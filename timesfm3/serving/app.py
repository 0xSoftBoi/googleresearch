"""The forecasting service.

    timesfm3 serve                       # http://localhost:8000
    curl -X POST localhost:8000/v1/forecast -H 'content-type: application/json' \\
         -d '{"targets":[{"values":[1,2,3,4,5,6,7,8]}],"horizon":4}'

Endpoints:

- ``GET  /``                dashboard
- ``GET  /healthz``         liveness + model count
- ``GET  /metrics``         Prometheus text exposition
- ``GET  /v1/models``       registered forecasters
- ``POST /v1/forecast``     point + quantile forecasts
- ``POST /v1/backtest``     walk-forward model comparison on your data
- ``POST /v1/volatility``   variance forecast and vol-targeted sizing
- ``POST /v1/anomalies``    walk-forward anomaly scoring
- ``GET  /v1/usage``        the calling key's metered usage this month
- ``GET  /v1/sample``       demo series for the dashboard

Auth and metering: see :mod:`timesfm3.serving.auth`.  Clients send
``X-API-Key: <key>`` or ``Authorization: Bearer <key>``; every metered
response carries ``X-Usage-Points`` (charged) and ``X-Usage-Remaining``.
"""

from __future__ import annotations

import os
import threading
import time
from collections import Counter, defaultdict

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, PlainTextResponse, Response

from .. import __version__
from ..anomaly import detect_anomalies
from ..evaluation import compare
from ..quant.volatility import HAR, TRADING_DAYS, ewma_variance, realized_variance
from ..tabular import future_timestamps, infer_step, parse_freq
from . import schemas
from .auth import ANONYMOUS, ApiKey, KeyStore, QuotaExceeded, UsageMeter
from .registry import ModelRegistry
from .x402 import X402Config, X402Gate, paid_identity

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class Metrics:
    """Minimal in-process counters, exported in Prometheus text format."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests: Counter = Counter()
        self.errors: Counter = Counter()
        self.latency_sum: dict[str, float] = defaultdict(float)
        self.series_forecast = 0
        self.steps_forecast = 0

    def record(self, endpoint: str, seconds: float, ok: bool = True) -> None:
        with self._lock:
            self.requests[endpoint] += 1
            self.latency_sum[endpoint] += seconds
            if not ok:
                self.errors[endpoint] += 1

    def exposition(self, registry: ModelRegistry) -> str:
        lines = [
            "# HELP timesfm3_requests_total Requests per endpoint.",
            "# TYPE timesfm3_requests_total counter",
        ]
        for ep, n in sorted(self.requests.items()):
            lines.append(f'timesfm3_requests_total{{endpoint="{ep}"}} {n}')
        lines += ["# HELP timesfm3_errors_total Failed requests per endpoint.",
                  "# TYPE timesfm3_errors_total counter"]
        for ep, n in sorted(self.errors.items()):
            lines.append(f'timesfm3_errors_total{{endpoint="{ep}"}} {n}')
        lines += ["# HELP timesfm3_latency_seconds_sum Total seconds spent per endpoint.",
                  "# TYPE timesfm3_latency_seconds_sum counter"]
        for ep, s in sorted(self.latency_sum.items()):
            lines.append(f'timesfm3_latency_seconds_sum{{endpoint="{ep}"}} {s:.6f}')
        lines += [
            "# HELP timesfm3_series_forecast_total Target series forecast.",
            "# TYPE timesfm3_series_forecast_total counter",
            f"timesfm3_series_forecast_total {self.series_forecast}",
            "# HELP timesfm3_steps_forecast_total Series x horizon steps forecast.",
            "# TYPE timesfm3_steps_forecast_total counter",
            f"timesfm3_steps_forecast_total {self.steps_forecast}",
            "# HELP timesfm3_models Registered forecasters.",
            "# TYPE timesfm3_models gauge",
            f"timesfm3_models {len(registry)}",
        ]
        return "\n".join(lines) + "\n"


def _to_array(series: schemas.Series) -> np.ndarray:
    return np.asarray(
        [np.nan if v is None else float(v) for v in series.values], dtype=np.float64
    )


def _names(series: list[schemas.Series], prefix: str) -> list[str]:
    return [s.name or f"{prefix}_{i}" for i, s in enumerate(series)]


def _finite(x: float) -> float | None:
    return float(x) if np.isfinite(x) else None


def create_app(
    registry: ModelRegistry | None = None,
    api_key: str | None = None,
    max_series: int | None = None,
    max_context: int | None = None,
    keys: KeyStore | None = None,
    meter: UsageMeter | None = None,
    x402: X402Config | None = None,
    x402_from_env: bool = True,
):
    """Builds the service. Reads ``TIMESFM3_*`` env vars for anything omitted.

    Returns the FastAPI app, or -- when x402 pay-per-call is configured -- an
    ASGI gate around it that proxies the app's attributes.
    """
    registry = registry or ModelRegistry.from_env()
    keys = keys or KeyStore.from_env(api_key)
    x402 = x402 if x402 is not None else (X402Config.from_env() if x402_from_env else None)
    meter = meter or UsageMeter(os.environ.get("TIMESFM3_USAGE_FILE") or None)
    max_series = max_series or int(os.environ.get("TIMESFM3_MAX_SERIES", "64"))
    max_context = max_context or int(os.environ.get("TIMESFM3_MAX_CONTEXT", "16384"))
    metrics = Metrics()
    anonymous = ApiKey(key="", name=ANONYMOUS, plan="open")

    app = FastAPI(
        title="TimesFM-3 Forecast Service",
        version=__version__,
        description=__doc__,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.registry = registry
    app.state.metrics = metrics
    app.state.keys = keys
    app.state.meter = meter
    app.state.x402 = x402

    async def require_key(request: Request) -> ApiKey:
        paid = paid_identity(request, x402)
        if paid is not None:
            return paid
        if keys.open:
            return anonymous
        header = request.headers.get("x-api-key")
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            header = header or auth[7:].strip()
        found = keys.lookup(header)
        if found is None:
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")
        return found

    def charge(key: ApiKey, points: int, response: Response) -> None:
        """Meters ``points`` against the key; 429 when the plan is exhausted."""
        try:
            bucket = meter.charge(key, points)
        except QuotaExceeded as e:
            raise HTTPException(
                status_code=429, detail=str(e),
                headers={"x-usage-remaining": "0", "retry-after": "86400"},
            )
        response.headers["x-usage-points"] = str(points)
        if not key.unlimited:
            response.headers["x-usage-remaining"] = str(
                max(0, key.monthly_points - bucket["points"])
            )

    def _timed(endpoint: str):
        class _Timer:
            def __enter__(self):
                self.t0 = time.perf_counter()
                self.ok = True
                return self

            def __exit__(self, exc_type, exc, tb):
                metrics.record(endpoint, time.perf_counter() - self.t0, ok=exc_type is None)
                return False

            @property
            def ms(self) -> float:
                return 1000.0 * (time.perf_counter() - self.t0)

        return _Timer()

    # -- static & ops ------------------------------------------------------

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard() -> str:
        with open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8") as f:
            return f.read()

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
            '<rect width="32" height="32" rx="6" fill="#171a21"/>'
            '<polyline points="4,22 11,15 17,19 28,7" fill="none" stroke="#5b9cff" '
            'stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        )
        return Response(svg, media_type="image/svg+xml",
                        headers={"cache-control": "public, max-age=86400"})

    @app.get("/healthz", response_model=schemas.Health, tags=["ops"])
    async def healthz() -> schemas.Health:
        default = registry.get() if registry.default else None
        return schemas.Health(
            status="ok",
            version=__version__,
            models=len(registry),
            default_model=registry.default,
            device=(default.meta.get("device", "cpu") if default else "cpu"),
        )

    @app.get("/v1/pricing", tags=["ops"])
    async def pricing() -> dict:
        """How to pay: plans via API keys, or x402 pay-per-call in USDC."""
        return {
            "plans": {"keys": keys.names() if not keys.open else [], "unit": "forecast points",
                      "note": "Send X-API-Key; see /v1/usage for quota."},
            "x402": x402.describe() if x402 else {"enabled": False},
        }

    @app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
    async def prometheus() -> str:
        return metrics.exposition(registry)

    @app.get("/v1/models", response_model=list[schemas.ModelInfo], tags=["models"],
             dependencies=[Depends(require_key)])
    async def models() -> list[dict]:
        return registry.describe()

    @app.get("/v1/usage", response_model=schemas.Usage, tags=["ops"])
    async def usage(key: ApiKey = Depends(require_key)) -> dict:
        """Metered usage of the calling key for the current month."""
        return meter.usage(key)

    @app.get("/v1/sample", tags=["models"], dependencies=[Depends(require_key)])
    async def sample(n: int = 336) -> dict:
        """A demo panel: a bundled ETTh2 slice if present, else synthetic."""
        return sample_panel(n)

    # -- forecasting -------------------------------------------------------

    @app.post("/v1/forecast", response_model=schemas.ForecastResponse, tags=["forecast"])
    async def forecast(
        req: schemas.ForecastRequest, response: Response, key: ApiKey = Depends(require_key)
    ) -> schemas.ForecastResponse:
        with _timed("forecast") as timer:
            try:
                entry = registry.get(req.model)
            except KeyError as e:
                raise HTTPException(status_code=404, detail=str(e))
            n_series = len(req.targets) + len(req.past_covariates) + len(req.future_covariates)
            if n_series > max_series:
                raise HTTPException(
                    status_code=413, detail=f"At most {max_series} series per request."
                )
            context = len(req.targets[0].values)
            if context > max_context:
                raise HTTPException(
                    status_code=413, detail=f"Context is capped at {max_context} steps."
                )
            if (req.past_covariates or req.future_covariates) and not entry.supports_covariates:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model {entry.name!r} does not accept covariates; "
                    "pick a TimesFM-3 checkpoint.",
                )
            if req.timestamps is not None and len(req.timestamps) != context:
                raise HTTPException(
                    status_code=400, detail="timestamps must have one entry per context step."
                )
            charge(key, len(req.targets) * req.horizon, response)
            targets = [_to_array(s) for s in req.targets]
            past = [_to_array(s) for s in req.past_covariates]
            future = [_to_array(s) for s in req.future_covariates]
            try:
                result = await run_in_threadpool(
                    entry.forecast, targets, req.horizon, past, future
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

            stamps = None
            if req.timestamps:
                step = parse_freq(req.freq) if req.freq else infer_step(req.timestamps)
                if step is not None:
                    try:
                        stamps = future_timestamps(req.timestamps[-1], req.horizon, step)
                    except ValueError:
                        stamps = None
            levels = list(result.quantile_levels)
            keys = [f"q{int(round(q * 100))}" for q in levels]
            out = []
            for i, name in enumerate(_names(req.targets, "target")):
                q = None
                if req.quantiles:
                    q = {k: result.quantiles[i, :, j].astype(float).tolist()
                         for j, k in enumerate(keys)}
                out.append(schemas.TargetForecast(
                    name=name, point=result.point[i].astype(float).tolist(), quantiles=q
                ))
            with metrics._lock:
                metrics.series_forecast += len(out)
                metrics.steps_forecast += len(out) * req.horizon
            return schemas.ForecastResponse(
                model=entry.name, horizon=req.horizon, quantile_levels=levels,
                timestamps=stamps, forecasts=out, latency_ms=timer.ms,
            )

    # -- backtest ----------------------------------------------------------

    @app.post("/v1/backtest", response_model=schemas.BacktestResponse, tags=["forecast"])
    async def backtest(
        req: schemas.BacktestRequest, response: Response, key: ApiKey = Depends(require_key)
    ) -> schemas.BacktestResponse:
        with _timed("backtest") as timer:
            names = req.models or registry.names()
            for m in list(names) + [req.reference]:
                if m not in registry:
                    raise HTTPException(status_code=404, detail=f"Unknown model {m!r}.")
            if req.reference not in names:
                names = list(names) + [req.reference]
            if len(req.series) > max_series:
                raise HTTPException(status_code=413, detail=f"At most {max_series} series.")
            series = [_to_array(s) for s in req.series]
            charge(key, len(series) * req.windows * req.horizon * len(set(names)), response)
            try:
                report = await run_in_threadpool(
                    run_backtest, registry, series, req.context, req.horizon,
                    names, req.reference, req.windows, req.metric, req.overlap,
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            return schemas.BacktestResponse(**report, latency_ms=timer.ms)

    # -- volatility --------------------------------------------------------

    @app.post("/v1/volatility", response_model=schemas.VolatilityResponse, tags=["quant"])
    async def volatility(
        req: schemas.VolatilityRequest, response: Response, key: ApiKey = Depends(require_key)
    ) -> schemas.VolatilityResponse:
        with _timed("volatility") as timer:
            charge(key, req.horizon, response)
            if req.returns is None and req.prices is None:
                raise HTTPException(status_code=400, detail="Provide returns or prices.")
            if req.returns is not None:
                r = np.asarray(req.returns, dtype=np.float64)
            else:
                r = np.diff(np.log(np.asarray(req.prices, dtype=np.float64)))
            r = r[np.isfinite(r)]
            if len(r) < 60:
                raise HTTPException(
                    status_code=400, detail="Need at least 60 daily returns (ideally 500+)."
                )
            report = await run_in_threadpool(
                variance_report, r, req.horizon, req.vol_target, req.max_leverage
            )
            return schemas.VolatilityResponse(**report, latency_ms=timer.ms)

    # -- anomalies ---------------------------------------------------------

    @app.post("/v1/anomalies", response_model=schemas.AnomalyResponse, tags=["forecast"])
    async def anomalies(
        req: schemas.AnomalyRequest, response: Response, key: ApiKey = Depends(require_key)
    ) -> schemas.AnomalyResponse:
        with _timed("anomalies") as timer:
            try:
                entry = registry.get(req.model)
            except KeyError as e:
                raise HTTPException(status_code=404, detail=str(e))
            if len(req.series) > max_series:
                raise HTTPException(status_code=413, detail=f"At most {max_series} series.")
            lengths = {len(s.values) for s in req.series}
            if max(lengths) > max_context * 4:
                raise HTTPException(
                    status_code=413, detail=f"Series are capped at {max_context * 4} steps."
                )
            if req.timestamps is not None and len(req.timestamps) not in lengths:
                raise HTTPException(
                    status_code=400, detail="timestamps must have one entry per step."
                )
            series = [_to_array(s) for s in req.series]
            charge(key, sum(max(0, len(x) - req.context) for x in series), response)
            out = []
            for name, x in zip(_names(req.series, "series"), series):
                try:
                    rep = await run_in_threadpool(
                        detect_anomalies, entry, x, req.context, req.block, req.threshold
                    )
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=str(e))
                stamps = req.timestamps if req.timestamps and len(req.timestamps) == len(x) else None
                item = schemas.SeriesAnomalies(
                    name=name, n_scored=int(np.isfinite(rep.scores).sum()),
                    n_flagged=int(rep.flagged.sum()),
                    anomalies=[schemas.Anomaly(**a) for a in rep.anomalies(x, stamps)],
                )
                if req.include_scores:
                    item.scores = [_finite(v) for v in rep.scores]
                    item.expected = [_finite(v) for v in rep.expected]
                    item.lower = [_finite(v) for v in rep.lower]
                    item.upper = [_finite(v) for v in rep.upper]
                out.append(item)
            return schemas.AnomalyResponse(
                model=entry.name, context=req.context, block=req.block,
                threshold=req.threshold, series=out, latency_ms=timer.ms,
            )

    if x402 is not None:
        return X402Gate(app, keys, x402)
    return app


# -- pure helpers (also used by the CLI) ------------------------------------


def run_backtest(
    registry: ModelRegistry,
    series: list[np.ndarray],
    context: int,
    horizon: int,
    models: list[str],
    reference: str,
    windows: int,
    metric: str = "mae",
    overlap: bool = False,
) -> dict:
    """Walk-forward comparison; returns the ``BacktestResponse`` fields."""
    models = list(models)
    if reference not in models:
        models.append(reference)
    span = context + horizon
    losses: dict[str, list[float]] = {m: [] for m in models}
    groups: list[int] = []
    n_windows = 0
    for g, x in enumerate(series):
        n = len(x)
        if n < span + 1:
            raise ValueError(
                f"Series {g} has {n} steps; need at least context + horizon + 1 = {span + 1}."
            )
        latest = n - span
        if overlap:
            starts = np.unique(np.linspace(0, latest, num=min(windows, latest + 1)).astype(int))
        else:
            max_nonoverlap = latest // horizon + 1
            k = min(windows, max_nonoverlap)
            starts = latest - horizon * np.arange(k)[::-1]
        n_windows = max(n_windows, len(starts))
        for s in starts:
            ctx = x[s : s + context]
            truth = x[s + context : s + span]
            if not np.isfinite(truth).any() or np.isfinite(ctx).sum() < 2:
                continue
            for m in models:
                pred = registry.get(m).forecast([ctx], horizon).point[0]
                err = pred - truth
                err = err[np.isfinite(err)]
                loss = float(np.mean(err ** 2) if metric == "mse" else np.mean(np.abs(err)))
                losses[m].append(loss)
            groups.append(g)
    if not groups:
        raise ValueError("No scorable windows (all-NaN targets?).")
    arrays = {m: np.asarray(v) for m, v in losses.items()}
    cmp = compare(arrays, reference, groups=np.asarray(groups) if len(series) > 1 else None,
                  resamples=1000)
    scores = []
    ref_loss = float(arrays[reference].mean())
    for m in models:
        if m == reference:
            scores.append(dict(
                model=m, mean_loss=ref_loss, ratio=1.0, ci_low=None, ci_high=None,
                p_value=None, p_adjusted=None, win_rate=0.0, n=len(arrays[m]),
                n_effective=float(len(arrays[m])), verdict="reference",
            ))
            continue
        c = cmp[m]
        scores.append(dict(
            model=m, mean_loss=c.mean_loss, ratio=c.ratio, ci_low=_finite(c.ci_low),
            ci_high=_finite(c.ci_high), p_value=_finite(c.p_value),
            p_adjusted=_finite(c.p_adjusted) if c.p_adjusted is not None else None,
            win_rate=c.win_rate, n=c.n, n_effective=c.n_effective, verdict=c.verdict,
        ))
    scores.sort(key=lambda s: s["ratio"])
    return dict(
        reference=reference, reference_loss=ref_loss, metric=metric, context=context,
        horizon=horizon, windows_per_series=n_windows, scores=scores,
    )


def variance_report(
    r: np.ndarray, horizon: int, vol_target: float, max_leverage: float
) -> dict:
    """HAR and RiskMetrics variance forecasts plus Moreira-Muir weights."""
    rv = realized_variance(r)
    out = []
    # RiskMetrics: flat at the filtered one-step-ahead variance.
    ewma = ewma_variance(r)
    last = float(ewma[-1]) if np.isfinite(ewma[-1]) else float(rv[-20:].mean())
    var_rm = rv[-1] * 0.06 + last * 0.94  # one more update with today's return
    paths = {"riskmetrics": np.full(horizon, var_rm)}
    har = HAR()
    if len(rv) >= HAR().monthly + 30:
        paths["har"] = np.asarray(har.forecast(rv[-min(len(rv), 756):], horizon))
    for name, path in paths.items():
        mean_var = float(np.mean(path))
        ann_vol = float(np.sqrt(mean_var * TRADING_DAYS))
        w = float(np.clip(vol_target / max(ann_vol, 1e-8), 0.0, max_leverage))
        out.append(dict(
            model=name, variance_path=[float(v) for v in path], mean_variance=mean_var,
            annualized_vol=ann_vol, weight=w,
        ))
    return dict(
        horizon=horizon, n_returns=int(len(r)),
        realized_vol_annualized=float(np.sqrt(rv[-21:].mean() * TRADING_DAYS)),
        forecasts=out,
    )


def sample_panel(n: int = 336) -> dict:
    """Demo data for the dashboard: hourly-looking seasonal series."""
    path = os.path.join(STATIC_DIR, "sample.csv")
    if os.path.exists(path):
        from ..tabular import read_series_csv

        t = read_series_csv(path)
        n = min(n, t.num_steps)
        return {
            "source": "ETTh2 tail (never used in training the starter model)",
            "names": t.names,
            "timestamps": t.timestamps[-n:] if t.timestamps else None,
            "values": [[None if not np.isfinite(v) else float(v) for v in row[-n:]]
                       for row in t.values],
        }
    rng = np.random.default_rng(7)
    idx = np.arange(n)
    daily = np.sin(2 * np.pi * idx / 24)
    weekly = np.sin(2 * np.pi * idx / 168)
    a = 10 + 3 * daily + 1.5 * weekly + rng.normal(0, 0.4, n)
    b = 5 - 2 * daily + 0.8 * weekly + rng.normal(0, 0.3, n)
    base = np.datetime64("2024-01-01T00:00")
    return {
        "source": "synthetic (daily + weekly seasonality)",
        "names": ["load_a", "load_b"],
        "timestamps": [str(base + np.timedelta64(int(i), "h")) for i in idx],
        "values": [a.round(4).tolist(), b.round(4).tolist()],
    }
