"""A volatility-specialized TimesFM-3: data, competitors, QLIKE fine-tuning, evaluation.

Variance is the forecasting problem quant desks pay for, and it has an
industry bar: RiskMetrics EWMA and Corsi's HAR-RV, scored by QLIKE. This
module turns that into a reproducible model-building protocol (see
``docs/VERIFICATION.md`` section 11 for the pre-registered splits and gate):

- the target is daily variance, proxied by squared daily log returns from
  the FRED multi-asset panel (:mod:`timesfm3.data.markets`);
- neural models see ``log`` variance over a 512-day context (floored at the
  context's own 1st percentile, so a zero-return day does not dominate) and
  forecast log variance; ``exp`` of the point forecast is the variance
  forecast;
- fine-tuning optimizes QLIKE on ``exp(point)`` directly. QLIKE is
  differentiable and minimized by the conditional mean of variance, so the
  model learns a mean-variance forecast without a log-normal correction;
- every model, classical or neural, gets one multiplicative calibration
  scalar per horizon fitted on the validation period and applied unchanged
  to the test period;
- losses are averaged across assets per date and compared with the
  repository's Diebold-Mariano / Holm machinery (:func:`timesfm3.evaluation.compare`).
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Callable, Sequence

import numpy as np
import torch

from ..embedding import ROLE_PAST_COVARIATE, ROLE_TARGET
from ..evaluation import Comparison, compare
from ..loss import quantile_loss
from ..model import TimesFM3Model
from .volatility import HAR, ewma_variance, qlike, realized_variance

CONTEXT = 512
HORIZONS = (1, 5)
EVAL_HORIZON = max(HORIZONS)
TRAIN_END = np.datetime64("2015-01-01")
TEST_START = np.datetime64("2020-01-01")

#: (n_origins, 2) variance forecasts for horizons (1, 5): column 0 is the
#: next-day variance, column 1 the mean variance over the next five days.
Forecaster = Callable[["AssetRV", np.ndarray], np.ndarray]


@dataclasses.dataclass
class AssetRV:
    name: str
    dates: np.ndarray    # datetime64[D] of trading days with a finite return
    returns: np.ndarray  # daily log returns on those days
    rv: np.ndarray       # squared returns (floored), the daily variance proxy
    features: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.features:
            self.features = _causal_features(self.rv)


#: Input channels a candidate may use. ``log_rv`` is the target channel; the
#: rest are causal smoothings of the same squared returns (value at day i uses
#: days <= i only), fed as past-only covariates.
CHANNELS = ("log_rv", "log_ewma94", "log_mean22")


def _causal_features(rv: np.ndarray) -> dict[str, np.ndarray]:
    ew = np.empty_like(rv)
    v = float(rv[: min(20, len(rv))].mean())
    for i, x in enumerate(rv):
        v = 0.94 * v + 0.06 * x
        ew[i] = v
    c = np.concatenate([[0.0], np.cumsum(rv)])
    idx = np.arange(1, len(rv) + 1)
    lo = np.maximum(0, idx - 22)
    m22 = (c[idx] - c[lo]) / (idx - lo)
    return {"log_ewma94": np.log(ew), "log_mean22": np.log(m22)}


def rv_panel(panel) -> list[AssetRV]:
    """One :class:`AssetRV` per asset of a :class:`~timesfm3.data.markets.MarketPanel`."""
    out = []
    for name, r in zip(panel.names, panel.returns):
        ok = np.isfinite(r)
        rr = np.asarray(r[ok], dtype=np.float64)
        out.append(AssetRV(name, panel.dates[ok], rr, realized_variance(rr, demean=False)))
    return out


def split_origins(asset: AssetRV, split: str, context: int = CONTEXT,
                  max_horizon: int = EVAL_HORIZON) -> np.ndarray:
    """Origin indices ``t`` (information through day ``t``) for one split.

    train: the whole target window ends before ``TRAIN_END``;
    val: origin on/after ``TRAIN_END`` and target window before ``TEST_START``;
    test: origin on/after ``TEST_START``.
    """
    n = len(asset.rv)
    t = np.arange(context - 1, n - max_horizon)
    if t.size == 0:
        return t
    start, end = asset.dates[t], asset.dates[t + max_horizon]
    if split == "train":
        keep = end < TRAIN_END
    elif split == "val":
        keep = (start >= TRAIN_END) & (end < TEST_START)
    elif split == "test":
        keep = start >= TEST_START
    else:
        raise ValueError(f"unknown split {split!r}")
    return t[keep]


def contexts(asset: AssetRV, t: np.ndarray, context: int = CONTEXT) -> np.ndarray:
    """(len(t), context) windows ``rv[t - context + 1 : t + 1]``."""
    windows = np.lib.stride_tricks.sliding_window_view(asset.rv, context)
    return windows[t - context + 1]


def targets(asset: AssetRV, t: np.ndarray) -> np.ndarray:
    """(len(t), 2): next-day variance and mean variance over the next five days."""
    future = np.lib.stride_tricks.sliding_window_view(asset.rv, EVAL_HORIZON)[t + 1]
    return np.stack([future[:, 0], future.mean(axis=1)], axis=1)


def channel_contexts(asset: AssetRV, t: np.ndarray, channels: Sequence[str] = ("log_rv",),
                     context: int = CONTEXT) -> np.ndarray:
    """(len(t), C, context) model inputs; channel 0 must be ``log_rv``."""
    if channels[0] != "log_rv":
        raise ValueError("the first channel must be the log_rv target")
    rows = [log_context(contexts(asset, t, context))]
    for ch in channels[1:]:
        series = asset.features[ch]
        rows.append(np.lib.stride_tricks.sliding_window_view(series, context)[t - context + 1])
    return np.stack(rows, axis=1)


def log_context(ctx: np.ndarray) -> np.ndarray:
    """``log`` variance with each window floored at its own 1st percentile."""
    floor = np.quantile(ctx, 0.01, axis=-1, keepdims=True)
    return np.log(np.maximum(ctx, floor))


def _to_horizons(var_path: np.ndarray) -> np.ndarray:
    """(n, >=5) variance path -> (n, 2) [next day, five-day mean]."""
    return np.stack([var_path[:, 0], var_path[:, :EVAL_HORIZON].mean(axis=1)], axis=1)


# -- forecasters -------------------------------------------------------------


def riskmetrics() -> Forecaster:
    def fc(asset: AssetRV, t: np.ndarray) -> np.ndarray:
        ewma = ewma_variance(asset.returns)  # ewma[t + 1] uses information through t
        f = ewma[t + 1]
        return np.stack([f, f], axis=1)  # RiskMetrics' multi-step forecast is flat
    return fc


def har(log_space: bool = False) -> Forecaster:
    model = HAR(log_space=log_space)

    def fc(asset: AssetRV, t: np.ndarray) -> np.ndarray:
        ctx = contexts(asset, t)
        return _to_horizons(np.stack([model.forecast(c, EVAL_HORIZON) for c in ctx]))
    return fc


def _patch_batch(x_log: np.ndarray, horizon_patches: int, patch_len: int):
    if x_log.ndim == 2:
        x_log = x_log[:, None, :]
    b, c, length = x_log.shape
    total = length + horizon_patches * patch_len
    values = torch.zeros((b, c, total), dtype=torch.float32)
    values[:, :, :length] = torch.as_tensor(x_log, dtype=torch.float32)
    observed = torch.zeros((b, c, total), dtype=torch.bool)
    observed[:, :, :length] = True
    roles = torch.full((b, c), ROLE_PAST_COVARIATE, dtype=torch.int64)
    roles[:, 0] = ROLE_TARGET
    return values, observed, roles


def timesfm_log_forward(model: TimesFM3Model, x_log: np.ndarray, horizon_patches: int = 1):
    """Independent forecasts for a batch of log-variance contexts.

    ``x_log`` is (B, L) for univariate input or (B, C, L) with the log_rv
    target in channel 0 and past-only covariates after it. Returns the model
    output and the context length; ``point[:, 0, L:, 0]`` is the log-variance
    forecast path.
    """
    length = x_log.shape[-1]
    cfg = model.config
    if length % cfg.patch_len or length > cfg.max_context_len:
        raise ValueError(f"context {length} must be a multiple of {cfg.patch_len} "
                         f"and at most {cfg.max_context_len}")
    values, observed, roles = _patch_batch(x_log, horizon_patches, cfg.patch_len)
    out = model(values=values, observed=observed, roles=roles,
                num_horizon_patches=horizon_patches)
    return out, length


def timesfm(model: TimesFM3Model, batch_size: int = 256,
            channels: Sequence[str] = ("log_rv",)) -> Forecaster:
    model.eval()

    @torch.no_grad()
    def fc(asset: AssetRV, t: np.ndarray) -> np.ndarray:
        x = channel_contexts(asset, t, channels)
        paths = []
        for i in range(0, len(x), batch_size):
            out, length = timesfm_log_forward(model, x[i:i + batch_size])
            p = out.point[:, 0, length:length + EVAL_HORIZON, 0].double().numpy()
            paths.append(np.exp(p))
        return _to_horizons(np.concatenate(paths))
    return fc


def chronos(pipeline, batch_size: int = 256) -> Forecaster:
    """Zero-shot Chronos / Chronos-Bolt / Chronos-2 on the same log-variance input."""

    def fc(asset: AssetRV, t: np.ndarray) -> np.ndarray:
        x = log_context(contexts(asset, t)).astype(np.float32)
        paths = []
        for i in range(0, len(x), batch_size):
            chunk = [torch.from_numpy(row) for row in x[i:i + batch_size]]
            _, mean = pipeline.predict_quantiles(chunk, prediction_length=EVAL_HORIZON,
                                                 quantile_levels=[0.5])
            if isinstance(mean, (list, tuple)):  # Chronos-2: one tensor per input
                mean = torch.cat([m.reshape(-1, EVAL_HORIZON) for m in mean])
            paths.append(np.exp(mean.double().numpy()))
        return _to_horizons(np.concatenate(paths))
    return fc


# -- evaluation --------------------------------------------------------------


@dataclasses.dataclass
class Forecasts:
    """Raw (uncalibrated) forecasts and targets on one split, all assets stacked."""

    dates: np.ndarray          # (n,) origin dates
    assets: np.ndarray         # (n,) asset names
    target: np.ndarray         # (n, 2)
    preds: dict[str, np.ndarray]  # name -> (n, 2)


def collect(assets: Sequence[AssetRV], split: str, models: dict[str, Forecaster],
            verbose: bool = False) -> Forecasts:
    dates, names, tgt = [], [], []
    preds: dict[str, list[np.ndarray]] = {m: [] for m in models}
    for a in assets:
        t = split_origins(a, split)
        if t.size == 0:
            continue
        dates.append(a.dates[t])
        names.append(np.full(t.size, a.name))
        tgt.append(targets(a, t))
        for m, fc in models.items():
            t0 = time.time()
            preds[m].append(fc(a, t))
            if verbose:
                print(f"  {split:5s} {a.name:8s} {m:20s} {t.size:6d} origins  {time.time() - t0:6.1f}s",
                      flush=True)
    return Forecasts(np.concatenate(dates), np.concatenate(names), np.concatenate(tgt),
                     {m: np.concatenate(v) for m, v in preds.items()})


def fit_calibration(f: Forecasts) -> dict[str, np.ndarray]:
    """Per model and horizon, the QLIKE-optimal scale ``c = mean(target / pred)``."""
    return {m: (f.target / p).mean(axis=0) for m, p in f.preds.items()}


def daily_losses(f: Forecasts, calibration: dict[str, np.ndarray] | None = None
                 ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Per-date QLIKE averaged across assets: returns (dates, {model: (n_dates, 2)})."""
    udates, inv = np.unique(f.dates, return_inverse=True)
    counts = np.bincount(inv, minlength=len(udates)).astype(np.float64)
    out = {}
    for m, p in f.preds.items():
        scale = calibration[m] if calibration is not None else np.ones(2)
        loss = qlike(p * scale, f.target)
        per_date = np.zeros((len(udates), 2))
        np.add.at(per_date, inv, loss)
        out[m] = per_date / counts[:, None]
    return udates, out


def compare_models(losses: dict[str, np.ndarray], reference: str, horizon_col: int
                   ) -> dict[str, Comparison]:
    return compare({m: v[:, horizon_col] for m, v in losses.items()}, reference=reference)


# -- QLIKE fine-tuning -------------------------------------------------------


def qlike_from_log(log_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """QLIKE with ``pred = exp(log_pred)``, computed without forming ``target / pred``."""
    return target * torch.exp(-log_pred) + log_pred - torch.log(target) - 1.0


def finetune_qlike(
    model: TimesFM3Model,
    assets: Sequence[AssetRV],
    val: Forecasts | None = None,
    val_assets: Sequence[AssetRV] = (),
    steps: int = 1000,
    batch_size: int = 32,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    warmup: int = 50,
    pinball_weight: float = 0.1,
    horizon_patches: int = 1,
    val_every: int = 100,
    seed: int = 0,
    channels: Sequence[str] = ("log_rv",),
    step_horizon: int | None = None,
    agg_weight: float = 1.0,
    log: Callable[[str], None] = print,
) -> tuple[dict, list[dict]]:
    """Fine-tunes ``model`` in place on training-split windows; returns (best_state, history).

    The objective is per-step QLIKE over the forecast patch, plus QLIKE of the
    five-day mean variance, plus a small pinball term on log variance (in the
    model's normalized units) that keeps the quantile head calibrated. Model
    selection uses calibrated five-day QLIKE on ``val`` when given.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    cfg = model.config
    horizon = horizon_patches * cfg.patch_len
    pools = []
    for a in assets:
        t = split_origins(a, "train", max_horizon=horizon)
        if t.size:
            pools.append((a, t))
    if not pools:
        raise ValueError("no training windows")
    quantiles = cfg.quantiles
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_at(step: int) -> float:
        if step < warmup:
            return lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    def val_score() -> float:
        fc = timesfm(model, channels=channels)
        preds = np.concatenate([fc(a, split_origins(a, "val")) for a in val_assets
                                if split_origins(a, "val").size])
        tgt = val.target
        scale = (tgt / preds).mean(axis=0)
        return float(qlike(preds * scale, tgt)[:, 1].mean())

    history: list[dict] = []
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best = val_score() if val is not None else float("inf")
    if val is not None:
        log(f"step      0  val QLIKE(5d, calibrated) {best:.4f}  (base)")
        history.append({"step": 0, "val": best})
    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step - 1)
        ctx_rows, fut_rows, x_rows = [], [], []
        for _ in range(batch_size):
            a, pool = pools[rng.integers(len(pools))]
            t = int(pool[rng.integers(len(pool))])
            ctx_rows.append(a.rv[t - CONTEXT + 1:t + 1])
            fut_rows.append(a.rv[t + 1:t + 1 + horizon])
            x_rows.append(channel_contexts(a, np.array([t]), channels)[0])
        ctx = np.stack(ctx_rows)
        fut = torch.as_tensor(np.stack(fut_rows), dtype=torch.float32)
        x_in = np.stack(x_rows)
        x = x_in[:, 0, :]
        out, length = timesfm_log_forward(model, x_in, horizon_patches)
        p = out.point[:, 0, length:, 0]
        lo = torch.as_tensor(x.min(axis=1, keepdims=True) - 8.0, dtype=p.dtype)
        hi = torch.as_tensor(x.max(axis=1, keepdims=True) + 8.0, dtype=p.dtype)
        p = torch.maximum(torch.minimum(p, hi), lo)
        k = step_horizon or horizon
        step_loss = qlike_from_log(p[:, :k], fut[:, :k]).mean()
        agg_pred = torch.exp(p[:, :EVAL_HORIZON]).mean(dim=1)
        agg_tgt = fut[:, :EVAL_HORIZON].mean(dim=1)
        agg_loss = (agg_tgt / agg_pred - torch.log(agg_tgt / agg_pred) - 1.0).mean()
        floor = torch.as_tensor(np.quantile(ctx, 0.01, axis=1, keepdims=True), dtype=fut.dtype)
        log_fut = torch.log(torch.maximum(fut, floor))
        std = out.std[:, 0, :]  # (B, 1)
        q = out.quantiles[:, 0, length:, :]
        pin = quantile_loss(q / std[:, :, None], log_fut / std, quantiles).mean()
        loss = step_loss + agg_weight * agg_loss + pinball_weight * pin
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        rec = {"step": step, "loss": loss.item(), "qlike_step": step_loss.item(),
               "qlike_5d": agg_loss.item(), "pinball": pin.item()}
        if val is not None and (step % val_every == 0 or step == steps):
            score = val_score()
            rec["val"] = score
            marker = ""
            if score < best:
                best = score
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                marker = "  (best)"
            log(f"step {step:6d}  loss {loss.item():.4f}  val QLIKE(5d, calibrated) {score:.4f}"
                f"{marker}  [{(time.time() - t0) / 60:.1f} min]")
        history.append(rec)
    return best_state, history
