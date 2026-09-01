"""A dependency-free Python client for the TimesFM-3 Forecast Service.

    from timesfm3.client import ForecastClient
    client = ForecastClient("http://localhost:8000", api_key=None)
    result = client.forecast([series], horizon=24)     # ForecastResult (numpy)
    report = client.backtest([series], context=256, horizon=24)

Only the standard library is used, so it works in any environment that can
reach the server, with or without torch installed.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence

import numpy as np

from .forecaster import ForecastResult


class ForecastServiceError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


def _series(values, name: str | None = None) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "name": name,
        "values": [None if not np.isfinite(v) else float(v) for v in arr],
    }


class ForecastClient:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str | None = None,
                 timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    # -- transport -----------------------------------------------------------

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, method=method)
        req.add_header("content-type", "application/json")
        req.add_header("accept", "application/json")
        if self.api_key:
            req.add_header("x-api-key", self.api_key)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            try:
                detail = json.loads(body).get("detail", body)
            except ValueError:
                detail = body
            raise ForecastServiceError(e.code, str(detail)) from None

    # -- API -------------------------------------------------------------------

    def health(self) -> dict:
        return self._request("GET", "/healthz")

    def models(self) -> list[dict]:
        return self._request("GET", "/v1/models")

    def forecast_raw(
        self,
        targets: Sequence,
        horizon: int,
        past_covariates: Sequence = (),
        future_covariates: Sequence = (),
        model: str | None = None,
        names: Sequence[str] | None = None,
        timestamps: Sequence[str] | None = None,
        freq: str | None = None,
        quantiles: bool = True,
    ) -> dict:
        names = list(names) if names else [None] * len(targets)
        payload = {
            "targets": [_series(t, n) for t, n in zip(targets, names)],
            "past_covariates": [_series(c) for c in past_covariates],
            "future_covariates": [_series(c) for c in future_covariates],
            "horizon": int(horizon),
            "model": model,
            "quantiles": quantiles,
            "timestamps": list(timestamps) if timestamps else None,
            "freq": freq,
        }
        return self._request("POST", "/v1/forecast", payload)

    def forecast(self, targets: Sequence, horizon: int, **kwargs) -> ForecastResult:
        """Same signature as :meth:`TimesFM3Forecaster.forecast`, over HTTP."""
        j = self.forecast_raw(targets, horizon, **kwargs)
        levels = tuple(j["quantile_levels"])
        keys = [f"q{int(round(q * 100))}" for q in levels]
        point = np.asarray([f["point"] for f in j["forecasts"]], dtype=np.float64)
        if j["forecasts"] and j["forecasts"][0]["quantiles"]:
            quantiles = np.stack(
                [np.stack([f["quantiles"][k] for k in keys], axis=-1) for f in j["forecasts"]]
            )
        else:
            quantiles = np.full(point.shape + (len(levels),), np.nan)
        result = ForecastResult(point=point, quantiles=quantiles, quantile_levels=levels)
        result.model = j["model"]  # type: ignore[attr-defined]
        result.timestamps = j.get("timestamps")  # type: ignore[attr-defined]
        return result

    def backtest(
        self,
        series: Sequence,
        context: int,
        horizon: int,
        models: Sequence[str] | None = None,
        reference: str = "last-value",
        windows: int = 20,
        metric: str = "mae",
        overlap: bool = False,
    ) -> dict:
        payload = {
            "series": [_series(s) for s in series],
            "context": int(context),
            "horizon": int(horizon),
            "models": list(models) if models else None,
            "reference": reference,
            "windows": int(windows),
            "metric": metric,
            "overlap": overlap,
        }
        return self._request("POST", "/v1/backtest", payload)

    def volatility(
        self,
        returns: Sequence | None = None,
        prices: Sequence | None = None,
        horizon: int = 5,
        vol_target: float = 0.10,
        max_leverage: float = 3.0,
    ) -> dict:
        payload = {
            "returns": [float(x) for x in returns] if returns is not None else None,
            "prices": [float(x) for x in prices] if prices is not None else None,
            "horizon": int(horizon),
            "vol_target": vol_target,
            "max_leverage": max_leverage,
        }
        return self._request("POST", "/v1/volatility", payload)
