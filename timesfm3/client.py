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

from .credits import CreditWallet
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
                 timeout: float = 120.0, credits: CreditWallet | None = None):
        """``credits``: a :class:`CreditWallet`; when set and no API key is given,
        each priced call spends unlinkable prepaid tokens instead."""
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.credits = credits

    # -- transport -----------------------------------------------------------

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, method=method)
        req.add_header("content-type", "application/json")
        req.add_header("accept", "application/json")
        if self.api_key:
            req.add_header("x-api-key", self.api_key)
        elif self.credits is not None and CreditWallet.is_priced(method, path):
            req.add_header("authorization", self.credits.take())
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

    def _request_raw(self, method: str, path: str, data: bytes | None = None,
                     headers: dict | None = None) -> tuple[int, dict, bytes]:
        """Bytes in / (status, headers, bytes) out; 4xx/5xx are returned, not raised."""
        req = urllib.request.Request(self.base_url + path, data=data, method=method)
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        if self.api_key:
            req.add_header("x-api-key", self.api_key)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read()

    # -- API -------------------------------------------------------------------

    def health(self) -> dict:
        return self._request("GET", "/healthz")

    def pricing(self) -> dict:
        return self._request("GET", "/v1/pricing")

    def models(self) -> list[dict]:
        return self._request("GET", "/v1/models")

    def buy_credits(self, count: int, wallet: CreditWallet | None = None) -> int:
        """Buys ``count`` Privacy Pass tokens into ``wallet`` (default: this client's),
        paid with the client's API key. For x402 use ``timesfm3 credits buy --private-key``."""
        from .privacypass import MEDIA_BATCH_REQUEST, MEDIA_REQUEST

        wallet = wallet or self.credits
        if wallet is None:
            raise ValueError("a CreditWallet is required")
        _, hdrs, _ = self._request_raw("GET", "/token-request/challenge")
        www = hdrs.get("www-authenticate", "")
        if not www:
            raise ForecastServiceError(500, "service did not offer a PrivateToken challenge")
        body, pending, batched = wallet.prepare(www, count)
        path = f"/token-request/batch/{count}" if batched else "/token-request"
        status, _, out = self._request_raw("POST", path, body, {"content-type": MEDIA_BATCH_REQUEST if batched else MEDIA_REQUEST})
        if status != 200:
            raise ForecastServiceError(status, out.decode(errors="replace")[:300])
        return wallet.finish(pending, out, batched)

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
