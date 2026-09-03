import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from timesfm3.client import ForecastClient, ForecastServiceError
from timesfm3.serving.app import create_app


@pytest.fixture
def client(registry, monkeypatch):
    http = TestClient(create_app(registry=registry, api_key="k"))

    def _request(self, method, path, payload=None):
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        r = http.request(method, path, content=json.dumps(payload) if payload is not None else None,
                         headers={**headers, "content-type": "application/json"})
        if r.status_code >= 400:
            raise ForecastServiceError(r.status_code, r.json().get("detail", r.text))
        return r.json()

    monkeypatch.setattr(ForecastClient, "_request", _request)
    return ForecastClient("http://test", api_key="k")


def test_client_roundtrip(client, seasonal):
    assert client.health()["status"] == "ok"
    assert any(m["name"] == "tiny" for m in client.models())
    x = seasonal[:128].copy()
    x[5] = np.nan
    res = client.forecast([x, x + 1], horizon=8, model="ewma", names=["a", "b"],
                          timestamps=[str(np.datetime64("2024-01-01") + np.timedelta64(i, "D"))
                                      for i in range(128)])
    assert res.point.shape == (2, 8) and res.quantiles.shape == (2, 8, 9)
    assert res.model == "ewma" and len(res.timestamps) == 8
    assert np.all(np.diff(res.quantiles, axis=-1) >= 0)

    raw = client.forecast_raw([x], 4, quantiles=False)
    assert raw["forecasts"][0]["quantiles"] is None
    bt = client.backtest([seasonal], context=64, horizon=8, windows=4, models=["ewma"])
    assert {s["model"] for s in bt["scores"]} == {"ewma", "last-value"}
    vol = client.volatility(prices=np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 300))))
    assert vol["n_returns"] == 299


def test_client_errors(client, seasonal):
    with pytest.raises(ForecastServiceError) as e:
        client.forecast([seasonal[:32]], 4, model="ghost")
    assert e.value.status == 404
    bad = ForecastClient("http://test", api_key="wrong")
    with pytest.raises(ForecastServiceError) as e:
        bad.models()
    assert e.value.status == 401
