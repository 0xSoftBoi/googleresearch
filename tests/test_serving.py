import numpy as np
import pytest
from fastapi.testclient import TestClient

from timesfm3.serving.app import create_app


@pytest.fixture(scope="module")
def client(registry):
    return TestClient(create_app(registry=registry, max_series=8, max_context=1024))


def _series(x, name=None):
    return {"name": name, "values": [None if not np.isfinite(v) else float(v) for v in x]}


def test_health_models_dashboard_sample(client):
    h = client.get("/healthz").json()
    assert h["status"] == "ok" and h["models"] == 7 and h["default_model"] == "tiny"
    models = client.get("/v1/models").json()
    assert {m["name"] for m in models} >= {"tiny", "ewma", "last-value"}
    assert sum(m["default"] for m in models) == 1
    page = client.get("/")
    assert page.status_code == 200 and "TimesFM-3 Forecast Service" in page.text
    s = client.get("/v1/sample?n=50").json()
    assert len(s["values"][0]) == 50 and len(s["names"]) == len(s["values"])
    assert client.get("/docs").status_code == 200
    assert client.get("/favicon.ico").headers["content-type"].startswith("image/svg")


def test_forecast_default_model_with_timestamps(client, seasonal):
    stamps = [str(np.datetime64("2024-01-01T00:00") + np.timedelta64(i, "h")) for i in range(96)]
    r = client.post("/v1/forecast", json={
        "targets": [_series(seasonal[:96], "load")], "horizon": 12, "timestamps": stamps,
    })
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["model"] == "tiny" and len(j["forecasts"]) == 1
    f = j["forecasts"][0]
    assert f["name"] == "load" and len(f["point"]) == 12
    assert set(f["quantiles"]) == {f"q{k}" for k in range(10, 100, 10)}
    assert j["timestamps"][0] == "2024-01-05T00:00:00" and len(j["timestamps"]) == 12
    assert j["latency_ms"] > 0


def test_forecast_classical_freq_and_no_quantiles(client, seasonal):
    r = client.post("/v1/forecast", json={
        "targets": [_series(seasonal[:50])], "horizon": 5, "model": "ar4", "quantiles": False,
        "timestamps": [f"2024-01-{d:02d}" for d in range(1, 51)] if False else
        [str(np.datetime64("2024-01-01") + np.timedelta64(i, "D")) for i in range(50)],
        "freq": "W",
    })
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["model"] == "ar4" and j["forecasts"][0]["quantiles"] is None
    assert j["forecasts"][0]["name"] == "target_0"
    assert j["timestamps"][0] == "2024-02-26T00:00:00"  # 2024-02-19 + 7 days


def test_forecast_with_covariates_and_nans(client, seasonal):
    x = seasonal[:128].copy()
    x[[3, 40]] = np.nan
    r = client.post("/v1/forecast", json={
        "targets": [_series(x, "a"), _series(x * 2, "b")],
        "past_covariates": [_series(x)],
        "future_covariates": [_series(np.ones(128 + 16))],
        "horizon": 16, "model": "tiny",
    })
    assert r.status_code == 200, r.text
    assert len(r.json()["forecasts"]) == 2


def test_forecast_validation_errors(client, seasonal):
    base = {"targets": [_series(seasonal[:64])], "horizon": 4}
    assert client.post("/v1/forecast", json={**base, "model": "nope"}).status_code == 404
    assert client.post("/v1/forecast", json={**base, "model": "ewma",
                       "past_covariates": [_series(seasonal[:64])]}).status_code == 400
    assert client.post("/v1/forecast", json={**base, "timestamps": ["x"]}).status_code == 400
    assert client.post("/v1/forecast", json={**base, "horizon": 0}).status_code == 422
    assert client.post("/v1/forecast", json={"targets": [], "horizon": 4}).status_code == 422
    assert client.post("/v1/forecast", json={"targets": [{"values": [None, None]}],
                       "horizon": 4}).status_code == 422
    too_many = {"targets": [_series(seasonal[:64])] * 9, "horizon": 4}
    assert client.post("/v1/forecast", json=too_many).status_code == 413
    too_long = {"targets": [_series(np.ones(1025))], "horizon": 4}
    assert client.post("/v1/forecast", json=too_long).status_code == 413
    # mismatched lengths surface as 400 from the forecaster, not 500
    bad = {"targets": [_series(seasonal[:64]), _series(seasonal[:60])], "horizon": 4, "model": "tiny"}
    assert client.post("/v1/forecast", json=bad).status_code == 400


def test_backtest(client, seasonal):
    r = client.post("/v1/backtest", json={
        "series": [_series(seasonal), _series(seasonal[::-1])],
        "context": 96, "horizon": 24, "windows": 5,
        "models": ["ewma", "drift", "ctx-mean"],
    })
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["reference"] == "last-value" and j["windows_per_series"] == 5
    names = [s["model"] for s in j["scores"]]
    assert set(names) == {"ewma", "drift", "ctx-mean", "last-value"}
    ref = next(s for s in j["scores"] if s["model"] == "last-value")
    assert ref["verdict"] == "reference" and ref["ratio"] == 1.0 and ref["n"] == 10
    other = next(s for s in j["scores"] if s["model"] == "ewma")
    assert other["verdict"] in {"better", "worse", "no difference"}
    assert 0 <= other["win_rate"] <= 1 and other["p_adjusted"] is not None
    assert [s["ratio"] for s in j["scores"]] == sorted(s["ratio"] for s in j["scores"])

    short = client.post("/v1/backtest", json={"series": [_series(seasonal[:100])],
                        "context": 96, "horizon": 24})
    assert short.status_code == 400
    assert client.post("/v1/backtest", json={"series": [_series(seasonal)], "context": 96,
                       "horizon": 24, "models": ["ghost"]}).status_code == 404


def test_backtest_overlap_and_mse(client, seasonal):
    r = client.post("/v1/backtest", json={
        "series": [_series(seasonal)], "context": 64, "horizon": 8, "windows": 30,
        "overlap": True, "metric": "mse", "models": ["ewma"],
    })
    assert r.status_code == 200, r.text
    assert r.json()["metric"] == "mse" and r.json()["windows_per_series"] == 30


def test_volatility(client):
    rng = np.random.default_rng(0)
    ret = rng.normal(0, 0.01, 600) * np.repeat([1.0, 2.0, 0.5], 200)
    r = client.post("/v1/volatility", json={"returns": ret.tolist(), "horizon": 5,
                                            "vol_target": 0.1, "max_leverage": 2.0})
    assert r.status_code == 200, r.text
    j = r.json()
    assert {f["model"] for f in j["forecasts"]} == {"riskmetrics", "har"}
    for f in j["forecasts"]:
        assert len(f["variance_path"]) == 5 and f["annualized_vol"] > 0
        assert 0 <= f["weight"] <= 2.0
    prices = np.exp(np.cumsum(ret)) * 100
    assert client.post("/v1/volatility", json={"prices": prices.tolist()}).status_code == 200
    assert client.post("/v1/volatility", json={"returns": [0.01] * 10}).status_code == 400
    assert client.post("/v1/volatility", json={}).status_code == 400
    assert client.post("/v1/volatility", json={"prices": [1, -1] * 40}).status_code == 422


def test_metrics_exposition(client):
    text = client.get("/metrics").text
    assert 'timesfm3_requests_total{endpoint="forecast"}' in text
    assert "timesfm3_models 7" in text
    assert "timesfm3_steps_forecast_total" in text


def test_api_key(registry, seasonal):
    c = TestClient(create_app(registry=registry, api_key="k3y"))
    assert c.get("/healthz").status_code == 200  # liveness stays open
    assert c.get("/v1/models").status_code == 401
    assert c.get("/v1/models", headers={"x-api-key": "wrong"}).status_code == 401
    assert c.get("/v1/models", headers={"x-api-key": "k3y"}).status_code == 200
    assert c.get("/v1/models", headers={"authorization": "Bearer k3y"}).status_code == 200
    body = {"targets": [_series(seasonal[:32])], "horizon": 2, "model": "ewma"}
    assert c.post("/v1/forecast", json=body).status_code == 401
    assert c.post("/v1/forecast", json=body, headers={"x-api-key": "k3y"}).status_code == 200


def test_env_configuration(monkeypatch, registry):
    monkeypatch.setenv("TIMESFM3_API_KEY", "envkey")
    monkeypatch.setenv("TIMESFM3_MAX_SERIES", "1")
    c = TestClient(create_app(registry=registry))
    assert c.get("/v1/models").status_code == 401
    two = {"targets": [_series(np.arange(20.0))] * 2, "horizon": 2, "model": "ewma"}
    assert c.post("/v1/forecast", json=two, headers={"x-api-key": "envkey"}).status_code == 413
