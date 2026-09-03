import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from timesfm3.serving.app import create_app
from timesfm3.serving.auth import ApiKey, KeyStore, QuotaExceeded, UsageMeter


def test_keystore_from_env_formats(tmp_path, monkeypatch):
    monkeypatch.setenv("TIMESFM3_API_KEY", "single")
    monkeypatch.setenv("TIMESFM3_API_KEYS", "team:t-key:1000,free:f-key")
    f = tmp_path / "keys.json"
    f.write_text(json.dumps({"keys": [{"key": "j-key", "name": "json", "plan": "pro",
                                       "monthly_points": 5}]}))
    monkeypatch.setenv("TIMESFM3_API_KEYS_FILE", str(f))
    ks = KeyStore.from_env()
    assert set(ks.names()) == {"default", "team", "free", "json"}
    assert ks.lookup("t-key").monthly_points == 1000 and ks.lookup("t-key").plan == "metered"
    assert ks.lookup("f-key").unlimited and ks.lookup("single").plan == "unlimited"
    assert ks.lookup("j-key").plan == "pro" and ks.lookup("nope") is None
    assert not ks.open and KeyStore().open

    monkeypatch.setenv("TIMESFM3_API_KEYS", "broken")
    with pytest.raises(ValueError):
        KeyStore.from_env()
    with pytest.raises(ValueError):
        KeyStore([ApiKey("a", "dup"), ApiKey("b", "dup")])


def test_meter_quota_and_persistence(tmp_path):
    path = tmp_path / "usage.json"
    meter = UsageMeter(str(path))
    key = ApiKey("k", "team", monthly_points=100)
    meter.charge(key, 60, month="2026-09")
    with pytest.raises(QuotaExceeded):
        meter.charge(key, 50, month="2026-09")
    meter.charge(key, 40, month="2026-09")
    assert meter.usage(key, month="2026-09")["points_remaining"] == 0
    # a new month starts fresh
    assert meter.charge(key, 10, month="2026-10")["points"] == 10
    reloaded = UsageMeter(str(path))
    assert reloaded.usage(key, month="2026-09")["points_used"] == 100
    assert reloaded.all_usage(month="2026-09")["team"]["requests"] == 2
    unlimited = ApiKey("u", "unl")
    assert meter.charge(unlimited, 10**9)["points"] == 10**9
    assert meter.usage(unlimited)["monthly_quota"] is None


@pytest.fixture
def metered(registry):
    ks = KeyStore([ApiKey("t-key", "team", plan="team", monthly_points=500),
                   ApiKey("u-key", "unlimited")])
    return TestClient(create_app(registry=registry, keys=ks, meter=UsageMeter()))


def _series(x):
    return {"values": [float(v) for v in x]}


def test_metering_headers_quota_and_usage(metered, seasonal):
    h = {"x-api-key": "t-key"}
    r = metered.post("/v1/forecast", json={"targets": [_series(seasonal[:64])] * 2,
                                           "horizon": 100, "model": "ewma"}, headers=h)
    assert r.status_code == 200
    assert r.headers["x-usage-points"] == "200" and r.headers["x-usage-remaining"] == "300"
    r = metered.post("/v1/forecast", json={"targets": [_series(seasonal[:64])] * 4,
                                           "horizon": 100, "model": "ewma"}, headers=h)
    assert r.status_code == 429 and "quota" in r.json()["detail"].lower()
    assert r.headers["retry-after"] == "86400"
    u = metered.get("/v1/usage", headers=h).json()
    assert u["points_used"] == 200 and u["requests"] == 1 and u["points_remaining"] == 300
    # a rejected request is not charged
    assert metered.get("/v1/usage", headers=h).json()["points_used"] == 200

    r = metered.post("/v1/backtest", json={"series": [_series(seasonal)], "context": 64,
                                           "horizon": 8, "windows": 3, "models": ["ewma"]},
                     headers=h)
    assert r.status_code == 200 and r.headers["x-usage-points"] == str(1 * 3 * 8 * 2)
    r = metered.post("/v1/volatility", json={"returns": list(np.random.default_rng(0).normal(0, 0.01, 200)),
                                             "horizon": 5}, headers=h)
    assert r.status_code == 200 and r.headers["x-usage-points"] == "5"

    unl = metered.post("/v1/forecast", json={"targets": [_series(seasonal[:64])], "horizon": 4000,
                                             "model": "ewma"}, headers={"x-api-key": "u-key"})
    assert unl.status_code == 200 and "x-usage-remaining" not in unl.headers
    assert metered.get("/v1/usage", headers={"x-api-key": "u-key"}).json()["monthly_quota"] is None
    assert metered.get("/v1/usage").status_code == 401


def test_open_mode_meters_anonymously(registry, seasonal):
    c = TestClient(create_app(registry=registry, keys=KeyStore(), meter=UsageMeter()))
    r = c.post("/v1/forecast", json={"targets": [_series(seasonal[:32])], "horizon": 3,
                                     "model": "ewma"})
    assert r.status_code == 200 and r.headers["x-usage-points"] == "3"
    u = c.get("/v1/usage").json()
    assert u["name"] == "anonymous" and u["points_used"] == 3
