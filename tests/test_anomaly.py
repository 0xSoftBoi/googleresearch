import numpy as np
import pytest
from fastapi.testclient import TestClient

from timesfm3.anomaly import detect_anomalies
from timesfm3.serving.app import create_app


@pytest.fixture(scope="module")
def spiky():
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.normal(0, 0.2, 400)) + 50
    x[250] += 8.0
    x[320] -= 8.0
    x[100] = np.nan
    return x


def test_detect_anomalies_flags_planted_points(registry, spiky):
    rep = detect_anomalies(registry.get("last-value"), spiky, context=48, block=8, threshold=2.0)
    assert rep.scores.shape == spiky.shape
    assert np.isnan(rep.scores[:48]).all() and np.isnan(rep.scores[100])
    assert rep.flagged[250] and rep.flagged[320]
    assert rep.flagged.sum() <= 6  # a few false alarms at most on a random walk
    items = rep.anomalies(spiky, [f"t{i}" for i in range(len(spiky))])
    by_idx = {a["index"]: a for a in items}
    assert by_idx[250]["direction"] == "high" and by_idx[320]["direction"] == "low"
    assert by_idx[250]["timestamp"] == "t250" and by_idx[250]["score"] > 2
    assert by_idx[250]["lower"] <= by_idx[250]["expected"] <= by_idx[250]["upper"]


def test_detect_anomalies_validation(registry):
    e = registry.get("ewma")
    with pytest.raises(ValueError):
        detect_anomalies(e, np.arange(8.0), context=8)
    with pytest.raises(ValueError):
        detect_anomalies(e, np.arange(50.0), context=4)
    with pytest.raises(ValueError):
        detect_anomalies(e, np.arange(50.0), context=8, block=0)
    # an all-NaN context is skipped, not an error
    x = np.arange(60.0)
    x[:16] = np.nan
    rep = detect_anomalies(e, x, context=16, block=8)
    assert np.isnan(rep.scores[16:24]).all() and np.isfinite(rep.scores[24:]).all()


def test_anomaly_endpoint(registry, spiky):
    c = TestClient(create_app(registry=registry, max_series=4))
    stamps = [str(np.datetime64("2024-01-01") + np.timedelta64(i, "h")) for i in range(len(spiky))]
    body = {"series": [{"name": "rw", "values": [None if np.isnan(v) else float(v) for v in spiky]}],
            "model": "last-value", "context": 48, "block": 8, "timestamps": stamps,
            "include_scores": True}
    r = c.post("/v1/anomalies", json=body)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["model"] == "last-value" and j["threshold"] == 2.0
    s = j["series"][0]
    assert s["name"] == "rw" and s["n_flagged"] == len(s["anomalies"]) >= 2
    assert {a["index"] for a in s["anomalies"]} >= {250, 320}
    assert s["anomalies"][0]["timestamp"].startswith("2024-01")
    assert len(s["scores"]) == len(spiky) and s["scores"][0] is None and s["scores"][100] is None
    assert r.headers["x-usage-points"] == str(len(spiky) - 48)

    lean = c.post("/v1/anomalies", json={**body, "include_scores": False}).json()["series"][0]
    assert lean["scores"] is None
    assert c.post("/v1/anomalies", json={**body, "model": "ghost"}).status_code == 404
    assert c.post("/v1/anomalies", json={**body, "timestamps": ["x"]}).status_code == 400
    assert c.post("/v1/anomalies", json={**body, "context": 1000}).status_code == 400
    assert c.post("/v1/anomalies", json={**body, "series": body["series"] * 5}).status_code == 413
    with_tiny = c.post("/v1/anomalies", json={**body, "model": "tiny", "context": 64, "block": 32})
    assert with_tiny.status_code == 200
