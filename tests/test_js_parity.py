"""The Cloudflare Worker's JS port must reproduce the Python numbers.

Runs cloudflare/public/js/forecast.js under Node on the same inputs the
Python registry / backtest see and compares point forecasts, quantile bands,
and the backtest report.  Skipped when Node is not installed.
"""

import json
import shutil
import subprocess

import numpy as np
import pytest

from timesfm3.serving.app import run_backtest
from timesfm3.serving.registry import CLASSICAL, ModelRegistry

JS = "cloudflare/public/js/forecast.js"
node = shutil.which("node")
pytestmark = pytest.mark.skipif(node is None, reason="node not installed")


def _run_js(script: str, payload: dict) -> dict:
    out = subprocess.run(
        [node, "--input-type=module", "-e", script], input=json.dumps(payload),
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(5)
    t = np.arange(700)
    a = 10 + 3 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.4, len(t))
    b = np.cumsum(rng.normal(0, 0.2, len(t))) + 50
    b[[10, 300]] = np.nan
    return [a, b]


def test_baselines_and_bands_match(panel):
    reg = ModelRegistry.from_env(include_bundled=False)
    horizon = 12
    expected = {}
    for name in CLASSICAL:
        r = reg.get(name).forecast(panel, horizon)
        expected[name] = {"point": r.point.tolist(), "quantiles": r.quantiles.tolist()}
    payload = {"series": [[None if not np.isfinite(v) else float(v) for v in s] for s in panel],
               "horizon": horizon, "models": list(CLASSICAL)}
    got = _run_js(f"""
import * as F from './{JS}';
let input=''; process.stdin.on('data',d=>input+=d); process.stdin.on('end',()=>{{
  const p=JSON.parse(input); const out={{}};
  for (const m of p.models) out[m]=F.forecastClassical(m, p.series, p.horizon);
  console.log(JSON.stringify(out));
}});""", payload)
    for name in CLASSICAL:
        np.testing.assert_allclose(got[name]["point"], expected[name]["point"], rtol=1e-6, atol=1e-6, err_msg=name)
        np.testing.assert_allclose(got[name]["quantiles"], expected[name]["quantiles"], rtol=1e-5, atol=1e-5, err_msg=name)


def test_backtest_report_matches(panel):
    reg = ModelRegistry.from_env(include_bundled=False)
    models = ["ewma", "ar4", "drift", "ctx-mean"]
    ref = run_backtest(reg, panel, 96, 24, models, "last-value", 8)
    payload = {"series": [[None if not np.isfinite(v) else float(v) for v in s] for s in panel],
               "models": models}
    got = _run_js(f"""
import * as F from './{JS}';
let input=''; process.stdin.on('data',d=>input+=d); process.stdin.on('end',async()=>{{
  const p=JSON.parse(input);
  const fn=async(m,ctx,h)=>F.forecastClassical(m,[ctx],h,false).point[0];
  console.log(JSON.stringify(await F.runBacktest(fn, p.series, 96, 24, p.models, 'last-value', 8)));
}});""", payload)
    assert got["windows_per_series"] == ref["windows_per_series"]
    assert got["reference_loss"] == pytest.approx(ref["reference_loss"], rel=1e-6)
    by_py = {s["model"]: s for s in ref["scores"]}
    by_js = {s["model"]: s for s in got["scores"]}
    assert set(by_js) == set(by_py)
    for m, s in by_py.items():
        j = by_js[m]
        assert j["mean_loss"] == pytest.approx(s["mean_loss"], rel=1e-6)
        assert j["ratio"] == pytest.approx(s["ratio"], rel=1e-6)
        assert j["n"] == s["n"] and j["n_effective"] == pytest.approx(s["n_effective"], rel=1e-6)
        if s["p_value"] is not None:
            assert j["p_value"] == pytest.approx(s["p_value"], abs=2e-6)   # erfc approximation
            assert j["win_rate"] == pytest.approx(s["win_rate"])
            # different RNGs: bootstrap intervals agree to a few percent
            assert j["ci_low"] == pytest.approx(s["ci_low"], rel=0.05)
            assert j["ci_high"] == pytest.approx(s["ci_high"], rel=0.05)
            assert j["verdict"] == s["verdict"]
