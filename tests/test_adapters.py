"""Proves the registry's adapter claim: a competitor's real weights, not a stub.

Runs the actual Amazon Chronos-Bolt pipeline (the smallest public checkpoint,
~9M parameters) through ``ChronosAdapter`` and the registry, end to end.
Skipped if ``chronos-forecasting`` is not installed or the checkpoint cannot
be fetched (no network) -- the point is to be honest when it runs, not to
require network access for the rest of the suite.
"""

import numpy as np
import pytest

chronos = pytest.importorskip("chronos", reason="optional 'chronos-forecasting' package not installed")

from timesfm3.serving.adapters import ChronosAdapter, add_chronos  # noqa: E402
from timesfm3.serving.registry import DEFAULT_QUANTILES, ModelRegistry  # noqa: E402

MODEL_ID = "amazon/chronos-bolt-tiny"


@pytest.fixture(scope="module")
def chronos_adapter():
    try:
        return ChronosAdapter(MODEL_ID, device="cpu")
    except Exception as exc:  # pragma: no cover - environment without network
        pytest.skip(f"could not fetch {MODEL_ID}: {exc}")


def test_chronos_adapter_forecasts_real_weights(chronos_adapter, seasonal):
    result = chronos_adapter.forecast([seasonal[:200], seasonal[:150] * 2.0], horizon=12)
    assert result.point.shape == (2, 12)
    assert result.quantiles.shape == (2, 12, 9)
    assert result.quantile_levels == DEFAULT_QUANTILES
    # a real forecast on a real model: finite, and quantiles are sorted.
    assert np.all(np.isfinite(result.point)) and np.all(np.isfinite(result.quantiles))
    assert np.all(np.diff(result.quantiles, axis=-1) >= 0)


def test_chronos_adapter_rejects_covariates(chronos_adapter, seasonal):
    with pytest.raises(ValueError, match="univariate"):
        chronos_adapter.forecast([seasonal[:100]], horizon=5, past_covariates=[seasonal[:100]])


def test_add_chronos_registers_a_real_forecaster_alongside_timesfm3():
    reg = ModelRegistry()
    try:
        entry = add_chronos(reg, MODEL_ID, name="chronos-tiny", device="cpu")
    except Exception as exc:  # pragma: no cover - environment without network
        pytest.skip(f"could not fetch {MODEL_ID}: {exc}")
    assert entry.kind == "chronos"
    assert entry.supports_covariates is False
    assert entry.parameters > 0
    assert "chronos-tiny" in reg
    # a foreign model becomes the registry's default over the classical baselines,
    # exactly like a TimesFM-3 checkpoint would (ModelRegistry.add_checkpoint).
    assert reg.default == "chronos-tiny"
    r = reg.get("chronos-tiny").forecast([np.sin(np.linspace(0, 10, 120))], horizon=6)
    assert r.point.shape == (1, 6)
