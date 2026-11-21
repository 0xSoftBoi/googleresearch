"""Proves the registry's adapter claim: a competitor's real weights, not a stub.

Runs the actual Amazon Chronos-Bolt pipeline (the smallest public checkpoint,
~9M parameters) through ``ChronosAdapter`` and the registry, end to end.
The real-weight tests skip if ``chronos-forecasting`` is not installed or the
checkpoint cannot be fetched (no network); the output-shape tests use stub
pipelines and always run -- the point is to be honest when it runs, not to
require network access for the rest of the suite.
"""

import numpy as np
import pytest

from timesfm3.serving.adapters import ChronosAdapter, add_chronos
from timesfm3.serving.registry import DEFAULT_QUANTILES, ModelRegistry

needs_chronos = pytest.mark.skipif(
    __import__("importlib").util.find_spec("chronos") is None,
    reason="optional 'chronos-forecasting' package not installed",
)

MODEL_ID = "amazon/chronos-bolt-tiny"


@pytest.fixture(scope="module")
def chronos_adapter():
    pytest.importorskip("chronos")
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


@needs_chronos
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


class _StubPipeline:
    """Mimics the two predict_quantiles return conventions without weights."""

    def __init__(self, listwise: bool):
        import torch

        self.listwise = listwise
        self.model = torch.nn.Linear(2, 2)

    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        import torch

        q = torch.stack([
            torch.stack([torch.full((prediction_length,), float(i) + lv) for lv in quantile_levels], -1)
            for i in range(len(inputs))
        ])  # (C, H, Q)
        m = q[..., len(quantile_levels) // 2]
        if self.listwise:  # Chronos-2: one (1, H, Q) / (1, H) tensor per input
            return [q[i : i + 1] for i in range(len(inputs))], [m[i : i + 1] for i in range(len(inputs))]
        return q, m  # Chronos / Chronos-Bolt: stacked tensors


@pytest.mark.parametrize("listwise", [False, True], ids=["bolt-stacked", "chronos2-listwise"])
def test_adapter_normalizes_both_pipeline_output_shapes(listwise):
    adapter = ChronosAdapter.__new__(ChronosAdapter)
    adapter.model_id, adapter.pipeline = "stub", _StubPipeline(listwise)
    adapter.model = adapter.pipeline.model
    r = adapter.forecast([np.zeros(50), np.ones(40), np.ones(30)], horizon=7)
    assert r.point.shape == (3, 7) and r.quantiles.shape == (3, 7, 9)
    assert np.allclose(r.point[:, 0], [0.5, 1.5, 2.5])  # series order preserved
    assert np.all(np.diff(r.quantiles, axis=-1) > 0)
