import os

import numpy as np
import pytest
import torch

from timesfm3 import TimesFM3Config, TimesFM3Model
from timesfm3.checkpoint import package_checkpoint
from timesfm3.serving.registry import CLASSICAL, ModelRegistry, empirical_quantiles


def test_classical_entries_shape_and_monotone_quantiles(registry, seasonal):
    for name in CLASSICAL:
        r = registry.get(name).forecast([seasonal, seasonal * 2], 12)
        assert r.point.shape == (2, 12)
        assert r.quantiles.shape == (2, 12, 9)
        assert np.all(np.diff(r.quantiles, axis=-1) >= 0)
        # the median band should straddle the point forecast reasonably
        assert np.all(r.quantiles[..., 0] <= r.point + 1e-9)
        assert np.all(r.quantiles[..., -1] >= r.point - 1e-9)


def test_bands_widen_with_horizon(registry, seasonal):
    r = registry.get("last-value").forecast([np.cumsum(np.random.default_rng(1).normal(size=300))], 24)
    width = r.quantiles[0, :, -1] - r.quantiles[0, :, 0]
    assert width[-1] > width[0]


def test_short_context_falls_back_to_gaussian_band():
    from timesfm3.baselines import LastValue

    ctx = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    q = empirical_quantiles(LastValue(), ctx, 3, np.full(3, 5.0))
    assert q.shape == (3, 9)
    assert np.all(np.diff(q, axis=-1) > 0)
    assert q[2, -1] - q[2, 0] > q[0, -1] - q[0, 0]


def test_nan_handling_classical(registry):
    x = np.arange(100, dtype=float)
    x[[0, 10, 50, 99]] = np.nan
    r = registry.get("drift").forecast([x], 5)
    assert np.all(np.isfinite(r.point)) and np.all(np.isfinite(r.quantiles))


def test_timesfm3_entry_with_covariates(registry, seasonal):
    e = registry.get("tiny")
    assert e.kind == "timesfm3" and e.supports_covariates and e.parameters > 0
    r = e.forecast([seasonal[:256]], 40, past_covariates=[seasonal[:256] * 0.5],
                   future_covariates=[np.ones(296)])
    assert r.point.shape == (1, 40) and r.quantiles.shape == (1, 40, 9)
    assert np.all(np.diff(r.quantiles, axis=-1) >= 0)


def test_default_selection_and_lookup(registry):
    assert registry.default == "tiny"
    assert "ewma" in registry and len(registry) == 7
    with pytest.raises(KeyError):
        registry.get("nope")
    info = {d["name"]: d for d in registry.describe()}
    assert info["tiny"]["default"] and not info["ewma"]["default"]


def test_from_env_named_checkpoints(tmp_path, monkeypatch):
    cfg = TimesFM3Config.tiny()
    torch.manual_seed(0)
    raw = tmp_path / "raw.pt"
    torch.save({"config": cfg, "model": TimesFM3Model(cfg).state_dict()}, raw)
    package_checkpoint(str(raw), str(tmp_path / "alpha.pt"), meta={"name": "alpha"})
    package_checkpoint(str(raw), str(tmp_path / "beta.pt"))

    monkeypatch.setenv("TIMESFM3_CHECKPOINTS", f"{tmp_path / 'alpha.pt'}")
    monkeypatch.setenv("TIMESFM3_DEFAULT_MODEL", "ewma")
    reg = ModelRegistry.from_env(checkpoints=[f"custom={tmp_path / 'beta.pt'}"],
                                 include_bundled=False, device="cpu")
    assert set(reg.names()) >= {"alpha", "custom", "ewma"}
    assert reg.default == "ewma"
    assert reg.get("custom").meta["path"].endswith("beta.pt")

    monkeypatch.setenv("TIMESFM3_DEFAULT_MODEL", "missing")
    with pytest.raises(KeyError):
        ModelRegistry.from_env(include_bundled=False)
    monkeypatch.delenv("TIMESFM3_DEFAULT_MODEL")
    monkeypatch.delenv("TIMESFM3_CHECKPOINTS")

    monkeypatch.setenv("TIMESFM3_MODEL_DIR", str(tmp_path))
    reg = ModelRegistry.from_env(include_bundled=False, device="cpu")
    assert "alpha" in reg and "beta" in reg
    assert reg.get().kind == "timesfm3"
