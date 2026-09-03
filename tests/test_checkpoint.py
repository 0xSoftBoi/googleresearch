import numpy as np
import torch

from timesfm3 import TimesFM3Config, TimesFM3Forecaster, TimesFM3Model
from timesfm3.checkpoint import load_checkpoint, package_checkpoint


def test_package_halves_and_loads(tmp_path):
    cfg = TimesFM3Config.tiny()
    torch.manual_seed(1)
    model = TimesFM3Model(cfg)
    raw = tmp_path / "train.pt"
    torch.save({"config": cfg, "model": model.state_dict()}, raw)
    packed = tmp_path / "packed.pt"
    meta = package_checkpoint(str(raw), str(packed), meta={"name": "unit", "steps": 3})
    assert meta["dtype"] == "float16" and meta["name"] == "unit"
    assert packed.stat().st_size < 0.6 * raw.stat().st_size

    state = load_checkpoint(str(packed))
    assert state["meta"]["steps"] == 3
    assert all(v.dtype == torch.float32 for v in state["model"].values()
               if torch.is_floating_point(v))

    a = TimesFM3Forecaster(cfg, model=model, device="cpu")
    b = TimesFM3Forecaster.from_checkpoint(str(packed), device="cpu")
    assert b.meta["name"] == "unit"
    x = np.sin(np.arange(96) / 5.0).astype(np.float32)
    fa, fb = a.forecast([x], 16), b.forecast([x], 16)
    assert np.allclose(fa.point, fb.point, atol=2e-2, rtol=2e-2)


def test_legacy_checkpoint_has_empty_meta(tmp_path):
    cfg = TimesFM3Config.tiny()
    path = tmp_path / "legacy.pt"
    torch.save({"config": cfg, "model": TimesFM3Model(cfg).state_dict()}, path)
    f = TimesFM3Forecaster.from_checkpoint(str(path), device="cpu")
    assert f.meta == {}
