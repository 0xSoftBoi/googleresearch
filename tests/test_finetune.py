import numpy as np
import pytest
import torch

from timesfm3 import TimesFM3Config, TimesFM3Forecaster, TimesFM3Model
from timesfm3.checkpoint import package_checkpoint
from timesfm3.cli import main
from timesfm3.data.real import RealSource, RealWindowDataset
from timesfm3.finetune import finetune


@pytest.fixture(scope="module")
def base_ckpt(tmp_path_factory):
    cfg = TimesFM3Config.tiny()
    torch.manual_seed(0)
    d = tmp_path_factory.mktemp("ft")
    raw = d / "raw.pt"
    torch.save({"config": cfg, "model": TimesFM3Model(cfg).state_dict()}, raw)
    packed = d / "base.pt"
    package_checkpoint(str(raw), str(packed), meta={"name": "tiny-base"})
    return str(packed)


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(0)
    t = np.arange(1200)
    a = 10 + 3 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.3, len(t))
    b = 5 + 2 * np.cos(2 * np.pi * t / 24) + rng.normal(0, 0.3, len(t))
    return np.stack([a, b])


def test_tail_windows_never_touch_training_region(panel):
    cfg = TimesFM3Config.tiny()
    src = RealSource("p", panel.astype(np.float32), periods=())
    tail = RealWindowDataset(cfg, [src], context_patches=2, horizon_patches=1,
                             train_fraction=0.8, tail=True, seed=1, strides=(1,))
    head = RealWindowDataset(cfg, [src], context_patches=2, horizon_patches=1,
                             train_fraction=0.8, seed=1, strides=(1,))
    split = int(1200 * 0.8)
    tail_vals = panel[:, split:]
    it = iter(tail)
    for _ in range(20):
        v = next(it)["values"][0].numpy()
        # every tail window must be found verbatim in the tail region
        assert any(np.allclose(tail_vals[c, i:i + len(v)], v)
                   for c in range(2) for i in range(tail_vals.shape[1] - len(v) + 1))
    with pytest.raises(ValueError):
        next(iter(RealWindowDataset(cfg, [src], context_patches=40, horizon_patches=1,
                                    train_fraction=0.99, tail=True, strides=(1,))))


def test_finetune_writes_package_and_evaluates(base_ckpt, panel, tmp_path):
    out = tmp_path / "ft.pt"
    rep = finetune(panel, base_ckpt, str(out), name="ft", steps=4, batch_size=4,
                   context_patches=2, horizon_patches=1, eval_windows=3, device="cpu",
                   verbose=False, periods=(24,))
    assert out.exists() and not (tmp_path / "ft.pt.train.pt").exists()
    assert rep.steps == 4 and np.isfinite(rep.best_val_loss)
    f = TimesFM3Forecaster.from_checkpoint(str(out), device="cpu")
    assert f.meta["name"] == "ft" and f.meta["base_checkpoint"] == base_ckpt
    assert f.meta["finetune_series"] == 2
    ev = rep.evaluation
    assert "error" not in ev
    assert {s["model"] for s in ev["scores"]} == {"base", "ft", "ewma", "ar4", "last-value"}
    assert ev["context"] == 64 and ev["horizon"] == 32

    with pytest.raises(ValueError):
        finetune(panel[:, :100], base_ckpt, str(tmp_path / "short.pt"), steps=1,
                 context_patches=2, horizon_patches=1, device="cpu", verbose=False)


def test_finetune_cli(base_ckpt, panel, tmp_path, capsys):
    p = tmp_path / "panel.csv"
    p.write_text("a,b\n" + "\n".join(f"{x:.4f},{y:.4f}" for x, y in panel.T) + "\n")
    out = tmp_path / "cli.pt"
    assert main(["finetune", str(p), "--out", str(out), "--from", base_ckpt, "--name", "mine",
                 "--steps", "3", "--batch-size", "4", "--context-patches", "2",
                 "--horizon-patches", "1", "--windows", "3", "--device", "cpu"]) == 0
    text = capsys.readouterr().out
    assert "fine-tuned vs base" in text and "serve it:" in text and out.exists()
    assert main(["models", "--no-bundled", "-c", f"mine={out}", "--device", "cpu"]) == 0
    assert "* mine" in capsys.readouterr().out


def test_anomalies_cli(tmp_path, capsys):
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(0, 0.1, 200)) + 20
    x[150] += 5
    p = tmp_path / "s.csv"
    p.write_text("t,x\n" + "\n".join(
        f"{np.datetime64('2024-01-01') + np.timedelta64(i, 'D')},{v:.4f}" for i, v in enumerate(x)
    ) + "\n")
    assert main(["anomalies", str(p), "--model", "last-value", "--context", "32", "--block", "8",
                 "--no-bundled"]) == 0
    text = capsys.readouterr().out
    assert "anomalies in" in text and "2024-05-30" in text  # index 150 as a date
    assert main(["anomalies", str(p), "--model", "last-value", "--context", "32", "--block", "8",
                 "--no-bundled", "--json"]) == 0
    import json
    j = json.loads(capsys.readouterr().out)
    assert j["series"][0]["name"] == "x" and any(a["index"] == 150 for a in j["series"][0]["anomalies"])
