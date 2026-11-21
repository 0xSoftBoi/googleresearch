import numpy as np
import pytest
import torch

from timesfm3 import TimesFM3Config, TimesFM3Forecaster
from timesfm3.quant import qlike
from timesfm3.quant import volmodel as vm


def _asset(start="2010-01-01", days=4800, seed=0):
    rng = np.random.default_rng(seed)
    dates = np.arange(np.datetime64(start), np.datetime64(start) + days)
    vol = 0.01 * np.exp(np.cumsum(rng.normal(0, 0.05, days)) * 0.1)
    r = rng.normal(0, vol)
    return vm.AssetRV("X", dates, r, r**2)


def test_splits_are_disjoint_and_respect_boundaries():
    a = _asset()
    tr, va, te = (vm.split_origins(a, s) for s in ("train", "val", "test"))
    assert len(set(tr) & set(va)) == 0 and len(set(va) & set(te)) == 0
    assert a.dates[tr + vm.EVAL_HORIZON].max() < vm.TRAIN_END       # train targets end before 2015
    assert a.dates[va].min() >= vm.TRAIN_END
    assert a.dates[va + vm.EVAL_HORIZON].max() < vm.TEST_START      # val targets end before 2020
    assert a.dates[te].min() >= vm.TEST_START
    assert tr.min() >= vm.CONTEXT - 1                               # a full context exists


def test_contexts_and_targets_align_with_origin():
    a = _asset()
    t = vm.split_origins(a, "val")[:3]
    ctx, tgt = vm.contexts(a, t), vm.targets(a, t)
    assert ctx.shape == (3, vm.CONTEXT)
    np.testing.assert_array_equal(ctx[0], a.rv[t[0] - vm.CONTEXT + 1:t[0] + 1])  # ends at day t
    assert tgt[0, 0] == a.rv[t[0] + 1]                                          # next day, not day t
    assert np.isclose(tgt[0, 1], a.rv[t[0] + 1:t[0] + 6].mean())


def test_log_context_floors_at_window_first_percentile():
    ctx = np.array([[1e-12] + [1.0] * 199, [2.0] * 200])
    x = vm.log_context(ctx)
    assert x[0, 0] >= np.log(np.quantile(ctx[0], 0.01)) - 1e-12
    assert np.allclose(x[1], np.log(2.0))


def test_qlike_from_log_matches_qlike():
    p = torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float64)
    y = torch.tensor([0.5, 1.0, 3.0], dtype=torch.float64)
    np.testing.assert_allclose(vm.qlike_from_log(p, y).numpy(), qlike(np.exp(p.numpy()), y.numpy()))


def test_batched_forward_matches_the_forecaster_for_each_series():
    torch.manual_seed(0)
    fc = TimesFM3Forecaster(TimesFM3Config.tiny(), device="cpu")
    fc.model.eval()
    rng = np.random.default_rng(1)
    x = rng.normal(-9, 1, size=(3, 256)).astype(np.float32)
    with torch.no_grad():
        out, length = vm.timesfm_log_forward(fc.model, x)
    batched = out.point[:, 0, length:length + 5, 0].numpy()
    for i in range(3):  # one independent univariate forecast per row
        single = fc.forecast([x[i]], horizon=5).point[0]
        np.testing.assert_allclose(batched[i], single, rtol=1e-4, atol=1e-4)


def test_calibration_and_daily_losses():
    f = vm.Forecasts(
        dates=np.array(["2020-01-02", "2020-01-02", "2020-01-03"], dtype="datetime64[D]"),
        assets=np.array(["A", "B", "A"]),
        target=np.array([[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]]),
        preds={"m": np.array([[2.0, 2.0], [4.0, 4.0], [8.0, 8.0]])},
    )
    cal = vm.fit_calibration(f)
    np.testing.assert_allclose(cal["m"], [0.5, 0.5])  # forecasts were uniformly 2x too high
    dates, losses = vm.daily_losses(f, cal)
    assert len(dates) == 2 and np.allclose(losses["m"], 0.0)  # perfect after calibration


def test_finetune_qlike_lowers_the_training_objective():
    torch.manual_seed(0)
    model = TimesFM3Forecaster(TimesFM3Config.tiny(), device="cpu").model
    a = _asset(start="2000-01-01", days=4000)
    _, hist = vm.finetune_qlike(model, [a], steps=30, batch_size=8, lr=1e-3, warmup=5, log=lambda s: None)
    first, last = np.mean([h["loss"] for h in hist[:5]]), np.mean([h["loss"] for h in hist[-5:]])
    assert np.isfinite(last) and last < first
