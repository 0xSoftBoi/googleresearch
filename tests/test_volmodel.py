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
    base = {k: v.detach().clone() for k, v in model.state_dict().items()}
    state, hist = vm.finetune_qlike(model, [a], steps=30, batch_size=8, lr=1e-3, warmup=5, log=lambda s: None)
    first, last = np.mean([h["loss"] for h in hist[:5]]), np.mean([h["loss"] for h in hist[-5:]])
    assert np.isfinite(last) and last < first
    # Without a validation set the trained weights come back, not the base ones.
    assert any(not torch.equal(state[k], base[k]) for k in base)
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())


def test_finetune_qlike_requires_val_assets_with_val():
    model = TimesFM3Forecaster(TimesFM3Config.tiny(), device="cpu").model
    a = _asset(start="2000-01-01", days=4000)
    f = vm.Forecasts(np.array([a.dates[0]]), np.array(["X"]), np.ones((1, 2)), {"m": np.ones((1, 2))})
    with pytest.raises(ValueError, match="val_assets"):
        vm.finetune_qlike(model, [a], val=f, steps=1, log=lambda s: None)


def test_feature_channels_are_causal():
    a = _asset()
    b = vm.AssetRV("X", a.dates, a.returns, a.rv.copy())
    b.rv[3000:] *= 50.0          # change the future only
    b.features = vm._causal_features(b.rv)
    for ch in ("log_ewma94", "log_mean22"):
        np.testing.assert_allclose(a.features[ch][:3000], b.features[ch][:3000])  # past unaffected
        assert not np.allclose(a.features[ch][3000:3010], b.features[ch][3000:3010])


def test_multichannel_forward_and_forecaster_shapes():
    torch.manual_seed(0)
    model = TimesFM3Forecaster(TimesFM3Config.tiny(), device="cpu").model
    a = _asset()
    t = vm.split_origins(a, "val")[:4]
    x = vm.channel_contexts(a, t, vm.CHANNELS)
    assert x.shape == (4, 4, vm.CONTEXT)
    assert np.isnan(x[:, 3]).all()  # no implied vol for this asset
    cfg = TimesFM3Config.tiny()
    if vm.CONTEXT <= cfg.max_context_len:
        out = vm.timesfm(model, channels=vm.CHANNELS)(a, t)
        assert out.shape == (4, 2) and np.all(out > 0)
    with pytest.raises(ValueError):
        vm.channel_contexts(a, t, ("log_ewma94",))


def test_missing_implied_vol_channel_equals_dropping_it():
    torch.manual_seed(0)
    model = TimesFM3Forecaster(TimesFM3Config.tiny(), device="cpu").model
    model.eval()
    a = _asset()
    t = vm.split_origins(a, "val")[:3]
    three = vm.timesfm(model, channels=vm.CHANNELS[:3])(a, t)
    four = vm.timesfm(model, channels=vm.CHANNELS)(a, t)   # log_iv all NaN -> masked out
    np.testing.assert_allclose(three, four, rtol=1e-5)


def test_implied_vol_competitors():
    a = _asset()
    t = vm.split_origins(a, "val")[:5]
    rm = vm.riskmetrics()(a, t)
    np.testing.assert_allclose(vm.iv_or_riskmetrics()(a, t), rm)       # no IV -> RiskMetrics
    np.testing.assert_allclose(vm.har_iv()(a, t), vm.har()(a, t))       # no IV -> plain HAR
    iv = np.full(len(a.rv), 2e-4)
    a.set_implied_var(iv)
    np.testing.assert_allclose(vm.iv_or_riskmetrics()(a, t), 2e-4)      # IV at the origin
    assert np.isfinite(a.features["log_iv"]).all()
    out = vm.har_iv()(a, t)
    assert out.shape == (5, 2) and np.all(out > 0)


def test_no_future_return_changes_any_earlier_input_target_or_benchmark():
    """Rebuild an asset with every return after day k replaced: nothing up to k may move."""
    rng = np.random.default_rng(3)
    dates = np.arange(np.datetime64("2010-01-01"), np.datetime64("2010-01-01") + 1600)
    r = rng.normal(0, 0.01, 1600)
    r[rng.choice(1600, 40, replace=False)] = 0.0          # zero-return days hit the floor
    k = 900
    r2 = r.copy()
    r2[k + 1:] = rng.normal(0.02, 0.2, 1600 - k - 1)      # wild, positive-mean future
    a1 = vm.AssetRV("X", dates, r, vm.realized_variance(r))
    a2 = vm.AssetRV("X", dates, r2, vm.realized_variance(r2))
    np.testing.assert_array_equal(a1.rv[:k + 1], a2.rv[:k + 1])
    for ch in ("log_ewma94", "log_mean22"):
        np.testing.assert_array_equal(a1.features[ch][:k + 1], a2.features[ch][:k + 1])
    t = np.arange(vm.CONTEXT - 1, k - vm.EVAL_HORIZON + 1)  # targets end at or before day k
    np.testing.assert_array_equal(vm.contexts(a1, t), vm.contexts(a2, t))
    np.testing.assert_array_equal(vm.targets(a1, t), vm.targets(a2, t))
    for fc in (vm.riskmetrics(), vm.har(), vm.har(log_space=True)):
        np.testing.assert_array_equal(fc(a1, t), fc(a2, t))


def test_ewma_variance_is_causal():
    from timesfm3.quant import ewma_variance
    rng = np.random.default_rng(4)
    r = rng.normal(0, 0.01, 800)
    r2 = r.copy()
    r2[500:] += 0.05
    np.testing.assert_array_equal(ewma_variance(r)[:501], ewma_variance(r2)[:501])
