"""Known-answer tests for the quant layer: backtester, vol models, strategies."""

import numpy as np
import pytest

from timesfm3.baselines import Drift
from timesfm3.quant import (
    HAR,
    backtest_portfolio,
    ewma_variance,
    forecast_signal_positions,
    performance_stats,
    qlike,
    tsmom_positions,
    vol_managed_weights,
)

TRADING_DAYS = 252


class TestBacktester:
    def test_position_at_t_earns_return_at_t_plus_one(self):
        # One asset, position held only on day 0; only returns[1] should count.
        returns = np.array([[0.0, 0.02, 0.05]])
        positions = np.array([[1.0, 0.0, 0.0]])
        dates = np.arange("2020-01-01", "2020-01-04", dtype="datetime64[D]")
        res = backtest_portfolio(positions, returns, dates, cost_bps=0.0)
        assert res.gross_returns == pytest.approx([0.02, 0.0])

    def test_costs_charged_on_position_changes(self):
        returns = np.array([[0.0, 0.0, 0.0]])
        positions = np.array([[1.0, -1.0, -1.0]])  # enter 1, flip to -1
        dates = np.arange("2020-01-01", "2020-01-04", dtype="datetime64[D]")
        res = backtest_portfolio(positions, returns, dates, cost_bps=100.0)
        # day 0: |0 -> 1| = 1 unit traded at 1%; day 1: |1 -> -1| = 2 units.
        assert res.net_returns == pytest.approx([-0.01, -0.02])

    def test_equal_risk_divides_by_live_assets(self):
        returns = np.array([[0.0, 0.10], [0.0, 0.10]])
        positions = np.array([[1.0, 1.0], [1.0, 1.0]])
        dates = np.arange("2020-01-01", "2020-01-03", dtype="datetime64[D]")
        res = backtest_portfolio(positions, returns, dates, cost_bps=0.0)
        assert res.gross_returns == pytest.approx([0.10])  # mean, not sum

    def test_performance_stats_on_constant_returns(self):
        r = np.full(504, 0.001)
        s = performance_stats(r)
        assert s["ann_return"] == pytest.approx(0.252)
        assert s["max_drawdown"] == 0.0
        assert s["hit_rate"] == 1.0


class TestVolatility:
    def test_ewma_is_one_step_ahead(self):
        # A shock at day t must not affect the forecast *for* day t.
        r = np.zeros(100)
        r[50] = 0.10
        var = ewma_variance(r, lam=0.9)
        assert var[50] < 1e-4  # forecast made before the shock
        assert var[51] > var[50] * 10  # shock enters the next day's forecast

    def test_ewma_handles_nan_head(self):
        r = np.concatenate([np.full(30, np.nan), np.random.default_rng(0).normal(0, 0.01, 200)])
        var = ewma_variance(r)
        assert np.isnan(var[:30]).all()
        assert np.isfinite(var[35:]).all()

    def test_qlike_zero_at_truth_and_positive_elsewhere(self):
        assert qlike(2.0, 2.0) == pytest.approx(0.0)
        assert qlike(1.0, 2.0) > 0 and qlike(4.0, 2.0) > 0

    def test_qlike_punishes_underforecast_more(self):
        assert qlike(1.0, 4.0) > qlike(16.0, 4.0)  # 4x under vs 4x over

    def test_har_tracks_a_persistent_level_shift(self):
        rng = np.random.default_rng(1)
        low = rng.uniform(0.9, 1.1, 300)
        high = rng.uniform(4.5, 5.5, 300)
        rv = np.concatenate([low, high])
        pred = HAR().forecast(rv, horizon=5)
        assert np.all(pred > 3.0)  # forecasts the new regime, not the old

    def test_har_constant_series_predicts_the_constant(self):
        rv = np.full(300, 2.5)
        pred = HAR().forecast(rv, horizon=3)
        assert pred == pytest.approx(np.full(3, 2.5), rel=1e-3)


class TestStrategies:
    def test_tsmom_goes_long_a_steady_uptrend(self):
        rng = np.random.default_rng(2)
        # Strong trend (annual Sharpe ~3) so the trailing-sum sign is
        # deterministic, not a coin flip on the seed.
        r = (0.002 + 0.01 * rng.standard_normal(2000))[None, :]
        pos = tsmom_positions(r, lookback=252, vol_target=0.10)
        live = np.isfinite(pos[0])
        assert live.sum() > 500
        assert (pos[0][live] > 0).mean() > 0.9

    def test_tsmom_vol_targets_position_size(self):
        rng = np.random.default_rng(3)
        daily_vol = 0.02
        r = (0.001 + daily_vol * rng.standard_normal(3000))[None, :]
        pos = tsmom_positions(r, lookback=252, vol_target=0.10)
        live = np.isfinite(pos[0])
        expected_lev = 0.10 / (daily_vol * np.sqrt(TRADING_DAYS))
        assert np.nanmean(np.abs(pos[0][live])) == pytest.approx(expected_lev, rel=0.15)

    def test_vol_managed_weights_capped_and_positive(self):
        rng = np.random.default_rng(4)
        r = 0.001 * rng.standard_normal(500)  # very quiet -> would want huge leverage
        w = vol_managed_weights(r, vol_target=0.20, max_leverage=3.0)
        finite = w[np.isfinite(w)]
        assert np.all(finite <= 3.0) and np.all(finite >= 0.0)

    def test_forecast_signal_follows_drift(self):
        rng = np.random.default_rng(5)
        r = (0.001 + 0.01 * rng.standard_normal(1500))[None, :]
        pos = forecast_signal_positions(r, Drift(), context_len=252, horizon=21)
        live = np.isfinite(pos[0])
        assert live.sum() > 300
        assert (pos[0][live] > 0).mean() > 0.9

    def test_zero_forecast_holds_zero_position(self):
        class Flat(Drift):
            def forecast(self, context, horizon):
                return np.full(horizon, context[-1])

        rng = np.random.default_rng(6)
        r = (0.01 * rng.standard_normal(1500))[None, :]
        pos = forecast_signal_positions(r, Flat(), context_len=252, horizon=21)
        live = np.isfinite(pos[0])
        assert np.nanmax(np.abs(pos[0][live])) < 1e-9
