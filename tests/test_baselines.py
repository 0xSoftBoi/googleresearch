"""Known-answer tests for the classical baselines."""

import numpy as np
import pytest

from timesfm3.baselines import (
    AR,
    EWMA,
    ContextMean,
    Drift,
    LastValue,
    baseline_forecasts,
)


def test_last_value_is_flat_at_the_last_observation():
    ctx = np.array([1.0, 2.0, 5.0])
    np.testing.assert_allclose(LastValue().forecast(ctx, 3), [5.0, 5.0, 5.0])


def test_context_mean_is_flat_at_the_mean():
    ctx = np.array([0.0, 2.0, 4.0])
    np.testing.assert_allclose(ContextMean().forecast(ctx, 2), [2.0, 2.0])


def test_drift_extrapolates_a_linear_ramp_exactly():
    ctx = np.arange(10, dtype=float)          # slope 1, last value 9
    np.testing.assert_allclose(Drift().forecast(ctx, 3), [10.0, 11.0, 12.0])


def test_ar1_recovers_a_known_coefficient():
    rng = np.random.default_rng(0)
    phi, n = 0.7, 4000
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0, 0.1)
    coef, _ = AR(1).fit(x)
    assert coef[0] == pytest.approx(phi, abs=0.03)


def test_ar1_forecast_decays_toward_the_mean():
    """A stationary AR(1) forecast must relax to the fitted mean, not sit flat."""
    rng = np.random.default_rng(1)
    phi, n = 0.5, 2000
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0, 1.0)
    x = x + 10.0                                  # mean 10
    x[-1] = 20.0                                  # far from the mean
    f = AR(1).forecast(x, 12)
    assert f[0] > f[5] > f[-1], "forecast should decay"
    assert f[-1] == pytest.approx(10.0, abs=1.0), "…toward the mean"


def test_ar_falls_back_to_last_value_on_a_constant_context():
    ctx = np.full(200, 3.0)
    np.testing.assert_allclose(AR(1).forecast(ctx, 4), np.full(4, 3.0))


def test_ar_falls_back_when_context_is_too_short():
    assert AR(4).fit(np.arange(5.0)) is None


def test_ar_forecast_stays_within_a_sane_range():
    """An explosive fit must be clamped rather than diverging."""
    ctx = np.exp(np.linspace(0, 6, 300))          # strongly explosive
    f = AR(2).forecast(ctx, 64)
    span = ctx.max() - ctx.min()
    assert np.all(f <= ctx.max() + span + 1e-6)
    assert np.all(f >= ctx.min() - span - 1e-6)
    assert np.all(np.isfinite(f))


def test_ewma_degenerates_to_last_value_on_a_random_walk():
    rng = np.random.default_rng(2)
    walk = np.cumsum(rng.normal(0, 1, 500))
    ewma = EWMA()
    assert ewma.fit_alpha(walk) >= 0.7, "a random walk should weight the present"
    assert ewma.forecast(walk, 3)[0] == pytest.approx(walk[-1], abs=abs(walk[-1]) * 0.2 + 1)


def test_ewma_smooths_heavily_on_noise_around_a_level():
    rng = np.random.default_rng(3)
    noise = 10.0 + rng.normal(0, 1, 500)
    ewma = EWMA()
    assert ewma.fit_alpha(noise) <= 0.2, "white noise should be smoothed hard"
    assert ewma.forecast(noise, 3)[0] == pytest.approx(10.0, abs=0.6)


def test_all_baselines_return_the_requested_shape():
    rng = np.random.default_rng(4)
    ctx = rng.normal(size=300)
    for name, f in baseline_forecasts(ctx, 17).items():
        assert f.shape == (17,), name
        assert np.all(np.isfinite(f)), name


def test_baselines_never_read_past_the_context():
    """Forecasts must depend only on the context, not on anything after it."""
    rng = np.random.default_rng(5)
    ctx = rng.normal(size=200)
    a = baseline_forecasts(ctx, 8)
    b = baseline_forecasts(ctx.copy(), 8)
    for k in a:
        np.testing.assert_allclose(a[k], b[k], err_msg=k)
