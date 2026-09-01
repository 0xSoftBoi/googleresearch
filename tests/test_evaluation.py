"""Known-answer tests for the forecast-comparison statistics."""

import numpy as np
import pytest

from timesfm3.evaluation import (
    compare,
    diebold_mariano,
    effective_sample_size,
    holm,
    newey_west_variance,
    paired_bootstrap,
)


def test_dm_reports_no_difference_for_identical_losses():
    x = np.random.default_rng(0).random(200)
    stat, p = diebold_mariano(x, x.copy())
    assert stat == 0.0 and p == 1.0


def test_dm_detects_a_large_consistent_difference():
    rng = np.random.default_rng(1)
    a = rng.normal(1.0, 0.1, 400)
    b = rng.normal(2.0, 0.1, 400)
    stat, p = diebold_mariano(a, b)
    assert stat < 0, "negative statistic favours the first argument"
    assert p < 1e-6


def test_dm_is_not_fooled_by_a_tiny_difference_in_noise():
    rng = np.random.default_rng(2)
    a = rng.normal(1.0, 1.0, 300)
    b = a + rng.normal(0.0, 1.0, 300)     # same mean, independent noise
    _, p = diebold_mariano(a, b)
    assert p > 0.05


def test_hac_variance_matches_iid_variance_on_independent_data():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 20000)
    assert newey_west_variance(x) == pytest.approx(x.var() / len(x), rel=0.25)


def test_hac_variance_exceeds_iid_under_positive_autocorrelation():
    """The whole point: correlated data carries less information than n suggests."""
    rng = np.random.default_rng(4)
    n, phi = 20000, 0.8
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0, 1)
    assert newey_west_variance(x, lags=60) > 3.0 * (x.var() / n)


def test_effective_sample_size_is_near_n_for_iid():
    x = np.random.default_rng(5).normal(size=5000)
    assert effective_sample_size(x) == pytest.approx(5000, rel=0.25)


def test_effective_sample_size_collapses_under_strong_autocorrelation():
    rng = np.random.default_rng(6)
    n, phi = 5000, 0.9
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0, 1)
    ess = effective_sample_size(x)
    assert ess < n / 4, f"expected a big reduction, got {ess:.0f}"


def test_bootstrap_point_estimate_is_the_ratio_of_means():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 2.0, 2.0, 2.0])
    ratio, lo, hi = paired_bootstrap(a, b, resamples=500)
    assert ratio == pytest.approx(2.5 / 2.0)
    assert lo <= ratio <= hi


def test_bootstrap_interval_brackets_one_when_forecasters_are_equivalent():
    rng = np.random.default_rng(7)
    a = rng.random(400) + 1.0
    b = rng.random(400) + 1.0
    _, lo, hi = paired_bootstrap(a, b, resamples=1000)
    assert lo < 1.0 < hi


def test_cluster_bootstrap_widens_the_interval_when_groups_are_correlated():
    """Ignoring within-market dependence understates uncertainty."""
    rng = np.random.default_rng(8)
    groups = np.repeat(np.arange(20), 30)
    offsets = rng.normal(0, 0.5, 20)            # a per-group shift
    a = 1.0 + offsets[groups] + rng.normal(0, 0.01, len(groups))
    b = np.full(len(groups), 1.0)
    _, lo_flat, hi_flat = paired_bootstrap(a, b, None, resamples=800, seed=1)
    _, lo_grp, hi_grp = paired_bootstrap(a, b, groups, resamples=800, seed=1)
    assert (hi_grp - lo_grp) > 3.0 * (hi_flat - lo_flat)


def test_holm_adjustment_matches_a_worked_example():
    adj = holm({"a": 0.01, "b": 0.02, "c": 0.03})
    # step-down: 3*0.01=0.03, then max(0.03, 2*0.02)=0.04, then max(.04, 0.03)=0.04
    assert adj["a"] == pytest.approx(0.03)
    assert adj["b"] == pytest.approx(0.04)
    assert adj["c"] == pytest.approx(0.04)


def test_holm_is_monotone_and_never_below_the_raw_p():
    raw = {"a": 0.001, "b": 0.04, "c": 0.2, "d": 0.9}
    adj = holm(raw)
    assert all(adj[k] >= raw[k] for k in raw)
    ordered = sorted(raw, key=raw.get)
    values = [adj[k] for k in ordered]
    assert values == sorted(values), "adjusted p-values must be non-decreasing"


def test_compare_flags_a_real_improvement_and_ignores_noise():
    rng = np.random.default_rng(9)
    ref = rng.normal(1.0, 0.1, 500)
    losses = {
        "reference": ref,
        "better": ref - 0.5,                       # clearly lower loss
        "same": ref + rng.normal(0, 1e-9, 500),    # indistinguishable
    }
    out = compare(losses, reference="reference")
    assert out["better"].verdict == "better"
    assert out["better"].ratio < 1.0
    assert out["same"].verdict == "no difference"


def test_compare_requires_a_known_reference():
    with pytest.raises(KeyError):
        compare({"a": np.ones(5)}, reference="missing")
