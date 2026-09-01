"""Statistics for comparing forecasters.

Forecast comparisons violate the assumptions of a naive t-test twice over.
Sliding windows overlap, so their errors are serially correlated; and windows
drawn from the same series are not independent of each other. Both inflate
apparent significance, so a raw win rate or an unqualified ratio of mean errors
says very little on its own. This module supplies what a comparison actually
needs:

- :func:`diebold_mariano` -- the standard test for equal predictive accuracy,
  with a Newey-West (HAC) variance so that h-step overlapping forecasts do not
  masquerade as independent draws.
- :func:`paired_bootstrap` -- a cluster (block) bootstrap that resamples whole
  *groups* (e.g. markets) rather than individual windows, so within-group
  dependence is preserved in the resampling.
- :func:`holm` -- Holm-Bonferroni, because testing every channel and reporting
  the one that won is a multiple-comparisons problem.
- :func:`effective_sample_size` -- how many independent observations a
  correlated series is really worth.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np


@dataclasses.dataclass
class Comparison:
    """Result of comparing one forecaster's losses against a reference."""

    name: str
    reference: str
    mean_loss: float
    reference_loss: float
    ratio: float
    ci_low: float
    ci_high: float
    p_value: float
    win_rate: float
    n: int
    n_effective: float
    p_adjusted: float | None = None

    @property
    def significant(self) -> bool:
        """Significant at 5% after correction, if a correction was applied."""
        p = self.p_adjusted if self.p_adjusted is not None else self.p_value
        return p < 0.05

    @property
    def verdict(self) -> str:
        if not self.significant:
            return "no difference"
        return "better" if self.ratio < 1.0 else "worse"


def newey_west_variance(x: np.ndarray, lags: int | None = None) -> float:
    """HAC (Bartlett-kernel) variance of the mean of a correlated series."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 2:
        return float("nan")
    if lags is None:
        # Standard rule of thumb; also the usual choice for h-step forecasts.
        lags = max(1, int(round(n ** (1.0 / 3.0))))
    lags = min(lags, n - 1)
    d = x - x.mean()
    gamma0 = float(np.dot(d, d)) / n
    total = gamma0
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = float(np.dot(d[lag:], d[:-lag])) / n
        total += 2.0 * weight * gamma
    return max(total, 0.0) / n


def diebold_mariano(
    loss_a: np.ndarray, loss_b: np.ndarray, lags: int | None = None
) -> tuple[float, float]:
    """Test equal predictive accuracy of two forecasters.

    Takes per-window losses (lower is better) and returns ``(statistic,
    two-sided p-value)`` for the null that their expected losses are equal. A
    negative statistic favours ``loss_a``. The variance of the loss
    differential is HAC-corrected, which is what makes the test usable on
    overlapping windows.
    """
    d = np.asarray(loss_a, dtype=np.float64) - np.asarray(loss_b, dtype=np.float64)
    d = d[np.isfinite(d)]
    if len(d) < 3:
        return float("nan"), float("nan")
    if np.allclose(d, 0.0):
        return 0.0, 1.0
    var = newey_west_variance(d, lags)
    if var <= 0:
        # A differential with no variance but a non-zero mean is the most
        # decisive case there is -- one forecaster is uniformly better -- not a
        # degenerate one. Returning NaN here would silently discard it.
        sign = math.copysign(1.0, float(d.mean()))
        return sign * float("inf"), 0.0
    stat = float(d.mean() / math.sqrt(var))
    # Normal approximation; with the sample sizes here the t correction is
    # immaterial next to the dependence the HAC term is absorbing.
    p = math.erfc(abs(stat) / math.sqrt(2.0))
    return stat, p


def effective_sample_size(x: np.ndarray, max_lag: int | None = None) -> float:
    """Independent-observation equivalent of an autocorrelated series.

    ``n / (1 + 2 * sum_k rho_k)``, truncated at the first non-positive
    autocorrelation. Overlapping forecast windows routinely give an effective
    size a fraction of the nominal one, which is the honest denominator for a
    standard error.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float(n)
    d = x - x.mean()
    denom = float(np.dot(d, d))
    if denom <= 0:
        return float(n)
    max_lag = max_lag or min(n - 1, 50)
    total = 0.0
    for lag in range(1, max_lag + 1):
        rho = float(np.dot(d[lag:], d[:-lag])) / denom
        if rho <= 0:
            break
        total += rho
    return float(n / (1.0 + 2.0 * total))


def paired_bootstrap(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    groups: np.ndarray | None = None,
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for the ratio ``mean(loss_a) / mean(loss_b)``.

    When ``groups`` is given, whole groups are resampled with replacement (a
    cluster bootstrap), which preserves dependence between windows drawn from
    the same market. Returns ``(ratio, ci_low, ci_high)`` at 95%.
    """
    a = np.asarray(loss_a, dtype=np.float64)
    b = np.asarray(loss_b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) == 0 or b.mean() == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(a.mean() / b.mean())

    rng = np.random.default_rng(seed)
    n = len(a)
    ratios = np.empty(resamples, dtype=np.float64)
    if groups is None:
        # Ordinary paired bootstrap: resample observations with replacement.
        for r in range(resamples):
            idx = rng.integers(0, n, n)
            denom = b[idx].mean()
            ratios[r] = a[idx].mean() / denom if denom != 0 else np.nan
    else:
        g = np.asarray(groups)[ok]
        idx_by_group = [np.flatnonzero(g == u) for u in np.unique(g)]
        n_groups = len(idx_by_group)
        for r in range(resamples):
            pick = rng.integers(0, n_groups, n_groups)
            idx = np.concatenate([idx_by_group[i] for i in pick])
            denom = b[idx].mean()
            ratios[r] = a[idx].mean() / denom if denom != 0 else np.nan
    ratios = ratios[np.isfinite(ratios)]
    if len(ratios) == 0:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down adjustment for a family of tests."""
    items = [(k, v) for k, v in p_values.items() if np.isfinite(v)]
    if not items:
        return {k: float("nan") for k in p_values}
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[key] = running
    for k, v in p_values.items():
        adjusted.setdefault(k, float("nan"))
    return adjusted


def compare(
    losses: dict[str, np.ndarray],
    reference: str,
    groups: np.ndarray | None = None,
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Comparison]:
    """Compare every entry of ``losses`` against ``losses[reference]``.

    Holm correction is applied across the comparisons in this family.
    """
    if reference not in losses:
        raise KeyError(f"reference {reference!r} not among {sorted(losses)}")
    ref = np.asarray(losses[reference], dtype=np.float64)
    out: dict[str, Comparison] = {}
    raw_p: dict[str, float] = {}
    for name, loss in losses.items():
        if name == reference:
            continue
        arr = np.asarray(loss, dtype=np.float64)
        ratio, lo, hi = paired_bootstrap(
            arr, ref, groups, resamples=resamples, seed=seed
        )
        _, p = diebold_mariano(arr, ref)
        raw_p[name] = p
        ok = np.isfinite(arr) & np.isfinite(ref)
        out[name] = Comparison(
            name=name,
            reference=reference,
            mean_loss=float(arr[ok].mean()) if ok.any() else float("nan"),
            reference_loss=float(ref[ok].mean()) if ok.any() else float("nan"),
            ratio=ratio,
            ci_low=lo,
            ci_high=hi,
            p_value=p,
            win_rate=float((arr[ok] < ref[ok]).mean()) if ok.any() else float("nan"),
            n=int(ok.sum()),
            n_effective=effective_sample_size(arr[ok] - ref[ok]),
        )
    for name, p_adj in holm(raw_p).items():
        if name in out:
            out[name].p_adjusted = p_adj
    return out
