"""Forecast-based anomaly detection.

An observation is anomalous when it falls far outside what the model
expected *before seeing it*.  The series is scanned walk-forward: from each
origin the model forecasts the next ``block`` steps from the preceding
``context`` steps, and every observation in the block is scored against the
forecast distribution for its own step:

    score = (x - median) / (q90 - median)   if x > median
    score = (median - x) / (median - q10)   if x < median

so ``score = 1`` sits exactly on the 80% band edge.  For a Gaussian
predictive distribution q90 - median is 1.28 sigma, so the default threshold
of 2 flags points beyond ~2.6 sigma (about 1% two-sided).  Any registry
model works -- the classical baselines with their empirical bands or a
TimesFM-3 checkpoint -- so the same detector runs on a laptop or a GPU box.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class AnomalyReport:
    scores: np.ndarray        # (T,) NaN where unscored (first `context` steps, NaN inputs)
    expected: np.ndarray      # (T,) predictive median
    lower: np.ndarray         # (T,) q10
    upper: np.ndarray         # (T,) q90
    flagged: np.ndarray       # (T,) bool

    def anomalies(self, values: np.ndarray, timestamps: list[str] | None = None) -> list[dict]:
        out = []
        for i in np.flatnonzero(self.flagged):
            item = {
                "index": int(i), "value": float(values[i]), "expected": float(self.expected[i]),
                "lower": float(self.lower[i]), "upper": float(self.upper[i]),
                "score": float(self.scores[i]),
                "direction": "high" if values[i] > self.expected[i] else "low",
            }
            if timestamps is not None:
                item["timestamp"] = timestamps[i]
            out.append(item)
        return out


def detect_anomalies(
    entry,
    values: np.ndarray,
    context: int = 96,
    block: int = 24,
    threshold: float = 2.0,
) -> AnomalyReport:
    """Scores one series with a registry ``entry`` (anything with ``.forecast``)."""
    x = np.asarray(values, dtype=np.float64)
    n = len(x)
    if context < 8:
        raise ValueError("context must be at least 8 steps.")
    if n < context + 1:
        raise ValueError(f"Series has {n} steps; need more than the context of {context}.")
    if block < 1 or threshold <= 0:
        raise ValueError("block must be >= 1 and threshold > 0.")
    scores = np.full(n, np.nan)
    expected = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    levels = None
    origin = context
    while origin < n:
        h = min(block, n - origin)
        ctx = x[origin - context : origin]
        if np.isfinite(ctx).sum() < 2:
            origin += h
            continue
        result = entry.forecast([ctx], h)
        if levels is None:
            levels = list(result.quantile_levels)
            i_lo, i_med, i_hi = levels.index(min(levels)), levels.index(0.5), levels.index(max(levels))
        q = result.quantiles[0]  # (h, Q)
        med, lo, hi = q[:, i_med], q[:, i_lo], q[:, i_hi]
        obs = x[origin : origin + h]
        up = np.maximum(hi - med, 1e-12)
        dn = np.maximum(med - lo, 1e-12)
        s = np.where(obs >= med, (obs - med) / up, (med - obs) / dn)
        s = np.where(np.isfinite(obs), s, np.nan)
        scores[origin : origin + h] = s
        expected[origin : origin + h] = med
        lower[origin : origin + h] = lo
        upper[origin : origin + h] = hi
        origin += h
    flagged = np.isfinite(scores) & (scores > threshold)
    return AnomalyReport(scores=scores, expected=expected, lower=lower, upper=upper, flagged=flagged)
