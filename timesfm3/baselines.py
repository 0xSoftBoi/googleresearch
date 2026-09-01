"""Classical forecasting baselines.

A foundation model is only interesting if it beats what a few lines of
statistics already achieve, so benchmarks in this repo score against these
rather than against last-value alone. Every baseline has the same shape --
``forecast(context, horizon) -> (horizon,)`` -- and fits whatever parameters it
needs from the context window itself, so nothing leaks across the forecast
boundary.

The set spans the two regimes a series can be in. ``LastValue`` and ``Drift``
are right for a random walk; ``ContextMean``, ``EWMA`` and ``AR`` are right for
a mean-reverting one. Which of them wins on a given channel is itself a useful
statement about that channel.
"""

from __future__ import annotations

import dataclasses

import numpy as np


class Baseline:
    """A forecaster fit from, and predicting beyond, a single context window."""

    name = "baseline"

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{self.name}>"


class LastValue(Baseline):
    """Random walk: the last observation, held flat. Optimal for a martingale."""

    name = "last-value"

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        return np.full(horizon, context[-1], dtype=np.float64)


class ContextMean(Baseline):
    """The context mean, held flat. Optimal for white noise around a level."""

    name = "ctx-mean"

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        return np.full(horizon, context.mean(), dtype=np.float64)


class Drift(Baseline):
    """Random walk with drift: extrapolates the context's average slope."""

    name = "drift"

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        n = len(context)
        if n < 2:
            return np.full(horizon, context[-1], dtype=np.float64)
        slope = (context[-1] - context[0]) / (n - 1)
        return context[-1] + slope * np.arange(1, horizon + 1, dtype=np.float64)


@dataclasses.dataclass
class EWMA(Baseline):
    """Exponentially weighted moving average, held flat over the horizon.

    The smoothing constant is chosen by minimising one-step-ahead squared error
    *within the context*, over a grid. That keeps the baseline honest (nothing
    beyond the forecast origin is used) while letting it adapt: a near-random
    walk drives alpha toward 1 and the baseline degenerates to last-value, a
    mean-reverting series drives alpha down toward the context mean.
    """

    alphas: tuple[float, ...] = (
        0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0
    )
    name = "ewma"

    def _level(self, context: np.ndarray, alpha: float) -> np.ndarray:
        """Recursive EWMA level; returns the level *after* each observation."""
        out = np.empty(len(context), dtype=np.float64)
        level = float(context[0])
        for i, x in enumerate(context):
            level = alpha * float(x) + (1.0 - alpha) * level
            out[i] = level
        return out

    def fit_alpha(self, context: np.ndarray) -> float:
        best, best_sse = self.alphas[-1], np.inf
        for alpha in self.alphas:
            level = self._level(context, alpha)
            # level[i] predicts context[i+1]
            err = context[1:] - level[:-1]
            sse = float(np.dot(err, err))
            if sse < best_sse:
                best, best_sse = alpha, sse
        return best

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        context = np.asarray(context, dtype=np.float64)
        if len(context) < 3:
            return np.full(horizon, context[-1], dtype=np.float64)
        alpha = self.fit_alpha(context)
        level = self._level(context, alpha)[-1]
        return np.full(horizon, level, dtype=np.float64)


@dataclasses.dataclass
class AR(Baseline):
    """Autoregressive model of order ``p``, fit by OLS on the context.

    The fitted recursion is iterated forward for the horizon, so the forecast
    decays toward the fitted mean at the rate the data implies -- the natural
    competitor for any channel with real autocorrelation. A singular or
    degenerate design (a constant context, or too few rows) falls back to
    last-value rather than producing a spurious fit.
    """

    p: int = 1
    ridge: float = 1e-8

    def __post_init__(self) -> None:
        self.name = f"ar{self.p}"

    def fit(self, context: np.ndarray) -> tuple[np.ndarray, float] | None:
        """Returns ``(coefficients, intercept)`` or ``None`` if unfittable."""
        p, n = self.p, len(context)
        if n < 3 * p + 2:
            return None
        # Centre first so the intercept is not fighting the scale.
        mu = float(context.mean())
        x = context - mu
        rows = n - p
        design = np.empty((rows, p), dtype=np.float64)
        for lag in range(p):
            design[:, lag] = x[p - lag - 1: n - lag - 1]
        target = x[p:]
        gram = design.T @ design + self.ridge * np.eye(p)
        try:
            coef = np.linalg.solve(gram, design.T @ target)
        except np.linalg.LinAlgError:  # pragma: no cover - ridge makes this rare
            return None
        if not np.all(np.isfinite(coef)):
            return None
        return coef, mu

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        context = np.asarray(context, dtype=np.float64)
        fitted = self.fit(context)
        if fitted is None or context.std() == 0:
            return np.full(horizon, context[-1], dtype=np.float64)
        coef, mu = fitted
        history = list(context[-self.p:] - mu)
        out = np.empty(horizon, dtype=np.float64)
        for i in range(horizon):
            nxt = float(np.dot(coef, history[::-1][: self.p]))
            out[i] = nxt
            history.append(nxt)
            history = history[-self.p:]
        forecast = out + mu
        # An explosive fit (|root| > 1) can diverge over a long horizon; clamp
        # to the range the context actually visited rather than emit garbage.
        lo, hi = context.min(), context.max()
        span = hi - lo
        return np.clip(forecast, lo - span, hi + span)


#: A reasonable default panel: one random-walk baseline, one flat-mean
#: baseline, and three that can express mean reversion at different speeds.
DEFAULT_BASELINES: tuple[Baseline, ...] = (
    LastValue(),
    ContextMean(),
    Drift(),
    EWMA(),
    AR(1),
    AR(4),
)


def baseline_forecasts(
    context: np.ndarray, horizon: int, baselines=DEFAULT_BASELINES
) -> dict[str, np.ndarray]:
    """Run every baseline on one context window."""
    context = np.asarray(context, dtype=np.float64)
    return {b.name: b.forecast(context, horizon) for b in baselines}
