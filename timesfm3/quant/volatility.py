"""Volatility forecasting: the most forecastable object in finance.

Daily *returns* are close to unpredictable (a good daily direction model
explains well under 1% of variance), but *volatility* clusters strongly and
is highly forecastable -- which is why volatility models, not return models,
are the first thing risk desks and options desks run in production.  This
module provides:

- :func:`realized_variance` -- squared demeaned daily returns, the standard
  daily proxy for latent variance when intraday data is unavailable.
- :func:`ewma_variance` -- the RiskMetrics filter (lambda = 0.94), the
  industry default since J.P. Morgan published it in 1994.
- :class:`HAR` -- the Heterogeneous Autoregressive model of Corsi (2009),
  regressing tomorrow's realized variance on daily / weekly / monthly
  averages.  Three OLS coefficients reproduce most of the long-memory
  behaviour of far fancier models, which makes HAR the benchmark that any
  machine-learning volatility forecaster must beat to claim an edge.
- :func:`qlike` -- the loss used to score variance forecasts.  QLIKE is
  robust to noisy volatility proxies (Patton 2011): squared-return proxies
  are unbiased but very noisy, and QLIKE keeps the ranking of forecasters
  consistent under that noise where naive MSE on variance need not.
- :func:`rolling_variance_forecasts` -- a walk-forward harness producing
  per-origin losses for each forecaster, shaped for
  :func:`timesfm3.evaluation.compare` (Diebold-Mariano + Holm).

``HAR`` implements the repo's :class:`~timesfm3.baselines.Baseline`
interface (``forecast(context, horizon)`` on a variance series), so the
classical baselines, HAR, and a TimesFM-3 forecaster all run through one
pipeline and one significance test.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ..baselines import Baseline

TRADING_DAYS = 252


def realized_variance(returns: np.ndarray, demean: bool = True) -> np.ndarray:
    """Daily variance proxy: squared (optionally demeaned) returns.

    With only daily data, r_t^2 is the standard unbiased-but-noisy proxy for
    the day's integrated variance; a tiny floor keeps log-space models and
    QLIKE defined on zero-return days.
    """
    r = np.asarray(returns, dtype=np.float64)
    mu = np.nanmean(r) if demean else 0.0
    rv = (r - mu) ** 2
    floor = max(np.nanmedian(rv) * 1e-4, 1e-12)
    return np.maximum(rv, floor)


def ewma_variance(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """RiskMetrics filter: sigma2_t = lam * sigma2_{t-1} + (1-lam) * r_{t-1}^2.

    ``out[t]`` uses information through ``t-1`` only, so it is a genuine
    one-step-ahead forecast series.  Seeded with the expanding mean of the
    first 20 squared returns.
    """
    rv = realized_variance(returns)
    out = np.full_like(rv, np.nan)
    finite = np.flatnonzero(np.isfinite(rv))
    if len(finite) < 2:
        return out
    # Seed with the mean of the first few finite squared returns (an asset
    # may be born mid-panel, so "first 20 rows" would be all-NaN).  The
    # seed peeks at most 20 days past the series start; callers require far
    # longer warmups than that before trading.
    var = float(np.mean(rv[finite[: min(20, len(finite))]]))
    start = int(finite[0])
    out[start] = var
    for t in range(start + 1, len(rv)):
        out[t] = var
        if np.isfinite(rv[t]):
            var = lam * var + (1.0 - lam) * rv[t]
    return out


@dataclasses.dataclass
class HAR(Baseline):
    """Corsi (2009) HAR-RV, fit by OLS on the context window.

    Regresses RV_{t+1} on the daily value, the 5-day mean, and the 22-day
    mean of RV -- a parsimonious "heterogeneous market" cascade that mimics
    long memory.  Corsi's original fits in *levels*, and that is the
    default here for a reason that bites with daily data: the squared
    return is an unbiased but extremely noisy variance proxy, so a levels
    fit stays unbiased, whereas a log fit needs the log-normal half-variance
    correction on the way back -- with log(r^2) residual variances of ~5,
    plain ``exp`` under-forecasts variance several-fold and QLIKE destroys
    it.  ``log_space=True`` applies that correction explicitly.

    Multi-step forecasts iterate the fitted recursion, feeding each
    prediction back into the daily/weekly/monthly averages.
    """

    daily: int = 1
    weekly: int = 5
    monthly: int = 22
    ridge: float = 1e-8
    log_space: bool = False

    def __post_init__(self) -> None:
        self.name = "har-log" if self.log_space else "har"

    def _design(self, series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = self.monthly
        n = len(series)
        rows = n - m
        x = np.empty((rows, 3), dtype=np.float64)
        for i in range(rows):
            t = m + i  # predict series[t] from info through t-1
            x[i, 0] = series[t - 1]
            x[i, 1] = series[t - self.weekly : t].mean()
            x[i, 2] = series[t - m : t].mean()
        y = series[m:]
        return x, y

    def fit(self, context: np.ndarray) -> tuple[np.ndarray, float] | None:
        """Returns ``([intercept, b_d, b_w, b_m], residual_variance)``."""
        rv = np.asarray(context, dtype=np.float64)
        if len(rv) < self.monthly + 30 or np.any(rv <= 0) or not np.all(np.isfinite(rv)):
            return None
        if self.log_space:
            series = np.log(rv)
        else:
            # Winsorize before a levels fit: one Black-Monday-sized squared
            # return in the window otherwise dominates the Gram matrix and
            # can flip coefficients negative for the following year.
            series = np.clip(rv, None, np.quantile(rv, 0.99))
        x, y = self._design(series)
        design = np.column_stack([np.ones(len(x)), x])
        gram = design.T @ design + self.ridge * np.eye(4)
        try:
            beta = np.linalg.solve(gram, design.T @ y)
        except np.linalg.LinAlgError:  # pragma: no cover - ridge makes this rare
            return None
        if not np.all(np.isfinite(beta)):
            return None
        resid = y - design @ beta
        return beta, float(resid.var())

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        rv = np.asarray(context, dtype=np.float64)
        fitted = self.fit(rv)
        if fitted is None:
            return np.full(horizon, rv[-1] if len(rv) else np.nan)
        beta, resid_var = fitted
        series = np.log(rv) if self.log_space else rv
        history = list(series[-self.monthly :])
        out = np.empty(horizon, dtype=np.float64)
        for i in range(horizon):
            arr = np.asarray(history[-self.monthly :])
            feats = np.array([1.0, arr[-1], arr[-self.weekly :].mean(), arr.mean()])
            nxt = float(beta @ feats)
            out[i] = nxt
            history.append(nxt)
        if self.log_space:
            # E[RV] = exp(mu + s^2/2) when log RV is normal with residual
            # variance s^2 -- the correction that makes a log fit unbiased.
            out = np.exp(
                np.clip(out + 0.5 * resid_var, np.log(rv.min()) - 5.0, np.log(rv.max()) + 5.0)
            )
        # A levels fit can still emit a near-zero or negative variance;
        # volatility does not collapse an order of magnitude in a week, so
        # floor at a fraction of the trailing month (QLIKE is unforgiving
        # of a forecast orders of magnitude below realized).
        lo = max(float(np.quantile(rv, 0.05)), 0.1 * float(rv[-self.monthly :].mean()))
        hi = float(rv.max() * 10.0)
        return np.clip(out, lo, hi)


def qlike(pred_var: np.ndarray, true_var: np.ndarray) -> np.ndarray:
    """QLIKE loss per observation: rv/pred - log(rv/pred) - 1 (>= 0).

    Minimized when pred equals the true conditional variance; robust to the
    noise of squared-return proxies (Patton 2011, "Volatility forecast
    comparison using imperfect volatility proxies").
    """
    pred = np.maximum(np.asarray(pred_var, dtype=np.float64), 1e-14)
    rv = np.maximum(np.asarray(true_var, dtype=np.float64), 1e-14)
    ratio = rv / pred
    return ratio - np.log(ratio) - 1.0


def rolling_variance_forecasts(
    returns: np.ndarray,
    forecasters: dict[str, Baseline],
    context_len: int = 504,
    horizon: int = 1,
    stride: int = 5,
    include_ewma: bool = True,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Walk-forward variance forecasting on one return series.

    At each origin t (every ``stride`` days), each forecaster sees the RV
    series up to and including t and predicts the mean variance over
    ``t+1 .. t+horizon``.  Returns ``(losses, origins)`` where
    ``losses[name]`` holds the QLIKE of each origin -- exactly the paired
    per-window losses that :func:`timesfm3.evaluation.compare` consumes.
    RiskMetrics EWMA is added automatically as the industry-default
    reference (its filtered forecast at each origin).
    """
    rv = realized_variance(returns)
    finite = np.flatnonzero(np.isfinite(rv))
    rv = rv[finite]
    ewma = ewma_variance(np.asarray(returns, dtype=np.float64)[finite])

    origins = np.arange(context_len, len(rv) - horizon, stride)
    losses: dict[str, list[float]] = {name: [] for name in forecasters}
    if include_ewma:
        losses["riskmetrics"] = []
    for t in origins:
        realized = rv[t + 1 : t + 1 + horizon].mean()
        context = rv[t - context_len + 1 : t + 1]
        # Variance is positive by definition; impose that domain floor on
        # every contender identically, so a generic level forecaster that
        # emits a negative value is scored on the constraint, not on -inf.
        floor = 0.1 * float(np.quantile(context, 0.05))
        for name, fc in forecasters.items():
            pred = float(np.mean(fc.forecast(context, horizon)))
            losses[name].append(float(qlike(max(pred, floor), realized)))
        if include_ewma:
            # ewma[t+1] is the filter's forecast made with info through t.
            losses["riskmetrics"].append(float(qlike(ewma[t + 1], realized)))
    return {k: np.asarray(v) for k, v in losses.items()}, origins
