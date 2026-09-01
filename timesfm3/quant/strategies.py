"""Reference strategies: what published fund research actually runs.

Three constructions, in increasing order of model dependence:

- :func:`tsmom_positions` -- time-series momentum exactly as in Moskowitz,
  Ooi & Pedersen (JFE 2012): long when the trailing 12-month return is
  positive, short when negative, scaled to a constant ex-ante volatility
  target.  Zero learned parameters; the benchmark every learned signal
  must beat.
- :func:`vol_managed_weights` -- the Moreira-Muir (JF 2017) overlay:
  exposure proportional to the inverse of forecast variance.  Uses no
  return forecast at all, which is precisely the point: the reliable
  edge of a forecaster in markets is in the *scale* of returns, not the
  sign.
- :func:`forecast_signal_positions` -- the Deep-Momentum-Networks pattern
  (Lim, Zohren & Roberts 2019): a forecaster's expected move over the
  next horizon, standardized by forecast volatility, squashed into a
  bounded position, and wrapped in the same volatility targeting.  Any
  object with ``forecast(context, horizon) -> (horizon,)`` plugs in --
  the classical baselines and a TimesFM-3 forecaster (via
  :class:`TimesFM3Signal`) run through identical code, so a model
  comparison is a strategy comparison.

All position functions obey the backtester's timing convention: the
position in column ``t`` uses information through day ``t`` only.
"""

from __future__ import annotations

import numpy as np

from ..baselines import Baseline
from .volatility import TRADING_DAYS, ewma_variance


def _ewma_vol(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """Annualized one-step-ahead vol forecast per day (info through t-1)."""
    var = ewma_variance(returns, lam=lam)
    return np.sqrt(np.maximum(var, 0.0) * TRADING_DAYS)


def tsmom_positions(
    returns: np.ndarray,
    lookback: int = 252,
    vol_target: float = 0.10,
    max_leverage: float = 4.0,
) -> np.ndarray:
    """MOP 2012 time-series momentum for a (N, T) return panel.

    position[i, t] = sign(sum of returns over t-lookback+1 .. t)
                     * vol_target / sigma_hat[i, t+1]

    where sigma_hat is the RiskMetrics ex-ante vol.  NaN until an asset has
    a full lookback of history.  Leverage is capped: quiet series would
    otherwise be levered absurdly, which no risk desk allows.
    """
    n, t_len = returns.shape
    pos = np.full((n, t_len), np.nan)
    for i in range(n):
        r = returns[i]
        finite = np.isfinite(r)
        if finite.sum() < lookback + 30:
            continue
        csum = np.nancumsum(np.where(finite, r, 0.0))
        vol = _ewma_vol(np.where(finite, r, np.nan))
        for t in range(lookback, t_len - 1):
            if not finite[max(0, t - lookback) : t + 1].any() or not finite[t]:
                continue
            trailing = csum[t] - csum[t - lookback]
            v = vol[t + 1] if t + 1 < t_len else vol[t]
            if not np.isfinite(v) or v <= 0:
                continue
            lev = min(vol_target / v, max_leverage)
            pos[i, t] = np.sign(trailing) * lev
    return pos


def vol_managed_weights(
    returns: np.ndarray,
    vol_target: float = 0.10,
    max_leverage: float = 3.0,
    lam: float = 0.94,
    variance_forecast: np.ndarray | None = None,
) -> np.ndarray:
    """Moreira-Muir scaling for one asset: weight[t] = c / sigma2_hat[t+1].

    ``variance_forecast`` may supply a better model's one-step-ahead
    variance path (e.g. HAR's, aligned so entry t is the forecast for day
    t+1 made with info through t); defaults to RiskMetrics.  The constant
    ``c`` is set so the *forecast* vol of the scaled position equals
    ``vol_target``; weights are capped at ``max_leverage``.
    """
    r = np.asarray(returns, dtype=np.float64)
    if variance_forecast is None:
        var = ewma_variance(r, lam=lam)
        var = np.concatenate([var[1:], [var[-1]]])  # var[t] forecasts day t+1
    else:
        var = np.asarray(variance_forecast, dtype=np.float64)
    ann_var = np.maximum(var * TRADING_DAYS, 1e-10)
    w = vol_target / np.sqrt(ann_var)
    return np.clip(np.where(np.isfinite(r), w, np.nan), 0.0, max_leverage)


def forecast_signal_positions(
    returns: np.ndarray,
    forecaster: Baseline,
    context_len: int = 252,
    horizon: int = 21,
    rebalance: int = 5,
    vol_target: float = 0.10,
    max_leverage: float = 4.0,
    signal_scale: float = 1.0,
) -> np.ndarray:
    """Deep-Momentum-Networks-style sizing from any forecaster.

    Every ``rebalance`` days, per asset: feed the last ``context_len``
    cumulative log returns to the forecaster, read the predicted cumulative
    move over ``horizon`` days, standardize it into

        z = predicted_move / (sigma_hat * sqrt(horizon))

    and hold ``tanh(z / signal_scale) * vol_target / sigma_hat_ann`` until
    the next rebalance.  tanh keeps positions bounded and makes the map
    from forecast to exposure smooth -- a large predicted move saturates
    instead of demanding unbounded leverage, and a near-zero forecast holds
    a near-zero position (unlike sign(), which trades noise at full size).
    """
    n, t_len = returns.shape
    pos = np.full((n, t_len), np.nan)
    for i in range(n):
        r = returns[i]
        finite = np.isfinite(r)
        if finite.sum() < context_len + 30:
            continue
        level = np.nancumsum(np.where(finite, r, 0.0))
        vol_ann = _ewma_vol(np.where(finite, r, np.nan))
        current = 0.0
        started = False
        first = np.flatnonzero(finite)[0]
        for t in range(first + context_len, t_len - 1):
            if not finite[t]:
                continue
            if (t - first - context_len) % rebalance == 0:
                context = level[t - context_len + 1 : t + 1]
                pred = forecaster.forecast(context, horizon)
                move = float(pred[-1] - level[t])
                v_ann = vol_ann[t + 1] if t + 1 < t_len else vol_ann[t]
                if np.isfinite(v_ann) and v_ann > 0:
                    daily_vol = v_ann / np.sqrt(TRADING_DAYS)
                    z = move / (daily_vol * np.sqrt(horizon))
                    lev = min(vol_target / v_ann, max_leverage)
                    current = float(np.tanh(z / signal_scale)) * lev
                    started = True
            if started:
                pos[i, t] = current
    return pos


class TimesFM3Signal(Baseline):
    """Adapter: a TimesFM-3 forecaster as a ``Baseline``.

    Wraps :class:`timesfm3.forecaster.TimesFM3Forecaster` (or anything with
    its ``forecast(targets=..., horizon=...)`` API) so it drops into
    :func:`forecast_signal_positions` and the volatility harness unchanged.
    ``quantile`` selects which forecast to read: the default 4 is the
    median; a risk-averse signal can size off q25 for longs and q75 for
    shorts by running the adapter twice.

    Torch is imported by the wrapped forecaster, not here, so the rest of
    the quant stack stays importable without torch.
    """

    def __init__(self, forecaster, quantile: int | None = None, name: str = "timesfm3"):
        self.forecaster = forecaster
        self.quantile = quantile
        self.name = name

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        result = self.forecaster.forecast(
            targets=[np.asarray(context, dtype=np.float32)], horizon=horizon
        )
        if self.quantile is None:
            return np.asarray(result.point[0], dtype=np.float64)
        return np.asarray(result.quantiles[0, :, self.quantile], dtype=np.float64)
