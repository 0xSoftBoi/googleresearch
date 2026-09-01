"""Quant-finance applications of the forecasting stack.

The modules here turn the repo's forecasters into the two workhorses of
systematic hedge funds:

- :mod:`volatility` -- variance forecasting (RiskMetrics EWMA, Corsi's
  HAR-RV, and any :class:`~timesfm3.baselines.Baseline` or TimesFM-3
  forecaster through the same interface), scored with QLIKE and compared
  with the Diebold-Mariano machinery in :mod:`timesfm3.evaluation`.
- :mod:`backtest` -- a daily portfolio backtester with volatility
  targeting, transaction costs, and Newey-West-adjusted performance
  statistics.
- :mod:`strategies` -- reference implementations of the strategies the
  published literature attributes to real funds: time-series momentum
  (Moskowitz-Ooi-Pedersen 2012), volatility-managed overlays
  (Moreira-Muir 2017), and a generic forecast-driven signal that accepts
  any forecaster with the ``forecast(context, horizon)`` shape.
"""

from .backtest import BacktestResult, backtest_portfolio, performance_stats
from .strategies import (
    forecast_signal_positions,
    tsmom_positions,
    vol_managed_weights,
)
from .volatility import (
    HAR,
    ewma_variance,
    qlike,
    realized_variance,
    rolling_variance_forecasts,
)

__all__ = [
    "BacktestResult",
    "HAR",
    "backtest_portfolio",
    "ewma_variance",
    "forecast_signal_positions",
    "performance_stats",
    "qlike",
    "realized_variance",
    "rolling_variance_forecasts",
    "tsmom_positions",
    "vol_managed_weights",
]
