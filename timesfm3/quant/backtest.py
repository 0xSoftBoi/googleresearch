"""A daily portfolio backtester that takes costs and inference seriously.

Backtests mislead in two standard ways: they trade on information not yet
available, and they ignore what trading costs.  This engine fixes the
timing convention once -- ``positions[:, t]`` is decided with information
through day t and earns ``returns[:, t+1]`` -- charges proportional costs
on every position change, and reports a Sharpe ratio with a Newey-West
standard error (daily strategy returns are autocorrelated, so the naive
``sqrt(T)`` t-stat overstates significance).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ..evaluation import newey_west_variance

TRADING_DAYS = 252


@dataclasses.dataclass
class BacktestResult:
    """Portfolio-level daily results plus summary statistics."""

    dates: np.ndarray  # (T,) datetime64[D] of the earning days
    gross_returns: np.ndarray  # (T,) portfolio return before costs
    net_returns: np.ndarray  # (T,) after transaction costs
    costs: np.ndarray  # (T,) cost drag per day
    turnover: float  # average daily one-sided turnover (sum |dpos|)
    stats: dict[str, float]

    def summary(self, name: str = "strategy") -> str:
        s = self.stats
        return (
            f"{name:24s} ann.ret {s['ann_return']:+7.2%}  ann.vol {s['ann_vol']:6.2%}  "
            f"Sharpe {s['sharpe']:5.2f} (t={s['sharpe_tstat']:4.1f})  "
            f"maxDD {s['max_drawdown']:7.2%}  turnover {self.turnover:5.2f}/day"
        )


def performance_stats(daily_returns: np.ndarray) -> dict[str, float]:
    """Annualized stats with an HAC-adjusted Sharpe t-statistic.

    The t-stat uses Newey-West variance of the daily mean, so overlapping
    signals and volatility clustering do not inflate apparent significance.
    """
    r = np.asarray(daily_returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return {k: np.nan for k in (
            "ann_return", "ann_vol", "sharpe", "sharpe_tstat",
            "max_drawdown", "hit_rate", "skew", "n_days",
        )}
    mean, std = r.mean(), r.std(ddof=1)
    ann_ret = mean * TRADING_DAYS
    ann_vol = std * np.sqrt(TRADING_DAYS)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    nw_var = newey_west_variance(r)  # variance of the *mean* under HAC
    tstat = mean / np.sqrt(nw_var) if nw_var > 0 else np.nan
    equity = np.cumsum(r)  # log-return convention: drawdown in log space
    drawdown = equity - np.maximum.accumulate(equity)
    active = r[r != 0.0]
    return {
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sharpe_tstat": float(tstat),
        "max_drawdown": float(drawdown.min()),
        "hit_rate": float((active > 0).mean()) if len(active) else np.nan,
        "skew": float(((r - mean) ** 3).mean() / std**3) if std > 0 else np.nan,
        "n_days": float(len(r)),
    }


def backtest_portfolio(
    positions: np.ndarray,
    returns: np.ndarray,
    dates: np.ndarray,
    cost_bps: float = 10.0,
    equal_risk: bool = True,
) -> BacktestResult:
    """Run the panel backtest.

    Arguments:
        positions: (N, T) signed exposures in units of notional (1.0 = one
            unit of capital long).  ``positions[:, t]`` must be computable
            from information through day t; NaN means "asset not tradable".
        returns: (N, T) daily log returns, NaN before an asset exists.
        dates: (T,) panel dates.
        cost_bps: proportional cost per unit of traded notional, one-sided,
            in basis points.  5-20 bps spans liquid futures/FX after
            slippage (Frazzini-Israel-Moskowitz measure much lower for
            institutional flow, so 10 bps is conservative).
        equal_risk: divide by the number of live assets each day, so the
            portfolio is an equal-weight average of per-asset strategies
            (the MOP construction) rather than a sum that grows with N.

    Portfolio return on day t+1 is ``sum_i w_i(t) * r_i(t+1) - costs`` with
    costs charged on ``|w(t) - w(t-1)|``.
    """
    pos = np.where(np.isfinite(positions), positions, 0.0)
    ret = np.where(np.isfinite(returns), returns, 0.0)
    live = np.isfinite(positions) & np.isfinite(returns)
    n_live = np.maximum(live.sum(axis=0), 1)
    if equal_risk:
        pos = pos / n_live[None, :]

    # positions decided at t earn returns at t+1
    gross = np.sum(pos[:, :-1] * ret[:, 1:], axis=0)
    dpos = np.abs(np.diff(pos, axis=1, prepend=np.zeros((pos.shape[0], 1))))
    costs = (cost_bps * 1e-4) * dpos.sum(axis=0)[:-1]
    net = gross - costs
    turnover = float(dpos.sum(axis=0).mean())

    return BacktestResult(
        dates=dates[1:],
        gross_returns=gross,
        net_returns=net,
        costs=costs,
        turnover=turnover,
        stats=performance_stats(net),
    )
