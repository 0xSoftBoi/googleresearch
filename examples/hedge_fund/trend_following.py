"""Time-series momentum on real multi-asset data, with costs.

Replicates the construction of Moskowitz, Ooi & Pedersen, "Time Series
Momentum" (JFE 2012) -- the strategy that managed-futures funds such as
Man AHL and AQR run at scale -- on the free FRED universe: sign of the
trailing 12-month return, volatility-targeted per asset, equal-weighted
across whatever assets are alive each day, with proportional transaction
costs swept from 0 to 20 bps.

Run:  python examples/hedge_fund/trend_following.py
"""

from __future__ import annotations

import numpy as np

from timesfm3.data.markets import load_universe
from timesfm3.quant import backtest_portfolio, performance_stats, tsmom_positions

START = "1975-01-01"  # enough assets alive for a diversified panel


def main() -> None:
    panel = load_universe(cache_dir="data/fred").slice(START, None)
    print(f"\nuniverse: {panel.num_assets} assets, {panel.dates[0]} .. {panel.dates[-1]}\n")

    positions = tsmom_positions(panel.returns, lookback=252, vol_target=0.10)

    print("== TSMOM (12m lookback, 10% per-asset vol target), cost sensitivity ==")
    for bps in (0.0, 5.0, 10.0, 20.0):
        res = backtest_portfolio(positions, panel.returns, panel.dates, cost_bps=bps)
        print(res.summary(f"tsmom @ {bps:4.0f} bps"))

    # Long-only, vol-targeted benchmark: same sizing, no sign -- isolates
    # how much the *signal* adds beyond diversified risk premia.
    long_only = np.where(np.isfinite(positions), np.abs(positions), np.nan)
    res_lo = backtest_portfolio(long_only, panel.returns, panel.dates, cost_bps=10.0)
    print(res_lo.summary("long-only @ 10 bps"))

    print("\n== per-asset-class TSMOM sleeves (net of 10 bps) ==")
    classes = sorted(set(panel.asset_classes))
    for cls in classes:
        idx = [i for i, c in enumerate(panel.asset_classes) if c == cls]
        res = backtest_portfolio(
            positions[idx], panel.returns[idx], panel.dates, cost_bps=10.0
        )
        print(res.summary(f"{cls} ({len(idx)} assets)"))

    print("\n== decade breakdown, full portfolio net of 10 bps ==")
    res = backtest_portfolio(positions, panel.returns, panel.dates, cost_bps=10.0)
    years = res.dates.astype("datetime64[Y]").astype(int) + 1970
    for decade in range(1970, 2030, 10):
        mask = (years >= decade) & (years < decade + 10)
        if mask.sum() < 252:
            continue
        s = performance_stats(res.net_returns[mask])
        print(
            f"{decade}s  ann.ret {s['ann_return']:+7.2%}  Sharpe {s['sharpe']:5.2f}  "
            f"maxDD {s['max_drawdown']:7.2%}"
        )


if __name__ == "__main__":
    main()
