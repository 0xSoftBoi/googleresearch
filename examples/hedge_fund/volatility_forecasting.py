"""Volatility forecasting horse race + the Moreira-Muir overlay, on real data.

Part 1 -- the forecasting problem every risk desk actually runs: predict
next-week variance.  Contestants: RiskMetrics EWMA (the industry default),
Corsi's HAR-RV (the academic benchmark ML must beat), and the repo's
classical baselines applied to the realized-variance series.  Scored by
QLIKE (Patton 2011) with Diebold-Mariano tests and Holm correction from
``timesfm3.evaluation`` -- the same machinery the repo uses for the
foundation model, because a variance forecaster is just a forecaster.

Part 2 -- why it matters for P&L: Moreira & Muir (JF 2017) showed that
scaling exposure by inverse forecast variance raises Sharpe ratios because
volatility is forecastable but expected returns do not move with it
one-for-one.  We apply the overlay to the NASDAQ series and report the
before/after.

Run:  python examples/hedge_fund/volatility_forecasting.py
"""

from __future__ import annotations

import numpy as np

from timesfm3.baselines import AR, EWMA, LastValue
from timesfm3.data.markets import load_universe
from timesfm3.evaluation import compare
from timesfm3.quant import HAR, rolling_variance_forecasts, vol_managed_weights
from timesfm3.quant.backtest import performance_stats

ASSETS = ("NASDAQ", "EURUSD", "WTI", "UST10Y")
HORIZON = 5  # one trading week
CONTEXT = 756  # three years


def horse_race() -> None:
    panel = load_universe(cache_dir="data/fred", verbose=False)
    forecasters = {
        "har": HAR(),
        "ar5-rv": AR(5),
        "ewma-rv": EWMA(),
        "last-rv": LastValue(),
    }
    print(f"== {HORIZON}-day-ahead variance, QLIKE, vs RiskMetrics (lambda=0.94) ==")
    print("ratio < 1 beats RiskMetrics; p-values are DM with HAC, Holm-corrected.\n")
    for name in ASSETS:
        i = panel.names.index(name)
        r = panel.returns[i]
        r = r[np.isfinite(r)]
        losses, origins = rolling_variance_forecasts(
            r, forecasters, context_len=CONTEXT, horizon=HORIZON, stride=HORIZON
        )
        results = compare(losses, reference="riskmetrics")
        print(f"-- {name} ({len(origins)} forecast origins) --")
        for res in results.values():
            print(
                f"   {res.name:10s} QLIKE ratio {res.ratio:5.3f}  "
                f"p={res.p_adjusted if res.p_adjusted is not None else res.p_value:7.1e}  "
                f"[{res.verdict}]"
            )
        print()


def overlay() -> None:
    panel = load_universe(cache_dir="data/fred", verbose=False)
    i = panel.names.index("NASDAQ")
    r = panel.returns[i]
    finite = np.isfinite(r)
    r = r[finite]

    w = vol_managed_weights(r, vol_target=0.15)
    managed = w[:-1] * r[1:]
    raw = r[1:]
    s_raw, s_mgd = performance_stats(raw), performance_stats(managed)
    print("== Moreira-Muir volatility-managed overlay, NASDAQ ==")
    for label, s in (("buy & hold", s_raw), ("vol-managed (15% target)", s_mgd)):
        print(
            f"   {label:26s} ann.ret {s['ann_return']:+7.2%}  ann.vol {s['ann_vol']:6.2%}  "
            f"Sharpe {s['sharpe']:5.2f}  maxDD {s['max_drawdown']:7.2%}  "
            f"skew {s['skew']:+5.2f}"
        )


if __name__ == "__main__":
    horse_race()
    overlay()
