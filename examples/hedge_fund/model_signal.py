"""Forecaster-driven positions: the Deep Momentum Networks pattern.

Lim, Zohren & Roberts ("Enhancing Time Series Momentum Strategies Using
Deep Neural Networks", JFDS 2019, from the Man-AHL-funded Oxford-Man
Institute) size positions from a neural forecaster's standardized expected
move inside the classic volatility-targeting wrapper.  This script runs
that exact harness (``forecast_signal_positions``) on the real FRED
universe with interchangeable forecasters:

- ``drift``  -- linear trend extrapolation: a continuous trend signal, the
  closest classical analogue of a learned momentum forecaster;
- ``ar4``    -- an autoregression: a mean-reversion-capable signal;
- ``ewma``   -- a level forecaster whose predicted move is ~0: positions
  stay near zero, a placebo that checks the harness does not conjure P&L
  out of sizing alone;
- ``timesfm3`` -- optionally, a trained TimesFM-3 checkpoint via
  ``TimesFM3Signal`` (pass its path as argv[1]).  Per Rahimikia et al.
  ("Re(Visiting) Time Series Foundation Models in Finance", 2025),
  randomly initialized or generic-pretrained weights should NOT be
  expected to beat the classical baselines on returns -- the published
  gains come from pre-training on financial data, which is what
  ``timesfm3.train`` + ``timesfm3.data.markets.to_real_source`` set up.

Because every forecaster runs through identical sizing, costs, and
statistics, differences in the table below are attributable to forecast
quality alone.

Run:  python examples/hedge_fund/model_signal.py [checkpoint.pt]
"""

from __future__ import annotations

import sys

from timesfm3.baselines import AR, Drift, EWMA
from timesfm3.data.markets import load_universe
from timesfm3.quant import backtest_portfolio, forecast_signal_positions

START = "1975-01-01"
COST_BPS = 10.0


def main() -> None:
    panel = load_universe(cache_dir="data/fred", verbose=False).slice(START, None)
    print(f"universe: {panel.num_assets} assets, {panel.dates[0]} .. {panel.dates[-1]}")
    print(f"harness: 252d context -> 21d horizon forecast, weekly rebalance, "
          f"10% vol target, {COST_BPS:.0f} bps costs\n")

    forecasters = {"drift": Drift(), "ar4": AR(4), "ewma": EWMA()}
    if len(sys.argv) > 1:
        from timesfm3.forecaster import TimesFM3Forecaster
        from timesfm3.quant.strategies import TimesFM3Signal

        forecasters["timesfm3"] = TimesFM3Signal(
            TimesFM3Forecaster.from_checkpoint(sys.argv[1])
        )

    for name, fc in forecasters.items():
        pos = forecast_signal_positions(
            panel.returns, fc, context_len=252, horizon=21, rebalance=5,
            vol_target=0.10,
        )
        res = backtest_portfolio(pos, panel.returns, panel.dates, cost_bps=COST_BPS)
        print(res.summary(f"signal[{name}]"))


if __name__ == "__main__":
    main()
