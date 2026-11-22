# A 1994 formula still beats time-series foundation models at volatility

*We tried twice to fix that, pre-registered both attempts, and published the
losing rows. Everything below reproduces from this repository.*

## The short version

Volatility is the most forecastable thing in finance, and it is what risk
desks, options desks and vol-targeting funds actually pay for. So we asked a
simple question: out of the box, are time-series foundation models any good
at it?

They are not. On 2020–2026 daily data, scored with QLIKE (the standard loss
for variance forecasts), five days ahead:

| model | QLIKE vs RiskMetrics (1994) | on 11 assets nobody trained on |
|---|---|---|
| RiskMetrics EWMA, λ = 0.94 | 1.00× | 1.00× |
| Chronos-Bolt-small (zero-shot) | 1.31× worse | 1.37× worse |
| Chronos-Bolt-tiny (zero-shot) | 1.32× worse | 1.40× worse |
| Chronos-2 (zero-shot) | 1.43× worse | 3.22× worse |
| our TimesFM-3 starter (zero-shot) | 2.70× worse | 3.53× worse |

That is *after* giving every model, classical and neural, its own
multiplicative calibration fitted on 2015–2019. Without it, the zero-shot
models are far worse still: the fitted scales were 3.7–11.6 for Chronos and
20–50 for our starter, against 1.15 for RiskMetrics.

## Then we tried to beat it

Attempt 1 fine-tuned our 5.2M-parameter TimesFM-3 directly on QLIKE with
causal smoothed-variance inputs. It went from 2.7× worse than RiskMetrics to
a statistical tie (0.988×, Holm p 0.53), and significantly beat HAR (by 7.9%)
and every Chronos model (by 29–41%).

Attempt 2 added 51% more training data, eight more assets and CBOE implied
volatility as an input, then tested on 11 assets the model had never seen:
crypto, emerging-market FX, the Nasdaq-100, the Dow and a new Treasury
maturity. Our model had the lowest five-day loss of all eleven forecasters,
3.4% below RiskMetrics and 1% below HAR. That gap was **not statistically
significant** (raw p 0.073 against RiskMetrics, before multiple-testing
correction).

Both attempts had a release gate written down and committed *before* the test
data was scored: beat every competitor, Holm-corrected p < 0.05. Both failed
it, so neither model shipped. The commit history shows the order.

## What we take from it

1. **Zero-shot foundation models are not volatility models.** If a vendor
   pitches one for risk, ask for the QLIKE table against RiskMetrics.
2. **Fine-tuning closes the gap to the classical frontier, not past it.**
   With only daily closes, a λ = 0.94 EWMA sits very close to the best you
   can do. Beating it plausibly needs better information, such as intraday
   realized variance, rather than a bigger model.
3. **The table is worth more than the model.** Most of what a buyer needs is
   to know, on their own series, whether the expensive model beats the cheap
   one. That is what we sell.

## What we built from this

The forecasting service runs the same walk-forward, HAC Diebold–Mariano,
Holm-corrected backtest on your data that produced these tables:

```bash
pip install "timesfm3[serve] @ git+https://github.com/0xSoftBoi/googleresearch"
timesfm3 backtest your_data.csv --context 256 --horizon 24   # does the model beat a random walk on YOUR data?
timesfm3 serve                                               # REST API + dashboard on :8000
```

`POST /v1/volatility` returns RiskMetrics and HAR variance forecasts with
vol-targeted position sizes: the models that won here, not the ones that
sound good. Chronos models load through the same registry
(`TIMESFM3_CHRONOS_MODELS=...`), so you can rerun this comparison yourself.

## Method, in one paragraph

Data: public FRED daily closes (equity indices, FX, commodities, Treasuries,
crypto). Variance proxy: squared daily log return. Splits by date: train
before 2015, validation 2015–2019, test 2020-01-01 to 2026-11. Context of 512
trading days. Targets: next-day variance and mean variance over the next five
days. Loss: QLIKE (Patton 2011), averaged across assets per date. Test: HAC
Diebold–Mariano per competitor with Holm correction, plus a cluster-bootstrap
CI on the loss ratio. Attempt 1: 16 assets, 28,398 test forecasts. Attempt 2:
11 never-seen assets, 21,410 forecasts. The full protocol, every training run
(including the ones that did worse) and the raw result files are in
[`docs/VERIFICATION.md`](../VERIFICATION.md) §11–16,
[`results/vol_test.json`](../../results/vol_test.json) and
[`results/vol_test_v2.json`](../../results/vol_test_v2.json).

## Caveats

- Daily closes only. Intraday data would change the frontier, and probably
  the ranking.
- Ratios against RiskMetrics in the table are derived from each test's QLIKE
  values; the formal significance tests were run against our fine-tuned
  candidate, as pre-registered.
- One horizon is shown here (five days). Next-day results point the same way
  and are in the ledger.
- Chronos models were run zero-shot on log-variance input. A fine-tuned
  Chronos was not tested.
