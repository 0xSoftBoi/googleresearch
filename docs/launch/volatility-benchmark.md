# A 1994 formula still beats time-series foundation models at volatility

*We tried twice to fix that, pre-registered both attempts, and published the
losing rows. An automated code review then found a look-ahead bug in our
harness. We fixed it, re-ran everything, and report the corrected numbers
below. Everything reproduces from this repository.*

## The short version

Volatility is the most forecastable thing in finance, and it is what risk
desks, options desks and vol-targeting funds actually pay for. So we asked a
simple question: used off the shelf, with no fine-tuning, are time-series
foundation models any good at it?

They are not. On 2020–2026 daily data, scored with QLIKE (the standard loss
for variance forecasts), five days ahead, as a ratio to RiskMetrics' loss:

| model | 16 assets, 2020–2026 | 11 assets nobody trained on |
|---|---|---|
| RiskMetrics EWMA, λ = 0.94 (1994) | 1.00× | 1.00× |
| HAR (Corsi 2009) | 1.09× | 0.97× |
| our TimesFM-3, fine-tuned on QLIKE | 1.03× | 0.97× |
| Chronos-Bolt-tiny, off the shelf | 1.32× | 1.41× |
| Chronos-Bolt-small, off the shelf | 1.33× | 1.39× |
| Chronos-2, off the shelf | 1.48× | 8.4× |
| our TimesFM-3 starter, off the shelf | 3.65× | 4.16× |

Lower is better. That is *after* giving every model, classical and neural,
its own multiplicative calibration fitted on 2015–2019. The fitted scales
show how far off the raw forecasts are: 3.8–12 for Chronos and 41–162 for our
starter, against 1.13–1.15 for RiskMetrics.

## Then we tried to beat it

Attempt 1 fine-tuned our 5.2M-parameter TimesFM-3 directly on QLIKE with
causal smoothed-variance inputs. It went from 3.6× RiskMetrics' loss to a
statistical tie with both RiskMetrics (2.8% higher loss, Holm p 0.12) and
HAR (5.8% lower loss, Holm p 0.12). It beat every Chronos model by 22–31%
in loss, significantly.

Attempt 2 added 51% more training data, eight more assets and CBOE implied
volatility as an input, then tested on 11 assets the model had never seen:
crypto, emerging-market FX, the Nasdaq-100, the Dow and a new Treasury
maturity. It landed 2.6% below RiskMetrics and level with HAR (0.4863 vs
0.4859), with no significant difference from either. Implied volatility did
not help on those assets: HAR with implied vol was 14% worse than plain HAR.

Both attempts had a release gate written down and committed *before* the
test data was scored: beat every competitor, Holm-corrected p < 0.05. Both
failed it, so neither model shipped. The commit history shows the order.

## The bug, and what it changed

After we published, automated reviewers (Devin and cubic) found that our
preprocessing used each asset's full history in two places: the floor for
zero-return days, and the mean RiskMetrics subtracted before squaring
returns. Later returns could therefore nudge earlier values. We fixed both,
added a test that rewrites every return after a given day and checks that
nothing earlier moves, and re-ran both tests on the same data with the same
locked models.

What changed: our earlier claim that the fine-tuned model *significantly*
beat HAR no longer holds, and HAR, not our model, now has the lowest loss on
the unseen assets. What held: no off-the-shelf foundation model came within
30% of RiskMetrics on any five-day view, before or after the fix. The full
before/after table is in the ledger (§17).

## What we take from it

1. **Off-the-shelf foundation models are not volatility models.** If a
   vendor pitches one for risk, ask for the QLIKE table against RiskMetrics.
2. **Fine-tuning closes the gap to the classical frontier, not past it.**
   Both attempts tied RiskMetrics and HAR; neither beat them significantly.
   With only daily closes, a λ = 0.94 EWMA sits very close to the best you
   can do. Beating it plausibly needs better information, such as intraday
   realized variance, rather than a bigger model.
3. **Log-variance models are fragile to preprocessing.** Changing only how
   zero-return days are floored moved Chronos-2 on the unseen assets from
   3.2× to 8.4×. RiskMetrics barely moved (0.4944 → 0.4940). Robust models
   are worth a lot in risk.
4. **The table is worth more than the model.** Most of what a buyer needs is
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

The self-hosted service's `POST /v1/volatility` returns RiskMetrics and HAR
variance forecasts with vol-targeted position sizes: the models that held up
here. Chronos models load through the same registry
(`TIMESFM3_CHRONOS_MODELS=...`), so you can rerun this comparison yourself.

## Method, in one paragraph

Data: public FRED daily closes (equity indices, FX, commodities, Treasuries,
crypto). Variance proxy: squared daily log return, with zero-return days
floored at a level fixed from each asset's first year. Splits by date: train
before 2015, validation 2015–2019, test 2020-01-01 to 2026-11. Context of 512
trading days. Targets: next-day variance and mean variance over the next five
days. Loss: QLIKE (Patton 2011), averaged across assets per date. Verdicts:
HAC Diebold–Mariano per competitor with Holm correction. Attempt 1: 16
assets, 28,398 test forecasts. Attempt 2: 11 never-seen assets, 21,410
forecasts. The full protocol, every training run (including the ones that
did worse), the erratum and the raw result files, before and after the fix,
are in [`docs/VERIFICATION.md`](../VERIFICATION.md) §11–17 and
[`results/`](../../results/).

## Caveats

- Daily closes only. Intraday data would change the frontier, and probably
  the ranking.
- Ratios against RiskMetrics are derived from each test's QLIKE values; the
  formal significance tests were run against our fine-tuned candidate, as
  pre-registered.
- The confidence intervals in the result files resample dates independently,
  which understates uncertainty for overlapping five-day targets; verdicts
  use the HAC p-values, which account for it.
- One horizon is shown here (five days). Next-day results point the same way
  and are in the ledger.
- "Off the shelf" means no fine-tuning by us. Chronos' training corpora are
  not fully published, so we cannot rule out that some of these series were
  in them. A fine-tuned Chronos was not tested.
- Our fine-tuned candidates were trained before the fix, on inputs built with
  the old floor, and were not retrained.
