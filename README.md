# TimesFM-3: A Zero-Shot Foundation Model for Multivariate Forecasting

A PyTorch implementation of the TimesFM-3 architecture described in the Google
Research blog post
["TimesFM-3: A zero-shot foundation model for multivariate forecasting"](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/).

TimesFM-3 is a decoder-style time-series foundation model that natively
supports:

- **Multiple simultaneous target series** with point and quantile forecasts.
- **Past-only covariates** (historical features whose future is unknown).
- **Past–future covariates** (signals known ahead of time, e.g. holidays,
  scheduled promotions, weather forecasts).
- **Single-forward-pass horizon decoding** — no autoregressive roll-out.

## Architecture (as implemented here)

| Component | Blog description | Where |
|---|---|---|
| Patching | Contiguous points grouped into patches of **32 time steps** | `timesfm3/embedding.py` |
| Normalization | Per-time-series (reversible) normalization from context statistics | `timesfm3/normalization.py` |
| Token construction | Standard series: one token per patch. Future-known covariates: **lookahead** — the current patch is concatenated with future patches so the model can peek at upcoming known signals | `timesfm3/embedding.py` |
| Alternating attention | A 2D grid of tokens (series × time). **Temporal attention** is strictly causal within each series; **cross-variate attention** lets a token attend to every other series at the same time step | `timesfm3/attention.py`, `timesfm3/blocks.py` |
| Decoding | **Contiguous Patch Masking**: target and past-only covariate patches are masked over the horizon while past–future covariates remain visible, and the whole horizon is produced in one forward pass. Horizons beyond the single-pass maximum roll forward chunk by chunk | `timesfm3/model.py`, `timesfm3/forecaster.py` |
| Output | **9 quantiles (q10 … q90, median at index 4)** plus a point forecast for every target at every horizon step, with optional quantile-crossing repair | `timesfm3/model.py`, `timesfm3/forecaster.py` |
| Scale | Base config matches the released TimesFM-3 dimensions — 20 layers, model dim 1280, 16 heads — at ~334 M parameters (`small` and `tiny` configs included for experimentation) | `timesfm3/configuration.py` |

A detailed walkthrough of the token grid, attention layout, and known
deviations from the released model lives in [ARCHITECTURE.md](ARCHITECTURE.md).

## Layout

```
timesfm3/
  configuration.py   # model / training hyper-parameters (base ≈ 334M params)
  normalization.py   # per-series reversible normalization (context statistics)
  embedding.py       # patching, residual-MLP patch embedding, lookahead tokens
  attention.py       # multi-head attention with rotary embeddings (temporal)
  blocks.py          # alternating temporal / cross-variate transformer layers
  model.py           # TimesFM3Model: contiguous patch masking + quantile head
  forecaster.py      # numpy-in / numpy-out API: rolling decode, NaN handling
  loss.py            # normalized quantile (pinball) + point losses
  baselines.py       # classical forecasters: last-value, drift, EWMA, AR(p)
  evaluation.py      # Diebold-Mariano (HAC), cluster bootstrap, Holm
  data/synthetic.py  # synthetic multivariate pre-training corpus generator
  data/real.py       # real-benchmark corpus (ETT, exchange rates) as RealSource
  data/polymarket.py # Polymarket prediction-market archive -> RealSource
  train.py           # pre-training loop with held-out validation
examples/
  forecast_example.py     # API demo: multivariate targets + covariates
  plot_forecast.py        # point + q10-q90 band vs ground truth
  evaluate.py             # held-out synthetic eval vs naive baselines
  evaluate_ett.py         # zero-shot eval on the real ETTh1 benchmark
  train_polymarket.py     # train on Polymarket, writing a held-out market split
  evaluate_polymarket.py  # benchmark forecasters on the Polymarket archive
tests/
  test_polymarket.py      # archive-loader unit tests (pytest)
  test_baselines.py       # known-answer tests for the classical baselines
  test_evaluation.py      # known-answer tests for the comparison statistics
```

## Quick start

```bash
pip install -e .
python examples/forecast_example.py
```

```python
import numpy as np
from timesfm3 import TimesFM3Config, TimesFM3Forecaster

forecaster = TimesFM3Forecaster(TimesFM3Config.small())  # or .base() for ~334M
# ... or load weights you trained: TimesFM3Forecaster.from_checkpoint("ckpt.pt")

result = forecaster.forecast(
    targets=[np.sin(np.arange(512) / 10.0)],       # one or more target series
    past_covariates=[np.random.randn(512)],        # optional, history only
    future_covariates=[np.random.randn(512 + 128)],# optional, known over horizon
    horizon=128,
)
result.point      # (num_targets, horizon) point forecast
result.quantiles  # (num_targets, horizon, 9) q10 ... q90
```

## Real-data pre-training (notebook)

[`notebooks/timesfm3_real_data.ipynb`](notebooks/timesfm3_real_data.ipynb)
is the productive-training path: it pre-trains on a **real corpus** (ETTh1,
ETTm1, ETTm2, exchange rates — `bash data/download.sh` fetches them) mixed
with synthetic data, using three tricks from `timesfm3/data/real.py`:

- **multi-frequency augmentation** — random-stride subsampling so one
  dataset teaches several sampling rates,
- **calendar covariates** — day/week sin/cos phases fed through the
  past-future covariate pathway, known arbitrarily far into the future,
- **role randomization** — real channels randomly demoted to past-only
  covariates.

It then evaluates a claim the synthetic demos cannot make: **zero-shot on a
held-out dataset** (ETTh2 is never seen in any form), with a calendar-
covariate ablation and a quantile-calibration analysis. Results from the
executed notebook (5.2M params, 25 min on CPU, scaled MAE):

| Evaluation | model | last-value | seasonal-naive |
|---|---|---|---|
| ETTh1, in-domain, held-out time, + calendar | **0.66** | 1.10 | 0.73 |
| ETTh1, in-domain, held-out time, no calendar | 0.81 | 1.10 | 0.73 |
| ETTh2, **zero-shot dataset**, + calendar | **0.76** | 1.06 | 0.92 |
| ETTh2, **zero-shot dataset**, no calendar | 0.87 | 1.06 | 0.92 |

Unlike the synthetic-only checkpoint, the real-corpus model **beats
seasonal-naive zero-shot on a dataset it has never seen**; calendar
covariates through the known-future pathway contribute a further ~13%,
and the 9 quantiles are calibrated to a mean absolute coverage gap of
0.064 zero-shot.

## Prediction markets (Polymarket order-book archive)

`timesfm3/data/polymarket.py` turns the public Polymarket order-book archive at
[archive.pendulumflow.com](https://archive.pendulumflow.com/) into TimesFM-3
inputs; `examples/train_polymarket.py` and `examples/evaluate_polymarket.py`
train and benchmark on it.

Every Polymarket market is binary: two outcome tokens, one `condition_id`. In
this archive the two mid prices are *exactly* complementary (`p_yes + p_no == 1`
to the tick, zero variance), so only one outcome carries information and the
loader uses one per market. The archive records every quote and fill with
microsecond arrival timestamps (~1 GB and ~10^8 rows *per hour*); the loader
streams those hours row-group by row-group and rebuilds, per grid cell:

| channel | meaning |
|---|---|
| `mid` | `(best_bid + best_ask) / 2`, last quote in the cell, forward-filled |
| `spread` | `best_ask - best_bid`, same treatment |
| `ret`, `abs_ret` | first difference of `mid`, and its magnitude (realized-volatility proxy) |
| `volume`, `trades` | summed fill size and fill count in the cell |
| `signed_flow` | fills signed by taker side (BUY +, SELL −) |
| `quotes` | book-update count in the cell |

```bash
pip install -e .[polymarket]                              # adds pyarrow
data/download_polymarket.sh 2026-08-28T08 2026-08-28T15   # checksum-verified
python examples/train_polymarket.py --panels panels.pkl --checkpoint pm.pt
python examples/evaluate_polymarket.py --panels test_panels.pkl --checkpoint pm.pt
```

### What is forecastable

Lag-1 autocorrelation over 13 verified hours, 163 continuously quoted markets,
15 s grid:

| channel | `mid` | `spread` | `ret` | `abs_ret` | `quotes` | `trades` | `volume` |
|---|---|---|---|---|---|---|---|
| lag-1 AC | 0.982 | 0.910 | **−0.009** | 0.204 | 0.812 | 0.568 | 0.332 |

Price *levels* are near-perfect random walks and their **returns carry
essentially no autocorrelation** — what an efficient market should look like.
Book *activity* is strongly autocorrelated, and that is where a sequence model
has something to learn.

### Evaluation protocol

Three properties of this data break naive benchmarking, so the harness handles
each explicitly:

- **Most horizons are frozen.** 44–90% of windows have a target that never
  moves; there last-value is exactly right by construction and nothing can beat
  it. Frozen and active windows are scored separately.
- **Scaled MAE is unusable.** Dividing by context standard deviation — what
  this repo's other benchmarks do — collapses on those flat contexts and yields
  meaningless six-figure "errors". Losses are reported in native units.
- **Windows are not independent.** Sliding windows share context and windows
  from one market share its regime. Windows are taken **non-overlapping**,
  significance uses a HAC-corrected [Diebold-Mariano](timesfm3/evaluation.py)
  test, confidence intervals come from a bootstrap resampling whole *markets*,
  and p-values are Holm-corrected across forecasters. Effective sample size is
  reported alongside nominal n.

Forecasters are scored against a panel of classical baselines
(`timesfm3/baselines.py`): last-value, context-mean, drift, EWMA with its
smoothing constant fit inside the context, and AR(1)/AR(4) fit by OLS on the
context. A foundation model that cannot beat a fitted AR(1) is not interesting.

### Benchmark

**Trained on 2026-08-28 08:00–15:59, evaluated on 2026-08-29 03:00–15:59** — a
strictly later day, with the 2 overlapping markets removed from training, so
all 163 test markets are unseen. 1M-parameter (`tiny`) model, 3.4 min on CPU.
Active windows, non-overlapping, MAE ratio against last-value (< 1 is better);
`*` = significant at 5% after Holm correction.

| channel | TimesFM-3 | EWMA | AR(1) | ctx-mean | n (effective) |
|---|---|---|---|---|---|
| `mid` | 1.074 * worse | 1.008 | 1.160 * worse | 1.876 * worse | 180 (158) |
| `spread` | 1.013 | 0.996 | 1.005 | 1.079 | 158 (147) |
| `abs_ret` | 0.811 | 0.844 | 0.738 | 0.739 | 180 (180) |
| **`quotes`** | **0.704 \*** [0.574, 0.853] | 0.901 | 0.871 | 0.870 | 429 (282) |

**Nothing beats a random walk on price.** TimesFM-3 is significantly *worse*
than last-value on `mid`, as is AR(1); EWMA ties it. On `spread` every
forecaster is statistically indistinguishable from last-value. On `abs_ret`
nothing survives multiple-comparison correction.

**On quote intensity the model wins, and the win is not something a linear
model reproduces** — EWMA, AR(1) and context-mean all sit near 0.87–0.90 and
none reach significance, while TimesFM-3 reaches 0.704.

Robustness of that one positive result. The seed, comparison-space and horizon
rows re-run the same cross-day protocol with `--channels mid,quotes`, which is
why seed 0 reads 0.731 there and 0.704 in the four-channel table above — the
joint forecast changes with the channel set, the conclusion does not:

| check | result |
|---|---|
| 3 independent training seeds | 0.725 / 0.728 / 0.731 (all significant) |
| scored in log1p space instead of native counts | 0.866 / 0.870 / 0.873 (all significant) |
| horizon 32 / 64 / 128 steps (8 / 16 / 32 min) | 0.841 (n.s.) / 0.731 \* / 0.655 \* |
| leak check: same-day eval including training data | 0.689 — essentially unchanged |

The edge **grows with horizon**, which is the mechanically sensible direction:
at 8 minutes quote intensity persists and last-value is hard to beat; over
longer horizons the mean-reverting structure the model has learned starts to
pay. What this buys is a liquidity/activity signal useful for execution
scheduling — not alpha. The price result says plainly that there is none to be
had here.

Caveats: two days of data from one archive, a 1M-parameter model, and a
per-window win rate on `quotes` well below half — the model wins on aggregate
error because it wins on the active minority of windows, not because it wins
most of them.

### Data integrity

Hourly files are ~1 GB and large transfers do get truncated in flight, so
`PolymarketArchive` verifies each download against `SHA256SUMS.txt` *before*
promoting it into the cache, repairs a cached file that fails its checksum, and
distinguishes two failures needing different responses: downloads that disagree
with *each other* (flaky transfer — retry helps) from downloads that agree with
each other but not with the manifest (the archive is serving bytes its own
manifest does not describe — retrying cannot help; `on_mismatch="warn"` accepts
them).

That distinction is not hypothetical. Of 24 hours audited, 23 verified and one
(`2026-08-29T02`) reproducibly hashed to a value the manifest does not list,
across four independent full-length downloads. The benchmark uses contiguous
verified blocks rather than forward-filling across a suspect hour.

## Pre-training (synthetic only)

The released model was pre-trained on a real-world plus synthetic corpus of
more than 1 trillion time points. This repo ships the training objective
(quantile + point loss under contiguous patch masking) and a synthetic
multivariate corpus generator so the pipeline is runnable end to end:

```bash
python -m timesfm3.train --config tiny --steps 600 --batch-size 16 \
    --context-patches 8 --horizon-patches 2
python examples/plot_forecast.py --checkpoint timesfm3_checkpoint.pt
```

The tiny (1M parameter) config trains in ~25 minutes on CPU with held-out
validation and best-checkpoint saving. After 8000 steps (best validation
loss 1.405), scaled MAE against naive baselines:

| Evaluation | model | last-value | seasonal-naive | context-mean |
|---|---|---|---|---|
| Held-out synthetic (248 target series) | **0.94** | 1.26 | — | 1.18 |
| ETTh1, real data, zero-shot (40×7 windows) | **0.90** | 1.04 | 0.70 | — |

On real ETTh1 data — never seen in training — the model transfers well
enough to beat the last-value baseline, though 1M parameters of
synthetic-only pretraining does not yet beat a daily seasonal-naive on
strongly daily-periodic data; the released model's zero-shot quality comes
from its >1T-point corpus and 334× larger capacity.

![Demo forecast](docs/forecast_demo.png)

The plot (`examples/plot_forecast.py`) shows the single-pass decode on two
correlated targets with a known-future covariate: the point forecast
continues the seasonal phase, anticipates the covariate-driven peak, and
the q10–q90 band widens with lead time.

This is an independent re-implementation of the publicly described
architecture; no pre-trained weights are included, and the released
TimesFM-3 checkpoint (which carries its own non-commercial license) is a
separate artifact from this code.
