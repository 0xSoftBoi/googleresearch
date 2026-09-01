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
  data/synthetic.py  # synthetic multivariate pre-training corpus generator
  data/real.py       # real-benchmark corpus (ETT, exchange rates) as RealSource
  data/polymarket.py # Polymarket prediction-market archive -> RealSource
  train.py           # pre-training loop with held-out validation
examples/
  forecast_example.py     # API demo: multivariate targets + covariates
  plot_forecast.py        # point + q10-q90 band vs ground truth
  evaluate.py             # held-out synthetic eval vs naive baselines
  evaluate_ett.py         # zero-shot eval on the real ETTh1 benchmark
  evaluate_polymarket.py  # benchmark forecasters on the Polymarket archive
tests/
  test_polymarket.py      # archive-loader unit tests (pytest)
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
inputs, and `examples/evaluate_polymarket.py` benchmarks forecasters on it.

Every Polymarket market is binary: two complementary outcome tokens, one
`condition_id`. The archive records each quote and fill with microsecond
arrival timestamps (~1 GB and ~10^8 rows *per hour*), so each market carries
its own microstructure. The loader streams those hours row-group by row-group
and rebuilds, per outcome token and per grid cell:

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
data/download_polymarket.sh 2026-08-29T03 2026-08-29T15   # checksum-verified
python examples/evaluate_polymarket.py --start 2026-08-29T03 --end 2026-08-29T15
```

```python
import datetime as dt
from timesfm3.data.polymarket import load_polymarket_sources

hour = dt.datetime(2026, 8, 29, 3, tzinfo=dt.timezone.utc)
sources = load_polymarket_sources(hour, hour, num_markets=64)  # -> RealSource list
```

### What is actually forecastable here

Measured on 13 contiguous checksum-verified hours (2026-08-29 03:00–15:59 UTC),
163 continuously quoted markets, 15 s grid. Lag-1 autocorrelation of each
channel, and which trivial predictor wins over a 16-minute horizon:

| channel | lag-1 AC | better baseline |
|---|---|---|
| `mid` | 0.982 | last-value |
| `spread` | 0.910 | last-value |
| **`ret`** | **-0.009** | context-mean |
| `abs_ret` | 0.204 | context-mean |
| `quotes` | 0.812 | last-value |
| `trades` | 0.568 | last-value |
| `volume` | 0.332 | last-value |

Price *levels* are near-perfect random walks (AC 0.982) and their **returns
carry essentially no autocorrelation (-0.009)** — which is what an efficient
market should look like. Book *activity* is a different story: quote intensity
is strongly autocorrelated, and that is where a sequence model has something to
learn.

Two further properties of this data shape any honest evaluation:

- **Most windows are frozen.** Over a 16-minute horizon, 71–90% of windows have
  a target that never moves at all. There last-value is exactly right by
  construction and no forecaster can beat it, so those windows are reported
  separately rather than being allowed to dominate an average.
- **Scaled MAE is unusable.** The repo's other benchmarks divide MAE by the
  context standard deviation; here contexts are frequently *perfectly flat*, so
  that denominator collapses and produces meaningless six-figure "errors". The
  Polymarket benchmark reports MAE in native units plus a ratio to the better
  baseline.

### Benchmark

A 1M-parameter (`tiny`) model, trained for 3.6 minutes on CPU on the 13-hour
window, forecasting `mid, spread, abs_ret, quotes` jointly. Two-way held out:
the last 20% of the time axis is never trained on, and 25% of markets are
excluded from training entirely. Numbers are model MAE ÷ best-baseline MAE
over **active** windows (frozen windows excluded); below 1.00 the model wins.

| channel | train markets | zero-shot markets |
|---|---|---|
| `mid` | 1.060 | 1.039 |
| `spread` | 0.997 | 1.008 |
| `abs_ret` | 1.128 | 1.441 |
| **`quotes`** | **0.931** | **0.918** |

A second, independent run over `mid, quotes, trades, volume` agrees on the two
findings that matter: `quotes` 0.910 / 0.975 and `mid` 1.047 / 1.102.

So: **the model does not beat a random walk on price, and beats both baselines
on book activity, including zero-shot on markets it never saw.** The first half
is the expected result, not a failure — a small model trained on 13 hours of one
day should not out-predict a liquid prediction market. The second half is real
signal: quote intensity has learnable temporal structure that neither trivial
baseline captures. Caveats worth keeping in view: this is one day of data, a
1M-parameter model, and the per-window win rate on `quotes` is only ~0.37 — the
model wins on aggregate error because it wins on the *active* minority of
windows, not because it wins most windows.

### Data integrity

Hourly files are ~1 GB and large transfers do get truncated in flight, so
`PolymarketArchive` verifies each download against the archive's
`SHA256SUMS.txt` *before* promoting it into the cache, repairs a cached file
that fails its checksum, and distinguishes two failures that need different
responses: downloads that disagree with *each other* (flaky transfer — retry
helps) from downloads that agree with each other but not with the manifest
(the archive is serving bytes its own manifest does not describe — retrying
cannot help; pass `on_mismatch="warn"` to accept them).

That distinction is not hypothetical. Auditing the 16 hours originally fetched
for this benchmark, 15 verified and one (`2026-08-29T02`) reproducibly hashed to
a value the published manifest does not list, across four independent
full-length downloads. The benchmark above therefore uses the contiguous
verified block 03:00–15:59 rather than forward-filling across a suspect hour.

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
