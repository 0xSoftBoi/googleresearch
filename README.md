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
  train.py           # pre-training loop with held-out validation
examples/
  forecast_example.py  # API demo: multivariate targets + covariates
  plot_forecast.py     # point + q10-q90 band vs ground truth
  evaluate.py          # held-out synthetic eval vs naive baselines
  evaluate_ett.py      # zero-shot eval on the real ETTh1 benchmark
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
