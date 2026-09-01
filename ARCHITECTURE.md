# TimesFM-3 architecture notes

This document maps the publicly described TimesFM-3 design onto this
implementation, and records where we deviate.

## The token grid

Every forecasting task is laid out as a 2D grid of tokens: one row per
series (targets, past-only covariates, past-future covariates), one column
per 32-step patch. The grid spans **context and horizon together**:

```
                     context patches          horizon patches
                 ┌─────┬─────┬─────┬─────┐  ┌─────┬─────┐
target A         │  t1 │  t2 │  t3 │  t4 │  │ ▒▒▒ │ ▒▒▒ │   ▒ = masked
target B         │  t1 │  t2 │  t3 │  t4 │  │ ▒▒▒ │ ▒▒▒ │       (to forecast)
past covariate   │  t1 │  t2 │  t3 │  t4 │  │ ▒▒▒ │ ▒▒▒ │
future covariate │  t1 │  t2 │  t3 │  t4 │  │  t5 │  t6 │   ← stays visible
                 └─────┴─────┴─────┴─────┘  └─────┴─────┘
                            time →
```

**Contiguous Patch Masking (CPM):** over the horizon region, target and
past-only covariate patches are masked — their values are hidden, an
indicator channel is set, and a learned mask embedding is added — while
past-future covariates remain visible so known future signals (holidays,
scheduled events) can steer the forecast. Every masked token directly
emits its own patch of predictions, so the whole horizon decodes in **one
forward pass**; there is no autoregressive roll-out within a pass. (The
high-level forecaster does roll the pass forward for horizons beyond the
single-pass maximum.)

## Token construction (`timesfm3/embedding.py`)

- Values are normalized **per series** from visible-context statistics
  (`timesfm3/normalization.py`) and grouped into 32-step patches.
- Standard tokens (targets, past-only covariates): a residual MLP embeds
  `[patch values, observed flags]` — 2×32 inputs — into the model
  dimension. NaNs and padding enter as zeros with the flag off.
- Future-covariate tokens use **lookahead**: the current patch is
  concatenated with the next `lookahead_patches` patches (values + flags),
  letting the model peek at upcoming known signals while temporal
  attention itself stays strictly causal.
- A learned role embedding (target / past covariate / future covariate)
  and, at masked positions, a learned mask embedding are added.

## Alternating attention (`timesfm3/attention.py`, `timesfm3/blocks.py`)

Layers alternate between the two grid axes (pre-norm, RMSNorm, SiLU FFN):

- **Temporal attention** (even layers): series are folded into the batch
  and tokens attend across time with a **strictly causal** mask and rotary
  position embeddings — a token only sees past tokens of its *own* series.
- **Cross-variate attention** (odd layers): time steps are folded into the
  batch and tokens attend across **all series at the same time step**.
  Series form a set: no positional information is injected along this
  axis, so the model is permutation-invariant over variates. Padding
  variates (from batching examples of different widths) are masked out of
  the keys.

## Output head (`timesfm3/model.py`)

Each token decodes its own 32-step patch through a residual MLP producing
`32 × (1 + 9)` values: a point forecast plus 9 quantiles (q10 … q90, the
median at index 4) per time step. Outputs are mapped back to original
units with the stored normalization statistics. The forecaster optionally
repairs quantile crossings by sorting per step.

## Training objective (`timesfm3/loss.py`, `timesfm3/train.py`)

Training replicates the inference regime: mask the trailing horizon
patches (the length is randomized per step so every offset is learned) and
minimize point MSE + mean pinball loss **in normalized units** over masked
positions. Masked past-only covariate patches contribute at half weight as
an auxiliary signal. Validation runs on a fixed held-out stream and the
best checkpoint is kept.

## Dimensions

The base config matches the released TimesFM-3: 20 layers, model dim 1280,
16 heads, 32-step patches, ≈334M parameters, contexts up to 16k steps.

## Known deviations from the released model

- The released model uses "CPM Iterative RevIN" — normalization statistics
  updated iteratively per patch. We fit statistics once from the visible
  context (classic reversible instance normalization).
- The released model decodes 64-step output patches; here every token
  decodes its own 32-step patch, and long horizons use more masked patches
  (or rolling passes) instead.
- The released model was pre-trained on >1T real + synthetic time points
  (GiftEvalPretrain, Wikipedia pageviews, Google Trends, synthetic). This
  repo ships only the synthetic side, so trained checkpoints here
  demonstrate the pipeline, not the released model's zero-shot quality.
