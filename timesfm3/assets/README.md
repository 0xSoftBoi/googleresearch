# Bundled checkpoints

`ModelRegistry.from_env()` — and therefore `timesfm3 serve`, `timesfm3
forecast` and `timesfm3 backtest` — registers every `*.pt` file in this
directory. Set `TIMESFM3_NO_BUNDLED=1` (or pass `--no-bundled`) to skip them.

## starter-small.pt

Produced by `scripts/train_starter.py` (`make starter-model`); provenance is
stored in the checkpoint's `meta` block and shown on `GET /v1/models`.

- Architecture: TimesFM-3 `small` config — 6 alternating layers, model dim
  256, 8 heads, 32-step patches, ≈5.2 M parameters, context up to 2048
  steps, 128 steps per single-pass decode (longer horizons roll forward).
- Corpus: ETTh1, ETTm1, ETTm2 (electricity-transformer telemetry) and daily
  exchange rates, mixed 70/30 with the synthetic multivariate generator;
  multi-frequency subsampling, calendar covariates through the known-future
  pathway, random demotion of channels to past-only covariates.
- Held out: ETTh2 in its entirety. The dashboard's sample panel is the
  ETTh2 tail, so the demo is a zero-shot forecast.
- Weights are stored in float16 (`timesfm3.checkpoint.package_checkpoint`)
  and restored to float32 on load.

Measured results are in `docs/PRODUCT.md` (model card section).
