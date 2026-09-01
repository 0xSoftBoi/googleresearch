# Changelog

## 0.2.0 — forecasting service

Turned the research implementation into a deployable product.

- **Service**: `timesfm3 serve` runs a FastAPI app with `/v1/forecast`,
  `/v1/backtest`, `/v1/volatility`, `/v1/models`, `/healthz`, `/metrics`,
  OpenAPI docs and a single-file dashboard (paste/upload CSV, fan chart,
  backtest table). Optional API key, request size limits.
- **Model registry**: serves TimesFM-3 checkpoints and the classical
  baselines through one interface; classical models get walk-forward
  empirical quantile bands.
- **Bundled starter checkpoint** (`timesfm3/assets/starter-small.pt`):
  small config pre-trained on real data, packaged in half precision with
  provenance metadata; `scripts/train_starter.py` reproduces it.
- **Checkpoint packaging** (`timesfm3.checkpoint`, `timesfm3 pack`): fp16
  weights + metadata, loadable by `TimesFM3Forecaster.from_checkpoint`.
- **CLI** (`timesfm3 serve|forecast|backtest|models|pack|train`) and a
  dependency-free Python client (`timesfm3.client.ForecastClient`).
- **Tabular I/O** for CSV in / long-format CSV or JSON out with timestamps.
- **Packaging**: `serve` extra, console script, Dockerfile, docker-compose,
  Makefile, GitHub Actions CI (tests on 3.10/3.12 + Docker boot probe).
- 33 new tests (102 total).

## 0.1.0

Independent PyTorch implementation of TimesFM-3 with synthetic and real
pre-training, classical baselines, significance testing, Polymarket and
FRED data loaders, and hedge-fund application harnesses.
