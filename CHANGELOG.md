# Changelog

## 0.3.1 — Cloudflare edge front end

- `cloudflare/`: one Worker (no build step) serving the landing page with a
  live forecast demo, pricing and waitlist; proxying `/v1/*`, `/healthz`,
  `/docs` to the service with the upstream key attached server-side
  (bring-your-own-key passes through); per-IP rate limiting and lead
  storage in Workers KV; 60 s edge cache for `/v1/models`, `/v1/sample`,
  `/healthz`; dashboard at `/app`; `/api/leads` for the founder;
  `/metrics` never exposed. GitHub Actions deploy on push to `main`.

## 0.3.0 — what the market sells

Driven by the competitive research in `docs/BUSINESS.md`.

- **Anomaly detection**: `POST /v1/anomalies` and `timesfm3 anomalies` score
  every observation against the forecast made before it was seen.
- **Fine-tuning**: `timesfm3 finetune` adapts a checkpoint to a CSV panel,
  validates on the held-out tail, packages the result and prints a
  DM-tested backtest of base vs fine-tuned vs classical baselines.
  `train()` accepts `init_state`; `RealWindowDataset(tail=True)` samples
  held-out windows.
- **API keys, plans, quotas, metering**: named keys from env or JSON file,
  monthly forecast-point quotas, `X-Usage-Points`/`X-Usage-Remaining`
  headers, `/v1/usage`, 429 on exhaustion, persistent counters.
- **Classical quantile bands** never narrower than the Gaussian √h estimate
  (cuts false alarms with few in-context origins).
- `LICENSE` (Apache-2.0) and `NOTICE`; `docs/BUSINESS.md`.
- 11 new tests (113 total).

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
