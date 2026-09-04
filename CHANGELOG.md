# Changelog

## Unreleased — verification pass

- `docs/VERIFICATION.md`: every measured figure in the README, model card,
  hedge-fund and Polymarket sections re-run from a clean container on
  2026-09-04, and every external citation re-fetched; ledger of what
  reproduced, what did not, and what was corrected.
- Corrections: Rahimikia, Ni & Wang (2025) is a Manchester/UCL paper, not
  "Man Group-affiliated"; Hurst–Ooi–Pedersen's century-scale *net* Sharpe is
  0.77, the 0.4 figure was a stress assumption; Nixtla Series A and ORATS /
  alt-data spend now cite pages that actually contain the figures; Pecan
  pricing restated from third-party listings; BigQuery `AI.FORECAST` serves
  TimesFM 2.5 by default; the released model's parameter count is stated by
  Google as 330 M (≈334 M as configured here) and its 16k context is a
  TimesFM 2.5 figure; the anomaly "two planted spikes" anecdote replaced by a
  reproducible statement; the fine-tune gain reported as a range across runs;
  unit economics re-measured (faster than the earlier draft claimed).

## 0.8.0 — built on Cloudflare's own primitives

- Credits are now standard **Privacy Pass** tokens (RFC 9576/9577/9578,
  token type `0x0002` Blind RSA): `timesfm3/privacypass.py` (challenges,
  token requests/responses, generic batched issuance, issuer directory,
  `PrivateToken` HTTP auth) and `timesfm3/serving/privacypass.py` (issuer +
  origin with lossless rotation). Endpoints: `/.well-known/private-token-issuer-directory`,
  `GET /token-request/challenge`, `POST /token-request`,
  `POST /token-request/batch/{10|25|100}`, `GET /token-request/stats`;
  priced calls answer `401 WWW-Authenticate: PrivateToken ...` and accept
  `Authorization: PrivateToken token="..."`. `timesfm3.credits.CreditWallet`,
  the client and `timesfm3 credits buy|status` speak the standard protocol.
  Replaces the bespoke `/v1/credits/*` and `X-Credit` API.
- Cloudflare Worker rewritten on **Hono** (`cloudflare/src/index.js`) with the
  official `@x402/hono` paywall middleware and Privacy Pass issuance and
  redemption through `@cloudflare/privacypass-ts` (`cloudflare/src/privacy-pass.js`);
  D1 binding `PRIVACY_PASS_DB`, secret `PRIVACY_PASS_PRIVATE_JWK`, vars
  `PRIVACY_PASS_OLD_PUBLIC_JWKS`, `PRIVACY_PASS_ISSUER_NAME`,
  `PRIVACY_PASS_ORIGIN`. Tokens issued by the Python service and the Worker
  are interchangeable; interop is tested in every issuer/client/origin
  combination, including against Cloudflare's library.
- Configuration renamed: `TIMESFM3_PRIVACY_PASS_{KEY_FILE,OLD_KEYS,LEDGER_FILE,PRICE,ORIGIN,ISSUER_NAME}`.

## 0.7.0 — standard blind signatures, key rotation, credits at the edge

- `timesfm3/blindrsa.py`: RFC 9474 RSA blind signatures (all four suites),
  pure Python, validated against the RFC's test vectors and cross-checked
  with `@cloudflare/blindrsa-ts` in both directions.
- The credit pool now uses RSABSSA-SHA384-PSSZERO-Deterministic; keys are
  JWK files shared by the service and the Worker; overlapping keys make
  rotation lossless (`TIMESFM3_CREDITS_OLD_KEYS`, `CREDITS_OLD_PUBLIC_JWKS`);
  `/v1/credits/pool` lists keys with the issuing flag; blinded messages and
  signatures travel as base64url.
- The Cloudflare Worker issues and redeems credits itself in edge-native
  mode: `cloudflare/src/credits.js`, D1 ledger (`migrations/0001_credits.sql`,
  database created in the account), x402-priced batches, `x-credits-spent`.
- `scripts/credits_keygen.py`; tests for vectors, interop, rotation, the
  edge pool with a D1 shim, and a Python-issued token redeemed at the edge.

## 0.6.0 — privacy pools

- **Unlinkable prepaid credits**: RSA-FDH blind signatures (`timesfm3/credits.py`
  client wallet, `timesfm3/serving/credits.py` pool). `GET /v1/credits/pool`,
  `POST /v1/credits/buy/{10|25|100}` (paid with x402 at $0.004/credit or with
  plan points), `X-Credit` redemption with nullifiers, persisted key and
  ledger. Costs: forecast 1, volatility 1, anomalies 2, backtest 4.
  `timesfm3 credits buy|status`; `ForecastClient(credits=...)` spends
  automatically. The edge passes `X-Credit` through the paywall.
- **docs/PRIVACY.md**: what each payment channel reveals, the credit pool's
  guarantees and limits, and funding purchases from a Privacy Pools
  (0xbow) withdrawal, with the operator-side treasury flow.

## 0.5.0 — pay per call with x402

- The service answers priced endpoints with x402 `402 Payment Required`
  challenges (USDC on Base via the official `x402` package): $0.005
  forecast, $0.005 volatility, $0.01 anomalies, $0.02 backtest. API-key
  holders bypass the paywall and stay metered; anonymous callers pay per
  request and are metered under an `x402` identity. `GET /v1/pricing`
  publishes both channels. Config via `TIMESFM3_X402_*`.
- The Cloudflare Worker enforces the same paywall in gateway mode with a
  small spec-exact implementation (`cloudflare/src/x402.js`): verify before
  the handler, settle after, `PAYMENT-RESPONSE` on success, CORS for the
  payment headers; optional on the edge classical API.
- Tests with a mocked facilitator for both, plus a cross-check that the
  Worker's challenge parses with the official Python models.

## 0.4.0 — the whole product on Cloudflare's free tier

- `scripts/export_onnx.py`: TimesFM-3 checkpoints export to ONNX (dynamic
  series/time axes, horizon as a tensor input) with a PyTorch parity check.
- `cloudflare/public/js/timesfm3-onnx.js`: the model runs in the browser via
  ONNX Runtime Web with the Python forecaster's padding, rolling decode and
  quantile repair.
- `cloudflare/public/js/forecast.js`: JS port of the classical baselines,
  empirical bands, Diebold–Mariano / bootstrap / Holm, walk-forward backtest
  and anomaly scoring; `tests/test_js_parity.py` pins it to the Python
  numbers.
- The Worker now has an edge-native mode (no `API_ORIGIN`): `/v1/forecast`,
  `/v1/backtest`, `/v1/anomalies` for classical models in the free plan's
  CPU budget, `/v1/models`, `/v1/sample`, `/healthz`, a static `/docs`, and
  the in-browser dashboard at `/app`. Rate limiting moved to the Workers
  rate-limit binding (no KV writes).
- Deploy-to-Cloudflare button; no token, no server, no bill.

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
