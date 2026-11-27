# Changelog

## 0.9.1 — stop the model research, launch the benchmark

- **Look-ahead fix (ledger §17).** `realized_variance` floored zero-return
  days with the full-history median and `ewma_variance` demeaned with the
  full-history mean, so later returns touched earlier values. The floor now
  comes from each asset's first 252 days and RiskMetrics runs on the same
  variance series as the targets (`ewma_of_variance`); a test rewrites every
  return after day k and asserts nothing up to k moves. Both volatility tests
  re-run with the locked candidates (`results/vol_test_causal.json`,
  `results/vol_test_v2_causal.json`; originals kept). Gates still fail.
  Withdrawn: the fine-tune "significantly beats HAR (7.9%)" and "lowest of
  eleven on unseen assets". HAR-IV now starts from raw history like HAR.
- Review fixes: `finetune_qlike` returned base weights without a validation
  set; `add_checkpoint(make_default=False)` displaced a Chronos default;
  `finetune --evaluate` no longer loads env-listed models; `/v1/forecast`
  rejects unequal-length targets when timestamps are given; Chronos
  quantiles are sorted; adapter tests skip only on network errors; the vol
  test commands refuse to overwrite a result file; the edge deploy job needs
  both Cloudflare secrets and applies D1 migrations; tests no longer
  hard-code a local checkout path.
- `docs/launch/volatility-benchmark.md`: launch write-up of the volatility
  results (ledger §11–17). Used off the shelf, Chronos models lose to
  RiskMetrics by 32–48% on 2020–2026 five-day QLIKE (the bundled starter by
  3.6×); two pre-registered fine-tunes tie RiskMetrics and HAR without
  significantly beating either.
- `docs/LAUNCH.md`: deploy checklist (the edge Worker is not yet deployed),
  Show HN / Reddit / X copy, a direct-outreach script for quant and risk
  teams, day-14 success metrics, and what we are deliberately not doing.
- Landing page: new volatility benchmark section; the unverifiable "two
  planted spikes" anomaly sentence (flagged in ledger §1) and the unproven
  "base config closes the gap" roadmap claim are replaced with the measured
  Chronos result. README links the write-up.

## 0.9.0 — measured against the competition, research re-verified

### Volatility attempt 2: more data (pre-registered; gate failed, not released)

- `timesfm3.data.markets`: `EXTRA_UNIVERSE` (Nasdaq-100, Dow, 12 currencies,
  1/5/30-year Treasuries), `CRYPTO_UNIVERSE` (separate calendar), CBOE
  implied-vol mapping. `timesfm3/quant/volmodel.py`: implied-variance input
  channel (masked when absent), HAR-IV and iv-or-riskmetrics competitors,
  implied-vol window oversampling. `examples/hedge_fund/train_vol_model_v2.py`.
- `docs/VERIFICATION.md` §14–16: four configurations never beat the
  attempt-1 weights on validation; on 11 never-seen assets the model has the
  lowest five-day QLIKE of eleven (3.4% below RiskMetrics, 1% below HAR) but
  not significantly, so not released. Results: `results/vol_test_v2.json`.
  *Corrected in 0.9.1 (ledger §17): HAR is lowest after the look-ahead fix.*

### Volatility specialization (pre-registered; gate failed, not released)

- `timesfm3/quant/volmodel.py`, `examples/hedge_fund/train_vol_model.py`:
  log-variance inputs with causal smoothed-variance covariates, date splits,
  batched TimesFM-3 and Chronos forecasters, symmetric validation-fitted
  calibration, and a QLIKE fine-tuning loop. Tests pin split boundaries,
  feature causality, and batched-vs-forecaster equivalence.
- `docs/VERIFICATION.md` §11–13: protocol and gate committed before the
  test; six configurations logged; the selected model, fine-tuned from
  starter-small, ties RiskMetrics on 2020–2026 five-day QLIKE (0.988×,
  p 0.53) and significantly beats HAR (7.9%) and every Chronos model
  (29–41%). The gate required beating RiskMetrics too, so it is not
  released. Results: `results/vol_test.json`. *Corrected in 0.9.1 (ledger
  §17): the HAR win is not significant after the look-ahead fix.*

### Re-audit and benchmark (2026-11-21)

- `docs/VERIFICATION.md` §9: Chronos-Bolt-tiny/-small and Chronos-2 served
  through the registry and backtested with the model-card command. All three
  beat the bundled starter (Bolt-tiny MAE ratio 0.783 [0.738, 0.845], Holm
  p < 0.001); the starter row reproduces the model card exactly. Leakage
  caveat recorded; per-model CPU latency measured.
- `ChronosAdapter` crashed on Chronos-2 (list-shaped `predict_quantiles`
  output; only Bolt had been tested). Fixed; an offline stub test pins both
  output conventions and runs without `chronos-forecasting` installed. The
  adapter's covariate error no longer claims Chronos has no covariate input.
- `docs/VERIFICATION.md` §10: every claim the 2026-09-13 BUSINESS.md draft
  added was re-read at its source. Removed as unsupported: the x402
  "$50M / 165M transactions / 69,000 agents" and "Solana 35M" figures,
  Toto "~70% private telemetry", Kronos "102M", the MLX/Node ports, the AP2
  paragraph. Reworded: Toto's benchmark position (now quoted from Datadog),
  TimeGPT 2.1 status, the x402 "$7B" figure (a token-category market cap),
  AIMA percentages and date.
- `docs/BUSINESS.md`: TimesFM 3.0 is now in BigQuery `AI.FORECAST` (Preview)
  and Google permits commercial use there, so §1/§3 no longer claim there is
  no commercial TimesFM-3; positioning leads with model-agnostic, measured
  serving; "classical models are cheaper still" corrected (they are slower
  as served). The landing-page FAQ no longer says the registry serves Toto,
  Granite or Kronos checkpoints.

### Chronos adapter

- `timesfm3/serving/adapters.py`: `ChronosAdapter` runs Amazon's own
  `chronos-forecasting` package (Chronos / Chronos-Bolt) through the
  registry's forecaster contract; `add_chronos()` registers it, and
  `TIMESFM3_CHRONOS_MODELS` loads one or more alongside TimesFM-3
  checkpoints with no code changes (`timesfm3[chronos]` extra).
- `ModelRegistry.add_forecaster` generalized to accept `kind` and
  `supports_covariates`; `ModelEntry.forecast()` dispatch generalized from
  `kind == "timesfm3"` to "has a forecaster", so a foreign model reports its
  real kind on `GET /v1/models` instead of being mislabeled `timesfm3`.
  Default-model selection generalized the same way.
- `tests/test_adapters.py` runs a real forecast against the actual
  Chronos-Bolt-tiny weights (skipped, not mocked, when the optional
  dependency or network is unavailable).
- `docs/BUSINESS.md`, `docs/PRODUCT.md`: corrected an earlier claim that the
  registry already served "any checkpoint" from a competitor — it only
  served this repo's own `.pt` format. That claim is now true for Chronos,
  documented as an adapter (a few dozen lines), not a config toggle.

### Verification pass (2026-09-04 audit, merged into this release)

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
