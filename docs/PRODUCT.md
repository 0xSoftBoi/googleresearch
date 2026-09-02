# TimesFM-3 Forecast Service — product guide

A self-hosted forecasting service. Send it your time series, get back point
forecasts with calibrated quantile bands, and — before you trust any of it —
run a statistically honest backtest that tells you whether the model beats a
random walk on *your* data.

Everything runs from one command, on CPU, with a bundled model:

```bash
pip install "timesfm3[serve] @ git+https://github.com/0xSoftBoi/googleresearch"
timesfm3 serve
# dashboard  http://localhost:8000/
# OpenAPI    http://localhost:8000/docs
```

```bash
curl -s -X POST localhost:8000/v1/forecast -H 'content-type: application/json' \
  -d '{"targets":[{"name":"sales","values":[12,15,14,18,21,19,24,27,25,30,33,31]}],"horizon":4}'
```

## Who it is for

- **Product and ops teams** that need demand, load, traffic or capacity
  forecasts with uncertainty bands, behind an API they control.
- **Data teams** that want to compare a foundation model against classical
  baselines on their own data with proper significance tests, not vibes.
- **Quant and risk desks** that need volatility forecasts and
  volatility-targeted position sizes from daily returns.

## What you get

| Piece | What it does |
|---|---|
| `timesfm3 serve` | REST API (FastAPI) + dashboard + Prometheus metrics, optional API key |
| Bundled starter model | TimesFM-3 `small` config pre-trained on real data, ships in the package (~10 MB) |
| Classical baselines | last-value, drift, context-mean, EWMA, AR(1), AR(4), each with quantile bands |
| `/v1/backtest` | Walk-forward comparison with cluster-bootstrap CIs, HAC Diebold–Mariano tests, Holm correction |
| `/v1/volatility` | HAR and RiskMetrics variance forecasts, Moreira–Muir vol-targeted weight |
| `/v1/anomalies` | Walk-forward anomaly scoring against the model's own predictive band |
| `timesfm3 finetune` | Fine-tune the starter on your CSV, validate on its held-out tail, print the same DM-tested backtest |
| API keys & metering | Named keys with plans and monthly forecast-point quotas, `/v1/usage`, usage headers, 429 on exhaustion |
| x402 pay-per-call | Anonymous callers and AI agents pay per request in USDC over HTTP 402; no signup, settled on Base |
| Apache-2.0 code **and weights** | Google's TimesFM-3 weights are non-commercial; ours are self-trained (see `NOTICE`) |
| `timesfm3` CLI | forecast / backtest / anomalies / finetune on CSV files, list models, package checkpoints, train |
| `timesfm3.client` | Dependency-free Python client returning numpy arrays |
| Docker image | CPU-only, ~1 GB, health-checked, extra checkpoints via a mounted volume |

## Quick start

### Local

```bash
git clone https://github.com/0xSoftBoi/googleresearch && cd googleresearch
make install            # pip install -e ".[serve]"
timesfm3 serve          # http://localhost:8000
```

### Docker

```bash
docker compose up --build        # or: docker build -t timesfm3 . && docker run -p 8000:8000 timesfm3
```

Mount packaged checkpoints into `/models` and they are served next to the
bundled one.

### Forecast a CSV without a server

```bash
timesfm3 forecast data.csv --horizon 24 --output forecast.csv     # long format, q10..q90
timesfm3 forecast data.csv --horizon 24 --format json --model ewma
```

CSV layout: one row per time step, one column per series, optional leading
timestamp column and header. Empty cells are missing values.

### Find anomalies

```bash
timesfm3 anomalies metrics.csv --context 192 --block 24 --threshold 2
```

Each point is scored against the forecast the model made *before seeing
it*: `score = 1` sits on the q10/q90 edge, so the default threshold of 2 is
roughly 2.6σ for a Gaussian band (~1% of points). On seasonal test data with
two planted spikes, the bundled model flags exactly those two with no false
alarms; EWMA misses one and raises five — seasonality is what the neural
model buys you here.

### Fine-tune on your data and prove it helped

```bash
timesfm3 finetune sales.csv --out sales-v1.pt --name sales-v1 --steps 300 --periods 7
```

Starts from the bundled checkpoint, trains on the first 80% of your panel,
validates on the last 20%, packages the result, then runs the same
walk-forward backtest the API serves on that held-out tail — base model vs
fine-tuned vs the classical baselines — and prints the verdict. On the ETTh2
demo panel, 200 CPU steps (under a minute) cut MAE 2.3% versus the base
model. Serve the result with `timesfm3 serve --checkpoint sales-v1=sales-v1.pt`.

### Backtest before you believe anything

```bash
timesfm3 backtest data.csv --context 256 --horizon 24 --windows 20
```

```
timesfm3/serving/static/sample.csv: 7 series, 20 non-overlapping windows/series, context 256, horizon 24, MAE ratio vs last-value (< 1 is better)
model               mean mae   ratio            95% CI  p (Holm)   win    n (eff)  verdict
starter-small          2.183   0.881    [0.751, 1.008]     0.085   78%  140 (  33)  no difference
ewma                   2.309   0.931    [0.854, 0.982]     0.000   65%  140 (  23)  better
ar4                    2.432   0.981    [0.864, 1.073]     1.000   61%  140 (  56)  no difference
last-value             2.479   1.000                 -         -     -  140 ( 140)  reference
drift                  2.488   1.004    [0.993, 1.015]     1.000   51%  140 ( 140)  no difference
```

Every model is scored on the same non-overlapping windows; the ratio is its
mean loss over the reference's (below 1 is better); the interval is a
bootstrap that resamples whole series so windows from one series are not
treated as independent; the p-value is a HAC-corrected Diebold–Mariano test
with Holm correction across the models tested. **Deploy a model only when
its verdict is `better`.** In the run above the bundled model has the lowest
error and wins 78% of windows, yet its corrected p-value is 0.085 — the test
is telling you that 140 autocorrelated windows (33 effective) are not enough
evidence at 5%, which is precisely the kind of thing this endpoint exists to
say out loud.

## API reference

All `/v1` endpoints accept and return JSON. When API keys are configured
(see *API keys and plans* below), send one as `X-API-Key: <key>` or
`Authorization: Bearer <key>`; `/healthz` stays open for load balancers.

### `POST /v1/forecast`

```json
{
  "targets":           [{"name": "sales", "values": [12, 15, null, 18]}],
  "past_covariates":   [{"name": "temp",  "values": [3.1, 4.0, 2.2, 5.5]}],
  "future_covariates": [{"name": "promo", "values": [0, 0, 1, 0, 1, 1]}],
  "horizon": 2,
  "model": "starter-small",
  "quantiles": true,
  "timestamps": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
  "freq": "D"
}
```

- `targets` — one or more series of equal length; `null` is a missing value.
- `past_covariates` — same length as targets; their future is unknown.
- `future_covariates` — length `context + horizon`; known ahead (calendar,
  promotions, weather forecasts). Only TimesFM-3 checkpoints accept covariates.
- `timestamps` / `freq` — optional; when given, the response carries future
  timestamps (`freq` is inferred from the timestamps if omitted).

Response:

```json
{
  "model": "starter-small", "horizon": 2, "quantile_levels": [0.1, "...", 0.9],
  "timestamps": ["2024-01-05T00:00:00", "2024-01-06T00:00:00"],
  "forecasts": [{"name": "sales", "point": [19.7, 21.2],
                 "quantiles": {"q10": [15.1, 15.9], "q20": ["..."], "q90": [24.4, 26.8]}}],
  "latency_ms": 41.3
}
```

Errors: `400` invalid inputs (mismatched lengths, covariates on a classical
model), `404` unknown model, `413` request over the configured limits,
`422` schema violations.

### `POST /v1/backtest`

```json
{"series": [{"values": ["..."]}], "context": 256, "horizon": 24,
 "models": ["starter-small", "ewma"], "reference": "last-value",
 "windows": 20, "metric": "mae", "overlap": false}
```

Returns one `ModelScore` per model: `mean_loss`, `ratio`, `ci_low`,
`ci_high`, `p_value`, `p_adjusted`, `win_rate`, `n`, `n_effective`,
`verdict` (`better` / `worse` / `no difference` / `reference`).

### `POST /v1/volatility`

```json
{"returns": [0.004, -0.011, "..."], "horizon": 5, "vol_target": 0.10, "max_leverage": 3.0}
```

or `{"prices": [...]}`. Returns, for `riskmetrics` (λ = 0.94) and `har`
(Corsi 2009), the daily variance path, annualized vol, and the
Moreira–Muir weight `vol_target / forecast_vol` capped at `max_leverage`.

### `POST /v1/anomalies`

```json
{"series": [{"name": "cpu", "values": ["..."]}], "model": "starter-small",
 "context": 96, "block": 24, "threshold": 2.0, "timestamps": ["..."], "include_scores": false}
```

Returns, per series, `n_scored`, `n_flagged` and the flagged points
(`index`, `timestamp`, `value`, `expected`, `lower`, `upper`, `score`,
`direction`); `include_scores` adds the full per-step score and band arrays
for plotting.

### `GET /v1/usage`

The calling key's metered usage this month: `points_used`, `requests`,
`monthly_quota`, `points_remaining`. Every metered response also carries
`X-Usage-Points` (charged by this request) and, for quota-limited keys,
`X-Usage-Remaining`. A request that would exceed the quota returns `429`
with `Retry-After` and is not charged.

**Billable unit: forecast points** — one series × one horizon step. A
forecast charges `targets × horizon`; a backtest `series × windows × horizon
× models`; anomaly scoring one point per scored observation; volatility one
per horizon step.

### `GET /v1/models`, `GET /healthz`, `GET /metrics`, `GET /v1/sample`

Model inventory (name, kind, parameters, provenance, default flag);
liveness; Prometheus text exposition (requests, errors, latency, series and
steps forecast); a demo panel for the dashboard.

## Models

### Bundled starter model

`starter-small` ships inside the package (`timesfm3/assets/`). It is the
`small` TimesFM-3 config (≈5.2 M parameters) pre-trained by
`scripts/train_starter.py` on ETTh1, ETTm1, ETTm2 and daily exchange rates
mixed 70/30 with the synthetic corpus, with calendar covariates and role
randomization. ETTh2 was never seen; the dashboard's sample data is the
ETTh2 tail, so what you see there is a genuine zero-shot forecast.

Measured on ETTh2 (7 series, context 256, horizon 24, 20 non-overlapping
windows per series, MAE ratio vs last-value; `timesfm3 backtest`):

| model | mean MAE | ratio vs last-value | 95% CI | p (Holm) | win rate | n (eff.) | verdict |
|---|---|---|---|---|---|---|---|
| **starter-small** | 2.183 | **0.881** | [0.751, 1.008] | 0.085 | 78% | 140 (33) | no difference |
| ewma | 2.309 | 0.931 | [0.854, 0.982] | 0.000 | 65% | 140 (23) | better |
| ar4 | 2.432 | 0.981 | [0.864, 1.073] | 1.000 | 61% | 140 (56) | no difference |
| ar1 | 2.448 | 0.988 | [0.870, 1.090] | 1.000 | 56% | 140 (57) | no difference |
| last-value | 2.479 | 1.000 | — | — | — | 140 (140) | reference |
| drift | 2.488 | 1.004 | [0.993, 1.015] | 1.000 | 51% | 140 (140) | no difference |
| ctx-mean | 2.555 | 1.031 | [0.891, 1.135] | 1.000 | 55% | 140 (60) | no difference |

Read it plainly: the starter model has the lowest mean error of the seven
and wins more windows than any other (78%), but with only 33 effective
independent windows the HAC Diebold–Mariano test does not reach 5%
significance after Holm correction (p = 0.085). EWMA's smaller edge is
significant because its loss differential is far less autocorrelated. The
notebook's scaled-MAE numbers for the same recipe (0.76 vs 1.06) use a
different, per-window-normalized metric; the table above is the stricter,
deployment-relevant one.

Retrain it with `make starter-model` (≈35 min on 4 CPU cores; a GPU and the
`base` config are the path to released-model quality).

### Bring your own checkpoint

```bash
timesfm3 train --config small --steps 20000 --checkpoint my.pt     # or your own loop
timesfm3 pack my.pt my-packed.pt --name demand-v1 --description "..."
timesfm3 serve --checkpoint demand-v1=my-packed.pt --default-model demand-v1
```

Packaged checkpoints are half precision with a provenance `meta` block; the
server lists that metadata on `/v1/models`. Multiple checkpoints can be
served at once (`--checkpoint` is repeatable, or set `TIMESFM3_MODEL_DIR`).

### Classical baselines

Always registered. They have no quantile head, so their bands are
walk-forward residual quantiles estimated inside the context window
(nothing beyond the forecast origin is used), falling back to a Gaussian
band scaled by the one-step residual times √h when the context is short.

## API keys and plans

```bash
# one unlimited key
TIMESFM3_API_KEY=s3cret timesfm3 serve
# several keys with monthly quotas (name:key[:monthly_points])
TIMESFM3_API_KEYS="acme:ak-1:2000000,beta:bk-2:20000000,internal:ik-3" timesfm3 serve
# or a file, with plan labels
TIMESFM3_API_KEYS_FILE=keys.json TIMESFM3_USAGE_FILE=usage.json timesfm3 serve
```

```json
{"keys": [{"key": "ak-1", "name": "acme", "plan": "starter", "monthly_points": 2000000}]}
```

Counters reset per calendar month (UTC). With `TIMESFM3_USAGE_FILE` they
survive restarts; without it they live in memory. With no keys configured
the service is open and metered under one `anonymous` key, so the usage
headers and `/v1/usage` still work. Rate limiting per second is left to your
gateway; quotas here are monthly.

## Pay per call with x402 (USDC, no signup)

Anyone — an AI agent, a script, a customer without an account — can call
the priced endpoints and pay per request in USDC using the
[x402](https://x402.org) protocol, which rides on HTTP 402:

```bash
export TIMESFM3_X402_PAY_TO=0xYourWallet          # enables the paywall
export TIMESFM3_X402_NETWORK=eip155:84532         # Base Sepolia to test; eip155:8453 for Base mainnet
timesfm3 serve
```

| Endpoint | Price per call |
|---|---|
| `POST /v1/forecast` | $0.005 |
| `POST /v1/volatility` | $0.005 |
| `POST /v1/anomalies` | $0.01 |
| `POST /v1/backtest` | $0.02 |

How it works: a request without an API key gets `402 Payment Required` with
a base64 `PAYMENT-REQUIRED` header stating the amount, the USDC contract,
the network and your wallet. The client signs a USDC transfer authorization
and retries with `PAYMENT-SIGNATURE`. The service asks a facilitator to
verify it, serves the response, settles the transfer on-chain, and returns
`PAYMENT-RESPONSE` with the transaction hash. API-key holders never see the
paywall; their requests are metered against their plan as before. Prices
are overridable with `TIMESFM3_X402_PRICES` (JSON), and `GET /v1/pricing`
publishes both channels.

Facilitators: on Base Sepolia the free `https://x402.org/facilitator` is
used by default; on Base mainnet the default is Coinbase's CDP facilitator
(`https://api.cdp.coinbase.com/platform/v2/x402`, first 1,000 settlements a
month free, then $0.001 each — set `TIMESFM3_X402_FACILITATOR_AUTH` to the
`Authorization` header value CDP issues). Any x402 facilitator URL works
via `TIMESFM3_X402_FACILITATOR`.

Paying from code takes one wrapper around the HTTP client:

```python
# pip install "x402[evm]"
from eth_account import Account
from x402 import x402ClientSync
from x402.http.clients import x402_requests
from x402.mechanisms.evm import EthAccountSigner
from x402.mechanisms.evm.exact import ExactEvmClientScheme

client = x402ClientSync().register("eip155:*", ExactEvmClientScheme(EthAccountSigner(Account.from_key("0x..."))))
session = x402_requests(client)                        # a requests.Session that pays 402s
session.post("https://api.example.com/v1/forecast", json={...})   # pays $0.005, returns the forecast
```

```js
// npm i @x402/fetch @x402/evm viem
import { wrapFetchWithPayment, x402Client } from "@x402/fetch";
import { ExactEvmScheme } from "@x402/evm";
import { privateKeyToAccount } from "viem/accounts";
const client = new x402Client().register("eip155:*", new ExactEvmScheme(privateKeyToAccount("0x...")));
const paidFetch = wrapFetchWithPayment(fetch, client);
await paidFetch("https://api.example.com/v1/forecast", { method: "POST", headers: {"content-type": "application/json"}, body });
```

The Cloudflare Worker enforces the same paywall in gateway mode (anonymous
callers pay, bring-your-own-key callers are metered upstream); set
`X402_PAY_TO` in `wrangler.jsonc`.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `TIMESFM3_API_KEY` | unset | One unlimited key named `default` |
| `TIMESFM3_API_KEYS` | unset | Inline `name:key[:monthly_points]` list |
| `TIMESFM3_API_KEYS_FILE` | unset | JSON file of keys with plans and quotas |
| `TIMESFM3_USAGE_FILE` | unset | Persist monthly usage counters here |
| `TIMESFM3_X402_PAY_TO` | unset | Wallet that receives x402 payments; enables pay-per-call |
| `TIMESFM3_X402_NETWORK` | `eip155:84532` | CAIP-2 network (`eip155:8453` for Base mainnet) |
| `TIMESFM3_X402_FACILITATOR` | by network | Facilitator base URL (`/verify`, `/settle`) |
| `TIMESFM3_X402_FACILITATOR_AUTH` | unset | `Authorization` header for the facilitator (CDP) |
| `TIMESFM3_X402_PRICES` | see above | JSON overrides, e.g. `{"POST /v1/forecast": "$0.01"}` |
| `TIMESFM3_CHECKPOINTS` | unset | Comma-separated `[name=]path` checkpoints to serve |
| `TIMESFM3_MODEL_DIR` | unset (`/models` in Docker) | Serve every `*.pt` in this directory |
| `TIMESFM3_DEFAULT_MODEL` | last checkpoint added, else `ewma` | Model used when a request omits `model` |
| `TIMESFM3_NO_BUNDLED` | `0` | `1` disables the bundled starter model |
| `TIMESFM3_MAX_SERIES` | `64` | Series per request (targets + covariates) |
| `TIMESFM3_MAX_CONTEXT` | `16384` | Context steps per request |
| `PORT` | `8000` | Listen port |

### Cloudflare edge front end (free tier, no backend)

`cloudflare/` is one Worker that runs the whole public product for $0: the
landing page, the classical-model API computed in JavaScript at the edge,
the dashboard at `/app` where the TimesFM-3 starter model runs **in the
visitor's browser** through ONNX Runtime (a 21 MB static asset, cached after
the first visit; forecasts, backtests and anomaly scans never upload data),
and waitlist capture in Workers KV. Deploy it with the *Deploy to Cloudflare*
button in the README or by importing the repository in the Cloudflare
dashboard with root directory `cloudflare`; no API token or server is
needed. Point `API_ORIGIN` at a self-hosted `timesfm3 serve` and the same
Worker becomes a gateway for the full server-side API. Details in
[cloudflare/README.md](../cloudflare/README.md).

## Operating it

- **Throughput.** Inference runs in a thread pool; the classical models take
  milliseconds, the starter model tens of milliseconds per request on CPU.
  Run several uvicorn workers behind a load balancer for more; the service
  is stateless.
- **GPU.** Build the image with `--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124`
  and pass `--device cuda`.
- **Observability.** Scrape `/metrics`; `latency_ms` is also on every response.
- **Safety limits.** Cap series count and context length with the env vars
  above; long horizons decode in rolling chunks and cost linearly.
- **Licensing.** Code and bundled weights are Apache-2.0 (`LICENSE`,
  `NOTICE`). The Google-released TimesFM-3 weights are *not* included and
  are non-commercial; do not drop them into `TIMESFM3_MODEL_DIR` for
  production use.

The market context, positioning and pricing model are in
[docs/BUSINESS.md](BUSINESS.md).

## Python client

```python
from timesfm3.client import ForecastClient

client = ForecastClient("http://localhost:8000", api_key=None)
result = client.forecast([history], horizon=24, names=["sales"])
result.point        # (1, 24)
result.quantiles    # (1, 24, 9)
client.backtest([history], context=256, horizon=24)
```

## Limits, stated plainly

- The starter model is a small CPU-trained checkpoint, not the released
  334 M-parameter TimesFM-3. On the held-out benchmark above it has the
  lowest error but not a statistically significant edge over EWMA at 5%;
  it is a working default, not a substitute for a model trained on your
  domain — which is exactly what the backtest endpoint is there to measure.
- Classical-model bands are empirical, not conformal guarantees.
- The volatility endpoint uses daily squared returns as the variance proxy;
  with intraday data HAR would do better.
- Nothing here is investment advice.
