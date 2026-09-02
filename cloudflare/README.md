# Cloudflare front end — free tier, zero servers

One Worker, no build step, nothing to pay for. The classical forecasters run
**at the edge** in JavaScript inside the free plan's CPU budget; the TimesFM-3
model runs **in the visitor's browser** through ONNX Runtime (WebAssembly)
from a static asset; signups land in Workers KV. There is no backend to host.

| Path | Edge-native mode (default) |
|---|---|
| `/` | Landing page: live edge forecast, benchmark, pricing, waitlist |
| `/app` | Dashboard — TimesFM-3 (21 MB ONNX, cached after first load) + classical models, backtests and anomaly scans, all computed locally in the browser; data never leaves the page |
| `/docs` | API reference for this deployment |
| `GET /healthz`, `GET /v1/models`, `GET /v1/sample` | Served by the Worker |
| `POST /v1/forecast` | Classical models in JS: ≤32 series, ≤4096 context, ≤512 horizon, quantile bands, future timestamps |
| `POST /v1/backtest` | Classical models: ≤4 series, ≤10 windows (the dashboard runs any size) |
| `POST /v1/anomalies` | Classical models: ≤4 series |
| `POST /api/waitlist`, `GET /api/leads` | Lead capture in KV (honeypot, dedupe by email); leads need `Authorization: Bearer $ADMIN_TOKEN` |
| `/metrics` | Never exposed |

Set `API_ORIGIN` to a self-hosted `timesfm3 serve` and the same Worker
becomes a **gateway**: `/v1/*` is proxied with the upstream key attached
server-side, bring-your-own keys pass through, cheap GETs are edge-cached.

**Pay-per-call with x402.** Set `X402_PAY_TO` (your wallet) in
`wrangler.jsonc`; in gateway mode anonymous callers then pay per request in
USDC ($0.005 forecast, $0.01 anomalies, $0.02 backtest, $0.005 volatility)
while bring-your-own-key callers are metered upstream. `X402_NETWORK`
defaults to Base Sepolia (`eip155:84532`, free facilitator at x402.org);
use `eip155:8453` plus the `X402_FACILITATOR_AUTH` secret (Coinbase CDP) on
mainnet. `X402_PAYWALL_EDGE_NATIVE=1` also charges for the edge classical
API. `GET /v1/pricing` publishes the terms; `GET /api/edge` shows the state.

**Unlinkable prepaid credits at the edge** (`docs/PRIVACY.md`): with the
secret `CREDITS_PRIVATE_JWK` (make one with `python scripts/credits_keygen.py`)
and the D1 binding, the Worker blind-signs credits (RFC 9474 via
`@cloudflare/blindrsa-ts`), sells them in batches of 10/25/100 through x402,
and redeems `X-Credit` tokens against a D1 nullifier ledger — the same key
and token format as the Python service, so credits are interchangeable. In
gateway mode `X-Credit` requests pass the paywall and are validated upstream.
The D1 database `timesfm3-credits` (`10503e65-8f18-4803-890f-809025735489`)
exists in the connected account; apply `migrations/0001_credits.sql` with
`npx wrangler d1 migrations apply timesfm3-credits` (`--local` for dev; dev
secrets go in `.dev.vars`).

Per-IP rate limiting (30/min) uses the Workers rate-limit binding, which is
free and makes no KV writes — the free tier's 1,000 KV writes/day are kept
for leads.

## Deploy in one click (free)

**Option A — Deploy to Cloudflare button** (creates a copy of the repo under
your GitHub account, provisions bindings, sets up Workers Builds so every push
deploys):

[![Deploy to Cloudflare](https://deploy.workers.cloudflare.com/button)](https://deploy.workers.cloudflare.com/?url=https://github.com/0xSoftBoi/googleresearch/tree/main/cloudflare)

**Option B — import this repo** (keeps deploying from this repo):
Cloudflare dashboard → *Workers & Pages* → *Create* → *Import a repository* →
pick `0xSoftBoi/googleresearch` → root directory `cloudflare` → deploy command
`npx wrangler deploy` → *Deploy*. Then, under the Worker's *Settings →
Variables and Secrets*, add the secret `ADMIN_TOKEN` (for `/api/leads`) and
optionally `LEADS_WEBHOOK`. The KV namespace `timesfm3-edge`
(`1df8ae6bdb5a41eabdd00beb32118817`) already exists in the account; the
rate-limit binding needs no provisioning.

**Option C — CLI**: `cd cloudflare && npm install && npx wrangler deploy`
with `CLOUDFLARE_API_TOKEN` set (Workers Scripts: Edit, Workers KV Storage:
Edit); `.github/workflows/cloudflare.yml` does the same on push when the
`CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` secrets exist.

The Worker lands at `timesfm3-edge.<account>.workers.dev`; attach a custom
domain in the dashboard when you have one.

## Public URL with no account at all (Quick Tunnel)

```bash
make edge-dev                 # terminal 1: Worker on :8787 (edge-native, or gateway with API_ORIGIN)
make edge-tunnel              # terminal 2: prints https://<random>.trycloudflare.com
```

Cloudflare Quick Tunnels need no account, token or DNS: `cloudflared`
opens an outbound connection and Cloudflare hands out a public
`trycloudflare.com` hostname for as long as the process runs. It is the
fastest way to put the full stack (including a local `timesfm3 serve` with
the neural model, via `API_ORIGIN`) in front of someone today; the Worker
deploy below is the permanent, always-on version.

## Run locally

```bash
cd cloudflare && npm install
npx wrangler dev --port 8787 --var ADMIN_TOKEN:dev        # edge-native, no backend
# or as a gateway for a local service:
timesfm3 serve --port 8000 &  npx wrangler dev --port 8787 --var API_ORIGIN:http://localhost:8000
```

## How the model gets into the browser

`scripts/export_onnx.py timesfm3/assets/starter-small.pt cloudflare/public/models/starter-small.onnx`
exports the checkpoint (dynamic series/time axes, horizon length as a tensor
input, parity-checked against PyTorch to ~1e-6) plus a `.json` model card.
`public/js/timesfm3-onnx.js` reproduces the Python forecaster's padding,
rolling decode and quantile repair around the graph; `public/js/forecast.js`
is the JS port of the classical baselines, empirical bands, backtest and
anomaly scoring — `tests/test_js_parity.py` proves it reproduces the Python
numbers.

Free-tier limits that shaped this: 10 ms CPU per request (hence the backtest
caps and 200 bootstrap resamples at the edge), 25 MiB per static asset (the
21 MB ONNX fits; the 334M `base` config would need R2), 1,000 KV writes/day.
