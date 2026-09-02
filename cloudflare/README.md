# Cloudflare edge front end

One Worker, no build step: the marketing site, the API gateway, the
dashboard, and lead capture for the TimesFM-3 Forecast Service.

| Path | What happens |
|---|---|
| `/` | Landing page from `public/` (Workers static assets) |
| `/app` | The product dashboard, proxied from the upstream service |
| `/v1/*`, `/healthz`, `/docs`, `/openapi.json` | Proxied to `API_ORIGIN`; the upstream `API_KEY` is attached server-side unless the caller brings their own; 30 req/min per IP via KV; `/v1/models`, `/v1/sample`, `/healthz` cached 60 s at the edge |
| `POST /api/waitlist` | Stores `{email, company, plan, use_case}` in KV (honeypot field `website`), optional webhook |
| `GET /api/leads` | Lists leads; needs `Authorization: Bearer $ADMIN_TOKEN` |
| `GET /api/edge` | Edge configuration and colo, for smoke tests |
| `/metrics` | Never proxied |

## Run locally

```bash
timesfm3 serve --port 8000 &                                   # the upstream
cd cloudflare && npm install
npx wrangler dev --var API_ORIGIN:http://localhost:8000 --var ADMIN_TOKEN:dev
open http://localhost:8787
```

## Deploy

Needs a Cloudflare API token with *Workers Scripts: Edit* and *Workers KV
Storage: Edit* (create one at dash.cloudflare.com → My Profile → API Tokens).

```bash
export CLOUDFLARE_API_TOKEN=...   CLOUDFLARE_ACCOUNT_ID=...
cd cloudflare
npx wrangler deploy                                  # -> https://timesfm3-edge.<account>.workers.dev
npx wrangler secret put API_KEY                      # the upstream key the edge uses for anonymous demo traffic
npx wrangler secret put ADMIN_TOKEN                  # for /api/leads
npx wrangler secret put LEADS_WEBHOOK                # optional Slack/Discord webhook
```

Set `API_ORIGIN` in `wrangler.jsonc` (or `npx wrangler deploy --var API_ORIGIN:https://...`)
to wherever the Docker image runs — a Fly/Railway/EC2 box, or a Cloudflare
Tunnel in front of an on-prem host. Give the edge a *quota-limited* upstream
key (`TIMESFM3_API_KEYS="edge-demo:<key>:2000000"`) so public demo traffic
can never exhaust the service; paying customers send their own key and the
edge passes it through untouched.

The GitHub Actions workflow `.github/workflows/cloudflare.yml` deploys on
every push to `main` when the `CLOUDFLARE_API_TOKEN` and
`CLOUDFLARE_ACCOUNT_ID` repository secrets exist.

The KV namespace `timesfm3-edge` (`1df8ae6bdb5a41eabdd00beb32118817`) was
created in the connected account for rate-limit counters and leads; keys are
`rl:<ip>:<minute>` (120 s TTL), `lead:<created>:<hash>` and `lead-email:<email>`.
