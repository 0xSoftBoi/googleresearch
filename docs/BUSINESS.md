# Business case: TimesFM-3 Forecast Service

*Research date: 2026-09-01. Sources are linked inline; everything without a
link is our own measurement or judgement.*

## 1. The moment

Google published TimesFM-3 on 31 August 2026 and it tops the public
zero-shot forecasting benchmarks — but the pretrained weights ship under the
**TimesFM Non-Commercial License v1.0**, which forbids "any revenue-generating
activity" and use "in direct or indirect interactions with end users or
production systems" ([license](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE),
[The New Stack: "You can't use it at work (yet)"](https://thenewstack.io/google-timesfm-3-multivariate-forecasting/)).
The code is Apache-2.0; only the weights are locked. Google's stated path to
commercial use is its own cloud: TimesFM inside BigQuery `AI.FORECAST` and
AlloyDB, billed at BigQuery ML rates
([docs](https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-ai-forecast),
[announcement](https://aiweekly.co/alerts/google-timesfm-3-tops-forecast-benchmarks-non-commercial)).

That leaves an open lane: **the TimesFM-3 architecture, with weights you are
allowed to run in production, on infrastructure you control.** This repo is a
from-scratch Apache-2.0 implementation (see `NOTICE`) with its own trained
weights, so it can be sold, self-hosted and fine-tuned without a Google
license — which is exactly what the two market leaders charge for.

## 2. Market map

| Player | What they sell | Price signal | License / lock-in | Gap we exploit |
|---|---|---|---|---|
| **Nixtla TimeGPT** | Hosted API, Azure, self-hosted; forecasting + anomaly detection + fine-tuning; TimeGPT 2.1 | Enterprise from **$12,000/month** flat; 30-day trial; Series A **$16M** (Feb 2026) | Proprietary model, never open | Price floor excludes SMB/mid-market; model is a black box you cannot inspect or retrain |
| **Google BigQuery `AI.FORECAST`** | Hosted TimesFM 2.0 (500M) as a SQL function; TimesFM-3 "in coming weeks" | BigQuery ML prediction rates; Vertex forecasting $0.20 → $0.02 per 1,000 predictions by volume | Data must live in BigQuery; no self-hosting; no covariate control at API level | Anyone not on GCP, anyone with data-residency limits, anyone who needs covariates or fine-tuning |
| **Amazon Chronos-2** | Open weights (Apache-2.0) via SageMaker JumpStart / Bedrock Marketplace / AutoGluon; Amazon Forecast closed to new customers (2024) | Instance-hours + Bedrock marketplace software fee | Open weights, AWS-shaped tooling | A model, not a product: no dashboard, backtest, metering, anomaly API out of the box |
| **IBM Granite TTM** | Tiny (1–5M) Apache-2.0 models, watsonx | Free weights; watsonx pricing | Open | Same: weights, not a service |
| **Salesforce Moirai 2.0** | Research-only open weights; proprietary version internal | n/a | Research-only license | Not usable commercially |
| **Demand-forecasting SaaS** (Pecan, Ikigai, Anodot, inventory tools) | Vertical apps for supply chain / ecommerce | **$0.50–$5 per SKU/month**; Pecan $760–$1,400/month; market $250–$28,000/month, 4-week to 6-month setup | Application lock-in | Long setup, no API-first path, no model transparency |
| **Quant data vendors** (ORATS, ExtractAlpha, Databento) | Signals and data, not forecasting infrastructure | ORATS from ~$100/month; large funds spend $15–60M/yr on alt data | Data, not models | Funds want to run models on their own data in their own perimeter |
| **Micro-APIs** (ForecastAPI etc.) | Simple hosted forecast calls | 200 free/month, then $0.0016–$0.0033 per call | None | Establishes that self-serve usage pricing at fractions of a cent per forecast is accepted |

Sources: [Nixtla plans](https://www.nixtla.io/docs/introduction/timegpt_subscription_plans),
[Nixtla pricing (SoftwareAdvice)](https://www.softwareadvice.com/product/527675-TimeGPT/),
[Nixtla Series A](https://newmarketpitch.com/blogs/news/foundation-model-list-deals),
[TimeGPT features](https://www.nixtla.io/docs/introduction/about_timegpt),
[Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing),
[Amazon Forecast status](https://www.signisys.com/blog/amazon-forecast-the-complete-guide-to-aws-time-series-forecasting/),
[Chronos on SageMaker](https://aws.amazon.com/blogs/machine-learning/fast-and-accurate-zero-shot-forecasting-with-chronos-bolt-and-autogluon/),
[Chronos license](https://github.com/amazon-science/chronos-forecasting),
[Granite TTM](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2/blob/main/README.md),
[Moirai 2.0](https://huggingface.co/Salesforce/moirai-2.0-R-small),
[demand-forecasting pricing](https://setupbots.com/blog/ai-demand-forecasting-pricing-guide),
[Pecan pricing](https://www.pecan.ai/pricing/),
[market cost range](https://shivlab.com/blog/demand-forecasting-software-cost-setup-use-cases/),
[ORATS / alt-data spend](https://permutable.ai/best-financial-market-data-providers-for-hedge-funds/),
[ForecastAPI](https://forecastapi.com/).

**What every serious competitor bundles** (and this product now has):
forecasting with uncertainty → anomaly detection → fine-tuning on customer
data → self-hosting → metered API keys. Nixtla's product page lists exactly
those four capabilities; they are table stakes, not differentiators.

## 3. Positioning

> **Production forecasting you can audit.** The TimesFM-3 architecture with
> commercially usable weights, self-hosted in one container, with a backtest
> that tells you — with real statistics — whether the model beats a random
> walk on your data before you pay for it.

Three claims, each backed by something in the repo:

1. **Legal to run in production.** Apache-2.0 code *and* weights (`NOTICE`);
   Google's are not. Chronos/Granite are, but ship as models, not services.
2. **Self-hosted, one command, CPU-first.** `docker compose up` — no BigQuery,
   no $12k/month. Data never leaves the customer's perimeter (the same
   argument Nixtla makes for its self-hosted tier, at its enterprise price).
3. **Honest by construction.** `/v1/backtest` reports Holm-corrected
   Diebold–Mariano verdicts; our own model card publishes a "no difference"
   result rather than hiding it. In a market where every vendor claims to
   "beat everyone", a verifiable benchmark on *the customer's* data is the
   sales tool.

## 4. Who buys, and why now

| Segment | Trigger | What they run | Willingness to pay |
|---|---|---|---|
| **Mid-market ops / data teams** (retail, energy, logistics, SaaS metrics) — 50–2,000 series | Amazon Forecast shut to new customers; TimeGPT too expensive; BigQuery not their stack | Forecast + anomalies on hourly/daily telemetry, demand, traffic | $300–$3,000/month (below Pecan's entry, far below Nixtla's) |
| **Platform / observability teams** embedding forecasts in their own product | Need an API with quotas and a license they can ship under | Metered `/v1/forecast`, `/v1/anomalies` behind their own keys | Per-point usage; OEM licensing |
| **Quant / risk desks** (small-to-mid funds, prop shops, treasury) | Want vol forecasts + a harness to test models on their own returns inside their perimeter | `/v1/volatility`, `timesfm3 finetune` on their panel, backtester | $2,000–$10,000/month seat-less licenses; they already spend orders of magnitude more on data |
| **Researchers / students** | Google's weights are non-commercial anyway; they want a trainable reference implementation | Library, notebooks | Free (funnel) |

## 5. Business model

**Open core + usage.** The library, CLI, server and starter model stay
Apache-2.0 (that *is* the distribution strategy: pip and Docker, no sales
call). Revenue comes from three layers built on the metering that ships in
this release (`X-Usage-Points`, `/v1/usage`, per-key monthly quotas):

| Tier | Price | Included | Mechanism in repo |
|---|---|---|---|
| **Community** | $0 | Self-host everything; starter model; unlimited local use | Apache-2.0 |
| **Cloud Starter** | $49/month | Hosted API, 2M forecast points/month, 3 keys | `TIMESFM3_API_KEYS_FILE` quotas |
| **Cloud Team** | $299/month | 20M points/month, anomaly + volatility endpoints, fine-tuning jobs, 10 keys | same, plus `timesfm3 finetune` as a job |
| **Overage** | $0.02 per 1,000 points | Falls between Vertex's $0.02–$0.20/1,000 predictions and ForecastAPI's ~$2–3/1,000 calls | metering headers |
| **Prepaid credit pool** | $0.004 per credit in batches of 10/25/100 | Blind-signed, unlinkable tokens: the privacy tier for quant and enterprise buyers whose query pattern is the secret; sold via x402 or plans | `timesfm3/credits.py`, `timesfm3/serving/credits.py` |
| **Pay-per-call (x402)** | $0.005 forecast · $0.005 volatility · $0.01 anomalies · $0.02 backtest, in USDC | No account, no invoice: the HTTP 402 flow AI agents already speak; settles on Base in seconds | `timesfm3/serving/x402.py`, `cloudflare/src/x402.js` |
| **Enterprise / self-hosted** | $2,000–$8,000/month | Private image, SSO, support SLA, larger checkpoints (`base` config trained on GPU), fine-tuning on their corpus, commercial warranty on weights | Docker + `TIMESFM3_MODEL_DIR` + NOTICE |

A **forecast point** (one series × one horizon step) is the billable unit
because it scales with compute and maps onto what customers already
understand from Vertex ("predictions"). One hourly series forecast 48 h
ahead every hour is ~35k points/month; 500 SKUs forecast daily 30 days out
is ~450k. The Starter tier covers a real deployment; Team covers a fleet.

**Unit economics (measured on this box, 4 vCPU, no GPU).** The starter
model serves ~55 ms per request for 7 series × 48 steps ≈ 6,000 points/s
single-threaded. At cloud-CPU prices (~$0.04/vCPU-hour) that is roughly
**$0.002 per 1,000 points** of compute — a 10× gross margin at the overage
price, before batching. Classical models are ~100× cheaper still.

### The agent channel: x402

The buyers who cannot sign up are AI agents. x402 (Coinbase's HTTP-402
payment protocol, adopted by Cloudflare's Agents SDK and MCP servers in
September 2025) lets any HTTP client pay per request in USDC with a signed
transfer authorization; a facilitator settles on Base for well under a cent.
This product now answers `402` with a price on every priced endpoint, so an
agent that discovers the API can forecast, backtest or scan for anomalies
and pay without a human, a card or an API key. Economics: the Coinbase
facilitator's first 1,000 settlements a month are free and $0.001 after, so
the $0.005 forecast price nets ~$0.004 at scale against ~$0.000002 of
compute. The metered plans stay the home for volume; x402 is the zero-friction
top of funnel that also turns every agent marketplace and x402 directory
("bazaar" listings) into a distribution channel.

### Privacy as a feature, not a footnote

Every competitor's API logs who asked what. For the quant segment that is a
reason not to use a hosted forecaster at all. This product sells a credit
pool whose tokens are blind-signed (the operator provably cannot link a call
to a buyer) and documents funding those credits from a Privacy Pools
withdrawal, the compliant on-chain privacy protocol from the Ethereum
research community. It is a differentiator no incumbent can copy without
redesigning billing, and it costs nothing to run.

## 6. Go-to-market (first 90 days)

1. **Ride the license news.** Blog: "TimesFM-3 is non-commercial. Here is an
   Apache-2.0 TimesFM-3 service you can run today" — with the honest ETTh2
   table. Post to HN / r/MachineLearning / the TimesFM GitHub discussions the
   week the weights are trending. Ship the `docker compose up` demo.
2. **Amazon Forecast refugees.** Search-intent content: "Amazon Forecast
   alternative self-hosted" — AWS explicitly points those users at Chronos
   plus DIY; we are the packaged version.
3. **Design partners: 3 mid-market ops teams + 1 quant shop.** Offer free
   Team tier for a case study using `/v1/backtest` on their data; publish the
   verdict either way (the honesty is the brand).
4. **Bigger bundled model.** Train the `base` config (334M) on a GPU for a
   week on public + licensed corpora and publish its card; that is the
   Enterprise up-sell and the answer to "how close to Google's numbers".

## 7. Moat and risks

- **Risk: Google relaxes the license** (or ships TimesFM-3 on Vertex with
  covariates). Mitigation: positioning rests on *self-hosted, auditable,
  fine-tunable*, not on the license alone; BigQuery still requires your data
  in BigQuery.
- **Risk: Chronos-2 + AutoGluon "good enough".** Mitigation: they sell
  weights; we sell the service layer (metering, anomalies, backtest,
  dashboard, fine-tune-and-prove-it). Also: the registry serves *any*
  checkpoint — adding a Chronos entry is a wrapper, not a rewrite.
- **Risk: starter model is small.** True, and the model card says so. The
  backtest endpoint turns this into a sales motion ("run it on your data;
  if `no difference`, use EWMA for free — we told you first") rather than a
  liability. The `base` config is the roadmap item that closes the gap.
- **Moat that compounds:** every fine-tuning job run with the product
  produces an evaluated checkpoint plus a backtest record; with customer
  consent, that is a domain-adaptation dataset nobody else has.

## 8. What changed in the product because of this research

| Finding | Change |
|---|---|
| Every competitor bundles anomaly detection | `POST /v1/anomalies`, `timesfm3 anomalies` — walk-forward scoring against the model's own predictive band |
| Fine-tuning is the paid feature at Nixtla and the source of gains in the finance literature | `timesfm3 finetune data.csv` — fine-tunes the starter, validates on the held-out tail, and prints the same DM-tested backtest the API serves, base vs fine-tuned vs classical |
| Usage-based pricing needs metering | Per-key plans and monthly quotas, `X-Usage-Points`/`X-Usage-Remaining` headers, `/v1/usage`, 429 on exhaustion, persistent counters |
| Google's weights are non-commercial | `LICENSE` (Apache-2.0) and `NOTICE` stating the bundled weights are self-trained and commercially usable |
