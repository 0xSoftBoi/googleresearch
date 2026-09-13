# Business case: TimesFM-3 Forecast Service

*Research date: 2026-09-13 (updated from 2026-09-01). Sources are linked
inline; everything without a link is our own measurement or judgement.*

## 1. The moment

Google published TimesFM-3 on 31 August 2026 and it tops the public
zero-shot forecasting benchmarks — but the pretrained weights ship under the
**TimesFM Non-Commercial License v1.0**, which forbids "any revenue-generating
activity" and use "in direct or indirect interactions with end users or
production systems" ([license](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE),
[The New Stack: "You can't use it at work (yet)"](https://thenewstack.io/google-timesfm-3-multivariate-forecasting/)).
This is a *regression* from TimesFM-2.5, whose weights were Apache-2.0 —
Google tightened the license specifically for its best model
([datanorth.ai](https://datanorth.ai/news/google-releases-timesfm-3)). The
code is Apache-2.0; only the weights are locked. Google's stated path to
commercial use is its own cloud: `AI.FORECAST`/`AI.EVALUATE` in BigQuery are
already GA for TimesFM-2.5, and Google says TimesFM-3 support is "coming in
the coming weeks" — as of this update (two weeks after launch) it still
has not shipped, so the non-commercial weights remain the only way to run
TimesFM-3 itself
([BigQuery docs](https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-ai-forecast),
[Google Cloud blog](https://cloud.google.com/blog/products/data-analytics/timesfm-models-in-bigquery-and-alloydb)).

That leaves an open lane: **the TimesFM-3 architecture, with weights you are
allowed to run in production, on infrastructure you control.** This repo is a
from-scratch Apache-2.0 implementation (see `NOTICE`) with its own trained
weights, so it can be sold, self-hosted and fine-tuned without a Google
license — which is exactly what the two market leaders charge for. We are
not the only ones who noticed the license gap: independent from-scratch
TimesFM reimplementations have appeared on GitHub within days of the release
(see §2b) — but none of them ship a server, metering, payments or a quant
toolkit, which is where this repo differs.

## 2. Market map

The field got more crowded and more open in the last two weeks than the
first version of this document assumed — two more well-funded labs
(Datadog, and the academic/industry team behind Kronos) shipped competitive
**open-weight** foundation models. That sharpens rather than closes the gap:
open weights are now normal; a *product* built around them (serving,
metering, payments, an honest backtest, a quant toolkit) is still rare.

| Player | What they sell | Price signal | License / lock-in | Gap we exploit |
|---|---|---|---|---|
| **Nixtla TimeGPT** | Hosted API, Azure, self-hosted; forecasting + anomaly detection + fine-tuning; **TimeGPT 2.1** (first multivariate TimeGPT, better with exogenous variables) is in **private preview only**, not GA | Enterprise from **$12,000/month** flat (reference price; actual deals are quote-only); 30-day trial; Series A **$16M** (Feb 2026, Energize Capital, True Ventures, GreatPoint Ventures) | Proprietary model, never open | Price floor excludes SMB/mid-market; the multivariate release mid-market actually wants is gated behind a preview waitlist, not shipping |
| **Google BigQuery `AI.FORECAST`** | Hosted TimesFM as a SQL function; TimesFM-2.5 is GA, TimesFM-3 still "coming weeks" two weeks after launch | BigQuery ML prediction rates; Vertex forecasting $0.20 → $0.02 per 1,000 predictions by volume | Data must live in BigQuery; no self-hosting; no covariate control at API level | Anyone not on GCP, anyone with data-residency limits, anyone who needs covariates or fine-tuning *today*, not "in coming weeks" |
| **Amazon Chronos-2** | Open weights (Apache-2.0), 120M params, encoder-only, zero-shot **univariate + multivariate + covariate-informed** forecasting; deployed via SageMaker JumpStart, AutoGluon-Cloud (3-line serverless/batch inference, June 2026) and Bedrock Marketplace; Amazon Forecast itself closed to new customers since **29 July 2024**, AWS points migrants at SageMaker Canvas, SageMaker+AutoGluon-TimeSeries, or Bedrock+Chronos | Instance-hours + Bedrock marketplace software fee | Open weights, AWS-shaped tooling | A model, not a product: no dashboard, backtest, metering, anomaly API, payments, or privacy layer out of the box |
| **Datadog Toto 2.0** | Open weights (Apache-2.0), five sizes **4M–2.5B params**, decoder-only, multivariate zero-shot; **currently #1 on BOOM, GIFT-Eval and the leakage-free TIME benchmark** — a stronger public benchmark position than this repo's starter model | Free weights; the product is Datadog's own observability platform (forecasting/anomaly features sold as part of a Datadog subscription, not standalone) | Open (Apache-2.0), but pretrained ~70% on Datadog's own private telemetry — biased toward infra/observability series | Datadog is not selling a general forecasting API — it's a $B-revenue SaaS company using the model as a feature inside its own product. Nobody can `pip install` a Datadog-hosted API with metering, x402 or fine-tuning-as-a-job; you get the weights and build the rest yourself, same gap as Chronos |
| **Kronos** (finance-native foundation model) | Open weights, up to 102M params, pretrained on 12B+ K-line (OHLCV) records from 45 exchanges; purpose-built tokenizer for candlestick sequences | Free weights (research release) | Open, no hosted product | Finance-only and price-native (candlesticks), not a general multivariate forecaster — no covariates, no arbitrary series, no volatility-targeting backtest harness. Validates that quant desks want a domain model, but it is a research artifact, not `timesfm3 finetune` + `/v1/volatility` + Diebold–Mariano-tested backtest in one CLI |
| **IBM Granite TTM** | Tiny (1–5M) Apache-2.0 models, watsonx | Free weights; watsonx pricing | Open | Same: weights, not a service |
| **Salesforce Moirai 2.0** | Research-only open weights; proprietary version internal | n/a | Research-only license | Not usable commercially |
| **Demand-forecasting SaaS** (Pecan, Ikigai, Anodot, inventory tools) | Vertical apps for supply chain / ecommerce | **$0.50–$5 per SKU/month**; Pecan $760–$1,400/month; market $250–$28,000/month, 4-week to 6-month setup | Application lock-in | Long setup, no API-first path, no model transparency |
| **Quant data vendors** (ORATS, ExtractAlpha, Databento) | Signals and data, not forecasting infrastructure | ORATS from ~$100/month; large funds spend $15–60M/yr on alt data | Data, not models | Funds want to run models on their own data in their own perimeter |
| **Micro-APIs** (ForecastAPI etc.) | Simple hosted forecast calls | 200 free/month, then $0.0016–$0.0033 per call | None | Establishes that self-serve usage pricing at fractions of a cent per forecast is accepted |

Sources: [Nixtla plans](https://www.nixtla.io/docs/introduction/timegpt_subscription_plans),
[Nixtla pricing (SoftwareAdvice)](https://www.softwareadvice.com/product/527675-TimeGPT/),
[Nixtla Series A](https://finance.yahoo.com/news/nixtla-raises-16-million-series-130000359.html),
[TimeGPT 2.1 announcement](https://www.nixtla.io/blog/timegpt-2-1-announcement),
[TimeGPT 2.1 waitlist](https://www.nixtla.io/timegpt-2-waitlist),
[Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing),
[Amazon Forecast closure](https://www.signisys.com/blog/amazon-forecast-the-complete-guide-to-aws-time-series-forecasting/),
[Amazon SageMaker Canvas migration](https://aws.amazon.com/blogs/machine-learning/transition-your-amazon-forecast-usage-to-amazon-sagemaker-canvas/),
[Chronos-2 announcement](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting),
[Chronos-2 on Hugging Face](https://huggingface.co/amazon/chronos-2),
[Chronos-2 + AutoGluon-Cloud](https://aws.amazon.com/blogs/machine-learning/how-decathlon-runs-demand-forecasting-at-scale-with-chronos-2/),
[Chronos license](https://github.com/amazon-science/chronos-forecasting),
[Datadog Toto/BOOM launch](https://www.datadoghq.com/blog/datadog-time-series-foundation-model/),
[Toto 2.0](https://www.datadoghq.com/blog/ai/toto-2/),
[Toto 2.0 on Hugging Face](https://huggingface.co/Datadog/Toto-2.0-2.5B),
[Kronos paper](https://arxiv.org/abs/2508.02739),
[Kronos context/hedge fund framing](https://blog.pebblous.ai/report/kronos-financial-foundation-model/en/),
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
those four capabilities; they are table stakes, not differentiators. What
*no* competitor above bundles, as of this update: a machine-payable API
(x402), an unlinkable prepaid tier (Privacy Pass), or a quant-specific
backtest harness with multiple-testing-corrected significance tests. Those
three remain differentiators, not table stakes — for now.

### 2b. Adjacent open-source TimesFM projects (evidence of demand, not competition)

Within days of the TimesFM-3 license change, independent developers started
publishing their own from-scratch reimplementations — proof the "give me
weights I can actually run" demand is real, and that we are early but not
alone in spotting it:

- [**FareedKhan-dev/timesfm-from-scratch**](https://github.com/FareedKhan-dev/timesfm-from-scratch):
  MIT-licensed, ships a 70M checkpoint plus 17M/203M configs, full
  pretraining pipeline (synthetic generator, RevIN, masked objective). It is
  a research/education codebase with inference and a training CLI — **no
  server, no API, no billing, no dashboard, no quant tooling.**
- A pure-MLX port of TimesFM-3 for Apple Silicon (numerically matches the
  reference to ~1e-6) and a Node.js/ONNX Runtime port of TimesFM-2.5 have
  also appeared, both libraries, not services.

None of these compete with the product; they validate the thesis and widen
the field of "TimesFM you can legally run" without touching the part we
actually sell: the service layer.

## 3. Positioning

> **Production forecasting you can audit.** The TimesFM-3 architecture with
> commercially usable weights, self-hosted in one container, with a backtest
> that tells you — with real statistics — whether the model beats a random
> walk on your data before you pay for it.

Four claims, each backed by something in the repo (a fourth was added this
update because Toto's benchmark position means "our weights are good" can no
longer carry the pitch alone):

1. **Legal to run in production.** Apache-2.0 code *and* weights (`NOTICE`);
   Google's are not. Chronos, Toto and Granite are, but ship as models, not
   services.
2. **Self-hosted, one command, CPU-first.** `docker compose up` — no BigQuery,
   no $12k/month. Data never leaves the customer's perimeter (the same
   argument Nixtla makes for its self-hosted tier, at its enterprise price,
   and the same argument Datadog *can't* make since Toto's edge is trained on
   Datadog's own telemetry and its distribution is a Datadog subscription).
3. **Honest by construction.** `/v1/backtest` reports Holm-corrected
   Diebold–Mariano verdicts; our own model card publishes a "no difference"
   result rather than hiding it. In a market where every vendor claims to
   "beat everyone" — and where Toto and Chronos-2 both currently *do* lead
   the public benchmarks — a verifiable benchmark on *the customer's* data,
   not a leaderboard, is the sales tool. We do not need to win GIFT-Eval; we
   need the customer's backtest to come back green.
4. **The only one that also sells the plumbing.** Metering, fine-tuning as a
   job, machine-payable pricing (x402), and an unlinkable credit tier
   (Privacy Pass) ship today. Every named competitor above sells either a
   model (Chronos, Toto, Granite, Kronos) or a walled hosted API (Nixtla,
   BigQuery) — none sell open, self-hostable infrastructure with billing
   built in.

## 4. Who buys, and why now

| Segment | Trigger | What they run | Willingness to pay |
|---|---|---|---|
| **Mid-market ops / data teams** (retail, energy, logistics, SaaS metrics) — 50–2,000 series | Amazon Forecast shut to new customers since 2024; TimeGPT's multivariate release is preview-only; TimeGPT enterprise too expensive; BigQuery not their stack | Forecast + anomalies on hourly/daily telemetry, demand, traffic | $300–$3,000/month (below Pecan's entry, far below Nixtla's) |
| **Platform / observability teams** embedding forecasts in their own product | Need an API with quotas and a license they can ship under, and are not already a Datadog customer (or want forecasting decoupled from an observability contract) | Metered `/v1/forecast`, `/v1/anomalies` behind their own keys | Per-point usage; OEM licensing |
| **Quant / risk desks** (small-to-mid funds, prop shops, treasury) | 95% of hedge fund managers now use generative AI in some form (up from 86% in 2023) and 58% expect to expand GenAI use in the *investment process* specifically; 60% of institutional allocators say meaningful AI investment makes them more likely to invest — funds that can show a working, testable pipeline have a fundraising argument, not just a research one ([AIMA 2025](https://www.aima.org/article/press-release-front-office-gen-ai-adoption-shifts-from-if-to-when-for-leading-fund-managers-aima-research-finds.html)) | `/v1/volatility`, `timesfm3 finetune` on their panel, HAR-RV and TSMOM strategy references, backtester | $2,000–$10,000/month seat-less licenses; they already spend orders of magnitude more on data |
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
| **Prepaid credit pool** | $0.004 per token, singly or in batches of 10/25/100 | Privacy Pass tokens (RFC 9578): the privacy tier for quant and enterprise buyers whose query pattern is the secret; sold via x402 or plans | `timesfm3/privacypass.py`, `timesfm3/serving/privacypass.py`, `cloudflare/src/privacy-pass.js` |
| **Pay-per-call (x402)** | $0.005 forecast · $0.005 volatility · $0.01 anomalies · $0.02 backtest, in USDC | No account, no invoice: the HTTP 402 flow AI agents already speak; settles on Base in seconds | `timesfm3/serving/x402.py`, `@x402/hono` in `cloudflare/src/index.js` |
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

### The agent channel: x402 — real, but still small, and that changes the plan

x402 (Coinbase's HTTP-402 payment protocol) is more mature infrastructure
than it looked two weeks ago, and less proven as a revenue channel. As of
this update: cumulative x402 volume across chains has passed **$50M** and
**165M transactions** from roughly **69,000 active agents**, and Solana
alone logged 35M x402 transactions by March 2026
([Chainalysis](https://www.chainalysis.com/blog/x402-agentic-payments-adoption/),
[Presenc.ai tracker](https://presenc.ai/research/x402-protocol-adoption-tracker-2026)).
But **genuine commercial daily volume is roughly $28,000/day, and about half
of that is estimated to be testing traffic rather than real purchases** —
the ecosystem is valued in the billions on infrastructure and speculative
activity, not on proven merchant revenue yet
([CoinDesk](https://www.coindesk.com/markets/2026/03/11/coinbase-backed-ai-payments-protocol-wants-to-fix-micropayment-but-demand-is-just-not-there-yet)).
Average transaction size is $0.20–$0.30 — squarely in the sub-cent-to-dime
range the protocol targets — which matches our per-call pricing well, but
confirms the *volume* isn't there yet at the ecosystem level. As of
1 January 2026 the Coinbase facilitator also started charging: first 1,000
settlements/month free, **$0.001 per settlement after that**
([Coinbase](https://x.com/CoinbaseDev/status/1995564027951665551)) — a real
but small tax on the $0.005–$0.02 prices we charge, and cheap enough that
alternative facilitators (Heurist, and others) are competing to stay free.

**Revision to the plan:** treat x402 as a distribution and positioning bet,
not a modeled revenue line for the first 90 days. It costs us nothing extra
to support (the same `/v1/pricing` 402 response works whether zero or a
thousand agents call it), it is real infrastructure with real facilitators
and a live fee schedule (not vaporware), and every x402-native agent
marketplace or "bazaar" listing is a distribution channel that costs us
nothing to be listed in. But the financial plan in §5's table should not
assume x402 volume; it should assume the metered plans carry revenue and
x402 is optionality. Separately, Google's **AP2** (Agent Payments Protocol)
is emerging as a complementary standard for *authorizing* agent purchases
via signed mandates, while x402 (and now an A2A x402 extension co-built with
Coinbase) handles settlement
([AP2 vs x402](https://ap2-protocol.org/topics/ap2-and-x402/)); worth
watching but not a blocker — the paywall we ship already speaks the
settlement half.

### Privacy as a feature, not a footnote

Every competitor's API logs who asked what. For the quant segment that is a
reason not to use a hosted forecaster at all. This product sells a credit
pool whose tokens are standard Privacy Pass (the operator provably cannot
link a call to a buyer) and documents funding those credits from a Privacy
Pools withdrawal, the compliant on-chain privacy protocol from the Ethereum
research community. It is a differentiator no incumbent above has any reason
to build — Nixtla and Google want to know who is calling for billing and
retention purposes, Datadog and Amazon do not sell a standalone forecasting
API at all — and it costs nothing to run since Privacy Pass is IETF-standard
infrastructure Cloudflare and Apple already operate at scale.

## 6. Go-to-market (first 90 days)

1. **Ride the license news.** Blog: "TimesFM-3 is non-commercial. Here is an
   Apache-2.0 TimesFM-3 service you can run today" — with the honest ETTh2
   table. Post to HN / r/MachineLearning / the TimesFM GitHub discussions the
   week the weights are trending. Ship the `docker compose up` demo.
2. **Amazon Forecast refugees.** Search-intent content: "Amazon Forecast
   alternative self-hosted" — AWS explicitly points those users at
   SageMaker Canvas, SageMaker+AutoGluon, or Bedrock+Chronos, none of which
   is a drop-in hosted API with metering; we are the packaged version.
3. **Design partners: 3 mid-market ops teams + 1 quant shop.** Offer free
   Team tier for a case study using `/v1/backtest` on their data; publish the
   verdict either way (the honesty is the brand).
4. **Bigger bundled model.** Train the `base` config (334M) on a GPU for a
   week on public + licensed corpora and publish its card against the public
   benchmarks Toto and Chronos-2 already lead — the goal is "competitive and
   auditable," not first place, because first place is a moving target we
   cannot win against a 2.5B-parameter model trained on Datadog's private
   telemetry with their infrastructure budget.

## 7. Moat and risks

- **Risk: Google relaxes the license** (or ships TimesFM-3 on Vertex/BigQuery
  with covariates — still not live two weeks after launch, per §1).
  Mitigation: positioning rests on *self-hosted, auditable, fine-tunable*,
  not on the license alone; BigQuery still requires your data in BigQuery.
- **Risk: open-weight foundation models are now commoditizing faster than
  expected.** Two weeks ago the open-weight field was Chronos and Granite;
  it now also includes Datadog's Toto 2.0 (currently #1 on three public
  benchmarks, up to 2.5B parameters, Apache-2.0) and Kronos (finance-native,
  open, backed by real hedge-fund-adoption momentum). Mitigation: none of
  them sell a service — Datadog monetizes Toto through its own SaaS
  subscription, not a standalone API; Kronos is a research release with no
  hosting, metering or billing at all. We stopped asserting this and shipped
  it: `timesfm3/serving/adapters.py` registers Amazon's own
  `chronos-forecasting` pipeline through the same `.forecast()` contract
  TimesFM-3 checkpoints use, `TIMESFM3_CHRONOS_MODELS` loads one alongside
  the starter model with no code changes, and a test runs a real Chronos-Bolt
  forecast end to end (skipped, not faked, when the optional dependency is
  absent). That is one adapter of a few dozen lines, not a fork of the
  serving stack — a Toto or Kronos adapter is the same shape of work, not
  done yet. It turns "they have a better model" into "we also serve their
  model, plus the metering, backtest, payments and privacy layer they
  don't sell."
- **Risk: x402 revenue does not materialize at ecosystem scale.** As of
  this update, real commercial x402 volume is ~$28k/day industry-wide and
  roughly half of that may be test traffic. Mitigation: §5 revises the plan
  to treat x402 as free optionality, not a forecast line; the metered plans
  and self-hosted enterprise tier carry the financial case.
- **Risk: Chronos-2 + AutoGluon "good enough".** Mitigation: they sell
  weights; we sell the service layer (metering, anomalies, backtest,
  dashboard, fine-tune-and-prove-it, payments, privacy). Also: a Chronos
  adapter already exists in this repo (see above) — a customer who prefers
  Chronos-2's accuracy on their series still needs everything else here to
  put it into production.
- **Risk: starter model is small and no longer leads any public benchmark.**
  True, and the model card says so. The backtest endpoint turns this into a
  sales motion ("run it on your data; if `no difference`, use EWMA for free
  — we told you first") rather than a liability. The `base` config is the
  roadmap item that closes the gap, and it does not need to beat Toto's
  2.5B-parameter model — it needs to beat the customer's current baseline,
  provably, on their data.
- **Moat that compounds:** every fine-tuning job run with the product
  produces an evaluated checkpoint plus a backtest record; with customer
  consent, that is a domain-adaptation dataset nobody else has. A second,
  newer moat: this is the only product in the market map that has shipped a
  machine-payable, self-hostable paywall (x402) *and* an unlinkable prepaid
  tier (Privacy Pass) end to end, interoperable between a self-hosted
  service and a $0 Cloudflare edge deployment — replicating that requires a
  competitor to redesign billing, not retrain a model.

## 8. What changed in the product because of this research

| Finding | Change |
|---|---|
| Every competitor bundles anomaly detection | `POST /v1/anomalies`, `timesfm3 anomalies` — walk-forward scoring against the model's own predictive band |
| Fine-tuning is the paid feature at Nixtla and the source of gains in the finance literature | `timesfm3 finetune data.csv` — fine-tunes the starter, validates on the held-out tail, and prints the same DM-tested backtest the API serves, base vs fine-tuned vs classical |
| Usage-based pricing needs metering | Per-key plans and monthly quotas, `X-Usage-Points`/`X-Usage-Remaining` headers, `/v1/usage`, 429 on exhaustion, persistent counters |
| Google's weights are non-commercial | `LICENSE` (Apache-2.0) and `NOTICE` stating the bundled weights are self-trained and commercially usable |
| AI agents cannot sign up for API keys | `POST`/`GET` 402 responses with x402 pricing on every priced endpoint, in both the Python service and the Cloudflare Worker |
| Every hosted competitor logs who asked what | Privacy Pass credit pool (RFC 9576/9577/9578): tokens the operator cannot link to the purchase or to each other, interoperable between the service and the edge |
| Open-weight competition (Toto, Chronos-2, Kronos) now leads public benchmarks | Positioning shifted from "best model" to "best-audited, best-plumbed service" (§3, claim 4) |
| x402 ecosystem volume is real but genuine commercial spend is still small | Revenue plan (§5) stopped counting on x402 volume; it is now framed explicitly as free distribution/optionality, not a forecast line |
| The registry's "serves any checkpoint" claim only held for this repo's own `.pt` format | `timesfm3/serving/adapters.py`: `ChronosAdapter` runs Amazon's real `chronos-forecasting` pipeline through the registry's forecaster contract; `TIMESFM3_CHRONOS_MODELS` loads it with no code changes; `ModelEntry` dispatch generalized from "kind == timesfm3" to "has a forecaster" so a foreign model reports its real kind (`GET /v1/models`) instead of masquerading as TimesFM-3; `tests/test_adapters.py` runs a real forecast, not a mock |
