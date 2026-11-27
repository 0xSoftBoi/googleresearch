# Business case: TimesFM-3 Forecast Service

*Research date: 2026-11-21 (re-audit of the 2026-09-01 and 2026-09-13
drafts). Every external claim below was read at its source on 2026-11-21 or
in the 2026-09-04 audit; every measured figure names the
[`docs/VERIFICATION.md`](VERIFICATION.md) section that reproduces it. Claims
from the 2026-09-13 draft that did not survive that check were corrected or
removed (ledger §10). Everything without a link or ledger reference is our
own judgement.*

## 1. The moment — and how it changed

Google published TimesFM-3 on 31 August 2026 and it tops the public
zero-shot forecasting benchmarks — but the pretrained weights ship under the
**TimesFM Non-Commercial License v1.0**, which forbids "any revenue-generating
activity" and use "in direct or indirect interactions with end users or
production systems" ([license](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE),
[The New Stack: "You can't use it at work (yet)"](https://thenewstack.io/google-timesfm-3-multivariate-forecasting/)).
Weights "up to version 2.5 remain Apache-2.0"; 3.0's are "restricted to
non-commercial, non-production use"
([google-research/timesfm](https://github.com/google-research/timesfm)).

**What changed since September:** the same README now says "commercial and
production use of TimesFM 3.0 is fully permitted through authorized Google
Cloud services, including BigQuery ML", and BigQuery's `AI.FORECAST`
reference lists "`TimesFM 2.5` and `TimesFM 3.0` (Preview)", with 2.5 the
default and multivariate forecasting requiring 3.0 under pre-GA terms
([AI.FORECAST reference](https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-ai-forecast),
read 2026-11-21). So "there is no commercial TimesFM-3" is no longer true.
What remains true: commercial TimesFM-3 exists **only inside Google Cloud,
with your data in BigQuery, in Preview**. Nothing lets you run it on your
own hardware, off GCP, or behind your own perimeter.

That is the lane, and it is narrower than the September draft said:
**production forecasting on infrastructure you control, with weights you are
allowed to run there.** This repo is a from-scratch Apache-2.0
implementation (see `NOTICE`) with its own trained weights, and — the more
important fact, measured in §2 — it can also serve Amazon's open Chronos
models, which beat our own starter on our own benchmark.

## 2. Market map

Open weights are now normal: Amazon (Chronos-2, Chronos-Bolt), Datadog
(Toto 2.0), IBM (Granite TTM) and the Kronos authors all publish models you
may run commercially. A *product* built around them — serving, metering,
payments, an honest backtest, a quant toolkit — is still rare. And the
benchmark that matters most for this document is our own: **on the model
card's ETTh2 panel, every open Amazon model we tried beats our bundled
starter by about 20% MAE** (ledger §9):

| Model (served through this repo's registry) | Params | Mean MAE | Ratio vs starter [95% CI] | Holm p | Verdict vs starter |
|---|---|---|---|---|---|
| Chronos-Bolt-small (Apache-2.0) | 47.7 M | 1.704 | 0.780 [0.735, 0.839] | < 0.001 | better |
| Chronos-Bolt-tiny (Apache-2.0) | 8.7 M | 1.709 | 0.783 [0.738, 0.845] | < 0.001 | better |
| Chronos-2 (Apache-2.0), univariate mode | 119.5 M | 1.770 | 0.811 [0.763, 0.872] | < 0.001 | better |
| **starter-small (this repo)** | 5.2 M | 2.183 | 1.000 | — | reference |
| EWMA (classical) | 0 | 2.309 | 1.057 [0.930, 1.228] | 0.33 | no difference |

7 series × 20 non-overlapping windows, context 256, horizon 24, HAC
Diebold–Mariano with Holm correction — the exact command that produced the
model card, plus `TIMESFM3_CHRONOS_MODELS`. The starter row reproduces the
model card to every digit. Caveat: ETT is a held-out ("not seen during
training") dataset for the original Chronos models, but the Chronos-Bolt
card does not itemize its corpus, so a zero-shot-clean comparison is
established for neither Bolt nor Chronos-2 (ledger §9).

| Player | What they sell | Price signal | License / lock-in | Gap we exploit |
|---|---|---|---|---|
| **Nixtla TimeGPT** | Hosted API, Azure, self-hosted; forecasting + anomaly detection + fine-tuning; **TimeGPT 2.1** (announced 2025-12-10: multivariate support, better exogenous-variable handling, on-prem / air-gapped deployment) launched as a **private preview with a waitlist**; no GA announcement on Nixtla's pages as of 2026-11-21 | Enterprise from **$12,000/month** flat (SoftwareAdvice; Nixtla lists no prices); 30-day trial; Series A **$16M** (Feb 2026, led by Energize Capital with True Ventures and GreatPoint Ventures) | Proprietary model, never open | Price floor excludes SMB/mid-market; model is a black box you cannot inspect or retrain |
| **Google BigQuery `AI.FORECAST`** | Hosted TimesFM as a SQL function: TimesFM 2.5 (default) and **TimesFM 3.0 (Preview)**; multivariate requires 3.0 | BigQuery ML prediction rates; Vertex forecasting $0.20 → $0.02 per 1,000 predictions by volume | Data must live in BigQuery; no self-hosting; Preview under pre-GA terms | Anyone not on GCP, anyone with data-residency limits, anyone who will not run production on a Preview feature, anyone who needs fine-tuning on their own corpus |
| **Amazon Chronos-2 / Chronos-Bolt** | Open weights (Apache-2.0). Chronos-2: "120M-parameter, encoder-only", zero-shot univariate, multivariate and covariate-informed; since 5 Jun 2026 deployable to AWS via AutoGluon-Cloud "in 3 lines of code"; also SageMaker JumpStart. Amazon Forecast itself closed to new customers in 2024, and AWS publishes a migration guide to SageMaker Canvas | Instance-hours on AWS; free to self-host | Open weights, AWS-shaped tooling | A model, not a product: no dashboard, backtest, metering, anomaly API, payments or privacy layer. **Also the strongest model we have measured on our own panel (table above)** — which is why this repo now serves it |
| **Datadog Toto 2.0** | Open weights (Apache-2.0), five checkpoints from 4M to 2.5B parameters, decoder-only, multivariate zero-shot. Datadog (May 2026): "takes the top spots on BOOM … GIFT-Eval … and TIME"; on GIFT-Eval its three largest sizes "rank first, second, and third among foundation models" by CRPS rank | Free weights; Datadog's commercial product is its observability platform (our reading: forecasting is a feature there, not a standalone API) | Open (Apache-2.0). Datadog: Toto 2.0 "does not see any public forecasting data during pretraining" — its corpus centres on Datadog's internal telemetry | No hosted general-purpose forecasting API with metering, payments or fine-tuning jobs; you get the weights and build the rest. No adapter in this repo yet |
| **Kronos** (finance-native foundation model) | Pretrained on "over 12 billion K-line records from 45 global exchanges" with a tokenizer for OHLCV sequences; its abstract claims price-forecasting RankIC "93% over the leading TSFM"; "pre-trained model is publicly available" | Free weights (research release) | Open, no hosted product | Price/candlestick-native rather than a general multivariate forecaster; a research artifact, not a served product with a significance-tested backtest. No adapter in this repo yet |
| **IBM Granite TTM** | Tiny (1–5M) Apache-2.0 models, watsonx | Free weights; watsonx pricing | Open | Same: weights, not a service |
| **Salesforce Moirai 2.0** | Research-only open weights; proprietary version internal | n/a | Research-only license | Not usable commercially |
| **Demand-forecasting SaaS** (Pecan, Ikigai, Anodot, inventory tools) | Vertical apps for supply chain / ecommerce | **$0.50–$5 per SKU/month** (not confirmed at the cited setupbots page, which renders no text; the same range appears on getmonetizely.com; ledger §6); Pecan's own page shows no prices — third-party listings (Capterra, GetApp) show $950–$1,750/month; market $250–$28,000/month, 4-week to 6-month setup | Application lock-in | Long setup, no API-first path, no model transparency |
| **Quant data vendors** (ORATS, ExtractAlpha, Databento) | Signals and data, not forecasting infrastructure | ORATS from $99/month; large multi-strategy funds average ~$5M/yr on alternative data (Hedgeweek) | Data, not models | Funds want to run models on their own data in their own perimeter |
| **Micro-APIs** (ForecastAPI etc.) | Simple hosted forecast calls | 200 free/month, then $0.0016–$0.0033 per call | None | Establishes that self-serve usage pricing at fractions of a cent per forecast is accepted |

Sources: [Nixtla plans](https://www.nixtla.io/docs/introduction/timegpt_subscription_plans),
[Nixtla pricing (SoftwareAdvice)](https://www.softwareadvice.com/product/527675-TimeGPT/),
[Nixtla Series A](https://www.accessnewswire.com/newsroom/en/computers-technology-and-internet/nixtla-raises-16-million-series-a-to-advance-time-series-intellig-1133806),
[TimeGPT 2.1 announcement](https://www.nixtla.io/blog/timegpt-2-1-announcement),
[TimeGPT 2.1 waitlist](https://www.nixtla.io/timegpt-2-waitlist),
[TimeGPT features](https://www.nixtla.io/docs/introduction/about_timegpt),
[Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing),
[BigQuery AI.FORECAST reference](https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-ai-forecast),
[Amazon Forecast closure](https://www.signisys.com/blog/amazon-forecast-the-complete-guide-to-aws-time-series-forecasting/),
[AWS Forecast → SageMaker Canvas migration guide](https://aws.amazon.com/blogs/machine-learning/transition-your-amazon-forecast-usage-to-amazon-sagemaker-canvas/),
[Chronos-2 model card](https://huggingface.co/amazon/chronos-2),
[chronos-forecasting README (license, AutoGluon-Cloud news)](https://github.com/amazon-science/chronos-forecasting),
[Chronos paper (ETT in held-out Benchmark II)](https://arxiv.org/abs/2403.07815),
[Toto 2.0](https://www.datadoghq.com/blog/ai/toto-2/),
[Toto 2.0 on Hugging Face](https://huggingface.co/Datadog/Toto-2.0-2.5B),
[Toto 1.0 / BOOM](https://www.datadoghq.com/blog/ai/toto-boom-unleashed/),
[Kronos paper](https://arxiv.org/abs/2508.02739),
[Granite TTM](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2/blob/main/README.md),
[Moirai 2.0](https://huggingface.co/Salesforce/moirai-2.0-R-small),
[demand-forecasting pricing](https://setupbots.com/blog/ai-demand-forecasting-pricing-guide),
[Pecan pricing](https://www.pecan.ai/pricing/) (no list prices; see [Capterra](https://www.capterra.com/p/199700/Pecan/)),
[market cost range](https://shivlab.com/blog/demand-forecasting-software-cost-setup-use-cases/),
[ORATS pricing](https://orats.com/data-api),
[alt-data spend (Hedgeweek)](https://www.hedgeweek.com/hedge-fund-alt-data-spending-set-to-surge-says-new-research/),
[ForecastAPI](https://forecastapi.com/).

**What every serious competitor bundles** (and this product now has):
forecasting with uncertainty → anomaly detection → fine-tuning on customer
data → self-hosting → metered API keys. Nixtla's product page lists exactly
those four capabilities; they are table stakes, not differentiators. What
*no* competitor above bundles, as of this update: a machine-payable API
(x402), an unlinkable prepaid tier (Privacy Pass), or a quant-specific
backtest harness with multiple-testing-corrected significance tests. Those
three remain differentiators, not table stakes — for now. The model itself
is not one of them (§2 table).

### 2b. Adjacent open-source TimesFM projects (evidence of demand, not competition)

Independent from-scratch TimesFM reimplementations exist — evidence that
"give me weights I can actually run" is a real demand, and that we are not
alone in serving it:

- [**FareedKhan-dev/timesfm-from-scratch**](https://github.com/FareedKhan-dev/timesfm-from-scratch):
  MIT-licensed, ships a 70M checkpoint plus 17M/203M configs, full
  pretraining pipeline (synthetic generator, RevIN, masked objective). It is
  a research/education codebase with inference and a training CLI — **no
  server, no API, no billing, no dashboard, no quant tooling.**

None of these compete with the product; they validate the thesis and widen
the field of "TimesFM you can legally run" without touching the part we
actually sell: the service layer.

## 3. Positioning

> **Production forecasting you can audit, on infrastructure you control.**
> One self-hosted container that serves the best open-weight model for
> *your* data — our TimesFM-3 implementation, Amazon's Chronos family, or a
> classical baseline — chosen by a Diebold–Mariano-tested backtest on your
> own history, with metering, machine payments and an unlinkable credit tier
> built in.

The September pitch led with "the TimesFM-3 architecture with commercially
usable weights". Two facts retire that as the lead: Google now sells
commercial TimesFM-3 inside BigQuery (§1), and our own starter loses to an
8.7M-parameter Apache-2.0 Chronos-Bolt by 22% MAE on our own panel (§2).
What survives is everything around the model:

1. **Model-agnostic and measured.** The registry serves TimesFM-3
   checkpoints, Chronos / Chronos-Bolt / Chronos-2 (`TIMESFM3_CHRONOS_MODELS`)
   and classical baselines behind one API; `/v1/backtest` tells the customer
   which one wins on their series, with Holm-corrected significance. We
   publish the table where our own model loses. That is the brand.
2. **Self-hosted, one command, CPU-first.** `docker compose up` — no
   BigQuery, no Preview terms, no $12k/month. Data never leaves the
   customer's perimeter. Every model above runs on CPU at a compute cost far
   below the price (ledger §9).
3. **The plumbing nobody else sells.** Metering and quotas, fine-tuning as a
   job, machine-payable pricing (x402) and an unlinkable credit tier
   (Privacy Pass) ship today. Every competitor above sells a model (Chronos,
   Toto, Granite, Kronos) or a walled hosted API (Nixtla, BigQuery).

## 4. Who buys, and why now

| Segment | Trigger | What they run | Willingness to pay |
|---|---|---|---|
| **Mid-market ops / data teams** (retail, energy, logistics, SaaS metrics) — 50–2,000 series | Amazon Forecast closed to new customers since 2024; TimeGPT enterprise priced for the Fortune 500; commercial TimesFM-3 only inside BigQuery, in Preview | Forecast + anomalies on hourly/daily telemetry, demand, traffic | $300–$3,000/month (below Pecan's entry, far below Nixtla's) |
| **Platform / observability teams** embedding forecasts in their own product | Need an API with quotas and a license they can ship under, and are not already a Datadog customer (or want forecasting decoupled from an observability contract) | Metered `/v1/forecast`, `/v1/anomalies` behind their own keys | Per-point usage; OEM licensing |
| **Quant / risk desks** (small-to-mid funds, prop shops, treasury) | AIMA (published 16 Sep 2025): 95% of fund-manager respondents use Gen AI in their work, up from 86% in 2023, and 58% "expect increased Gen AI use in investment processes over the next year"; of the 18 large institutional investors also surveyed, 60% would be more likely to invest in a fund that "allocates a meaningful portion of its budget" to Gen AI ([AIMA](https://www.aima.org/article/press-release-front-office-gen-ai-adoption-shifts-from-if-to-when-for-leading-fund-managers-aima-research-finds.html)). Our reading: a working, testable pipeline is a fundraising argument, not only a research one | `/v1/volatility`, `timesfm3 finetune` on their panel, HAR-RV and TSMOM strategy references, backtester | $2,000–$10,000/month seat-less licenses; they already spend orders of magnitude more on data |
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

**Unit economics (ledger §6 and §9; CPU only, one thread, 7 series × 48
steps).** Model time per request, re-measured 2026-11-21: starter 17 ms
(≈19,000 points/s; the 2026-09-04 audit measured 11 ms on another torch
build), Chronos-Bolt-tiny 28 ms (≈12,000 points/s), Chronos-2 502 ms
(≈670 points/s), EWMA 71 ms. At ~$0.04/vCPU-hour that is roughly
$0.000001 per 1,000 points for the starter or Bolt and $0.00002 for
Chronos-2 — three to four orders of magnitude under the $0.02 overage
price, so model choice is an accuracy decision, not a margin one. Note the
classical baselines are *not* the cheap option as served: their bands refit
the baseline at 16 earlier origins, which makes EWMA ~4× slower than the
starter. (An earlier draft said "classical models are cheaper still"; that
does not hold for the served path.)

### The agent channel: x402 — real infrastructure, small demand

x402 (Coinbase's HTTP-402 payment protocol) is real infrastructure with
modest demand. Chainalysis (3 Jun 2026): agentic payments on Base "crossed
100 million transactions in approximately three quarters"
([Chainalysis](https://www.chainalysis.com/blog/x402-agentic-payments-adoption/));
Presenc's tracker (snapshot 2026-05-15) counts ~167M cumulative settled
transactions, ~85% on Base
([Presenc](https://presenc.ai/research/x402-protocol-adoption-tracker-2026)).
But CoinDesk (11 Mar 2026): "x402 currently processes only about $28,000 in
daily volume", "roughly half of observed x402 transactions reflect
artificial activity", average payment "around $0.20"; the ~$7B figure
sometimes quoted is a CoinGecko token-category market cap that the same
article says is dominated by Chainlink's LINK, not a valuation of x402
([CoinDesk](https://www.coindesk.com/markets/2026/03/11/coinbase-backed-ai-payments-protocol-wants-to-fix-micropayment-but-demand-is-just-not-there-yet)).
The Coinbase CDP facilitator charges $0.001 per on-chain settlement after
1,000 free a month (ledger §6) — small next to our $0.005–$0.02 prices.

**Plan:** treat x402 as distribution and optionality, not a modeled revenue
line. It costs nothing extra to support (the same 402 response works for
zero or a thousand agents), and agent marketplaces that list x402 endpoints
are free distribution. The financial case rests on the metered plans and
the self-hosted enterprise tier.

### Privacy as a feature, not a footnote

Every competitor's API logs who asked what. For the quant segment that is a
reason not to use a hosted forecaster at all. This product sells a credit
pool whose tokens are standard Privacy Pass (the operator provably cannot
link a call to a buyer) and documents funding those credits from a Privacy
Pools withdrawal, the compliant on-chain privacy protocol from the Ethereum
research community. Our reading: no incumbent above has a reason to build it — Nixtla and
Google bill per identified account, and Datadog and Amazon do not sell a
standalone forecasting API — and it costs almost nothing to run because
Privacy Pass is an IETF standard (RFC 9576–9578) with Cloudflare's own
library doing the cryptography at the edge.

## 6. Go-to-market (next 90 days)

1. **Publish the table where we lose.** Blog: "Commercial TimesFM-3 lives
   inside BigQuery. Here is a self-hosted forecasting service — and the
   backtest showing Amazon's Chronos-Bolt beats our own model by 22% on our
   own benchmark, so we serve it too." The reproducible command is the
   post. No incumbent can publish that sentence about itself.
2. **Amazon Forecast refugees.** Search-intent content: "Amazon Forecast
   alternative self-hosted". AWS's own migration path is SageMaker Canvas;
   we are the packaged, self-hostable API that serves Amazon's best open
   model with metering and a backtest.
3. **Design partners: 3 mid-market ops teams + 1 quant shop.** Free Team
   tier for a case study: run `/v1/backtest` across starter, Chronos and
   classical models on their data and publish the verdict either way.
4. **Decide on the bundled model (founder decision, see §7).** Spending GPU
   weeks training the 334M `base` config to chase Toto and Chronos is no
   longer the obvious next step: an 8.7M Apache-2.0 model already beats ours
   by 22%. The cheaper move is to ship Chronos-Bolt as the recommended model
   and keep TimesFM-3 for the covariate-heavy multivariate cases where our
   architecture is designed to differ — after measuring that it actually
   does (Chronos-2 also takes covariates; this repo's adapter does not wire
   them up yet).

## 7. Moat and risks

- **Happened: Google opened a commercial path.** TimesFM 3.0 is in BigQuery
  `AI.FORECAST` (Preview), and Google's README says commercial use is fully
  permitted there. Mitigation: the lane is now *self-hosted / off-GCP /
  production-grade today*, not "the only commercial TimesFM-3". Watch for
  GA and for any self-hostable commercial offering — that would close the
  lane for TimesFM specifically, not for the model-agnostic service.
- **Happened: our own model is not competitive.** Every Chronos model we
  served beats the bundled starter by ~20% MAE on its own panel (§2).
  Mitigation: the service is model-agnostic and the adapter is shipped and
  tested; the decision is what to *default* to (recommendation in §6, item
  4). Selling the starter as the headline model would contradict our own
  backtest.
- **Tried: specializing our model for volatility (ledger §11–13).** A
  pre-registered fine-tune of the starter on QLIKE, with causal smoothed-
  variance covariates, went from 2.7× worse than RiskMetrics to a
  statistical tie on 2020–2026 five-day QLIKE (0.988×, p 0.53), and beat
  HAR by 7.9% and every Chronos model by 29–41%. The pre-registered gate
  required beating RiskMetrics too, so it was not released. What it is
  good for: honest content ("zero-shot foundation models are bad at
  volatility; here is what fine-tuning buys, with the losing row"). What
  it is not: a reason for a risk desk to pay, since a 1994 EWMA ties it.
  On daily closes, the next gain likely needs better data (intraday
  realized variance), which is a data-cost decision, not a training one.
- **Tried: more data (ledger §14–16).** 51% more training windows, eight
  more assets and CBOE implied vol as an input did not improve validation,
  and a pre-registered test on 11 never-seen assets (crypto, EM FX,
  Nasdaq-100, Dow) found our model with the lowest five-day loss of all
  eleven forecasters — 3.4% below RiskMetrics, 1% below HAR — but not
  significantly, so it was not released. It beats every Chronos model by
  42% to 3.3× on those assets. Free daily data looks exhausted as a lever;
  the remaining one is intraday data.
- **Risk: open weights commoditize further.** Toto 2.0 (Apache-2.0, up to
  2.5B parameters, top spots on Datadog's reported benchmarks) and Kronos
  join Chronos and Granite. Mitigation: none of them ship a service. Toto and
  Kronos adapters are the same shape of work as the Chronos one (a few dozen
  lines against the registry contract) and are not done yet.
- **Risk: x402 revenue does not materialize.** CoinDesk puts real volume at
  ~$28k/day with roughly half of transactions artificial. Mitigation: the
  plan does not count on it (§5).
- **Risk: Chronos + AutoGluon-Cloud is "good enough" for AWS shops.**
  Mitigation: that serves AWS customers on AWS. Off-AWS, self-hosted, with
  metering, a significance-tested backtest, payments and privacy, it is
  still the same model served by us.
- **Moat that compounds:** every fine-tuning job and backtest run through
  the product yields an evaluated checkpoint and a model-comparison record
  on real customer data; with consent, that is a dataset nobody else has.
  And the billing layer — x402 plus Privacy Pass, interoperable between a
  self-hosted service and a $0 Cloudflare edge — takes a competitor a
  billing redesign to copy, not a retrain.

## 8. What changed in the product because of this research

| Finding | Change |
|---|---|
| Every competitor bundles anomaly detection | `POST /v1/anomalies`, `timesfm3 anomalies` — walk-forward scoring against the model's own predictive band |
| Fine-tuning is the paid feature at Nixtla and the source of gains in the finance literature | `timesfm3 finetune data.csv` — fine-tunes the starter, validates on the held-out tail, and prints the same DM-tested backtest the API serves, base vs fine-tuned vs classical |
| Usage-based pricing needs metering | Per-key plans and monthly quotas, `X-Usage-Points`/`X-Usage-Remaining` headers, `/v1/usage`, 429 on exhaustion, persistent counters |
| Google's weights are non-commercial | `LICENSE` (Apache-2.0) and `NOTICE` stating the bundled weights are self-trained and commercially usable |
| AI agents cannot sign up for API keys | `POST`/`GET` 402 responses with x402 pricing on every priced endpoint, in both the Python service and the Cloudflare Worker |
| Every hosted competitor logs who asked what | Privacy Pass credit pool (RFC 9576/9577/9578): tokens the operator cannot link to the purchase or to each other, interoperable between the service and the edge |
| Open-weight competition (Toto, Chronos-2, Kronos) leads public benchmarks | Positioning shifted from "best model" to "best-audited, best-plumbed service" (§3) |
| x402 ecosystem volume is real but genuine commercial spend is still small | Revenue plan (§5) stopped counting on x402 volume; it is now framed explicitly as free distribution/optionality, not a forecast line |
| The registry's "serves any checkpoint" claim only held for this repo's own `.pt` format | `timesfm3/serving/adapters.py`: `ChronosAdapter` runs Amazon's real `chronos-forecasting` pipeline through the registry's forecaster contract; `TIMESFM3_CHRONOS_MODELS` loads it with no code changes; `ModelEntry` dispatch generalized from "kind == timesfm3" to "has a forecaster" so a foreign model reports its real kind (`GET /v1/models`) instead of masquerading as TimesFM-3; `tests/test_adapters.py` runs a real forecast, not a mock |
| Commercial TimesFM-3 now exists inside BigQuery (Preview) | §1 and §3 rewritten: the lane is self-hosted / off-GCP, not "the only commercial TimesFM-3" |
| Chronos-Bolt-tiny beats our starter by 22% MAE on the model-card panel (ledger §9) | Positioning leads with model-agnostic serving; GTM item 1 publishes the losing table; bundled-model roadmap reopened as a founder decision |
| The Chronos adapter crashed on Chronos-2 (list-shaped outputs) — it had only been tested on Bolt | Adapter normalizes both output conventions; an offline stub test pins both shapes and runs without the optional dependency |
| "Classical models are cheaper still" did not hold for the served path | Unit economics re-measured per model; the sentence corrected (§5) |
| Zero-shot foundation models are poor at volatility; a QLIKE fine-tune of ours ties RiskMetrics and beats HAR and all Chronos models (ledger §13) | Harness and fine-tuning code shipped (`timesfm3/quant/volmodel.py`); the model is not released because it does not beat RiskMetrics, per the pre-registered gate |
| More daily data and implied vol did not move the model past classical forecasters on unseen assets (ledger §16) | Extended FRED universes and implied-vol input shipped in the harness; model still not released; the next lever is intraday data, a cost decision |
