# Launch kit

The model research is closed: two pre-registered volatility attempts tied the
classical frontier and did not beat it (`docs/VERIFICATION.md` §11–16). The
lead asset is now the **benchmark itself**
([`docs/launch/volatility-benchmark.md`](launch/volatility-benchmark.md)) and
the product that produces it for any dataset. This file holds the steps and
the copy, ready to paste.

## Day 0: get a public URL (founder, about 10 minutes)

The edge Worker is **not deployed yet**. On 2026-11-22 the Cloudflare
account had one Worker, `suwappu-router`, and no `timesfm3-edge`.

The KV namespace `timesfm3-edge` and the D1 database `timesfm3-credits` already
exist in the account. D1 has no tables yet; the first deploy applies the
migration.

1. Merge this branch to `main`. The Deploy button in the README points at
   `main/cloudflare`.
2. Fastest: create a Cloudflare API token (Workers Scripts:Edit, Workers KV
   Storage:Edit, D1:Edit) and add it to the GitHub repo as the
   `CLOUDFLARE_API_TOKEN` secret, plus `CLOUDFLARE_ACCOUNT_ID` (your
   personal account's ID, shown on the Workers dashboard). Then run the **cloudflare-edge**
   workflow (Actions tab, *Run workflow*). From then on, every push to `main`
   that touches `cloudflare/` deploys. Alternatives: the **Deploy to
   Cloudflare** button (then run `npx wrangler d1 migrations apply
   timesfm3-credits --remote` once from `cloudflare/`, or Privacy Pass credits
   fail on a missing table), or `cd cloudflare && npm install && npx wrangler
   deploy && npx wrangler d1 migrations apply timesfm3-credits --remote`. You get
   `timesfm3-edge.<account>.workers.dev`: landing page, JS classical API,
   in-browser TimesFM-3, waitlist in KV. There is no backend to pay for.
3. Set the `ADMIN_TOKEN` secret so you can read signups: `cd cloudflare &&
   npx wrangler secret put ADMIN_TOKEN` (any long random string), or add it
   under the Worker's *Settings → Variables and Secrets*. Without it,
   `GET /api/leads` is locked and the day-14 table below has no signup count.
4. Optional: attach a custom domain in the Cloudflare dashboard, then replace
   `<URL>` in the copy below.
5. Smoke test: open the landing page, run the in-browser forecast, submit
   the waitlist form once with your own email, then check it arrived:
   `curl -H "authorization: Bearer $ADMIN_TOKEN" <URL>/api/leads`.

## Day 1: post the benchmark (one channel at a time, reply to every comment)

Order matters. Post on HN first, then Reddit the next day, then X.

### Show HN

**Title** (80 characters max):
`Show HN: A 1994 formula still beats time-series foundation models at volatility`

**Body:**

> We build an open-source (Apache-2.0) forecasting service around an
> independent TimesFM-3 implementation. Before claiming anything for finance,
> we benchmarked volatility forecasting on 2020–2026 FRED daily data.
>
> Zero-shot, five days ahead, QLIKE: Chronos-Bolt is 31–40% worse than
> RiskMetrics (J.P. Morgan's 1994 EWMA), and Chronos-2 is 43% worse, or
> 3.2× worse on assets outside our training panel. That is after a
> calibration fitted on 2015–2019 for every model.
>
> We then fine-tuned our own model on QLIKE twice, pre-registering the
> release gate before scoring the test set. It reached a statistical tie with
> RiskMetrics and beat HAR and every Chronos model. On 11 never-seen assets
> it had the lowest loss of all eleven forecasters, but not significantly. So
> it did not ship, and the losing rows are published.
>
> The product is the harness: walk-forward backtest, HAC Diebold–Mariano,
> Holm correction, on your own CSV, so you can see whether a fancy model
> beats a cheap one on your data before paying for it.
>
> Write-up: <link to docs/launch/volatility-benchmark.md>
> Repo: https://github.com/0xSoftBoi/googleresearch
> Live demo, no signup: <URL>

**Expected first questions, with answers:**

- *"Why not intraday data?"* Cost. It is the one lever left, and we say so.
- *"Did you fine-tune Chronos?"* No, zero-shot only. That is in the caveats.
- *"Isn't this just GARCH?"* RiskMetrics is IGARCH with fixed λ. HAR is the
  standard academic benchmark. We beat HAR significantly, not RiskMetrics.

### Reddit: r/algotrading (day 2)

r/algotrading allows self-posts with code. Check r/quant's current
self-promotion rule before posting there; if in doubt, skip it.

**Title:** `I benchmarked Chronos / TimesFM against RiskMetrics for vol forecasting. The 1994 EWMA wins.`

**Body:** paste the table from the write-up, add two sentences on the
pre-registration, and link the repo. Do not lead with pricing.

### X / LinkedIn thread (day 3)

1. Time-series foundation models are everywhere. We tested them on the
   thing quants actually pay for: volatility. A 1994 formula wins. 🧵
2. Zero-shot, 5-day QLIKE vs RiskMetrics: Chronos-Bolt 1.31×, Chronos-2
   1.43×, and 3.2× on unseen assets. (Table image: screenshot the write-up
   table.)
3. We fine-tuned our own model twice and pre-registered the gate. Result: a
   tie with RiskMetrics, a significant win over HAR and every Chronos model.
   The gate said beat everything, so it didn't ship.
4. Lesson: on daily closes, the classical frontier is hard to move. Ask any
   vendor for the QLIKE table.
5. The harness that produced this runs on your CSV, open source: <repo>

## Day 2–10: direct outreach (the step that actually produces revenue)

Target: 30 named people at small-to-mid funds, prop shops, crypto market
makers, and fintechs with a risk team. Use LinkedIn search for "quantitative
researcher", "risk", "volatility" at firms with 10–200 staff. Send 5 a day.

**Email or DM (under 100 words):**

> Subject: Foundation models vs RiskMetrics for vol: our numbers
>
> Hi {name}, we benchmarked Chronos and TimesFM against RiskMetrics and HAR
> on 2020–2026 daily data. Zero-shot, the foundation models lose by 31–43%
> on QLIKE, and a pre-registered fine-tune only ties RiskMetrics. Write-up:
> <link>.
>
> If {firm} is evaluating ML forecasters, we'll run the same significance-
> tested backtest on your own series for free and send you the table. Is
> 20 minutes next week useful?

The offer is the design-partner deal already on the landing page: Cloud Team
free for three months in exchange for a case study.

## What to measure (check every evening)

| signal | where | "this is working" by day 14 |
|---|---|---|
| Waitlist signups | `GET /api/leads` with `Authorization: Bearer $ADMIN_TOKEN` (set the `ADMIN_TOKEN` secret at deploy; see `cloudflare/README.md`) | 50+ |
| Repo stars | GitHub | 100+ |
| Outreach replies | your inbox | 5 replies, 2 calls booked |
| Free backtests run for prospects | manual | 2 |

If HN and Reddit fall flat but outreach gets replies, the story works and the
channel didn't: keep doing outreach. If nobody replies to 30 targeted
messages, the offer is wrong. Change the offer (for example, to a paid
benchmark report), not the model.

## Not doing (deliberately)

- No third volatility training attempt on daily data. The ledger shows that
  lever is exhausted.
- No intraday data purchase until a paying prospect asks for it.
- No new endpoints until a design partner needs one.
