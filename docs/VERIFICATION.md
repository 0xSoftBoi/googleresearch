# Verification ledger

*Audit date: 2026-09-04, from a fresh clone in a clean 4-vCPU CPU-only
container (torch 2.14.0+cpu, Python 3.11). Every measured number in
`README.md`, `docs/PRODUCT.md`, `docs/HEDGE_FUND_APPLICATIONS.md` and
`docs/BUSINESS.md` was re-run from the commands below, and every external
citation was re-fetched and read. Verdicts:*

- **REPRODUCED** — re-run gives the printed figure (to the printed precision,
  or within the drift explained in the note).
- **PARTIAL** — the claim's mechanism was checked on a subset; the full
  original run was not repeated (and why).
- **NOT REPRODUCED** — the re-run gave a different figure; the doc has been
  corrected to what was measured.
- **UNVERIFIABLE** — no artifact or script exists that could reproduce it;
  the doc has been rewritten or the claim removed.
- **CITATION OK / CITATION FIXED** — the cited page does / did not support
  the statement attributed to it.

Nothing in this ledger was taken on trust from the earlier sessions that
wrote the documents; where a figure could not be re-derived it is marked.

---

## 1. Bundled model and model card (`docs/PRODUCT.md`, `timesfm3/assets/`)

| Claim | Verdict | Evidence |
|---|---|---|
| `starter-small.pt` is the `small` config, ≈5.2 M parameters, trained 6000 steps, batch 16, on ETTh1/ETTm1/ETTm2/exchange, holdout ETTh2 | REPRODUCED | Checkpoint `meta` block (also in `cloudflare/public/models/starter-small.json`): `num_parameters 5191808`, `training_steps 6000`, `batch_size 16`, `corpus [ETTh1, ETTm1, ETTm2, exchange]`, `holdout [ETTh2]`, `best_val_loss 45.12`, `train_minutes 26.0`, created `2026-09-01T23:28:44Z`. Instantiating `TimesFM3Config.small()` gives 5.19 M parameters. |
| The dashboard sample panel is the ETTh2 tail (so the demo is zero-shot) | REPRODUCED | `timesfm3/serving/static/sample.csv` is byte-for-byte the last 2000 rows of `data/raw/ETTh2.csv` (same dates, max abs value diff 0.0). |
| Model-card table: ETTh2, 7 series, context 256, horizon 24, 20 windows — starter 2.183 / 0.881 / [0.751, 1.008] / p 0.085 / 78% / 140 (33) / no difference; ewma 2.309 / 0.931 / better; ar4 2.432; ar1 2.448; last-value 2.479; drift 2.488; ctx-mean 2.555 | REPRODUCED | `timesfm3 backtest timesfm3/serving/static/sample.csv --context 256 --horizon 24 --windows 20` — every cell identical. 70 s wall. |
| "tiny/small/base ≈ 1 M / 5.2 M / 334 M parameters; base = 20 layers, dim 1280, 16 heads" | REPRODUCED (with a caveat, §5) | Instantiated: tiny 0.97 M, small 5.19 M, base 334.08 M; 20 / 1280 / 16. |
| Fine-tune: "200 CPU steps (under a minute) cut MAE 2.3% versus the base model" on the ETTh2 demo panel | NOT REPRODUCED (direction yes, size no) | `timesfm3 finetune timesfm3/serving/static/sample.csv --steps 200 --periods 24,168`: 0.8 min, fine-tuned MAE 2.823 vs base 2.855 = **−1.1%**, verdict vs last-value "no difference"; AR(4) 2.471 ("better") beat both on that tail. Same seed (`seed=0`), different torch build. Doc now reports the range and the AR(4) result. |
| Anomalies: "on seasonal test data with two planted spikes, the bundled model flags exactly those two with no false alarms; EWMA misses one and raises five" | UNVERIFIABLE → rewritten | No script or dataset for this sentence exists in the repo. On a synthetic daily+weekly sinusoid (noise σ 0.8, ±12 spikes at t=400, 520; `--context 192 --block 24 --threshold 2`) the starter scored the spikes 1.78 / 1.90 and EWMA 0.51 / 0.98 — neither crossed the threshold of 2; nothing else scored above 2 for the starter, 54 points scored above 1 for EWMA. The only pinned behaviour is `tests/test_anomaly.py` (random walk, ±8 spikes, last-value model: both flagged, ≤6 false alarms). The paragraph now says that. |
| Test counts: "113 total" (0.3.0), "102 total" (0.2.0) | PARTIAL | Current suite: **131 passed, 3 skipped, 1 failed** — the failure (`tests/test_x402.py::test_worker_x402_flow_matches_spec`) is only because `cloudflare/node_modules` was not installed; after `npm install` in `cloudflare/` all x402 and JS-parity tests pass (8 passed). Historical counts not re-checked. |
| Docker image "~1 GB" | UNVERIFIABLE here | No Docker daemon in the audit container. |

## 2. Real-data notebook (`notebooks/timesfm3_real_data.ipynb`, README table)

| Claim | Verdict | Evidence |
|---|---|---|
| Notebook was actually executed: 5.2 M params, 25 min on CPU | REPRODUCED (provenance) | Cell execution metadata: training cell ran 03:39:03 → 04:04:02 UTC on 2026-09-01 (25.0 min, printed `wall time: 25.0 min`); git commits `bc6dfe5` (03:39) and `faebb3b` (04:04) bracket it. Execution counts are sequential 1–8. |
| Scaled-MAE table: ETTh1 +cal 0.66 / 1.10 / 0.73; ETTh1 no-cal 0.81; ETTh2 zero-shot +cal 0.76 / 1.06 / 0.92; no-cal 0.87; calendar ≈13%; calibration gap 0.064 | REPRODUCED (from stored outputs) | Cell outputs contain exactly 0.659 / 1.097 / 0.734, 0.807, 0.760 / 1.061 / 0.921, 0.872, "+12.9% improvement", "0.064". The notebook itself was **not** re-executed (25 min of training whose checkpoint is not shipped); the bundled starter uses the identical recipe and its ETTh2 backtest reproduced (§1). |

## 3. Hedge-fund applications (`docs/HEDGE_FUND_APPLICATIONS.md`, README)

FRED data re-downloaded 2026-09-04 (two trading days longer than the
2026-09-01 run); all 17 default instruments fetched except `GOLDPMGBD228NLBM`
(FRED 404 — the loader skips it, giving the documented 16-asset universe).

| Claim | Verdict | Evidence (`python examples/hedge_fund/…`) |
|---|---|---|
| TSMOM 1975–2026: 0 bps +5.10% / 4.53% / Sharpe 1.13 / t 7.2; 5 bps 0.89 / 5.6; 10 bps +2.95% / 0.65 / 4.1; 20 bps +0.80% / 0.18 / 1.1; long-only 10 bps +1.32% / 5.50% / 0.24 / 1.5 | REPRODUCED | `trend_following.py`: +5.11% / 4.53% / 1.13 / 7.2; +4.04% / 0.89 / 5.7; +2.96% / 0.65 / 4.1; +0.81% / 0.18 / 1.1; +1.33% / 5.50% / 0.24 / 1.5. Differences ≤0.01 from two extra days. |
| Decade decay "1980s Sharpe 1.67 → 2010s −0.53" | REPRODUCED | 1.67 and −0.52. |
| HAR vs RiskMetrics QLIKE ratios: NASDAQ 0.960 (p=0.056, tie), EURUSD 1.024, WTI 1.009, UST10Y 1.112 (worse); last-value loses by ~10³ | REPRODUCED | `volatility_forecasting.py`: 0.960 p=5.6e-02 [no difference]; 1.024; 1.009; 1.112 [worse]; last-rv ratios 415–3111. |
| Moreira–Muir overlay on NASDAQ: Sharpe 0.49 → 0.71, vol 20.1% → 15.8% | REPRODUCED | 0.49 → 0.71; 20.07% → 15.81%. |
| Signal harness net 10 bps: drift +2.19% / 1.25; AR(4) −2.78% / −1.44; EWMA placebo −0.03% / −0.87 | REPRODUCED | `model_signal.py`: +2.19% / 1.26; −2.77% / −1.44; −0.04% / −0.90. |
| TimesFM-3 `tiny`, 2000 CPU steps on the FRED panel: −3.39% / Sharpe −1.57 | see §3a | Re-trained from scratch (`pretrain_markets.py 2000`, 4.0 min, best val loss 6.06) and run through `model_signal.py <ckpt>`. |
| "16 assets: 9 FX majors, NASDAQ and S&P 500, WTI/Brent/natural gas, 2y and 10y Treasury proxies, histories to 1971" | REPRODUCED | Loader output: 16 assets, 1975-01-02 .. 2026-09-03; GBPUSD etc. from 1971-01-04. Note: `DEFAULT_UNIVERSE` also lists LBMA gold, which FRED no longer serves (404) — it is skipped at load time, so the doc's 16 is what runs. |

### 3a. TimesFM-3 tiny signal row (re-trained in this audit)

Documented: **−3.39% / Sharpe −1.57**. Re-trained from scratch
(`pretrain_markets.py 2000`: 16 channels × 16,527 steps, 1.0 M params,
best val loss 6.06 at step 1750, 4.0 min) and scored:
`signal[timesfm3]  ann.ret −1.89%  ann.vol 1.27%  Sharpe −1.49 (t = −9.0)`.
Verdict **PARTIAL**: a fresh random initialization on a different torch
build cannot be expected to land on the same weights; what the doc claims —
that an under-trained tiny checkpoint loses money through the harness and
clusters with AR(4) (−1.44) rather than drift (+1.26) — holds. The doc now
prints both runs.

## 4. Polymarket (`README.md`, `timesfm3/data/polymarket.py`)

Full re-run needs ~21 GB of hourly parquet (8 training hours + 13 test
hours) plus training; not repeated. Two hours were downloaded in full and the
verifiable mechanisms checked on them.

| Claim | Verdict | Evidence |
|---|---|---|
| Hourly files are ~1 GB, ~10⁸ rows/hour | REPRODUCED | `2026-08-29T02.parquet` 1,164,084,589 bytes; `T03` 1,051,910,499 bytes. |
| "Of 24 hours audited, 23 verified and one (`2026-08-29T02`) reproducibly hashed to a value the manifest does not list" | REPRODUCED | Downloaded `T02` once more, full length (matches `content-length`): sha256 `696c6356…6bce8`; the archive's `v3/SHA256SUMS.txt` (fetched 2026-09-04) lists `467d071c…f591` for that file. `T03` hashes to exactly its manifest entry `6c4bc48f…7970`. So the discrepancy is still live three days later. |
| "the two mid prices are *exactly* complementary (`p_yes + p_no == 1` to the tick, zero variance)" | REPRODUCED | On `T03`, top-300 traded markets → 267 continuously quoted: max |p_yes + p_no − 1| over every grid cell of every market = 0.000000; per-market std 0.000000. |
| Lag-1 AC table over 13 hours / 163 markets / 15 s grid: mid 0.982, spread 0.910, ret −0.009, abs_ret 0.204, quotes 0.812, trades 0.568, volume 0.332 | PARTIAL | One hour (`T03`, 240-step grids, 267 markets), median per market: mid **+0.962**, spread +0.872, ret **−0.032**, abs_ret +0.192, quotes +0.580, trades +0.216, volume +0.045. Same ordering and the same conclusion (levels ≈ random walk, returns ≈ 0, book activity persistent); absolute values on a 240-step window are biased toward zero relative to the 13-hour (3120-step) series, which explains the lower activity numbers. No script in the repo produces the README table; the code used here is recorded below. |
| Cross-day benchmark table (TimesFM-3 `quotes` 0.704 \*, `mid` 1.074 \* worse, seeds 0.725/0.728/0.731, horizon sweep, leak check) | NOT RE-RUN | Requires the 21-hour download and four training runs. The scripts (`examples/train_polymarket.py`, `examples/evaluate_polymarket.py`) exist and implement the described protocol (non-overlapping windows, HAC DM, market-cluster bootstrap, Holm, frozen/active split); the numbers themselves were not independently regenerated. Treat them as reported-by-one-run until someone re-runs. |
| "183,641 markets appear in an hour; most are five-minute contracts" | (new measurement) | `scan_markets` on `T03`: 183,641 distinct market ids, 3,213 with two tokens and at least one fill, 267 of the top 300 quoted ≥98% of the hour. |

Lag-1 AC code used (one hour):

```python
from timesfm3.data.polymarket import top_markets, build_market_panels, select_covered
paths = ["data/polymarket/v3/2026-08-29/03/2026-08-29T03.parquet"]
cov = select_covered(build_market_panels(paths, top_markets(paths, 300), freq_seconds=15.0))
def lag1(x):
    x = x[np.isfinite(x)]; x = x - x.mean()
    return float((x[:-1] * x[1:]).sum() / (x * x).sum()) if x.size > 10 and x.std() else np.nan
{ch: np.nanmedian([lag1(p.features[ch][0]) for p in cov]) for ch in
 ["mid", "spread", "ret", "abs_ret", "quotes", "trades", "volume"]}
```

## 5. Architecture claims about the *released* model (README, `ARCHITECTURE.md`)

| Claim | Verdict | Source read |
|---|---|---|
| Patches of 32 steps; 9 quantiles q10…q90; >1 T time points; Contiguous Patch Masking; alternating (temporal / variate) attention | CITATION OK | Google Research blog, 2026-08-31. |
| 20 layers, model dim 1280, 16 heads; "CPM Iterative RevIN"; 64-step output patches; GiftEvalPretrain / Wikipedia pageviews / Google Trends corpora | CITATION OK (source is the HF model card, not the blog) | `huggingface.co/google/timesfm-3.0-pytorch` README. |
| "≈334 M parameters" | CITATION FIXED | Google says "330 million" (blog) / "0.3B" (card). 334.08 M is what this implementation's `base` config instantiates. README and ARCHITECTURE now say both. |
| "contexts up to 16k steps" | CITATION FIXED | Stated by Google for TimesFM **2.5**, not in the TimesFM-3 blog or card. Now labelled as this implementation's configured limit. |
| License: "TimesFM Non-Commercial License v1.0", forbids "any revenue-generating activity" and use "in direct or indirect interactions with end users or production systems" | CITATION OK | HF `LICENSE` file, verbatim. |
| The New Stack "You can't use it at work (yet)"; aiweekly 31 Aug 2026 release, BigQuery "in coming weeks" | CITATION OK | Both pages fetched. |
| BigQuery `AI.FORECAST` "hosts TimesFM 2.0 (500M)" | CITATION FIXED | Docs page: supports TimesFM 2.0 and 2.5, default 2.5; no parameter count on the page. |

## 6. Business research (`docs/BUSINESS.md`)

| Claim | Verdict | Note |
|---|---|---|
| Nixtla Enterprise "$12,000/month flat", 30-day trial | CITATION OK | SoftwareAdvice ("$12,000.00 flat rate per month"); Nixtla plans page (30-day trial, no prices listed). |
| Nixtla Series A $16 M, Feb 2026 | CITATION FIXED | Fact is right (AccessNewswire / Axios, 2026-02-05, led by Energize Capital) but the cited newmarketpitch.com page never mentions Nixtla. Now cites the press release. |
| Vertex forecasting $0.20 → $0.02 per 1,000 predictions | CITATION OK | Vertex pricing table: $0.20 / $0.10 / $0.02 per 1,000 by volume tier. |
| Pecan "$760–$1,400/month" | CITATION FIXED | pecan.ai/pricing shows no prices. Capterra/GetApp list $950 and $1,750/month. Doc restated. |
| ForecastAPI 200 free/month, $0.0016–$0.0033/call | CITATION OK | forecastapi.com. |
| "$0.50–$5 per SKU/month" (setupbots) | PARTIAL | Cited page is JS-rendered and returned no text; the same range appears on getmonetizely.com. Left in place, flagged here. |
| "$250–$28,000/month, 4-week to 6-month setup" (shivlab) | CITATION OK | Verbatim on page. |
| "ORATS from ~$100/month; large funds spend $15–60M/yr on alt data" (permutable.ai) | CITATION FIXED | permutable.ai page mentions neither ORATS nor any spend figure. ORATS: $99/month (orats.com). Alt-data: Hedgeweek — large multi-strategy funds average ~$5 M/yr. No source found for $60 M; removed. |
| Amazon Forecast closed to new customers 2024 | CITATION OK | signisys.com. |
| Moirai 2.0 research-only; Granite TTM Apache-2.0 | CITATION OK | HF cards (TTM card says "starting from 1M params"; the 5 M upper bound is from the TTM paper). |
| x402: Coinbase protocol, Cloudflare Agents SDK + MCP support Sept 2025; CDP facilitator 1,000 free settlements/month then $0.001 | CITATION OK | Cloudflare blog 2025-09-23; CDP docs. Note the fee is per on-chain settlement, not per HTTP request. |
| Unit economics "~55 ms per request for 7 × 48 ≈ 6,000 points/s" | NOT REPRODUCED (faster) | Measured: 11.5 ms model time single-threaded, 14.6 ms HTTP round trip through the FastAPI app in-process (server-reported `latency_ms` 10.9), ≈23,000 points/s. Doc updated; conclusion unchanged. |

## 7. Hedge-fund literature citations (`docs/HEDGE_FUND_APPLICATIONS.md`)

| Claim | Verdict | Note |
|---|---|---|
| Rahimikia et al. 2025 "Man Group-affiliated" | CITATION FIXED | arXiv 2511.18578 authors are at Alliance Manchester Business School, UCL and Shanghai University; no Man Group / Oxford-Man affiliation anywhere in the PDF. Finding as described (zero-shot TSFMs underperform CatBoost/LightGBM; financially pre-trained ones gain) is accurate. |
| Hurst–Ooi–Pedersen "Century of Evidence": positive every decade, "~0.4 net Sharpe" | CITATION FIXED | Positive net-of-fee Sharpe in every decade: yes. Full-sample net Sharpe is **0.77**; 0.4 appears only as "suppose that the strategy only realizes a Sharpe ratio of 0.4 net of fees". Doc corrected. |
| Gu–Kelly–Xiu ~0.4% monthly OOS R² | CITATION OK | "R²oos … peaks at 0.40% for NN3". |
| Voleon "a little bit better than 50%" | CITATION OK | Bloomberg Businessweek, 2019-12-04 (body confirmed via reprint). |
| Man AHL ML in client portfolios since early 2014 | CITATION OK | man.com, 2016. |
| Lim–Zohren–Roberts DMN; Wood et al. SMFR "~33% Sharpe improvement 1995–2020"; Momentum Transformer; TFT; 101 Formulaic Alphas (0.6–6.4 days, ~16% correlation); Demystifying Managed Futures gross Sharpe 1.8; Frazzini–Israel–Moskowitz ~$1 T, ~10× below academic estimates; Harvey et al. JPM 2018; D. E. Shaw 2018; Two Sigma, Cubist pages | CITATION OK | All fetched and matched (SSRN pages returned 403; AQR/Quantpedia mirrors used). |
| arXiv 2606.27100 "small and sparse" | CITATION OK | Noguer i Alonso & Pereira Franklin, June 2026, TimesFM-2.5 on five US equities; abstract says exactly that. |
| Renaissance ~66% gross 1988–2018 | as stated | Doc already labels this folklore-with-sources; not re-checked. |

## 8. Synthetic-only pre-training (README "Pre-training (synthetic only)")

| Claim | Verdict | Evidence |
|---|---|---|
| tiny config, 8000 steps: best val loss 1.405; held-out synthetic scaled MAE 0.94 vs last-value 1.26 / ctx-mean 1.18; ETTh1 zero-shot 0.90 vs 1.04 / seasonal-naive 0.70; "~25 minutes on CPU" | IN PROGRESS | No checkpoint for this run is shipped, so it is being re-trained from scratch with the README command at 8000 steps and evaluated with `examples/evaluate.py` and `examples/evaluate_ett.py`; the result is appended in §8a by the follow-up commit. |

---

## How to re-run this audit

```bash
pip install -e ".[serve,test,polymarket,examples]"
bash data/download.sh                                   # ETT + exchange (≈25 MB)
python -m pytest -q                                     # 131 passed, 3 skipped (+8 after `cd cloudflare && npm install`)
timesfm3 backtest timesfm3/serving/static/sample.csv --context 256 --horizon 24 --windows 20
timesfm3 finetune timesfm3/serving/static/sample.csv --out ft.pt --steps 200 --periods 24,168
python examples/hedge_fund/trend_following.py           # downloads FRED (≈4 MB)
python examples/hedge_fund/volatility_forecasting.py
python examples/hedge_fund/model_signal.py
python examples/hedge_fund/pretrain_markets.py 2000 && python examples/hedge_fund/model_signal.py data/markets_tiny.pt
python -m timesfm3.train --config tiny --steps 8000 --batch-size 16 --context-patches 8 --horizon-patches 2 --checkpoint tiny.pt
python examples/evaluate.py --checkpoint tiny.pt && python examples/evaluate_ett.py --checkpoint tiny.pt --data data/raw/ETTh1.csv
data/download_polymarket.sh 2026-08-29T02 2026-08-29T03   # ≈2.2 GB; T02 will fail its checksum
```
