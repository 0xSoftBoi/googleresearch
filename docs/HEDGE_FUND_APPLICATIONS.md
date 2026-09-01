# Hedge-Fund Applications of This Codebase

What systematic funds actually do with time-series forecasting, mapped onto
this repository, with measured results on real data. Every strategy here is
runnable from `examples/hedge_fund/` against a free, reproducible data source
(FRED — no API key), and every claim about the industry carries a citation.

**TL;DR of the honest economics before the details:**

- Daily *return direction* is close to unforecastable. The best neural
  networks in the flagship academic study reach ~0.4% out-of-sample monthly
  R² on individual US stocks ([Gu, Kelly & Xiu, RFS 2020](https://academic.oup.com/rfs/article/33/5/2223/5758276)).
  Voleon's founder describes production edge as being
  ["a little bit better than 50%"](https://prod.cm.bloomberg.com/news/articles/2019-12-04/voleon-s-kharitonov-and-mcauliffe-are-the-killer-quants).
- *Volatility* is highly forecastable
  ([Andersen & Bollerslev, IER 1998](https://www.semanticscholar.org/paper/93ffc319bf39882dd471e1594ce242aee101ef38)),
  which is why the reliable ways to monetize a forecaster are **position
  sizing and risk**, not direction-calling.
- Generic-pretrained time-series foundation models transfer poorly to
  financial returns; the same architectures **pre-trained on financial
  data** produce real gains
  ([Rahimikia et al. 2025, Man Group-affiliated](https://arxiv.org/abs/2511.18578)).
  This repo is a trainable TimesFM-3 implementation with a financial
  pre-training corpus — exactly the setup that paper argues for.

---

## 1. What real funds do (and where the evidence is)

### Man AHL / Man Group and the Oxford-Man Institute

Man AHL has traded machine-learning systems in client portfolios
[since early 2014](https://www.man.com/insights/the-rise-of-machine-learning)
and co-funds the Oxford-Man Institute, whose published papers are the most
direct public blueprint for using a neural forecaster in a futures/FX
program:

- **Deep Momentum Networks** — an LSTM inside the classic
  volatility-scaling wrapper, sizing positions from the network's
  standardized forecast ([Lim, Zohren & Roberts, JFDS 2019](https://arxiv.org/abs/1904.04912)).
  `timesfm3.quant.strategies.forecast_signal_positions` is this harness.
- **Slow Momentum with Fast Reversion** — adds changepoint detection;
  ~33% Sharpe improvement over classic trend, 1995–2020
  ([Wood, Roberts & Zohren, JFDS 2022](https://arxiv.org/abs/2105.13727)).
- **Momentum Transformer** — attention-based version, robust to costs
  ([Wood et al. 2021](https://arxiv.org/abs/2112.08534)).
- **Temporal Fusion Transformers** — multi-horizon *quantile* forecasting
  with known-future covariates ([Lim et al., IJF 2021](https://arxiv.org/abs/1912.09363)) —
  architecturally the closest published ancestor of TimesFM-3's
  point-plus-quantile, covariate-aware head.

Man Group researchers also wrote the definitive study of volatility
targeting ([Harvey et al., JPM 2018](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3175538)):
Sharpe gains concentrate in equities and credit, but tail-risk reduction
shows up everywhere.

### AQR

AQR principals wrote the canonical trend-following literature this repo's
baseline strategy implements:

- [Moskowitz, Ooi & Pedersen, "Time Series Momentum", JFE 2012](https://w4.stern.nyu.edu/facdir/lpederse/papers/TimeSeriesMomentum.pdf):
  sign of the trailing 12-month return, volatility-scaled, across ~58
  futures/forwards → diversified Sharpe above 1, low correlation to equities.
- [Hurst, Ooi & Pedersen, "Demystifying Managed Futures", JOIM 2013](https://www.aqr.com/-/media/AQR/Documents/Insights/Journal-Article/Demystifying-Managed-Futures.pdf):
  a gross Sharpe of 1.8 for the multi-horizon version, and TSMOM largely
  explains CTA-index returns.
- [Hurst, Ooi & Pedersen, "A Century of Evidence on Trend-Following", JPM 2017](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2993026):
  positive in every decade 1880–2016, **~0.4 net Sharpe** at the century
  scale — the honest long-run number after costs and crowding.
- [Frazzini, Israel & Moskowitz, "Trading Costs of Asset Pricing Anomalies"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2294498):
  ~$1T of AQR's own executions show institutional costs are ~10x below
  academic estimates — which is why the cost sweep in our backtests
  (0–20 bps) brackets reality rather than assuming one number.

### Two Sigma, D.E. Shaw, Cubist, WorldQuant, Renaissance, Voleon

- Two Sigma publishes distributional/regime models on its Factor Lens
  ([ML regime modeling, 2021](https://www.twosigma.com/wp-content/uploads/2021/10/Machine-Learning-Approach-to-Regime-Modeling_.pdf));
  production alpha details are not public.
- D.E. Shaw formed a dedicated ML research group in 2018 and says ML has
  been deployed in production for decades
  ([announcement](https://www.prnewswire.com/news-releases/d-e-shaw-group-forms-new-machine-learning-research-group-300698027.html)).
- Cubist (Point72's quant arm) publicly runs computer-driven strategies
  across equities, futures and FX ([point72.com/cubist](https://point72.com/cubist/)).
- WorldQuant's "alpha factory" is documented in
  ["101 Formulaic Alphas"](https://arxiv.org/abs/1601.00991): 101 real
  production-style signals, holding periods 0.6–6.4 days, pairwise
  correlation ~16% — the public template for "many weak signals, combined".
- Renaissance: the famous Medallion numbers (~66% gross annually,
  1988–2018) trace to [Zuckerman's book](https://www.bloomberg.com/news/articles/2019-11-12/the-unsolved-mystery-of-the-medallion-fund-s-success)
  and are **not** first-party verifiable; treat as folklore-with-sources.
- Voleon is the ML-first fund; its founder's "slightly better than 50%"
  framing above is the right calibration for what a return forecaster can
  achieve.

---

## 2. What this repo implements, and the measured results

All numbers below are from real FRED data (16 assets: 9 FX majors, NASDAQ
and S&P 500, WTI/Brent/natural gas, 2y and 10y Treasury total-return
proxies built from constant-maturity yields), produced by the scripts named
in each subsection on 2026-09-01. Drawdowns are in log-return units.

### 2.1 Time-series momentum (`examples/hedge_fund/trend_following.py`)

MOP 2012 construction: sign of trailing 12-month return, 10% per-asset vol
target (RiskMetrics ex-ante), equal-weighted across live assets, daily
positions, 1975–2026:

| Cost assumption | Ann. return | Ann. vol | Sharpe | HAC t-stat |
|---|---|---|---|---|
| 0 bps (gross) | +5.10% | 4.53% | **1.13** | 7.2 |
| 5 bps | +4.03% | 4.53% | 0.89 | 5.6 |
| 10 bps | +2.95% | 4.54% | 0.65 | 4.1 |
| 20 bps | +0.80% | 4.56% | 0.18 | 1.1 |
| long-only, vol-targeted, 10 bps | +1.32% | 5.50% | 0.24 | 1.5 |

The gross Sharpe of 1.13 replicates MOP's ">1 diversified" result on free
data; the collapse from 1.13 to 0.18 as costs rise from 0 to 20 bps is the
cost sensitivity Frazzini-Israel-Moskowitz warn about; and the decade
breakdown reproduces the well-documented post-2010 decay of classic trend
(1980s Sharpe 1.67 → 2010s −0.53), which is precisely the regime the
Oxford-Man deep-learning papers were written to address.

### 2.2 Volatility forecasting (`examples/hedge_fund/volatility_forecasting.py`)

One-week-ahead variance, QLIKE loss
([Patton, J. Econometrics 2011](https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf)-robust),
walk-forward with 3-year contexts, scored against RiskMetrics EWMA
(the industry default) with the repo's Diebold-Mariano + Holm machinery:

| Asset | HAR vs RiskMetrics (QLIKE ratio) | verdict |
|---|---|---|
| NASDAQ | 0.960 | statistical tie (p=0.056) |
| EURUSD | 1.024 | tie |
| WTI | 1.009 | tie |
| UST10Y | 1.112 | worse |

Two honest lessons the harness surfaces: (a) with only daily squared-return
proxies, [HAR](https://academic.oup.com/jfec/article-abstract/7/2/174/856522)
cannot beat a well-tuned EWMA — its documented edge needs intraday realized
variance; (b) naive level-forecasters applied to variance (last-value, raw
AR) lose by factors of 10³ because QLIKE brutally punishes under-forecasts.
Any model claiming a volatility edge must clear this exact bar — plug it in
as a `Baseline` and the DM test adjudicates.

**Why volatility matters for P&L** — the
[Moreira-Muir (JF 2017)](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12513)
overlay, NASDAQ 1971–2026: scaling exposure by inverse forecast variance
takes buy-and-hold from Sharpe 0.49 to **0.71** at lower realized vol
(20.1% → 15.8%), with no return forecast anywhere in the loop. This is the
single most reliable "application of a forecaster" in the equity literature,
and it is a *scale* forecast, not a direction forecast.

### 2.3 Forecaster-driven positions (`examples/hedge_fund/model_signal.py`)

The Deep-Momentum-Networks harness: any forecaster with the repo's
`forecast(context, horizon)` interface sizes positions via its
standardized predicted move inside the same vol-targeting wrapper —
identical sizing, costs, and statistics for classical and neural models,
so the table isolates forecast quality. Net of 10 bps, 1975–2026:

| Forecaster | Ann. return | Sharpe | Interpretation |
|---|---|---|---|
| drift (trend extrapolation) | +2.19% | **+1.25** | markets trend |
| AR(4) (mean reversion) | −2.78% | −1.44 | ...and punish mean-reversion at this horizon |
| EWMA (flat forecast, placebo) | −0.03% | −0.87 | harness conjures no P&L from sizing alone |

The placebo row is the important one: a forecaster that predicts "no move"
holds ~zero positions and earns ~zero — the pipeline has no hidden long
bias. Sharpe differences between rows are attributable to the forecasts.

### 2.4 The TimesFM-3 path (`examples/hedge_fund/pretrain_markets.py`)

[Rahimikia et al.](https://arxiv.org/abs/2511.18578) (Man Group-affiliated)
find generic-pretrained foundation models underperform gradient-boosting
baselines on daily returns, while financially-pretrained versions of the
same architectures deliver forecasting and portfolio gains; independent
evaluations of zero-shot TimesFM-class models on equities agree the gains
are ["small and sparse"](https://arxiv.org/abs/2606.27100). The pipeline
here follows the paper's recipe rather than the zero-shot shortcut:

1. `pretrain_markets.py` pre-trains a TimesFM-3 config on the FRED panel
   (log-price levels via `to_real_source`) mixed 70/30 with the synthetic
   corpus, with windows drawn **only from the first 80% of history** —
   evaluation on recent years is out-of-sample in time.
2. `model_signal.py <checkpoint>` runs the trained model through the
   identical harness as §2.3 via `TimesFM3Signal`, which also exposes the
   quantile head (size longs off q25, shorts off q75, for a risk-averse
   variant — the distributional sizing that
   [Kelly-style position sizing](https://gwern.net/doc/statistics/decision/2006-thorp.pdf)
   needs and point forecasters cannot provide).

Do not expect a tiny CPU-trained checkpoint to beat `drift`; the point of
the harness is that when you train a real one (the `small`/`base` configs
on a GPU, more data via additional `Instrument`s), the comparison against
every classical baseline — same costs, same DM tests — is one command.

---

## 3. Limitations, stated plainly

- **FRED spot series are not tradable futures.** Spot FX ignores carry
  (interest-rate differentials), spot oil ignores roll yield, and the
  Treasury proxies are duration-approximated from par yields (no roll-down,
  no financing). Signal *research* on these series is meaningful;
  P&L levels would shift with real futures data.
- **Survivorship/lookahead**: FRED archives are point-final, not
  point-in-time (revisions are overwritten). For daily market prices this
  is minor; for anything macro it would not be.
- **No intraday data** — which is exactly why HAR ties rather than beats
  RiskMetrics here (§2.2).
- **Costs are proportional-only** (no market impact); MOP-style weekly
  turnover keeps this defensible at these leverages, per
  Frazzini-Israel-Moskowitz.
- **The drawdown convention is log-space**, so "−151%" for NASDAQ
  buy-and-hold is the dot-com −78% expressed in log units.
- Nothing here is investment advice; it is a research harness with honest
  statistics.
