/**
 * Pure-JS port of timesfm3's classical baselines, empirical quantile bands and
 * comparison statistics.  Runs identically in a Cloudflare Worker (the free
 * tier's 10 ms CPU budget is plenty) and in the browser.  Mirrors
 * timesfm3/baselines.py, timesfm3/serving/registry.py and
 * timesfm3/evaluation.py; the Python test-suite is the reference.
 */

export const QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

/* ---------- small numerics ---------- */

export function mean(a) { let s = 0; for (const v of a) s += v; return a.length ? s / a.length : NaN; }
function std(a, ddof = 0) { const m = mean(a); let s = 0; for (const v of a) s += (v - m) ** 2; return Math.sqrt(s / Math.max(1, a.length - ddof)); }
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

/** numpy.quantile default (linear, "type 7"). */
export function quantile(sorted, q) {
  const n = sorted.length;
  if (!n) return NaN;
  const pos = (n - 1) * q, lo = Math.floor(pos), hi = Math.min(lo + 1, n - 1), w = pos - lo;
  return sorted[lo] * (1 - w) + sorted[hi] * w;
}
function quantiles(values, levels) { const s = [...values].sort((a, b) => a - b); return levels.map((q) => quantile(s, q)); }

/** Acklam's inverse-normal CDF (rel. error < 1.2e-9). */
export function normalPpf(p) {
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.383577518672690e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  const pl = 0.02425, ph = 1 - pl;
  let q, r;
  if (p < pl) { q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  if (p > ph) { q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  q = p - 0.5; r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/** Complementary error function (Numerical Recipes erfcc, |err| < 1.2e-7). */
export function erfc(x) {
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}

/** Deterministic PRNG (mulberry32) so bootstrap CIs are reproducible. */
export function rng(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

/** Solve A x = b (small dense, Gaussian elimination with partial pivoting). */
function solve(A, b) {
  const n = b.length, M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (Math.abs(M[p][c]) < 1e-300) return null;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) { let s = M[r][n]; for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k]; x[r] = s / M[r][r]; }
  return x.every(Number.isFinite) ? x : null;
}

/* ---------- baselines (timesfm3/baselines.py) ---------- */

const EWMA_ALPHAS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0];

function ewmaLevel(ctx, alpha) { const out = new Array(ctx.length); let level = ctx[0]; for (let i = 0; i < ctx.length; i++) { level = alpha * ctx[i] + (1 - alpha) * level; out[i] = level; } return out; }

export const BASELINES = {
  "last-value": (ctx, h) => new Array(h).fill(ctx[ctx.length - 1]),
  "ctx-mean": (ctx, h) => new Array(h).fill(mean(ctx)),
  drift: (ctx, h) => { const n = ctx.length; if (n < 2) return new Array(h).fill(ctx[n - 1]); const slope = (ctx[n - 1] - ctx[0]) / (n - 1); return Array.from({ length: h }, (_, i) => ctx[n - 1] + slope * (i + 1)); },
  ewma: (ctx, h) => {
    const n = ctx.length;
    if (n < 3) return new Array(h).fill(ctx[n - 1]);
    // One pass for every candidate alpha: level_k predicts ctx[i+1].
    const K = EWMA_ALPHAS.length, level = new Float64Array(K).fill(ctx[0]), sse = new Float64Array(K);
    for (let i = 0; i < n; i++) {
      const x = ctx[i];
      if (i > 0) for (let k = 0; k < K; k++) { const e = x - level[k]; sse[k] += e * e; }
      for (let k = 0; k < K; k++) level[k] = EWMA_ALPHAS[k] * x + (1 - EWMA_ALPHAS[k]) * level[k];
    }
    let best = K - 1;
    for (let k = 0; k < K; k++) if (sse[k] < sse[best]) best = k;
    return new Array(h).fill(level[best]);
  },
  ar1: (ctx, h) => ar(ctx, h, 1),
  ar4: (ctx, h) => ar(ctx, h, 4),
};

export const BASELINE_INFO = {
  "last-value": "Random walk: holds the last observation flat.",
  "ctx-mean": "Holds the context mean flat.",
  drift: "Random walk with drift: extrapolates the context slope.",
  ewma: "Exponentially weighted level, smoothing fit in-context.",
  ar1: "AR(1) fit by OLS on the context, iterated forward.",
  ar4: "AR(4) fit by OLS on the context, iterated forward.",
};

function ar(ctx, h, p, ridge = 1e-8) {
  const n = ctx.length;
  const fallback = () => new Array(h).fill(ctx[n - 1]);
  if (n < 3 * p + 2) return fallback();
  let sum = 0, lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n; i++) { const v = ctx[i]; sum += v; if (v < lo) lo = v; if (v > hi) hi = v; }
  const mu = sum / n;
  if (hi === lo) return fallback(); // constant context: std == 0
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = ctx[i] - mu;
  const gram = new Float64Array(p * p), rhs = new Float64Array(p);
  for (let t = p; t < n; t++) {
    const y = x[t];
    for (let a = 0; a < p; a++) {
      const xa = x[t - a - 1];
      rhs[a] += xa * y;
      const row = a * p;
      for (let b2 = a; b2 < p; b2++) gram[row + b2] += xa * x[t - b2 - 1];
    }
  }
  const A = [];
  for (let a = 0; a < p; a++) { const row = new Array(p); for (let b2 = 0; b2 < p; b2++) row[b2] = a <= b2 ? gram[a * p + b2] : gram[b2 * p + a]; row[a] += ridge; A.push(row); }
  const coef = solve(A, Array.from(rhs));
  if (!coef) return fallback();
  const hist = new Float64Array(p);
  for (let k = 0; k < p; k++) hist[k] = x[n - p + k]; // oldest..newest
  const out = new Array(h), span = hi - lo;
  for (let i = 0; i < h; i++) {
    let nxt = 0;
    for (let k = 0; k < p; k++) nxt += coef[k] * hist[p - 1 - k];
    out[i] = Math.min(Math.max(nxt + mu, lo - span), hi + span);
    for (let k = 1; k < p; k++) hist[k - 1] = hist[k];
    hist[p - 1] = nxt;
  }
  return out;
}

/* ---------- missing values + empirical bands (registry.py) ---------- */

export function fillMissing(x) {
  const out = x.map((v) => (v === null || v === undefined || !Number.isFinite(v) ? NaN : v));
  const n = out.length; let first = -1;
  for (let i = 0; i < n; i++) if (!Number.isNaN(out[i])) { first = i; break; }
  if (first < 0) return new Array(n).fill(0);
  for (let i = 0; i < first; i++) out[i] = out[first];
  for (let i = first + 1; i < n; i++) if (Number.isNaN(out[i])) out[i] = out[i - 1];
  return out;
}

export function empiricalQuantiles(baseline, ctx, horizon, point, levels = QUANTILE_LEVELS, maxOrigins = 16) {
  const n = ctx.length, minCtx = Math.max(8, Math.floor(n / 4));
  const errors = [];
  if (n - minCtx >= 2) {
    const k = Math.min(maxOrigins, n - minCtx);
    const origins = [...new Set(Array.from({ length: k }, (_, i) => Math.floor(minCtx + ((n - 1 - minCtx) * i) / Math.max(1, k - 1))))].sort((a, b) => a - b);
    for (const o of origins) {
      const h = Math.min(horizon, n - o), fc = baseline(ctx.slice(0, o), h), row = new Array(horizon).fill(NaN);
      for (let j = 0; j < h; j++) row[j] = ctx[o + j] - fc[j];
      errors.push(row);
    }
  }
  const one = errors.map((r) => r[0]).filter(Number.isFinite);
  let sigma1;
  if (one.length >= 2) { const s = std(one); sigma1 = s > 0 ? s : mean(one.map(Math.abs)); }
  else { const d = []; for (let i = 1; i < n; i++) d.push(ctx[i] - ctx[i - 1]); sigma1 = d.length > 1 ? std(d) : mean(ctx.map(Math.abs)) * 0.1; }
  sigma1 = Math.max(sigma1, 1e-6 * Math.max(1, mean(ctx.map(Math.abs))));
  const z = levels.map(normalPpf);
  const out = [];
  for (let j = 0; j < horizon; j++) {
    const gauss = z.map((zz) => zz * sigma1 * Math.sqrt(j + 1));
    const e = errors.map((r) => r[j]).filter(Number.isFinite);
    let offsets;
    if (e.length >= 4) { const emp = quantiles(e, levels); offsets = emp.map((v, i) => (Math.abs(v) >= Math.abs(gauss[i]) ? v : gauss[i])); }
    else offsets = gauss;
    out.push(offsets.map((o) => point[j] + o).sort((a, b) => a - b));
  }
  return out;
}

/** Same result shape as the Python API: {point: [[...]], quantiles: [[[...]]]} */
export function forecastClassical(name, targets, horizon, withQuantiles = true) {
  const baseline = BASELINES[name];
  if (!baseline) throw new Error(`Unknown classical model ${name}`);
  if (!(horizon > 0)) throw new Error("horizon must be positive.");
  const point = [], quants = [];
  for (const series of targets) {
    const ctx = fillMissing(series);
    if (ctx.length < 2) throw new Error("Each target needs at least 2 observations.");
    const p = baseline(ctx, horizon);
    point.push(p);
    if (withQuantiles) quants.push(empiricalQuantiles(baseline, ctx, horizon, p));
  }
  return { point, quantiles: withQuantiles ? quants : null, quantile_levels: QUANTILE_LEVELS };
}

/* ---------- comparison statistics (evaluation.py) ---------- */

export function neweyWestVariance(x, lags = null) {
  const n = x.length; if (n < 2) return NaN;
  if (lags === null) lags = Math.max(1, Math.round(Math.cbrt(n)));
  lags = Math.min(lags, n - 1);
  const m = mean(x), d = x.map((v) => v - m);
  let total = dot(d, d) / n;
  for (let lag = 1; lag <= lags; lag++) { let g = 0; for (let i = lag; i < n; i++) g += d[i] * d[i - lag]; total += 2 * (1 - lag / (lags + 1)) * (g / n); }
  return Math.max(total, 0) / n;
}

export function dieboldMariano(a, b) {
  const d = []; for (let i = 0; i < a.length; i++) { const v = a[i] - b[i]; if (Number.isFinite(v)) d.push(v); }
  if (d.length < 3) return [NaN, NaN];
  if (d.every((v) => Math.abs(v) < 1e-12)) return [0, 1];
  const v = neweyWestVariance(d);
  if (v <= 0) return [Math.sign(mean(d)) * Infinity, 0];
  const stat = mean(d) / Math.sqrt(v);
  return [stat, erfc(Math.abs(stat) / Math.SQRT2)];
}

export function effectiveSampleSize(x, maxLag = null) {
  x = x.filter(Number.isFinite); const n = x.length; if (n < 3) return n;
  const m = mean(x), d = x.map((v) => v - m), denom = dot(d, d); if (denom <= 0) return n;
  maxLag = maxLag || Math.min(n - 1, 50); let total = 0;
  for (let lag = 1; lag <= maxLag; lag++) { let g = 0; for (let i = lag; i < n; i++) g += d[i] * d[i - lag]; const rho = g / denom; if (rho <= 0) break; total += rho; }
  return n / (1 + 2 * total);
}

export function pairedBootstrap(a, b, groups = null, resamples = 1000, seed = 0) {
  const ok = a.map((v, i) => Number.isFinite(v) && Number.isFinite(b[i]));
  const A = a.filter((_, i) => ok[i]), B = b.filter((_, i) => ok[i]);
  if (!A.length || mean(B) === 0) return [NaN, NaN, NaN];
  const point = mean(A) / mean(B), random = rng(seed), n = A.length, ratios = [];
  if (groups === null) {
    for (let r = 0; r < resamples; r++) { let sa = 0, sb = 0; for (let i = 0; i < n; i++) { const k = Math.floor(random() * n); sa += A[k]; sb += B[k]; } if (sb !== 0) ratios.push(sa / sb); }
  } else {
    const g = groups.filter((_, i) => ok[i]), uniq = [...new Set(g)].sort((x, y) => x - y), idx = uniq.map((u) => g.map((v, i) => (v === u ? i : -1)).filter((i) => i >= 0));
    for (let r = 0; r < resamples; r++) { let sa = 0, sb = 0, cnt = 0; for (let j = 0; j < uniq.length; j++) { const pick = idx[Math.floor(random() * uniq.length)]; for (const i of pick) { sa += A[i]; sb += B[i]; cnt++; } } if (sb !== 0 && cnt) ratios.push(sa / sb); }
  }
  if (!ratios.length) return [point, NaN, NaN];
  const s = ratios.sort((x, y) => x - y);
  return [point, quantile(s, 0.025), quantile(s, 0.975)];
}

export function holm(pvals) {
  const items = Object.entries(pvals).filter(([, p]) => Number.isFinite(p)).sort((x, y) => x[1] - y[1]);
  const m = items.length, out = {}; let running = 0;
  items.forEach(([k, p], i) => { running = Math.max(running, Math.min(1, (m - i) * p)); out[k] = running; });
  for (const k of Object.keys(pvals)) if (!(k in out)) out[k] = NaN;
  return out;
}

export function compare(losses, reference, groups = null, resamples = 1000, seed = 0) {
  const ref = losses[reference], out = {}, raw = {};
  for (const [name, arr] of Object.entries(losses)) {
    if (name === reference) continue;
    const [ratio, lo, hi] = pairedBootstrap(arr, ref, groups, resamples, seed);
    const [, p] = dieboldMariano(arr, ref);
    raw[name] = p;
    const okA = [], okR = []; arr.forEach((v, i) => { if (Number.isFinite(v) && Number.isFinite(ref[i])) { okA.push(v); okR.push(ref[i]); } });
    let wins = 0; okA.forEach((v, i) => { if (v < okR[i]) wins++; });
    out[name] = { name, reference, mean_loss: okA.length ? mean(okA) : NaN, reference_loss: okA.length ? mean(okR) : NaN, ratio, ci_low: lo, ci_high: hi, p_value: p, win_rate: okA.length ? wins / okA.length : NaN, n: okA.length, n_effective: effectiveSampleSize(okA.map((v, i) => v - okR[i])), p_adjusted: null };
  }
  const adj = holm(raw);
  for (const name of Object.keys(out)) { out[name].p_adjusted = adj[name]; const p = Number.isFinite(adj[name]) ? adj[name] : out[name].p_value; out[name].significant = p < 0.05; out[name].verdict = !out[name].significant ? "no difference" : out[name].ratio < 1 ? "better" : "worse"; }
  return out;
}

/* ---------- walk-forward backtest (serving/app.py run_backtest) ---------- */

/**
 * forecastFn(modelName, contextArray, horizon) -> Promise<number[]> point path.
 * Returns the same report shape as the Python /v1/backtest endpoint.
 */
export async function runBacktest(forecastFn, series, context, horizon, models, reference, windows, metric = "mae", overlap = false, resamples = 1000) {
  models = [...models]; if (!models.includes(reference)) models.push(reference);
  const span = context + horizon, losses = Object.fromEntries(models.map((m) => [m, []])), groups = [];
  let nWindows = 0;
  for (let g = 0; g < series.length; g++) {
    const x = series[g].map((v) => (v === null || v === undefined ? NaN : v)), n = x.length;
    if (n < span + 1) throw new Error(`Series ${g} has ${n} steps; need at least context + horizon + 1 = ${span + 1}.`);
    const latest = n - span;
    let starts;
    if (overlap) { const k = Math.min(windows, latest + 1); starts = [...new Set(Array.from({ length: k }, (_, i) => Math.floor((latest * i) / Math.max(1, k - 1))))]; }
    else { const k = Math.min(windows, Math.floor(latest / horizon) + 1); starts = Array.from({ length: k }, (_, i) => latest - horizon * (k - 1 - i)); }
    nWindows = Math.max(nWindows, starts.length);
    for (const s of starts) {
      const ctx = x.slice(s, s + context), truth = x.slice(s + context, s + span);
      if (!truth.some(Number.isFinite) || ctx.filter(Number.isFinite).length < 2) continue;
      for (const m of models) {
        const r = forecastFn(m, ctx, horizon);
        const pred = r && typeof r.then === "function" ? await r : r;
        let acc = 0, cnt = 0;
        for (let j = 0; j < horizon; j++) { const e = pred[j] - truth[j]; if (Number.isFinite(e)) { acc += metric === "mse" ? e * e : Math.abs(e); cnt++; } }
        losses[m].push(cnt ? acc / cnt : NaN);
      }
      groups.push(g);
    }
  }
  if (!groups.length) throw new Error("No scorable windows (all-NaN targets?).");
  const cmp = compare(losses, reference, series.length > 1 ? groups : null, resamples);
  const refLoss = mean(losses[reference]);
  const fin = (v) => (Number.isFinite(v) ? v : null);
  const scores = models.map((m) => m === reference
    ? { model: m, mean_loss: refLoss, ratio: 1, ci_low: null, ci_high: null, p_value: null, p_adjusted: null, win_rate: 0, n: losses[m].length, n_effective: losses[m].length, verdict: "reference" }
    : { model: m, mean_loss: cmp[m].mean_loss, ratio: cmp[m].ratio, ci_low: fin(cmp[m].ci_low), ci_high: fin(cmp[m].ci_high), p_value: fin(cmp[m].p_value), p_adjusted: fin(cmp[m].p_adjusted), win_rate: cmp[m].win_rate, n: cmp[m].n, n_effective: cmp[m].n_effective, verdict: cmp[m].verdict });
  scores.sort((a, b) => a.ratio - b.ratio);
  return { reference, reference_loss: refLoss, metric, context, horizon, windows_per_series: nWindows, scores };
}

/* ---------- anomaly scoring (timesfm3/anomaly.py) ---------- */

export async function detectAnomalies(forecastFn, values, context = 96, block = 24, threshold = 2.0) {
  const x = values.map((v) => (v === null || v === undefined ? NaN : v)), n = x.length;
  if (context < 8) throw new Error("context must be at least 8 steps.");
  if (n < context + 1) throw new Error(`Series has ${n} steps; need more than the context of ${context}.`);
  const scores = new Array(n).fill(NaN), expected = new Array(n).fill(NaN), lower = new Array(n).fill(NaN), upper = new Array(n).fill(NaN);
  let origin = context;
  while (origin < n) {
    const h = Math.min(block, n - origin), ctx = x.slice(origin - context, origin);
    if (ctx.filter(Number.isFinite).length < 2) { origin += h; continue; }
    const r = await forecastFn(ctx, h); // {point:[h], quantiles:[h][Q]}
    for (let j = 0; j < h; j++) {
      const q = r.quantiles[j], med = q[4], lo = q[0], hi = q[8], obs = x[origin + j];
      expected[origin + j] = med; lower[origin + j] = lo; upper[origin + j] = hi;
      if (Number.isFinite(obs)) scores[origin + j] = obs >= med ? (obs - med) / Math.max(hi - med, 1e-12) : (med - obs) / Math.max(med - lo, 1e-12);
    }
    origin += h;
  }
  const anomalies = [];
  for (let i = 0; i < n; i++) if (Number.isFinite(scores[i]) && scores[i] > threshold) anomalies.push({ index: i, value: x[i], expected: expected[i], lower: lower[i], upper: upper[i], score: scores[i], direction: x[i] > expected[i] ? "high" : "low" });
  return { scores, expected, lower, upper, anomalies, n_scored: scores.filter(Number.isFinite).length };
}
