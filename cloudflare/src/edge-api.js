/** Edge-native API: the classical forecasters in JS, model inventory, demo data, static docs. */
import * as F from "../public/js/forecast.js";

let modelMeta = null; // cached starter-small.json
const LIMITS = { series: 32, context: 4096, horizon: 512, btSeries: 4, btContext: 1024, btHorizon: 128, btWindows: 10, anomalySeries: 4, anomalySteps: 4096 };

export async function edgeApi(c, opts) {
  try { return await edgeApiInner(c, opts); } catch (err) { if (err instanceof HttpError) return json({ detail: err.message }, err.status); throw err; }
}

async function edgeApiInner(c, { version }) {
  const request = c.req.raw, env = c.env, url = new URL(request.url);
  const path = url.pathname;
  const t0 = Date.now();
  if (path === "/healthz") return json({ status: "ok", version: version + "-edge", models: (await models(env, request)).length, default_model: "ewma", device: "cloudflare-edge + browser-wasm" });
  if (path === "/docs" || path === "/docs/") return asset(env, request, "/docs/index.html");
  if (path === "/openapi.json" || path === "/redoc") return json({ detail: "OpenAPI is served by the self-hosted service; see /docs." }, 404);
  if (path === "/v1/models" && request.method === "GET") return json(await models(env, request));
  if (path === "/v1/sample" && request.method === "GET") {
    const n = Math.max(2, Math.min(2000, Number(url.searchParams.get("n") || 336)));
    const doc = await (await asset(env, request, "/data/sample.json")).json();
    return json({ source: doc.source, names: doc.names, timestamps: doc.timestamps.slice(-n), values: doc.values.map((v) => v.slice(-n)) });
  }
  if (path === "/v1/usage") return json({ name: "anonymous", plan: "edge-free", month: new Date().toISOString().slice(0, 7), points_used: 0, requests: 0, monthly_quota: null, points_remaining: null, note: "Per-key metering runs in the self-hosted service." });
  if (request.method !== "POST") return json({ detail: "Not found." }, 404);
  let body;
  try { body = await request.json(); } catch { return json({ detail: "Body must be JSON." }, 422); }
  await models(env, request);

  if (path === "/v1/forecast") {
    const targets = seriesList(body.targets, "targets", 1);
    const horizon = int(body.horizon, 1, LIMITS.horizon, "horizon");
    const name = body.model || "ewma";
    if (modelMeta && name === modelMeta.name) return json({ detail: `${name} runs in the browser on the free edge: open /app, or self-host the service for a server-side API.` }, 400);
    if (!F.BASELINES[name]) return json({ detail: `Unknown model '${name}'; available: ${Object.keys(F.BASELINES).join(", ")}` }, 404);
    if ((body.past_covariates && body.past_covariates.length) || (body.future_covariates && body.future_covariates.length)) return json({ detail: `Model '${name}' does not accept covariates; use the TimesFM-3 model at /app.` }, 400);
    if (targets.length > LIMITS.series) return json({ detail: `At most ${LIMITS.series} series per request.` }, 413);
    const context = targets[0].values.length;
    if (context > LIMITS.context) return json({ detail: `Context is capped at ${LIMITS.context} steps on the free edge.` }, 413);
    if (body.timestamps && body.timestamps.length !== context) return json({ detail: "timestamps must have one entry per context step." }, 400);
    let r;
    try { r = F.forecastClassical(name, targets.map((s) => s.values), horizon, body.quantiles !== false); } catch (e) { return json({ detail: e.message }, 400); }
    const keys = F.QUANTILE_LEVELS.map((q) => "q" + Math.round(q * 100));
    const stamps = body.timestamps ? futureTimestamps(body.timestamps, horizon, body.freq) : null;
    const res = json({
      model: name, horizon, quantile_levels: F.QUANTILE_LEVELS, timestamps: stamps,
      forecasts: targets.map((s, i) => ({ name: s.name || `target_${i}`, point: r.point[i], quantiles: r.quantiles ? Object.fromEntries(keys.map((k, j) => [k, r.quantiles[i].map((row) => row[j])])) : null })),
      latency_ms: Date.now() - t0,
    });
    res.headers.set("x-usage-points", String(targets.length * horizon));
    return res;
  }

  if (path === "/v1/backtest") {
    const series = seriesList(body.series, "series", 1);
    const context = int(body.context, 8, LIMITS.btContext, "context");
    const horizon = int(body.horizon, 1, LIMITS.btHorizon, "horizon");
    const windows = int(body.windows ?? 20, 3, LIMITS.btWindows, "windows");
    const reference = body.reference || "last-value";
    const modelsReq = body.models && body.models.length ? body.models : Object.keys(F.BASELINES);
    for (const m of [...modelsReq, reference]) {
      if (modelMeta && m === modelMeta.name) return json({ detail: `${m} backtests run in the browser at /app on the free edge.` }, 400);
      if (!F.BASELINES[m]) return json({ detail: `Unknown model '${m}'.` }, 404);
    }
    if (series.length > LIMITS.btSeries) return json({ detail: `At most ${LIMITS.btSeries} series per backtest on the free edge (the browser dashboard at /app has no such limit).` }, 413);
    const fn = (m, ctxArr, h) => F.forecastClassical(m, [ctxArr], h, false).point[0];
    let report;
    // 200 bootstrap resamples keeps a full run inside the free plan's 10 ms CPU budget.
    try { report = await F.runBacktest(fn, series.map((s) => s.values), context, horizon, modelsReq, reference, windows, body.metric === "mse" ? "mse" : "mae", Boolean(body.overlap), 200); } catch (e) { return json({ detail: e.message }, 400); }
    const res = json({ ...report, latency_ms: Date.now() - t0 });
    res.headers.set("x-usage-points", String(series.length * windows * horizon * new Set([...modelsReq, reference]).size));
    return res;
  }

  if (path === "/v1/anomalies") {
    const series = seriesList(body.series, "series", 1);
    const name = body.model || "ewma";
    if (modelMeta && name === modelMeta.name) return json({ detail: `${name} anomaly scoring runs in the browser at /app on the free edge.` }, 400);
    if (!F.BASELINES[name]) return json({ detail: `Unknown model '${name}'.` }, 404);
    if (series.length > LIMITS.anomalySeries) return json({ detail: `At most ${LIMITS.anomalySeries} series per request on the free edge.` }, 413);
    const context = int(body.context ?? 96, 8, LIMITS.context, "context");
    const block = int(body.block ?? 24, 1, 512, "block");
    const threshold = Number(body.threshold ?? 2.0);
    if (!(threshold > 0)) return json({ detail: "threshold must be > 0." }, 422);
    const fn = async (ctxArr, h) => { const r = F.forecastClassical(name, [ctxArr], h, true); return { point: r.point[0], quantiles: r.quantiles[0] }; };
    const out = [];
    for (const [i, s] of series.entries()) {
      if (s.values.length > LIMITS.anomalySteps) return json({ detail: `Series are capped at ${LIMITS.anomalySteps} steps.` }, 413);
      let rep;
      try { rep = await F.detectAnomalies(fn, s.values, context, block, threshold); } catch (e) { return json({ detail: e.message }, 400); }
      const stamps = body.timestamps && body.timestamps.length === s.values.length ? body.timestamps : null;
      const fin = (v) => (Number.isFinite(v) ? v : null);
      out.push({ name: s.name || `series_${i}`, n_scored: rep.n_scored, n_flagged: rep.anomalies.length, anomalies: rep.anomalies.map((a) => ({ ...a, timestamp: stamps ? stamps[a.index] : null })), scores: body.include_scores ? rep.scores.map(fin) : null, expected: body.include_scores ? rep.expected.map(fin) : null, lower: body.include_scores ? rep.lower.map(fin) : null, upper: body.include_scores ? rep.upper.map(fin) : null });
    }
    return json({ model: name, context, block, threshold, series: out, latency_ms: Date.now() - t0 });
  }
  if (path === "/v1/volatility") return json({ detail: "Volatility forecasting runs in the self-hosted service (docker compose up)." }, 501);
  return json({ detail: "Not found." }, 404);
}

async function models(env, request) {
  if (!modelMeta) {
    try { modelMeta = await (await asset(env, request, "/models/starter-small.json")).json(); } catch { modelMeta = { name: "starter-small" }; }
  }
  const classical = Object.entries(F.BASELINE_INFO).map(([name, description]) => ({ name, kind: "classical", description, parameters: 0, supports_covariates: false, default: name === "ewma", meta: { runs_in: "edge" } }));
  const browser = { name: modelMeta.name || "starter-small", kind: "timesfm3-browser", description: (modelMeta.description || "TimesFM-3 starter model") + " Runs in your browser via ONNX Runtime (WebAssembly).", parameters: modelMeta.num_parameters || 0, supports_covariates: true, default: false, meta: { runs_in: "browser", onnx: "/models/starter-small.onnx", onnx_bytes: modelMeta.onnx_bytes || null, best_val_loss: modelMeta.best_val_loss || null, training_steps: modelMeta.training_steps || null, corpus: modelMeta.corpus || null, holdout: modelMeta.holdout || null } };
  return [...classical, browser];
}

function seriesList(list, field, min) {
  if (!Array.isArray(list) || list.length < min) throw new HttpError(422, `${field} must be a non-empty list of {name?, values}.`);
  return list.map((s, i) => {
    const values = Array.isArray(s && s.values) ? s.values.map((v) => (v === null || v === undefined || v === "" ? null : Number(v))) : null;
    if (!values || values.length < 2) throw new HttpError(422, `${field}[${i}].values needs at least 2 entries.`);
    if (values.some((v) => v !== null && !Number.isFinite(v))) throw new HttpError(422, `${field}[${i}].values must be numbers or null.`);
    if (!values.some((v) => v !== null)) throw new HttpError(422, `${field}[${i}] needs at least one non-null value.`);
    return { name: s.name ? String(s.name).slice(0, 128) : null, values };
  });
}

function int(v, lo, hi, name) {
  const n = Number(v);
  if (!Number.isInteger(n) || n < lo || n > hi) throw new HttpError(422, `${name} must be an integer in [${lo}, ${hi}].`);
  return n;
}

class HttpError extends Error { constructor(status, detail) { super(detail); this.status = status; } }

const UNITS = { s: 1e3, sec: 1e3, second: 1e3, seconds: 1e3, m: 6e4, min: 6e4, minute: 6e4, minutes: 6e4, t: 6e4, h: 36e5, hr: 36e5, hour: 36e5, hours: 36e5, d: 864e5, day: 864e5, days: 864e5, w: 6048e5, wk: 6048e5, week: 6048e5, weeks: 6048e5 };

function futureTimestamps(stamps, horizon, freq) {
  let stepMs = null;
  if (freq) { const m = /^\s*(\d+)?\s*([a-zA-Z]+)\s*$/.exec(freq); if (m && UNITS[m[2].toLowerCase()]) stepMs = (Number(m[1] || 1)) * UNITS[m[2].toLowerCase()]; }
  const times = stamps.slice(-64).map((s) => Date.parse(/[zZ]|[+-]\d\d:?\d\d$/.test(s) ? s : s + "Z"));
  if (times.some(Number.isNaN)) return null;
  if (stepMs === null) {
    if (times.length < 2) return null;
    const counts = new Map();
    for (let i = 1; i < times.length; i++) { const d = times[i] - times[i - 1]; counts.set(d, (counts.get(d) || 0) + 1); }
    stepMs = [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0];
    if (!(stepMs > 0)) return null;
  }
  const last = times[times.length - 1];
  return Array.from({ length: horizon }, (_, k) => new Date(last + (k + 1) * stepMs).toISOString().slice(0, 19));
}


async function asset(env, request, path) {
  const u = new URL(request.url); u.pathname = path; u.search = "";
  return env.ASSETS.fetch(new Request(u.toString(), { method: "GET" }));
}
function json(obj, status = 200) { return new Response(JSON.stringify(obj), { status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } }); }
