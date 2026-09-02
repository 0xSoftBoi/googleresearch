/**
 * TimesFM-3 Forecast Service — Cloudflare edge, free tier.
 *
 * Two modes, chosen by whether API_ORIGIN is set:
 *
 *  edge-native (default, $0): everything runs on Cloudflare's free tier and
 *    in the visitor's browser.  The Worker serves the site, the classical
 *    forecasters (`/v1/forecast`, `/v1/backtest`, `/v1/anomalies`) in pure JS
 *    inside the 10 ms CPU budget, and the demo data; the TimesFM-3 model runs
 *    in the browser at /app through ONNX Runtime Web from a static asset.
 *
 *  gateway: with API_ORIGIN pointing at a self-hosted `timesfm3 serve`, the
 *    Worker proxies `/v1/*`, `/healthz`, `/docs` there with the upstream key
 *    attached server-side (bring-your-own keys pass through), and caches the
 *    cheap GETs at the edge.
 *
 * Both modes: waitlist capture in KV (`POST /api/waitlist`), founder-only
 * `GET /api/leads`, per-IP rate limiting through the Workers rate-limit
 * binding (no KV writes, so the free tier's 1,000 writes/day go to leads).
 */

import * as F from "../public/js/forecast.js";
import { describe as describeX402, finalize as x402Finalize, requirePayment, x402Config } from "./x402.js";

const VERSION = "0.5.0";
const PROXY_PREFIXES = ["/v1/", "/healthz", "/docs", "/openapi.json", "/redoc"];
const CACHEABLE = new Set(["/v1/models", "/v1/sample", "/healthz"]);
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;
const LIMITS = { series: 32, context: 4096, horizon: 512, btSeries: 4, btContext: 1024, btHorizon: 128, btWindows: 10, anomalySeries: 4, anomalySteps: 4096 };

let modelMeta = null; // cached starter-small.json

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const path = url.pathname;
    if (request.method === "OPTIONS") return cors(new Response(null, { status: 204 }), request, env);
    if (path === "/metrics") return json({ detail: "Not exposed at the edge." }, 404);
    try {
      if (path === "/api/waitlist" && request.method === "POST") return await waitlist(request, env, ctx);
      if (path === "/api/leads" && request.method === "GET") return await leads(request, env);
      if (path === "/api/edge") return json(edgeInfo(request, env));
      if (path === "/" || path === "/index.html") return withSecurityHeaders(await asset(env, request, "/index.html"));
      if (path === "/app" || path === "/app/") return withSecurityHeaders(await asset(env, request, "/app/index.html"));
      if (path === "/favicon.ico") return asset(env, request, "/favicon.svg");

      const gateway = Boolean((env.API_ORIGIN || "").trim());
      if (path === "/v1/pricing") return cors(json(pricing(env, gateway)), request, env);
      if (PROXY_PREFIXES.some((p) => path === p || path.startsWith(p))) {
        const limited = await rateLimit(request, env);
        if (limited) return cors(limited, request, env);
        // x402: anonymous callers pay per call; bring-your-own-key callers are
        // metered upstream instead (gateway mode only, where keys are validated).
        const cfg = x402Config(env);
        const hasKey = Boolean(request.headers.get("x-api-key") || bearer(request.headers.get("authorization")));
        const paywall = cfg && (gateway ? !hasKey : env.X402_PAYWALL_EDGE_NATIVE === "1");
        let payment = null;
        if (paywall) {
          const gate = await requirePayment(cfg, request, path);
          if (gate && gate.response) return cors(gate.response, request, env);
          if (gate) payment = gate.payment;
        }
        let response = gateway ? await proxy(request, env, ctx, path + url.search) : cors(await edgeApi(request, env, url), request, env);
        if (payment) response = cors(await x402Finalize(cfg, payment, response), request, env);
        return response;
      }
      return withSecurityHeaders(await env.ASSETS.fetch(request));
    } catch (err) {
      if (err instanceof HttpError) return cors(json({ detail: err.message }, err.status), request, env);
      console.error("edge error", err);
      return json({ detail: "Edge error: " + (err && err.message ? err.message : String(err)) }, 502);
    }
  },
};

/* ---------- edge-native API (classical models in JS) ---------- */

async function edgeApi(request, env, url) {
  const path = url.pathname;
  const t0 = Date.now();
  if (path === "/healthz") return json({ status: "ok", version: VERSION + "-edge", models: (await models(env, request)).length, default_model: "ewma", device: "cloudflare-edge + browser-wasm" });
  if (path === "/docs" || path === "/docs/") return withSecurityHeaders(await asset(env, request, "/docs/index.html"));
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
  await models(env, request); // ensures modelMeta is loaded

  if (path === "/v1/forecast") {
    const targets = seriesList(body.targets, "targets", 1);
    const horizon = int(body.horizon, 1, LIMITS.horizon, "horizon");
    const name = body.model || "ewma";
    if (name === modelMeta.name) return json({ detail: `${name} runs in the browser on the free edge: open /app, or self-host the service for a server-side API.` }, 400);
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
      if (m === modelMeta.name) return json({ detail: `${m} backtests run in the browser at /app on the free edge.` }, 400);
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
    if (name === modelMeta.name) return json({ detail: `${name} anomaly scoring runs in the browser at /app on the free edge.` }, 400);
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

/* ---------- gateway mode ---------- */

async function proxy(request, env, ctx, upstreamPath) {
  const origin = env.API_ORIGIN.replace(/\/+$/, "");
  const isGet = request.method === "GET" || request.method === "HEAD";
  const maxBody = Number(env.MAX_BODY_BYTES || 1048576);
  if (Number(request.headers.get("content-length") || 0) > maxBody) return cors(json({ detail: `Request body over ${maxBody} bytes.` }, 413), request, env);
  const upstreamUrl = origin + upstreamPath;
  const cacheKey = new Request(upstreamUrl, { method: "GET" });
  const cache = caches.default;
  const cacheable = isGet && CACHEABLE.has(new URL(upstreamUrl).pathname);
  if (cacheable) { const hit = await cache.match(cacheKey); if (hit) return cors(withHeader(hit, "x-edge-cache", "HIT"), request, env); }
  const headers = new Headers(request.headers);
  headers.delete("host"); headers.delete("cookie");
  headers.delete("payment-signature"); headers.delete("x-payment");
  const clientKey = headers.get("x-api-key") || bearer(headers.get("authorization"));
  if (!clientKey && env.API_KEY) headers.set("x-api-key", env.API_KEY);
  headers.set("x-forwarded-for", clientIp(request));
  headers.set("x-edge-worker", "timesfm3-edge");
  const init = { method: request.method, headers, redirect: "manual" };
  if (!isGet) init.body = request.body;
  let response = await fetch(upstreamUrl, init);
  response = new Response(response.body, response);
  response.headers.set("x-edge-cache", "MISS");
  response.headers.delete("server");
  if (cacheable && response.ok) { const toCache = new Response(response.clone().body, response); toCache.headers.set("cache-control", "public, s-maxage=60"); ctx.waitUntil(cache.put(cacheKey, toCache)); }
  const ct = response.headers.get("content-type") || "";
  return cors(ct.includes("text/html") ? withSecurityHeaders(response) : response, request, env);
}

/* ---------- rate limiting (Workers rate-limit binding; no KV writes) ---------- */

async function rateLimit(request, env) {
  if (!env.RATE_LIMITER) return null;
  const { success } = await env.RATE_LIMITER.limit({ key: clientIp(request) });
  if (success) return null;
  const r = json({ detail: `Rate limit of ${env.RATE_LIMIT_PER_MINUTE || 30} requests/minute exceeded.` }, 429);
  r.headers.set("retry-after", "60");
  r.headers.set("x-ratelimit-limit", String(env.RATE_LIMIT_PER_MINUTE || 30));
  r.headers.set("x-ratelimit-remaining", "0");
  return r;
}

/* ---------- leads ---------- */

async function waitlist(request, env, ctx) {
  let body;
  try { body = await request.json(); } catch { return cors(json({ detail: "Send JSON: {email, company?, plan?, use_case?}." }, 400), request, env); }
  if (body.website) return cors(json({ ok: true }), request, env); // honeypot
  const email = String(body.email || "").trim().toLowerCase();
  if (!EMAIL_RE.test(email) || email.length > 254) return cors(json({ detail: "A valid email address is required." }, 400), request, env);
  if (!env.EDGE_KV) return cors(json({ detail: "Lead storage not configured." }, 503), request, env);
  const lead = { email, company: clip(body.company, 120), plan: clip(body.plan, 40) || "unspecified", use_case: clip(body.use_case, 1000), source: clip(body.source, 80) || "landing", country: request.cf && request.cf.country ? request.cf.country : null, ip_hash: await sha256(clientIp(request)), created: new Date().toISOString() };
  const existing = await env.EDGE_KV.get(`lead-email:${email}`);
  const id = existing || `${lead.created}:${(await sha256(email)).slice(0, 12)}`;
  if (existing) {
    const prev = await env.EDGE_KV.get(`lead:${existing}`);
    if (prev) { const p = JSON.parse(prev); lead.created = p.created || lead.created; lead.updated = new Date().toISOString(); lead.signups = (p.signups || 1) + 1; for (const k of ["company", "use_case"]) if (!lead[k] && p[k]) lead[k] = p[k]; }
  }
  await env.EDGE_KV.put(`lead:${id}`, JSON.stringify(lead));
  await env.EDGE_KV.put(`lead-email:${email}`, id);
  if (env.LEADS_WEBHOOK) ctx.waitUntil(fetch(env.LEADS_WEBHOOK, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ text: `New TimesFM-3 lead: ${email} (${lead.plan}) ${lead.company || ""} ${lead.use_case ? "— " + lead.use_case : ""}`.trim(), content: `New TimesFM-3 lead: ${email} (${lead.plan})` }) }).catch((e) => console.error("webhook failed", e)));
  return cors(json({ ok: true, duplicate: Boolean(existing) }), request, env);
}

async function leads(request, env) {
  const token = bearer(request.headers.get("authorization"));
  if (!env.ADMIN_TOKEN || !token || token !== env.ADMIN_TOKEN) return json({ detail: "Admin token required." }, 401);
  const out = []; let cursor;
  do {
    const page = await env.EDGE_KV.list({ prefix: "lead:", cursor, limit: 1000 });
    for (const k of page.keys) { const v = await env.EDGE_KV.get(k.name); if (v) out.push(JSON.parse(v)); }
    cursor = page.list_complete ? undefined : page.cursor;
  } while (cursor);
  out.sort((a, b) => (a.created < b.created ? 1 : -1));
  return json({ count: out.length, leads: out });
}

/* ---------- helpers ---------- */

async function asset(env, request, path) {
  const u = new URL(request.url); u.pathname = path; u.search = "";
  return env.ASSETS.fetch(new Request(u.toString(), { method: "GET" }));
}
function edgeInfo(request, env) { return { worker: "timesfm3-edge", version: VERSION, mode: (env.API_ORIGIN || "").trim() ? "gateway" : "edge-native", upstream_key_configured: Boolean(env.API_KEY), rate_limiter: Boolean(env.RATE_LIMITER), x402: describeX402(x402Config(env)), colo: request.cf && request.cf.colo ? request.cf.colo : null, country: request.cf && request.cf.country ? request.cf.country : null }; }
function pricing(env, gateway) {
  const cfg = x402Config(env);
  const paywall = cfg && (gateway || env.X402_PAYWALL_EDGE_NATIVE === "1");
  return {
    free: gateway ? "GET endpoints; bring your own API key for metered plans" : "classical-model API at the edge, TimesFM-3 in the browser at /app",
    plans: gateway ? { how: "X-API-Key from the service operator; metered in forecast points; see /v1/usage" } : null,
    x402: paywall ? describeX402(cfg) : { enabled: false, note: cfg ? "configured but not enforced in edge-native mode" : "set X402_PAY_TO to enable" },
  };
}
function clientIp(request) { return request.headers.get("cf-connecting-ip") || request.headers.get("x-forwarded-for") || "0.0.0.0"; }
function bearer(v) { if (!v) return null; const m = /^bearer\s+(.+)$/i.exec(v.trim()); return m ? m[1].trim() : null; }
function clip(v, n) { return v === undefined || v === null ? "" : String(v).trim().slice(0, n); }
async function sha256(text) { const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)); return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join(""); }
function json(obj, status = 200) { return new Response(JSON.stringify(obj), { status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } }); }
function withHeader(response, name, value) { const r = new Response(response.body, response); r.headers.set(name, value); return r; }
function withSecurityHeaders(response) { const r = new Response(response.body, response); r.headers.set("x-content-type-options", "nosniff"); r.headers.set("x-frame-options", "DENY"); r.headers.set("referrer-policy", "strict-origin-when-cross-origin"); return r; }
function cors(response, request, env) {
  const allowed = (env.ALLOWED_ORIGINS || "*").split(",").map((s) => s.trim()).filter(Boolean);
  const origin = request.headers.get("origin");
  const r = new Response(response.body, response);
  if (allowed.includes("*")) r.headers.set("access-control-allow-origin", "*");
  else if (origin && allowed.includes(origin)) { r.headers.set("access-control-allow-origin", origin); r.headers.set("vary", "origin"); }
  r.headers.set("access-control-allow-methods", "GET, POST, OPTIONS");
  r.headers.set("access-control-allow-headers", "content-type, x-api-key, authorization, payment-signature, x-payment");
  r.headers.set("access-control-expose-headers", "x-usage-points, x-usage-remaining, x-usage-paid, x-edge-cache, x-ratelimit-limit, x-ratelimit-remaining, retry-after, payment-required, payment-response");
  r.headers.set("access-control-max-age", "86400");
  return r;
}
