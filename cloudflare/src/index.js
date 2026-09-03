/**
 * TimesFM-3 Forecast Service — Cloudflare edge, built on Cloudflare's own pieces.
 *
 *   Hono                      routing, CORS
 *   @x402/hono                pay-per-call in USDC (official middleware)
 *   @cloudflare/privacypass-ts unlinkable prepaid tokens (RFC 9576/9577/9578)
 *   Workers static assets     landing page, dashboard, ONNX model
 *   D1                        Privacy Pass double-spend ledger
 *   Rate-limit binding        per-IP abuse brake
 *   KV                        waitlist leads
 *
 * Modes (by API_ORIGIN): edge-native (classical models here, TimesFM-3 in the
 * browser) or gateway (proxy to a self-hosted `timesfm3 serve`).
 */

import { Hono } from "hono";
import { cors } from "hono/cors";
import { paymentMiddleware, x402ResourceServer } from "@x402/hono";
import { HTTPFacilitatorClient } from "@x402/core/server";
import { ExactEvmScheme } from "@x402/evm/exact/server";

import * as F from "../public/js/forecast.js";
import { edgeApi } from "./edge-api.js";
import { PrivacyPassOrigin, privacyPassRoutes } from "./privacy-pass.js";
import { leadsRoutes } from "./leads.js";

const VERSION = "0.8.0";
const PRICED = { "POST /v1/forecast": "$0.005", "POST /v1/anomalies": "$0.01", "POST /v1/backtest": "$0.02", "POST /v1/volatility": "$0.005" };
const DESCRIPTIONS = {
  "POST /v1/forecast": "TimesFM-3 forecast: point + 9 quantiles per series and step",
  "POST /v1/anomalies": "Walk-forward anomaly scoring against the model's predictive band",
  "POST /v1/backtest": "Walk-forward model comparison with Diebold-Mariano tests",
  "POST /v1/volatility": "HAR + RiskMetrics variance forecasts and vol-targeted sizing",
};
const FACILITATORS = { "eip155:8453": "https://api.cdp.coinbase.com/platform/v2/x402", "eip155:84532": "https://x402.org/facilitator" };

/* ---------- configuration (per env; memoized) ---------- */

const configs = new WeakMap();
export function config(env) {
  let c = configs.get(env);
  if (c) return c;
  const gateway = Boolean((env.API_ORIGIN || "").trim());
  const x402 = (env.X402_PAY_TO || "").trim() ? {
    payTo: env.X402_PAY_TO.trim(), network: (env.X402_NETWORK || "eip155:84532").trim(),
    facilitator: (env.X402_FACILITATOR || FACILITATORS[(env.X402_NETWORK || "eip155:84532").trim()] || FACILITATORS["eip155:84532"]).replace(/\/+$/, ""),
    auth: env.X402_FACILITATOR_AUTH || null,
    prices: { ...PRICED, ...(env.X402_PRICES ? JSON.parse(env.X402_PRICES) : {}) },
    edgeNativePaywall: env.X402_PAYWALL_EDGE_NATIVE === "1",
  } : null;
  const tokenPrice = Number(env.CREDITS_PRICE_USD || 0.004);
  c = { gateway, x402, tokenPrice, denominations: [10, 25, 100], rateLimit: Number(env.RATE_LIMIT_PER_MINUTE || 30), maxBody: Number(env.MAX_BODY_BYTES || 1048576) };
  configs.set(env, c);
  return c;
}

const usd = (v) => "$" + v.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");

/** x402 route table: which POSTs are paywalled in this mode, plus token issuance. */
function x402Routes(env) {
  const c = config(env);
  if (!c.x402) return null;
  const opt = (price, description) => ({ accepts: { scheme: "exact", price, network: c.x402.network, payTo: c.x402.payTo, maxTimeoutSeconds: 300 }, description, mimeType: "application/json" });
  const routes = {};
  if (c.gateway || c.x402.edgeNativePaywall) for (const [k, p] of Object.entries(c.x402.prices)) if (PRICED[k] || k.startsWith("POST /v1/")) routes[k] = opt(p, DESCRIPTIONS[k] || k);
  if (env.PRIVACY_PASS_PRIVATE_JWK) {
    routes["POST /token-request"] = opt(usd(c.tokenPrice), "One Privacy Pass token (RFC 9578, Blind RSA) = one API call");
    for (const n of c.denominations) routes[`POST /token-request/batch/${n}`] = opt(usd(c.tokenPrice * n), `${n} Privacy Pass tokens, batched issuance`);
  }
  return routes;
}

const x402Servers = new WeakMap();
async function x402Server(env) {
  let s = x402Servers.get(env);
  if (s) return s;
  const c = config(env);
  const facilitator = new HTTPFacilitatorClient({ url: c.x402.facilitator, ...(c.x402.auth ? { createAuthHeaders: async () => ({ verify: { authorization: c.x402.auth }, settle: { authorization: c.x402.auth }, supported: { authorization: c.x402.auth } }) } : {}) });
  const server = new x402ResourceServer(facilitator).register(c.x402.network, new ExactEvmScheme());
  s = server.initialize().then(() => server); // one /supported fetch per isolate
  x402Servers.set(env, s);
  return s;
}

/* ---------- app ---------- */

const app = new Hono();

app.use("*", cors({
  origin: (o, c) => { const allowed = (c.env.ALLOWED_ORIGINS || "*").split(",").map((s) => s.trim()); return allowed.includes("*") ? "*" : (allowed.includes(o) ? o : null); },
  allowHeaders: ["content-type", "x-api-key", "authorization", "payment-signature", "x-payment"],
  exposeHeaders: ["x-usage-points", "x-usage-remaining", "x-usage-paid", "x-edge-cache", "x-ratelimit-limit", "x-ratelimit-remaining", "retry-after", "payment-required", "payment-response", "www-authenticate"],
  maxAge: 86400,
}));

// Security headers on HTML.
app.use("*", async (c, next) => {
  await next();
  if ((c.res.headers.get("content-type") || "").includes("text/html")) {
    c.res.headers.set("x-content-type-options", "nosniff"); c.res.headers.set("x-frame-options", "DENY"); c.res.headers.set("referrer-policy", "strict-origin-when-cross-origin");
  }
});

// Per-IP rate limit on the API (Workers rate-limit binding: free, no KV writes).
const clientIp = (c) => c.req.header("cf-connecting-ip") || c.req.header("x-forwarded-for") || "0.0.0.0";
app.use("/v1/*", rateLimit); app.use("/healthz", rateLimit); app.use("/token-request", rateLimit); app.use("/token-request/*", rateLimit);
async function rateLimit(c, next) {
  if (!c.env.RATE_LIMITER) return next();
  const { success } = await c.env.RATE_LIMITER.limit({ key: clientIp(c) });
  if (success) return next();
  return c.json({ detail: `Rate limit of ${config(c.env).rateLimit} requests/minute exceeded.` }, 429, { "retry-after": "60", "x-ratelimit-limit": String(config(c.env).rateLimit), "x-ratelimit-remaining": "0" });
}

app.get("/api/edge", (c) => {
  const cfg = config(c.env);
  return c.json({ worker: "timesfm3-edge", version: VERSION, mode: cfg.gateway ? "gateway" : "edge-native", upstream_key_configured: Boolean(c.env.API_KEY), rate_limiter: Boolean(c.env.RATE_LIMITER),
    x402: cfg.x402 ? { enabled: true, protocol: "x402 v2", network: cfg.x402.network, pay_to: cfg.x402.payTo, facilitator: cfg.x402.facilitator, prices_usd: cfg.x402.prices } : { enabled: false },
    privacy_pass: Boolean(c.env.PRIVACY_PASS_PRIVATE_JWK), colo: c.req.raw.cf?.colo || null, country: c.req.raw.cf?.country || null });
});
app.get("/metrics", (c) => c.json({ detail: "Not exposed at the edge." }, 404));

// Leads (KV) and Privacy Pass issuer/origin.
app.route("/api", leadsRoutes());
app.route("/", privacyPassRoutes({ VERSION }));

// Privacy Pass redemption: a valid unspent token pays for one priced call and
// skips the x402 paywall; an invalid one is a 401 with a fresh challenge.
app.use("/v1/*", async (c, next) => {
  const key = `${c.req.method} ${new URL(c.req.url).pathname}`;
  if (!PRICED[key] || !c.env.PRIVACY_PASS_PRIVATE_JWK) return next();
  const origin = await PrivacyPassOrigin.get(c.env, c.req.url);
  const auth = c.req.header("authorization") || "";
  if (!/^PrivateToken\s/i.test(auth)) { c.set("challenge", origin.challengeHeader()); return next(); }
  const r = await origin.redeem(auth);
  if (!r.ok) return c.json({ detail: `PrivateToken rejected: ${r.why}` }, 401, { "www-authenticate": origin.challengeHeader() });
  c.set("paidByToken", true);
  return next();
});

// x402 paywall (official middleware). Bring-your-own-key and token payers skip it.
app.use("*", async (c, next) => {
  const routes = x402Routes(c.env);
  if (!routes) return next();
  const hasKey = Boolean(c.req.header("x-api-key") || /^bearer\s/i.test(c.req.header("authorization") || ""));
  if (c.get("paidByToken") || (config(c.env).gateway && hasKey)) return next();
  const mw = paymentMiddleware(routes, await x402Server(c.env), undefined, undefined, false);
  return mw(c, async () => { await next(); });
});

// Add the Privacy Pass challenge to every 402 so clients can pick either payment.
app.use("/v1/*", async (c, next) => {
  await next();
  const ch = c.get("challenge");
  if (ch && (c.res.status === 402 || c.res.status === 401) && !c.res.headers.get("www-authenticate")) c.res.headers.set("www-authenticate", ch);
});

app.get("/v1/pricing", async (c) => {
  const cfg = config(c.env);
  const ppOrigin = c.env.PRIVACY_PASS_PRIVATE_JWK ? await PrivacyPassOrigin.get(c.env, c.req.url) : null;
  const paywall = cfg.x402 && (cfg.gateway || cfg.x402.edgeNativePaywall);
  return c.json({
    free: cfg.gateway ? "GET endpoints; bring your own API key for metered plans" : "classical-model API at the edge, TimesFM-3 in the browser at /app",
    plans: cfg.gateway ? { how: "X-API-Key from the service operator; metered in forecast points; see /v1/usage" } : null,
    x402: paywall ? { enabled: true, network: cfg.x402.network, pay_to: cfg.x402.payTo, facilitator: cfg.x402.facilitator, prices_usd: cfg.x402.prices } : { enabled: false, note: cfg.x402 ? "configured but not enforced in edge-native mode" : "set X402_PAY_TO to enable" },
    privacy_pass: c.env.PRIVACY_PASS_PRIVATE_JWK ? { enabled: true, standard: "RFC 9576/9577/9578, token type 0x0002 (Blind RSA)", one_token: "one priced call", price_per_token_usd: cfg.tokenPrice, issuer_directory: "/.well-known/private-token-issuer-directory", issue: "POST /token-request (single) or /token-request/batch/{10|25|100}", redeem: "Authorization: PrivateToken token=...", challenge: "GET /token-request/challenge", www_authenticate: ppOrigin ? ppOrigin.challengeHeader() : null } : { enabled: false, note: "set PRIVACY_PASS_PRIVATE_JWK + D1 binding" },
  });
});

/* ---------- API: gateway or edge-native ---------- */

const CACHEABLE = new Set(["/v1/models", "/v1/sample", "/healthz"]);
const API_PATHS = ["/v1/*", "/healthz", "/docs", "/openapi.json", "/redoc"];
for (const p of API_PATHS) app.all(p, async (c) => {
  const cfg = config(c.env);
  const url = new URL(c.req.url);
  if (cfg.gateway) return proxy(c, url.pathname + url.search);
  return edgeApi(c, { version: VERSION });
});

async function proxy(c, upstreamPath) {
  const env = c.env, request = c.req.raw;
  const origin = env.API_ORIGIN.replace(/\/+$/, "");
  const isGet = request.method === "GET" || request.method === "HEAD";
  if (Number(request.headers.get("content-length") || 0) > config(env).maxBody) return c.json({ detail: `Request body over ${config(env).maxBody} bytes.` }, 413);
  const upstreamUrl = origin + upstreamPath;
  const cacheKey = new Request(upstreamUrl, { method: "GET" });
  const cacheable = isGet && CACHEABLE.has(new URL(upstreamUrl).pathname);
  if (cacheable) { const hit = await caches.default.match(cacheKey); if (hit) { const r = new Response(hit.body, hit); r.headers.set("x-edge-cache", "HIT"); return r; } }
  const headers = new Headers(request.headers);
  for (const h of ["host", "cookie", "payment-signature", "x-payment"]) headers.delete(h);
  if (c.get("paidByToken") || !(headers.get("x-api-key") || /^bearer\s/i.test(headers.get("authorization") || ""))) { if (env.API_KEY) headers.set("x-api-key", env.API_KEY); }
  if (c.get("paidByToken")) headers.delete("authorization");
  headers.set("x-forwarded-for", clientIp(c)); headers.set("x-edge-worker", "timesfm3-edge");
  const init = { method: request.method, headers, redirect: "manual" };
  if (!isGet) init.body = request.body;
  let response = await fetch(upstreamUrl, init);
  response = new Response(response.body, response);
  response.headers.set("x-edge-cache", "MISS"); response.headers.delete("server");
  if (cacheable && response.ok) { const toCache = new Response(response.clone().body, response); toCache.headers.set("cache-control", "public, s-maxage=60"); c.executionCtx.waitUntil(caches.default.put(cacheKey, toCache)); }
  return response;
}

/* ---------- static site ---------- */

const page = (path) => async (c) => { const u = new URL(c.req.url); u.pathname = path; u.search = ""; return c.env.ASSETS.fetch(new Request(u.toString(), { method: "GET" })); };
app.get("/", page("/index.html")); app.get("/index.html", page("/index.html"));
app.get("/app", page("/app/index.html")); app.get("/app/", page("/app/index.html"));
app.get("/favicon.ico", page("/favicon.svg"));
app.all("*", (c) => c.env.ASSETS.fetch(c.req.raw));

export default app;

