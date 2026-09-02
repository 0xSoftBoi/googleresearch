/**
 * TimesFM-3 Forecast Service — Cloudflare edge front end.
 *
 * One Worker does four jobs:
 *   1. Serves the marketing site from ./public (Workers static assets).
 *   2. Proxies the API (/v1/*, /healthz, /docs, /openapi.json) to the
 *      self-hosted service at API_ORIGIN, attaching the upstream API key
 *      server-side so browsers never see it, with per-IP rate limiting in KV
 *      and edge caching of the cheap GETs.
 *   3. Serves the product dashboard at /app by proxying the upstream "/".
 *   4. Captures signups (POST /api/waitlist) into KV, optionally forwarding
 *      them to a webhook, and lists them for the founder (GET /api/leads).
 *
 * Bindings (wrangler.jsonc): ASSETS (static), EDGE_KV (rate limits + leads).
 * Vars: API_ORIGIN, RATE_LIMIT_PER_MINUTE, MAX_BODY_BYTES, ALLOWED_ORIGINS.
 * Secrets: API_KEY (upstream key), ADMIN_TOKEN (for /api/leads),
 *          LEADS_WEBHOOK (optional Slack/Discord-style webhook URL).
 */

const PROXY_PREFIXES = ["/v1/", "/healthz", "/docs", "/openapi.json", "/redoc", "/favicon.ico"];
const CACHEABLE = new Set(["/v1/models", "/v1/sample", "/healthz"]);
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;

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
      if (path === "/app" || path === "/app/") return await proxy(request, env, ctx, "/");
      if (PROXY_PREFIXES.some((p) => path === p || path.startsWith(p))) {
        return await proxy(request, env, ctx, path + url.search);
      }
      // Everything else is the static site (with its own 404 page).
      return withSecurityHeaders(await env.ASSETS.fetch(request));
    } catch (err) {
      console.error("edge error", err);
      return json({ detail: "Edge error: " + (err && err.message ? err.message : String(err)) }, 502);
    }
  },
};

/* ---------- API proxy ---------- */

async function proxy(request, env, ctx, upstreamPath) {
  const origin = (env.API_ORIGIN || "").replace(/\/+$/, "");
  if (!origin) return json({ detail: "API_ORIGIN is not configured on the edge." }, 503);

  const isGet = request.method === "GET" || request.method === "HEAD";
  const rateLimited = await rateLimit(request, env, ctx);
  if (rateLimited) return cors(rateLimited, request, env);

  const maxBody = Number(env.MAX_BODY_BYTES || 1048576);
  const declared = Number(request.headers.get("content-length") || 0);
  if (declared > maxBody) {
    return cors(json({ detail: `Request body over ${maxBody} bytes.` }, 413), request, env);
  }

  const upstreamUrl = origin + upstreamPath;
  const cacheKey = new Request(upstreamUrl, { method: "GET" });
  const cache = caches.default;
  if (isGet && CACHEABLE.has(new URL(upstreamUrl).pathname)) {
    const hit = await cache.match(cacheKey);
    if (hit) return cors(withHeader(hit, "x-edge-cache", "HIT"), request, env);
  }

  const headers = new Headers(request.headers);
  headers.delete("host");
  headers.delete("cookie");
  // Bring-your-own-key callers keep their key; anonymous callers ride the
  // edge's own (quota-limited) upstream key, if one is configured.
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

  if (isGet && response.ok && CACHEABLE.has(new URL(upstreamUrl).pathname)) {
    const toCache = new Response(response.clone().body, response);
    toCache.headers.set("cache-control", "public, s-maxage=60");
    ctx.waitUntil(cache.put(cacheKey, toCache));
  }
  const ct = response.headers.get("content-type") || "";
  return cors(ct.includes("text/html") ? withSecurityHeaders(response) : response, request, env);
}

async function rateLimit(request, env, ctx) {
  const limit = Number(env.RATE_LIMIT_PER_MINUTE || 30);
  if (!limit || !env.EDGE_KV) return null;
  const ip = clientIp(request);
  const minute = Math.floor(Date.now() / 60000);
  const key = `rl:${ip}:${minute}`;
  const current = Number((await env.EDGE_KV.get(key)) || 0);
  if (current >= limit) {
    const reset = (minute + 1) * 60 - Math.floor(Date.now() / 1000);
    const r = json({ detail: `Rate limit of ${limit} requests/minute exceeded.` }, 429);
    r.headers.set("retry-after", String(Math.max(1, reset)));
    r.headers.set("x-ratelimit-limit", String(limit));
    r.headers.set("x-ratelimit-remaining", "0");
    return r;
  }
  // KV is eventually consistent; this is a soft limit, which is fine for an
  // abuse brake in front of a service that also meters per key.
  ctx.waitUntil(env.EDGE_KV.put(key, String(current + 1), { expirationTtl: 120 }));
  return null;
}

/* ---------- Leads ---------- */

async function waitlist(request, env, ctx) {
  let body;
  try {
    body = await request.json();
  } catch {
    return cors(json({ detail: "Send JSON: {email, company?, plan?, use_case?}." }, 400), request, env);
  }
  if (body.website) return cors(json({ ok: true }), request, env); // honeypot: pretend success
  const email = String(body.email || "").trim().toLowerCase();
  if (!EMAIL_RE.test(email) || email.length > 254) {
    return cors(json({ detail: "A valid email address is required." }, 400), request, env);
  }
  const lead = {
    email,
    company: clip(body.company, 120),
    plan: clip(body.plan, 40) || "unspecified",
    use_case: clip(body.use_case, 1000),
    source: clip(body.source, 80) || "landing",
    country: request.cf && request.cf.country ? request.cf.country : null,
    ip_hash: await sha256(clientIp(request)),
    created: new Date().toISOString(),
  };
  if (!env.EDGE_KV) return cors(json({ detail: "Lead storage not configured." }, 503), request, env);
  // One record per email: a repeat signup updates the existing lead (keeping
  // its first-seen time) rather than adding a second row.
  const existing = await env.EDGE_KV.get(`lead-email:${email}`);
  const id = existing || `${lead.created}:${(await sha256(email)).slice(0, 12)}`;
  if (existing) {
    const prev = await env.EDGE_KV.get(`lead:${existing}`);
    if (prev) {
      const p = JSON.parse(prev);
      lead.created = p.created || lead.created;
      lead.updated = new Date().toISOString();
      lead.signups = (p.signups || 1) + 1;
      for (const k of ["company", "use_case"]) if (!lead[k] && p[k]) lead[k] = p[k];
    }
  }
  await env.EDGE_KV.put(`lead:${id}`, JSON.stringify(lead));
  await env.EDGE_KV.put(`lead-email:${email}`, id);
  if (env.LEADS_WEBHOOK) {
    ctx.waitUntil(
      fetch(env.LEADS_WEBHOOK, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          text: `New TimesFM-3 lead: ${email} (${lead.plan}) ${lead.company || ""} ${lead.use_case ? "— " + lead.use_case : ""}`.trim(),
          content: `New TimesFM-3 lead: ${email} (${lead.plan})`,
        }),
      }).catch((e) => console.error("webhook failed", e)),
    );
  }
  return cors(json({ ok: true, duplicate: Boolean(existing) }), request, env);
}

async function leads(request, env) {
  const token = bearer(request.headers.get("authorization"));
  if (!env.ADMIN_TOKEN || !token || token !== env.ADMIN_TOKEN) {
    return json({ detail: "Admin token required." }, 401);
  }
  const out = [];
  let cursor;
  do {
    const page = await env.EDGE_KV.list({ prefix: "lead:", cursor, limit: 1000 });
    for (const k of page.keys) {
      const v = await env.EDGE_KV.get(k.name);
      if (v) out.push(JSON.parse(v));
    }
    cursor = page.list_complete ? undefined : page.cursor;
  } while (cursor);
  out.sort((a, b) => (a.created < b.created ? 1 : -1));
  return json({ count: out.length, leads: out });
}

/* ---------- helpers ---------- */

function edgeInfo(request, env) {
  return {
    worker: "timesfm3-edge",
    api_origin_configured: Boolean(env.API_ORIGIN),
    upstream_key_configured: Boolean(env.API_KEY),
    rate_limit_per_minute: Number(env.RATE_LIMIT_PER_MINUTE || 30),
    colo: request.cf && request.cf.colo ? request.cf.colo : null,
    country: request.cf && request.cf.country ? request.cf.country : null,
  };
}

function clientIp(request) {
  return request.headers.get("cf-connecting-ip") || request.headers.get("x-forwarded-for") || "0.0.0.0";
}

function bearer(value) {
  if (!value) return null;
  const m = /^bearer\s+(.+)$/i.exec(value.trim());
  return m ? m[1].trim() : null;
}

function clip(v, n) {
  if (v === undefined || v === null) return "";
  return String(v).trim().slice(0, n);
}

async function sha256(text) {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });
}

function withHeader(response, name, value) {
  const r = new Response(response.body, response);
  r.headers.set(name, value);
  return r;
}

function withSecurityHeaders(response) {
  const r = new Response(response.body, response);
  r.headers.set("x-content-type-options", "nosniff");
  r.headers.set("x-frame-options", "DENY");
  r.headers.set("referrer-policy", "strict-origin-when-cross-origin");
  r.headers.set("permissions-policy", "camera=(), microphone=(), geolocation=()");
  return r;
}

function cors(response, request, env) {
  const allowed = (env.ALLOWED_ORIGINS || "*").split(",").map((s) => s.trim()).filter(Boolean);
  const origin = request.headers.get("origin");
  const r = new Response(response.body, response);
  if (allowed.includes("*")) r.headers.set("access-control-allow-origin", "*");
  else if (origin && allowed.includes(origin)) {
    r.headers.set("access-control-allow-origin", origin);
    r.headers.set("vary", "origin");
  }
  r.headers.set("access-control-allow-methods", "GET, POST, OPTIONS");
  r.headers.set("access-control-allow-headers", "content-type, x-api-key, authorization");
  r.headers.set("access-control-expose-headers", "x-usage-points, x-usage-remaining, x-edge-cache, x-ratelimit-limit, x-ratelimit-remaining, retry-after");
  r.headers.set("access-control-max-age", "86400");
  return r;
}
