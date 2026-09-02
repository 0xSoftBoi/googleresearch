/** Waitlist capture in KV (honeypot, dedupe by email) and founder-only listing. */
import { Hono } from "hono";

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;
const clip = (v, n) => (v === undefined || v === null ? "" : String(v).trim().slice(0, n));
async function sha256(text) { const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)); return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join(""); }

export function leadsRoutes() {
  const r = new Hono();
  r.post("/waitlist", async (c) => {
    let body; try { body = await c.req.json(); } catch { return c.json({ detail: "Send JSON: {email, company?, plan?, use_case?}." }, 400); }
    if (body.website) return c.json({ ok: true }); // honeypot
    const email = String(body.email || "").trim().toLowerCase();
    if (!EMAIL_RE.test(email) || email.length > 254) return c.json({ detail: "A valid email address is required." }, 400);
    if (!c.env.EDGE_KV) return c.json({ detail: "Lead storage not configured." }, 503);
    const ip = c.req.header("cf-connecting-ip") || "0.0.0.0";
    const lead = { email, company: clip(body.company, 120), plan: clip(body.plan, 40) || "unspecified", use_case: clip(body.use_case, 1000), source: clip(body.source, 80) || "landing", country: c.req.raw.cf?.country || null, ip_hash: await sha256(ip), created: new Date().toISOString() };
    const existing = await c.env.EDGE_KV.get(`lead-email:${email}`);
    const id = existing || `${lead.created}:${(await sha256(email)).slice(0, 12)}`;
    if (existing) { const prev = await c.env.EDGE_KV.get(`lead:${existing}`); if (prev) { const p = JSON.parse(prev); lead.created = p.created || lead.created; lead.updated = new Date().toISOString(); lead.signups = (p.signups || 1) + 1; for (const k of ["company", "use_case"]) if (!lead[k] && p[k]) lead[k] = p[k]; } }
    await c.env.EDGE_KV.put(`lead:${id}`, JSON.stringify(lead)); await c.env.EDGE_KV.put(`lead-email:${email}`, id);
    if (c.env.LEADS_WEBHOOK) c.executionCtx.waitUntil(fetch(c.env.LEADS_WEBHOOK, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ text: `New TimesFM-3 lead: ${email} (${lead.plan}) ${lead.company || ""}`.trim(), content: `New TimesFM-3 lead: ${email} (${lead.plan})` }) }).catch(() => {}));
    return c.json({ ok: true, duplicate: Boolean(existing) });
  });
  r.get("/leads", async (c) => {
    const m = /^bearer\s+(.+)$/i.exec(c.req.header("authorization") || "");
    if (!c.env.ADMIN_TOKEN || !m || m[1].trim() !== c.env.ADMIN_TOKEN) return c.json({ detail: "Admin token required." }, 401);
    const out = []; let cursor;
    do { const page = await c.env.EDGE_KV.list({ prefix: "lead:", cursor, limit: 1000 }); for (const k of page.keys) { const v = await c.env.EDGE_KV.get(k.name); if (v) out.push(JSON.parse(v)); } cursor = page.list_complete ? undefined : page.cursor; } while (cursor);
    out.sort((a, b) => (a.created < b.created ? 1 : -1));
    return c.json({ count: out.length, leads: out });
  });
  return r;
}
