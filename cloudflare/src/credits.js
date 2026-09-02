/**
 * Credit pool at the edge: RFC 9474 blind signatures with @cloudflare/blindrsa-ts,
 * nullifiers in D1. Same token format and suite as the Python service, so a
 * token bought from either can be redeemed at either that shares the key.
 *
 *   secret CREDITS_PRIVATE_JWK   private RSA JWK (p,q,dp,dq,qi) -> issuing key
 *   var    CREDITS_OLD_PUBLIC_JWKS  JSON array of public JWKs still redeemable
 *   var    CREDITS_PRICE_USD     per credit (default 0.004)
 *   binding CREDITS_DB           D1 (migrations/0001_credits.sql)
 */

import { RSABSSA } from "@cloudflare/blindrsa-ts";

export const SUITE_NAME = "RSABSSA-SHA384-PSSZERO-Deterministic";
export const SERIAL_BYTES = 32;
export const DENOMINATIONS = [10, 25, 100];
export const CREDIT_COSTS = { "POST /v1/forecast": 1, "POST /v1/volatility": 1, "POST /v1/anomalies": 2, "POST /v1/backtest": 4 };
export const POINTS_PER_CREDIT = 256;

const suite = RSABSSA.SHA384.PSSZero.Deterministic();
const enc = new TextEncoder();

export const b64 = {
  encode: (u8) => btoa(String.fromCharCode(...u8)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, ""),
  decode: (s) => Uint8Array.from(atob(s.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - (s.length % 4)) % 4)), (c) => c.charCodeAt(0)),
};

async function sha256hex(bytes) {
  return [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** kid = first 12 hex of SHA-256 over the big-endian modulus (matches Python). */
export async function keyIdFromJwk(jwk) {
  return (await sha256hex(b64.decode(jwk.n))).slice(0, 12);
}

let cached = null; // { issuing: {kid, priv, pub, jwk}, keys: Map(kid -> {pub, jwk, issuing}), price }

export async function creditPool(env) {
  if (!env.CREDITS_PRIVATE_JWK) return null;
  if (cached && cached.raw === env.CREDITS_PRIVATE_JWK) return cached;
  const priv = JSON.parse(env.CREDITS_PRIVATE_JWK);
  const pubJwk = { kty: "RSA", n: priv.n, e: priv.e, alg: "PS384", use: "sig" };
  const algo = { name: "RSA-PSS", hash: "SHA-384" };
  const privateKey = await crypto.subtle.importKey("jwk", { ...priv, alg: "PS384", ext: true }, algo, true, ["sign"]);
  const publicKey = await crypto.subtle.importKey("jwk", { ...pubJwk, ext: true }, algo, true, ["verify"]);
  const kid = await keyIdFromJwk(pubJwk);
  const keys = new Map([[kid, { kid, pub: publicKey, jwk: pubJwk, issuing: true }]]);
  for (const old of JSON.parse(env.CREDITS_OLD_PUBLIC_JWKS || "[]")) {
    const okid = await keyIdFromJwk(old);
    if (!keys.has(okid)) keys.set(okid, { kid: okid, pub: await crypto.subtle.importKey("jwk", { kty: "RSA", n: old.n, e: old.e, alg: "PS384", ext: true }, algo, true, ["verify"]), jwk: { kty: "RSA", n: old.n, e: old.e, alg: "PS384", use: "sig" }, issuing: false });
  }
  cached = { raw: env.CREDITS_PRIVATE_JWK, issuing: { kid, priv: privateKey, pub: publicKey, jwk: pubJwk }, keys, price: Number(env.CREDITS_PRICE_USD || 0.004) };
  return cached;
}

export function priceUsd(pool, count) {
  return "$" + (count * pool.price).toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

export async function describe(pool, env) {
  let stats = { issued: 0, redeemed: 0 };
  if (env.CREDITS_DB) {
    const row = await env.CREDITS_DB.prepare("SELECT COALESCE(SUM(issued),0) AS issued, COALESCE(SUM(redeemed),0) AS redeemed FROM stats").first();
    if (row) stats = { issued: Number(row.issued), redeemed: Number(row.redeemed) };
  }
  return {
    suite: SUITE_NAME, spec: "RFC 9474", serial_bytes: SERIAL_BYTES,
    keys: [...pool.keys.values()].map((k) => ({ kid: k.kid, jwk: k.jwk, issuing: k.issuing })),
    kid: pool.issuing.kid, denominations: DENOMINATIONS, price_per_credit_usd: pool.price,
    points_per_credit: POINTS_PER_CREDIT, costs: CREDIT_COSTS,
    pool: { ...stats, outstanding: stats.issued - stats.redeemed },
    ephemeral_key: false, runs_in: "edge",
    token_format: "kid.base64url(serial).base64url(signature); header X-Credit: token[,token]",
  };
}

export async function signBlinded(pool, env, blindedList) {
  const out = [];
  for (const b of blindedList) {
    let blinded;
    try { blinded = b64.decode(String(b)); } catch { throw new Error("bad blinded message"); }
    out.push(b64.encode(await suite.blindSign(pool.issuing.priv, blinded)));
  }
  if (env.CREDITS_DB) {
    await env.CREDITS_DB.prepare("INSERT INTO stats (kid, issued, redeemed) VALUES (?1, ?2, 0) ON CONFLICT(kid) DO UPDATE SET issued = issued + ?2").bind(pool.issuing.kid, out.length).run();
  }
  return out;
}

async function verifyToken(pool, token) {
  const parts = token.trim().split(".");
  if (parts.length !== 3) return null;
  const key = pool.keys.get(parts[0]);
  if (!key) return null;
  let serial, sig;
  try { serial = b64.decode(parts[1]); sig = b64.decode(parts[2]); } catch { return null; }
  if (serial.length !== SERIAL_BYTES) return null;
  try { if (!(await suite.verify(key.pub, sig, serial))) return null; } catch { return null; }
  return { kid: key.kid, serial };
}

/** All-or-nothing redemption of `cost` tokens from the X-Credit header. */
export async function redeem(pool, env, header, cost) {
  const tokens = header.split(",").map((t) => t.trim()).filter(Boolean);
  if (tokens.length < cost) return { ok: false, why: `this call costs ${cost} credit(s); ${tokens.length} presented` };
  const nulls = [];
  for (const t of tokens.slice(0, cost)) {
    const v = await verifyToken(pool, t);
    if (!v) return { ok: false, why: "invalid credit token (bad signature or unknown key id)" };
    const nul = await sha256hex(new Uint8Array([...enc.encode("nullifier"), ...v.serial]));
    nulls.push({ nul, kid: v.kid });
  }
  if (new Set(nulls.map((n) => n.nul)).size !== nulls.length) return { ok: false, why: "duplicate credit token in request" };
  if (!env.CREDITS_DB) return { ok: false, why: "credit ledger not configured" };
  const ts = Date.now();
  // INSERT OR IGNORE per nullifier; if any was already present, roll back the batch.
  const results = await env.CREDITS_DB.batch(nulls.map((n) => env.CREDITS_DB.prepare("INSERT OR IGNORE INTO spent (nullifier, kid, ts) VALUES (?1, ?2, ?3)").bind(n.nul, n.kid, ts)));
  const inserted = results.map((r) => Number((r.meta && r.meta.changes) ?? 0));
  if (inserted.some((c) => c === 0)) {
    const mine = nulls.filter((_, i) => inserted[i] === 1).map((n) => n.nul);
    if (mine.length) await env.CREDITS_DB.batch(mine.map((nul) => env.CREDITS_DB.prepare("DELETE FROM spent WHERE nullifier = ?1 AND ts = ?2").bind(nul, ts)));
    return { ok: false, why: "credit token already spent" };
  }
  await env.CREDITS_DB.prepare("INSERT INTO stats (kid, issued, redeemed) VALUES (?1, 0, ?2) ON CONFLICT(kid) DO UPDATE SET redeemed = redeemed + ?2").bind(pool.issuing.kid, cost).run();
  return { ok: true, why: "" };
}

export function creditCost(method, path) {
  return CREDIT_COSTS[`${method.toUpperCase()} ${path}`] ?? null;
}
