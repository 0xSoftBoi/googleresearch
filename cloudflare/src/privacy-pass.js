/**
 * Privacy Pass at the edge (RFC 9576/9577/9578) with @cloudflare/privacypass-ts.
 *
 * The Worker is both Issuer and Origin. Publicly verifiable tokens (type
 * 0x0002, Blind RSA, PSS mode): one token pays for one priced API call.
 *
 *   GET  /.well-known/private-token-issuer-directory   (RFC 9578 §4.3)
 *   POST /token-request                                (single or generic batch, by Content-Type)
 *   POST /token-request/batch/{n}                      (generic batch, fixed-price for x402)
 *   Authorization: PrivateToken token=...              on /v1/* priced routes
 *
 * Secrets/vars: PRIVACY_PASS_PRIVATE_JWK (issuing key), PRIVACY_PASS_OLD_PUBLIC_JWKS
 * (JSON array; still redeemable), binding PRIVACY_PASS_DB (D1 nullifier ledger).
 */

import { Hono } from "hono";
import { publicVerif, genericBatched, AuthorizationHeader, WWWAuthenticateHeader, TOKEN_TYPES, MediaType, PRIVATE_TOKEN_ISSUER_DIRECTORY } from "@cloudflare/privacypass-ts";

const MODE = publicVerif.BlindRSAMode.PSS; // RFC 9578 §6: PSS with 48-byte salt
const ALGO = { name: "RSA-PSS", hash: "SHA-384" };
const hex = (u8) => [...u8].map((b) => b.toString(16).padStart(2, "0")).join("");

const pools = new WeakMap();
async function pool(env) {
  let p = pools.get(env);
  if (p) return p;
  if (!env.PRIVACY_PASS_PRIVATE_JWK) return null;
  const priv = JSON.parse(env.PRIVACY_PASS_PRIVATE_JWK);
  const privateKey = await crypto.subtle.importKey("jwk", { ...priv, alg: "PS384", ext: true }, ALGO, true, ["sign"]);
  const pubJwk = { kty: "RSA", n: priv.n, e: priv.e, alg: "PS384", ext: true };
  const publicKey = await crypto.subtle.importKey("jwk", pubJwk, ALGO, true, ["verify"]);
  const issuerName = (env.PRIVACY_PASS_ISSUER_NAME || "").trim() || null;
  const issuer = new publicVerif.Issuer(MODE, issuerName || "issuer", privateKey, publicKey);
  const issuing = { publicKey, spki: await publicVerif.getPublicKeyBytes(publicKey), keyId: await issuer.tokenKeyID(), issuing: true };
  const keys = [issuing];
  for (const old of JSON.parse(env.PRIVACY_PASS_OLD_PUBLIC_JWKS || "[]")) {
    const pk = await crypto.subtle.importKey("jwk", { kty: "RSA", n: old.n, e: old.e, alg: "PS384", ext: true }, ALGO, true, ["verify"]);
    const spki = await publicVerif.getPublicKeyBytes(pk);
    keys.push({ publicKey: pk, spki, keyId: new Uint8Array(await crypto.subtle.digest("SHA-256", spki)), issuing: false });
  }
  p = { issuer, batched: new genericBatched.Issuer(issuer), issuing, keys, issuerName };
  pools.set(env, p);
  return p;
}

/** Origin role: challenge + redemption against the D1 ledger. */
export class PrivacyPassOrigin {
  static async get(env, url) {
    const p = await pool(env);
    if (!p) return null;
    const host = (env.PRIVACY_PASS_ORIGIN || "").trim() || new URL(url).host;
    return new PrivacyPassOrigin(env, p, host);
  }
  constructor(env, p, host) {
    this.env = env; this.pool = p; this.host = host;
    this.origin = new publicVerif.Origin(MODE, [host]);
    // Static per-origin challenge (empty redemption context): tokens are pre-purchasable.
    this.challenge = this.origin.createTokenChallenge(p.issuerName || host, new Uint8Array(0));
  }
  challengeHeader() {
    return new WWWAuthenticateHeader(this.challenge, this.pool.issuing.spki, 86400 * 30).toString();
  }
  async redeem(authorization) {
    let headers;
    try { headers = AuthorizationHeader.parse(TOKEN_TYPES.BLIND_RSA, authorization); } catch { return { ok: false, why: "malformed PrivateToken header" }; }
    if (!headers.length) return { ok: false, why: "no PrivateToken credential" };
    const token = headers[0].token;
    const key = this.pool.keys.find((k) => hex(k.keyId) === hex(token.authInput.tokenKeyId));
    if (!key) return { ok: false, why: "unknown token key" };
    const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", this.challenge.serialize()));
    if (hex(digest) !== hex(token.authInput.challengeDigest)) return { ok: false, why: "token was issued for a different challenge" };
    if (!(await this.origin.verify(token, key.publicKey))) return { ok: false, why: "invalid token" };
    if (!this.env.PRIVACY_PASS_DB) return { ok: false, why: "ledger not configured" };
    const nullifier = hex(new Uint8Array(await crypto.subtle.digest("SHA-256", new Uint8Array([...new TextEncoder().encode("nullifier"), ...token.authInput.nonce]))));
    const r = await this.env.PRIVACY_PASS_DB.prepare("INSERT OR IGNORE INTO spent (nullifier, kid, ts) VALUES (?1, ?2, ?3)").bind(nullifier, hex(key.keyId).slice(0, 12), Date.now()).run();
    if (!Number(r.meta?.changes ?? 0)) return { ok: false, why: "token already spent" };
    await this.env.PRIVACY_PASS_DB.prepare("INSERT INTO stats (kid, issued, redeemed) VALUES (?1, 0, 1) ON CONFLICT(kid) DO UPDATE SET redeemed = redeemed + 1").bind(hex(this.pool.issuing.keyId).slice(0, 12)).run();
    return { ok: true };
  }
}

async function countIssued(env, kid, n) {
  if (!env.PRIVACY_PASS_DB) return;
  await env.PRIVACY_PASS_DB.prepare("INSERT INTO stats (kid, issued, redeemed) VALUES (?1, ?2, 0) ON CONFLICT(kid) DO UPDATE SET issued = issued + ?2").bind(kid, n).run();
}

export function privacyPassRoutes({ VERSION }) {
  const r = new Hono();
  const b64 = (u8) => btoa(String.fromCharCode(...u8)); // padded base64url per RFC 9577
  const b64url = (u8) => b64(u8).replace(/\+/g, "-").replace(/\//g, "_");

  r.get(PRIVATE_TOKEN_ISSUER_DIRECTORY, async (c) => {
    const p = await pool(c.env);
    if (!p) return c.json({ detail: "Privacy Pass issuer not configured (set PRIVACY_PASS_PRIVATE_JWK)." }, 404);
    const body = { "issuer-request-uri": "/token-request", "token-keys": p.keys.map((k) => ({ "token-type": TOKEN_TYPES.BLIND_RSA.value, "token-key": b64url(k.spki), ...(k.issuing ? {} : { "not-after": null }) })) };
    return new Response(JSON.stringify(body), { headers: { "content-type": MediaType.PRIVATE_TOKEN_ISSUER_DIRECTORY, "cache-control": "public, max-age=300" } });
  });

  // A resource that always challenges: the discoverable way to fetch the challenge
  // (RFC 9577 puts it on 401/402 responses; the free edge API rarely returns those).
  r.get("/token-request/challenge", async (c) => {
    const origin = await PrivacyPassOrigin.get(c.env, c.req.url);
    if (!origin) return c.json({ detail: "Privacy Pass issuer not configured." }, 404);
    return c.json({ detail: "Present a PrivateToken (see WWW-Authenticate) or buy tokens at POST /token-request." }, 401, { "www-authenticate": origin.challengeHeader() });
  });

  r.get("/token-request/stats", async (c) => {
    const p = await pool(c.env);
    if (!p) return c.json({ detail: "not configured" }, 404);
    let stats = { issued: 0, redeemed: 0 };
    if (c.env.PRIVACY_PASS_DB) { const row = await c.env.PRIVACY_PASS_DB.prepare("SELECT COALESCE(SUM(issued),0) AS issued, COALESCE(SUM(redeemed),0) AS redeemed FROM stats").first(); if (row) stats = { issued: Number(row.issued), redeemed: Number(row.redeemed) }; }
    return c.json({ standard: "RFC 9578 type 0x0002 (Blind RSA, PSS)", keys: p.keys.map((k) => ({ kid: hex(k.keyId).slice(0, 12), issuing: k.issuing })), pool: { ...stats, outstanding: stats.issued - stats.redeemed }, version: VERSION });
  });

  const single = async (c, p, bytes) => {
    let req;
    try { req = publicVerif.TokenRequest.deserialize(TOKEN_TYPES.BLIND_RSA, bytes); } catch { return c.json({ detail: "malformed TokenRequest" }, 422); }
    if (req.truncatedTokenKeyId !== p.issuing.keyId[p.issuing.keyId.length - 1]) return c.json({ detail: "TokenRequest is for a key this issuer does not issue with" }, 422);
    let res;
    try { res = await p.issuer.issue(req); } catch (e) { return c.json({ detail: `issuance failed: ${e.message}` }, 422); }
    await countIssued(c.env, hex(p.issuing.keyId).slice(0, 12), 1);
    return new Response(res.serialize(), { headers: { "content-type": MediaType.PRIVATE_TOKEN_RESPONSE, "cache-control": "no-store" } });
  };
  const batch = async (c, p, bytes, expected) => {
    let reqs;
    try { reqs = genericBatched.BatchedTokenRequest.deserialize(bytes); } catch { return c.json({ detail: "malformed BatchedTokenRequest" }, 422); }
    const n = reqs.tokenRequests.length;
    if (expected !== null && n !== expected) return c.json({ detail: `this batch endpoint issues exactly ${expected} tokens; ${n} requested` }, 422);
    if (n < 1 || n > 100) return c.json({ detail: "batches are 1-100 tokens" }, 422);
    const res = await p.batched.issue(reqs);
    const issued = res.tokenResponses.filter((t) => t.tokenResponse !== null).length;
    await countIssued(c.env, hex(p.issuing.keyId).slice(0, 12), issued);
    return new Response(res.serialize(), { headers: { "content-type": MediaType.GENERIC_BATCHED_TOKEN_RESPONSE, "cache-control": "no-store", "x-tokens-issued": String(issued) } });
  };

  r.post("/token-request", async (c) => {
    const p = await pool(c.env);
    if (!p) return c.json({ detail: "Privacy Pass issuer not configured." }, 404);
    const ct = (c.req.header("content-type") || "").split(";")[0].trim();
    const bytes = new Uint8Array(await c.req.arrayBuffer());
    if (ct === MediaType.GENERIC_BATCHED_TOKEN_REQUEST) return batch(c, p, bytes, null);
    if (ct !== MediaType.PRIVATE_TOKEN_REQUEST) return c.json({ detail: `Content-Type must be ${MediaType.PRIVATE_TOKEN_REQUEST} or ${MediaType.GENERIC_BATCHED_TOKEN_REQUEST}` }, 415);
    return single(c, p, bytes);
  });
  r.post("/token-request/batch/:n", async (c) => {
    const p = await pool(c.env);
    if (!p) return c.json({ detail: "Privacy Pass issuer not configured." }, 404);
    const n = Number(c.req.param("n"));
    if (![10, 25, 100].includes(n)) return c.json({ detail: "batch denominations: 10, 25, 100" }, 404);
    return batch(c, p, new Uint8Array(await c.req.arrayBuffer()), n);
  });
  return r;
}
