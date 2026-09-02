/**
 * x402 v2 pay-per-call for the edge Worker (spec: coinbase/x402, "exact" scheme).
 *
 * Anonymous callers get `402` with a base64 `PAYMENT-REQUIRED` header (the same
 * JSON is in the body for humans); they retry with `PAYMENT-SIGNATURE` carrying
 * a signed USDC transfer authorization. The facilitator verifies before the
 * handler runs and settles after it succeeds; the response carries
 * `PAYMENT-RESPONSE`. Bring-your-own-key callers (gateway mode) skip the paywall
 * and are metered by the upstream service instead.
 *
 * Vars: X402_PAY_TO (enables), X402_NETWORK (default eip155:84532 = Base
 * Sepolia; mainnet eip155:8453), X402_FACILITATOR (default by network),
 * X402_PRICES (JSON overrides), X402_PAYWALL_EDGE_NATIVE ("1" to also charge
 * for the free classical API). Secret: X402_FACILITATOR_AUTH (Authorization
 * header value, e.g. for Coinbase CDP on mainnet).
 */

export const USDC = {
  "eip155:8453": { asset: "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913", name: "USD Coin", version: "2", decimals: 6 },
  "eip155:84532": { asset: "0x036CbD53842c5426634e7929541eC2318f3dCF7e", name: "USDC", version: "2", decimals: 6 },
};
export const DEFAULT_FACILITATORS = {
  "eip155:8453": "https://api.cdp.coinbase.com/platform/v2/x402",
  "eip155:84532": "https://x402.org/facilitator",
};
export const DEFAULT_PRICES = {
  "POST /v1/forecast": "$0.005",
  "POST /v1/anomalies": "$0.01",
  "POST /v1/backtest": "$0.02",
  "POST /v1/volatility": "$0.005",
};
const DESCRIPTIONS = {
  "POST /v1/forecast": "TimesFM-3 forecast: point + 9 quantiles per series and step",
  "POST /v1/anomalies": "Walk-forward anomaly scoring against the model's predictive band",
  "POST /v1/backtest": "Walk-forward model comparison with Diebold-Mariano tests",
  "POST /v1/volatility": "HAR + RiskMetrics variance forecasts and vol-targeted sizing",
};

export function x402Config(env) {
  const payTo = (env.X402_PAY_TO || "").trim();
  if (!payTo) return null;
  const network = (env.X402_NETWORK || "eip155:84532").trim();
  const asset = USDC[network];
  if (!asset) throw new Error(`x402: no USDC metadata for network ${network}`);
  let prices = { ...DEFAULT_PRICES };
  if (env.X402_PRICES) prices = { ...prices, ...JSON.parse(env.X402_PRICES) };
  return {
    payTo, network, asset, prices,
    facilitator: (env.X402_FACILITATOR || DEFAULT_FACILITATORS[network] || DEFAULT_FACILITATORS["eip155:84532"]).replace(/\/+$/, ""),
    auth: env.X402_FACILITATOR_AUTH || null,
    maxTimeoutSeconds: 300,
  };
}

export function describe(cfg) {
  return cfg ? { enabled: true, protocol: "x402 v2", network: cfg.network, pay_to: cfg.payTo, facilitator: cfg.facilitator, prices_usd: cfg.prices } : { enabled: false };
}

/** "$0.005" -> "5000" atomic USDC (6 decimals). */
export function toAtomic(price, decimals = 6) {
  const m = /^\$?\s*(\d+)(?:\.(\d+))?$/.exec(String(price).trim());
  if (!m) throw new Error(`x402: bad price ${price}`);
  const frac = (m[2] || "").padEnd(decimals, "0");
  if (frac.length > decimals) throw new Error(`x402: price ${price} has more than ${decimals} decimals`);
  return String(BigInt(m[1]) * 10n ** BigInt(decimals) + BigInt(frac || "0"));
}

export function requirementsFor(cfg, method, path, url) {
  const key = `${method.toUpperCase()} ${path}`;
  const price = cfg.prices[key];
  if (!price) return null;
  return {
    resource: { url, description: DESCRIPTIONS[key] || key, mimeType: "application/json" },
    requirements: {
      scheme: "exact", network: cfg.network, amount: toAtomic(price, cfg.asset.decimals),
      asset: cfg.asset.asset, payTo: cfg.payTo, maxTimeoutSeconds: cfg.maxTimeoutSeconds,
      extra: { name: cfg.asset.name, version: cfg.asset.version },
    },
  };
}

const b64 = {
  encode: (obj) => btoa(unescape(encodeURIComponent(JSON.stringify(obj)))),
  decode: (s) => JSON.parse(decodeURIComponent(escape(atob(s.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - (s.length % 4)) % 4))))),
};

export function paymentRequired(req, error) {
  const body = { x402Version: 2, ...(error ? { error } : {}), resource: req.resource, accepts: [req.requirements] };
  return new Response(JSON.stringify(body), {
    status: 402,
    headers: { "content-type": "application/json; charset=utf-8", "PAYMENT-REQUIRED": b64.encode(body), "cache-control": "no-store" },
  });
}

function sameRequirements(a, b) {
  return a && b && a.scheme === b.scheme && a.network === b.network && String(a.amount) === String(b.amount)
    && String(a.asset).toLowerCase() === String(b.asset).toLowerCase() && String(a.payTo).toLowerCase() === String(b.payTo).toLowerCase();
}

async function facilitator(cfg, op, payload, requirements) {
  const headers = { "content-type": "application/json" };
  if (cfg.auth) headers.authorization = cfg.auth;
  const r = await fetch(`${cfg.facilitator}/${op}`, { method: "POST", headers, body: JSON.stringify({ x402Version: 2, paymentPayload: payload, paymentRequirements: requirements }) });
  let j = {};
  try { j = await r.json(); } catch { /* non-JSON facilitator error */ }
  if (!r.ok && !("isValid" in j) && !("success" in j)) throw new Error(`facilitator ${op} HTTP ${r.status}`);
  return j;
}

/**
 * Enforces payment for a priced route.  Returns:
 *   {response}   -> a 402 (or error) to send as-is
 *   {payment}    -> verified; call handler, then finalize(payment, response)
 *   null         -> route not priced; nothing to do
 */
export async function requirePayment(cfg, request, path) {
  const req = requirementsFor(cfg, request.method, path, new URL(request.url).toString());
  if (!req) return null;
  const header = request.headers.get("PAYMENT-SIGNATURE") || request.headers.get("X-PAYMENT");
  if (!header) return { response: paymentRequired(req) };
  let payload;
  try { payload = b64.decode(header); } catch { return { response: paymentRequired(req, "Malformed PAYMENT-SIGNATURE header") }; }
  if (payload.x402Version !== 2 || !sameRequirements(payload.accepted, req.requirements)) {
    return { response: paymentRequired(req, "Payment does not match the required amount, asset, network or recipient") };
  }
  let verdict;
  try { verdict = await facilitator(cfg, "verify", payload, req.requirements); }
  catch (e) { return { response: paymentRequired(req, `Payment verification unavailable: ${e.message}`) }; }
  if (!verdict.isValid) return { response: paymentRequired(req, verdict.invalidReason || "Payment invalid") };
  return { payment: { payload, requirements: req.requirements, payer: verdict.payer || null } };
}

/** Settles after a successful handler; attaches PAYMENT-RESPONSE. */
export async function finalize(cfg, payment, response) {
  if (!response.ok) return response;
  let settle;
  try { settle = await facilitator(cfg, "settle", payment.payload, payment.requirements); }
  catch (e) { settle = { success: false, errorReason: `settlement_unavailable: ${e.message}`, transaction: "", network: payment.requirements.network }; }
  if (!settle.success) {
    return new Response(JSON.stringify({ x402Version: 2, error: settle.errorReason || "Settlement failed", accepts: [payment.requirements] }), {
      status: 402, headers: { "content-type": "application/json; charset=utf-8", "PAYMENT-RESPONSE": b64.encode(settle), "cache-control": "no-store" },
    });
  }
  const out = new Response(response.body, response);
  out.headers.set("PAYMENT-RESPONSE", b64.encode({ success: true, transaction: settle.transaction || "", network: settle.network || payment.requirements.network, payer: settle.payer || payment.payer || undefined, amount: payment.requirements.amount }));
  out.headers.set("x-usage-paid", `${payment.requirements.amount} atomic ${cfg.asset.name} on ${cfg.network}`);
  return out;
}
