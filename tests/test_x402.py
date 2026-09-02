"""x402 pay-per-call: the FastAPI paywall and the Worker's implementation.

The facilitator is mocked (no chain access in CI). The Worker's 402 is
validated with the official Python models so the two implementations stay
interoperable.
"""

import base64
import json
import shutil
import subprocess
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("x402")
from x402.http.facilitator_client import HTTPFacilitatorClient  # noqa: E402
from x402.http.utils import decode_payment_required_header  # noqa: E402
from x402.schemas import PaymentPayload, PaymentRequired, SettleResponse, VerifyResponse  # noqa: E402

from timesfm3.serving.app import create_app  # noqa: E402
from timesfm3.serving.auth import ApiKey, KeyStore, UsageMeter  # noqa: E402
from timesfm3.serving.x402 import DEFAULT_PRICES, X402Config  # noqa: E402

PAY_TO = "0x000000000000000000000000000000000000dEaD"
PAYER = "0x00000000000000000000000000000000000000A1"


def _b64(obj) -> str:
    return base64.b64encode(json.dumps(obj).encode()).decode()


def _unb64(s: str) -> dict:
    return json.loads(base64.b64decode(s + "=" * (-len(s) % 4)))


def _payment_header(requirements: dict, resource: dict | None = None) -> str:
    payload = {
        "x402Version": 2,
        "resource": resource or {},
        "accepted": requirements,
        "payload": {
            "signature": "0x" + "ab" * 65,
            "authorization": {
                "from": PAYER, "to": requirements["payTo"], "value": requirements["amount"],
                "validAfter": "0", "validBefore": str(int(time.time()) + 600),
                "nonce": "0x" + "11" * 32,
            },
        },
    }
    return _b64(payload)


@pytest.fixture
def facilitator(monkeypatch):
    calls = {"verify": [], "settle": []}

    async def verify(self, payload, requirements):
        calls["verify"].append((payload, requirements))
        return VerifyResponse(is_valid=True, payer=PAYER)

    async def settle(self, payload, requirements):
        calls["settle"].append((payload, requirements))
        return SettleResponse(success=True, payer=PAYER, transaction="0x" + "cd" * 32,
                              network=requirements.network, amount=requirements.amount)

    monkeypatch.setattr(HTTPFacilitatorClient, "verify", verify)
    monkeypatch.setattr(HTTPFacilitatorClient, "settle", settle)
    return calls


@pytest.fixture
def paid_client(registry):
    keys = KeyStore([ApiKey("k1", "team", plan="team", monthly_points=100000)])
    app = create_app(registry=registry, keys=keys, meter=UsageMeter(),
                     x402=X402Config(pay_to=PAY_TO))
    return TestClient(app)


def _body():
    return {"targets": [{"values": list(np.arange(32.0))}], "horizon": 3, "model": "ewma"}


def test_402_challenge_is_spec_shaped(paid_client):
    r = paid_client.post("/v1/forecast", json=_body())
    assert r.status_code == 402
    pr = decode_payment_required_header(r.headers["payment-required"])
    assert isinstance(pr, PaymentRequired) and pr.x402_version == 2
    req = pr.accepts[0]
    assert req.scheme == "exact" and req.network == "eip155:84532"
    assert req.amount == "5000" and req.pay_to == PAY_TO  # $0.005 in atomic USDC
    assert req.asset.lower() == "0x036cbd53842c5426634e7929541ec2318f3dcf7e"
    assert pr.resource.url.endswith("/v1/forecast")
    # unpriced routes are untouched by the paywall
    assert paid_client.get("/v1/models").status_code == 401  # keys configured, no key
    assert paid_client.get("/healthz").status_code == 200


def test_paid_request_is_served_and_settled(paid_client, facilitator):
    challenge = paid_client.post("/v1/forecast", json=_body())
    pr = _unb64(challenge.headers["payment-required"])
    header = _payment_header(pr["accepts"][0], pr.get("resource"))
    r = paid_client.post("/v1/forecast", json=_body(), headers={"PAYMENT-SIGNATURE": header})
    assert r.status_code == 200, r.text
    assert r.json()["model"] == "ewma"
    assert len(facilitator["verify"]) == 1 and len(facilitator["settle"]) == 1
    settled = _unb64(r.headers["payment-response"])
    assert settled["success"] is True and settled["transaction"].startswith("0x")
    assert r.headers["x-usage-points"] == "3"
    usage = paid_client.get("/v1/usage", headers={"x-api-key": "k1"}).json()
    assert usage["name"] == "team"  # plan usage is separate from the anonymous paid identity


def test_wrong_amount_is_rejected_before_facilitator(paid_client, facilitator):
    challenge = paid_client.post("/v1/forecast", json=_body())
    req = _unb64(challenge.headers["payment-required"])["accepts"][0]
    req["amount"] = "1"
    r = paid_client.post("/v1/forecast", json=_body(), headers={"PAYMENT-SIGNATURE": _payment_header(req)})
    assert r.status_code == 402
    assert facilitator["settle"] == []


def test_key_holders_bypass_paywall(paid_client, facilitator):
    r = paid_client.post("/v1/forecast", json=_body(), headers={"x-api-key": "k1"})
    assert r.status_code == 200 and facilitator["verify"] == []
    assert r.headers["x-usage-remaining"] == str(100000 - 3)


def test_pricing_endpoint_and_env_config(paid_client, monkeypatch):
    p = paid_client.get("/v1/pricing", headers={"x-api-key": "k1"}).json()
    assert p["x402"]["enabled"] and p["x402"]["prices_usd"] == DEFAULT_PRICES
    assert p["x402"]["facilitator"] == "https://x402.org/facilitator"
    monkeypatch.setenv("TIMESFM3_X402_PAY_TO", PAY_TO)
    monkeypatch.setenv("TIMESFM3_X402_NETWORK", "eip155:8453")
    monkeypatch.setenv("TIMESFM3_X402_PRICES", '{"POST /v1/forecast": "$0.01"}')
    cfg = X402Config.from_env()
    assert cfg.mainnet and cfg.facilitator.startswith("https://api.cdp.coinbase.com")
    assert cfg.prices["POST /v1/forecast"] == "$0.01" and cfg.prices["POST /v1/backtest"] == "$0.02"
    monkeypatch.delenv("TIMESFM3_X402_PAY_TO")
    assert X402Config.from_env() is None


# ---- the Worker's implementation, run under Node with a fake env -----------

node = shutil.which("node")


@pytest.mark.skipif(node is None, reason="node not installed")
def test_worker_x402_flow_matches_spec():
    script = r"""
import worker from './cloudflare/src/worker.js';
import fs from 'fs';
const files = { '/models/starter-small.json': 'cloudflare/public/models/starter-small.json' };
const env = {
  ASSETS: { fetch: async (req) => { const p = new URL(req.url).pathname; const f = files[p]; return f ? new Response(fs.readFileSync(f)) : new Response('nf', {status: 404}); } },
  X402_PAY_TO: '0x000000000000000000000000000000000000dEaD', X402_PAYWALL_EDGE_NATIVE: '1',
};
const calls = [];
globalThis.fetch = async (url, init) => {
  calls.push(url);
  const body = JSON.parse(init.body);
  if (url.endsWith('/verify')) return new Response(JSON.stringify({ isValid: true, payer: body.paymentPayload.payload.authorization.from }));
  if (url.endsWith('/settle')) return new Response(JSON.stringify({ success: true, transaction: '0x' + 'ef'.repeat(32), network: body.paymentRequirements.network, payer: body.paymentPayload.payload.authorization.from }));
  return new Response('nope', { status: 500 });
};
const ctx = { waitUntil() {} };
const body = JSON.stringify({ targets: [{ values: Array.from({length: 32}, (_, i) => i) }], horizon: 3, model: 'ewma' });
const post = (headers) => new Request('https://edge.test/v1/forecast', { method: 'POST', headers: { 'content-type': 'application/json', ...headers }, body });
const out = {};
const r1 = await worker.fetch(post({}), env, ctx);
out.challenge = { status: r1.status, header: r1.headers.get('PAYMENT-REQUIRED'), body: await r1.json() };
const req = out.challenge.body.accepts[0];
const payload = { x402Version: 2, resource: out.challenge.body.resource, accepted: req, payload: { signature: '0x' + 'ab'.repeat(65), authorization: { from: '0x00000000000000000000000000000000000000A1', to: req.payTo, value: req.amount, validAfter: '0', validBefore: String(Math.floor(Date.now()/1000) + 600), nonce: '0x' + '11'.repeat(32) } } };
const sig = btoa(JSON.stringify(payload));
const r2 = await worker.fetch(post({ 'PAYMENT-SIGNATURE': sig }), env, ctx);
out.paid = { status: r2.status, response: r2.headers.get('PAYMENT-RESPONSE'), paid: r2.headers.get('x-usage-paid'), expose: r2.headers.get('access-control-expose-headers'), model: (await r2.json()).model, calls };
const bad = { ...payload, accepted: { ...req, amount: '1' } };
const r3 = await worker.fetch(post({ 'PAYMENT-SIGNATURE': btoa(JSON.stringify(bad)) }), env, ctx);
out.wrongAmount = { status: r3.status, error: (await r3.json()).error };
const r4 = await worker.fetch(new Request('https://edge.test/v1/pricing'), env, ctx);
out.pricing = await r4.json();
const r5 = await worker.fetch(new Request('https://edge.test/v1/models'), { ...env }, ctx);
out.freeGet = r5.status;
console.log(JSON.stringify(out));
"""
    res = subprocess.run([node, "--input-type=module", "-e", script], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["challenge"]["status"] == 402
    pr = decode_payment_required_header(out["challenge"]["header"])  # official parser accepts it
    assert isinstance(pr, PaymentRequired) and pr.accepts[0].amount == "5000"
    assert pr.accepts[0].pay_to == PAY_TO and pr.accepts[0].extra == {"name": "USDC", "version": "2"}
    assert out["challenge"]["body"]["accepts"][0] == json.loads(pr.accepts[0].model_dump_json(by_alias=True, exclude_none=True))
    assert out["paid"]["status"] == 200 and out["paid"]["model"] == "ewma"
    settled = _unb64(out["paid"]["response"])
    assert settled["success"] and settled["transaction"].startswith("0x") and settled["amount"] == "5000"
    assert out["paid"]["paid"].startswith("5000 atomic USDC")
    assert "payment-response" in out["paid"]["expose"]
    assert [u.rsplit("/", 1)[1] for u in out["paid"]["calls"]] == ["verify", "settle"]
    assert out["wrongAmount"]["status"] == 402 and "match" in out["wrongAmount"]["error"]
    assert out["pricing"]["x402"]["enabled"] and out["pricing"]["x402"]["prices_usd"]["POST /v1/forecast"] == "$0.005"
    assert out["freeGet"] == 200
    # the payment payload the Worker forwarded is a valid v2 PaymentPayload
    PaymentPayload.model_validate({"x402Version": 2, "accepted": pr.accepts[0].model_dump(by_alias=True),
                                   "payload": {"signature": "0x00", "authorization": {}}})
