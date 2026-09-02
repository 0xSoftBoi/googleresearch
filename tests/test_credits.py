"""Unlinkable prepaid credits: blind signatures, redemption, API, client, CLI, edge."""

import json
import shutil
import subprocess

import numpy as np
import pytest
from fastapi.testclient import TestClient

from timesfm3.cli import main
from timesfm3.client import ForecastClient, ForecastServiceError
from timesfm3.credits import (
    CREDIT_COSTS,
    CreditWallet,
    decode_token,
    encode_token,
    full_domain_hash,
    verify_token,
)
from timesfm3.serving.app import create_app
from timesfm3.serving.auth import ApiKey, KeyStore, UsageMeter
from timesfm3.serving.credits import DENOMINATIONS, POINTS_PER_CREDIT, CreditPool


@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    d = tmp_path_factory.mktemp("pool")
    return CreditPool(key_file=str(d / "key.json"), ledger_file=str(d / "ledger.json"), bits=1024)


def _buy(pool, wallet, n):
    pending = wallet.prepare(pool.describe(), n)
    sigs = pool.sign_blinded([format(p.blinded, "x") for p in pending])
    return wallet.finish(pending, sigs)


def test_blind_signature_round_trip_and_unlinkability(pool):
    w = CreditWallet()
    pending = w.prepare(pool.describe(), 3)
    blinded = [format(p.blinded, "x") for p in pending]
    # the server never sees the serials: blinded values are unrelated to FDH(serial)
    for p in pending:
        assert p.blinded != full_domain_hash(p.serial, pool.n)
    sigs = pool.sign_blinded(blinded)
    assert w.finish(pending, sigs) == 3 and len(w) == 3
    for t in w.tokens:
        assert verify_token(t, pool.n, pool.e, pool.kid) is not None
    # tokens carry no trace of the blinded messages the server signed
    for t in w.tokens:
        _, serial, sig = decode_token(t)
        assert format(sig, "x") not in sigs
    assert pool.issued == 3


def test_tampered_or_foreign_tokens_fail(pool):
    w = CreditWallet(); _buy(pool, w, 1)
    t = w.tokens[0]
    kid, serial, sig = decode_token(t)
    assert verify_token(encode_token(kid, serial, sig + 1, pool.n), pool.n, pool.e, pool.kid) is None
    assert verify_token(encode_token("000000000000", serial, sig, pool.n), pool.n, pool.e, pool.kid) is None
    assert verify_token("garbage", pool.n, pool.e, pool.kid) is None
    other = CreditPool(bits=1024)
    assert verify_token(t, other.n, other.e, other.kid) is None
    with pytest.raises(ValueError):
        pool.sign_blinded(["zz"])
    with pytest.raises(ValueError):
        pool.sign_blinded([format(pool.n + 1, "x")])


def test_redeem_costs_double_spend_and_persistence(tmp_path):
    p = CreditPool(key_file=str(tmp_path / "k.json"), ledger_file=str(tmp_path / "l.json"), bits=1024)
    w = CreditWallet(str(tmp_path / "w.json")); _buy(p, w, 6)
    header = w.take(4)
    assert p.redeem(header, 4) == (True, "")
    assert p.redeem(header, 4)[1] == "credit token already spent"
    ok, why = p.redeem(w.take(1), 2)
    assert not ok and "costs 2" in why
    dup = w.tokens[0]
    assert p.redeem(f"{dup},{dup}", 2)[1] == "duplicate credit token in request"
    # a reloaded pool remembers spent serials and counters; the wallet file persists tokens
    p2 = CreditPool(key_file=str(tmp_path / "k.json"), ledger_file=str(tmp_path / "l.json"))
    assert p2.kid == p.kid and p2.redeemed == 4 and p2.issued == 6
    assert p2.redeem(header, 4)[1] == "credit token already spent"
    w2 = CreditWallet(str(tmp_path / "w.json"))
    assert len(w2) == 1 and w2.pool["kid"] == p.kid
    assert p.describe()["pool"] == {"issued": 6, "redeemed": 4, "outstanding": 2}


@pytest.fixture
def client(registry, pool):
    keys = KeyStore([ApiKey("k1", "team", plan="team", monthly_points=100000)])
    return TestClient(create_app(registry=registry, keys=keys, meter=UsageMeter(),
                                 x402_from_env=False, credits=pool))


def _body():
    return {"targets": [{"values": list(np.arange(32.0))}], "horizon": 3, "model": "ewma"}


def test_api_buy_with_plan_and_spend(client, pool):
    info = client.get("/v1/credits/pool").json()
    assert info["kid"] == pool.kid and info["denominations"] == list(DENOMINATIONS)
    w = CreditWallet()
    pending = w.prepare(info, 10)
    r = client.post("/v1/credits/buy/10", json={"blinded": [format(p.blinded, "x") for p in pending]},
                    headers={"x-api-key": "k1"})
    assert r.status_code == 200 and r.headers["x-usage-points"] == str(10 * POINTS_PER_CREDIT)
    assert w.finish(pending, r.json()["blind_signatures"]) == 10

    r = client.post("/v1/forecast", json=_body(), headers={"X-Credit": w.take(1)})
    assert r.status_code == 200 and r.headers["x-usage-points"] == "3"
    bt = {"series": [{"values": list(np.arange(200.0))}], "context": 64, "horizon": 8, "windows": 3, "models": ["ewma"]}
    assert client.post("/v1/backtest", json=bt, headers={"X-Credit": w.take(4)}).status_code == 200
    short = client.post("/v1/backtest", json=bt, headers={"X-Credit": w.take(1)})
    assert short.status_code == 402 and short.headers["x-credit-cost"] == "4"
    spent = w.take(1)
    assert client.post("/v1/forecast", json=_body(), headers={"X-Credit": spent}).status_code == 200
    again = client.post("/v1/forecast", json=_body(), headers={"X-Credit": spent})
    assert again.status_code == 402 and "already spent" in again.json()["detail"]
    assert client.get("/v1/models", headers={"X-Credit": w.take(1)}).status_code == 400  # not a credit route
    assert client.post("/v1/credits/buy/7", json={"blinded": []}, headers={"x-api-key": "k1"}).status_code == 404
    assert client.post("/v1/credits/buy/10", json={"blinded": ["1"]}, headers={"x-api-key": "k1"}).status_code == 422
    assert client.post("/v1/credits/buy/10", json={"blinded": ["1"] * 10}, headers={"X-Credit": w.take(1)}).status_code == 400
    assert client.get("/v1/pricing", headers={"x-api-key": "k1"}).json()["credits"]["costs"] == CREDIT_COSTS


def test_credits_bought_with_x402_bypass_paywall_later(registry, pool, monkeypatch):
    x402 = pytest.importorskip("x402")
    from x402.http.facilitator_client import HTTPFacilitatorClient
    from x402.schemas import SettleResponse, VerifyResponse

    from timesfm3.serving.x402 import X402Config
    from tests.test_x402 import PAY_TO, PAYER, _payment_header, _unb64

    async def verify(self, payload, requirements):
        return VerifyResponse(is_valid=True, payer=PAYER)

    async def settle(self, payload, requirements):
        return SettleResponse(success=True, payer=PAYER, transaction="0x" + "ab" * 32,
                              network=requirements.network, amount=requirements.amount)

    monkeypatch.setattr(HTTPFacilitatorClient, "verify", verify)
    monkeypatch.setattr(HTTPFacilitatorClient, "settle", settle)
    c = TestClient(create_app(registry=registry, keys=KeyStore(), meter=UsageMeter(),
                              x402=X402Config(pay_to=PAY_TO), credits=pool))
    w = CreditWallet()
    pending = w.prepare(c.get("/v1/credits/pool").json(), 10)
    body = {"blinded": [format(p.blinded, "x") for p in pending]}
    challenge = c.post("/v1/credits/buy/10", json=body)
    assert challenge.status_code == 402
    pr = _unb64(challenge.headers["payment-required"])
    req = pr["accepts"][0]
    assert req["amount"] == "40000"  # $0.04 for 10 credits
    # the payload must echo the challenge's resource (official clients do)
    paid = c.post("/v1/credits/buy/10", json=body, headers={"PAYMENT-SIGNATURE": _payment_header(req, pr["resource"])})
    assert paid.status_code == 200 and "payment-response" in {k.lower() for k in paid.headers}
    w.finish(pending, paid.json()["blind_signatures"])
    # spending a credit needs no payment header and no key
    r = c.post("/v1/forecast", json=_body(), headers={"X-Credit": w.take(1)})
    assert r.status_code == 200
    assert c.post("/v1/forecast", json=_body()).status_code == 402  # still paywalled without credits


def test_client_and_cli(registry, pool, tmp_path, monkeypatch, capsys):
    keys = KeyStore([ApiKey("k1", "team", plan="team", monthly_points=100000)])
    http = TestClient(create_app(registry=registry, keys=keys, meter=UsageMeter(),
                                 x402_from_env=False, credits=pool))

    def _request(self, method, path, payload=None):
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        elif self.credits is not None and method == "POST" and not path.startswith("/v1/credits/"):
            headers["x-credit"] = self.credits.take(CreditWallet.cost_of(method, path))
        r = http.request(method, path, content=json.dumps(payload) if payload is not None else None, headers=headers)
        if r.status_code >= 400:
            raise ForecastServiceError(r.status_code, r.json().get("detail", r.text))
        return r.json()

    monkeypatch.setattr(ForecastClient, "_request", _request)
    wallet_path = str(tmp_path / "credits.json")
    assert main(["credits", "buy", "--api", "http://t", "--count", "10", "--api-key", "k1", "--wallet", wallet_path]) == 0
    assert "bought 10 credits" in capsys.readouterr().out
    assert main(["credits", "status", "--wallet", wallet_path]) == 0
    assert "10 unspent" in capsys.readouterr().out

    anon = ForecastClient("http://t", credits=CreditWallet(wallet_path))
    res = anon.forecast([np.arange(32.0)], horizon=3, model="ewma")
    assert res.point.shape == (1, 3) and len(anon.credits) == 9
    anon.backtest([np.arange(200.0)], context=64, horizon=8, windows=3, models=["ewma"])
    assert len(anon.credits) == 5
    assert anon.pricing()["credits"]["price_per_credit_usd"] == pool.price_per_credit_usd
    with pytest.raises(ValueError):
        anon.credits.take(6)


node = shutil.which("node")


@pytest.mark.skipif(node is None, reason="node not installed")
def test_edge_lets_credit_holders_through_paywall():
    script = r"""
import worker from './cloudflare/src/worker.js';
const env = { API_ORIGIN: 'https://up.test', X402_PAY_TO: '0x000000000000000000000000000000000000dEaD', ASSETS: { fetch: async () => new Response('nf', {status: 404}) } };
const seen = [];
globalThis.fetch = async (url, init) => { seen.push([url, init && init.headers && init.headers.get && init.headers.get('x-credit')]); return new Response(JSON.stringify({ ok: true }), { headers: { 'content-type': 'application/json' } }); };
globalThis.caches = { default: { match: async () => undefined, put: async () => {} } };
const ctx = { waitUntil() {} };
const body = JSON.stringify({ targets: [{ values: [1,2,3,4,5,6,7,8] }], horizon: 2 });
const r1 = await worker.fetch(new Request('https://edge.test/v1/forecast', { method: 'POST', headers: { 'content-type': 'application/json' }, body }), env, ctx);
const r2 = await worker.fetch(new Request('https://edge.test/v1/forecast', { method: 'POST', headers: { 'content-type': 'application/json', 'X-Credit': 'kid.serial.sig' }, body }), env, ctx);
console.log(JSON.stringify({ anon: r1.status, credit: r2.status, forwarded: seen.map(s => s[1]), allow: r2.headers.get('access-control-allow-headers') }));
"""
    res = subprocess.run([node, "--input-type=module", "-e", script], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["anon"] == 402 and out["credit"] == 200
    assert out["forwarded"] == ["kid.serial.sig"] and "x-credit" in out["allow"]
