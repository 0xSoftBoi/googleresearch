"""Privacy Pass (RFC 9576/9577/9578): structures, interop with Cloudflare's library,
the service's issuer/origin, the client wallet and CLI, and the Worker at the edge."""

import json
import os
import shutil
import subprocess

import numpy as np
import pytest
from fastapi.testclient import TestClient

from timesfm3 import blindrsa as B
from timesfm3 import privacypass as PP
from timesfm3.cli import main
from timesfm3.client import ForecastClient, ForecastServiceError
from timesfm3.credits import CreditWallet
from timesfm3.serving.app import create_app
from timesfm3.serving.auth import ApiKey, KeyStore, UsageMeter
from timesfm3.serving.privacypass import DENOMINATIONS, POINTS_PER_TOKEN, PrivacyPassService, generate_private_jwk

NODE = shutil.which("node")
PP_TS = "/home/user/googleresearch/cloudflare/node_modules/@cloudflare/privacypass-ts/lib/src/index.js"
HAS_TS = NODE is not None and os.path.exists(PP_TS)


def _node(script: str, payload: dict) -> dict:
    res = subprocess.run([NODE, "--input-type=module", "-e", script], input=json.dumps(payload),
                         capture_output=True, text=True, cwd="/home/user/googleresearch/cloudflare")
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def jwk():
    return generate_private_jwk(bits=2048)


@pytest.fixture(scope="module")
def priv(jwk):
    return B.private_from_jwk(jwk)


def test_structures_and_headers_round_trip(priv):
    ch = PP.TokenChallenge(PP.TOKEN_TYPE_BLIND_RSA, "issuer.example", b"", "api.example")
    assert PP.TokenChallenge.deserialize(ch.serialize()) == ch
    with pytest.raises(ValueError):
        PP.TokenChallenge.deserialize(PP.TokenChallenge(2, "i", b"x" * 5, "").serialize())
    client, issuer, origin = PP.Client(priv.public), PP.Issuer([priv]), PP.Origin([priv.public])
    req, st = client.create_request(ch)
    assert PP.TokenRequest.deserialize(req.serialize()) == req and req.truncated_token_key_id == PP.token_key_id(priv.public)[-1]
    tok = client.finalize(st, PP.TokenResponse.deserialize(issuer.issue(req).serialize()))
    assert PP.Token.deserialize(tok.serialize()) == tok and origin.verify(tok, ch)
    assert not origin.verify(tok, PP.TokenChallenge(2, "issuer.example", b"", "other.example"))
    bad = PP.Token(tok.token_type, tok.nonce, tok.challenge_digest, tok.token_key_id, bytes([tok.authenticator[0] ^ 1]) + tok.authenticator[1:])
    assert not origin.verify(bad, ch)
    hdr = PP.www_authenticate(ch, priv.public, 600)
    assert "=" in hdr.split("challenge=")[1].split(",")[0]  # RFC 9577: padded base64url
    p = PP.parse_www_authenticate(hdr)[0]
    assert p["challenge"] == ch and p["token_key"] == priv.public and p["max_age"] == 600
    assert PP.parse_authorization(PP.authorization(tok)) == tok
    assert PP.parse_authorization("Bearer abc") is None
    assert PP.public_key_from_spki(PP.spki_rsassa_pss(priv.public)) == priv.public
    assert PP.issuer_directory([priv.public])["token-keys"][0]["token-type"] == 2


def test_batched_issuance_format(priv):
    ch = PP.TokenChallenge(2, "i", b"", "o")
    client, issuer = PP.Client(priv.public), PP.Issuer([priv])
    reqs, states = zip(*(client.create_request(ch) for _ in range(3)))
    body = PP.serialize_batched_request(list(reqs))
    assert PP.count_batched_request(body) == 3 and PP.deserialize_batched_request(body) == list(reqs)
    for v in (5, 100, 20000, 3_000_000):
        enc = PP.quic_varint_encode(v)
        assert PP.quic_varint_decode(enc) == (v, len(enc))
    responses = [issuer.issue(r) for r in reqs]
    responses[1] = None
    out = PP.deserialize_batched_response(PP.serialize_batched_response(responses))
    assert out[1] is None and out[0] == responses[0] and out[2] == responses[2]
    with pytest.raises(ValueError):
        PP.deserialize_batched_request(body[:-1])


@pytest.mark.skipif(not HAS_TS, reason="node / privacypass-ts not installed")
def test_interop_with_cloudflare_privacypass_ts(jwk, priv):
    """Every role combination across Python and @cloudflare/privacypass-ts, including batches."""
    ch = PP.TokenChallenge(2, "issuer.example", b"", "api.example")
    client = PP.Client(priv.public)
    req, st = client.create_request(ch)
    reqs, states = zip(*(client.create_request(ch) for _ in range(3)))
    issuer = PP.Issuer([priv])
    py_req, py_st = client.create_request(ch)
    py_tok = client.finalize(py_st, issuer.issue(py_req))
    out = _node(r"""
import { publicVerif, genericBatched, WWWAuthenticateHeader, AuthorizationHeader, TOKEN_TYPES } from '@cloudflare/privacypass-ts';
let input=''; process.stdin.on('data',d=>input+=d); process.stdin.on('end',async()=>{
  const p=JSON.parse(input); const b64d=(s)=>Uint8Array.from(Buffer.from(s,'base64url')); const b64e=(u)=>Buffer.from(u).toString('base64url');
  const algo={name:'RSA-PSS',hash:'SHA-384'};
  const privateKey=await crypto.subtle.importKey('jwk',{...p.jwk,alg:'PS384',ext:true},algo,true,['sign']);
  const publicKey=await crypto.subtle.importKey('jwk',{kty:'RSA',n:p.jwk.n,e:p.jwk.e,alg:'PS384',ext:true},algo,true,['verify']);
  const issuer=new publicVerif.Issuer(publicVerif.BlindRSAMode.PSS,'issuer.example',privateKey,publicKey);
  const out={ spki: b64e(await publicVerif.getPublicKeyBytes(publicKey)), kid: b64e(await issuer.tokenKeyID()) };
  out.single=b64e((await issuer.issue(publicVerif.TokenRequest.deserialize(TOKEN_TYPES.BLIND_RSA, b64d(p.req)))).serialize());
  const batched=new genericBatched.Issuer(issuer);
  out.batch=b64e((await batched.issue(genericBatched.BatchedTokenRequest.deserialize(b64d(p.batch)))).serialize());
  const hdr=WWWAuthenticateHeader.parse(p.www)[0]; const client=new publicVerif.Client(publicVerif.BlindRSAMode.PSS);
  const tok=await client.finalize(await issuer.issue(await client.createTokenRequest(hdr.challenge, hdr.tokenKey)));
  out.js_auth=new AuthorizationHeader(tok).toString();
  const origin=new publicVerif.Origin(publicVerif.BlindRSAMode.PSS,['api.example']);
  out.js_verifies_py=await origin.verify(AuthorizationHeader.parse(TOKEN_TYPES.BLIND_RSA, p.py_auth)[0].token, publicKey);
  console.log(JSON.stringify(out)); });
""", {"jwk": jwk, "req": PP.b64e(req.serialize()), "batch": PP.b64e(PP.serialize_batched_request(list(reqs))),
      "www": PP.www_authenticate(ch, priv.public), "py_auth": PP.authorization(py_tok)})
    assert PP.b64d(out["spki"]) == PP.spki_rsassa_pss(priv.public)
    assert PP.b64d(out["kid"]) == PP.token_key_id(priv.public)
    origin = PP.Origin([priv.public])
    assert origin.verify(client.finalize(st, PP.TokenResponse.deserialize(PP.b64d(out["single"]))), ch)
    batch = PP.deserialize_batched_response(PP.b64d(out["batch"]))
    assert len(batch) == 3 and all(origin.verify(client.finalize(s, r), ch) for s, r in zip(states, batch))
    assert origin.verify(PP.parse_authorization(out["js_auth"]), ch)
    assert out["js_verifies_py"] is True


# ---- the service --------------------------------------------------------------

@pytest.fixture(scope="module")
def service(tmp_path_factory, jwk):
    d = tmp_path_factory.mktemp("pp")
    key = d / "key.json"; key.write_text(json.dumps(jwk))
    return PrivacyPassService(key_file=str(key), ledger_file=str(d / "ledger.json"), issuer_name="api.test", origin_name="api.test")


@pytest.fixture
def client(registry, service):
    keys = KeyStore([ApiKey("k1", "team", plan="team", monthly_points=100000)])
    return TestClient(create_app(registry=registry, keys=keys, meter=UsageMeter(), x402_from_env=False, privacy_pass=service))


def _body():
    return {"targets": [{"values": list(np.arange(32.0))}], "horizon": 3, "model": "ewma"}


def test_service_issue_redeem_and_double_spend(client, service):
    r = client.post("/v1/forecast", json=_body())
    assert r.status_code == 401 and r.headers["www-authenticate"].startswith("PrivateToken challenge=")
    assert client.get("/token-request/challenge").headers["www-authenticate"] == r.headers["www-authenticate"]
    assert "www-authenticate" not in {k.lower() for k in client.get("/v1/models").headers}  # unpriced: no challenge
    www = r.headers["www-authenticate"]
    d = client.get(PP.ISSUER_DIRECTORY_PATH)
    assert d.status_code == 200 and d.headers["content-type"].startswith(PP.MEDIA_DIRECTORY)
    assert d.json()["issuer-request-uri"] == "/token-request" and PP.b64d(d.json()["token-keys"][0]["token-key"]) == PP.spki_rsassa_pss(service.public)

    w = CreditWallet()
    body, pending, batched = w.prepare(www, 1)
    r = client.post("/token-request", content=body, headers={"content-type": PP.MEDIA_REQUEST, "x-api-key": "k1"})
    assert r.status_code == 200 and r.headers["content-type"].startswith(PP.MEDIA_RESPONSE) and r.headers["x-usage-points"] == str(POINTS_PER_TOKEN)
    assert w.finish(pending, r.content, batched) == 1

    body, pending, batched = w.prepare(www, 10)
    r = client.post("/token-request/batch/10", content=body, headers={"content-type": PP.MEDIA_BATCH_REQUEST, "x-api-key": "k1"})
    assert r.status_code == 200 and r.headers["x-tokens-issued"] == "10" and r.headers["x-usage-points"] == str(10 * POINTS_PER_TOKEN)
    assert w.finish(pending, r.content, batched) == 10 and len(w) == 11
    # generic batch by media type at the standard URI too
    body, pending, batched = w.prepare(www, 3)
    r = client.post("/token-request", content=body, headers={"content-type": PP.MEDIA_BATCH_REQUEST, "x-api-key": "k1"})
    assert r.status_code == 200 and w.finish(pending, r.content, batched) == 3

    auth = w.take()
    r = client.post("/v1/forecast", json=_body(), headers={"authorization": auth})
    assert r.status_code == 200 and r.headers["x-usage-points"] == "3"
    again = client.post("/v1/forecast", json=_body(), headers={"authorization": auth})
    assert again.status_code == 401 and "already spent" in again.json()["detail"] and again.headers["www-authenticate"]
    assert client.post("/v1/backtest", json={"series": [{"values": list(np.arange(200.0))}], "context": 64, "horizon": 8, "windows": 3, "models": ["ewma"]},
                       headers={"authorization": w.take()}).status_code == 200
    assert client.get("/v1/models", headers={"authorization": w.take()}).status_code == 400  # not a priced route
    assert client.post("/v1/forecast", json=_body(), headers={"authorization": 'PrivateToken token="AAAA"'}).status_code == 401
    assert client.post("/token-request/batch/7", content=b"", headers={"content-type": PP.MEDIA_BATCH_REQUEST, "x-api-key": "k1"}).status_code == 404
    assert client.post("/token-request", content=b"xx", headers={"content-type": "text/plain", "x-api-key": "k1"}).status_code == 415
    assert client.post("/token-request", content=b"xx", headers={"content-type": PP.MEDIA_REQUEST, "x-api-key": "k1"}).status_code == 422
    assert client.post("/token-request", content=b"xx", headers={"content-type": PP.MEDIA_REQUEST, "authorization": w.take()}).status_code == 400
    pr = client.get("/v1/pricing", headers={"x-api-key": "k1"}).json()["privacy_pass"]
    assert pr["denominations"] == list(DENOMINATIONS) and pr["pool"]["issued"] == 14


def test_tokens_bought_with_x402_bypass_paywall(registry, service, monkeypatch):
    pytest.importorskip("x402")
    from x402.http.facilitator_client import HTTPFacilitatorClient
    from x402.schemas import SettleResponse, VerifyResponse

    from tests.test_x402 import PAY_TO, PAYER, _payment_header, _unb64
    from timesfm3.serving.x402 import X402Config

    async def verify(self, payload, requirements):
        return VerifyResponse(is_valid=True, payer=PAYER)

    async def settle(self, payload, requirements):
        return SettleResponse(success=True, payer=PAYER, transaction="0x" + "ab" * 32, network=requirements.network, amount=requirements.amount)

    monkeypatch.setattr(HTTPFacilitatorClient, "verify", verify)
    monkeypatch.setattr(HTTPFacilitatorClient, "settle", settle)
    c = TestClient(create_app(registry=registry, keys=KeyStore(), meter=UsageMeter(), x402=X402Config(pay_to=PAY_TO), privacy_pass=service))
    challenge = c.post("/v1/forecast", json=_body())
    assert challenge.status_code == 402 and challenge.headers["www-authenticate"].startswith("PrivateToken")  # both ways to pay
    w = CreditWallet()
    body, pending, batched = w.prepare(challenge.headers["www-authenticate"], 10)
    r = c.post("/token-request/batch/10", content=body, headers={"content-type": PP.MEDIA_BATCH_REQUEST})
    assert r.status_code == 402
    req = _unb64(r.headers["payment-required"])
    assert req["accepts"][0]["amount"] == "40000"  # $0.04 for 10 tokens
    paid = c.post("/token-request/batch/10", content=body, headers={"content-type": PP.MEDIA_BATCH_REQUEST, "PAYMENT-SIGNATURE": _payment_header(req["accepts"][0], req.get("resource"))})
    assert paid.status_code == 200 and w.finish(pending, paid.content, batched) == 10
    assert c.post("/v1/forecast", json=_body(), headers={"authorization": w.take()}).status_code == 200
    assert c.post("/v1/forecast", json=_body()).status_code == 402


def test_rotation_keeps_old_tokens_redeemable(tmp_path, registry):
    old = PrivacyPassService(key_file=str(tmp_path / "old.json"), issuer_name="i", origin_name="o")
    ch = old.challenge("o")
    client = PP.Client(old.public); req, st = client.create_request(ch)
    tok = client.finalize(st, PP.TokenResponse.deserialize(old.issue(req.serialize())))
    new = PrivacyPassService(key_file=str(tmp_path / "new.json"), old_key_files=[str(tmp_path / "old.json")], issuer_name="i", origin_name="o")
    assert [k["issuing"] for k in new.describe()["keys"]] == [True, False] and len(new.directory()["token-keys"]) == 2
    assert new.redeem(PP.authorization(tok), "o") == (True, "")
    assert new.redeem(PP.authorization(tok), "o")[1] == "token already spent"
    reloaded = PrivacyPassService(key_file=str(tmp_path / "new.json"), old_key_files=[str(tmp_path / "old.json")], issuer_name="i", origin_name="o")
    assert reloaded.kid == new.kid


def test_client_and_cli(registry, service, tmp_path, monkeypatch, capsys):
    keys = KeyStore([ApiKey("k1", "team", plan="team", monthly_points=100000)])
    http = TestClient(create_app(registry=registry, keys=keys, meter=UsageMeter(), x402_from_env=False, privacy_pass=service))

    def _request(self, method, path, payload=None):
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        elif self.credits is not None and CreditWallet.is_priced(method, path):
            headers["authorization"] = self.credits.take()
        r = http.request(method, path, content=json.dumps(payload) if payload is not None else None, headers=headers)
        if r.status_code >= 400:
            raise ForecastServiceError(r.status_code, r.json().get("detail", r.text))
        return r.json()

    def _request_raw(self, method, path, data=None, headers=None):
        h = dict(headers or {})
        if self.api_key:
            h["x-api-key"] = self.api_key
        r = http.request(method, path, content=data, headers=h)
        return r.status_code, {k.lower(): v for k, v in r.headers.items()}, r.content

    monkeypatch.setattr(ForecastClient, "_request", _request)
    monkeypatch.setattr(ForecastClient, "_request_raw", _request_raw)
    wallet_path = str(tmp_path / "wallet.json")
    assert main(["credits", "buy", "--api", "http://t", "--count", "10", "--api-key", "k1", "--wallet", wallet_path]) == 0
    assert "bought 10" in capsys.readouterr().out
    assert main(["credits", "buy", "--api", "http://t", "--count", "1", "--api-key", "k1", "--wallet", wallet_path]) == 0
    capsys.readouterr()
    assert main(["credits", "status", "--wallet", wallet_path]) == 0
    assert "11 unspent" in capsys.readouterr().out
    anon = ForecastClient("http://t", credits=CreditWallet(wallet_path))
    res = anon.forecast([np.arange(32.0)], horizon=3, model="ewma")
    assert res.point.shape == (1, 3) and len(anon.credits) == 10
    anon.backtest([np.arange(200.0)], context=64, horizon=8, windows=3, models=["ewma"])
    assert len(anon.credits) == 9
    assert anon.pricing()["privacy_pass"]["price_per_token_usd"] == service.price_per_token_usd


# ---- the Worker ---------------------------------------------------------------

@pytest.mark.skipif(not HAS_TS, reason="node / privacypass-ts not installed")
def test_worker_privacy_pass_end_to_end_and_interchangeable(jwk, service):
    """The Hono Worker issues (single + batch) and redeems standard tokens with a D1
    shim; tokens are interchangeable with the Python service under the same key."""
    ch = service.challenge("api.test")
    client = PP.Client(service.public); req, st = client.create_request(ch)
    py_tok = client.finalize(st, PP.TokenResponse.deserialize(service.issue(req.serialize())))
    w = CreditWallet()
    body1, pend1, _ = w.prepare(service.challenge_header("api.test"), 1)
    body10, pend10, _ = w.prepare(service.challenge_header("api.test"), 10)
    out = _node(r"""
import app from './src/index.js';
let input=''; process.stdin.on('data',d=>input+=d); process.stdin.on('end',async()=>{
  const p=JSON.parse(input); const b64d=(s)=>Buffer.from(s,'base64url'); const b64e=(u)=>Buffer.from(u).toString('base64url');
  const spent=new Map(), stats=new Map();
  const stmt=(sql)=>({bind:(...a)=>({run:async()=>run(sql,a),first:async()=>first(sql,a)}),run:async()=>run(sql,[]),first:async()=>first(sql,[])});
  const run=async(sql,a)=>{ if(sql.startsWith('INSERT OR IGNORE INTO spent')){ if(spent.has(a[0])) return {meta:{changes:0}}; spent.set(a[0],a); return {meta:{changes:1}}; }
    if(sql.startsWith('INSERT INTO stats')){ const s=stats.get(a[0])||{issued:0,redeemed:0}; if(sql.includes('issued = issued')) s.issued+=a[1]; else s.redeemed+=1; stats.set(a[0],s); return {meta:{changes:1}}; } return {meta:{}}; };
  const first=async()=>{ let i=0,r=0; for(const s of stats.values()){i+=s.issued;r+=s.redeemed;} return {issued:i,redeemed:r}; };
  const env={ PRIVACY_PASS_PRIVATE_JWK: JSON.stringify(p.jwk), PRIVACY_PASS_ORIGIN:'api.test', PRIVACY_PASS_ISSUER_NAME:'api.test', PRIVACY_PASS_DB:{prepare:stmt}, ASSETS:{fetch:async(req)=>{ const u=new URL(req.url).pathname; if(u==='/models/starter-small.json') return new Response(require('fs').readFileSync('public/models/starter-small.json')); return new Response('nf',{status:404}); }} };
  const ctx={waitUntil(){}, passThroughOnException(){}};
  const call=(path,init={})=>app.fetch(new Request('https://api.test'+path,init),env,ctx);
  const out={};
  const dir=await call('/.well-known/private-token-issuer-directory'); out.dir={status:dir.status, ct:dir.headers.get('content-type'), body:await dir.json()};
  const chal=await call('/token-request/challenge'); out.challenge={status:chal.status, www:chal.headers.get('www-authenticate')};
  const s1=await call('/token-request',{method:'POST',headers:{'content-type':'application/private-token-request'},body:b64d(p.body1)}); out.single={status:s1.status, ct:s1.headers.get('content-type'), body:b64e(new Uint8Array(await s1.arrayBuffer()))};
  const s10=await call('/token-request/batch/10',{method:'POST',headers:{'content-type':'application/private-token-generic-batch-request'},body:b64d(p.body10)}); out.batch={status:s10.status, issued:s10.headers.get('x-tokens-issued'), body:b64e(new Uint8Array(await s10.arrayBuffer()))};
  const fc=(auth)=>call('/v1/forecast',{method:'POST',headers:{'content-type':'application/json',authorization:auth},body:JSON.stringify({targets:[{values:Array.from({length:32},(_,i)=>i)}],horizon:2,model:'ewma'})});
  const r1=await fc(p.py_auth); out.pyTokenAtEdge={status:r1.status, model:(await r1.json()).model};
  const r2=await fc(p.py_auth); out.pyTokenAgain={status:r2.status, detail:(await r2.json()).detail, www:Boolean(r2.headers.get('www-authenticate'))};
  const r3=await fc('PrivateToken token="AAAA"'); out.bad=r3.status;
  out.stats=await (await call('/token-request/stats')).json();
  out.pricing=(await (await call('/v1/pricing')).json()).privacy_pass.enabled;
  console.log(JSON.stringify(out)); });
""", {"jwk": jwk, "body1": PP.b64e(body1), "body10": PP.b64e(body10), "py_auth": PP.authorization(py_tok)})
    assert out["dir"]["status"] == 200 and out["dir"]["ct"] == PP.MEDIA_DIRECTORY
    assert PP.b64d(out["dir"]["body"]["token-keys"][0]["token-key"]) == PP.spki_rsassa_pss(service.public)
    assert out["challenge"]["status"] == 401
    assert PP.parse_www_authenticate(out["challenge"]["www"])[0]["challenge"] == ch  # identical challenge on both
    assert out["single"]["status"] == 200 and out["single"]["ct"] == PP.MEDIA_RESPONSE
    assert w.finish(pend1, PP.b64d(out["single"]["body"]), False) == 1
    assert out["batch"]["status"] == 200 and out["batch"]["issued"] == "10"
    assert w.finish(pend10, PP.b64d(out["batch"]["body"]), True) == 10
    # edge-issued tokens redeem in the Python service
    assert service.redeem(w.take(), "api.test") == (True, "")
    assert out["pyTokenAtEdge"] == {"status": 200, "model": "ewma"}
    assert out["pyTokenAgain"]["status"] == 401 and "already spent" in out["pyTokenAgain"]["detail"] and out["pyTokenAgain"]["www"]
    assert out["bad"] == 401
    assert out["stats"]["pool"] == {"issued": 11, "redeemed": 1, "outstanding": 10}
    assert out["pricing"] is True
