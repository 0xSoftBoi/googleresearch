"""RFC 9474 implementation: official test vectors and cross-implementation checks."""

import json
import os
import shutil
import subprocess

import pytest

from timesfm3 import blindrsa as B

VECTORS = os.path.join(os.path.dirname(__file__), "data", "rfc9474_vectors.json")


@pytest.mark.parametrize("section", list(json.load(open(VECTORS))))
def test_rfc9474_vectors(section):
    v = json.load(open(VECTORS))[section]
    suite = B.SUITES[section.split()[1]]
    h = lambda s: bytes.fromhex(s) if s else b""
    pub = B.PublicKey(int(v["n"], 16), int(v["e"], 16))
    priv = B.PrivateKey(pub.n, pub.e, int(v["d"], 16))
    prepared = suite.prepare(h(v["msg"]), prefix=h(v["msg_prefix"]) if v["msg_prefix"] else None)
    assert prepared.hex() == v["prepared_msg"]
    assert B.emsa_pss_encode(prepared, pub.n.bit_length() - 1, h(v["salt"])).hex() == v["encoded_msg"]
    inv = int(v["inv"], 16)
    blinded, inv2 = suite.blind(pub, prepared, salt=h(v["salt"]), r=pow(inv, -1, pub.n))
    assert blinded.hex() == v["blinded_msg"] and inv2 == inv
    blind_sig = suite.blind_sign(priv, blinded)
    assert blind_sig.hex() == v["blind_sig"]
    sig = suite.finalize(pub, prepared, blind_sig, inv2)
    assert sig.hex() == v["sig"]
    assert suite.verify(pub, sig, prepared)
    assert not suite.verify(pub, bytes([sig[0] ^ 1]) + sig[1:], prepared)
    assert not suite.verify(pub, sig, prepared + b"x")


def test_jwk_round_trip_and_key_helpers():
    v = json.load(open(VECTORS))["A.4.  RSABSSA-SHA384-PSSZERO-Deterministic Test Vector"]
    priv = B.PrivateKey(int(v["n"], 16), int(v["e"], 16), int(v["d"], 16))
    jwk = B.private_jwk(priv, int(v["p"], 16), int(v["q"], 16))
    assert {"n", "e", "d", "p", "q", "dp", "dq", "qi"} <= set(jwk)
    assert B.private_from_jwk(jwk) == priv and B.public_from_jwk(B.public_jwk(priv.public)) == priv.public
    with pytest.raises(ValueError):
        B.RSABSSA_SHA384_PSSZERO_DETERMINISTIC.finalize(priv.public, b"m", b"\x00" * priv.public.klen, 1)


node = shutil.which("node")
BLINDRSA_JS = "cloudflare/node_modules/@cloudflare/blindrsa-ts/lib/src/index.js"


@pytest.mark.skipif(node is None or not os.path.exists(BLINDRSA_JS), reason="node / blindrsa-ts not installed")
def test_interop_with_cloudflare_blindrsa_ts(tmp_path):
    """Signatures issued by either implementation verify in the other."""
    from timesfm3.serving.credits import generate_private_jwk

    suite = B.RSABSSA_SHA384_PSSZERO_DETERMINISTIC
    jwk = generate_private_jwk(bits=1024)
    priv = B.private_from_jwk(jwk)
    msg = os.urandom(32)
    blinded, inv = suite.blind(priv.public, msg)
    py_sig = suite.finalize(priv.public, msg, suite.blind_sign(priv, blinded), inv)
    payload = {"jwk": jwk, "msg": msg.hex(), "py_sig": py_sig.hex()}
    script = f"""
import {{ RSABSSA }} from './{BLINDRSA_JS}';
let input=''; process.stdin.on('data',d=>input+=d); process.stdin.on('end',async()=>{{
  const p=JSON.parse(input); const hex=(h)=>Uint8Array.from(Buffer.from(h,'hex'));
  const algo={{name:'RSA-PSS',hash:'SHA-384'}};
  const priv=await crypto.subtle.importKey('jwk',{{...p.jwk,alg:'PS384',ext:true}},algo,true,['sign']);
  const pub=await crypto.subtle.importKey('jwk',{{kty:'RSA',n:p.jwk.n,e:p.jwk.e,alg:'PS384',ext:true}},algo,true,['verify']);
  const suite=RSABSSA.SHA384.PSSZero.Deterministic();
  const pyOk=await suite.verify(pub,hex(p.py_sig),hex(p.msg));
  const msg2=crypto.getRandomValues(new Uint8Array(32)); const prep=suite.prepare(msg2);
  const {{blindedMsg,inv}}=await suite.blind(pub,prep); const bs=await suite.blindSign(priv,blindedMsg);
  const sig=await suite.finalize(pub,prep,bs,inv);
  console.log(JSON.stringify({{pyOk, msg2:Buffer.from(msg2).toString('hex'), sig:Buffer.from(sig).toString('hex')}}));
}});"""
    res = subprocess.run([node, "--input-type=module", "-e", script], input=json.dumps(payload), capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["pyOk"] is True
    assert suite.verify(priv.public, bytes.fromhex(out["sig"]), bytes.fromhex(out["msg2"]))
