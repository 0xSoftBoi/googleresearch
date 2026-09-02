"""Privacy Pass (RFC 9576/9577/9578), publicly verifiable tokens, type 0x0002.

Wire-compatible with Cloudflare's ``@cloudflare/privacypass-ts`` and any other
conforming implementation.  This module covers everything the service and
the client need, on top of :mod:`timesfm3.blindrsa` (RFC 9474):

- :class:`TokenChallenge` / :class:`Token` / :class:`TokenRequest` /
  :class:`TokenResponse` TLS-style structures (RFC 9577 §2, RFC 9578 §6)
- issuer public key encoding as DER SPKI with RSASSA-PSS parameters and
  ``token_key_id = SHA256(SPKI)`` (RFC 9578 §6.5)
- the ``WWW-Authenticate: PrivateToken`` challenge and
  ``Authorization: PrivateToken`` credential header formats (RFC 9577)
- issuer (blind sign), client (blind / finalize), origin (verify) roles

Only the standard library is used.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import secrets
import struct

from . import blindrsa as B

TOKEN_TYPE_BLIND_RSA = 0x0002
SUITE = B.RSABSSA_SHA384_PSS_DETERMINISTIC  # RFC 9578 §6.4: PSS, 48-byte salt
NK = 256  # modulus bytes for the 2048-bit token type
MEDIA_REQUEST = "application/private-token-request"
MEDIA_RESPONSE = "application/private-token-response"
MEDIA_DIRECTORY = "application/private-token-issuer-directory"
MEDIA_BATCH_REQUEST = "application/private-token-generic-batch-request"
MEDIA_BATCH_RESPONSE = "application/private-token-generic-batch-response"
ISSUER_DIRECTORY_PATH = "/.well-known/private-token-issuer-directory"
AUTH_SCHEME = "PrivateToken"


def b64e(b: bytes) -> str:
    """base64url *with* padding: RFC 9577 requires the '=' characters in headers."""
    return base64.urlsafe_b64encode(b).decode()


def b64d(s: str) -> bytes:
    """Accepts padded or unpadded base64url."""
    s = s.strip().rstrip("=")
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


# -- DER helpers -----------------------------------------------------------------

def _der_len(n: int) -> bytes:
    if n < 0x80:
        return bytes([n])
    body = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(body)]) + body


def _der(tag: int, payload: bytes) -> bytes:
    return bytes([tag]) + _der_len(len(payload)) + payload


def _der_int(x: int) -> bytes:
    body = x.to_bytes((x.bit_length() + 8) // 8, "big")  # leading 0 keeps it positive
    return _der(0x02, body)


def _der_oid(oid: str) -> bytes:
    parts = [int(p) for p in oid.split(".")]
    out = bytes([40 * parts[0] + parts[1]])
    for p in parts[2:]:
        chunk = [p & 0x7F]
        p >>= 7
        while p:
            chunk.append(0x80 | (p & 0x7F))
            p >>= 7
        out += bytes(reversed(chunk))
    return _der(0x06, out)


_SHA384_ALG = _der(0x30, _der_oid("2.16.840.1.101.3.4.2.2"))  # id-sha384, absent params
_MGF1_SHA384_ALG = _der(0x30, _der_oid("1.2.840.113549.1.1.8") + _SHA384_ALG)  # id-mgf1 { sha384 }


def spki_rsassa_pss(pub: B.PublicKey, salt_len: int = 48) -> bytes:
    """DER SubjectPublicKeyInfo: id-RSASSA-PSS with {sha384, mgf1-sha384, saltLength}."""
    params = _der(0x30,
                  _der(0xA0, _SHA384_ALG) + _der(0xA1, _MGF1_SHA384_ALG) + _der(0xA2, _der_int(salt_len)))
    alg = _der(0x30, _der_oid("1.2.840.113549.1.1.10") + params)
    rsa_pub = _der(0x30, _der_int(pub.n) + _der_int(pub.e))
    return _der(0x30, alg + _der(0x03, b"\x00" + rsa_pub))


def public_key_from_spki(spki: bytes) -> B.PublicKey:
    """Parses n and e out of an SPKI (RSASSA-PSS or plain rsaEncryption)."""
    bitstr = spki.rfind(b"\x03")  # locate the BIT STRING holding RSAPublicKey
    # walk properly instead of guessing: parse outer SEQUENCE
    def read(buf, i):
        tag = buf[i]; i += 1
        ln = buf[i]; i += 1
        if ln & 0x80:
            k = ln & 0x7F; ln = int.from_bytes(buf[i:i + k], "big"); i += k
        return tag, buf[i:i + ln], i + ln
    tag, body, _ = read(spki, 0)
    assert tag == 0x30
    _, _alg, j = read(body, 0)
    tag, bits, _ = read(body, j)
    assert tag == 0x03 and bits[0] == 0
    _, rsa, _ = read(bits, 1)
    tag, n_bytes, k = read(rsa, 0)
    tag2, e_bytes, _ = read(rsa, k)
    return B.PublicKey(int.from_bytes(n_bytes, "big"), int.from_bytes(e_bytes, "big"))


def token_key_id(pub: B.PublicKey) -> bytes:
    return hashlib.sha256(spki_rsassa_pss(pub)).digest()


# -- structures -------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class TokenChallenge:
    token_type: int
    issuer_name: str
    redemption_context: bytes = b""
    origin_info: str = ""

    def serialize(self) -> bytes:
        name = self.issuer_name.encode("ascii")
        origin = self.origin_info.encode("ascii")
        return (struct.pack(">H", self.token_type) + struct.pack(">H", len(name)) + name
                + bytes([len(self.redemption_context)]) + self.redemption_context
                + struct.pack(">H", len(origin)) + origin)

    @classmethod
    def deserialize(cls, b: bytes) -> "TokenChallenge":
        tt = struct.unpack(">H", b[:2])[0]
        ln = struct.unpack(">H", b[2:4])[0]; i = 4
        name = b[i:i + ln].decode("ascii"); i += ln
        rl = b[i]; i += 1
        ctx = b[i:i + rl]; i += rl
        ol = struct.unpack(">H", b[i:i + 2])[0]; i += 2
        origin = b[i:i + ol].decode("ascii")
        if rl not in (0, 32):
            raise ValueError("invalid redemption_context length")
        return cls(tt, name, ctx, origin)

    def digest(self) -> bytes:
        return hashlib.sha256(self.serialize()).digest()


@dataclasses.dataclass(frozen=True)
class TokenRequest:
    truncated_token_key_id: int
    blinded_msg: bytes
    token_type: int = TOKEN_TYPE_BLIND_RSA

    def serialize(self) -> bytes:
        return struct.pack(">HB", self.token_type, self.truncated_token_key_id) + self.blinded_msg

    @classmethod
    def deserialize(cls, b: bytes) -> "TokenRequest":
        if len(b) != 3 + NK:
            raise ValueError("bad TokenRequest length")
        tt, tk = struct.unpack(">HB", b[:3])
        return cls(tk, b[3:], tt)


@dataclasses.dataclass(frozen=True)
class TokenResponse:
    blind_sig: bytes

    def serialize(self) -> bytes:
        return self.blind_sig

    @classmethod
    def deserialize(cls, b: bytes) -> "TokenResponse":
        if len(b) != NK:
            raise ValueError("bad TokenResponse length")
        return cls(b)


@dataclasses.dataclass(frozen=True)
class Token:
    token_type: int
    nonce: bytes
    challenge_digest: bytes
    token_key_id: bytes
    authenticator: bytes

    def authenticator_input(self) -> bytes:
        return struct.pack(">H", self.token_type) + self.nonce + self.challenge_digest + self.token_key_id

    def serialize(self) -> bytes:
        return self.authenticator_input() + self.authenticator

    @classmethod
    def deserialize(cls, b: bytes) -> "Token":
        if len(b) != 2 + 32 + 32 + 32 + NK:
            raise ValueError("bad Token length")
        return cls(struct.unpack(">H", b[:2])[0], b[2:34], b[34:66], b[66:98], b[98:])


# -- HTTP header formats (RFC 9577) -------------------------------------------

def www_authenticate(challenge: TokenChallenge, pub: B.PublicKey, max_age: int | None = None) -> str:
    v = f'{AUTH_SCHEME} challenge="{b64e(challenge.serialize())}", token-key="{b64e(spki_rsassa_pss(pub))}"'
    if max_age is not None:
        v += f", max-age={int(max_age)}"
    return v


def parse_www_authenticate(header: str) -> list[dict]:
    """Returns [{challenge: TokenChallenge, token_key: PublicKey, max_age}] per PrivateToken entry."""
    out = []
    for part in header.split(f"{AUTH_SCHEME} ")[1:]:
        params = {}
        for kv in part.split(","):
            if "=" in kv:
                k, v = kv.strip().split("=", 1)
                params[k.strip()] = v.strip().strip('"')
        if "challenge" in params and "token-key" in params:
            out.append({
                "challenge": TokenChallenge.deserialize(b64d(params["challenge"])),
                "token_key": public_key_from_spki(b64d(params["token-key"])),
                "token_key_bytes": b64d(params["token-key"]),
                "max_age": int(params["max-age"]) if "max-age" in params else None,
            })
    return out


def authorization(token: Token) -> str:
    return f'{AUTH_SCHEME} token="{b64e(token.serialize())}"'


def parse_authorization(header: str) -> Token | None:
    h = header.strip()
    if not h.startswith(AUTH_SCHEME):
        return None
    for kv in h[len(AUTH_SCHEME):].split(","):
        if "=" in kv:
            k, v = kv.strip().split("=", 1)
            if k.strip() == "token":
                return Token.deserialize(b64d(v.strip().strip('"')))
    return None


# -- roles ----------------------------------------------------------------------

def issuer_directory(keys: list[B.PublicKey], request_uri: str = "/token-request") -> dict:
    """RFC 9578 §4.3 issuer directory; ``token-keys`` are base64url SPKI (padded)."""
    return {"issuer-request-uri": request_uri,
            "token-keys": [{"token-type": TOKEN_TYPE_BLIND_RSA, "token-key": b64e(spki_rsassa_pss(k))} for k in keys]}


class Client:
    """RFC 9578 §6.1/§6.3: builds a TokenRequest for a challenge; finalizes a Token."""

    def __init__(self, pub: B.PublicKey):
        self.pub = pub
        self.key_id = token_key_id(pub)

    def create_request(self, challenge: TokenChallenge, nonce: bytes | None = None) -> tuple[TokenRequest, dict]:
        nonce = nonce or secrets.token_bytes(32)
        token_input = struct.pack(">H", TOKEN_TYPE_BLIND_RSA) + nonce + challenge.digest() + self.key_id
        blinded, inv = SUITE.blind(self.pub, SUITE.prepare(token_input))
        req = TokenRequest(self.key_id[-1], blinded)
        return req, {"nonce": nonce, "challenge_digest": challenge.digest(), "token_input": token_input, "inv": inv}

    def finalize(self, state: dict, response: TokenResponse) -> Token:
        authenticator = SUITE.finalize(self.pub, state["token_input"], response.blind_sig, state["inv"])
        return Token(TOKEN_TYPE_BLIND_RSA, state["nonce"], state["challenge_digest"], self.key_id, authenticator)


class Issuer:
    """RFC 9578 §6.2: blind-signs token requests for the keys it holds."""

    def __init__(self, keys: list[B.PrivateKey]):
        self.keys = {token_key_id(k.public)[-1]: k for k in keys}
        self.by_full_id = {token_key_id(k.public): k for k in keys}

    def issue(self, req: TokenRequest) -> TokenResponse:
        if req.token_type != TOKEN_TYPE_BLIND_RSA:
            raise ValueError("unsupported token_type")
        priv = self.keys.get(req.truncated_token_key_id)
        if priv is None:
            raise ValueError("unknown token key id")
        if len(req.blinded_msg) != priv.public.klen:
            raise ValueError("blinded_msg has the wrong size")
        return TokenResponse(SUITE.blind_sign(priv, req.blinded_msg))


class Origin:
    """RFC 9578 §6.4 verification against the keys this origin accepts."""

    def __init__(self, keys: list[B.PublicKey]):
        self.keys = {token_key_id(k): k for k in keys}

    def verify(self, token: Token, challenge: TokenChallenge | None = None) -> bool:
        pub = self.keys.get(token.token_key_id)
        if pub is None or token.token_type != TOKEN_TYPE_BLIND_RSA:
            return False
        if challenge is not None and token.challenge_digest != challenge.digest():
            return False
        return SUITE.verify(pub, token.authenticator, token.authenticator_input())


# -- generic batched issuance (draft-ietf-privacypass-batched-tokens) --------
# Wire format as implemented by @cloudflare/privacypass-ts: a QUIC variable-
# length integer (RFC 9000 §16) giving the total byte length, then the items.

def quic_varint_encode(v: int) -> bytes:
    if v < 0x40:
        return bytes([v])
    if v < 0x4000:
        return struct.pack(">H", v | 0x4000)
    if v < 0x40000000:
        return struct.pack(">I", v | 0x80000000)
    if v < 0x4000000000000000:
        return struct.pack(">Q", v | 0xC000000000000000)
    raise ValueError("varint too large")


def quic_varint_decode(b: bytes) -> tuple[int, int]:
    """Returns (value, bytes consumed)."""
    prefix = b[0] >> 6
    size = 1 << prefix
    v = int.from_bytes(b[:size], "big") & ((1 << (8 * size - 2)) - 1)
    return v, size


def serialize_batched_request(reqs: list[TokenRequest]) -> bytes:
    items = b"".join(r.serialize() for r in reqs)
    return quic_varint_encode(len(items)) + items


def deserialize_batched_request(b: bytes) -> list[TokenRequest]:
    total, off = quic_varint_decode(b)
    if off + total != len(b):
        raise ValueError("batched request length mismatch")
    out = []
    item = 3 + NK  # type 0x0002 requests are fixed size
    while off < len(b):
        if b[off:off + 2] != struct.pack(">H", TOKEN_TYPE_BLIND_RSA):
            raise ValueError("unsupported token type in batch")
        out.append(TokenRequest.deserialize(b[off:off + item]))
        off += item
    return out


def count_batched_request(b: bytes) -> int:
    total, _ = quic_varint_decode(b)
    return total // (3 + NK)


def serialize_batched_response(responses: list["TokenResponse | None"]) -> bytes:
    items = b""
    for r in responses:
        if r is None:
            items += b"\x00"
        else:
            items += b"\x01" + struct.pack(">H", TOKEN_TYPE_BLIND_RSA) + r.serialize()
    return quic_varint_encode(len(items)) + items


def deserialize_batched_response(b: bytes) -> list["TokenResponse | None"]:
    total, off = quic_varint_decode(b)
    if off + total != len(b):
        raise ValueError("batched response length mismatch")
    out: list[TokenResponse | None] = []
    while off < len(b):
        present = b[off]; off += 1
        if not present:
            out.append(None)
            continue
        tt = struct.unpack(">H", b[off:off + 2])[0]; off += 2
        if tt != TOKEN_TYPE_BLIND_RSA:
            raise ValueError("unsupported token type in batch response")
        out.append(TokenResponse.deserialize(b[off:off + NK])); off += NK
    return out
