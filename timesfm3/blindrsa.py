"""RFC 9474 RSA blind signatures (RSABSSA), pure Python.

Implements the four named variants -- SHA-384 with PSS (48-byte salt) or
PSS-Zero (no salt), Randomized (a 32-byte random prefix is prepended to the
message) or Deterministic (identity prepare) -- exactly as specified, so
tokens issued here verify with any conforming library (for example
Cloudflare's ``@cloudflare/blindrsa-ts`` in the edge Worker) and vice
versa.  Only the standard library is used; RSA arithmetic is on Python
integers.

    suite = RSABSSA_SHA384_PSSZERO_DETERMINISTIC
    prepared = suite.prepare(msg)
    blinded, inv = suite.blind(pub, prepared)
    blind_sig = suite.blind_sign(priv, blinded)
    sig = suite.finalize(pub, prepared, blind_sig, inv)
    assert suite.verify(pub, sig, prepared)
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import secrets

HASH = hashlib.sha384
HLEN = 48


@dataclasses.dataclass(frozen=True)
class PublicKey:
    n: int
    e: int

    @property
    def klen(self) -> int:  # bytes in the modulus
        return (self.n.bit_length() + 7) // 8


@dataclasses.dataclass(frozen=True)
class PrivateKey:
    n: int
    e: int
    d: int

    @property
    def public(self) -> PublicKey:
        return PublicKey(self.n, self.e)


def i2osp(x: int, length: int) -> bytes:
    if x < 0 or x >= 1 << (8 * length):
        raise ValueError("integer too large")
    return x.to_bytes(length, "big")


def os2ip(b: bytes) -> int:
    return int.from_bytes(b, "big")


def mgf1(seed: bytes, length: int) -> bytes:
    out = b""
    for counter in range((length + HLEN - 1) // HLEN):
        out += HASH(seed + counter.to_bytes(4, "big")).digest()
    return out[:length]


def emsa_pss_encode(msg: bytes, em_bits: int, salt: bytes) -> bytes:
    """RFC 8017 9.1.1 with SHA-384 / MGF1-SHA384 and the given salt."""
    em_len = (em_bits + 7) // 8
    s_len = len(salt)
    mhash = HASH(msg).digest()
    if em_len < HLEN + s_len + 2:
        raise ValueError("encoding error")
    h = HASH(b"\x00" * 8 + mhash + salt).digest()
    ps = b"\x00" * (em_len - s_len - HLEN - 2)
    db = ps + b"\x01" + salt
    db_mask = mgf1(h, em_len - HLEN - 1)
    masked = bytes(a ^ b for a, b in zip(db, db_mask))
    # clear the leftmost 8*emLen - emBits bits
    top = 8 * em_len - em_bits
    masked = bytes([masked[0] & (0xFF >> top)]) + masked[1:]
    return masked + h + b"\xbc"


def emsa_pss_verify(msg: bytes, em: bytes, em_bits: int, s_len: int) -> bool:
    """RFC 8017 9.1.2."""
    em_len = (em_bits + 7) // 8
    mhash = HASH(msg).digest()
    if em_len < HLEN + s_len + 2 or len(em) != em_len or em[-1] != 0xBC:
        return False
    masked, h = em[: em_len - HLEN - 1], em[em_len - HLEN - 1 : -1]
    top = 8 * em_len - em_bits
    if masked[0] & ~(0xFF >> top) & 0xFF:
        return False
    db_mask = mgf1(h, em_len - HLEN - 1)
    db = bytearray(a ^ b for a, b in zip(masked, db_mask))
    db[0] &= 0xFF >> top
    pad_len = em_len - HLEN - s_len - 2
    if any(db[:pad_len]) or db[pad_len] != 0x01:
        return False
    salt = bytes(db[pad_len + 1 :]) if s_len else b""
    if len(salt) != s_len:
        return False
    return secrets.compare_digest(HASH(b"\x00" * 8 + mhash + salt).digest(), h)


def rsassa_pss_verify(pub: PublicKey, msg: bytes, sig: bytes, s_len: int) -> bool:
    if len(sig) != pub.klen:
        return False
    s = os2ip(sig)
    if s >= pub.n:
        return False
    em_bits = pub.n.bit_length() - 1
    em = i2osp(pow(s, pub.e, pub.n), (em_bits + 7) // 8)
    return emsa_pss_verify(msg, em, em_bits, s_len)


@dataclasses.dataclass(frozen=True)
class Suite:
    name: str
    salt_len: int
    randomized: bool

    # -- client ---------------------------------------------------------------

    def prepare(self, msg: bytes, prefix: bytes | None = None) -> bytes:
        if not self.randomized:
            return msg
        return (prefix if prefix is not None else secrets.token_bytes(32)) + msg

    def blind(self, pub: PublicKey, prepared: bytes, salt: bytes | None = None,
              r: int | None = None) -> tuple[bytes, int]:
        """Returns (blinded_msg, inv). ``salt``/``r`` only for test vectors."""
        em_bits = pub.n.bit_length() - 1
        if salt is None:
            salt = secrets.token_bytes(self.salt_len)
        if len(salt) != self.salt_len:
            raise ValueError("salt length mismatch")
        em = emsa_pss_encode(prepared, em_bits, salt)
        m = os2ip(em)
        if m >= pub.n:
            raise ValueError("message representative out of range")
        if math.gcd(m, pub.n) != 1:
            raise ValueError("invalid input")
        while True:
            if r is None:
                r = secrets.randbelow(pub.n - 1) + 1
            if math.gcd(r, pub.n) == 1:
                break
            r = None
        inv = pow(r, -1, pub.n)
        x = pow(r, pub.e, pub.n)
        z = (m * x) % pub.n
        return i2osp(z, pub.klen), inv

    def finalize(self, pub: PublicKey, prepared: bytes, blind_sig: bytes, inv: int) -> bytes:
        if len(blind_sig) != pub.klen:
            raise ValueError("unexpected input size")
        z = os2ip(blind_sig)
        s = (z * inv) % pub.n
        sig = i2osp(s, pub.klen)
        if not self.verify(pub, sig, prepared):
            raise ValueError("invalid signature")
        return sig

    # -- server ---------------------------------------------------------------

    def blind_sign(self, priv: PrivateKey, blinded_msg: bytes) -> bytes:
        pub = priv.public
        if len(blinded_msg) != pub.klen:
            raise ValueError("unexpected input size")
        m = os2ip(blinded_msg)
        if m >= pub.n:
            raise ValueError("invalid message length")
        s = pow(m, priv.d, priv.n)
        if pow(s, pub.e, pub.n) != m:
            raise ValueError("signing failure")
        return i2osp(s, pub.klen)

    def verify(self, pub: PublicKey, sig: bytes, prepared: bytes) -> bool:
        return rsassa_pss_verify(pub, prepared, sig, self.salt_len)


RSABSSA_SHA384_PSS_RANDOMIZED = Suite("RSABSSA-SHA384-PSS-Randomized", 48, True)
RSABSSA_SHA384_PSSZERO_RANDOMIZED = Suite("RSABSSA-SHA384-PSSZERO-Randomized", 0, True)
RSABSSA_SHA384_PSS_DETERMINISTIC = Suite("RSABSSA-SHA384-PSS-Deterministic", 48, False)
RSABSSA_SHA384_PSSZERO_DETERMINISTIC = Suite("RSABSSA-SHA384-PSSZERO-Deterministic", 0, False)
SUITES = {s.name: s for s in (
    RSABSSA_SHA384_PSS_RANDOMIZED, RSABSSA_SHA384_PSSZERO_RANDOMIZED,
    RSABSSA_SHA384_PSS_DETERMINISTIC, RSABSSA_SHA384_PSSZERO_DETERMINISTIC,
)}


# -- key helpers ---------------------------------------------------------------

def _b64url_uint(x: int) -> str:
    import base64
    return base64.urlsafe_b64encode(i2osp(x, (x.bit_length() + 7) // 8)).decode().rstrip("=")


def _uint_b64url(s: str) -> int:
    import base64
    return os2ip(base64.urlsafe_b64decode(s + "=" * (-len(s) % 4)))


def public_jwk(pub: PublicKey) -> dict:
    return {"kty": "RSA", "n": _b64url_uint(pub.n), "e": _b64url_uint(pub.e), "alg": "PS384", "use": "sig"}


def public_from_jwk(jwk: dict) -> PublicKey:
    return PublicKey(_uint_b64url(jwk["n"]), _uint_b64url(jwk["e"]))


def private_from_jwk(jwk: dict) -> PrivateKey:
    return PrivateKey(_uint_b64url(jwk["n"]), _uint_b64url(jwk["e"]), _uint_b64url(jwk["d"]))


def private_jwk(priv: PrivateKey, p: int | None = None, q: int | None = None) -> dict:
    """Minimal private JWK (n, e, d) plus CRT parameters when p and q are known
    -- WebCrypto requires p, q, dp, dq, qi to import a private RSA key."""
    jwk = {"kty": "RSA", "alg": "PS384", "n": _b64url_uint(priv.n), "e": _b64url_uint(priv.e), "d": _b64url_uint(priv.d)}
    if p and q:
        jwk.update({"p": _b64url_uint(p), "q": _b64url_uint(q), "dp": _b64url_uint(priv.d % (p - 1)),
                    "dq": _b64url_uint(priv.d % (q - 1)), "qi": _b64url_uint(pow(q, -1, p))})
    return jwk
