"""Unlinkable prepaid credits: a privacy pool for API usage.

A payment ties a wallet to a request; a plan ties an API key to every
request it makes. Both let the operator (and, for x402, the facilitator)
reconstruct a customer's query history. For a quant desk the *pattern* of
forecasts is the secret, so the service offers a third way to pay:

1. The buyer generates random serials, **blinds** them, and buys a batch of
   signatures (`POST /v1/credits/buy/{n}`, paid once with x402 or a plan).
2. The server signs the blinded messages without seeing the serials.
3. The buyer unblinds and holds single-use tokens. Each API call redeems
   one or more (`X-Credit` header). The server can verify a token is
   genuine and unspent, but cannot tell which purchase it came from or
   which other tokens belong to the same buyer.

The signature scheme is **RFC 9474** (RSABSSA-SHA384-PSSZERO-Deterministic),
implemented in :mod:`timesfm3.blindrsa` and interoperable with any
conforming library -- the Cloudflare Worker issues and redeems the same
tokens with ``@cloudflare/blindrsa-ts``. Serials double as nullifiers
against double spending. This module is the shared token format and the
*client* wallet; it depends only on the standard library.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import os
import secrets
import tempfile
from typing import Iterable

from . import blindrsa as B

SUITE = B.RSABSSA_SHA384_PSSZERO_DETERMINISTIC
SERIAL_BYTES = 32

#: Credits charged per call; a token is one credit.
CREDIT_COSTS: dict[str, int] = {
    "POST /v1/forecast": 1,
    "POST /v1/volatility": 1,
    "POST /v1/anomalies": 2,
    "POST /v1/backtest": 4,
}


def b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def key_id(pub: B.PublicKey) -> str:
    """12 hex chars of SHA-256 over the big-endian modulus (same in the Worker)."""
    return hashlib.sha256(B.i2osp(pub.n, pub.klen)).hexdigest()[:12]


def encode_token(kid: str, serial: bytes, signature: bytes) -> str:
    return f"{kid}.{b64e(serial)}.{b64e(signature)}"


def decode_token(token: str) -> tuple[str, bytes, bytes]:
    kid, s, sig = token.strip().split(".")
    return kid, b64d(s), b64d(sig)


def verify_token(token: str, pub: B.PublicKey, kid: str) -> bytes | None:
    """Returns the serial if the token carries a valid signature under ``pub``."""
    try:
        tkid, serial, sig = decode_token(token)
    except (ValueError, TypeError):
        return None
    if tkid != kid or len(serial) != SERIAL_BYTES:
        return None
    return serial if SUITE.verify(pub, sig, serial) else None


def nullifier(serial: bytes) -> str:
    return hashlib.sha256(b"nullifier" + serial).hexdigest()


# ---------------------------------------------------------------------------
# Client-side wallet


@dataclasses.dataclass
class PendingBlind:
    serial: bytes
    inv: int
    blinded: bytes


def issuing_key(pool: dict) -> dict:
    keys = pool.get("keys") or []
    for k in keys:
        if k.get("issuing"):
            return k
    if keys:
        return keys[0]
    raise ValueError("pool has no keys")


class CreditWallet:
    """Holds unspent tokens and the pool's public keys in a JSON file."""

    def __init__(self, path: str | None = None):
        self.path = path
        self.tokens: list[str] = []
        self.pool: dict | None = None  # {suite, keys: [{kid, jwk, issuing}]}
        if path and os.path.exists(path):
            with open(path) as f:
                doc = json.load(f)
            self.tokens = list(doc.get("tokens", []))
            self.pool = doc.get("pool")

    def save(self) -> None:
        if not self.path:
            return
        d = os.path.dirname(os.path.abspath(self.path))
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".credits-", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump({"pool": self.pool, "tokens": self.tokens}, f)
        os.replace(tmp, self.path)

    def __len__(self) -> int:
        return len(self.tokens)

    def prepare(self, pool: dict, count: int) -> list[PendingBlind]:
        """Generates serials and blinds them for a purchase of ``count`` credits."""
        if pool.get("suite", SUITE.name) != SUITE.name:
            raise ValueError(f"unsupported suite {pool.get('suite')}")
        key = issuing_key(pool)
        pub = B.public_from_jwk(key["jwk"])
        self.pool = {"suite": SUITE.name, "keys": [{"kid": k["kid"], "jwk": k["jwk"], "issuing": bool(k.get("issuing"))} for k in pool["keys"]]}
        self._issuing = key
        out = []
        for _ in range(count):
            serial = secrets.token_bytes(SERIAL_BYTES)
            blinded, inv = SUITE.blind(pub, SUITE.prepare(serial))
            out.append(PendingBlind(serial=serial, inv=inv, blinded=blinded))
        return out

    def finish(self, pending: Iterable[PendingBlind], blind_signatures: Iterable[str], kid: str | None = None) -> int:
        """Unblinds the server's signatures, verifies them, stores the tokens."""
        key = self._issuing if kid is None else next(k for k in self.pool["keys"] if k["kid"] == kid)
        pub = B.public_from_jwk(key["jwk"])
        added = 0
        for p, sig_b64 in zip(pending, blind_signatures):
            sig = SUITE.finalize(pub, p.serial, b64d(sig_b64), p.inv)  # raises if invalid
            self.tokens.append(encode_token(key["kid"], p.serial, sig))
            added += 1
        self.save()
        return added

    def take(self, count: int) -> str:
        """Removes ``count`` tokens and returns the ``X-Credit`` header value."""
        if count > len(self.tokens):
            raise ValueError(f"wallet holds {len(self.tokens)} credits; {count} needed")
        chosen, self.tokens = self.tokens[:count], self.tokens[count:]
        self.save()
        return ",".join(chosen)

    @staticmethod
    def cost_of(method: str, path: str) -> int:
        return CREDIT_COSTS.get(f"{method.upper()} {path}", 1)
