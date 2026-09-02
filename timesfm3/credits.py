"""Unlinkable prepaid credits: a privacy pool for API usage.

A payment ties a wallet to a request; a plan ties an API key to every
request it makes. Both let the operator (and, for x402, the facilitator)
reconstruct a customer's query history. For a quant desk the *pattern* of
forecasts is the secret, so the service offers a third way to pay:

1. The buyer generates random serials, **blinds** them, and buys a batch of
   signatures (`POST /v1/credits/buy/{n}`, paid once with x402 or a plan).
2. The server signs the blinded messages with its RSA key without seeing
   the serials.
3. The buyer unblinds and holds single-use tokens. Each API call redeems
   one or more (`X-Credit` header). The server can verify a token is
   genuine and unspent, but cannot tell which purchase it came from or
   which other tokens belong to the same buyer.

All buyers' credits are indistinguishable inside the pool, which is what
makes the pool private. The scheme is Chaum's RSA blind signature with a
full-domain hash (RSA-FDH); serials double as nullifiers against double
spending. This module is the shared arithmetic and the *client* wallet; it
depends only on the standard library so any client can use it.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import math
import os
import secrets
import tempfile
from typing import Iterable

DOMAIN = b"timesfm3-credit-v1"
SERIAL_BYTES = 32

#: Credits charged per call; a token is one credit.
CREDIT_COSTS: dict[str, int] = {
    "POST /v1/forecast": 1,
    "POST /v1/volatility": 1,
    "POST /v1/anomalies": 2,
    "POST /v1/backtest": 4,
}


def key_id(n: int) -> str:
    return hashlib.sha256(n.to_bytes((n.bit_length() + 7) // 8, "big")).hexdigest()[:12]


def full_domain_hash(serial: bytes, n: int) -> int:
    """Deterministically maps a serial to an integer in [0, n) (MGF1-style)."""
    nbytes = (n.bit_length() + 7) // 8
    out = b""
    counter = 0
    while len(out) < nbytes:
        out += hashlib.sha256(DOMAIN + serial + counter.to_bytes(4, "big")).digest()
        counter += 1
    # Clear the top byte so the value is always below n for keys >= 16 bits.
    return int.from_bytes(b"\x00" + out[1:nbytes], "big") % n


def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def encode_token(kid: str, serial: bytes, signature: int, n: int) -> str:
    nbytes = (n.bit_length() + 7) // 8
    return f"{kid}.{_b64e(serial)}.{_b64e(signature.to_bytes(nbytes, 'big'))}"


def decode_token(token: str) -> tuple[str, bytes, int]:
    kid, s, sig = token.strip().split(".")
    return kid, _b64d(s), int.from_bytes(_b64d(sig), "big")


def verify_token(token: str, n: int, e: int, kid: str) -> bytes | None:
    """Returns the serial if the token carries a valid signature, else None."""
    try:
        tkid, serial, sig = decode_token(token)
    except (ValueError, TypeError):
        return None
    if tkid != kid or len(serial) != SERIAL_BYTES or not (0 < sig < n):
        return None
    return serial if pow(sig, e, n) == full_domain_hash(serial, n) else None


def nullifier(serial: bytes) -> str:
    return hashlib.sha256(b"nullifier" + serial).hexdigest()


# ---------------------------------------------------------------------------
# Client-side wallet


@dataclasses.dataclass
class PendingBlind:
    serial: bytes
    r: int
    blinded: int


class CreditWallet:
    """Holds unspent tokens and in-flight blinded purchases in a JSON file."""

    def __init__(self, path: str | None = None):
        self.path = path
        self.tokens: list[str] = []
        self.pool: dict | None = None  # {kid, n, e}
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
        n, e = int(pool["n"], 16), int(pool["e"])
        self.pool = {"kid": pool["kid"], "n": pool["n"], "e": pool["e"]}
        out = []
        for _ in range(count):
            serial = secrets.token_bytes(SERIAL_BYTES)
            while True:
                r = secrets.randbelow(n - 2) + 2
                if math.gcd(r, n) == 1:
                    break
            blinded = (full_domain_hash(serial, n) * pow(r, e, n)) % n
            out.append(PendingBlind(serial=serial, r=r, blinded=blinded))
        return out

    def finish(self, pending: Iterable[PendingBlind], blind_signatures: Iterable[str]) -> int:
        """Unblinds the server's signatures and stores verified tokens."""
        n, e, kid = int(self.pool["n"], 16), int(self.pool["e"]), self.pool["kid"]
        added = 0
        for p, sig_hex in zip(pending, blind_signatures):
            sig = (int(sig_hex, 16) * pow(p.r, -1, n)) % n
            token = encode_token(kid, p.serial, sig, n)
            if verify_token(token, n, e, kid) is None:
                raise ValueError("server returned an invalid blind signature")
            self.tokens.append(token)
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
