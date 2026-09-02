"""Server side of the credit pool: RFC 9474 blind signing, redemption, nullifiers.

Keys are JWK files (private JWKs carry p/q/dp/dq/qi so the same file can be
loaded by WebCrypto in the Cloudflare Worker). One key issues; any number
of older keys stay valid for redemption so rotation never strands tokens:

    TIMESFM3_CREDITS_KEY_FILE=keys/credits-2026-09.json        # issuing (created if missing)
    TIMESFM3_CREDITS_OLD_KEYS=keys/credits-2026-06.json,...    # redeem-only
    TIMESFM3_CREDITS_LEDGER_FILE=credits-ledger.json           # spent serials, counters
"""

from __future__ import annotations

import json
import math
import os
import secrets
import tempfile
import threading

from .. import blindrsa as B
from ..credits import CREDIT_COSTS, SERIAL_BYTES, SUITE, b64d, b64e, key_id, nullifier, verify_token
from .auth import ApiKey

CREDIT_IDENTITY = ApiKey(key="", name="credit-pool", plan="prepaid")
DENOMINATIONS = (10, 25, 100)
DEFAULT_PRICE_PER_CREDIT_USD = 0.004  # 20% below pay-per-call
POINTS_PER_CREDIT = 256  # what a plan holder is charged per credit bought


# -- RSA key generation in pure Python (runs once; persisted as a JWK).
#    Miller-Rabin with 40 rounds: error < 2^-80.

_SMALL_PRIMES = [p for p in range(3, 2000, 2) if all(p % q for q in range(3, int(p ** 0.5) + 1, 2))]


def _is_probable_prime(n: int, rounds: int = 40) -> bool:
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for _ in range(rounds):
        a = secrets.randbelow(n - 3) + 2
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _random_prime(bits: int) -> int:
    while True:
        c = secrets.randbits(bits) | (1 << (bits - 1)) | 1
        if _is_probable_prime(c):
            return c


def generate_private_jwk(bits: int = 2048, e: int = 65537) -> dict:
    """A private RSA JWK (with CRT parameters) usable by Python and WebCrypto."""
    while True:
        p, q = _random_prime(bits // 2), _random_prime(bits // 2)
        if p == q:
            continue
        n = p * q
        phi = (p - 1) * (q - 1)
        if n.bit_length() == bits and math.gcd(e, phi) == 1:
            break
    if p < q:  # JWK/CRT convention: p > q
        p, q = q, p
    d = pow(e, -1, phi)
    return B.private_jwk(B.PrivateKey(n, e, d), p, q)


class PoolKey:
    def __init__(self, jwk: dict, issuing: bool):
        self.private = B.private_from_jwk(jwk) if "d" in jwk else None
        self.public = B.public_from_jwk(jwk)
        self.kid = key_id(self.public)
        self.issuing = issuing
        self.public_jwk = B.public_jwk(self.public)


class CreditPool:
    def __init__(self, key_file: str | None = None, ledger_file: str | None = None,
                 price_per_credit_usd: float = DEFAULT_PRICE_PER_CREDIT_USD, bits: int = 2048,
                 old_key_files: list[str] | None = None):
        self._lock = threading.Lock()
        self.ledger_file = ledger_file
        self.price_per_credit_usd = price_per_credit_usd
        self.ephemeral = key_file is None
        if key_file and os.path.exists(key_file):
            with open(key_file) as f:
                jwk = json.load(f)
            if "kty" not in jwk:
                raise ValueError(f"{key_file} is not a JWK; regenerate the credit-pool key")
        else:
            jwk = generate_private_jwk(bits)
            if key_file:
                d = os.path.dirname(os.path.abspath(key_file))
                os.makedirs(d, exist_ok=True)
                fd, tmp = tempfile.mkstemp(dir=d, prefix=".credits-key-")
                with os.fdopen(fd, "w") as f:
                    json.dump(jwk, f)
                os.chmod(tmp, 0o600)
                os.replace(tmp, key_file)
        self.issuing_key = PoolKey(jwk, issuing=True)
        self.keys: dict[str, PoolKey] = {self.issuing_key.kid: self.issuing_key}
        for path in old_key_files or []:
            with open(path) as f:
                k = PoolKey(json.load(f), issuing=False)
            self.keys.setdefault(k.kid, k)
        self.spent: set[str] = set()
        self.issued = 0
        self.redeemed = 0
        if ledger_file and os.path.exists(ledger_file):
            with open(ledger_file) as f:
                doc = json.load(f)
            self.spent = set(doc.get("spent", []))
            self.issued = int(doc.get("issued", 0))
            self.redeemed = int(doc.get("redeemed", 0))

    # convenience for tests / single-key callers
    @property
    def kid(self) -> str:
        return self.issuing_key.kid

    @property
    def public(self) -> B.PublicKey:
        return self.issuing_key.public

    def _persist(self) -> None:
        if not self.ledger_file:
            return
        d = os.path.dirname(os.path.abspath(self.ledger_file))
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".credits-", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump({"kid": self.kid, "issued": self.issued, "redeemed": self.redeemed,
                       "spent": sorted(self.spent)}, f)
        os.replace(tmp, self.path_or_none())

    def path_or_none(self):
        return self.ledger_file

    # -- public description ------------------------------------------------

    def describe(self) -> dict:
        return {
            "suite": SUITE.name,
            "spec": "RFC 9474",
            "serial_bytes": SERIAL_BYTES,
            "keys": [{"kid": k.kid, "jwk": k.public_jwk, "issuing": k.issuing} for k in self.keys.values()],
            "kid": self.kid,
            "denominations": list(DENOMINATIONS),
            "price_per_credit_usd": self.price_per_credit_usd,
            "points_per_credit": POINTS_PER_CREDIT,
            "costs": dict(CREDIT_COSTS),
            "pool": {"issued": self.issued, "redeemed": self.redeemed, "outstanding": self.issued - self.redeemed},
            "ephemeral_key": self.ephemeral,
            "token_format": "kid.base64url(serial).base64url(signature); header X-Credit: token[,token]",
        }

    # -- issuing -----------------------------------------------------------

    def sign_blinded(self, blinded_b64: list[str]) -> list[str]:
        priv = self.issuing_key.private
        out = []
        for h in blinded_b64:
            try:
                out.append(b64e(SUITE.blind_sign(priv, b64d(h))))
            except (ValueError, TypeError) as e:
                raise ValueError(f"bad blinded message: {e}")
        with self._lock:
            self.issued += len(out)
            self._persist()
        return out

    # -- redemption --------------------------------------------------------

    def redeem(self, header: str, cost: int) -> tuple[bool, str]:
        """Spends exactly ``cost`` valid tokens from the header; all-or-nothing."""
        tokens = [t for t in header.split(",") if t.strip()]
        if len(tokens) < cost:
            return False, f"this call costs {cost} credit(s); {len(tokens)} presented"
        serials = []
        for t in tokens[:cost]:
            kid = t.split(".")[0]
            key = self.keys.get(kid)
            serial = verify_token(t, key.public, kid) if key else None
            if serial is None:
                return False, "invalid credit token (bad signature or unknown key id)"
            serials.append(nullifier(serial))
        if len(set(serials)) != len(serials):
            return False, "duplicate credit token in request"
        with self._lock:
            if any(s in self.spent for s in serials):
                return False, "credit token already spent"
            self.spent.update(serials)
            self.redeemed += cost
            self._persist()
        return True, ""


def credit_cost(method: str, path: str) -> int | None:
    return CREDIT_COSTS.get(f"{method.upper()} {path}")


__all__ = ["CreditPool", "CREDIT_IDENTITY", "DENOMINATIONS", "POINTS_PER_CREDIT", "credit_cost",
           "generate_private_jwk", "SERIAL_BYTES"]
