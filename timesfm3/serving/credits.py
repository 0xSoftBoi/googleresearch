"""Server side of the credit pool: signing, redemption, nullifiers."""

from __future__ import annotations

import json
import math
import os
import secrets
import tempfile
import threading

from ..credits import CREDIT_COSTS, SERIAL_BYTES, key_id, nullifier, verify_token
from .auth import ApiKey


# -- RSA key generation in pure Python (no OpenSSL binding needed; runs once
#    and is persisted). Miller-Rabin with 40 rounds: error < 2^-80.

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


def generate_rsa(bits: int = 2048, e: int = 65537) -> dict:
    while True:
        p, q = _random_prime(bits // 2), _random_prime(bits // 2)
        if p == q:
            continue
        n = p * q
        phi = (p - 1) * (q - 1)
        if n.bit_length() == bits and math.gcd(e, phi) == 1:
            return {"n": n, "e": e, "d": pow(e, -1, phi)}

CREDIT_IDENTITY = ApiKey(key="", name="credit-pool", plan="prepaid")
DENOMINATIONS = (10, 25, 100)
DEFAULT_PRICE_PER_CREDIT_USD = 0.004  # 20% below pay-per-call
POINTS_PER_CREDIT = 256  # what a plan holder is charged per credit bought


class CreditPool:
    def __init__(self, key_file: str | None = None, ledger_file: str | None = None,
                 price_per_credit_usd: float = DEFAULT_PRICE_PER_CREDIT_USD, bits: int = 2048):
        self._lock = threading.Lock()
        self.ledger_file = ledger_file
        self.price_per_credit_usd = price_per_credit_usd
        self.ephemeral = key_file is None
        if key_file and os.path.exists(key_file):
            with open(key_file) as f:
                key = {k: int(v, 16) for k, v in json.load(f).items()}
        else:
            key = generate_rsa(bits)
            if key_file:
                d = os.path.dirname(os.path.abspath(key_file))
                os.makedirs(d, exist_ok=True)
                fd, tmp = tempfile.mkstemp(dir=d, prefix=".credits-key-")
                with os.fdopen(fd, "w") as f:
                    json.dump({k: format(v, "x") for k, v in key.items()}, f)
                os.chmod(tmp, 0o600)
                os.replace(tmp, key_file)
        self.n, self.e, self._d = key["n"], key["e"], key["d"]
        self.kid = key_id(self.n)
        self.spent: set[str] = set()
        self.issued = 0
        self.redeemed = 0
        if ledger_file and os.path.exists(ledger_file):
            with open(ledger_file) as f:
                doc = json.load(f)
            self.spent = set(doc.get("spent", []))
            self.issued = int(doc.get("issued", 0))
            self.redeemed = int(doc.get("redeemed", 0))

    def _persist(self) -> None:
        if not self.ledger_file:
            return
        d = os.path.dirname(os.path.abspath(self.ledger_file))
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".credits-", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump({"kid": self.kid, "issued": self.issued, "redeemed": self.redeemed,
                       "spent": sorted(self.spent)}, f)
        os.replace(tmp, self.ledger_file)

    # -- public description ------------------------------------------------

    def describe(self) -> dict:
        return {
            "kid": self.kid, "n": format(self.n, "x"), "e": self.e, "bits": self.n.bit_length(),
            "scheme": "RSA-FDH blind signature (Chaum); serial = nullifier",
            "denominations": list(DENOMINATIONS),
            "price_per_credit_usd": self.price_per_credit_usd,
            "points_per_credit": POINTS_PER_CREDIT,
            "costs": dict(CREDIT_COSTS),
            "pool": {"issued": self.issued, "redeemed": self.redeemed, "outstanding": self.issued - self.redeemed},
            "ephemeral_key": self.ephemeral,
        }

    def price_usd(self, count: int) -> str:
        return f"${count * self.price_per_credit_usd:.4f}".rstrip("0").rstrip(".")

    # -- issuing -----------------------------------------------------------

    def sign_blinded(self, blinded_hex: list[str]) -> list[str]:
        out = []
        for h in blinded_hex:
            try:
                m = int(h, 16)
            except ValueError:
                raise ValueError("blinded messages must be hex integers")
            if not (0 < m < self.n):
                raise ValueError("blinded message out of range")
            out.append(format(pow(m, self._d, self.n), "x"))
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
            serial = verify_token(t, self.n, self.e, self.kid)
            if serial is None:
                return False, "invalid credit token (bad signature or wrong key id)"
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


__all__ = ["CreditPool", "CREDIT_IDENTITY", "DENOMINATIONS", "POINTS_PER_CREDIT", "credit_cost", "SERIAL_BYTES"]
