"""Server side of Privacy Pass for the service: issuer, origin, ledger, keys.

One token pays for one priced call. Keys are private JWKs (with CRT
parameters) shared with the Cloudflare Worker, so tokens issued by either
redeem at either. Rotation is lossless: the issuing key signs new tokens,
older keys stay redeemable.

    TIMESFM3_PRIVACY_PASS_KEY_FILE=keys/pp-2026-09.json     # issuing (created if missing)
    TIMESFM3_PRIVACY_PASS_OLD_KEYS=keys/pp-2026-06.json,... # redeem-only
    TIMESFM3_PRIVACY_PASS_LEDGER_FILE=pp-ledger.json         # spent nonces, counters
    TIMESFM3_PRIVACY_PASS_ISSUER_NAME=api.example.com        # default: request host
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import tempfile
import threading

from .. import blindrsa as B
from .. import privacypass as PP
from .auth import ApiKey

TOKEN_IDENTITY = ApiKey(key="", name="privacy-pass", plan="prepaid")
DENOMINATIONS = (10, 25, 100)
DEFAULT_PRICE_PER_TOKEN_USD = 0.004
POINTS_PER_TOKEN = 256  # what a plan holder is charged per token bought


# -- RSA key generation in pure Python (runs once; persisted as a JWK).

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
    if p < q:
        p, q = q, p
    return B.private_jwk(B.PrivateKey(n, e, pow(e, -1, phi)), p, q)


class PoolKey:
    def __init__(self, jwk: dict, issuing: bool):
        self.private = B.private_from_jwk(jwk) if "d" in jwk else None
        self.public = B.public_from_jwk(jwk)
        self.key_id = PP.token_key_id(self.public)
        self.kid = self.key_id.hex()[:12]
        self.issuing = issuing


class PrivacyPassService:
    """Issuer + Origin + double-spend ledger for the FastAPI service."""

    def __init__(self, key_file: str | None = None, ledger_file: str | None = None,
                 price_per_token_usd: float = DEFAULT_PRICE_PER_TOKEN_USD, bits: int = 2048,
                 old_key_files: list[str] | None = None, issuer_name: str | None = None,
                 origin_name: str | None = None):
        self._lock = threading.Lock()
        self.ledger_file = ledger_file
        self.price_per_token_usd = price_per_token_usd
        self.issuer_name = issuer_name
        self.origin_name = origin_name  # default: the request host (per-origin tokens)
        self.ephemeral = key_file is None
        if key_file and os.path.exists(key_file):
            with open(key_file) as f:
                jwk = json.load(f)
            if "kty" not in jwk:
                raise ValueError(f"{key_file} is not a JWK; regenerate it with scripts/credits_keygen.py")
        else:
            jwk = generate_private_jwk(bits)
            if key_file:
                d = os.path.dirname(os.path.abspath(key_file))
                os.makedirs(d, exist_ok=True)
                fd, tmp = tempfile.mkstemp(dir=d, prefix=".pp-key-")
                with os.fdopen(fd, "w") as f:
                    json.dump(jwk, f)
                os.chmod(tmp, 0o600)
                os.replace(tmp, key_file)
        self.issuing_key = PoolKey(jwk, issuing=True)
        self.keys: dict[bytes, PoolKey] = {self.issuing_key.key_id: self.issuing_key}
        for path in old_key_files or []:
            with open(path) as f:
                k = PoolKey(json.load(f), issuing=False)
            self.keys.setdefault(k.key_id, k)
        self.issuer = PP.Issuer([self.issuing_key.private])
        self.origin = PP.Origin([k.public for k in self.keys.values()])
        self.spent: set[str] = set()
        self.issued = 0
        self.redeemed = 0
        if ledger_file and os.path.exists(ledger_file):
            with open(ledger_file) as f:
                doc = json.load(f)
            self.spent = set(doc.get("spent", []))
            self.issued = int(doc.get("issued", 0))
            self.redeemed = int(doc.get("redeemed", 0))

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
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".pp-", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump({"kid": self.kid, "issued": self.issued, "redeemed": self.redeemed, "spent": sorted(self.spent)}, f)
        os.replace(tmp, self.ledger_file)

    # -- origin: challenge and redemption ----------------------------------

    def challenge(self, host: str) -> PP.TokenChallenge:
        """Static per-origin challenge (empty redemption context): tokens are pre-purchasable."""
        origin = self.origin_name or host
        return PP.TokenChallenge(PP.TOKEN_TYPE_BLIND_RSA, self.issuer_name or origin, b"", origin)

    def challenge_header(self, host: str) -> str:
        return PP.www_authenticate(self.challenge(host), self.public, max_age=86400 * 30)

    def redeem(self, authorization: str, host: str) -> tuple[bool, str]:
        try:
            token = PP.parse_authorization(authorization)
        except (ValueError, TypeError):
            return False, "malformed PrivateToken header"
        if token is None:
            return False, "no PrivateToken credential"
        if token.token_key_id not in self.keys:
            return False, "unknown token key"
        if not self.origin.verify(token, self.challenge(host)):
            return False, "invalid token or wrong challenge"
        nul = hashlib.sha256(b"nullifier" + token.nonce).hexdigest()
        with self._lock:
            if nul in self.spent:
                return False, "token already spent"
            self.spent.add(nul)
            self.redeemed += 1
            self._persist()
        return True, ""

    # -- issuer -------------------------------------------------------------

    def directory(self) -> dict:
        return PP.issuer_directory([k.public for k in self.keys.values()], "/token-request")

    def issue(self, body: bytes) -> bytes:
        req = PP.TokenRequest.deserialize(body)
        res = self.issuer.issue(req)
        with self._lock:
            self.issued += 1
            self._persist()
        return res.serialize()

    def issue_batch(self, body: bytes, expected: int | None = None) -> tuple[bytes, int]:
        reqs = PP.deserialize_batched_request(body)
        if expected is not None and len(reqs) != expected:
            raise ValueError(f"this batch endpoint issues exactly {expected} tokens; {len(reqs)} requested")
        if not 1 <= len(reqs) <= 100:
            raise ValueError("batches are 1-100 tokens")
        out: list[PP.TokenResponse | None] = []
        issued = 0
        for r in reqs:
            try:
                out.append(self.issuer.issue(r))
                issued += 1
            except ValueError:
                out.append(None)
        with self._lock:
            self.issued += issued
            self._persist()
        return PP.serialize_batched_response(out), issued

    def describe(self) -> dict:
        return {
            "standard": "RFC 9576/9577/9578, token type 0x0002 (Blind RSA, PSS)",
            "keys": [{"kid": k.kid, "issuing": k.issuing} for k in self.keys.values()],
            "kid": self.kid,
            "issuer_directory": PP.ISSUER_DIRECTORY_PATH,
            "issue": "POST /token-request (application/private-token-request or generic batch) or /token-request/batch/{10|25|100}",
            "redeem": "Authorization: PrivateToken token=...  (one token = one priced call)",
            "denominations": list(DENOMINATIONS),
            "price_per_token_usd": self.price_per_token_usd,
            "points_per_token": POINTS_PER_TOKEN,
            "pool": {"issued": self.issued, "redeemed": self.redeemed, "outstanding": self.issued - self.redeemed},
            "ephemeral_key": self.ephemeral,
        }


__all__ = ["PrivacyPassService", "TOKEN_IDENTITY", "DENOMINATIONS", "POINTS_PER_TOKEN", "generate_private_jwk"]
