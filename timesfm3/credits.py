"""Privacy Pass token wallet: prepaid, unlinkable API credits.

A payment ties a wallet to a request; a plan ties an API key to every
request it makes. Both let the operator (and, for x402, the facilitator)
reconstruct a customer's query history. For a quant desk the *pattern* of
forecasts is the secret, so the service also accepts **Privacy Pass**
tokens (RFC 9576/9577/9578, publicly verifiable, Blind RSA):

1. The buyer fetches the service's challenge (``WWW-Authenticate:
   PrivateToken``) and issuer directory, blinds random nonces and buys a
   batch of tokens (``POST /token-request``, paid once with x402 or a plan).
2. The issuer signs blinded messages without seeing the nonces.
3. Each priced call presents one token in ``Authorization: PrivateToken
   token=...``. The server verifies it and rejects reuse, but cannot link it
   to the purchase or to other tokens.

Tokens are interchangeable with the Cloudflare Worker's issuer and with any
conforming Privacy Pass client. Standard library only.
"""

from __future__ import annotations

import json
import os
import tempfile

from . import blindrsa as B
from . import privacypass as PP

#: Priced routes; every one costs exactly one token.
PRICED_ROUTES = ("POST /v1/forecast", "POST /v1/volatility", "POST /v1/anomalies", "POST /v1/backtest")


class CreditWallet:
    """Holds unspent serialized tokens (base64url) plus the challenge they were issued for."""

    def __init__(self, path: str | None = None):
        self.path = path
        self.tokens: list[str] = []
        self.challenge: str | None = None  # base64url TokenChallenge
        self.token_key: str | None = None  # base64url SPKI of the issuing key
        if path and os.path.exists(path):
            with open(path) as f:
                doc = json.load(f)
            self.tokens = list(doc.get("tokens", []))
            self.challenge = doc.get("challenge")
            self.token_key = doc.get("token_key")

    def save(self) -> None:
        if not self.path:
            return
        d = os.path.dirname(os.path.abspath(self.path))
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".pp-wallet-", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump({"challenge": self.challenge, "token_key": self.token_key, "tokens": self.tokens}, f)
        os.replace(tmp, self.path)

    def __len__(self) -> int:
        return len(self.tokens)

    # -- issuance ---------------------------------------------------------------

    def prepare(self, www_authenticate: str, count: int) -> tuple[bytes, list, bool]:
        """Builds the request body for ``count`` tokens from a challenge header.

        Returns ``(body, pending, batched)``: a single ``TokenRequest`` when
        count is 1, else a generic batched request.
        """
        entries = PP.parse_www_authenticate(www_authenticate)
        if not entries:
            raise ValueError("no PrivateToken challenge in WWW-Authenticate")
        e = entries[0]
        self.challenge = PP.b64e(e["challenge"].serialize())
        self.token_key = PP.b64e(e["token_key_bytes"])
        client = PP.Client(e["token_key"])
        pending = []
        reqs = []
        for _ in range(count):
            req, state = client.create_request(e["challenge"])
            reqs.append(req)
            pending.append((client, state))
        if count == 1:
            return reqs[0].serialize(), pending, False
        return PP.serialize_batched_request(reqs), pending, True

    def finish(self, pending: list, body: bytes, batched: bool) -> int:
        """Finalizes the issuer's response into stored tokens; returns how many."""
        responses = PP.deserialize_batched_response(body) if batched else [PP.TokenResponse.deserialize(body)]
        added = 0
        for (client, state), res in zip(pending, responses):
            if res is None:
                continue
            token = client.finalize(state, res)  # verifies the signature
            self.tokens.append(PP.b64e(token.serialize()))
            added += 1
        self.save()
        return added

    # -- spending ---------------------------------------------------------------

    def take(self) -> str:
        """Removes one token and returns the ``Authorization`` header value."""
        if not self.tokens:
            raise ValueError("wallet is empty; buy tokens first")
        t = self.tokens.pop(0)
        self.save()
        return f'{PP.AUTH_SCHEME} token="{t}"'

    @staticmethod
    def is_priced(method: str, path: str) -> bool:
        return f"{method.upper()} {path}" in PRICED_ROUTES
