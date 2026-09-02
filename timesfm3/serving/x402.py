"""Pay-per-call with x402: USDC over plain HTTP, no signup.

Anonymous callers -- AI agents in particular -- hit a priced endpoint, get a
``402 Payment Required`` whose ``PAYMENT-REQUIRED`` header states the price
and where to pay, sign a USDC transfer authorization, and retry with a
``PAYMENT-SIGNATURE`` header.  A facilitator verifies and settles the
transfer on-chain; the response carries ``PAYMENT-RESPONSE`` with the
transaction.  API-key holders (metered plans) bypass the paywall entirely.

Configuration (all optional; the paywall is off unless ``pay_to`` is set):

- ``TIMESFM3_X402_PAY_TO``       recipient wallet (0x..., EVM)
- ``TIMESFM3_X402_NETWORK``      CAIP-2, default ``eip155:84532`` (Base
                                 Sepolia testnet); production: ``eip155:8453``
- ``TIMESFM3_X402_FACILITATOR``  default by network: x402.org for testnet,
                                 Coinbase CDP for Base mainnet
- ``TIMESFM3_X402_FACILITATOR_AUTH``  optional ``Authorization`` header value
- ``TIMESFM3_X402_PRICES``       JSON ``{"POST /v1/forecast": "$0.005", ...}``

Built on the official ``x402`` package (``pip install "timesfm3[x402]"``).
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any

from .auth import ApiKey, KeyStore

#: Default USD prices per call. A forecast call may carry up to the service's
#: configured limits; larger workloads belong on a metered plan.
DEFAULT_PRICES: dict[str, str] = {
    "POST /v1/forecast": "$0.005",
    "POST /v1/anomalies": "$0.01",
    "POST /v1/backtest": "$0.02",
    "POST /v1/volatility": "$0.005",
}

DESCRIPTIONS = {
    "POST /v1/forecast": "TimesFM-3 forecast: point + 9 quantiles per series and step",
    "POST /v1/anomalies": "Walk-forward anomaly scoring against the model's predictive band",
    "POST /v1/backtest": "Walk-forward model comparison with Diebold-Mariano tests",
    "POST /v1/volatility": "HAR + RiskMetrics variance forecasts and vol-targeted sizing",
}

TESTNET = "eip155:84532"
MAINNET = "eip155:8453"
DEFAULT_FACILITATORS = {
    TESTNET: "https://x402.org/facilitator",
    MAINNET: "https://api.cdp.coinbase.com/platform/v2/x402",
}

X402_IDENTITY = ApiKey(key="", name="x402", plan="pay-per-call")


@dataclasses.dataclass(frozen=True)
class X402Config:
    pay_to: str
    network: str = TESTNET
    facilitator_url: str | None = None
    facilitator_auth: str | None = None
    prices: dict[str, str] = dataclasses.field(default_factory=lambda: dict(DEFAULT_PRICES))

    @property
    def facilitator(self) -> str:
        return self.facilitator_url or DEFAULT_FACILITATORS.get(self.network, DEFAULT_FACILITATORS[TESTNET])

    @property
    def mainnet(self) -> bool:
        return self.network == MAINNET

    @classmethod
    def from_env(cls) -> "X402Config | None":
        pay_to = os.environ.get("TIMESFM3_X402_PAY_TO", "").strip()
        if not pay_to:
            return None
        prices = dict(DEFAULT_PRICES)
        raw = os.environ.get("TIMESFM3_X402_PRICES")
        if raw:
            prices.update(json.loads(raw))
        return cls(
            pay_to=pay_to,
            network=os.environ.get("TIMESFM3_X402_NETWORK", TESTNET).strip() or TESTNET,
            facilitator_url=os.environ.get("TIMESFM3_X402_FACILITATOR") or None,
            facilitator_auth=os.environ.get("TIMESFM3_X402_FACILITATOR_AUTH") or None,
            prices=prices,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "network": self.network,
            "pay_to": self.pay_to,
            "facilitator": self.facilitator,
            "prices_usd": {k: v for k, v in self.prices.items()},
            "protocol": "x402 v2",
        }


def is_priced(config: X402Config | None, method: str, path: str) -> bool:
    return bool(config) and f"{method.upper()} {path}" in config.prices


def build_paid_app(app, config: X402Config):
    """Wraps a FastAPI app in the official x402 payment middleware."""
    from x402.http import FacilitatorConfig, HTTPFacilitatorClient, PaymentOption
    from x402.http.middleware.fastapi import PaymentMiddlewareASGI
    from x402.http.types import RouteConfig
    from x402.mechanisms.evm.exact import ExactEvmServerScheme
    from x402.server import x402ResourceServer

    fac_cfg: dict[str, Any] = {"url": config.facilitator}
    if config.facilitator_auth:
        auth = config.facilitator_auth
        fac_cfg["create_headers"] = lambda: {"Authorization": auth}
    facilitator = HTTPFacilitatorClient(FacilitatorConfig(**fac_cfg))
    server = x402ResourceServer(facilitator)
    server.register(config.network, ExactEvmServerScheme())
    routes = {
        route: RouteConfig(
            accepts=[PaymentOption(scheme="exact", pay_to=config.pay_to, price=price,
                                   network=config.network)],
            mime_type="application/json",
            description=DESCRIPTIONS.get(route, route),
        )
        for route, price in config.prices.items()
    }
    return PaymentMiddlewareASGI(app, routes=routes, server=server)


class X402Gate:
    """ASGI front door: plan customers (API key) skip the paywall, everyone
    else goes through the x402 middleware for priced routes."""

    def __init__(self, app, keys: KeyStore, config: X402Config):
        self.app = app
        self.keys = keys
        self.config = config
        self.paid_app = build_paid_app(app, config)

    def __getattr__(self, name):  # TestClient / uvicorn see the FastAPI app's attributes
        return getattr(self.app, name)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        key = headers.get("x-api-key")
        auth = headers.get("authorization", "")
        if not key and auth.lower().startswith("bearer "):
            key = auth[7:].strip()
        if key and self.keys.lookup(key):
            return await self.app(scope, receive, send)
        scope.setdefault("state", {})["x402_gate"] = True
        return await self.paid_app(scope, receive, send)


def paid_identity(request, config: X402Config | None) -> ApiKey | None:
    """The anonymous-but-paid identity for a request that cleared the paywall."""
    if config is None or not request.scope.get("state", {}).get("x402_gate"):
        return None
    if not is_priced(config, request.method, request.url.path):
        return None
    if not (request.headers.get("payment-signature") or request.headers.get("x-payment")):
        return None
    return X402_IDENTITY
