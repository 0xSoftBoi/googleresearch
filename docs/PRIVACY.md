# Privacy pools: paying without being profiled

*What the service can and cannot learn about you, and how to pay so that it
learns as little as possible.*

For a trading desk, an ops team or an agent, the **pattern of questions**
asked of a forecaster is the sensitive thing — which series, how often, at
what horizon. Every payment channel leaks some of it. This document lays out
the leak per channel and the two "pools" that close it: the on-chain
[Privacy Pools](https://privacypools.com) protocol for *funds*, and this
service's blind-signed **credit pool** for *usage*.

## What each channel reveals

| Channel | Who learns what |
|---|---|
| API key (plan) | The operator links every call to the key holder; usage is metered per key. |
| x402 pay-per-call | The operator sees the payer wallet on every call; the facilitator sees payer, amount and the resource URL; the chain sees payer → operator transfers. |
| **Prepaid credits** | The operator sees *one* purchase (via x402 or a plan) and then a stream of anonymous, single-use tokens it cannot link to the purchase or to each other. |
| **Credits bought from a Privacy Pools withdrawal address** | Additionally, the paying wallet cannot be linked on-chain to the buyer's main wallet. |

The service stores no payer addresses. x402 traffic is metered under one
shared `x402` identity and token traffic under `privacy-pass`; the usage
ledger holds counts, not identities. Rate limiting at the edge hashes IPs
per minute and keeps nothing.

## The credit pool (this repo): Privacy Pass

Credits are **Privacy Pass tokens** — the IETF architecture (RFC 9576),
HTTP authentication scheme (RFC 9577) and issuance protocol (RFC 9578) that
Cloudflare, Apple and Fastly already ship. Token type `0x0002` (Blind RSA,
2048-bit, RFC 9474 RSABSSA-SHA384-PSS-Deterministic) is implemented in
`timesfm3/privacypass.py` on top of `timesfm3/blindrsa.py` (pure Python,
checked against the RFC 9474 vectors) and, at the edge, with Cloudflare's
own `@cloudflare/privacypass-ts`. Tokens issued by either verify in the
other (tested in every issuer/client/origin combination), so any conforming
Privacy Pass client can buy and spend credits.

The flow is the standard one. A priced call answered with
`401 WWW-Authenticate: PrivateToken challenge=..., token-key=...` carries a
challenge bound to the origin. The client picks a random 32-byte nonce,
blinds the token input, and posts a `TokenRequest` to
`POST /token-request` (one token, `application/private-token-request`) or a
generic batched request to `POST /token-request/batch/{10|25|100}` (RFC 9578
§6.3, one payment for the batch). The issuer signs without seeing the nonce.
The client unblinds and holds tokens; each is spent once as
`Authorization: PrivateToken token="..."`. The origin checks the signature
against a key from `GET /.well-known/private-token-issuer-directory`, checks
the challenge digest, and records the nonce as a nullifier. Nothing links a
redeemed token to the purchase or to other tokens.

```bash
timesfm3 credits buy --api https://api.example.com --count 25 --private-key 0x...   # pay with x402
timesfm3 credits buy --api https://api.example.com --count 25 --api-key K           # or with a plan
timesfm3 credits status
```

```python
from timesfm3.client import ForecastClient
from timesfm3.credits import CreditWallet
client = ForecastClient("https://api.example.com", credits=CreditWallet("credits.json"))
client.forecast([history], horizon=24)   # spends 1 token; nothing identifies you
```

One token pays for one priced call (forecast, volatility, anomalies,
backtest). Denominations: 1 token at $0.004, batches of 10/25/100 at
$0.04/$0.10/$0.40 (20% under pay-per-call). `GET /token-request/stats`
publishes aggregate issued/redeemed counts per key so buyers can judge the
anonymity set; `GET /token-request/challenge` always answers 401 with a
fresh challenge for clients that want to buy before their first call.

What the pool does not hide: request timing and IP address (use a proxy,
Tor, or the Cloudflare edge, which forwards only a per-minute hashed
counter), and the fact that *someone* bought N tokens at time T.

Operator notes: `python scripts/credits_keygen.py keys/privacypass.json`
makes a private JWK that both the service (`TIMESFM3_PRIVACY_PASS_KEY_FILE`)
and the Cloudflare Worker (secret `PRIVACY_PASS_PRIVATE_JWK`) load, so
tokens bought from one redeem at the other. Set
`TIMESFM3_PRIVACY_PASS_LEDGER_FILE` so spent nonces persist (the Worker
keeps them in D1). Challenges are bound to the origin name: set
`TIMESFM3_PRIVACY_PASS_ORIGIN` / `PRIVACY_PASS_ORIGIN` to the public host
when the service sits behind a proxy so both sides issue the same challenge.
**Rotation never strands tokens**: point the issuing key at a new file and
list the old keys in `TIMESFM3_PRIVACY_PASS_OLD_KEYS` (files) or
`PRIVACY_PASS_OLD_PUBLIC_JWKS` (public JWKs) — the issuer directory lists
every key, new purchases use the newest, and old tokens stay redeemable.
The scheme is single-issuer: the operator holds the signing key and could
issue itself free tokens, which affects its own revenue, not buyers'
privacy.

The free Cloudflare deployment issues and redeems tokens **at the edge**
with no backend: the Worker runs Cloudflare's Privacy Pass library, keeps
nullifiers in a free D1 database, and takes payment for batches with the
official `@x402/hono` middleware. That makes the privacy tier available on
the $0 deployment, not only to self-hosters.

## Privacy Pools (on-chain)

[Privacy Pools](https://docs.privacypools.com) — the design from Buterin,
Illum, Nadler, Schär and Soleimani, shipped by 0xbow — lets a wallet deposit
funds and later withdraw them to a **fresh address with no on-chain link**,
while proving in zero knowledge that the deposit belongs to an *association
set* screened by an Association Set Provider. It is privacy with a
compliance story: withdrawals must carry a proof of membership in the
approved set, and a rejected depositor can always "ragequit" their own
funds publicly.

As of September 2026 the deployment is Ethereum mainnet (chain id 1;
entrypoint `0x6818809eefce719e480a7526d76bd3e561526b46`) with ETH and the
stablecoins the team has added since launch (USDC, USDT, DAI, USDS). No Base
deployment is listed, and the x402 facilitators do not settle on Ethereum
mainnet, so the flow today bridges once:

1. Deposit USDC into Privacy Pools from your main wallet and wait for ASP
   approval.
2. Withdraw to a **fresh** address (the app at privacypools.com or the
   `@0xbow/privacy-pools-core-sdk` toolkit generates the withdrawal proof and
   a relayer submits it, so the fresh address never needs gas from you).
3. Bridge that USDC to Base with Circle's CCTP (the fresh address is the
   only party involved).
4. From the fresh address, buy a batch of tokens with x402:
   `timesfm3 credits buy --private-key <fresh key> --count 100`.
5. Spend tokens. The chain shows a Privacy Pools withdrawal to an unknown
   address and one small purchase; the service shows anonymous tokens.

When Privacy Pools deploys on Base (the team has said multi-chain is the
next expansion), step 3 disappears. The service itself needs no change: it
never sees more than an address that paid for a batch.

**For the operator**: revenue arrives as many small USDC transfers into the
`pay_to` wallet, which makes income and customer count public. Sweeping the
wallet periodically into Privacy Pools and withdrawing to the treasury with
the ASP proof breaks that link as well, and keeps a compliance trail.

## Threat model summary

- Operator honest-but-curious: cannot link token redemptions to buyers;
  can count pool activity; can see x402 payers (only if you pay per call).
- Facilitator: sees x402 payer and resource URL; buying tokens in batches
  reduces this to one event per batch, on `/token-request/batch/{n}`.
- Chain observer: sees payer → operator transfers; a Privacy Pools
  withdrawal address hides which depositor paid.
- Network observer / edge: sees IPs and timing. Out of scope here; use the
  usual tools.
- Not protected: an operator who logs request bodies could still fingerprint
  a buyer by the *content* of the series. Self-host if that matters.
