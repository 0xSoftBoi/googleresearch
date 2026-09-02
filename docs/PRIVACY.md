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
shared `x402` identity and credit traffic under `credit-pool`; the usage
ledger holds counts, not identities. Rate limiting at the edge hashes IPs
per minute and keeps nothing.

## The credit pool (this repo)

The scheme is **RFC 9474, RSABSSA-SHA384-PSSZERO-Deterministic** — the IETF
standard for RSA blind signatures, implemented in `timesfm3/blindrsa.py`
(pure Python, checked against all four of the RFC's test vectors) and, at
the edge, with Cloudflare's `@cloudflare/blindrsa-ts`. Tokens issued by
either verify in the other (tested), so any conforming client library can
buy and spend credits. The buyer picks random 32-byte serials, blinds them,
and asks the server to sign the blinded values; the server signs without
seeing the serials and charges once for the batch. The buyer unblinds and
holds tokens (`kid.serial.signature`). Redeeming a token proves it was
signed by a pool key and has not been spent (the serial is its nullifier),
and nothing more.

```bash
timesfm3 credits buy --api https://api.example.com --count 25 --private-key 0x...   # pay with x402
timesfm3 credits buy --api https://api.example.com --count 25 --api-key K           # or with a plan
timesfm3 credits status
```

```python
from timesfm3.client import ForecastClient
from timesfm3.credits import CreditWallet
client = ForecastClient("https://api.example.com", credits=CreditWallet("credits.json"))
client.forecast([history], horizon=24)   # spends 1 credit; nothing identifies you
```

Costs per call: forecast 1, volatility 1, anomalies 2, backtest 4 credits.
Denominations: 10, 25, 100 credits at $0.004 each (20% under pay-per-call).
`GET /v1/credits/pool` publishes the pool key and aggregate pool statistics
(issued, redeemed, outstanding) so buyers can judge the anonymity set.

What the pool does not hide: request timing and IP address (use a proxy,
Tor, or the Cloudflare edge, which forwards only a per-minute hashed
counter), and the fact that *someone* bought N credits at time T.

Operator notes: `python scripts/credits_keygen.py keys/credits.json` makes a
private JWK that both the service (`TIMESFM3_CREDITS_KEY_FILE`) and the
Cloudflare Worker (secret `CREDITS_PRIVATE_JWK`) load, so credits bought
from one redeem at the other. Set `TIMESFM3_CREDITS_LEDGER_FILE` so spent
serials persist (the Worker keeps them in D1). **Rotation never strands
tokens**: point the issuing key at a new file and list the old keys in
`TIMESFM3_CREDITS_OLD_KEYS` (files) or `CREDITS_OLD_PUBLIC_JWKS` (public
JWKs) — they stay valid for redemption while new purchases use the new key;
`GET /v1/credits/pool` lists every key with its `issuing` flag. The scheme
is single-server: the operator holds the signing key and could issue itself
free credits, which affects its own revenue, not buyers' privacy.

The free Cloudflare deployment sells and redeems credits **at the edge**
with no backend: the Worker blind-signs with the same library Cloudflare
uses for Privacy Pass, keeps nullifiers in a free D1 database, and takes
payment for batches with x402. That makes the privacy tier available on
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
4. From the fresh address, buy a batch of credits with x402:
   `timesfm3 credits buy --private-key <fresh key> --count 100`.
5. Spend credits. The chain shows a Privacy Pools withdrawal to an unknown
   address and one small purchase; the service shows anonymous tokens.

When Privacy Pools deploys on Base (the team has said multi-chain is the
next expansion), step 3 disappears. The service itself needs no change: it
never sees more than an address that paid for a batch.

**For the operator**: revenue arrives as many small USDC transfers into the
`pay_to` wallet, which makes income and customer count public. Sweeping the
wallet periodically into Privacy Pools and withdrawing to the treasury with
the ASP proof breaks that link as well, and keeps a compliance trail.

## Threat model summary

- Operator honest-but-curious: cannot link credit redemptions to buyers;
  can count pool activity; can see x402 payers (only if you pay per call).
- Facilitator: sees x402 payer and resource URL; buying credits in batches
  reduces this to one event per batch, on `/v1/credits/buy/{n}`.
- Chain observer: sees payer → operator transfers; a Privacy Pools
  withdrawal address hides which depositor paid.
- Network observer / edge: sees IPs and timing. Out of scope here; use the
  usual tools.
- Not protected: an operator who logs request bodies could still fingerprint
  a buyer by the *content* of the series. Self-host if that matters.
