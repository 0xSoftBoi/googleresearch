"""Generate a credit-pool signing key shared by the service and the edge Worker.

    python scripts/credits_keygen.py keys/credits-2026-09.json
    # service:  TIMESFM3_CREDITS_KEY_FILE=keys/credits-2026-09.json timesfm3 serve
    # worker:   cd cloudflare && npx wrangler secret put CREDITS_PRIVATE_JWK < keys/credits-2026-09.json
    # dev:      echo "CREDITS_PRIVATE_JWK=$(cat keys/credits-2026-09.json)" >> cloudflare/.dev.vars

Rotation: generate a new file, point TIMESFM3_CREDITS_KEY_FILE / the secret at
it, and list the old *public* JWK (printed below) in TIMESFM3_CREDITS_OLD_KEYS
(service, as a file) or CREDITS_OLD_PUBLIC_JWKS (worker, JSON array) so tokens
already sold keep redeeming.
"""

import json
import os
import sys

from timesfm3 import blindrsa as B
from timesfm3.credits import key_id
from timesfm3.serving.credits import generate_private_jwk


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "credits-key.json"
    jwk = generate_private_jwk()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(jwk, f)
    os.chmod(out, 0o600)
    pub = B.public_jwk(B.public_from_jwk(jwk))
    print(f"private JWK -> {out} (kid {key_id(B.public_from_jwk(jwk))})")
    print("public JWK (for CREDITS_OLD_PUBLIC_JWKS after rotation):")
    print(json.dumps(pub))


if __name__ == "__main__":
    main()
