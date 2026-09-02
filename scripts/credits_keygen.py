"""Generate a Privacy Pass issuer key shared by the service and the edge Worker.

    python scripts/credits_keygen.py keys/privacypass-2026-09.json
    # service:  TIMESFM3_PRIVACY_PASS_KEY_FILE=keys/privacypass-2026-09.json timesfm3 serve
    # worker:   cd cloudflare && npx wrangler secret put PRIVACY_PASS_PRIVATE_JWK < keys/privacypass-2026-09.json
    # dev:      echo "PRIVACY_PASS_PRIVATE_JWK=$(cat keys/privacypass-2026-09.json)" >> cloudflare/.dev.vars

Rotation: generate a new file, point TIMESFM3_PRIVACY_PASS_KEY_FILE / the
secret at it, and list the old *public* JWK (printed below) in
TIMESFM3_PRIVACY_PASS_OLD_KEYS (service, as a file) or
PRIVACY_PASS_OLD_PUBLIC_JWKS (worker, JSON array) so tokens already sold keep
redeeming; the issuer directory lists every key.
"""

import json
import os
import sys

from timesfm3 import blindrsa as B
from timesfm3.privacypass import token_key_id
from timesfm3.serving.privacypass import generate_private_jwk


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "credits-key.json"
    jwk = generate_private_jwk()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(jwk, f)
    os.chmod(out, 0o600)
    pub = B.public_jwk(B.public_from_jwk(jwk))
    print(f"private JWK -> {out} (kid {token_key_id(B.public_from_jwk(jwk)).hex()[:12]})")
    print("public JWK (for PRIVACY_PASS_OLD_PUBLIC_JWKS after rotation):")
    print(json.dumps(pub))


if __name__ == "__main__":
    main()
