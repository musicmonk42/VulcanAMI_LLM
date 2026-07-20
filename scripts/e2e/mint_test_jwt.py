#!/usr/bin/env python3
"""Mint a scoped HS256 JWT for the canonical runtime e2e harness."""
from __future__ import annotations
import argparse, base64, hashlib, hmac, json, os, time, uuid


def b64(obj: dict[str, object]) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--secret", default=os.environ.get("VULCAN_JWT_SECRET", ""))
    p.add_argument("--issuer", default=os.environ.get("VULCAN_JWT_ISSUER", "vulcan"))
    p.add_argument("--audience", default=os.environ.get("VULCAN_JWT_AUDIENCE", "vulcan-runtime"))
    p.add_argument("--subject", default="runtime-e2e")
    p.add_argument("--tenant", default="runtime-e2e")
    p.add_argument("--scope", action="append", default=[])
    p.add_argument("--ttl", type=int, default=900)
    args = p.parse_args()
    if len(args.secret.encode()) < 32:
        raise SystemExit("VULCAN_JWT_SECRET/--secret must be at least 32 bytes")
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT", "kid": "v1"}
    payload = {
        "iss": args.issuer,
        "aud": args.audience,
        "sub": args.subject,
        "tenant": args.tenant,
        "scope": " ".join(args.scope or ["reason:write", "audit:read", "operator:read", "memory:write", "memory:read"]),
        "iat": now,
        "nbf": now,
        "exp": now + args.ttl,
        "jti": "jti-" + uuid.uuid4().hex,
    }
    signing_input = f"{b64(header)}.{b64(payload)}"
    sig = hmac.new(args.secret.encode(), signing_input.encode("ascii"), hashlib.sha256).digest()
    print(signing_input + "." + base64.urlsafe_b64encode(sig).rstrip(b"=").decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
