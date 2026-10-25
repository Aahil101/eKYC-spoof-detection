"""Authentication helpers.

- KYC flow: per-session bearer token (X-Session-Token), stored as SHA-256 only.
- Admin / monitor API: X-API-Key == ADMIN_API_KEY. In DEMO_MODE without a key,
  admin endpoints are allowed only from loopback clients (or any client when
  DEMO_ALLOW_ANY_CLIENT=1, which docker-compose sets while publishing the port on
  127.0.0.1 only).
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets

from fastapi import HTTPException, Request, status

LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def new_token() -> tuple[str, str]:
    t = secrets.token_urlsafe(32)
    return t, hash_token(t)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def token_matches(token: str | None, token_hash: str) -> bool:
    return bool(token) and hmac.compare_digest(hash_token(token), token_hash)


def is_admin(request: Request) -> bool:
    settings = request.app.state.settings
    key = request.headers.get("x-api-key")
    if settings.admin_api_key:
        return bool(key) and hmac.compare_digest(key, settings.admin_api_key)
    if settings.demo_mode:
        if os.environ.get("DEMO_ALLOW_ANY_CLIENT") == "1":
            return True
        host = request.client.host if request.client else ""
        return host in LOOPBACK
    return False


def require_admin(request: Request) -> None:
    if not is_admin(request):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "admin API key required")
