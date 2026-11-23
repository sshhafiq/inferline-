"""API-key authentication.

When ``api_keys`` is empty, authentication is disabled and every request is
treated as ``"anonymous"`` — convenient for local development. Otherwise the
``Authorization: Bearer <key>`` header is required and the key must be listed.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer_scheme = HTTPBearer(auto_error=False)

ANONYMOUS = "anonymous"


def is_valid_key(api_keys: list[str], candidate: str | None) -> bool:
    """True when auth is disabled or ``candidate`` is one of ``api_keys``."""
    if not api_keys:
        return True
    return candidate is not None and candidate in api_keys


def make_auth_dependency(api_keys: list[str]):
    """Build a FastAPI dependency that returns the caller's key identity."""

    async def verify(
        credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    ) -> str:
        if not api_keys:
            return ANONYMOUS
        if credentials is None or credentials.credentials not in api_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return credentials.credentials

    return verify
