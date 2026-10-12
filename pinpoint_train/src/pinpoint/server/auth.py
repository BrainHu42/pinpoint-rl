from __future__ import annotations

import secrets
from dataclasses import dataclass

from fastapi import HTTPException, Request, status

from pinpoint.server.config import ServerConfig


PINPOINT_UPLOAD_TOKEN_HEADER = "X-Pinpoint-Upload-Token"


@dataclass(frozen=True)
class AuthenticatedRequester:
    login: str | None
    name: str | None
    authenticated_via: str


def _clean_header_value(raw_value: str | None) -> str | None:
    if raw_value is None:
        return None
    cleaned = raw_value.strip()
    return cleaned or None


def authenticate_request(
    request: Request,
    config: ServerConfig,
) -> AuthenticatedRequester:
    if not config.secret_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server secret token is not configured",
        )

    secret_token = _clean_header_value(request.headers.get(PINPOINT_UPLOAD_TOKEN_HEADER))
    auth_header = _clean_header_value(request.headers.get("Authorization"))
    if secret_token is None and auth_header and auth_header.lower().startswith("bearer "):
        secret_token = auth_header[7:].strip() or None

    if secret_token and secrets.compare_digest(secret_token, config.secret_token):
        return AuthenticatedRequester(
            login=None,
            name="secret-token",
            authenticated_via="secret-token",
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrected or missing secret upload token",
    )
