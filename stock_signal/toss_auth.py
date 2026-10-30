from __future__ import annotations

import time
from urllib.parse import urljoin

import requests


_TOKEN_CACHE: dict[tuple[str, str], tuple[str, float]] = {}


def _base_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    if base.endswith("/api"):
        base = base[:-4]
    return base + "/"


def _url(base_url: str, path: str) -> str:
    return urljoin(_base_url(base_url), path.lstrip("/"))


def get_cached_access_token(
    *,
    base_url: str,
    client_id: str,
    client_secret: str,
    timeout_seconds: int = 10,
) -> str:
    key = (_base_url(base_url), client_id)
    cached = _TOKEN_CACHE.get(key)
    now = time.time()
    if cached and cached[1] > now:
        return cached[0]

    response = requests.post(
        _url(base_url, "/oauth2/token"),
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    token = payload.get("access_token")
    if not token:
        raise ValueError("Toss OAuth response did not include access_token")
    expires_in = float(payload.get("expires_in") or payload.get("expiresIn") or 3600)
    _TOKEN_CACHE[key] = (str(token), now + max(60.0, expires_in - 60.0))
    return str(token)


def invalidate_cached_access_token(*, base_url: str, client_id: str) -> None:
    _TOKEN_CACHE.pop((_base_url(base_url), client_id), None)
