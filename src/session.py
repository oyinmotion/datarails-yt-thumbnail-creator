"""Keep a signed-in user signed in across page loads.

Streamlit's session_state dies on every full page load, so without this a user
re-runs the whole Google round trip each time they open the tool. A cookie holds
a signed token; the credentials themselves never leave the server.

The token is an HMAC over (email, expiry) using a server-side secret. It proves
"the server issued this, and it has not expired" and nothing else — so a stolen
cookie is worth only what the domain gate already allows, and only until it
expires.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time

log = logging.getLogger(__name__)

COOKIE_NAME = "dr_yt_session"
# A working week: long enough that nobody signs in twice in a day, short enough
# that removing someone from ALLOWED_EMAILS takes effect within days.
DEFAULT_TTL_SECONDS = 7 * 24 * 60 * 60
SEPARATOR = "."


def _signature(payload: str, secret: str) -> str:
    return hmac.new(
        secret.encode(), payload.encode(), hashlib.sha256
    ).hexdigest()


def mint_token(email: str, secret: str, ttl: int = DEFAULT_TTL_SECONDS,
               now: float | None = None) -> str:
    """A signed "<email>.<expiry>.<signature>"."""
    if not secret:
        raise ValueError("a signing secret is required")
    expires = int((now if now is not None else time.time()) + ttl)
    payload = f"{email.strip().lower()}{SEPARATOR}{expires}"
    return f"{payload}{SEPARATOR}{_signature(payload, secret)}"


def read_token(token: str | None, secret: str,
               now: float | None = None) -> str | None:
    """The email a valid token belongs to, or None.

    Returns None for anything at all suspect: wrong shape, bad signature,
    expired, or no secret configured. Never raises — a malformed cookie must
    show a sign-in screen, not an error page.
    """
    if not token or not secret:
        return None
    # rsplit, not split: real addresses contain dots (omer.y@datarails.com), so
    # splitting from the left shatters the email across parts. The expiry and
    # signature never contain the separator, so parsing from the right is exact.
    parts = token.rsplit(SEPARATOR, 2)
    if len(parts) != 3:
        return None
    email, expires_raw, provided = parts
    try:
        expires = int(expires_raw)
    except ValueError:
        return None

    payload = f"{email}{SEPARATOR}{expires}"
    # compare_digest, not ==, so a forged token cannot be found byte by byte.
    if not hmac.compare_digest(provided, _signature(payload, secret)):
        log.warning("session cookie failed signature check")
        return None
    if (now if now is not None else time.time()) >= expires:
        return None
    return email or None
