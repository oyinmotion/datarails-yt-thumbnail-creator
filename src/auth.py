"""Google sign-in. One consent covers identity and Drive access."""

from __future__ import annotations

import logging
import re

from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from oauthlib.oauth2.rfc6749.errors import OAuth2Error
from requests.exceptions import RequestException

from .config import ALLOWED_EMAIL_DOMAIN, GOOGLE_SCOPES

log = logging.getLogger(__name__)


class AuthError(RuntimeError):
    """Sign-in failed or the account isn't allowed."""


def parse_allowlist(raw: object | None) -> set[str]:
    """Read an ALLOWED_EMAILS secret into a set.

    Accepts commas, semicolons, newlines or spaces as separators so whoever
    maintains the secret cannot get the format wrong. An empty or missing value
    means "no allowlist" — every @datarails.com account is accepted.
    """
    if not raw:
        return set()
    # TOML allows either form and both are reasonable to write, so accept both:
    #   ALLOWED_EMAILS = "a@x.com, b@x.com"
    #   ALLOWED_EMAILS = ["a@x.com", "b@x.com"]
    if isinstance(raw, (list, tuple, set)):
        raw = ",".join(str(item) for item in raw)
    parts = re.split(r"[,;\s]+", str(raw).strip())
    return {p.strip().lower() for p in parts if p.strip()}


def is_allowed_email(email: str | None, allowlist: set[str] | None = None) -> bool:
    """Domain gate, plus an optional per-person allowlist.

    The domain check is never skipped: an allowlisted address still has to be on
    the company domain, so a typo in the secret cannot let an outsider in.
    """
    if not email:
        return False
    cleaned = email.strip().lower()
    # Require non-empty local part before the @ and exact domain match
    if "@" not in cleaned:
        return False
    local_part, domain = cleaned.rsplit("@", 1)
    if not (bool(local_part) and domain == ALLOWED_EMAIL_DOMAIN):
        return False
    if allowlist:
        return cleaned in allowlist
    return True


def build_flow(client_id: str, client_secret: str, redirect_uri: str) -> Flow:
    return Flow.from_client_config(
        {
            "web": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [redirect_uri],
            }
        },
        scopes=GOOGLE_SCOPES,
        redirect_uri=redirect_uri,
    )


def authorization_url(flow: Flow) -> tuple[str, str]:
    return flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
        hd=ALLOWED_EMAIL_DOMAIN,     # send users straight to their work account
    )


def _email_from_credentials(credentials) -> str:
    try:
        service = build("oauth2", "v2", credentials=credentials,
                        cache_discovery=False)
        return service.userinfo().get().execute().get("email", "")
    except (OAuth2Error, RequestException) as exc:
        log.warning("Failed to fetch email from credentials", exc_info=True)
        raise AuthError(
            "Sign-in didn't complete. Please try again. "
            f"(technical reason: {type(exc).__name__})"
        ) from exc
    except Exception as exc:
        log.warning("Unexpected error fetching email from credentials",
                    exc_info=True)
        raise AuthError(
            "Sign-in didn't complete. Please try again. "
            f"(technical reason: {type(exc).__name__})"
        ) from exc


def exchange_code(
    flow,
    code: str,
    allowlist: set[str] | None = None,
    code_verifier: str | None = None,
) -> tuple[object, str]:
    # PKCE: google-auth-oauthlib generates a code_verifier when the sign-in URL
    # is built and Google requires the same value back here. The Flow that built
    # the URL is long gone by the time the browser returns, so the caller carries
    # the verifier across and hands it to us. Without it Google answers
    # "invalid_grant: Missing code verifier".
    if code_verifier:
        flow.code_verifier = code_verifier

    try:
        flow.fetch_token(code=code)
    except (OAuth2Error, RequestException) as exc:
        log.warning("OAuth token exchange failed", exc_info=True)
        raise AuthError(
            "Sign-in didn't complete. Please try again. "
            f"(technical reason: {type(exc).__name__})"
        ) from exc
    except Exception as exc:
        log.warning("Unexpected error during OAuth token exchange",
                    exc_info=True)
        raise AuthError(
            "Sign-in didn't complete. Please try again. "
            f"(technical reason: {type(exc).__name__})"
        ) from exc

    credentials = flow.credentials
    email = _email_from_credentials(credentials)
    if not is_allowed_email(email, allowlist):
        if allowlist and email and email.strip().lower().endswith(
            f"@{ALLOWED_EMAIL_DOMAIN}"
        ):
            raise AuthError(
                f"{email} isn't on this tool's access list yet. Ask Omer to add "
                "you."
            )
        raise AuthError(
            f"{email or 'That account'} isn't a {ALLOWED_EMAIL_DOMAIN} account. "
            "Sign in with your Datarails email."
        )
    return credentials, email
