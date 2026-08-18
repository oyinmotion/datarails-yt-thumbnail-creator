"""Google sign-in. One consent covers identity and Drive access."""

from __future__ import annotations

from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

from .config import ALLOWED_EMAIL_DOMAIN, GOOGLE_SCOPES


class AuthError(RuntimeError):
    """Sign-in failed or the account isn't allowed."""


def is_allowed_email(email: str | None) -> bool:
    if not email:
        return False
    cleaned = email.strip().lower()
    return cleaned.endswith(f"@{ALLOWED_EMAIL_DOMAIN}")


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
    service = build("oauth2", "v2", credentials=credentials,
                    cache_discovery=False)
    return service.userinfo().get().execute().get("email", "")


def exchange_code(flow, code: str) -> tuple[object, str]:
    try:
        flow.fetch_token(code=code)
    except Exception as exc:
        raise AuthError(f"Sign-in didn't complete: {exc}") from exc

    credentials = flow.credentials
    email = _email_from_credentials(credentials)
    if not is_allowed_email(email):
        raise AuthError(
            f"{email or 'That account'} isn't a {ALLOWED_EMAIL_DOMAIN} account. "
            "Sign in with your Datarails email."
        )
    return credentials, email
