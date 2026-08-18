import pytest

from src import auth
from src.config import GOOGLE_SCOPES


@pytest.mark.parametrize("email", [
    "omer.y@datarails.com",
    "Someone.Else@Datarails.com",
    "  omer.y@datarails.com  ",
])
def test_datarails_emails_are_allowed(email):
    assert auth.is_allowed_email(email)


@pytest.mark.parametrize("email", [
    "someone@gmail.com",
    "attacker@datarails.com.evil.com",
    "datarails.com",
    "@datarails.com",
    "x@sub.datarails.com",
    "",
    None,
])
def test_everything_else_is_rejected(email):
    assert not auth.is_allowed_email(email)


def test_build_flow_requests_the_drive_scope():
    flow = auth.build_flow("id.apps.googleusercontent.com", "secret",
                           "https://example.streamlit.app")
    assert set(GOOGLE_SCOPES).issubset(set(flow.oauth2session.scope))


def test_authorization_url_forces_the_datarails_domain_hint():
    flow = auth.build_flow("id.apps.googleusercontent.com", "secret",
                           "https://example.streamlit.app")
    url, state = auth.authorization_url(flow)
    assert "hd=datarails.com" in url
    assert state


def test_exchange_code_rejects_a_non_datarails_account(monkeypatch):
    class FakeFlow:
        credentials = object()

        def fetch_token(self, code=None):
            return None

    monkeypatch.setattr(auth, "_email_from_credentials",
                        lambda creds: "someone@gmail.com")
    with pytest.raises(auth.AuthError, match="datarails.com"):
        auth.exchange_code(FakeFlow(), "code123")


def test_exchange_code_returns_credentials_and_email(monkeypatch):
    sentinel = object()

    class FakeFlow:
        credentials = sentinel

        def fetch_token(self, code=None):
            return None

    monkeypatch.setattr(auth, "_email_from_credentials",
                        lambda creds: "omer.y@datarails.com")
    creds, email = auth.exchange_code(FakeFlow(), "code123")
    assert creds is sentinel
    assert email == "omer.y@datarails.com"


def test_exchange_code_hides_exception_details_from_user(monkeypatch):
    """Verify that sensitive details in exceptions are not leaked to the UI."""
    sentinel_secret = "client_secret=SUPERSECRET"

    class FakeFlow:
        credentials = object()

        def fetch_token(self, code=None):
            # Simulate an exception that contains sensitive information
            raise ValueError(sentinel_secret)

    with pytest.raises(auth.AuthError) as exc_info:
        auth.exchange_code(FakeFlow(), "code123")

    # The user-facing error message should be generic
    assert "Sign-in didn't complete" in str(exc_info.value)
    # The sensitive detail should NOT appear in the error message
    assert sentinel_secret not in str(exc_info.value)
