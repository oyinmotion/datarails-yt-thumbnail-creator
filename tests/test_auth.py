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


# --- optional per-person allowlist ----------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("a@datarails.com,b@datarails.com", {"a@datarails.com", "b@datarails.com"}),
    ("a@datarails.com b@datarails.com", {"a@datarails.com", "b@datarails.com"}),
    ("a@datarails.com; b@datarails.com", {"a@datarails.com", "b@datarails.com"}),
    ("a@datarails.com\nb@datarails.com", {"a@datarails.com", "b@datarails.com"}),
    ("  A@Datarails.com , b@datarails.com  ",
     {"a@datarails.com", "b@datarails.com"}),
    ("", set()),
    (None, set()),
])
def test_parse_allowlist_accepts_any_reasonable_format(raw, expected):
    assert auth.parse_allowlist(raw) == expected


def test_no_allowlist_means_the_whole_domain_is_welcome():
    assert auth.is_allowed_email("anyone@datarails.com", set())
    assert auth.is_allowed_email("anyone@datarails.com", None)


def test_an_allowlisted_person_gets_in():
    allowed = {"omer.y@datarails.com"}
    assert auth.is_allowed_email("omer.y@datarails.com", allowed)
    assert auth.is_allowed_email("  Omer.Y@Datarails.com  ", allowed)


def test_a_colleague_not_on_the_list_is_refused():
    assert not auth.is_allowed_email(
        "someone.else@datarails.com", {"omer.y@datarails.com"}
    )


def test_the_domain_gate_still_applies_to_allowlisted_addresses():
    """A typo in the secret must not become a way in for an outsider."""
    assert not auth.is_allowed_email(
        "attacker@gmail.com", {"attacker@gmail.com"}
    )
    assert not auth.is_allowed_email(
        "attacker@datarails.com.evil.com", {"attacker@datarails.com.evil.com"}
    )


def test_a_colleague_off_the_list_is_told_to_ask_rather_than_that_they_are_wrong(
    monkeypatch,
):
    class FakeFlow:
        credentials = object()

        def fetch_token(self, code=None):
            return None

    monkeypatch.setattr(auth, "_email_from_credentials",
                        lambda creds: "someone.else@datarails.com")
    with pytest.raises(auth.AuthError, match="access list"):
        auth.exchange_code(FakeFlow(), "code", allowlist={"omer.y@datarails.com"})


def test_a_non_datarails_account_still_gets_the_domain_message(monkeypatch):
    class FakeFlow:
        credentials = object()

        def fetch_token(self, code=None):
            return None

    monkeypatch.setattr(auth, "_email_from_credentials",
                        lambda creds: "someone@gmail.com")
    with pytest.raises(auth.AuthError, match="datarails.com account"):
        auth.exchange_code(FakeFlow(), "code", allowlist={"omer.y@datarails.com"})


# --- PKCE code verifier ---------------------------------------------------


def test_the_code_verifier_is_set_on_the_flow_before_the_exchange():
    """Google answers "invalid_grant: Missing code verifier" without this."""
    class FakeFlow:
        credentials = object()
        code_verifier = None

        def fetch_token(self, code=None):
            assert self.code_verifier == "the-verifier", (
                "the verifier must be on the flow BEFORE fetch_token runs"
            )

    flow = FakeFlow()
    import src.auth as auth_mod
    original = auth_mod._email_from_credentials
    auth_mod._email_from_credentials = lambda creds: "omer.y@datarails.com"
    try:
        auth.exchange_code(flow, "code", code_verifier="the-verifier")
    finally:
        auth_mod._email_from_credentials = original
    assert flow.code_verifier == "the-verifier"


def test_no_verifier_leaves_the_flow_untouched():
    """PKCE may legitimately be off; passing None must not clobber the flow."""
    class FakeFlow:
        credentials = object()
        code_verifier = "already-set"

        def fetch_token(self, code=None):
            return None

    flow = FakeFlow()
    import src.auth as auth_mod
    original = auth_mod._email_from_credentials
    auth_mod._email_from_credentials = lambda creds: "omer.y@datarails.com"
    try:
        auth.exchange_code(flow, "code", code_verifier=None)
    finally:
        auth_mod._email_from_credentials = original
    assert flow.code_verifier == "already-set"


def test_build_flow_produces_a_verifier_when_the_url_is_generated():
    """The real Flow must actually give us something to carry across."""
    flow = auth.build_flow("id.apps.googleusercontent.com", "secret",
                           "https://example.streamlit.app")
    url, state = auth.authorization_url(flow)
    assert state
    assert getattr(flow, "code_verifier", None), (
        "authorization_url should generate a PKCE verifier to carry forward"
    )
    assert "code_challenge" in url


def test_parse_allowlist_accepts_a_toml_array():
    """ALLOWED_EMAILS = ["a@x", "b@x"] is a natural way to write it in TOML."""
    assert auth.parse_allowlist(["A@datarails.com", "b@datarails.com"]) == {
        "a@datarails.com", "b@datarails.com",
    }
    assert auth.parse_allowlist(("a@datarails.com",)) == {"a@datarails.com"}
    assert auth.parse_allowlist([]) == set()
