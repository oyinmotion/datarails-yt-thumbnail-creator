import time

import pytest

from src import session

SECRET = "a-server-side-secret"


def test_a_freshly_minted_token_reads_back():
    token = session.mint_token("omer.y@datarails.com", SECRET)
    assert session.read_token(token, SECRET) == "omer.y@datarails.com"


def test_the_email_is_normalised():
    token = session.mint_token("  Omer.Y@Datarails.com ", SECRET)
    assert session.read_token(token, SECRET) == "omer.y@datarails.com"


def test_a_token_signed_with_another_secret_is_refused():
    token = session.mint_token("omer.y@datarails.com", "other-secret")
    assert session.read_token(token, SECRET) is None


def test_tampering_with_the_email_is_refused():
    """The whole point: you cannot promote yourself by editing the cookie."""
    token = session.mint_token("someone@datarails.com", SECRET)
    email, expires, sig = token.rsplit(".", 2)
    forged = f"attacker@datarails.com.{expires}.{sig}"
    assert session.read_token(forged, SECRET) is None


def test_extending_the_expiry_is_refused():
    token = session.mint_token("omer.y@datarails.com", SECRET, ttl=10)
    email, expires, sig = token.rsplit(".", 2)
    forged = f"{email}.{int(expires) + 999999}.{sig}"
    assert session.read_token(forged, SECRET) is None


def test_an_expired_token_is_refused():
    now = time.time()
    token = session.mint_token("omer.y@datarails.com", SECRET, ttl=60, now=now)
    assert session.read_token(token, SECRET, now=now + 61) is None
    assert session.read_token(token, SECRET, now=now + 30)


@pytest.mark.parametrize("junk", [
    None, "", "not-a-token", "a.b", "a.notanumber.c",
    "omer.y@datarails.com.9999999999.deadbeef",   # right shape, wrong signature
])
def test_malformed_cookies_are_refused_without_raising(junk):
    assert session.read_token(junk, SECRET) is None


def test_no_secret_means_no_session():
    """Misconfiguration must fail closed, not open."""
    token = session.mint_token("omer.y@datarails.com", SECRET)
    assert session.read_token(token, "") is None
    with pytest.raises(ValueError):
        session.mint_token("omer.y@datarails.com", "")


def test_the_default_lifetime_is_about_a_week():
    assert 6 * 86400 <= session.DEFAULT_TTL_SECONDS <= 8 * 86400
