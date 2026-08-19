import time

from src import backoff


def test_delay_grows_exponentially():
    assert backoff.delay_for(1) < backoff.delay_for(2) < backoff.delay_for(3)
    assert backoff.delay_for(2) == backoff.delay_for(1) * 2


def test_first_delay_is_not_zero():
    """An immediate retry lands inside the same rate-limit window."""
    assert backoff.delay_for(1) >= 1.0


def test_delay_is_capped():
    assert backoff.delay_for(50) == backoff.MAX_DELAY


def test_offset_staggers_concurrent_callers():
    """Five variants must not wake from backoff at the same instant."""
    delays = {backoff.delay_for(1, offset=i * backoff.STAGGER)
              for i in range(1, 6)}
    assert len(delays) == 5


def test_negative_offset_is_ignored():
    assert backoff.delay_for(1, offset=-99) == backoff.delay_for(1)


def test_wait_sleeps_the_delay_it_returns():
    slept = []
    delay = backoff.wait(2, sleeper=slept.append)
    assert slept == [delay] == [backoff.delay_for(2)]


def test_default_sleeper_is_real_sleep():
    assert backoff.wait.__defaults__[0] is time.sleep
