"""Exponential backoff for retried API calls.

Five renders fire concurrently, so a rate limit hits all five at once. Retrying
immediately — which is what both retry loops used to do — lands the second
attempt inside the same rate window and the user gets five empty tiles.

`sleeper` is injectable so tests stay instant.
"""

from __future__ import annotations

import time

BASE_DELAY = 2.0        # seconds before the first retry
MAX_DELAY = 30.0        # never make a user wait longer than this
# Concurrent variants must not wake up together, or they re-collide. A small
# per-variant offset staggers them deterministically — no randomness, so tests
# and reruns behave identically.
STAGGER = 0.5


def delay_for(attempt: int, offset: float = 0.0) -> float:
    """Seconds to wait after `attempt` failed. 1-based, doubling each time."""
    steps = max(0, attempt - 1)
    return min(BASE_DELAY * (2 ** steps) + max(0.0, offset), MAX_DELAY)


def wait(attempt: int, sleeper=time.sleep, offset: float = 0.0) -> float:
    """Sleep the backoff for `attempt` and return the delay used."""
    delay = delay_for(attempt, offset)
    sleeper(delay)
    return delay
