"""One place that constructs the OpenAI client.

Three modules make API calls (plan, render, qa) and each used to build its own
bare `OpenAI()`, which inherited the SDK's 600s default timeout. A render that
stalls holds one of the five pool workers for ten minutes and the user sees
nothing until every variant finishes, so the timeout is explicit and shared.

Every caller still accepts an injected `client`, which is what the test suite
passes; this factory only decides what a *real* client looks like.
"""

from __future__ import annotations

from .config import OPENAI_MAX_RETRIES, OPENAI_TIMEOUT_SECONDS


def get_client(client=None):
    """Return the injected client, or a configured real one."""
    if client is not None:
        return client
    from openai import OpenAI

    return OpenAI(
        timeout=OPENAI_TIMEOUT_SECONDS,
        max_retries=OPENAI_MAX_RETRIES,
    )
