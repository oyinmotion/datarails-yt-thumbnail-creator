"""Turn frames and audio into five validated thumbnail concepts."""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path

from pydantic import ValidationError

from . import backoff
from .config import PLANNER_MODEL, TRANSCRIBE_MODEL
from .models import MAX_HEADLINE_WORDS, BatchPlan
from .openai_client import get_client
from .prompts import planner_prompt

log = logging.getLogger(__name__)


class PlanError(RuntimeError):
    """The planner could not produce a usable batch plan."""


# Module-level so tests can swap in a no-op and stay instant.
DEFAULT_SLEEPER = time.sleep


def _client(client=None):
    return get_client(client)


def _violations(exc: ValidationError) -> str:
    """The specific rules the model broke, in a form it can act on."""
    parts = []
    for error in exc.errors():
        where = ".".join(str(piece) for piece in error.get("loc", ()))
        parts.append(f"{where or 'plan'}: {error.get('msg', 'invalid')}")
    return "; ".join(parts) or str(exc)


def _data_url(image: Path) -> str:
    encoded = base64.b64encode(Path(image).read_bytes()).decode()
    return f"data:image/jpeg;base64,{encoded}"


def transcribe(audio: Path, client=None) -> str | None:
    """Transcribe the ad's audio. Returns None on failure — never fatal.

    A missing transcript weakens hook quality; losing the whole batch over it
    would be worse.
    """
    try:
        with Path(audio).open("rb") as handle:
            result = _client(client).audio.transcriptions.create(
                model=TRANSCRIBE_MODEL, file=handle,
            )
        text = (getattr(result, "text", "") or "").strip()
        return text or None
    except Exception:
        log.warning("transcription failed; planning from frames alone",
                    exc_info=True)
        return None


def build_plan(
    frames: list[Path],
    audio: Path | None,
    headline_override: str | None = None,
    context: str | None = None,
    client=None,
    sleeper=None,
) -> BatchPlan:
    if not frames:
        raise PlanError("No frames to plan from.")

    sleeper = sleeper or DEFAULT_SLEEPER

    active = _client(client)
    transcript = transcribe(audio, client=active) if audio else None

    instructions = planner_prompt(transcript, context, headline_override)
    frame_list = "\n".join(f"- {f.name}" for f in frames)
    instructions += (
        "\n## Available frames\n\nUse one of these exact filenames for "
        f"`frame_id`:\n\n{frame_list}\n"
    )
    base_instructions = instructions

    images = [
        {"type": "input_image", "image_url": _data_url(frame)}
        for frame in frames
    ]

    attempts = (1, 2)
    last_attempt = attempts[-1]
    last_error: Exception | None = None
    for attempt in attempts:
        content: list[dict] = [{"type": "input_text", "text": instructions}]
        content.extend(images)
        try:
            response = active.responses.parse(
                model=PLANNER_MODEL,
                input=[{"role": "user", "content": content}],
                text_format=BatchPlan,
            )
            result: BatchPlan = response.output_parsed
            result.validate_matrix()
            result.transcript_used = transcript is not None
            return result
        except ValidationError as exc:
            # A field rule was broken — nearly always a headline that ran long.
            # ValidationError subclasses ValueError, so it MUST be caught above
            # the matrix branch. This is correctable, so tell the model exactly
            # what it got wrong and let it try again.
            last_error = exc
            detail = _violations(exc)
            log.warning("planner attempt %s produced an invalid plan: %s",
                        attempt, detail)
            if attempt == last_attempt:
                raise PlanError(
                    f"The planner kept writing hooks we can't use: {detail}"
                ) from exc
            instructions = (
                base_instructions
                + "\n## Fix this\n\nYour previous answer was rejected: "
                + f"{detail}\n\nEvery headline must be "
                + f"{MAX_HEADLINE_WORDS} words or fewer. Rewrite all five and "
                + "obey every rule above.\n"
            )
            backoff.wait(attempt, sleeper=sleeper)
        except ValueError as exc:
            # validate_matrix() only. The locked pairing is generated from
            # MATRIX in the prompt, so breaking it means the model ignored a
            # spelled-out instruction; a reroll is not worth the user's wait.
            raise PlanError(f"The planner broke the matrix: {exc}") from exc
        except Exception as exc:
            last_error = exc
            log.warning("planner attempt %s failed", attempt, exc_info=True)
            if attempt != last_attempt:
                backoff.wait(attempt, sleeper=sleeper)

    raise PlanError(
        f"Couldn't plan thumbnails for that ad. ({last_error})"
    ) from last_error
