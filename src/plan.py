"""Turn frames and audio into five validated thumbnail concepts."""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from .config import PLANNER_MODEL, TRANSCRIBE_MODEL
from .models import BatchPlan
from .prompts import planner_prompt

log = logging.getLogger(__name__)


class PlanError(RuntimeError):
    """The planner could not produce a usable batch plan."""


def _client(client=None):
    if client is not None:
        return client
    from openai import OpenAI

    return OpenAI()


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
) -> BatchPlan:
    if not frames:
        raise PlanError("No frames to plan from.")

    active = _client(client)
    transcript = transcribe(audio, client=active) if audio else None

    instructions = planner_prompt(transcript, context, headline_override)
    frame_list = "\n".join(f"- {f.name}" for f in frames)
    instructions += (
        "\n## Available frames\n\nUse one of these exact filenames for "
        f"`frame_id`:\n\n{frame_list}\n"
    )

    content: list[dict] = [{"type": "input_text", "text": instructions}]
    for frame in frames:
        content.append({"type": "input_image", "image_url": _data_url(frame)})

    last_error: Exception | None = None
    for attempt in (1, 2):
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
        except ValueError as exc:          # includes matrix violations
            raise PlanError(f"The planner broke the matrix: {exc}") from exc
        except Exception as exc:
            last_error = exc
            log.warning("planner attempt %s failed", attempt, exc_info=True)

    raise PlanError(
        f"Couldn't plan thumbnails for that ad. ({last_error})"
    ) from last_error
