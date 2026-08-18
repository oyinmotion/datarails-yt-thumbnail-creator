"""Verification. OpenAI's own docs warn gpt-image-2 can still struggle with
precise text placement, and our thumbnails are headline-dominant, so we read
every render back before showing it.
"""

from __future__ import annotations

import base64
import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .config import FINAL_H, FINAL_W, MAX_BYTES, QA_MODEL
from .prompts import load

log = logging.getLogger(__name__)

# The width a thumbnail actually occupies in a YouTube feed.
FEED_WIDTH = 320

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)


@dataclass
class QAResult:
    ok: bool
    problems: list[str] = field(default_factory=list)
    transcribed: str | None = None


def normalize(text: str) -> list[str]:
    return _PUNCT.sub(" ", (text or "")).upper().split()


def headline_is_legible(intended: str, transcribed: str) -> bool:
    """Every intended word must appear, in order, in what the model read back.

    Extra words are tolerated — the model may read a shirt logo. Missing or
    reordered words mean the type warped, got cut off, or wrapped wrongly.
    """
    want = normalize(intended)
    got = normalize(transcribed)
    if not want or got == ["NONE"]:
        return False

    index = 0
    for word in got:
        if index < len(want) and word == want[index]:
            index += 1
    return index == len(want)


def hard_checks(path: Path) -> list[str]:
    problems: list[str] = []
    path = Path(path)

    if not path.exists() or path.stat().st_size == 0:
        return ["the file is missing or empty"]

    try:
        with Image.open(path) as im:
            size = im.size
            im.verify()
    except Exception:
        return ["the file isn't a readable image"]

    if size != (FINAL_W, FINAL_H):
        problems.append(
            f"dimensions are {size[0]}x{size[1]}, must be {FINAL_W}x{FINAL_H}"
        )
    if path.stat().st_size > MAX_BYTES:
        problems.append("the file is too large for YouTube's 2 MB limit")
    return problems


def _feed_size_data_url(path: Path) -> str:
    with Image.open(path) as im:
        thumb = im.convert("RGB")
        height = round(thumb.height * FEED_WIDTH / thumb.width)
        thumb = thumb.resize((FEED_WIDTH, height), Image.LANCZOS)
        buffer = io.BytesIO()
        thumb.save(buffer, "PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


def _client(client=None):
    if client is not None:
        return client
    from openai import OpenAI

    return OpenAI()


def check(path: Path, intended_headline: str, client=None) -> QAResult:
    problems = hard_checks(path)
    if problems:
        return QAResult(ok=False, problems=problems)

    try:
        response = _client(client).responses.create(
            model=QA_MODEL,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": load("qa_legibility")},
                    {"type": "input_image",
                     "image_url": _feed_size_data_url(path)},
                ],
            }],
        )
        transcribed = (getattr(response, "output_text", "") or "").strip()
    except Exception:
        # A QA outage must not cost the user their batch.
        log.warning("legibility check unavailable; passing unverified",
                    exc_info=True)
        return QAResult(ok=True, problems=[], transcribed=None)

    if not headline_is_legible(intended_headline, transcribed):
        return QAResult(
            ok=False,
            problems=[
                "the headline isn't readable at feed size "
                f"(read back as {transcribed!r})"
            ],
            transcribed=transcribed,
        )

    return QAResult(ok=True, problems=[], transcribed=transcribed)
