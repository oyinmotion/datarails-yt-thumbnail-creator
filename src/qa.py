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
from .openai_client import get_client
from .prompts import load

log = logging.getLogger(__name__)

# The width a thumbnail actually occupies in a YouTube feed.
FEED_WIDTH = 320

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)


@dataclass
class QAResult:
    ok: bool
    problems: list[str] = field(default_factory=list)
    # None means the vision model never read this render back: either the check
    # was skipped (hard checks already failed) or it was unavailable. Combined
    # with ok=True it means "passed unverified" — the pipeline turns that into a
    # user-visible warning so a silent QA outage can't masquerade as five
    # perfect renders.
    transcribed: str | None = None
    # SAME / DIFFERENT / NOBODY / UNCLEAR, or None when the check did not run.
    likeness: str | None = None

    @property
    def unverified(self) -> bool:
        return self.ok and self.transcribed is None


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


def hard_checks(
    path: Path, expected_size: tuple[int, int] = (FINAL_W, FINAL_H)
) -> list[str]:
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

    if size != expected_size:
        problems.append(
            f"dimensions are {size[0]}x{size[1]}, must be "
            f"{expected_size[0]}x{expected_size[1]}"
        )
    if path.stat().st_size > MAX_BYTES:
        problems.append("the file is too large for YouTube's 2 MB limit")
    return problems


def _feed_size_data_url(path: Path, width: int = FEED_WIDTH) -> str:
    """Downscale to `width` and return a data URL.

    The legibility check uses FEED_WIDTH (320) deliberately — that is the size a
    thumbnail actually occupies in a feed, so text that survives it is text a
    viewer can read. The likeness check needs more pixels than that to judge a
    face, so it passes a larger width.
    """
    with Image.open(path) as im:
        thumb = im.convert("RGB")
        height = round(thumb.height * width / thumb.width)
        thumb = thumb.resize((width, height), Image.LANCZOS)
        buffer = io.BytesIO()
        thumb.save(buffer, "PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


def _client(client=None):
    return get_client(client)


def likeness_verdict(
    render: Path, reference_frame: Path, client=None
) -> str | None:
    """Is the person in the render the actor from the ad?

    Returns SAME / DIFFERENT / NOBODY / UNCLEAR, or None if the check could not
    run. Prompt instructions alone have already proven insufficient here — a
    model asked for "a real professional" invented one — so this is the gate that
    actually enforces it.
    """
    try:
        response = _client(client).responses.create(
            model=QA_MODEL,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": load("qa_likeness")},
                    {"type": "input_image",
                     "image_url": _feed_size_data_url(render, width=512)},
                    {"type": "input_image",
                     "image_url": _feed_size_data_url(reference_frame, width=512)},
                ],
            }],
        )
    except Exception:
        log.warning("likeness check unavailable; passing unverified",
                    exc_info=True)
        return None

    answer = (getattr(response, "output_text", "") or "").strip().upper()
    for verdict in ("DIFFERENT", "NOBODY", "UNCLEAR", "SAME"):
        if verdict in answer:
            return verdict
    return None


def check(
    path: Path,
    intended_headline: str,
    reference_frame: Path | None = None,
    client=None,
    people_in_ad: bool = True,
    expected_size: tuple[int, int] = (FINAL_W, FINAL_H),
) -> QAResult:
    problems = hard_checks(path, expected_size)
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

    # Legibility passed. Now: is it actually our actor?
    verdict = None
    if reference_frame is not None:
        verdict = likeness_verdict(path, reference_frame, client=client)

        if not people_in_ad:
            # A motion-graphic ad has no actor, so ANY person here was invented
            # by the model — usually lifted out of a style reference. NOBODY is
            # the correct outcome, not a defect.
            if verdict in ("SAME", "DIFFERENT"):
                return QAResult(
                    ok=False,
                    problems=[
                        "this ad has no people in it, but the thumbnail shows a "
                        "person the model invented"
                    ],
                    transcribed=transcribed,
                    likeness=verdict,
                )
            return QAResult(
                ok=True, problems=[], transcribed=transcribed, likeness=verdict
            )

        if verdict in ("DIFFERENT", "NOBODY"):
            problem = (
                "the person in this thumbnail is not the actor from the ad"
                if verdict == "DIFFERENT"
                else "this thumbnail has no person in it at all"
            )
            return QAResult(
                ok=False,
                problems=[problem],
                transcribed=transcribed,
                likeness=verdict,
            )

    return QAResult(
        ok=True, problems=[], transcribed=transcribed, likeness=verdict
    )
