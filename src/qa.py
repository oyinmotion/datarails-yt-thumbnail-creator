"""Verification. The headline is set by us, so the only thing left to check on
a render is the person: is it really the actor from the ad?
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .config import FINAL_H, FINAL_W, MAX_BYTES, QA_MODEL
from .openai_client import get_client
from .prompts import load

log = logging.getLogger(__name__)

# The width a thumbnail actually occupies in a YouTube feed. The likeness
# check passes a larger width (512): it needs more pixels than that to judge a face.
FEED_WIDTH = 320


@dataclass
class QAResult:
    ok: bool
    problems: list[str] = field(default_factory=list)
    # SAME / DIFFERENT / NOBODY / UNCLEAR, or None when the check did not run.
    likeness: str | None = None
    # True when the vision model actually read this render. ok=True with
    # checked=False is "passed unverified" — the pipeline warns about it so a
    # silent outage cannot masquerade as a clean batch.
    checked: bool = False

    @property
    def unverified(self) -> bool:
        return self.ok and not self.checked


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


def likeness_gate(
    path: Path,
    reference_frame: Path | None,
    client=None,
    people_in_ad: bool = True,
) -> QAResult:
    """Pass, fail, or pass-unverified, on likeness alone.

    hard_checks are not run here: the art is at generation size at this point
    and the size/2MB rules apply to the delivered file, which
    postprocess.finalize_image guards.
    """
    if reference_frame is None:
        return QAResult(ok=True)

    verdict = likeness_verdict(path, reference_frame, client=client)
    if verdict is None:
        return QAResult(ok=True)                       # outage: unverified

    if not people_in_ad:
        if verdict in ("SAME", "DIFFERENT"):
            return QAResult(
                ok=False,
                problems=["this ad has no people in it, but the thumbnail shows a "
                          "person the model invented"],
                likeness=verdict, checked=True,
            )
        return QAResult(ok=True, likeness=verdict, checked=True)

    if verdict in ("DIFFERENT", "NOBODY"):
        problem = ("the person in this thumbnail is not the actor from the ad"
                   if verdict == "DIFFERENT"
                   else "this thumbnail has no person in it at all")
        return QAResult(ok=False, problems=[problem], likeness=verdict, checked=True)

    return QAResult(ok=True, likeness=verdict, checked=True)
