"""One variant in, image bytes out. One gpt-image-2 call per variant."""

from __future__ import annotations

import base64
import contextlib
import logging
from pathlib import Path

from .config import GEN_SIZE, IMAGE_MODEL, IMAGE_QUALITY
from .models import Variant
from .prompts import render_prompt
from .refs import pick_refs

log = logging.getLogger(__name__)

# images.edit accepts at most 16 input images for gpt-image models.
MAX_INPUT_IMAGES = 16


class RenderError(RuntimeError):
    """The image call failed."""


class RenderBlocked(RenderError):
    """Refused by the content filter — expected occasionally on real faces."""


def _client(client=None):
    if client is not None:
        return client
    from openai import OpenAI

    return OpenAI()


def _is_moderation(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(
        marker in text
        for marker in ("moderation", "content_policy", "content policy", "safety system")
    )


def render_variant(
    variant: Variant,
    frames: dict[str, Path],
    client=None,
    extra_instruction: str = "",
    frame_override: Path | None = None,
) -> bytes:
    """Render one thumbnail at GEN_SIZE. Returns raw PNG bytes."""
    primary = frame_override or frames.get(variant.frame_id)
    if primary is None:
        raise RenderError(
            f"The planner picked a frame we don't have: {variant.frame_id}"
        )

    image_paths: list[Path] = [primary]
    if variant.second_frame_id:
        if variant.second_frame_id in frames:
            image_paths.append(frames[variant.second_frame_id])
        else:
            log.warning(
                "second_frame_id %r not found in frames; rendering with "
                "only the primary frame",
                variant.second_frame_id,
            )
    image_paths.extend(pick_refs(variant.treatment, limit=3))
    image_paths = image_paths[:MAX_INPUT_IMAGES]

    prompt = render_prompt(variant)
    if extra_instruction:
        prompt += f"\n\n## Correction for this attempt\n\n{extra_instruction}"

    with contextlib.ExitStack() as stack:
        handles = [stack.enter_context(Path(p).open("rb")) for p in image_paths]
        try:
            result = _client(client).images.edit(
                model=IMAGE_MODEL,
                image=handles,
                prompt=prompt,
                size=GEN_SIZE,
                quality=IMAGE_QUALITY,
                output_format="png",
                n=1,
                # No input_fidelity: gpt-image-2 does not accept it and already
                # processes every input at high fidelity.
            )
        except Exception as exc:
            if _is_moderation(exc):
                raise RenderBlocked(
                    "The content filter refused this render."
                ) from exc
            raise RenderError(f"Image generation failed: {exc}") from exc

    return base64.b64decode(result.data[0].b64_json)
