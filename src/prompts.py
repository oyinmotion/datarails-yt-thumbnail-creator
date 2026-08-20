"""Load and fill the prompt files. No prompt text lives in Python."""

from __future__ import annotations

from functools import lru_cache

from .config import PROMPTS_DIR
from .models import STYLE_BRIEF, TREATMENT_BRIEF, Variant


@lru_cache(maxsize=None)
def load(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"No prompt file at {path}")
    return path.read_text(encoding="utf-8")


NO_PEOPLE_RULE = """

## This ad contains no people

The reference frames for this ad show no person at all — it is motion graphics,
screen recording or product footage. So this thumbnail must contain NO person:
no face, no hand, no silhouette, no figure in the background. Do not add a
presenter, a customer, a model or a stock person to make it feel human. Build
the frame from the ad's own graphics, the product surface, and typography.

Any people you can see in the house-style reference images are from an unrelated
ad. They are there to show palette, lighting and type treatment only. Never copy
a person out of a reference image.
"""


def render_prompt(variant: Variant, people_in_ad: bool = True) -> str:
    filled = (
        load("render")
        .replace("{style_brief}", STYLE_BRIEF[variant.style])
        .replace("{treatment_brief}", TREATMENT_BRIEF[variant.treatment])
        .replace("{scene_direction}", variant.scene_direction)
        .replace("{headline}", variant.headline)
    )
    if not people_in_ad:
        filled += NO_PEOPLE_RULE
    return filled


def planner_prompt(
    transcript: str | None,
    context: str | None,
    headline_override: str | None,
) -> str:
    parts = [load("planner")]

    if transcript:
        parts.append(f"\n## What is said in the ad\n\n{transcript.strip()}")
    else:
        parts.append(
            "\n## What is said in the ad\n\nNo transcript is available for this "
            "ad. Plan from the frames alone."
        )

    if context:
        parts.append(f"\n## Extra context from the team\n\n{context.strip()}")

    if headline_override:
        parts.append(
            "\n## Headline is fixed\n\nUse this exact headline, unchanged, for "
            f"all five variants:\n\n{headline_override.strip()}\n\nVary only the "
            "frame choice and scene direction."
        )

    # Frame filenames are appended by plan.py, which knows what it extracted.
    return "\n".join(parts)
