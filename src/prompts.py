"""Load and fill the prompt files. No prompt text lives in Python."""

from __future__ import annotations

from functools import lru_cache

from .config import PROMPTS_DIR
from .models import MATRIX, TREATMENT_BRIEF, Variant


@lru_cache(maxsize=None)
def load(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"No prompt file at {path}")
    return path.read_text(encoding="utf-8")


def render_prompt(variant: Variant) -> str:
    return (
        load("render")
        .replace("{treatment_brief}", TREATMENT_BRIEF[variant.treatment])
        .replace("{scene_direction}", variant.scene_direction)
        .replace("{headline}", variant.headline)
    )


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
