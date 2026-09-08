"""Load and fill the prompt files. No prompt text lives in Python."""

from __future__ import annotations

from functools import lru_cache

from .config import PRIMARY_RATIO, PROMPTS_DIR
from .models import (
    DEFAULT_VARIANTS,
    ROW_INTENT,
    STYLE_BRIEF,
    TREATMENT_BRIEF,
    Variant,
    matrix_for,
)
from .typeset import zone_instruction


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


def render_prompt(
    variant: Variant, people_in_ad: bool = True,
    style: str | None = None, ratio: str = PRIMARY_RATIO,
) -> str:
    """The art-only prompt. `style` overrides the slot's style (swap look)."""
    filled = (
        load("render")
        .replace("{style_brief}", STYLE_BRIEF[style or variant.style])
        .replace("{treatment_brief}", TREATMENT_BRIEF[variant.treatment])
        .replace("{scene_direction}", variant.scene_direction)
        .replace("{zone_instruction}", zone_instruction(variant.treatment, ratio))
    )
    if not people_in_ad:
        filled += NO_PEOPLE_RULE
    return filled


NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def matrix_instruction(count: int) -> str:
    """The pairing table, for exactly the rows this batch asked for."""
    rows = matrix_for(count)
    word = NUMBER_WORDS.get(count, str(count))
    thing = "variant" if count == 1 else "variants"
    lines = [
        f"You must return exactly {word} {thing}, one per row, using these "
        "exact pairings:",
        "",
        "| index | hook_type | treatment | what it is |",
        "|---|---|---|---|",
    ]
    for index, hook, treatment, _style in rows:
        lines.append(
            f"| {index} | {hook} | {treatment} | {ROW_INTENT.get(index, '')} |"
        )
    if count < DEFAULT_VARIANTS:
        lines += [
            "",
            f"Only these {word} rows. Do not add the others — this batch was "
            "deliberately asked for fewer, and the rows are ordered so the "
            "strongest, safest concepts come first.",
        ]
    return "\n".join(lines)


def planner_prompt(
    transcript: str | None,
    context: str | None,
    headline_override: str | None,
    variant_count: int = DEFAULT_VARIANTS,
) -> str:
    parts = [load("planner").replace(
        "{matrix_instruction}", matrix_instruction(variant_count)
    )]

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
