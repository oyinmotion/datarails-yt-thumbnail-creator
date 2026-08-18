"""Plan data model. Pure data — no I/O, no API calls.

The hook/treatment pairing is fixed in code, not chosen by the model, so every
batch spans the space instead of clustering on one idea.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

HookType = Literal["stat", "question", "conflict", "pain", "outcome"]
Treatment = Literal[
    "split_screen", "face_closeup", "full_bleed", "text_dominant",
    "product_forward",
]

# (index, hook_type, treatment) — locked pairing.
MATRIX: list[tuple[int, HookType, Treatment]] = [
    (1, "stat", "split_screen"),
    (2, "question", "face_closeup"),
    (3, "conflict", "full_bleed"),
    (4, "pain", "text_dominant"),
    (5, "outcome", "product_forward"),
]

TREATMENT_BRIEF: dict[str, str] = {
    "split_screen": (
        "Both actors face off, one on each side, a hard vertical seam of light "
        "between them. Headline centered on the seam."
    ),
    "face_closeup": (
        "One actor's face fills roughly half the frame with a clear reaction. "
        "Headline stacked in the remaining space."
    ),
    "full_bleed": (
        "A single dramatic energy burst fills the frame behind both actors. "
        "Headline centered and dominant."
    ),
    "text_dominant": (
        "Typography carries the frame; the actor is smaller and offset to one "
        "side. The headline is the subject."
    ),
    "product_forward": (
        "The FinanceOS product surface or its mark is visible and legible, with "
        "one actor presenting it. Headline supports rather than competes."
    ),
}

MAX_HEADLINE_WORDS = 5


class Variant(BaseModel):
    index: int = Field(ge=1, le=5)
    hook_type: HookType
    treatment: Treatment
    headline: str
    frame_id: str
    second_frame_id: str | None = None
    scene_direction: str
    rationale: str

    @field_validator("headline")
    @classmethod
    def clean_headline(cls, v: str) -> str:
        v = " ".join(v.split()).rstrip(".").strip().upper()
        if not v:
            raise ValueError("headline cannot be empty")
        if len(v.split()) > MAX_HEADLINE_WORDS:
            raise ValueError(
                f"headline must be {MAX_HEADLINE_WORDS} words or fewer, got "
                f"{len(v.split())}: {v!r}"
            )
        return v


class BatchPlan(BaseModel):
    ad_summary: str
    transcript_used: bool
    variants: list[Variant] = Field(min_length=5, max_length=5)

    def validate_matrix(self) -> None:
        """Fail loudly if the model drifted off the locked pairing."""
        actual = [(v.index, v.hook_type, v.treatment) for v in self.variants]
        if sorted(actual) != sorted(MATRIX):
            raise ValueError(
                f"plan does not follow the locked matrix.\n"
                f"expected: {sorted(MATRIX)}\ngot:      {sorted(actual)}"
            )
