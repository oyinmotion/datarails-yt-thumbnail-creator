import json

import pytest
from openai.lib._pydantic import to_strict_json_schema
from pydantic import ValidationError

from src.models import MATRIX, BatchPlan, Variant


def _variant(index=1, hook="stat", treatment="split_screen", headline="47K OVER"):
    return Variant(
        index=index,
        hook_type=hook,
        treatment=treatment,
        headline=headline,
        frame_id="scene_004.jpg",
        second_frame_id=None,
        scene_direction="orange versus blue split, sparks at the seam",
        rationale="the ad names a budget overrun",
    )


def _plan(variants=None):
    return BatchPlan(
        ad_summary="two colleagues get different answers from the same AI",
        transcript_used=True,
        variants=variants or [_variant(i, h, t) for i, h, t in MATRIX],
    )


def test_matrix_has_five_unique_rows():
    assert len(MATRIX) == 5
    assert len({row[1] for row in MATRIX}) == 5   # 5 distinct hooks
    assert len({row[2] for row in MATRIX}) == 5   # 5 distinct treatments
    assert [row[0] for row in MATRIX] == [1, 2, 3, 4, 5]


def test_valid_plan_passes_matrix_validation():
    _plan().validate_matrix()


def test_plan_with_four_variants_is_rejected():
    # Enforced in validate_matrix(), not by Field(min_length=...): a length
    # constraint would put minItems/maxItems in the strict schema.
    with pytest.raises(ValueError, match="matrix"):
        _plan(variants=[_variant(i, h, t) for i, h, t in MATRIX[:4]]).validate_matrix()


def test_plan_with_six_variants_is_rejected():
    six = [_variant(i, h, t) for i, h, t in MATRIX] + [_variant(1, "stat", "split_screen")]
    with pytest.raises(ValueError, match="matrix"):
        _plan(variants=six).validate_matrix()


def test_plan_with_an_out_of_range_index_is_rejected():
    bad = [_variant(i, h, t) for i, h, t in MATRIX]
    bad[0].index = 9
    with pytest.raises(ValueError, match="matrix"):
        _plan(variants=bad).validate_matrix()


def test_plan_with_a_duplicated_index_is_rejected():
    bad = [_variant(i, h, t) for i, h, t in MATRIX]
    bad[1].index = 1
    with pytest.raises(ValueError, match="matrix"):
        _plan(variants=bad).validate_matrix()


def test_strict_schema_emits_no_length_or_range_keywords():
    """Regression guard: these four keywords 400 the structured-output call.

    Field(min_length=5, max_length=5) on variants and Field(ge=1, le=5) on
    index used to put all four in the emitted schema, which would have failed
    every planner call. The rules live in validate_matrix() now.
    """
    schema = json.dumps(to_strict_json_schema(BatchPlan))
    for keyword in ("minItems", "maxItems", "minimum", "maximum"):
        assert keyword not in schema, f"{keyword} is back in the strict schema"


def test_headline_validator_adds_no_schema_keyword():
    """A field_validator runs after parsing and must not shape the schema."""
    schema = to_strict_json_schema(BatchPlan)
    headline = schema["$defs"]["Variant"]["properties"]["headline"]
    assert set(headline) <= {"type", "title", "description"}


def test_plan_with_wrong_pairing_is_rejected():
    bad = [_variant(i, h, t) for i, h, t in MATRIX]
    bad[0].treatment = "full_bleed"          # stat must pair with split_screen
    with pytest.raises(ValueError, match="matrix"):
        _plan(variants=bad).validate_matrix()


def test_unknown_treatment_is_rejected():
    with pytest.raises(ValidationError):
        _variant(treatment="interpretive_dance")


def test_headline_longer_than_five_words_is_rejected():
    with pytest.raises(ValidationError):
        _variant(headline="this headline has far too many words in it")


def test_headline_trailing_period_is_stripped():
    assert _variant(headline="SAME AI DIFFERENT ANSWER.").headline.endswith("ANSWER")


def test_headline_is_uppercased_and_trimmed():
    assert _variant(headline="  same ai  ").headline == "SAME AI"
