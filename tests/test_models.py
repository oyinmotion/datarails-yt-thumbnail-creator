import pytest
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
    with pytest.raises(ValidationError):
        _plan(variants=[_variant(i, h, t) for i, h, t in MATRIX[:4]])


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
