from pathlib import Path

import pytest
from PIL import Image

from src import typeset
from src.config import RATIOS, TEXT_BOTTOM_RESERVE
from src.models import MATRIX

TREATMENTS = [row[2] for row in MATRIX]


@pytest.mark.parametrize("treatment", TREATMENTS)
@pytest.mark.parametrize("ratio", list(RATIOS))
def test_every_zone_lies_inside_the_canvas_and_above_the_logo_band(treatment, ratio):
    size = RATIOS[ratio][0]
    x0, y0, x1, y1 = typeset.zone_box(treatment, ratio, size)
    w, h = size
    assert 0 <= x0 < x1 <= w
    assert 0 <= y0 < y1 <= h
    assert y1 <= h * (1 - TEXT_BOTTOM_RESERVE), "zone must stay out of the logo band"
    assert (x1 - x0) >= w * 0.3, "a zone narrower than 30% cannot hold a headline"


def test_tall_ratio_uses_its_own_zone():
    wide = typeset.zone_for("face_closeup", "16x9")
    tall = typeset.zone_for("face_closeup", "9x16")
    assert wide != tall
    assert tall.x1 - tall.x0 > 0.8, "on 9:16 the headline spans the width"


def test_square_shares_the_wide_zone():
    assert typeset.zone_for("split_screen", "1x1") == typeset.zone_for("split_screen", "16x9")


def test_alignment_follows_treatment():
    assert typeset.zone_for("text_dominant", "16x9").align == "left"
    assert typeset.zone_for("face_closeup", "16x9").align == "left"
    assert typeset.zone_for("split_screen", "16x9").align == "centre"


def test_unknown_treatment_or_ratio_is_a_programming_error():
    with pytest.raises(KeyError):
        typeset.zone_for("poster", "16x9")
    with pytest.raises(KeyError):
        typeset.zone_for("split_screen", "4x3")


def test_zone_instruction_tells_the_model_where_to_leave_room_and_to_render_no_text():
    text = typeset.zone_instruction("face_closeup", "16x9").lower()
    assert "no text" in text
    assert "left" in text
    assert "%" in text, "the instruction gives the model concrete bounds"
