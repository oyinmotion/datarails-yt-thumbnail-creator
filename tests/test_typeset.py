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


# --- fitting -----------------------------------------------------------------
from PIL import ImageFont
from src.config import HEADLINE_FONT, TEXT_FLOOR_FRACTION


def _box(w=800, h=500):
    return (0, 0, w, h)


def _cap_height(font):
    bbox = font.getbbox("H")
    return bbox[3] - bbox[1]


def test_a_short_headline_is_set_as_large_as_the_box_allows():
    """Largest size wins, even if that stacks a two-word line — big type is
    the point of a thumbnail. One step bigger must no longer fit."""
    fit = typeset.fit_headline("SAME AI", HEADLINE_FONT, _box(), floor=40)
    assert " ".join(fit.lines) == "SAME AI"
    assert fit.fits
    bigger = ImageFont.truetype(str(HEADLINE_FONT), fit.size + 4)
    lines = typeset._wrap("SAME AI".split(), bigger, 800)
    assert (any(bigger.getlength(l) > 800 for l in lines)
            or typeset._block_height(bigger, len(lines)) > 500
            or len(lines) > 3)


def test_a_five_word_headline_wraps_to_two_or_three_lines():
    fit = typeset.fit_headline("YOUR FORECAST IS ALREADY WRONG", HEADLINE_FONT, _box(), floor=40)
    assert 2 <= len(fit.lines) <= 3
    assert " ".join(fit.lines) == "YOUR FORECAST IS ALREADY WRONG"
    assert fit.fits


def test_lines_are_filled_greedily_and_each_fits_the_width():
    fit = typeset.fit_headline("ONE TWO THREE FOUR FIVE", HEADLINE_FONT, _box(w=600), floor=40)
    for line in fit.lines:
        assert fit.font.getlength(line) <= 600


def test_never_goes_below_the_floor_and_flags_overflow_instead():
    fit = typeset.fit_headline("EXTRAORDINARILY LONG WORDS EVERYWHERE HERE", HEADLINE_FONT,
                               _box(w=300, h=120), floor=90)
    assert fit.size >= 90
    assert not fit.fits


def test_floor_is_seven_percent_of_canvas_height():
    assert typeset.floor_px(1152) == round(1152 * TEXT_FLOOR_FRACTION)


def test_headline_is_uppercased_and_whitespace_collapsed():
    fit = typeset.fit_headline("  same   ai ", HEADLINE_FONT, _box(), floor=40)
    assert " ".join(fit.lines) == "SAME AI"


def test_missing_font_falls_back_then_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(typeset, "HEADLINE_FONT", tmp_path / "missing.ttf")
    font = typeset.load_font(50)          # falls back to SemiBold
    assert "SemiBold" in font.path
    monkeypatch.setattr(typeset, "HEADLINE_FALLBACK_FONT", tmp_path / "also-missing.ttf")
    with pytest.raises(typeset.TypesetError, match="font"):
        typeset.load_font(50)


# --- treatments, scrim, typeset() ---------------------------------------------
from src import branding
from src.config import NAVY, CREAM, WHITE


def _flat_art(ratio="16x9", colour=(30, 60, 120)):
    return Image.new("RGB", RATIOS[ratio][0], colour)


def _busy_art(ratio="16x9"):
    """High-frequency noise everywhere: the model ignored the zone."""
    return Image.effect_noise(RATIOS[ratio][0], 80).convert("RGB")


def test_every_style_has_a_type_treatment():
    from src.models import STYLE_BRIEF
    assert set(typeset.TYPE_TREATMENTS) == set(STYLE_BRIEF)


def test_house_energy_is_white_with_navy_stroke_and_shadow():
    t = typeset.TYPE_TREATMENTS["house_energy"]
    assert t.fill == WHITE and t.stroke == NAVY and t.shadow is not None


def test_dark_cinematic_and_clean_corporate_are_bare():
    for style, fill in (("dark_cinematic", CREAM), ("clean_corporate", NAVY)):
        t = typeset.TYPE_TREATMENTS[style]
        assert t.fill == fill and t.stroke is None and t.shadow is None


def test_typeset_draws_the_headline_inside_the_zone_and_nowhere_else():
    art = _flat_art()
    result = typeset.typeset(art, "SAME AI", "house_energy", "split_screen", "16x9")
    out = result.image
    assert out.size == art.size, "typeset works at the art's native size"
    box = typeset.zone_box("split_screen", "16x9", art.size)
    inside = out.crop(box).convert("L")
    outside = out.crop((0, box[3] + 5, out.width, out.height)).convert("L")
    assert max(inside.get_flattened_data()) > 240, "white type appears in the zone"
    assert max(outside.get_flattened_data()) < 120, "nothing is drawn below the zone"
    assert result.notes == [] and not result.scrimmed and not result.overflowed


def test_text_edges_are_antialiased_not_stepped():
    """The pixelation guard. Type drawn at 2x and composited down has many
    intermediate grey levels along its edges; nearest-neighbour or 1x drawing
    onto a flat field would show only the fill and the background."""
    art = _flat_art(colour=(0, 0, 0))
    out = typeset.typeset(art, "SAME AI", "dark_cinematic", "full_bleed", "16x9").image
    box = typeset.zone_box("full_bleed", "16x9", art.size)
    levels = set(out.crop(box).convert("L").get_flattened_data())
    assert len(levels) > 24, f"only {len(levels)} grey levels: edges look stepped"


def test_flat_graphic_picks_navy_on_light_and_cream_on_dark():
    light = typeset.typeset(_flat_art(colour=(250, 240, 220)), "SAME AI",
                            "flat_graphic", "text_dominant", "16x9").image
    dark = typeset.typeset(_flat_art(colour=(10, 10, 30)), "SAME AI",
                           "flat_graphic", "text_dominant", "16x9").image
    box = typeset.zone_box("text_dominant", "16x9", light.size)
    assert min(light.crop(box).convert("L").get_flattened_data()) < 60, "navy type on a light field"
    assert max(dark.crop(box).convert("L").get_flattened_data()) > 220, "cream type on a dark field"


def test_zone_is_busy_detects_art_painted_into_the_zone():
    box = typeset.zone_box("split_screen", "16x9", RATIOS["16x9"][0])
    assert not typeset.zone_is_busy(_flat_art(), box)
    assert typeset.zone_is_busy(_busy_art(), box)


def test_scrim_is_applied_only_when_asked_and_is_noted():
    art = _busy_art()
    plain = typeset.typeset(art, "SAME AI", "house_energy", "split_screen", "16x9")
    scrimmed = typeset.typeset(art, "SAME AI", "house_energy", "split_screen", "16x9", scrim=True)
    assert not plain.scrimmed and scrimmed.scrimmed
    assert any("scrim" in n for n in scrimmed.notes)
    box = typeset.zone_box("split_screen", "16x9", art.size)
    # A dark scrim under white type lowers the zone's spread noticeably.
    assert (branding.busy_score(scrimmed.image, box)
            < branding.busy_score(plain.image, box))


def test_overflow_is_drawn_and_noted():
    result = typeset.typeset(_flat_art("9x16"), "EXTRAORDINARILY LONG WORDS EVERYWHERE HERE",
                             "house_energy", "product_forward", "9x16")
    assert result.overflowed
    assert any("too long" in n for n in result.notes)


def test_typeset_never_mutates_the_art_it_is_given():
    art = _flat_art()
    before = art.tobytes()
    typeset.typeset(art, "SAME AI", "house_energy", "split_screen", "16x9")
    assert art.tobytes() == before


def test_a_smooth_gradient_is_a_calm_zone_not_a_busy_one():
    """Regression: the first cut used branding.busy_score, whose luminance spread
    flagged every gradient as busy. A calm zone in real art IS a gradient."""
    size = RATIOS["16x9"][0]
    box = typeset.zone_box("split_screen", "16x9", size)
    gradient = Image.linear_gradient("L").resize(size).convert("RGB")
    assert not typeset.zone_is_busy(gradient, box)


REF = Path(__file__).resolve().parent.parent / "refs" / "style" / "Claude-Vs-ClaudeFOS1.png"


@pytest.mark.skipif(not REF.exists(), reason="reference thumbnail not present")
def test_real_faces_and_type_in_the_zone_read_as_busy_and_a_blurred_zone_does_not():
    from PIL import ImageFilter
    size = RATIOS["16x9"][0]
    box = typeset.zone_box("split_screen", "16x9", size)
    ref = Image.open(REF).convert("RGB").resize(size, Image.LANCZOS)
    assert typeset.zone_is_busy(ref, box), "the reference has type and faces in that zone"
    calm = ref.copy()
    calm.paste(ref.crop(box).filter(ImageFilter.GaussianBlur(30)), box[:2])
    assert not typeset.zone_is_busy(calm, box), "blurred to a gradient, it is calm"
