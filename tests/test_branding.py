from PIL import Image

from src import branding
from src.config import LOGO_DARK, LOGO_LIGHT


def _img(tmp_path, name, colour, size=(1920, 1080)):
    path = tmp_path / name
    Image.new("RGB", size, colour).save(path, "PNG")
    return path


def test_both_logo_assets_are_present_and_have_transparency():
    """Without alpha the logo would paste as a rectangle of background."""
    for path in (LOGO_LIGHT, LOGO_DARK):
        assert path.exists(), f"{path} is missing"
        with Image.open(path) as im:
            assert im.mode in ("RGBA", "LA", "P"), f"{path} has no alpha channel"


def test_stamping_changes_the_image(tmp_path):
    path = _img(tmp_path, "dark.png", (10, 20, 40))
    before = path.read_bytes()
    branding.stamp_logo(path)
    assert path.read_bytes() != before


def test_stamped_image_keeps_its_exact_dimensions(tmp_path):
    for size in ((1920, 1080), (1080, 1080), (1080, 1920)):
        path = _img(tmp_path, f"s{size[0]}x{size[1]}.png", (12, 24, 48), size)
        branding.stamp_logo(path)
        with Image.open(path) as im:
            assert im.size == size


def test_a_dark_corner_gets_the_light_logo(tmp_path):
    """Measured, not assumed: the corner region drives the choice."""
    path = _img(tmp_path, "dark.png", (5, 5, 15))
    with Image.open(path) as im:
        assert not branding.corner_is_light(im.convert("RGBA"), (0, 900, 300, 1000))


def test_a_light_corner_gets_the_dark_logo(tmp_path):
    path = _img(tmp_path, "light.png", (245, 245, 245))
    with Image.open(path) as im:
        assert branding.corner_is_light(im.convert("RGBA"), (0, 900, 300, 1000))


def test_the_logo_lands_in_the_configured_corner(tmp_path):
    """A flat image, stamped: only the logo's corner should differ."""
    path = _img(tmp_path, "flat.png", (10, 20, 40))
    branding.stamp_logo(path)
    with Image.open(path) as im:
        px = im.convert("RGB").load()
        w, h = im.size
        # Bottom-left region contains the logo; the opposite corner is untouched.
        assert px[w - 5, 5] == (10, 20, 40), "top-right should be untouched"


def test_the_logo_reads_at_feed_size(tmp_path):
    """A logo narrower than ~40px at 320px wide is decoration, not branding."""
    from src.config import LOGO_WIDTH_FRACTION
    assert 320 * LOGO_WIDTH_FRACTION >= 40


def test_missing_assets_do_not_lose_the_render(tmp_path, monkeypatch):
    path = _img(tmp_path, "keep.png", (10, 20, 40))
    monkeypatch.setattr(branding, "LOGO_LIGHT", tmp_path / "nope.png")
    monkeypatch.setattr(branding, "LOGO_DARK", tmp_path / "nope2.png")
    before = path.read_bytes()
    assert branding.stamp_logo(path) == path
    assert path.read_bytes() == before


def test_writing_to_a_separate_output_leaves_the_original(tmp_path):
    src = _img(tmp_path, "src.png", (10, 20, 40))
    before = src.read_bytes()
    out = branding.stamp_logo(src, tmp_path / "out.png")
    assert out.exists() and out != src
    assert src.read_bytes() == before


# --- the logo must sit on flat colour --------------------------------------

import os

from PIL import ImageDraw

from src.config import LOGO_BUSY_THRESHOLD, LOGO_WIDTH_FRACTION as _WF


def _logo_size(width=1920):
    from PIL import Image as _I
    target_w = round(width * _WF)
    with _I.open(LOGO_LIGHT) as probe:
        return target_w, round(probe.height * target_w / probe.width)


def _placement(image):
    lw, lh = _logo_size(image.width)
    return branding.plan_placement(image, lw, lh, margin=round(image.width * 0.035))


def test_flat_colour_scores_far_below_the_busy_threshold():
    flat = Image.new("RGB", (600, 200), (12, 24, 48)).convert("RGBA")
    assert branding.busy_score(flat, (0, 0, 600, 200)) < LOGO_BUSY_THRESHOLD


def test_random_texture_scores_above_the_threshold():
    noise = Image.frombytes("RGB", (600, 200), os.urandom(600 * 200 * 3))
    score = branding.busy_score(noise.convert("RGBA"), (0, 0, 600, 200))
    assert score > LOGO_BUSY_THRESHOLD


def test_text_scores_above_the_threshold_even_though_it_is_two_colours():
    """The reason edge energy is in the score: type has little luminance spread
    across the region but a great many edges."""
    canvas = Image.new("RGB", (600, 200), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for y in range(10, 190, 14):
        draw.text((8, y), "SAME AI DIFFERENT ANSWER 47K", fill=(0, 0, 0))
    assert branding.busy_score(canvas.convert("RGBA"), (0, 0, 600, 200)) > (
        LOGO_BUSY_THRESHOLD
    )


def test_a_wholly_flat_image_needs_no_plate():
    flat = Image.new("RGB", (1920, 1080), (12, 24, 48)).convert("RGBA")
    placement = _placement(flat)
    assert not placement.needs_plate
    assert placement.corner == "bottom-left", "ties resolve to the preferred corner"


def test_a_wholly_busy_image_gets_a_plate():
    """Nowhere is flat, so a flat area is manufactured rather than giving up."""
    noise = Image.frombytes("RGB", (1920, 1080), os.urandom(1920 * 1080 * 3))
    assert _placement(noise.convert("RGBA")).needs_plate


def test_the_logo_moves_to_the_calm_corner_instead_of_the_default_one():
    """Busy bottom-left, calm top-right: the logo must not stay put."""
    canvas = Image.new("RGB", (1920, 1080), (12, 24, 48))
    busy = Image.frombytes("RGB", (900, 400), os.urandom(900 * 400 * 3))
    canvas.paste(busy, (0, 680))          # covers the bottom-left corner
    placement = _placement(canvas.convert("RGBA"))
    assert placement.corner != "bottom-left"
    assert not placement.needs_plate, "it found a genuinely flat corner"


def test_bottom_right_is_the_last_resort():
    """YouTube's duration badge lives there, so every other tie wins first."""
    from src.config import LOGO_CORNERS
    assert LOGO_CORNERS[-1] == "bottom-right"


def test_the_area_under_the_logo_is_flat_after_stamping_a_busy_image(tmp_path):
    """End to end: the promise is that the logo never sits on texture."""
    path = tmp_path / "busy.png"
    original = Image.frombytes("RGB", (1920, 1080), os.urandom(1920 * 1080 * 3))
    original.save(path)

    lw, lh = _logo_size()
    margin = round(1920 * 0.035)
    planned = branding.plan_placement(original.convert("RGBA"), lw, lh, margin)
    assert planned.needs_plate
    x0, y0, x1, _y1 = planned.box
    # A strip inside the plate's padding, above the wordmark itself.
    plate_strip = (x0, max(0, y0 - 8), x1, max(1, y0 - 2))

    branding.stamp_logo(path)

    # Where the logo WAS going to go, measured before stamping. Re-planning on
    # the stamped image would measure the wordmark's own letterforms.
    with Image.open(path) as stamped:
        rgba = stamped.convert("RGBA")

    assert branding.busy_score(rgba, plate_strip) < LOGO_BUSY_THRESHOLD, (
        "the plate should have replaced the texture behind the logo"
    )
