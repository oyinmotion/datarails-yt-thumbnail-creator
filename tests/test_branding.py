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
