import io
import os

from PIL import Image

from src.config import FINAL_H, FINAL_W, MAX_BYTES
from src.postprocess import finalize


def _png_bytes(w, h, noisy=False):
    """A flat image compresses tiny; true random noise is incompressible, which
    is what a dense full-bleed thumbnail behaves like against the 2MB cap."""
    if noisy:
        im = Image.frombytes("RGB", (w, h), os.urandom(w * h * 3))
    else:
        im = Image.new("RGB", (w, h), (12, 24, 48))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def test_downscales_generated_size_to_exactly_1920x1080(tmp_path):
    out = finalize(_png_bytes(2048, 1152), tmp_path / "v1.png")
    with Image.open(out) as im:
        assert im.size == (FINAL_W, FINAL_H)


def test_flat_image_stays_png(tmp_path):
    out = finalize(_png_bytes(2048, 1152), tmp_path / "v1.png")
    assert out.suffix == ".png"
    assert out.stat().st_size <= MAX_BYTES


def test_dense_image_falls_back_to_jpeg_under_the_cap(tmp_path):
    """A noisy full-bleed PNG at 1920x1080 blows past YouTube's 2MB cap."""
    out = finalize(_png_bytes(2048, 1152, noisy=True), tmp_path / "v2.png")
    assert out.suffix == ".jpg"
    assert out.stat().st_size <= MAX_BYTES
    with Image.open(out) as im:
        assert im.size == (FINAL_W, FINAL_H)


def test_oversize_png_is_removed_when_jpeg_replaces_it(tmp_path):
    png = tmp_path / "v3.png"
    out = finalize(_png_bytes(2048, 1152, noisy=True), png)
    assert out != png
    assert not png.exists()


def test_rgba_input_is_flattened(tmp_path):
    im = Image.new("RGBA", (2048, 1152), (255, 0, 0, 128))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    out = finalize(buf.getvalue(), tmp_path / "v4.png")
    with Image.open(out) as result:
        assert result.mode == "RGB"


def test_already_final_size_input_is_untouched_in_dimensions(tmp_path):
    out = finalize(_png_bytes(FINAL_W, FINAL_H), tmp_path / "v5.png")
    with Image.open(out) as im:
        assert im.size == (FINAL_W, FINAL_H)
