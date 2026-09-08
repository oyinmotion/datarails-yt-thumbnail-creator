"""Turn raw model output into a deliverable YouTube thumbnail file.

Two jobs: land on exactly 1920x1080, and stay under YouTube's 2MB cap.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from .config import FINAL_H, FINAL_W, JPEG_FALLBACK_QUALITY, MAX_BYTES


def finalize_image(
    image: Image.Image,
    out_path: Path,
    final_size: tuple[int, int] = (FINAL_W, FINAL_H),
) -> Path:
    """One Lanczos downscale to `final_size`, PNG, JPEG-92 if over the 2MB cap.

    This is the ONLY place the art is resampled. typeset() works at native
    generation size and resamples only its own text layer.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = image.convert("RGB")
    if rgb.size != final_size:
        # Exact ratio, no crop; the supersampling visibly sharpens the type.
        rgb = rgb.resize(final_size, Image.LANCZOS)
    rgb.save(out_path, "PNG", optimize=True)
    if out_path.stat().st_size <= MAX_BYTES:
        return out_path

    # Expected on dense full-bleed art. YouTube accepts JPEG.
    jpeg_path = out_path.with_suffix(".jpg")
    rgb.save(jpeg_path, "JPEG", quality=JPEG_FALLBACK_QUALITY, optimize=True, progressive=True)
    out_path.unlink()
    return jpeg_path


def finalize(
    image_bytes: bytes,
    out_path: Path,
    final_size: tuple[int, int] = (FINAL_W, FINAL_H),
) -> Path:
    """Bytes-in wrapper kept for existing callers and tests.

    Returns the path actually written — the suffix may differ from out_path.
    """
    with Image.open(io.BytesIO(image_bytes)) as raw:
        return finalize_image(raw, out_path, final_size)
