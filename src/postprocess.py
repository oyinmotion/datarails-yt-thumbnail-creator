"""Turn raw model output into a deliverable YouTube thumbnail file.

Two jobs: land on exactly 1920x1080, and stay under YouTube's 2MB cap.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from .config import FINAL_H, FINAL_W, JPEG_FALLBACK_QUALITY, MAX_BYTES


def finalize(
    image_bytes: bytes,
    out_path: Path,
    final_size: tuple[int, int] = (FINAL_W, FINAL_H),
) -> Path:
    """Downscale to 1920x1080 and write, falling back to JPEG if oversize.

    Returns the path actually written — the suffix may differ from out_path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(io.BytesIO(image_bytes)) as raw:
        image = raw.convert("RGB")
        if image.size != final_size:
            # Lanczos downscale from 2048x1152: exact ratio, no crop, and the
            # supersampling visibly sharpens the headline type.
            image = image.resize(final_size, Image.LANCZOS)

        image.save(out_path, "PNG", optimize=True)
        if out_path.stat().st_size <= MAX_BYTES:
            return out_path

        # Expected on dense full-bleed art. YouTube accepts JPEG.
        jpeg_path = out_path.with_suffix(".jpg")
        image.save(
            jpeg_path, "JPEG",
            quality=JPEG_FALLBACK_QUALITY, optimize=True, progressive=True,
        )

    out_path.unlink()
    return jpeg_path
