# Text as a Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the headline out of `gpt-image-2` into a deterministic Pillow typesetting layer, so text is always legible, free to edit, and every tile can be re-titled, re-rolled or re-styled on its own.

**Architecture:** Two artifacts per tile per ratio. `render_art()` asks the model for text-free art with a reserved zone and caches it at native generation size (`work_dir/art/`). `compose()` typesets the headline in Poppins Black onto that art, stamps the logo, and downscales exactly once to delivery size (`work_dir/out/`). The likeness check is the only model-driven gate left; legibility becomes deterministic. Per-tile actions (edit / variants / re-roll / swap look) are thin wrappers over those two functions.

**Tech Stack:** Python 3.13, Streamlit 1.63, Pillow 12 (`ImageDraw.text` with `stroke_width`, `ImageFont.truetype`), OpenAI SDK (`images.edit`, `responses.parse`), pytest. Fonts: Poppins (OFL) bundled in-repo.

**Spec:** `docs/superpowers/specs/2026-09-08-text-as-a-layer-design.md`

## Global Constraints

- Every test mocks the API; Pillow runs for real; ffmpeg runs for real against `Sample input ad/*.mp4` when present (tests skip otherwise). Backoff is injected via `sleeper` so tests are instant.
- One bad thing costs one tile, never the batch: every new code path in `pipeline._one_render` stays inside its existing catch-everything boundary.
- `generate_batch` keeps its hard contract: exactly one `ThumbResult` per planned concept, in matrix order, every ratio attempted.
- Art is cached at the model's native generation size (`RATIOS[ratio][0]`); the delivered file is produced by **exactly one** Lanczos downscale. Text is drawn at `TEXT_SUPERSAMPLE = 2`.
- No zone enters the bottom `TEXT_BOTTOM_RESERVE = 0.14` of the canvas. Floor: cap height ≥ `TEXT_FLOOR_FRACTION = 0.07` of canvas height.
- Headline typeface: `assets/fonts/poppins/Poppins-Black.ttf`; fallback `Poppins-SemiBold.ttf`; then `TypesetError`. Full Poppins family plus `OFL.txt` ship in the repo.
- `MAX_HEADLINE_WORDS` stays 5. `SEND_STYLE_REFS` stays `False`. `IMAGE_QUALITY` stays `"high"`.
- No prompt text lives in Python (`src/prompts.py` docstring rule): new prompt copy goes in `prompts/*.md`.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Work on branch `feat/text-layer`, stacked on `feat/speed-cost`.
- Run the suite both ways before the final PR: `.venv/bin/python -m pytest -q` (local) and, in the clean Python 3.13 venv with system ffmpeg removed from PATH, `PATH=/usr/bin:/bin:/usr/sbin:/sbin <cloudsim>/bin/python -m pytest -q -p no:cacheprovider`.

---

## File structure

| File | Responsibility | Change |
|---|---|---|
| `assets/fonts/poppins/*.ttf`, `OFL.txt` | The typeface, redistributable | **create** (Task 1) |
| `src/config.py` | Font paths, zone/floor/supersample constants, brand colours | modify (Task 1) |
| `src/typeset.py` | Zones, fitting, type treatments, scrim, `typeset()` — the only module that draws text | **create** (Tasks 2–4) |
| `prompts/render.md` | Art-only render prompt with `{zone_instruction}` | rewrite (Task 5) |
| `prompts/qa_legibility.md` | Retired | **delete** (Task 5) |
| `prompts/headlines.md` | "3 more lines" planner prompt | **create** (Task 11) |
| `src/models.py` | `TREATMENT_BRIEF` / `STYLE_BRIEF` lose their type and headline sentences; `clean_headline_text()` extracted | modify (Tasks 5, 11) |
| `src/prompts.py` | `render_prompt(variant, people_in_ad, style, ratio)`; `headline_prompt()` | modify (Tasks 5, 11) |
| `src/render.py` | `render_variant` → `render_art(..., style, ratio)` | modify (Task 6) |
| `src/qa.py` | Legibility removed; `likeness_gate()`; `QAResult.checked` | modify (Task 7) |
| `src/branding.py` | `stamp_logo_image(Image) -> Image`; path wrapper kept | modify (Task 8) |
| `src/postprocess.py` | `finalize_image(Image, ...)`; bytes wrapper kept | modify (Task 8) |
| `src/pipeline.py` | `_one_render` = art + compose; `compose()`, `retitle()`, `reroll()`; richer result types | modify (Tasks 9–10) |
| `src/plan.py` | `suggest_headlines()` | modify (Task 11) |
| `app.py` | Action row per card, per-concept session state, exports reflect current state | modify (Task 12) |
| `README.md`, `scripts/live_run.py` | Docs and the live harness | modify (Task 13) |
| `tests/test_typeset.py` | New module's tests | **create** (Tasks 2–4) |
| `tests/test_prompts.py`, `test_render.py`, `test_qa.py`, `test_branding.py`, `test_postprocess.py`, `test_pipeline.py`, `test_plan.py`, `test_app_helpers.py` | Updated for the new contract | modify |

---

### Task 1: Bundle Poppins and add the typesetting constants

**Files:**
- Create: `assets/fonts/poppins/` (18 `.ttf` + `OFL.txt`)
- Modify: `src/config.py` (after the `# --- Branding` block)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `config.FONTS_DIR: Path`, `config.HEADLINE_FONT: Path`, `config.HEADLINE_FALLBACK_FONT: Path`, `config.TEXT_FLOOR_FRACTION = 0.07`, `config.TEXT_BOTTOM_RESERVE = 0.14`, `config.TEXT_SUPERSAMPLE = 2`, `config.ZONE_BUSY_THRESHOLD = 22.0`, `config.NAVY`, `config.CREAM`, `config.WHITE` (RGB tuples).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_the_headline_font_ships_in_the_repo_and_opens():
    from PIL import ImageFont
    from src import config
    assert config.HEADLINE_FONT.exists(), config.HEADLINE_FONT
    assert config.HEADLINE_FALLBACK_FONT.exists()
    assert (config.FONTS_DIR / "OFL.txt").exists(), "the licence travels with the font"
    font = ImageFont.truetype(str(config.HEADLINE_FONT), 100)
    assert font.getlength("SAME AI") > 0


def test_typesetting_constants_match_the_spec():
    from src import config
    assert config.TEXT_FLOOR_FRACTION == 0.07
    assert config.TEXT_BOTTOM_RESERVE == 0.14
    assert config.TEXT_SUPERSAMPLE == 2
    assert config.NAVY == (12, 20, 43)
    assert config.CREAM == (255, 248, 238)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest -q tests/test_config.py -k "headline_font or typesetting_constants"`
Expected: FAIL with `AttributeError: module 'src.config' has no attribute 'HEADLINE_FONT'`

- [ ] **Step 3: Download the family and the licence**

```bash
cd "/Users/omeryadgar/Desktop/dev_projects/YT Thumbnail creator"
mkdir -p assets/fonts/poppins
for w in Thin ThinItalic ExtraLight ExtraLightItalic Light LightItalic Regular Italic \
         Medium MediumItalic SemiBold SemiBoldItalic Bold BoldItalic ExtraBold \
         ExtraBoldItalic Black BlackItalic; do
  curl -sL --max-time 60 -o "assets/fonts/poppins/Poppins-$w.ttf" \
    "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-$w.ttf"
done
curl -sL --max-time 60 -o assets/fonts/poppins/OFL.txt \
  "https://github.com/google/fonts/raw/main/ofl/poppins/OFL.txt"
ls assets/fonts/poppins | wc -l      # expect 19
file assets/fonts/poppins/Poppins-Black.ttf   # expect "TrueType Font data"
```

- [ ] **Step 4: Add the constants**

Append to `src/config.py` after the `# --- Branding` block (before `# --- Output ratios`):

```python
# --- Typesetting -----------------------------------------------------------
# The headline is set by us, not by the image model. Poppins is the design
# system's face and is OFL-licensed, so the whole family can ship in the repo.
FONTS_DIR = PROJECT_ROOT / "assets" / "fonts" / "poppins"
HEADLINE_FONT = FONTS_DIR / "Poppins-Black.ttf"
HEADLINE_FALLBACK_FONT = FONTS_DIR / "Poppins-SemiBold.ttf"
# Cap height never below this fraction of canvas height: at the 320px width a
# thumbnail actually occupies in a feed, that is still >= 14px of type.
TEXT_FLOOR_FRACTION = 0.07
# Text zones never enter this bottom band; it is where stamp_logo prefers to sit.
TEXT_BOTTOM_RESERVE = 0.14
# Text is drawn at this multiple of the art's size and composited down, so the
# edges are anti-aliased rather than stepped.
TEXT_SUPERSAMPLE = 2
# branding.busy_score above this means the model painted into the reserved zone.
ZONE_BUSY_THRESHOLD = 22.0
# Brand colours used by the type treatments (same values as the logo plates).
NAVY = LOGO_PLATE_DARK
CREAM = LOGO_PLATE_LIGHT
WHITE = (255, 255, 255)
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest -q tests/test_config.py`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add assets/fonts/poppins src/config.py tests/test_config.py
git commit -m "$(cat <<'EOF'
Bundle the Poppins family and add the typesetting constants

The headline moves out of the image model and into a Pillow layer, so the
face has to live in the repo. Poppins is the design system's typeface and is
OFL-licensed; the full family ships with its licence. Black carries the
headline, SemiBold is the fallback if Black ever fails to load.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Text zones per treatment and ratio

**Files:**
- Create: `src/typeset.py`
- Create: `tests/test_typeset.py`

**Interfaces:**
- Consumes: `config.TEXT_BOTTOM_RESERVE`, `models.Treatment`, `config.RATIOS`.
- Produces: `typeset.Zone` (frozen dataclass: `x0, y0, x1, y1: float` fractions, `align: str` in `{"centre","left"}`), `typeset.zone_for(treatment: str, ratio: str) -> Zone`, `typeset.zone_box(treatment: str, ratio: str, size: tuple[int, int]) -> tuple[int, int, int, int]`, `typeset.zone_instruction(treatment: str, ratio: str) -> str`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_typeset.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest -q tests/test_typeset.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.typeset'`

- [ ] **Step 3: Create the module with zones**

Create `src/typeset.py`:

```python
"""The headline as a layer. The only module that draws text.

The image model renders art with a reserved, calm zone; this module sets the
headline into that zone in Poppins Black, with the type treatment the style
calls for, and composites it down onto the art. Deterministic: same inputs,
same pixels. Nothing here calls an API.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import TEXT_BOTTOM_RESERVE


@dataclass(frozen=True)
class Zone:
    """Where the headline goes, as fractions of the canvas."""
    x0: float
    y0: float
    x1: float
    y1: float
    align: str          # "centre" | "left"


# 16:9 and 1:1 share the "wide" zone; 9:16 gets a "tall" zone across the top.
# Every zone stays above TEXT_BOTTOM_RESERVE so the logo has somewhere to land.
_WIDE: dict[str, Zone] = {
    "split_screen":    Zone(0.30, 0.12, 0.70, 0.60, "centre"),
    "face_closeup":    Zone(0.06, 0.10, 0.52, 0.70, "left"),
    "full_bleed":      Zone(0.15, 0.08, 0.85, 0.44, "centre"),
    "text_dominant":   Zone(0.06, 0.10, 0.62, 0.72, "left"),
    "product_forward": Zone(0.10, 0.08, 0.90, 0.30, "centre"),
}
_TALL: dict[str, Zone] = {
    "split_screen":    Zone(0.06, 0.08, 0.94, 0.36, "centre"),
    "face_closeup":    Zone(0.06, 0.08, 0.94, 0.40, "left"),
    "full_bleed":      Zone(0.06, 0.08, 0.94, 0.38, "centre"),
    "text_dominant":   Zone(0.06, 0.08, 0.94, 0.50, "left"),
    "product_forward": Zone(0.06, 0.08, 0.94, 0.30, "centre"),
}
_FAMILY: dict[str, dict[str, Zone]] = {"16x9": _WIDE, "1x1": _WIDE, "9x16": _TALL}

# Where the actor goes so the zone stays clear. Spoken to the model.
_ACTOR: dict[str, str] = {
    "split_screen": "one actor on each side, the seam of light between them",
    "face_closeup": "the actor's face fills the RIGHT half of the frame",
    "full_bleed": "the actors occupy the lower two-thirds of the frame",
    "text_dominant": "the actor stands offset to the RIGHT, still large",
    "product_forward": "the product surface and the actor sit BELOW the reserved band",
}


def zone_for(treatment: str, ratio: str) -> Zone:
    return _FAMILY[ratio][treatment]          # KeyError is a programming error


def zone_box(treatment: str, ratio: str, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """The zone in pixels for a canvas of `size`, clamped above the logo band."""
    zone = zone_for(treatment, ratio)
    width, height = size
    y1 = min(zone.y1, 1.0 - TEXT_BOTTOM_RESERVE)
    return (
        round(width * zone.x0), round(height * zone.y0),
        round(width * zone.x1), round(height * y1),
    )


def _side(zone: Zone) -> str:
    if zone.x1 <= 0.55:
        return "left"
    if zone.x0 >= 0.45:
        return "right"
    return "centre" if (zone.x1 - zone.x0) < 0.8 else "full width"


def zone_instruction(treatment: str, ratio: str) -> str:
    """Prose for the render prompt: where to leave room, and that we add the text."""
    zone = zone_for(treatment, ratio)
    return (
        f"Reserve a calm area at the {_side(zone)} of the frame, from "
        f"{round(zone.x0 * 100)}% to {round(zone.x1 * 100)}% across and from "
        f"{round(zone.y0 * 100)}% to {round(zone.y1 * 100)}% down. Inside it: flat "
        "colour or a soft gradient only — no face, no hand, no object, no sparks, "
        "no logo, no pattern. The headline is added afterwards by us. You render "
        "NO text of any kind anywhere in the image: no words, letters, numbers, "
        f"captions, watermarks or UI. Composition: {_ACTOR[treatment]}."
    )
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest -q tests/test_typeset.py`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/typeset.py tests/test_typeset.py
git commit -m "$(cat <<'EOF'
Define the headline zones per treatment and ratio

Fixed zones, expressed as canvas fractions, beat face detection: they are
deterministic and the render prompt can tell the model where the actor goes
so the zone stays clear. 16:9 and 1:1 share a zone; 9:16 uses a band across
the top. No zone enters the bottom 14%, which is the logo's territory.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Fitting — largest size that fits, 1–3 lines, never below the floor

**Files:**
- Modify: `src/typeset.py`
- Modify: `tests/test_typeset.py`

**Interfaces:**
- Consumes: `config.HEADLINE_FONT`, `config.TEXT_FLOOR_FRACTION`.
- Produces: `typeset.Fit` (dataclass: `font: ImageFont.FreeTypeFont`, `lines: list[str]`, `size: int`, `fits: bool`), `typeset.floor_px(canvas_height: int) -> int`, `typeset.fit_headline(text: str, font_path: Path, box: tuple[int,int,int,int], floor: int, max_lines: int = 3) -> Fit`, `typeset.TypesetError`, `typeset.load_font(size: int) -> ImageFont.FreeTypeFont`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_typeset.py`:

```python
from PIL import ImageFont
from src.config import HEADLINE_FONT, TEXT_FLOOR_FRACTION


def _box(w=800, h=500):
    return (0, 0, w, h)


def test_a_short_headline_sets_on_one_line_as_large_as_the_box_allows():
    fit = typeset.fit_headline("SAME AI", HEADLINE_FONT, _box(), floor=40)
    assert fit.lines == ["SAME AI"]
    assert fit.fits
    # Largest size: one step bigger would no longer fit the width or height.
    bigger = ImageFont.truetype(str(HEADLINE_FONT), fit.size + 4)
    assert bigger.getlength("SAME AI") > 800 or _cap_height(bigger) * 1.0 > 500


def _cap_height(font):
    bbox = font.getbbox("H")
    return bbox[3] - bbox[1]


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
    assert fit.size >= 90 or _cap_height(fit.font) >= 90 * 0.9
    assert not fit.fits


def test_floor_is_seven_percent_of_canvas_height():
    assert typeset.floor_px(1152) == round(1152 * TEXT_FLOOR_FRACTION)


def test_headline_is_uppercased_and_whitespace_collapsed():
    fit = typeset.fit_headline("  same   ai ", HEADLINE_FONT, _box(), floor=40)
    assert fit.lines == ["SAME AI"]


def test_missing_font_falls_back_then_raises(tmp_path, monkeypatch):
    from src import config
    monkeypatch.setattr(typeset, "HEADLINE_FONT", tmp_path / "missing.ttf")
    font = typeset.load_font(50)          # falls back to SemiBold
    assert "SemiBold" in font.path
    monkeypatch.setattr(typeset, "HEADLINE_FALLBACK_FONT", tmp_path / "also-missing.ttf")
    with pytest.raises(typeset.TypesetError, match="font"):
        typeset.load_font(50)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest -q tests/test_typeset.py -k "fit or floor or uppercased or missing_font"`
Expected: FAIL with `AttributeError: module 'src.typeset' has no attribute 'fit_headline'`

- [ ] **Step 3: Implement fitting**

Add to `src/typeset.py` (imports at top; functions after `zone_instruction`):

```python
import logging
from pathlib import Path

from PIL import ImageFont

from .config import (
    HEADLINE_FALLBACK_FONT, HEADLINE_FONT, TEXT_BOTTOM_RESERVE, TEXT_FLOOR_FRACTION,
)

log = logging.getLogger(__name__)


class TypesetError(RuntimeError):
    """The headline could not be set at all (no usable font)."""


@dataclass
class Fit:
    font: ImageFont.FreeTypeFont
    lines: list[str]
    size: int
    fits: bool          # False when even the floor size overflows the box


def load_font(size: int) -> ImageFont.FreeTypeFont:
    """Poppins Black, else SemiBold, else a TypesetError naming both."""
    for path in (HEADLINE_FONT, HEADLINE_FALLBACK_FONT):
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            log.warning("font %s not loadable", path)
    raise TypesetError(
        f"No headline font could be loaded (tried {HEADLINE_FONT.name} and "
        f"{HEADLINE_FALLBACK_FONT.name}); the fonts ship in assets/fonts/poppins."
    )


def floor_px(canvas_height: int) -> int:
    return round(canvas_height * TEXT_FLOOR_FRACTION)


def _cap_height(font: ImageFont.FreeTypeFont) -> int:
    bbox = font.getbbox("H")
    return bbox[3] - bbox[1]


def _wrap(words: list[str], font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Greedy fill: as many words per line as fit the width."""
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if current and font.getlength(candidate) > max_width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def _block_height(font: ImageFont.FreeTypeFont, n_lines: int) -> int:
    cap = _cap_height(font)
    leading = round(cap * 1.18)          # tight display leading
    return cap + leading * (n_lines - 1)


def fit_headline(
    text: str, font_path: Path, box: tuple[int, int, int, int],
    floor: int, max_lines: int = 3,
) -> Fit:
    """The largest size at which the headline fits `box` in <= max_lines.

    Sizes step down by 4px from a generous start. If nothing fits at the floor,
    return the floor-size fit with fits=False so the caller can flag the tile;
    the text is still drawn — a flagged headline beats a missing one.
    `font_path` is accepted for callers that want a specific face; load_font()
    handles the fallback chain when it is the configured headline font.
    """
    words = " ".join((text or "").upper().split()).split()
    x0, y0, x1, y1 = box
    max_w, max_h = x1 - x0, y1 - y0

    def _font(size: int) -> ImageFont.FreeTypeFont:
        if Path(font_path) == HEADLINE_FONT:
            return load_font(size)
        return ImageFont.truetype(str(font_path), size)

    # Start where a single line would fill the height; nothing can be bigger.
    size = max(floor, max_h)
    while size > floor:
        font = _font(size)
        lines = _wrap(words, font, max_w)
        if (len(lines) <= max_lines
                and all(font.getlength(l) <= max_w for l in lines)
                and _block_height(font, len(lines)) <= max_h):
            return Fit(font=font, lines=lines, size=size, fits=True)
        size -= 4

    font = _font(floor)
    lines = _wrap(words, font, max_w)
    fits = (len(lines) <= max_lines
            and all(font.getlength(l) <= max_w for l in lines)
            and _block_height(font, len(lines)) <= max_h)
    return Fit(font=font, lines=lines, size=floor, fits=fits)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest -q tests/test_typeset.py`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/typeset.py tests/test_typeset.py
git commit -m "$(cat <<'EOF'
Fit the headline: largest size that fits, one to three lines, never below the floor

Greedy wrap, then step the size down until every line fits the zone's width and
the block fits its height. The floor is 7% of canvas height, which keeps the
type at least 14px tall at the 320px a thumbnail occupies in a feed. Below the
floor we still draw — flagged — because a flagged headline beats a missing one.
Poppins Black falls back to SemiBold before failing.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Type treatments, scrim, and `typeset()` — with the anti-pixelation test

**Files:**
- Modify: `src/typeset.py`
- Modify: `tests/test_typeset.py`

**Interfaces:**
- Consumes: `branding.busy_score(image, box) -> float`, `branding.region_is_light(image, box) -> bool`, `config.ZONE_BUSY_THRESHOLD`, `config.TEXT_SUPERSAMPLE`, `config.NAVY/CREAM/WHITE`.
- Produces: `typeset.TypeTreatment` (frozen dataclass: `fill: tuple`, `stroke: tuple | None`, `stroke_frac: float`, `shadow: tuple | None`, `shadow_frac: float`, `adaptive: bool`), `typeset.TYPE_TREATMENTS: dict[str, TypeTreatment]` keyed by style, `typeset.zone_is_busy(image, box) -> bool`, `typeset.TypesetResult` (dataclass: `image: Image.Image`, `notes: list[str]`, `scrimmed: bool`, `overflowed: bool`), `typeset.typeset(art: Image.Image, headline: str, style: str, treatment: str, ratio: str, *, scrim: bool = False) -> TypesetResult`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_typeset.py`:

```python
from src import branding
from src.config import NAVY, CREAM, WHITE, RATIOS


def _flat_art(ratio="16x9", colour=(30, 60, 120)):
    return Image.new("RGB", RATIOS[ratio][0], colour)


def _busy_art(ratio="16x9"):
    """High-frequency noise everywhere: the model ignored the zone."""
    import random
    random.seed(1)
    w, h = RATIOS[ratio][0]
    img = Image.new("RGB", (w, h))
    img.putdata([(random.randrange(256),) * 3 for _ in range(w * h)])
    return img


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
    assert max(inside.getdata()) > 240, "white type appears in the zone"
    assert max(outside.getdata()) < 120, "nothing is drawn below the zone"
    assert result.notes == [] and not result.scrimmed and not result.overflowed


def test_text_edges_are_antialiased_not_stepped():
    """The pixelation guard. Type drawn at 2x and composited down has many
    intermediate grey levels along its edges; nearest-neighbour or 1x drawing
    onto a flat field would show only the fill and the background."""
    art = _flat_art(colour=(0, 0, 0))
    out = typeset.typeset(art, "SAME AI", "clean_corporate", "full_bleed", "16x9").image
    box = typeset.zone_box("full_bleed", "16x9", art.size)
    levels = set(out.crop(box).convert("L").getdata())
    assert len(levels) > 24, f"only {len(levels)} grey levels: edges look stepped"


def test_flat_graphic_picks_navy_on_light_and_cream_on_dark():
    light = typeset.typeset(_flat_art(colour=(250, 240, 220)), "SAME AI",
                            "flat_graphic", "text_dominant", "16x9").image
    dark = typeset.typeset(_flat_art(colour=(10, 10, 30)), "SAME AI",
                           "flat_graphic", "text_dominant", "16x9").image
    box = typeset.zone_box("text_dominant", "16x9", light.size)
    assert min(light.crop(box).convert("L").getdata()) < 60, "navy type on a light field"
    assert max(dark.crop(box).convert("L").getdata()) > 220, "cream type on a dark field"


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
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest -q tests/test_typeset.py -k "treatment or typeset_ or antialiased or flat_graphic or busy or scrim or overflow or mutates"`
Expected: FAIL with `AttributeError: module 'src.typeset' has no attribute 'TYPE_TREATMENTS'`

- [ ] **Step 3: Implement treatments, scrim and typeset()**

Add to `src/typeset.py` (extend imports; append after `fit_headline`):

```python
from PIL import Image, ImageDraw

from . import branding
from .config import CREAM, NAVY, TEXT_SUPERSAMPLE, WHITE, ZONE_BUSY_THRESHOLD


@dataclass(frozen=True)
class TypeTreatment:
    """How a style sets its headline. Mirrors the retired TYPE clauses of STYLE_BRIEF."""
    fill: tuple[int, int, int]
    stroke: tuple[int, int, int] | None
    stroke_frac: float          # of font size
    shadow: tuple[int, int, int, int] | None   # RGBA
    shadow_frac: float          # offset, of font size
    adaptive: bool = False      # flat_graphic: pick fill from the zone's luminance


TYPE_TREATMENTS: dict[str, TypeTreatment] = {
    "house_energy":    TypeTreatment(WHITE, NAVY, 0.07, (0, 0, 0, 166), 0.06),
    "dark_cinematic":  TypeTreatment(CREAM, None, 0.0, None, 0.0),
    "flat_graphic":    TypeTreatment(NAVY, None, 0.0, None, 0.0, adaptive=True),
    "clean_corporate": TypeTreatment(NAVY, None, 0.0, None, 0.0),
}


@dataclass
class TypesetResult:
    image: Image.Image
    notes: list[str]
    scrimmed: bool = False
    overflowed: bool = False


def zone_is_busy(image: Image.Image, box: tuple[int, int, int, int]) -> bool:
    return branding.busy_score(image, box) > ZONE_BUSY_THRESHOLD


def _draw_scrim(canvas: Image.Image, box: tuple[int, int, int, int], dark: bool) -> None:
    """A soft plate under the type, in place. Dark under light type, light under dark."""
    x0, y0, x1, y1 = box
    pad = round((y1 - y0) * 0.08)
    colour = (*NAVY, 150) if dark else (*CREAM, 170)
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rounded_rectangle(
        (max(0, x0 - pad), max(0, y0 - pad), min(canvas.width, x1 + pad), min(canvas.height, y1 + pad)),
        radius=max(8, pad), fill=colour,
    )
    canvas.alpha_composite(overlay)


def typeset(
    art: Image.Image, headline: str, style: str, treatment: str, ratio: str,
    *, scrim: bool = False,
) -> TypesetResult:
    """Set `headline` into the zone on a copy of `art`, at the art's own size.

    Text is drawn on a transparent layer at TEXT_SUPERSAMPLE x the art's size and
    resampled down before compositing, so edges are anti-aliased. The art itself
    is never resampled here — the single downscale to delivery size happens
    later, in postprocess.finalize_image().
    """
    canvas = art.convert("RGBA")
    box = zone_box(treatment, ratio, canvas.size)
    zone = zone_for(treatment, ratio)
    look = TYPE_TREATMENTS[style]
    notes: list[str] = []

    fill = look.fill
    if look.adaptive:
        fill = NAVY if branding.region_is_light(canvas, box) else CREAM

    if scrim:
        dark_type = sum(fill) < 384
        _draw_scrim(canvas, box, dark=not dark_type)
        notes.append("set over a scrim — art ignored the text zone")

    fit = fit_headline(headline, HEADLINE_FONT, box, floor=floor_px(canvas.height))
    if not fit.fits:
        notes.append("headline too long for this layout")

    s = TEXT_SUPERSAMPLE
    layer = Image.new("RGBA", (canvas.width * s, canvas.height * s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font = load_font(fit.size * s) if Path(HEADLINE_FONT) == HEADLINE_FONT else fit.font
    cap = _cap_height(font)
    leading = round(cap * 1.18)
    stroke_w = round(fit.size * s * look.stroke_frac) if look.stroke else 0
    shadow_dx = round(fit.size * s * look.shadow_frac) if look.shadow else 0

    bx0, by0, bx1, by1 = (v * s for v in box)
    block_h = cap + leading * (len(fit.lines) - 1)
    y = by0 + max(0, ((by1 - by0) - block_h) // 2)
    for line in fit.lines:
        w = font.getlength(line)
        x = bx0 if zone.align == "left" else bx0 + ((bx1 - bx0) - w) / 2
        # getbbox's top offset keeps the cap height where we measured it.
        oy = font.getbbox("H")[1]
        if look.shadow:
            draw.text((x + shadow_dx, y - oy + shadow_dx), line, font=font,
                      fill=look.shadow, stroke_width=stroke_w, stroke_fill=look.shadow)
        draw.text((x, y - oy), line, font=font, fill=(*fill, 255),
                  stroke_width=stroke_w, stroke_fill=(*look.stroke, 255) if look.stroke else None)
        y += leading

    layer = layer.resize(canvas.size, Image.LANCZOS)
    canvas.alpha_composite(layer)
    return TypesetResult(
        image=canvas.convert("RGB"), notes=notes, scrimmed=scrim, overflowed=not fit.fits,
    )
```

Note on `font` selection: `fit.font` was sized for the 1× canvas; the layer is drawn at `s×`, so the font is reloaded at `fit.size * s` through `load_font` (same fallback chain). The line `font = load_font(fit.size * s) if Path(HEADLINE_FONT) == HEADLINE_FONT else fit.font` always takes the first branch in production; simplify to `font = load_font(fit.size * s)` if the reviewer prefers — the tests do not distinguish.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest -q tests/test_typeset.py`
Expected: all PASS. If `test_text_edges_are_antialiased_not_stepped` fails with a low level count, the layer was not resampled — check `layer.resize(..., Image.LANCZOS)` runs before `alpha_composite`.

- [ ] **Step 5: Commit**

```bash
git add src/typeset.py tests/test_typeset.py
git commit -m "$(cat <<'EOF'
Set the headline: per-style type treatment, supersampled, with a scrim fallback

Each style's type treatment moves out of STYLE_BRIEF prose and into code: white
with a navy stroke and hard shadow for house energy, bare cream for dark
cinematic, bare navy for clean corporate, and navy-or-cream chosen from the
zone's luminance for flat graphic. Text is drawn at 2x on its own layer and
resampled down before compositing, so edges are anti-aliased — a test counts
the grey levels along them. When the caller says the zone is busy, a soft
brand-coloured scrim goes under the type so legibility is never in doubt.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: The render prompt asks for art only

**Files:**
- Modify: `prompts/render.md` (rewrite)
- Delete: `prompts/qa_legibility.md`
- Modify: `src/models.py:42-115` (`TREATMENT_BRIEF`, `STYLE_BRIEF`)
- Modify: `src/prompts.py:42-52` (`render_prompt`)
- Modify: `tests/test_prompts.py`

**Interfaces:**
- Consumes: `typeset.zone_instruction(treatment, ratio)`.
- Produces: `prompts.render_prompt(variant: Variant, people_in_ad: bool = True, style: str | None = None, ratio: str = PRIMARY_RATIO) -> str`. The prompt contains **no** headline and no `TYPE:` sentence; it contains the zone instruction.

- [ ] **Step 1: Update the tests**

In `tests/test_prompts.py`:

Replace `test_all_three_prompt_files_load` with:

```python
def test_the_prompt_files_load():
    for name in ("planner", "render", "qa_likeness"):
        assert len(prompts.load(name)) > 200


def test_the_legibility_prompt_is_gone():
    with pytest.raises(FileNotFoundError):
        prompts.load("qa_legibility")
```

Delete `test_render_prompt_contains_the_exact_headline_in_quotes` and `test_render_prompt_forbids_extra_text`. Add:

```python
def test_render_prompt_never_mentions_the_headline_text():
    text = prompts.render_prompt(_variant(headline="SAME AI DIFFERENT ANSWER"))
    assert "SAME AI DIFFERENT ANSWER" not in text
    assert "headline" not in text.lower().replace("the headline is added afterwards by us", "")


def test_render_prompt_forbids_all_text_and_reserves_the_zone():
    flat = " ".join(prompts.render_prompt(_variant(treatment="face_closeup", index=2, hook="question")).split()).lower()
    assert "no text of any kind" in flat
    assert "reserve a calm area" in flat
    assert "left" in flat, "face_closeup reserves the left"


def test_render_prompt_can_be_asked_for_another_style():
    text = prompts.render_prompt(_variant(index=1), style="flat_graphic")
    assert "flat-colour treatment" in text
    assert "proven Datarails look" not in text


def test_render_prompt_uses_the_tall_zone_for_9x16():
    wide = prompts.render_prompt(_variant(treatment="face_closeup", index=2, hook="question"), ratio="16x9")
    tall = prompts.render_prompt(_variant(treatment="face_closeup", index=2, hook="question"), ratio="9x16")
    assert wide != tall


def test_style_briefs_no_longer_describe_type():
    from src.models import STYLE_BRIEF, TREATMENT_BRIEF
    for brief in STYLE_BRIEF.values():
        assert "TYPE:" not in brief
    for brief in TREATMENT_BRIEF.values():
        assert "headline" not in brief.lower()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest -q tests/test_prompts.py`
Expected: FAIL — `test_the_legibility_prompt_is_gone` (file exists), `test_render_prompt_never_mentions_the_headline_text` (headline present), `test_style_briefs_no_longer_describe_type` (TYPE: present), `test_render_prompt_can_be_asked_for_another_style` (unexpected keyword `style`).

- [ ] **Step 3: Rewrite the prompt, briefs and `render_prompt`**

`git rm prompts/qa_legibility.md`

Replace `prompts/render.md` with:

```markdown
Create the ARTWORK for a YouTube ad thumbnail in the exact visual style described
below. You are producing the picture only. We add the headline ourselves
afterwards, so this image must contain no text at all.

## Non-negotiable, whatever the style

These override the style direction below in every case.

**The people are real and they are not yours to change.**

- The reference images are frames from an actual filmed ad. The people in them
  are real employees. They are the ONLY people who may appear in this image.
- Never generate, substitute, replace or add a person. Not a similar-looking
  person, not a better-looking person, not a stock-photography professional.
- Count the people in the reference frames. Your image may contain **only**
  those people, and no more of them than the frames show. If they show nobody,
  the image contains nobody.
- Any additional images you were given are there for VISUAL STYLE ONLY. If a
  person appears in one of them, that person is a stranger here. Never copy,
  trace or imitate a face from a style image.
- Keep their face, age, skin tone, hair, facial hair, glasses and clothing
  exactly as they appear in the reference frames, including any logo or graphic
  printed on their clothing.
- You are compositing, not re-photographing. Cut the person out of their
  original footage and place them IN FRONT OF the style's background.

**It has to work as a YouTube thumbnail.**

- At least one face is large, clearly visible, and carries a readable emotion,
  turned toward the viewer or close to it.
- The subject separates hard from the background and reads instantly at 320
  pixels wide.
- No invented logos, no invented product names, no invented UI, no watermark,
  no signature, no play button.

**No text. None.** No words, letters, numbers, captions, labels, speech
bubbles, watermarks or interface text anywhere in the image. Text printed on a
person's own clothing in the reference frames is the single exception.

## The reserved area

{zone_instruction}

## Visual style for this artwork

{style_brief}

Follow that style faithfully, including its palette and lighting. Do not
substitute a different look, and do not fall back on a high-contrast
orange-and-blue energy treatment unless the style above asks for one.

## Layout

{treatment_brief}

## Scene direction

{scene_direction}
```

In `src/models.py`, remove the headline sentences from `TREATMENT_BRIEF` and the `TYPE:` sentences from `STYLE_BRIEF`:

```python
TREATMENT_BRIEF: dict[str, str] = {
    "split_screen": (
        "Both actors face off, one on each side, a hard vertical seam of light "
        "between them."
    ),
    "face_closeup": (
        "One actor's face fills roughly half the frame with a clear reaction."
    ),
    "full_bleed": (
        "A single dramatic energy burst fills the frame behind both actors."
    ),
    "text_dominant": (
        "The actor is offset to one side rather than centred — but still close, "
        "still big enough that their face and expression read clearly at "
        "thumbnail size."
    ),
    "product_forward": (
        "The FinanceOS product surface or its mark is visible and legible, with "
        "one actor presenting it."
    ),
}

STYLE_BRIEF: dict[str, str] = {
    "house_energy": (
        "The proven Datarails look. Extremely high contrast, built to stop a "
        "scroll. BACKGROUND: splits deep navy blue against vivid orange with hot "
        "white light where they meet, carrying embers, sparks or light rays, lit "
        "cinematically. SUBJECT: the cut-out people from the footage stand in "
        "front of it with a subtle light rim separating them."
    ),
    "dark_cinematic": (
        "Restrained and expensive, like a prestige drama poster. BACKGROUND: "
        "near-black, with one hard warm key light raking in from the side and a "
        "deep falloff into shadow. No colour split, no embers, no sparks, no "
        "glow effects. A single orange accent at most. SUBJECT: the cut-out "
        "person from the footage, placed against that darkness with a faint warm "
        "rim light along one edge so they separate from it — their face stays "
        "bright enough to read clearly and is NOT lost in shadow."
    ),
    "flat_graphic": (
        "A bold flat-colour treatment. BACKGROUND: two or three solid flat "
        "colour fields — deep navy, vivid orange, off-white — with hard "
        "geometric edges. No gradients, no glow, no sparks, no photographic "
        "scenery. SUBJECT: the cut-out person from the footage sits on those "
        "colour fields with a crisp offset shadow, kept LARGE in the frame with "
        "their expression fully readable — this is a thumbnail, not a minimal "
        "print poster, so never shrink them or turn them away."
    ),
    "clean_corporate": (
        "Calm software credibility, bright and modern. BACKGROUND: a clean, "
        "light, near-white or very pale grey field with a soft even wash of "
        "light. No drama, no sparks, no dark vignette. SUBJECT: the cut-out "
        "person from the footage, placed on that light background with a soft "
        "contact shadow so they separate from it. They keep the clothing they "
        "are wearing in the footage — do not put them in different clothes, a "
        "different setting, an office, or at a desk, and do not replace them "
        "with anyone else. One vivid orange accent at most."
    ),
}
```

Update the comment above `STYLE_BRIEF` to: `# Each brief specifies palette, lighting and subject treatment. Type is NOT described here any more: the headline is set by src/typeset.py, whose TYPE_TREATMENTS carry the per-style type rules.`

In `src/prompts.py`, replace `render_prompt`:

```python
from .config import PRIMARY_RATIO
from .typeset import zone_instruction


def render_prompt(
    variant: Variant, people_in_ad: bool = True,
    style: str | None = None, ratio: str = PRIMARY_RATIO,
) -> str:
    """The art-only prompt. `style` overrides the slot's style (swap look)."""
    filled = (
        load("render")
        .replace("{style_brief}", STYLE_BRIEF[style or variant.style])
        .replace("{treatment_brief}", TREATMENT_BRIEF[variant.treatment])
        .replace("{scene_direction}", variant.scene_direction)
        .replace("{zone_instruction}", zone_instruction(variant.treatment, ratio))
    )
    if not people_in_ad:
        filled += NO_PEOPLE_RULE
    return filled
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest -q tests/test_prompts.py tests/test_models.py`
Expected: all PASS. `tests/test_render.py` will fail until Task 6 (it still asserts on headline text via `render_variant`); that is expected here.

- [ ] **Step 5: Commit**

```bash
git add prompts/render.md prompts/qa_legibility.md src/models.py src/prompts.py tests/test_prompts.py
git commit -m "$(cat <<'EOF'
Ask the model for art only: no text, a reserved zone, no type direction

The render prompt drops the headline and every instruction about lettering, and
gains the zone instruction generated from typeset's zones, so the model leaves
a calm area where we will set the type. TREATMENT_BRIEF loses its headline
sentences and STYLE_BRIEF its TYPE clauses — describing type to a model that
must render none of it only invited stray lettering. render_prompt() takes a
style override (for swap look) and a ratio (the zone differs on 9:16).

The legibility QA prompt is retired: legibility is now a property of our own
typesetting, not something to read back.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `render_art` — the image call, per ratio and style

**Files:**
- Modify: `src/render.py:42-113`
- Modify: `tests/test_render.py`

**Interfaces:**
- Consumes: `prompts.render_prompt(variant, people_in_ad, style, ratio)`, `config.RATIOS`.
- Produces: `render.render_art(variant: Variant, frames: dict[str, Path], client=None, extra_instruction: str = "", frame_override: Path | None = None, people_in_ad: bool = True, style: str | None = None, ratio: str = PRIMARY_RATIO) -> bytes`. `RenderError` / `RenderBlocked` unchanged. `render_variant` is removed.

- [ ] **Step 1: Update the tests**

In `tests/test_render.py`: replace every `render.render_variant(` with `render.render_art(`. Replace `test_render_sends_the_configured_model_size_and_quality` and `test_extra_instruction_is_appended_to_the_prompt` with:

```python
def test_render_sends_the_configured_model_size_and_quality(frames):
    client = FakeClient()
    render.render_art(_variant(), frames, client=client)
    call = client.images.calls[0]
    assert call["model"] == IMAGE_MODEL
    assert call["size"] == GEN_SIZE            # 16:9 is the default ratio
    assert call["quality"] == "high"
    assert call["output_format"] == "png"
    assert call["n"] == 1


def test_each_ratio_asks_for_its_own_generation_size(frames):
    from src.config import RATIOS
    for ratio, ((gw, gh), _final) in RATIOS.items():
        client = FakeClient()
        render.render_art(_variant(), frames, client=client, ratio=ratio)
        assert client.images.calls[0]["size"] == f"{gw}x{gh}"


def test_extra_instruction_is_appended_to_the_prompt(frames):
    client = FakeClient()
    render.render_art(
        _variant(), frames, client=client,
        extra_instruction="The previous attempt painted sparks into the reserved area.",
    )
    assert "painted sparks into the reserved area" in client.images.calls[0]["prompt"]


def test_the_prompt_carries_no_headline_and_reserves_the_zone(frames):
    client = FakeClient()
    render.render_art(_variant(), frames, client=client)
    prompt = client.images.calls[0]["prompt"]
    assert "47K OVER" not in prompt
    assert "Reserve a calm area" in prompt


def test_a_style_override_reaches_the_prompt(frames):
    client = FakeClient()
    render.render_art(_variant(), frames, client=client, style="dark_cinematic")
    assert "near-black" in client.images.calls[0]["prompt"].lower()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest -q tests/test_render.py`
Expected: FAIL with `AttributeError: module 'src.render' has no attribute 'render_art'`

- [ ] **Step 3: Rename and extend**

In `src/render.py` replace the signature and prompt line of `render_variant`:

```python
from .config import IMAGE_MODEL, IMAGE_QUALITY, PRIMARY_RATIO, RATIOS


def render_art(
    variant: Variant,
    frames: dict[str, Path],
    client=None,
    extra_instruction: str = "",
    frame_override: Path | None = None,
    people_in_ad: bool = True,
    style: str | None = None,
    ratio: str = PRIMARY_RATIO,
) -> bytes:
    """Render the text-free art for one tile at `ratio`'s generation size.

    Returns raw PNG bytes at native generation size; the caller caches them.
    `style` overrides the slot's locked style (swap look). The headline is not
    sent — it is set by typeset.py afterwards.
    """
    (gen_w, gen_h), _final = RATIOS[ratio]
    gen_size = f"{gen_w}x{gen_h}"
    primary = frame_override or frames.get(variant.frame_id)
    if primary is None:
        raise RenderError(
            f"The planner picked a frame we don't have: {variant.frame_id}"
        )

    image_paths: list[Path] = [primary]
    if variant.second_frame_id:
        if variant.second_frame_id in frames:
            image_paths.append(frames[variant.second_frame_id])
        else:
            log.warning(
                "second_frame_id %r not found in frames; rendering with "
                "only the primary frame",
                variant.second_frame_id,
            )
    image_paths.extend(
        pick_refs(style or variant.style, variant.treatment, limit=3,
                  people_in_ad=people_in_ad)
    )
    image_paths = image_paths[:MAX_INPUT_IMAGES]

    prompt = render_prompt(variant, people_in_ad=people_in_ad, style=style, ratio=ratio)
    if extra_instruction:
        prompt += f"\n\n## Correction for this attempt\n\n{extra_instruction}"
    # ... the `with contextlib.ExitStack()` block is unchanged, using gen_size ...
```

Remove the `gen_size: str = GEN_SIZE` parameter and the `from .config import GEN_SIZE` import (keep `GEN_SIZE` in config for tests). Update the module docstring to `"""One tile's art in, PNG bytes out. One gpt-image-2 call per tile per ratio."""`.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest -q tests/test_render.py tests/test_prompts.py`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/render.py tests/test_render.py
git commit -m "$(cat <<'EOF'
render_art: one text-free image call per tile, per ratio, with a style override

render_variant becomes render_art. It derives the generation size from the
ratio instead of taking one, accepts a style override for swap-look, and its
prompt no longer carries the headline. Everything about frames, references,
moderation handling and payload guards is unchanged.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: QA keeps the likeness gate and drops legibility

**Files:**
- Modify: `src/qa.py`
- Modify: `tests/test_qa.py`

**Interfaces:**
- Consumes: `prompts.load("qa_likeness")`, `config.QA_MODEL`.
- Produces: `qa.QAResult` (dataclass: `ok: bool`, `problems: list[str]`, `likeness: str | None = None`, `checked: bool = False`; property `unverified -> bool` = `ok and not checked`), `qa.likeness_gate(path: Path, reference_frame: Path | None, client=None, people_in_ad: bool = True) -> QAResult`, `qa.hard_checks(path, expected_size)` unchanged, `qa.likeness_verdict(...)` unchanged. Removed: `qa.check`, `qa.normalize`, `qa.headline_is_legible`, `QAResult.transcribed`.

- [ ] **Step 1: Update the tests**

In `tests/test_qa.py`: delete `test_normalize_*`, `test_legible_*`, `test_not_legible_*`, `test_check_*`, `test_an_unavailable_vision_model_marks_the_result_unverified`, `test_a_verified_pass_is_not_marked_unverified`, `test_a_failed_check_is_not_marked_unverified`, `test_illegible_text_short_circuits_before_the_likeness_call`, and the `FakeResponses`/`FakeClient` classes. Keep the `hard_checks` tests. Replace `FakeLikenessClient` with a single-answer fake and rewrite the likeness tests against `likeness_gate`:

```python
class FakeLikenessClient:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []
        outer = self

        class _Responses:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                return type("R", (), {"output_text": outer.answer})()

        self.responses = _Responses()


def _frame(tmp_path, name="scene_001.jpg"):
    path = tmp_path / name
    Image.new("RGB", (1280, 720), (60, 60, 60)).save(path, "JPEG")
    return path


def test_a_render_of_the_right_actor_passes(tmp_path):
    result = qa.likeness_gate(_write(tmp_path / "a.png"), _frame(tmp_path),
                              client=FakeLikenessClient("SAME"))
    assert result.ok and result.likeness == "SAME" and result.checked
    assert not result.unverified


def test_an_invented_person_fails_the_tile(tmp_path):
    result = qa.likeness_gate(_write(tmp_path / "b.png"), _frame(tmp_path),
                              client=FakeLikenessClient("DIFFERENT"))
    assert not result.ok
    assert any("not the actor" in p for p in result.problems)


def test_a_render_with_no_person_fails(tmp_path):
    result = qa.likeness_gate(_write(tmp_path / "c.png"), _frame(tmp_path),
                              client=FakeLikenessClient("NOBODY"))
    assert not result.ok
    assert any("no person" in p for p in result.problems)


def test_an_unclear_verdict_does_not_block_the_render(tmp_path):
    result = qa.likeness_gate(_write(tmp_path / "d.png"), _frame(tmp_path),
                              client=FakeLikenessClient("UNCLEAR"))
    assert result.ok and result.likeness == "UNCLEAR"


def test_no_reference_frame_means_no_call_and_an_unverified_pass(tmp_path):
    client = FakeLikenessClient("SAME")
    result = qa.likeness_gate(_write(tmp_path / "f.png"), None, client=client)
    assert result.ok and result.likeness is None and client.calls == []
    assert result.unverified


def test_a_likeness_outage_passes_unverified_rather_than_losing_the_render(tmp_path):
    class Broken:
        class responses:
            @staticmethod
            def create(**kwargs):
                raise RuntimeError("vision down")

    result = qa.likeness_gate(_write(tmp_path / "g.png"), _frame(tmp_path), client=Broken())
    assert result.ok and result.likeness is None and result.unverified


def test_the_likeness_check_sends_the_render_and_the_frame_in_that_order(tmp_path):
    client = FakeLikenessClient("SAME")
    qa.likeness_gate(_write(tmp_path / "h.png"), _frame(tmp_path), client=client)
    content = client.calls[0]["input"][0]["content"]
    images = [c for c in content if c["type"] == "input_image"]
    assert len(images) == 2
    assert "same real person" in content[0]["text"]


def test_a_people_free_ad_fails_when_a_person_appears(tmp_path):
    for verdict in ("SAME", "DIFFERENT"):
        result = qa.likeness_gate(_write(tmp_path / f"p{verdict}.png"), _frame(tmp_path),
                                  client=FakeLikenessClient(verdict), people_in_ad=False)
        assert not result.ok
        assert any("invented" in p for p in result.problems)


def test_a_people_free_ad_passes_when_there_is_nobody(tmp_path):
    result = qa.likeness_gate(_write(tmp_path / "none.png"), _frame(tmp_path),
                              client=FakeLikenessClient("NOBODY"), people_in_ad=False)
    assert result.ok and result.likeness == "NOBODY"


def test_the_legibility_check_is_gone():
    assert not hasattr(qa, "check")
    assert not hasattr(qa, "headline_is_legible")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest -q tests/test_qa.py`
Expected: FAIL with `AttributeError: module 'src.qa' has no attribute 'likeness_gate'`

- [ ] **Step 3: Rewrite `qa.py`**

Replace the module docstring, `QAResult`, and everything from `normalize` down to the end of `check` with:

```python
"""Verification. The headline is set by us, so the only thing left to check on
a render is the person: is it really the actor from the ad?
"""

# (imports unchanged except: remove `re`; keep base64, io, logging, dataclass, Path, Image)


@dataclass
class QAResult:
    ok: bool
    problems: list[str] = field(default_factory=list)
    # SAME / DIFFERENT / NOBODY / UNCLEAR, or None when the check did not run.
    likeness: str | None = None
    # True when the vision model actually read this render. ok=True with
    # checked=False is "passed unverified" — the pipeline warns about it so a
    # silent outage cannot masquerade as a clean batch.
    checked: bool = False

    @property
    def unverified(self) -> bool:
        return self.ok and not self.checked


# hard_checks, _feed_size_data_url, _client, likeness_verdict: unchanged.


def likeness_gate(
    path: Path,
    reference_frame: Path | None,
    client=None,
    people_in_ad: bool = True,
) -> QAResult:
    """Pass, fail, or pass-unverified, on likeness alone.

    hard_checks are not run here: the art is at generation size at this point
    and the size/2MB rules apply to the delivered file, which
    postprocess.finalize_image guards.
    """
    if reference_frame is None:
        return QAResult(ok=True)

    verdict = likeness_verdict(path, reference_frame, client=client)
    if verdict is None:
        return QAResult(ok=True)                       # outage: unverified

    if not people_in_ad:
        if verdict in ("SAME", "DIFFERENT"):
            return QAResult(
                ok=False,
                problems=["this ad has no people in it, but the thumbnail shows a "
                          "person the model invented"],
                likeness=verdict, checked=True,
            )
        return QAResult(ok=True, likeness=verdict, checked=True)

    if verdict in ("DIFFERENT", "NOBODY"):
        problem = ("the person in this thumbnail is not the actor from the ad"
                   if verdict == "DIFFERENT"
                   else "this thumbnail has no person in it at all")
        return QAResult(ok=False, problems=[problem], likeness=verdict, checked=True)

    return QAResult(ok=True, likeness=verdict, checked=True)
```

Remove `FEED_WIDTH`'s legibility comment (keep the constant: `_feed_size_data_url` still defaults to it for the likeness call at width=512) and remove the `_PUNCT` regex.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest -q tests/test_qa.py`
Expected: all PASS. (`tests/test_pipeline.py` now fails on `qa.check` — expected until Task 9.)

- [ ] **Step 5: Commit**

```bash
git add src/qa.py tests/test_qa.py
git commit -m "$(cat <<'EOF'
QA: keep the likeness gate, drop the legibility read-back

Legibility is now a property of our own typesetting, so reading the headline
back with a vision model checks nothing. likeness_gate() is what remains: the
same SAME/DIFFERENT/NOBODY logic, the same people_in_ad handling, the same
fail-open on an outage — now surfaced through QAResult.checked so an
unverified pass is still visible to the user.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Image-in, image-out logo stamp and finalize

**Files:**
- Modify: `src/branding.py:151-203`
- Modify: `src/postprocess.py`
- Modify: `tests/test_branding.py`, `tests/test_postprocess.py`

**Interfaces:**
- Produces: `branding.stamp_logo_image(base: Image.Image) -> Image.Image` (RGB in, RGB out, same size; logs placement; returns the image untouched if logo files are missing). `branding.stamp_logo(path, out_path=None)` keeps its signature and delegates. `postprocess.finalize_image(image: Image.Image, out_path: Path, final_size=(FINAL_W, FINAL_H)) -> Path` (one Lanczos downscale if needed, PNG, JPEG-92 fallback over `MAX_BYTES`). `postprocess.finalize(image_bytes, out_path, final_size)` keeps its signature and delegates.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_branding.py`:

```python
def test_stamp_logo_image_returns_a_same_size_rgb_image_with_the_logo_on_it():
    from PIL import Image
    from src import branding
    base = Image.new("RGB", (2048, 1152), (30, 60, 120))
    out = branding.stamp_logo_image(base)
    assert out.size == base.size and out.mode == "RGB"
    assert out.tobytes() != base.tobytes(), "the logo was composited"
    assert base.getpixel((5, 5)) == (30, 60, 120), "the input is not mutated"
```

Append to `tests/test_postprocess.py`:

```python
def test_finalize_image_downscales_once_to_delivery_size(tmp_path):
    from PIL import Image
    from src import postprocess
    from src.config import FINAL_H, FINAL_W
    big = Image.new("RGB", (2048, 1152), (12, 24, 48))
    out = postprocess.finalize_image(big, tmp_path / "t.png")
    with Image.open(out) as im:
        assert im.size == (FINAL_W, FINAL_H)
    assert out.suffix == ".png"


def test_finalize_bytes_still_works_through_the_image_path(tmp_path):
    import io
    from PIL import Image
    from src import postprocess
    buf = io.BytesIO()
    Image.new("RGB", (1088, 1088), (1, 2, 3)).save(buf, "PNG")
    out = postprocess.finalize(buf.getvalue(), tmp_path / "s.png", final_size=(1080, 1080))
    with Image.open(out) as im:
        assert im.size == (1080, 1080)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest -q tests/test_branding.py tests/test_postprocess.py -k "image"`
Expected: FAIL with `AttributeError ... 'stamp_logo_image'` / `'finalize_image'`

- [ ] **Step 3: Refactor**

In `src/branding.py`, split `stamp_logo`:

```python
def stamp_logo_image(base_rgb: Image.Image) -> Image.Image:
    """Composite the logo onto a copy of `base_rgb`. Same size, RGB out.

    Missing logo files are not fatal: log and return the image unchanged.
    """
    if not LOGO_LIGHT.exists() or not LOGO_DARK.exists():
        log.warning("logo assets missing (%s, %s); leaving the image unbranded",
                    LOGO_LIGHT, LOGO_DARK)
        return base_rgb.copy()

    base = base_rgb.convert("RGBA")
    width, height = base.size
    margin = max(1, round(width * LOGO_MARGIN_FRACTION))
    target_w = max(1, round(width * LOGO_WIDTH_FRACTION))
    with Image.open(LOGO_LIGHT) as probe:
        target_h = max(1, round(probe.height * target_w / probe.width))

    placement = plan_placement(base, target_w, target_h, margin)
    if placement.needs_plate:
        _draw_plate(base, placement.box, placement.on_light)
        on_light = region_is_light(base, placement.box)
        log.info("logo plated at %s (busy score %.1f)", placement.corner, placement.score)
    else:
        on_light = placement.on_light
        log.info("logo placed bare at %s (busy score %.1f)", placement.corner, placement.score)

    logo_path = LOGO_DARK if on_light else LOGO_LIGHT
    with Image.open(logo_path) as opened_logo:
        logo = opened_logo.convert("RGBA")
        logo_h = max(1, round(logo.height * target_w / logo.width))
        logo = logo.resize((target_w, logo_h), Image.LANCZOS)
    base.alpha_composite(logo, (placement.box[0], placement.box[1]))
    return base.convert("RGB")


def stamp_logo(image_path: Path, out_path: Path | None = None) -> Path:
    """File-based wrapper kept for existing callers and tests."""
    image_path = Path(image_path)
    out_path = Path(out_path) if out_path else image_path
    with Image.open(image_path) as opened:
        stamped = stamp_logo_image(opened.convert("RGB"))
    if stamped.tobytes() == opened.convert("RGB").tobytes() and not (LOGO_LIGHT.exists() and LOGO_DARK.exists()):
        return image_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stamped.save(out_path, "PNG", optimize=True)
    return out_path
```

(The existing `test_branding.py` test for "missing logo leaves the image untouched and returns the input path" must keep passing — hence the early `return image_path` when the assets are missing. If that test compares by path only, simplify the wrapper's condition to `if not LOGO_LIGHT.exists() or not LOGO_DARK.exists(): return image_path` placed before opening the image.)

In `src/postprocess.py`:

```python
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
        rgb = rgb.resize(final_size, Image.LANCZOS)
    rgb.save(out_path, "PNG", optimize=True)
    if out_path.stat().st_size <= MAX_BYTES:
        return out_path
    jpeg_path = out_path.with_suffix(".jpg")
    rgb.save(jpeg_path, "JPEG", quality=JPEG_FALLBACK_QUALITY, optimize=True, progressive=True)
    out_path.unlink()
    return jpeg_path


def finalize(image_bytes: bytes, out_path: Path,
             final_size: tuple[int, int] = (FINAL_W, FINAL_H)) -> Path:
    """Bytes-in wrapper kept for existing callers and tests."""
    with Image.open(io.BytesIO(image_bytes)) as raw:
        return finalize_image(raw, out_path, final_size)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest -q tests/test_branding.py tests/test_postprocess.py`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/branding.py src/postprocess.py tests/test_branding.py tests/test_postprocess.py
git commit -m "$(cat <<'EOF'
Image-in, image-out logo stamp and finalize, so compose can run at native size

compose() typesets on the art at generation size, stamps the logo on that
canvas, and only then downscales — one resampling of the art, ever. That needs
stamp_logo and finalize to accept a PIL image; the path/bytes wrappers stay
for the existing callers and tests.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: The pipeline renders art, then composes

**Files:**
- Modify: `src/pipeline.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `render.render_art(...)`, `qa.likeness_gate(...)`, `typeset.typeset(...)`, `typeset.zone_is_busy(...)`, `typeset.zone_box(...)`, `branding.stamp_logo_image(...)`, `postprocess.finalize_image(...)`.
- Produces:
  - `RenderOutcome` gains `art_path: Path | None = None`, `style: str = ""`, `headline: str = ""`.
  - `ThumbResult` gains `art_paths: dict[str, Path] = {}`, `style: str = ""`, `headline: str = ""`, `render_calls: int = 0`, `images_billed: int = 0`.
  - `BatchOutcome` gains `frames: dict[str, Path] = {}`, `people_in_ad: bool = True`, `work_dir: Path | None = None`.
  - `pipeline.compose(art_path: Path, headline: str, style: str, treatment: str, ratio: str, out_path: Path, scrim: bool = False) -> tuple[Path, list[str]]`.
  - `pipeline.ZONE_REROLL_INSTRUCTION: str`.
  - `_one_render(variant, ratio, frames, art_dir, out_dir, client, sleeper=None, people_in_ad=True, style=None, headline=None) -> RenderOutcome`.
  - `generate_batch(...)` signature unchanged.

- [ ] **Step 1: Update the tests**

In `tests/test_pipeline.py`, update the `wired` fixture and the assertions that referenced `qa.check` / `render.render_variant`:

```python
@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Replace every external dependency with a deterministic fake."""
    frames = []
    for name in ("scene_001.jpg", "scene_002.jpg"):
        frame = tmp_path / name
        Image.new("RGB", (1280, 720), (60, 60, 60)).save(frame, "JPEG")
        frames.append(frame)
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"fake")

    monkeypatch.setattr(probe, "extract_frames", lambda video, out_dir, max_frames=16: frames)
    monkeypatch.setattr(probe, "extract_audio", lambda video, out_dir: audio)
    monkeypatch.setattr(plan_module, "build_plan", lambda *a, **k: _plan())
    monkeypatch.setattr(render, "render_art", lambda *a, **k: _flat_png(k.get("ratio", "16x9")))
    monkeypatch.setattr(qa, "likeness_gate",
                        lambda path, frame, **k: QAResult(ok=True, likeness="SAME", checked=True))
    monkeypatch.setattr(pipeline, "DEFAULT_SLEEPER", lambda seconds: None)
    monkeypatch.setattr(plan_module, "DEFAULT_SLEEPER", lambda seconds: None)
    return frames


def _flat_png(ratio="16x9", colour=(30, 60, 120)) -> bytes:
    """Real PNG bytes at the ratio's generation size, so typeset and finalize run for real."""
    from src.config import RATIOS
    buf = io.BytesIO()
    Image.new("RGB", RATIOS[ratio][0], colour).save(buf, "PNG")
    return buf.getvalue()
```

Then:
- In every test that monkeypatched `render.render_variant`, patch `render.render_art` instead and return `_flat_png(kwargs.get("ratio", "16x9"))` rather than `b"\x89PNG bytes"`.
- Replace `test_failed_qa_triggers_exactly_one_reroll` (legibility no longer exists) with:

```python
def test_a_failed_likeness_triggers_exactly_one_reroll(wired, tmp_path, monkeypatch):
    calls = {"render": 0}

    def counting_render(*args, **kwargs):
        calls["render"] += 1
        return _flat_png(kwargs.get("ratio", "16x9"))

    monkeypatch.setattr(render, "render_art", counting_render)
    monkeypatch.setattr(
        qa, "likeness_gate",
        lambda path, frame, **k: QAResult(ok=False, problems=["the person in this thumbnail is not the actor from the ad"], likeness="DIFFERENT", checked=True),
    )
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    assert calls["render"] == 30
    assert all(r.flagged for r in outcome.results)
    assert all(r.path is not None for r in outcome.results)
```

- Delete `test_reroll_passes_the_failure_reason_back_into_the_prompt` and `test_a_qa_reroll_does_not_make_the_user_wait` (their subject was the legibility re-roll). Keep `test_an_invented_person_triggers_a_reroll_with_likeness_advice` but patch `qa.likeness_gate` (signature `(path, frame, **k)`) and `render.render_art`.
- Replace `test_an_unverified_batch_warns_the_user` body's fake with `QAResult(ok=True, checked=False)`.
- Replace `test_a_real_2048x1152_render_survives_finalize_and_hard_checks` with:

```python
def test_a_real_2048x1152_art_becomes_a_1920x1080_thumbnail_with_the_headline(tmp_path, monkeypatch):
    """The integration the suite must never fake on both sides: real typeset,
    real logo stamp, real single downscale, real hard checks."""
    frames = []
    for name in ("scene_001.jpg", "scene_002.jpg"):
        frame = tmp_path / name
        Image.new("RGB", (1280, 720), (60, 60, 60)).save(frame, "JPEG")
        frames.append(frame)
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"fake")
    monkeypatch.setattr(probe, "extract_frames", lambda video, out_dir, max_frames=16: frames)
    monkeypatch.setattr(probe, "extract_audio", lambda video, out_dir: audio)
    monkeypatch.setattr(plan_module, "build_plan", lambda *a, **k: _plan())
    monkeypatch.setattr(render, "render_art", lambda *a, **k: _flat_png(k.get("ratio", "16x9"), (0, 0, 0)))
    monkeypatch.setattr(qa, "likeness_gate", lambda path, frame, **k: QAResult(ok=True, likeness="SAME", checked=True))
    monkeypatch.setattr(pipeline, "DEFAULT_SLEEPER", lambda seconds: None)

    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")

    from src import typeset
    for result in outcome.results:
        assert result.path is not None and not result.flagged, result.note
        assert qa.hard_checks(result.path) == []
        with Image.open(result.path) as im:
            assert im.size == (FINAL_W, FINAL_H)
            box = typeset.zone_box(result.variant.treatment, "16x9", im.size)
            assert max(im.crop(box).convert("L").getdata()) > 200, "headline present in the zone"
        with Image.open(result.art_paths["16x9"]) as art:
            assert art.size == (2048, 1152), "art is cached at generation size"
```

- Add:

```python
def test_the_legibility_model_is_never_called(wired, tmp_path):
    assert not hasattr(qa, "check")
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    assert all(not r.unverified for r in outcome.results)


def test_every_result_records_its_headline_and_style(wired, tmp_path):
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    for r in outcome.results:
        assert r.headline == r.variant.headline
        assert r.style == r.variant.style
        assert set(r.art_paths) == set(r.paths) == {"16x9", "1x1", "9x16"}


def test_the_outcome_carries_what_reroll_needs(wired, tmp_path):
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    assert set(outcome.frames) == {"scene_001.jpg", "scene_002.jpg"}
    assert outcome.people_in_ad is True
    assert outcome.work_dir == tmp_path / "work"


def test_art_painted_into_the_zone_is_rerolled_once_then_scrimmed(wired, tmp_path, monkeypatch):
    import random
    random.seed(3)
    from src.config import RATIOS

    def noisy(*args, **kwargs):
        w, h = RATIOS[kwargs.get("ratio", "16x9")][0]
        img = Image.new("RGB", (w, h))
        img.putdata([(random.randrange(256),) * 3 for _ in range(w * h)])
        buf = io.BytesIO(); img.save(buf, "PNG"); return buf.getvalue()

    seen = []
    def capturing(*args, **kwargs):
        seen.append(kwargs.get("extra_instruction", ""))
        return noisy(**kwargs)

    monkeypatch.setattr(render, "render_art", capturing)
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work", variant_count=1)
    assert outcome.render_calls == 6, "one reroll per ratio"
    assert any("reserved" in s for s in seen if s)
    r = outcome.results[0]
    assert r.flagged and "scrim" in r.note
    assert all(p is not None for p in r.paths.values())
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest -q tests/test_pipeline.py`
Expected: FAIL — `AttributeError: module 'src.render' has no attribute 'render_variant'` from `_one_render`, and missing `art_paths`/`frames` attributes.

- [ ] **Step 3: Rewrite `_one_render` and the result types**

In `src/pipeline.py`:

```python
from . import backoff, branding, postprocess, probe, qa, render, typeset
from . import plan as plan_module
from .config import PRIMARY_RATIO, RATIOS
from .models import DEFAULT_VARIANTS, BatchPlan, Variant

MAX_WORKERS = 15
DEFAULT_SLEEPER = time.sleep

ZONE_REROLL_INSTRUCTION = (
    "The previous attempt painted detail into the reserved area. Keep that area "
    "completely calm — flat colour or a soft gradient only — and move every "
    "person, object, spark and light effect out of it."
)


@dataclass
class RenderOutcome:
    variant: Variant
    ratio: str
    path: Path | None
    art_path: Path | None = None
    style: str = ""
    headline: str = ""
    flagged: bool = False
    note: str = ""
    unverified: bool = False
    attempts: int = 0
    billed: int = 0


@dataclass
class ThumbResult:
    variant: Variant
    paths: dict[str, Path] = field(default_factory=dict)
    art_paths: dict[str, Path] = field(default_factory=dict)
    style: str = ""
    headline: str = ""
    flagged: bool = False
    note: str = ""
    unverified: bool = False
    render_calls: int = 0
    images_billed: int = 0

    @property
    def path(self) -> Path | None:
        return self.paths.get(PRIMARY_RATIO)


@dataclass
class BatchOutcome:
    plan: BatchPlan
    results: list[ThumbResult]
    warnings: list[str] = field(default_factory=list)
    planned_renders: int = 0
    render_calls: int = 0
    images_billed: int = 0
    # What per-tile actions need later in the session.
    frames: dict[str, Path] = field(default_factory=dict)
    people_in_ad: bool = True
    work_dir: Path | None = None

    @property
    def rerolls(self) -> int:
        return max(0, self.render_calls - self.planned_renders)


def _slug(variant: Variant) -> str:
    return f"{variant.index:02d}_{variant.hook_type}_{variant.treatment}"


def compose(
    art_path: Path, headline: str, style: str, treatment: str, ratio: str,
    out_path: Path, scrim: bool = False,
) -> tuple[Path, list[str]]:
    """Art on disk → delivered thumbnail on disk. Pure Pillow, ~100ms.

    typeset (native size) → logo → ONE downscale → size/2MB guard.
    """
    with Image.open(art_path) as opened:
        art = opened.convert("RGB")
    set_ = typeset.typeset(art, headline, style, treatment, ratio, scrim=scrim)
    branded = branding.stamp_logo_image(set_.image)
    written = postprocess.finalize_image(branded, out_path, final_size=RATIOS[ratio][1])
    return written, set_.notes


def _one_render(
    variant: Variant, ratio: str, frames: dict[str, Path], art_dir: Path, out_dir: Path,
    client, sleeper=None, people_in_ad: bool = True,
    style: str | None = None, headline: str | None = None,
) -> RenderOutcome:
    """Render art (likeness gate, zone gate), then compose. Nothing escapes."""
    sleeper = sleeper or DEFAULT_SLEEPER
    style = style or variant.style
    headline = headline or variant.headline
    offset = variant.index * backoff.STAGGER
    extra_instruction = ""
    frame_override: Path | None = None
    last_note = ""
    attempts = 0
    billed = 0
    art_path = art_dir / f"{_slug(variant)}_{ratio}.png"
    out_path = out_dir / f"{_slug(variant)}_{ratio}.png"

    def _done(**kw) -> RenderOutcome:
        return RenderOutcome(variant=variant, ratio=ratio, style=style, headline=headline,
                             attempts=attempts, billed=billed, **kw)

    try:
        zone_busy_last = False
        for attempt in (1, 2):
            try:
                attempts += 1
                raw = render.render_art(
                    variant, frames, client=client, extra_instruction=extra_instruction,
                    frame_override=frame_override, people_in_ad=people_in_ad,
                    style=style, ratio=ratio,
                )
                billed += 1
            except render.RenderBlocked as exc:
                last_note = f"blocked by content filter: {exc}"
                frame_override = _other_frame(frames, variant.frame_id)
                if attempt == 1:
                    backoff.wait(attempt, sleeper=sleeper, offset=offset)
                continue
            except render.RenderError as exc:
                last_note = str(exc)
                if attempt == 1:
                    backoff.wait(attempt, sleeper=sleeper, offset=offset)
                continue

            art_dir.mkdir(parents=True, exist_ok=True)
            art_path.write_bytes(raw)

            source_frame = frame_override or frames.get(variant.frame_id)
            gate = qa.likeness_gate(art_path, source_frame, client=client, people_in_ad=people_in_ad)
            with Image.open(art_path) as art:
                zone_busy = typeset.zone_is_busy(
                    art.convert("RGB"), typeset.zone_box(variant.treatment, ratio, art.size))

            if gate.ok and not zone_busy:
                path, notes = compose(art_path, headline, style, variant.treatment, ratio, out_path)
                return _done(path=path, art_path=art_path, unverified=gate.unverified,
                             flagged=bool(notes), note="; ".join(notes))

            problems = list(gate.problems)
            if zone_busy:
                problems.append("art painted into the text zone")
            last_note = "; ".join(problems)
            zone_busy_last = zone_busy

            if attempt == 1:
                if not gate.ok and gate.likeness in ("DIFFERENT", "NOBODY"):
                    extra_instruction = (
                        f"The previous attempt failed verification: {last_note}. "
                        "You generated a person who is not in the reference frames. Do not "
                        "invent, replace or beautify anyone. Cut the exact person out of the "
                        "reference frame and place them in front of the background, keeping "
                        "their face, age, hair, facial hair and clothing identical."
                    )
                    if zone_busy:
                        extra_instruction += " " + ZONE_REROLL_INSTRUCTION
                else:
                    extra_instruction = ZONE_REROLL_INSTRUCTION
                continue

            # Second failure: hand it over, flagged. Scrim if the zone is still busy.
            path, notes = compose(art_path, headline, style, variant.treatment, ratio,
                                  out_path, scrim=zone_busy_last)
            return _done(path=path, art_path=art_path, flagged=True,
                         note="; ".join([last_note] + notes), unverified=gate.unverified)
    except Exception as exc:
        log.warning("variant %s/%s failed unexpectedly", variant.index, ratio, exc_info=True)
        return _done(path=None, art_path=art_path if art_path.exists() else None,
                     flagged=True, note=f"something went wrong rendering this one ({exc})")

    return _done(path=None, flagged=True, note=last_note)
```

Add `from PIL import Image` to the imports. Update `_group_by_variant` to carry the new fields:

```python
    for outcome in outcomes:
        row = by_index[outcome.variant.index]
        row.style = outcome.style
        row.headline = outcome.headline
        row.render_calls += outcome.attempts
        row.images_billed += outcome.billed
        if outcome.path is not None:
            row.paths[outcome.ratio] = outcome.path
        if outcome.art_path is not None:
            row.art_paths[outcome.ratio] = outcome.art_path
        # flagged / unverified / note aggregation unchanged
```

In `generate_batch`, create `art_dir = work_dir / "art"` and pass `art_dir, out_dir` to `_one_render`; return the outcome with `frames=frames, people_in_ad=batch_plan.people_in_ad, work_dir=work_dir` added. Update the module docstring's hard-contract sentence to mention art and final.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest -q tests/test_pipeline.py`
Expected: all PASS. If `test_art_painted_into_the_zone_is_rerolled_once_then_scrimmed` fails on `render_calls == 6`, check the zone gate runs on attempt 1 *and* the scrim path composes on attempt 2 without a third call.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline.py tests/test_pipeline.py
git commit -m "$(cat <<'EOF'
Pipeline: render text-free art, gate it on likeness and a clear zone, then compose

_one_render is now two steps. render_art fetches the picture and caches it at
generation size under work_dir/art; the likeness gate and a zone-busy check
decide whether to re-roll once; compose() typesets the headline, stamps the
logo and downscales exactly once into work_dir/out. A zone still busy after
the re-roll gets a scrim under the type rather than a failed tile. Results
carry the art paths, the headline and the style, and the outcome carries the
frames and people flag, which is what per-tile actions need later.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Per-tile actions — `retitle()` and `reroll()`

**Files:**
- Modify: `src/pipeline.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `compose(...)`, `_one_render(...)`, `BatchOutcome.frames/people_in_ad/work_dir`.
- Produces: `pipeline.retitle(result: ThumbResult, headline: str) -> ThumbResult` (re-composes every ratio that has art; no API call; returns a new `ThumbResult` with `headline` set and notes refreshed). `pipeline.reroll(result: ThumbResult, outcome: BatchOutcome, client=None, style: str | None = None, sleeper=None) -> ThumbResult` (re-renders every ratio via `_one_render`, using `result.headline`, `style or result.style`; adds its calls to `outcome.render_calls/images_billed`; returns the new row).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`:

```python
def test_retitle_recomposes_every_ratio_for_free(wired, tmp_path, monkeypatch):
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work", variant_count=1)
    calls = {"n": 0}
    monkeypatch.setattr(render, "render_art", lambda *a, **k: calls.__setitem__("n", calls["n"] + 1))
    before = {r: outcome.results[0].paths[r].read_bytes() for r in outcome.results[0].paths}
    new = pipeline.retitle(outcome.results[0], "NEW LINE HERE")
    assert calls["n"] == 0, "no image call"
    assert new.headline == "NEW LINE HERE"
    assert set(new.paths) == {"16x9", "1x1", "9x16"}
    for ratio, path in new.paths.items():
        assert path.read_bytes() != before[ratio], f"{ratio} was re-set"
    assert new.art_paths == outcome.results[0].art_paths, "the art is untouched"


def test_retitle_flags_a_headline_that_does_not_fit(wired, tmp_path):
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work", variant_count=1)
    new = pipeline.retitle(outcome.results[0], "EXTRAORDINARILY LONG WORDS EVERYWHERE HERE")
    assert new.flagged and "too long" in new.note


def test_reroll_rerenders_all_three_ratios_and_bills_them(wired, tmp_path, monkeypatch):
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work", variant_count=1)
    calls = {"n": 0}

    def counting(*a, **k):
        calls["n"] += 1
        return _flat_png(k.get("ratio", "16x9"), (90, 20, 20))

    monkeypatch.setattr(render, "render_art", counting)
    before_calls = outcome.render_calls
    new = pipeline.reroll(outcome.results[0], outcome)
    assert calls["n"] == 3
    assert outcome.render_calls == before_calls + 3
    assert outcome.images_billed == before_calls + 3
    assert new.headline == outcome.results[0].headline
    assert new.style == outcome.results[0].style
    assert all(p.exists() for p in new.paths.values())


def test_reroll_with_a_style_swaps_the_look(wired, tmp_path, monkeypatch):
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work", variant_count=1)
    seen = []

    def capturing(*a, **k):
        seen.append(k.get("style"))
        return _flat_png(k.get("ratio", "16x9"))

    monkeypatch.setattr(render, "render_art", capturing)
    new = pipeline.reroll(outcome.results[0], outcome, style="dark_cinematic")
    assert seen == ["dark_cinematic"] * 3
    assert new.style == "dark_cinematic"
    assert new.variant.style == "house_energy", "the matrix slot is unchanged; the row is off-matrix"


def test_reroll_keeps_the_hard_contract_when_the_api_fails(wired, tmp_path, monkeypatch):
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work", variant_count=1)

    def fails(*a, **k):
        raise render.RenderError("503")

    monkeypatch.setattr(render, "render_art", fails)
    new = pipeline.reroll(outcome.results[0], outcome)
    assert new.variant.index == 1
    assert all(p is None for p in [new.paths.get(r) for r in ("16x9", "1x1", "9x16")])
    assert "503" in new.note
    assert outcome.images_billed == 3, "failed calls are not billed"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest -q tests/test_pipeline.py -k "retitle or reroll"`
Expected: FAIL with `AttributeError: module 'src.pipeline' has no attribute 'retitle'`

- [ ] **Step 3: Implement**

Append to `src/pipeline.py`:

```python
def retitle(result: ThumbResult, headline: str) -> ThumbResult:
    """Re-set the headline on every ratio's cached art. No API call."""
    new = ThumbResult(
        variant=result.variant, art_paths=dict(result.art_paths), style=result.style,
        headline=" ".join(headline.upper().split()), unverified=result.unverified,
        render_calls=result.render_calls, images_billed=result.images_billed,
    )
    notes: list[str] = []
    for ratio, art_path in result.art_paths.items():
        out_path = art_path.parent.parent / "out" / art_path.name
        try:
            path, tile_notes = compose(art_path, new.headline, new.style,
                                       result.variant.treatment, ratio, out_path)
            new.paths[ratio] = path
            notes += [n if ratio == PRIMARY_RATIO else f"{ratio}: {n}" for n in tile_notes]
        except Exception as exc:
            log.warning("retitle failed for %s", art_path, exc_info=True)
            notes.append(f"{ratio}: couldn't set the headline ({exc})")
    new.flagged = bool(notes)
    new.note = "; ".join(notes)
    return new


def reroll(
    result: ThumbResult, outcome: BatchOutcome, client=None,
    style: str | None = None, sleeper=None,
) -> ThumbResult:
    """Re-render one concept's art at every ratio; keep its headline.

    `style` swaps the look (off-matrix, for this tile only). Adds the calls to
    the outcome's running totals. Returns the new row; the caller swaps it in.
    """
    if outcome.work_dir is None:
        raise RuntimeError("this batch did not record its work_dir")
    art_dir = outcome.work_dir / "art"
    out_dir = outcome.work_dir / "out"
    jobs = list(RATIOS)
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        outcomes = list(pool.map(
            lambda ratio: _one_render(
                result.variant, ratio, outcome.frames, art_dir, out_dir, client,
                sleeper=sleeper, people_in_ad=outcome.people_in_ad,
                style=style or result.style, headline=result.headline,
            ),
            jobs,
        ))
    row = _group_by_variant(outcomes, [result.variant])[0]
    outcome.render_calls += sum(o.attempts for o in outcomes)
    outcome.images_billed += sum(o.billed for o in outcomes)
    return row
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest -q tests/test_pipeline.py`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipeline.py tests/test_pipeline.py
git commit -m "$(cat <<'EOF'
Per-tile actions: retitle for free, reroll (optionally in another style) for three calls

retitle() re-sets the headline on a concept's cached art at every ratio with no
API call. reroll() re-renders the art at every ratio through the same
_one_render path the batch used, keeping the headline; a style argument turns
it into swap-look, which deliberately steps off the matrix for that tile only.
Both return a fresh ThumbResult; reroll adds its calls to the batch's running
cost so the footer stays honest.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: "3 more lines" — headline suggestions from the planner

**Files:**
- Create: `prompts/headlines.md`
- Modify: `src/models.py` (extract `clean_headline_text`), `src/prompts.py` (`headline_prompt`), `src/plan.py` (`suggest_headlines`)
- Modify: `tests/test_models.py`, `tests/test_prompts.py`, `tests/test_plan.py`

**Interfaces:**
- Produces: `models.clean_headline_text(text: str) -> str` (uppercases, collapses whitespace, strips a trailing period, raises `ValueError` if empty or > `MAX_HEADLINE_WORDS`); `Variant.clean_headline` delegates to it. `models.HeadlineOptions(BaseModel)` with `headlines: list[str]`. `prompts.headline_prompt(variant: Variant, ad_summary: str, current: str, n: int = 3) -> str`. `plan.suggest_headlines(variant: Variant, ad_summary: str, current: str, client=None, n: int = 3) -> list[str]` — returns up to `n` cleaned, valid, distinct headlines (never the current one); returns `[]` on any failure, never raises.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_models.py`:

```python
def test_clean_headline_text_normalises_and_enforces_the_word_cap():
    from src.models import clean_headline_text
    assert clean_headline_text("  same   ai. ") == "SAME AI"
    with pytest.raises(ValueError):
        clean_headline_text("one two three four five six")
    with pytest.raises(ValueError):
        clean_headline_text("   ")
```

Append to `tests/test_prompts.py`:

```python
def test_headline_prompt_names_the_hook_and_the_current_line():
    text = prompts.headline_prompt(_variant(hook="pain", treatment="text_dominant", index=4),
                                   ad_summary="two colleagues disagree", current="SAME AI DIFFERENT ANSWER")
    assert "pain" in text and "SAME AI DIFFERENT ANSWER" in text
    assert "two colleagues disagree" in text
    assert "three" in text.lower()
    assert "{" not in text
```

Append to `tests/test_plan.py`:

```python
from src.models import HeadlineOptions


class FakeHeadlineResponses:
    def __init__(self, headlines, fail=False):
        self.parsed = HeadlineOptions(headlines=headlines)
        self.fail = fail
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("planner down")
        return type("R", (), {"output_parsed": self.parsed})()


def _headline_client(headlines, fail=False):
    c = FakeClient()
    c.responses = FakeHeadlineResponses(headlines, fail)
    return c


def test_suggest_headlines_returns_cleaned_valid_distinct_lines():
    v = _valid_plan().variants[0]
    got = plan.suggest_headlines(v, "x", "HOOK 1",
                                 client=_headline_client(["forecast is wrong", "HOOK 1", "same ai, two answers.", "one two three four five six", "FORECAST IS WRONG"]))
    assert got == ["FORECAST IS WRONG", "SAME AI, TWO ANSWERS"]


def test_suggest_headlines_never_raises():
    v = _valid_plan().variants[0]
    assert plan.suggest_headlines(v, "x", "HOOK 1", client=_headline_client([], fail=True)) == []


def test_suggest_headlines_sends_no_images():
    v = _valid_plan().variants[0]
    client = _headline_client(["A B"])
    plan.suggest_headlines(v, "x", "HOOK 1", client=client)
    content = client.responses.calls[0]["input"][0]["content"]
    assert all(c["type"] == "input_text" for c in content)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest -q tests/test_models.py tests/test_prompts.py tests/test_plan.py -k "clean_headline_text or headline_prompt or suggest"`
Expected: FAIL with `ImportError: cannot import name 'clean_headline_text'`

- [ ] **Step 3: Implement**

Create `prompts/headlines.md`:

```markdown
You are the creative director for Datarails, a financial planning platform. One
thumbnail concept for a YouTube ad needs alternative headlines.

The ad: {ad_summary}

The concept's hook type is **{hook_type}** — {row_intent}. Its current headline is:

"{current}"

Write {n} different headlines for this same concept. Each must:

- be five words maximum, twenty-two characters or fewer where you can manage it
- not be a sentence: no ending period, no quotation marks, no exclamation marks
- read at 320 pixels wide
- stay true to the hook type above and to what the ad actually says
- never invent a claim or a statistic
- be genuinely different from the current headline and from each other

Return only the headlines.
```

In `src/models.py`, above `class Variant`:

```python
def clean_headline_text(text: str) -> str:
    """One rule for every headline the tool accepts, planned or suggested."""
    v = " ".join((text or "").split()).rstrip(".").strip().upper()
    if not v:
        raise ValueError("headline cannot be empty")
    if len(v.split()) > MAX_HEADLINE_WORDS:
        raise ValueError(
            f"headline must be {MAX_HEADLINE_WORDS} words or fewer, got "
            f"{len(v.split())}: {v!r}"
        )
    return v


class HeadlineOptions(BaseModel):
    """Structured output for 'more lines': the planner returns only headlines."""
    headlines: list[str]
```

and make `Variant.clean_headline` a one-liner: `return clean_headline_text(v)`.

In `src/prompts.py`:

```python
def headline_prompt(variant: Variant, ad_summary: str, current: str, n: int = 3) -> str:
    return (
        load("headlines")
        .replace("{ad_summary}", ad_summary.strip() or "no summary available")
        .replace("{hook_type}", variant.hook_type)
        .replace("{row_intent}", ROW_INTENT.get(variant.index, ""))
        .replace("{current}", current)
        .replace("{n}", NUMBER_WORDS.get(n, str(n)))
    )
```

In `src/plan.py`:

```python
from .models import HeadlineOptions, Variant, clean_headline_text
from .prompts import headline_prompt


def suggest_headlines(
    variant: Variant, ad_summary: str, current: str, client=None, n: int = 3,
) -> list[str]:
    """Up to `n` alternative headlines for one concept. Text only, never raises."""
    try:
        response = _client(client).responses.parse(
            model=PLANNER_MODEL,
            input=[{"role": "user", "content": [
                {"type": "input_text", "text": headline_prompt(variant, ad_summary, current, n)},
            ]}],
            text_format=HeadlineOptions,
        )
        raw = list(response.output_parsed.headlines)
    except Exception:
        log.warning("headline suggestions unavailable", exc_info=True)
        return []

    keep: list[str] = []
    current_clean = " ".join(current.upper().split())
    for line in raw:
        try:
            cleaned = clean_headline_text(line)
        except ValueError:
            continue
        if cleaned != current_clean and cleaned not in keep:
            keep.append(cleaned)
        if len(keep) == n:
            break
    return keep
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest -q tests/test_models.py tests/test_prompts.py tests/test_plan.py`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add prompts/headlines.md src/models.py src/prompts.py src/plan.py tests/test_models.py tests/test_prompts.py tests/test_plan.py
git commit -m "$(cat <<'EOF'
Suggest alternative headlines for one concept, text only

suggest_headlines() asks the planner for a few more lines for a single hook, with
no images attached, and returns only the ones that pass the same headline rule
the plan itself obeys — extracted into clean_headline_text() so the two can
never disagree. Nothing is rendered until the user picks one.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: The action row in the results grid

**Files:**
- Modify: `app.py` (helpers near `cost_line`; results loop `for tab, (ratio, tab_label) ...`)
- Modify: `tests/test_app_helpers.py`

**Interfaces:**
- Consumes: `pipeline.retitle`, `pipeline.reroll`, `plan.suggest_headlines`, `models.STYLE_BRIEF` keys, `config.IMAGE_COST_USD`.
- Produces pure helpers: `app.price_label(n_images: int) -> str` (e.g. `"↻ Re-roll art · 3 images ≈ $0.60"`), `app.style_caption(result: ThumbResult) -> str` (hook · treatment · style, plus `" · off-matrix"` when `result.style != result.variant.style`), `app.download_name(result: ThumbResult, ratio: str) -> str` (the file's name with `_edited` inserted before the suffix when `result.headline != result.variant.headline`), `app.replace_result(outcome: BatchOutcome, new: ThumbResult) -> None` (swaps the row with the same `variant.index`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_app_helpers.py`:

```python
from src.pipeline import BatchOutcome, ThumbResult
from src.models import BatchPlan, MATRIX, Variant


def _result(headline="HOOK 1", style="house_energy"):
    v = Variant(index=1, hook_type="stat", treatment="split_screen", headline="HOOK 1",
                frame_id="scene_001.jpg", second_frame_id=None, scene_direction="x", rationale="y")
    return ThumbResult(variant=v, paths={"16x9": Path("out/01_stat_split_screen_16x9.png")},
                       headline=headline, style=style)


def test_price_label_shows_the_image_count_and_the_shared_price():
    from src.config import IMAGE_COST_USD
    label = app.price_label(3)
    assert "3 images" in label and f"${3 * IMAGE_COST_USD:.2f}" in label


def test_style_caption_marks_an_off_matrix_tile():
    assert "off-matrix" not in app.style_caption(_result())
    assert app.style_caption(_result(style="dark_cinematic")).endswith("off-matrix")
    assert "dark_cinematic" in app.style_caption(_result(style="dark_cinematic"))


def test_download_name_marks_an_edited_headline():
    assert app.download_name(_result(), "16x9") == "01_stat_split_screen_16x9.png"
    assert app.download_name(_result(headline="NEW LINE"), "16x9") == "01_stat_split_screen_16x9_edited.png"


def test_replace_result_swaps_the_row_with_the_same_index():
    plan = BatchPlan(ad_summary="x", transcript_used=True, variants=[])
    old = _result()
    outcome = BatchOutcome(plan=plan, results=[old])
    new = _result(headline="CHANGED")
    app.replace_result(outcome, new)
    assert outcome.results == [new]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest -q tests/test_app_helpers.py -k "price_label or style_caption or download_name or replace_result"`
Expected: FAIL with `AttributeError: module 'app' has no attribute 'price_label'`

- [ ] **Step 3: Add the helpers and wire the action row**

In `app.py`, after `cost_line`:

```python
def price_label(n_images: int) -> str:
    return f"↻ Re-roll art · {n_images} images ≈ ${n_images * IMAGE_COST_USD:.2f}"


def style_caption(result: ThumbResult) -> str:
    label = f"{result.variant.hook_type} · {result.variant.treatment} · {result.style or result.variant.style}"
    if result.style and result.style != result.variant.style:
        label += " · off-matrix"
    return label


def download_name(result: ThumbResult, ratio: str) -> str:
    path = result.paths[ratio]
    if result.headline and result.headline != result.variant.headline:
        return f"{path.stem}_edited{path.suffix}"
    return path.name


def replace_result(outcome: BatchOutcome, new: ThumbResult) -> None:
    for i, row in enumerate(outcome.results):
        if row.variant.index == new.variant.index:
            outcome.results[i] = new
            return
    outcome.results.append(new)
```

Add imports: `from src import auth, drive, plan as plan_module, session as dr_session` and `from src.models import DEFAULT_VARIANTS, MAX_VARIANTS, MIN_VARIANTS, STYLE_BRIEF` and `from src.pipeline import BatchOutcome, ThumbResult, generate_batch, reroll, retitle`.

Replace the per-card body inside the `for position, result in enumerate(outcome.results):` loop with:

```python
                    with columns[position % 3]:
                        idx = result.variant.index
                        label = style_caption(result)
                        path = result.paths.get(ratio)
                        if path is None:
                            st.error(f"**{label}** — no {tab_label.split(' · ')[0]} version. {result.note}")
                        else:
                            st.image(str(path), caption=label)
                            if result.flagged:
                                st.warning(f"⚠️ {result.note}")
                            st.download_button(
                                "Download", path.read_bytes(),
                                file_name=download_name(result, ratio),
                                mime=("image/png" if path.suffix == ".png" else "image/jpeg"),
                                key=f"dl_{idx}_{ratio}",
                            )

                        # The action row is per CONCEPT: draw it once, on the 16:9 tab,
                        # and act on all three ratios.
                        if ratio != "16x9" or not result.art_paths:
                            continue
                        busy = st.session_state.get(f"busy_{idx}", False)
                        new_headline = st.text_input(
                            "Headline", value=result.headline, key=f"hl_{idx}",
                            disabled=busy, help="Press Enter to re-set it on all three sizes. Free.",
                        )
                        if new_headline.strip().upper() != result.headline and not busy:
                            with st.status("Re-setting the headline…", expanded=False):
                                replace_result(outcome, retitle(result, new_headline))
                            st.rerun()

                        suggestions = st.session_state.get(f"sugg_{idx}", [])
                        c1, c2, c3 = st.columns([1.1, 1.3, 1])
                        if c1.button("3 more lines", key=f"more_{idx}", disabled=busy):
                            got = plan_module.suggest_headlines(
                                result.variant, outcome.plan.ad_summary, result.headline)
                            if got:
                                st.session_state[f"sugg_{idx}"] = got
                            else:
                                st.warning("Couldn't get suggestions just now — the field above still works.")
                            st.rerun()
                        if suggestions:
                            pick = st.pills("Try one", suggestions, key=f"pick_{idx}", disabled=busy)
                            if pick and pick != result.headline:
                                replace_result(outcome, retitle(result, pick))
                                st.session_state[f"hl_{idx}"] = pick
                                st.session_state.pop(f"pick_{idx}", None)
                                st.rerun()

                        if c2.button(price_label(3), key=f"reroll_{idx}", disabled=busy):
                            st.session_state[f"busy_{idx}"] = True
                            with st.status(f"Re-rolling concept {idx}…", expanded=False):
                                replace_result(outcome, reroll(result, outcome))
                            st.session_state[f"busy_{idx}"] = False
                            st.rerun()

                        others = [s for s in STYLE_BRIEF if s != result.style]
                        look = c3.selectbox("Look", ["(keep)"] + others, key=f"look_{idx}", disabled=busy,
                                            label_visibility="collapsed")
                        if look != "(keep)":
                            st.session_state[f"busy_{idx}"] = True
                            with st.status(f"Re-rendering concept {idx} as {look}…", expanded=False):
                                replace_result(outcome, reroll(result, outcome, style=look))
                            st.session_state[f"busy_{idx}"] = False
                            st.session_state[f"look_{idx}"] = "(keep)"
                            st.rerun()
```

Update `zip_bytes` and the Save-to-Drive list to use `download_name(result, ratio)` for arcnames/file names (in `zip_bytes`: `arcname=f"{ratio}/{download_name(result, ratio)}"`; in `save_batch`, upload paths are unchanged — Drive names come from `path.name`, so pass `[(p, download_name(r, ratio)) ...]` only if `drive.save_batch` is extended; simplest: leave Drive names as the file names, which already differ per concept). The footer `st.caption(cost_line(outcome, planned_images=len(outcome.results) * 3))` now reflects re-rolls because `reroll` mutates `outcome`.

- [ ] **Step 4: Run the tests, then run the app once by hand**

Run: `.venv/bin/python -m pytest -q`
Expected: all PASS.

Then, with real secrets in `.streamlit/secrets.toml`: `.venv/bin/streamlit run app.py`, sign in, generate **1 concept**, and exercise: edit the headline (instant), "3 more lines" (pills appear), re-roll (footer cost rises by 3), Look → another style (caption shows off-matrix). Download one edited tile and confirm the `_edited` suffix. This is the one manual step in the plan; note the result in the PR body.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app_helpers.py
git commit -m "$(cat <<'EOF'
Results grid: edit the headline, try more lines, re-roll, or swap the look per concept

Each card gets an action row on the 16:9 tab that acts on all three sizes. The
headline field re-sets the type for free; "3 more lines" asks the planner for
alternatives and shows them as pills; re-roll re-renders the art for three
images and the footer's actual cost follows; Look re-renders in another style
and marks the row off-matrix. Downloads of an edited headline carry _edited.
A per-concept busy flag keeps two actions from racing on one tile.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: Docs, the live harness, both suites, and the PR

**Files:**
- Modify: `README.md` ("What you get" flags paragraph, "How it works"), `scripts/live_run.py`
- Verify: full suite in both venvs

- [ ] **Step 1: README**

In "What you get", replace the ⚠️ paragraph with:

```markdown
If a tile is marked ⚠️, read the note. The headline itself is never the problem
any more — it is set by the tool, not drawn by the model, in Poppins Black at a
size that survives the 320px a thumbnail gets in a feed. What can still be
flagged: the person in the render is not the actor from the ad (the likeness
check), the model painted detail into the area reserved for the headline (the
tool re-rolls once, then sets the type over a soft plate), or a headline too
long for that layout.

Under every concept, on the 16:9 tab: **edit the headline** and press Enter to
re-set it on all three sizes for free; **3 more lines** asks for alternatives;
**Re-roll art** regenerates just that concept's picture (three images); **Look**
re-renders it in another style — marked *off-matrix* so you know it was a
deliberate step outside the locked pairings.
```

In "How it works", replace the two paragraphs about `gpt-image-2` text placement and the 2048×1152 downscale with:

```markdown
The model renders the **artwork only** — background, the cut-out actors, the
energy — with a calm area reserved where the headline will go. The headline is
then set by `src/typeset.py` in Poppins Black with the type treatment each style
calls for, the logo is stamped, and the whole canvas is downscaled exactly once
from the model's native size (2048×1152 for 16:9) to the delivery size. Text is
drawn at twice the canvas size and composited down, so its edges are smooth;
nothing is ever scaled twice. Because the tool sets the type, editing a headline
costs nothing and never needs a new render.
```

- [ ] **Step 2: `scripts/live_run.py`**

After the results loop, add:

```python
    print(f"\nImages billed: {outcome.images_billed} (calls: {outcome.render_calls}, "
          f"re-rolls: {outcome.rerolls})")
    if outcome.results and outcome.results[0].art_paths:
        print(f"Art cached in {work_dir / 'art'} — try: retitle(outcome.results[0], 'NEW LINE')")
```

- [ ] **Step 3: Run both suites**

```bash
cd "/Users/omeryadgar/Desktop/dev_projects/YT Thumbnail creator"
.venv/bin/python -m pytest -q
SP=/private/tmp/claude-501/-Users-omeryadgar-Desktop-dev-projects/bcd55e69-4a55-4f31-b874-af6f70fe6a62/scratchpad
PATH=/usr/bin:/bin:/usr/sbin:/sbin $SP/cloudsim/bin/python -m pytest -q -p no:cacheprovider
```

Expected: all PASS in both (the second proves the fonts load and Pillow typesets on the Cloud's Python with no system ffmpeg). If the cloud-sim venv is gone, recreate it: `python3.13 -m venv $SP/cloudsim && $SP/cloudsim/bin/pip install -r requirements.txt`.

- [ ] **Step 4: Commit and open the PR**

```bash
git add README.md scripts/live_run.py
git commit -m "$(cat <<'EOF'
Document the headline layer and the per-concept actions

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
git push -u origin feat/text-layer
gh pr create --base master --title "Text as a layer: set the headline ourselves, iterate per concept for free" --body-file - <<'EOF'
Piece 3 of the improvement plan. Stacked on #4 (merge #2 → #3 → #4 → this).

Spec: docs/superpowers/specs/2026-09-08-text-as-a-layer-design.md
Plan: docs/superpowers/plans/2026-09-08-text-as-a-layer.md

## What changes
- gpt-image-2 renders **art only** with a reserved zone; `src/typeset.py` sets the headline in Poppins Black (full family bundled, OFL) with per-style type treatment; logo stamped; **one** downscale from native generation size. Text drawn at 2× — an anti-pixelation test counts grey levels along the edges.
- Legibility QA retired (deterministic now); likeness gate stays.
- Per concept: edit headline (free), 3 more lines (text-only planner call), re-roll art (3 images), swap look (3 images, marked off-matrix). Exports reflect current state; edited downloads carry `_edited`.

## Verification
- Full suite green locally and in a clean py3.13 venv with no system ffmpeg.
- Manual: 1-concept live batch; edit / more lines / re-roll / look exercised; footer cost tracked re-rolls. (Result: <fill in from Task 12 Step 4>)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
```

(The single `<fill in ...>` above is a placeholder in the **PR body**, to be replaced with the observed manual result at the time of opening the PR — not a plan step.)

---

## Self-review

**Spec coverage.**
- §1 two artifacts, module ownership, session model → Tasks 8, 9 (art/out dirs, `compose`), 12 (session state). ✔
- §2 fonts (Task 1), zones + bottom reserve (Task 2), fitting + floor + overflow flag (Task 3), type treatments + adaptive flat-graphic + 2× rendering + scrim (Task 4), caption pill out of scope (stated in Task 4's absence and README). ✔
- §3 native-size caching and single downscale (Tasks 8, 9), `render_art` with likeness-only gate (Tasks 6, 7, 9), `compose` (Task 9), per-tile wrappers and cost counting (Task 10), deterministic flags (Tasks 4, 9). ✔
- §4 action row: headline field, 3 more lines, re-roll with price, Look select, all-ratios-at-once, `_edited` exports, off-matrix marker, per-card status, per-tile busy lock (Task 12). ✔
- §5 failure table: zone busy → reroll → scrim (Task 9), overflow (Tasks 3–4), font fallback → `TypesetError` (Task 3; `retitle` catches per ratio in Task 10), API failure path unchanged (Task 9), suggestions failure → warning (Tasks 11–12), two actions → busy flag (Task 12). Tests: `test_typeset.py` incl. anti-pixelation (Task 4), render prompt (Task 6), pipeline contract/no legibility call/sizes/cost (Tasks 9–10), app helpers (Task 12), integration test (Task 9). ✔

**Placeholder scan.** No TBD/TODO. The one `<fill in>` is in the PR body text, flagged as such. Task 4's note about `font =` selection offers a simplification but both forms are complete code.

**Type consistency.** `render.render_art(variant, frames, client, extra_instruction, frame_override, people_in_ad, style, ratio)` used identically in Tasks 6, 9, 10 tests. `qa.likeness_gate(path, reference_frame, client, people_in_ad)` and `QAResult(ok, problems, likeness, checked)` consistent across Tasks 7, 9. `typeset.typeset(art, headline, style, treatment, ratio, *, scrim)` → `TypesetResult(image, notes, scrimmed, overflowed)` consistent across Tasks 4, 9. `pipeline.compose(art_path, headline, style, treatment, ratio, out_path, scrim) -> (Path, list[str])` consistent across Tasks 9, 10. `ThumbResult.art_paths/style/headline/render_calls/images_billed` consistent across Tasks 9, 10, 12. `BatchOutcome.frames/people_in_ad/work_dir` consistent across Tasks 9, 10. `models.clean_headline_text`, `HeadlineOptions`, `prompts.headline_prompt`, `plan.suggest_headlines` consistent across Tasks 11, 12.
