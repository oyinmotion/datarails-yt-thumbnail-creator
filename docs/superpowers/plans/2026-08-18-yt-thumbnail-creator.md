# YT Thumbnail Creator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Streamlit tool where a Datarails marketer pastes a Google Drive link to a video ad and receives five finished 1920×1080 YouTube ad thumbnails in one batch.

**Architecture:** A linear pipeline of small, single-purpose modules. `drive` fetches the ad, `probe` extracts frames and audio with ffmpeg, `plan` asks a text model for five hook/treatment variants as validated JSON, `render` makes five concurrent `gpt-image-2` calls using the ad's own frames plus the locked house-style thumbnails as references, `postprocess` downscales each render to exactly 1920×1080, and `qa` verifies the headline is actually legible before the grid appears. No module knows about Streamlit; `app.py` holds all UI and nothing else.

**Tech Stack:** Python 3.14, Streamlit 1.58, OpenAI Python SDK, Pydantic v2, Pillow, ffmpeg (system binary), google-api-python-client + google-auth-oauthlib, pytest.

**Spec:** `docs/superpowers/specs/2026-08-18-yt-thumbnail-creator-design.md`

## Global Constraints

Every task's requirements implicitly include this section.

- **Python only.** Datarails Data team policy. No JS/TS build step anywhere.
- **Final output is exactly 1920×1080.** Generated at `2048x1152` and downscaled with Lanczos. `gpt-image-2` requires both edges to be multiples of 16 and 1080 is not (67.5), so 1920×1080 cannot be requested directly.
- **File size ≤ 2 MB** (YouTube's thumbnail cap). On overflow, re-encode as JPEG quality 92.
- **Exactly 5 variants per batch**, using the fixed hook/treatment matrix. Never 4, never 6.
- **Headline rules:** ≤ 5 words, ≤ 22 characters where possible, no full sentences, no ending period, readable at 320px wide.
- **No invented claims or statistics.** Hooks come from what the ad actually says or shows, or from the user's context field.
- **Model IDs live only in `src/config.py`.** No model string appears anywhere else in the codebase.
- **Access is limited to `@datarails.com`.**
- **No secrets in the repo.** `OPENAI_API_KEY`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `REDIRECT_URI` come from `st.secrets` / environment only.
- **Prompt text lives in `prompts/*.md`**, never as inline Python strings, so the creative team can tune copy rules without touching code.
- **Every API call is mocked in the automated test suite.** Real calls happen only in `scripts/live_run.py`.

## File Structure

| File | Responsibility |
|------|----------------|
| `src/config.py` | Model IDs, sizes, thresholds, paths. The only place these live. |
| `src/models.py` | `Variant`, `BatchPlan`, the fixed `MATRIX`. Pure data, no I/O. |
| `src/probe.py` | ffmpeg only: video → frames + audio. No network, no AI. |
| `src/drive.py` | Drive URL parsing (pure) + download and upload (network). |
| `src/plan.py` | Transcription + planner call → validated `BatchPlan`. |
| `src/refs.py` | Owns `refs/style/` and `refs/winners/`; picks references per call. |
| `src/render.py` | One variant → raw image bytes from `gpt-image-2`. |
| `src/postprocess.py` | Raw bytes → exactly 1920×1080 file under 2 MB. Pure Pillow. |
| `src/qa.py` | Hard checks + legibility verification. |
| `src/pipeline.py` | Orchestration, concurrency, partial-failure handling. |
| `src/auth.py` | Google OAuth flow, domain gate, credential handling. |
| `app.py` | Streamlit UI only. |
| `prompts/planner.md` | Hook strategy and copy rules. |
| `prompts/render.md` | House-style render prompt template. |
| `prompts/qa_legibility.md` | The transcribe-what-you-see prompt. |
| `scripts/live_run.py` | Manual real-API run against the sample ad. |

---

### Task 1: Project scaffold, config, and its own venv

**Files:**
- Create: `.venv/` (via command), `requirements.txt`, `packages.txt`, `pytest.ini`, `src/__init__.py`, `src/config.py`, `tests/__init__.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `src.config` module exposing `PLANNER_MODEL`, `QA_MODEL`, `TRANSCRIBE_MODEL`, `IMAGE_MODEL`, `GEN_SIZE`, `FINAL_W`, `FINAL_H`, `MAX_BYTES`, `JPEG_FALLBACK_QUALITY`, `MAX_FRAMES`, `SCENE_THRESHOLD`, `MIN_SCENE_FRAMES`, `FRAME_WIDTH`, `PROJECT_ROOT`, `PROMPTS_DIR`, `REFS_STYLE_DIR`, `REFS_WINNERS_DIR`, `ALLOWED_EMAIL_DOMAIN`, `GOOGLE_SCOPES`.

- [ ] **Step 1: Create the project venv**

`python3` on this machine currently resolves to another project's venv (`dr-marketing-projects/.venv`), which would silently install into the wrong place. Use the absolute system interpreter.

```bash
cd "/Users/omeryadgar/Desktop/dev_projects/YT Thumbnail creator"
/opt/homebrew/bin/python3 -m venv .venv || /usr/bin/python3 -m venv .venv
.venv/bin/python --version
```

Expected: `Python 3.1x.x`. Every later command uses `.venv/bin/python` and `.venv/bin/pytest` explicitly.

- [ ] **Step 2: Write `requirements.txt`**

```
streamlit>=1.58
openai>=2.0
pydantic>=2.9
pillow>=11.0
google-api-python-client>=2.140
google-auth-oauthlib>=1.2
pytest>=8.0
```

- [ ] **Step 3: Write `packages.txt`**

This is how Streamlit Community Cloud installs system binaries.

```
ffmpeg
```

- [ ] **Step 4: Install dependencies**

```bash
.venv/bin/pip install -q -r requirements.txt && .venv/bin/python -c "import openai, streamlit, pydantic, PIL; print('deps ok')"
```

Expected: `deps ok`

- [ ] **Step 5: Write `pytest.ini`**

```ini
[pytest]
testpaths = tests
addopts = -q
```

- [ ] **Step 6: Write the failing test**

`tests/test_config.py`:

```python
from src import config


def test_generation_size_edges_are_multiples_of_16():
    """gpt-image-2 rejects any edge that is not a multiple of 16."""
    w, h = (int(v) for v in config.GEN_SIZE.split("x"))
    assert w % 16 == 0
    assert h % 16 == 0


def test_generation_size_is_16_by_9():
    w, h = (int(v) for v in config.GEN_SIZE.split("x"))
    assert round(w / h, 4) == round(16 / 9, 4)


def test_final_dimensions_are_1920_by_1080():
    assert (config.FINAL_W, config.FINAL_H) == (1920, 1080)


def test_generation_size_is_larger_than_final_so_we_downscale():
    w, h = (int(v) for v in config.GEN_SIZE.split("x"))
    assert w > config.FINAL_W and h > config.FINAL_H


def test_max_bytes_is_youtube_thumbnail_cap():
    assert config.MAX_BYTES == 2 * 1024 * 1024


def test_drive_scope_allows_creating_folders_beside_the_source_ad():
    """drive.file cannot create a subfolder in a folder the app did not create."""
    assert "https://www.googleapis.com/auth/drive" in config.GOOGLE_SCOPES
```

- [ ] **Step 7: Run the test to verify it fails**

```bash
.venv/bin/pytest tests/test_config.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.config'`

- [ ] **Step 8: Write `src/config.py` and empty `__init__.py` files**

```bash
touch src/__init__.py tests/__init__.py
```

`src/config.py`:

```python
"""Single source of truth for models, sizes, and paths.

Model IDs appear ONLY in this file. Refreshing to a newer model is a one-line
change here and nowhere else.
"""

from pathlib import Path

# --- Models (verified against OpenAI's model list, 2026-08-18) -------------
PLANNER_MODEL = "gpt-5.6-sol"      # hooks and copy: the highest-leverage output
QA_MODEL = "gpt-5.6-terra"         # mechanical legibility read: cheaper tier
TRANSCRIBE_MODEL = "gpt-transcribe"
IMAGE_MODEL = "gpt-image-2"

# --- Output geometry -------------------------------------------------------
# gpt-image-2 requires both edges to be multiples of 16. 1080 / 16 = 67.5, so
# 1920x1080 cannot be requested. Generate the nearest native 16:9 size that
# satisfies the constraint, then downscale. Supersampling also sharpens type.
GEN_SIZE = "2048x1152"
FINAL_W = 1920
FINAL_H = 1080
MAX_BYTES = 2 * 1024 * 1024        # YouTube thumbnail cap
JPEG_FALLBACK_QUALITY = 92
IMAGE_QUALITY = "high"

# --- Frame extraction ------------------------------------------------------
MAX_FRAMES = 16
SCENE_THRESHOLD = 0.3
MIN_SCENE_FRAMES = 6               # below this, fall back to interval sampling
FRAME_WIDTH = 1280

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
REFS_STYLE_DIR = PROJECT_ROOT / "refs" / "style"
REFS_WINNERS_DIR = PROJECT_ROOT / "refs" / "winners"

# --- Access ----------------------------------------------------------------
ALLOWED_EMAIL_DOMAIN = "datarails.com"
GOOGLE_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    # Full drive scope, not drive.file: saving results means creating a
    # subfolder inside a folder this app did not create.
    "https://www.googleapis.com/auth/drive",
]
```

- [ ] **Step 9: Run the tests to verify they pass**

```bash
.venv/bin/pytest tests/test_config.py -v
```

Expected: 6 passed

- [ ] **Step 10: Commit**

```bash
git add requirements.txt packages.txt pytest.ini src/ tests/
git commit -m "feat: project scaffold and config with verified geometry constraints"
```

---

### Task 2: `probe.py` — frames and audio via ffmpeg

**Files:**
- Create: `src/probe.py`, `tests/test_probe.py`
- Test fixture: `Sample input ad/claude-vs-claude-with-fos-v2_vid_16x9_47s_high_bus_awa_skit_aic_hall_rhal_mul_n_y_fos_cld_aif.mp4` (already present, 1920×1080, 25fps, 46.6s)

**Interfaces:**
- Consumes: `src.config` (`MAX_FRAMES`, `SCENE_THRESHOLD`, `MIN_SCENE_FRAMES`, `FRAME_WIDTH`).
- Produces:
  - `ProbeError(RuntimeError)`
  - `video_duration(video: Path) -> float`
  - `extract_frames(video: Path, out_dir: Path, max_frames: int = MAX_FRAMES) -> list[Path]`
  - `extract_audio(video: Path, out_dir: Path) -> Path`

- [ ] **Step 1: Write the failing test**

These tests run real ffmpeg against the real sample ad — this module's whole job is shelling out correctly, and mocking `subprocess` would test nothing.

`tests/test_probe.py`:

```python
from pathlib import Path

import pytest
from PIL import Image

from src import probe
from src.config import PROJECT_ROOT

SAMPLE = next((PROJECT_ROOT / "Sample input ad").glob("*.mp4"), None)
needs_sample = pytest.mark.skipif(SAMPLE is None, reason="sample ad not present")


@needs_sample
def test_video_duration_matches_the_sample_ad():
    assert probe.video_duration(SAMPLE) == pytest.approx(46.6, abs=0.5)


@needs_sample
def test_extract_frames_returns_real_jpegs_capped_at_max(tmp_path):
    frames = probe.extract_frames(SAMPLE, tmp_path, max_frames=8)
    assert 1 <= len(frames) <= 8
    for f in frames:
        assert f.exists() and f.stat().st_size > 0
        with Image.open(f) as im:
            assert im.format == "JPEG"
            assert im.width == 1280


@needs_sample
def test_extract_frames_are_distinct_moments(tmp_path):
    """Scene selection must not return the same frame repeatedly."""
    frames = probe.extract_frames(SAMPLE, tmp_path, max_frames=6)
    digests = {f.read_bytes()[:2048] for f in frames}
    assert len(digests) == len(frames)


@needs_sample
def test_extract_audio_produces_a_small_mono_file(tmp_path):
    audio = probe.extract_audio(SAMPLE, tmp_path)
    assert audio.exists()
    assert audio.stat().st_size > 1024
    # Mono 16kHz 32kbps over ~47s must be far smaller than the 86MB source.
    assert audio.stat().st_size < 1_000_000


def test_missing_video_raises_probe_error(tmp_path):
    with pytest.raises(probe.ProbeError):
        probe.extract_frames(tmp_path / "nope.mp4", tmp_path)


def test_interval_timestamps_are_evenly_spread_and_inside_the_video():
    ts = probe._interval_timestamps(duration=40.0, count=4)
    assert len(ts) == 4
    assert all(0 < t < 40.0 for t in ts)
    assert ts == sorted(ts)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/pytest tests/test_probe.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.probe'`

- [ ] **Step 3: Write `src/probe.py`**

```python
"""ffmpeg wrappers. No network, no AI, no Streamlit."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .config import FRAME_WIDTH, MAX_FRAMES, MIN_SCENE_FRAMES, SCENE_THRESHOLD


class ProbeError(RuntimeError):
    """Raised when ffmpeg/ffprobe cannot read the video."""


def _binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise ProbeError(
            f"{name} is not installed. On Streamlit Cloud this comes from "
            "packages.txt; locally, `brew install ffmpeg`."
        )
    return path


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ProbeError(
            f"Couldn't read that video file. ({cmd[0]} said: "
            f"{result.stderr.strip()[:300]})"
        )
    return result.stdout


def video_duration(video: Path) -> float:
    if not Path(video).exists():
        raise ProbeError("That video file doesn't exist.")
    out = _run([
        _binary("ffprobe"), "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video),
    ])
    try:
        return float(out.strip())
    except ValueError as exc:
        raise ProbeError("That file doesn't look like a video.") from exc


def _interval_timestamps(duration: float, count: int) -> list[float]:
    """Evenly spread timestamps, avoiding the very first and last frame."""
    step = duration / (count + 1)
    return [round(step * (i + 1), 3) for i in range(count)]


def extract_frames(
    video: Path, out_dir: Path, max_frames: int = MAX_FRAMES
) -> list[Path]:
    """Scene-change frames, falling back to even sampling on flat footage."""
    video = Path(video)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not video.exists():
        raise ProbeError("That video file doesn't exist.")

    _run([
        _binary("ffmpeg"), "-v", "error", "-i", str(video),
        "-vf", f"select='gt(scene,{SCENE_THRESHOLD})',scale={FRAME_WIDTH}:-2",
        "-vsync", "vfr", "-q:v", "3",
        str(out_dir / "scene_%03d.jpg"),
    ])
    frames = sorted(out_dir.glob("scene_*.jpg"))

    if len(frames) < MIN_SCENE_FRAMES:
        # Flat footage (one continuous shot): sample by time instead.
        for f in frames:
            f.unlink()
        duration = video_duration(video)
        for i, ts in enumerate(_interval_timestamps(duration, max_frames)):
            _run([
                _binary("ffmpeg"), "-v", "error", "-ss", str(ts),
                "-i", str(video), "-frames:v", "1",
                "-vf", f"scale={FRAME_WIDTH}:-2", "-q:v", "3",
                str(out_dir / f"interval_{i:03d}.jpg"),
            ])
        frames = sorted(out_dir.glob("interval_*.jpg"))

    if not frames:
        raise ProbeError("Couldn't pull any frames out of that video.")

    # Keep an even spread rather than only the first N.
    if len(frames) > max_frames:
        stride = len(frames) / max_frames
        keep = {frames[int(i * stride)] for i in range(max_frames)}
        for f in frames:
            if f not in keep:
                f.unlink()
        frames = sorted(keep)

    return frames


def extract_audio(video: Path, out_dir: Path) -> Path:
    """Mono 16kHz m4a — small enough to upload for transcription."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio = out_dir / "audio.m4a"
    _run([
        _binary("ffmpeg"), "-v", "error", "-y", "-i", str(video),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "aac", "-b:a", "32k",
        str(audio),
    ])
    if not audio.exists() or audio.stat().st_size == 0:
        raise ProbeError("That video has no usable audio track.")
    return audio
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/pytest tests/test_probe.py -v
```

Expected: 6 passed. If `test_extract_frames_are_distinct_moments` fails, the scene threshold is too low for this footage — raise `SCENE_THRESHOLD` in config to `0.4` and rerun.

- [ ] **Step 5: Commit**

```bash
git add src/probe.py tests/test_probe.py
git commit -m "feat: frame and audio extraction with scene detection and interval fallback"
```

---

### Task 3: `models.py` — the plan schema and the fixed matrix

**Files:**
- Create: `src/models.py`, `tests/test_models.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `HookType = Literal["stat", "question", "conflict", "pain", "outcome"]`
  - `Treatment = Literal["split_screen", "face_closeup", "full_bleed", "text_dominant", "product_forward"]`
  - `MATRIX: list[tuple[int, str, str]]` — the five fixed `(index, hook_type, treatment)` rows
  - `Variant` with fields `index, hook_type, treatment, headline, frame_id, second_frame_id, scene_direction, rationale`
  - `BatchPlan` with fields `ad_summary, transcript_used, variants`
  - `BatchPlan.validate_matrix() -> None` raising `ValueError` on any deviation

- [ ] **Step 1: Write the failing test**

`tests/test_models.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/pytest tests/test_models.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.models'`

- [ ] **Step 3: Write `src/models.py`**

```python
"""Plan data model. Pure data — no I/O, no API calls.

The hook/treatment pairing is fixed in code, not chosen by the model, so every
batch spans the space instead of clustering on one idea.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

HookType = Literal["stat", "question", "conflict", "pain", "outcome"]
Treatment = Literal[
    "split_screen", "face_closeup", "full_bleed", "text_dominant",
    "product_forward",
]

# (index, hook_type, treatment) — locked pairing.
MATRIX: list[tuple[int, HookType, Treatment]] = [
    (1, "stat", "split_screen"),
    (2, "question", "face_closeup"),
    (3, "conflict", "full_bleed"),
    (4, "pain", "text_dominant"),
    (5, "outcome", "product_forward"),
]

TREATMENT_BRIEF: dict[str, str] = {
    "split_screen": (
        "Both actors face off, one on each side, a hard vertical seam of light "
        "between them. Headline centered on the seam."
    ),
    "face_closeup": (
        "One actor's face fills roughly half the frame with a clear reaction. "
        "Headline stacked in the remaining space."
    ),
    "full_bleed": (
        "A single dramatic energy burst fills the frame behind both actors. "
        "Headline centered and dominant."
    ),
    "text_dominant": (
        "Typography carries the frame; the actor is smaller and offset to one "
        "side. The headline is the subject."
    ),
    "product_forward": (
        "The FinanceOS product surface or its mark is visible and legible, with "
        "one actor presenting it. Headline supports rather than competes."
    ),
}

MAX_HEADLINE_WORDS = 5


class Variant(BaseModel):
    index: int = Field(ge=1, le=5)
    hook_type: HookType
    treatment: Treatment
    headline: str
    frame_id: str
    second_frame_id: str | None = None
    scene_direction: str
    rationale: str

    @field_validator("headline")
    @classmethod
    def clean_headline(cls, v: str) -> str:
        v = " ".join(v.split()).rstrip(".").strip().upper()
        if not v:
            raise ValueError("headline cannot be empty")
        if len(v.split()) > MAX_HEADLINE_WORDS:
            raise ValueError(
                f"headline must be {MAX_HEADLINE_WORDS} words or fewer, got "
                f"{len(v.split())}: {v!r}"
            )
        return v


class BatchPlan(BaseModel):
    ad_summary: str
    transcript_used: bool
    variants: list[Variant] = Field(min_length=5, max_length=5)

    def validate_matrix(self) -> None:
        """Fail loudly if the model drifted off the locked pairing."""
        actual = [(v.index, v.hook_type, v.treatment) for v in self.variants]
        if sorted(actual) != sorted(MATRIX):
            raise ValueError(
                f"plan does not follow the locked matrix.\n"
                f"expected: {sorted(MATRIX)}\ngot:      {sorted(actual)}"
            )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/pytest tests/test_models.py -v
```

Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/models.py tests/test_models.py
git commit -m "feat: batch plan schema with locked hook/treatment matrix"
```

---

### Task 4: `drive.py` — link parsing and file transfer

**Files:**
- Create: `src/drive.py`, `tests/test_drive.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `DriveError(RuntimeError)`, `DriveLinkError(DriveError)`
  - `parse_file_id(url: str) -> str`
  - `fetch_video(file_id: str, creds, dest_dir: Path) -> tuple[Path, str]` returning `(local_path, parent_folder_id)`
  - `save_batch(files: list[Path], parent_id: str, folder_name: str, creds) -> str` returning the new folder's web URL

- [ ] **Step 1: Write the failing test**

Link parsing is pure logic and gets real tests. The two network functions get a fake Drive service, because testing Google's transport is not our job.

`tests/test_drive.py`:

```python
import pytest

from src import drive


@pytest.mark.parametrize("url,expected", [
    ("https://drive.google.com/file/d/1AbC_dEfGhIjKlMnOpQr/view?usp=sharing",
     "1AbC_dEfGhIjKlMnOpQr"),
    ("https://drive.google.com/file/d/1AbC_dEfGhIjKlMnOpQr/view",
     "1AbC_dEfGhIjKlMnOpQr"),
    ("https://drive.google.com/open?id=1AbC_dEfGhIjKlMnOpQr",
     "1AbC_dEfGhIjKlMnOpQr"),
    ("https://drive.google.com/uc?export=download&id=1AbC_dEfGhIjKlMnOpQr",
     "1AbC_dEfGhIjKlMnOpQr"),
    ("https://drive.google.com/drive/u/0/file/d/1AbC_dEfGhIjKlMnOpQr/view",
     "1AbC_dEfGhIjKlMnOpQr"),
    ("  https://drive.google.com/file/d/1AbC_dEfGhIjKlMnOpQr/view  ",
     "1AbC_dEfGhIjKlMnOpQr"),
    ("1AbC_dEfGhIjKlMnOpQrStUvWxYz123456", "1AbC_dEfGhIjKlMnOpQrStUvWxYz123456"),
])
def test_parse_file_id_handles_every_drive_link_form(url, expected):
    assert drive.parse_file_id(url) == expected


@pytest.mark.parametrize("bad", [
    "",
    "   ",
    "not a url at all",
    "https://example.com/file/d/1AbC_dEfGhIjKlMnOpQr/view",
    "https://drive.google.com/drive/folders/1AbC_dEfGhIjKlMnOpQr",
])
def test_parse_file_id_rejects_bad_input(bad):
    with pytest.raises(drive.DriveLinkError):
        drive.parse_file_id(bad)


def test_folder_link_error_names_the_problem():
    with pytest.raises(drive.DriveLinkError, match="folder"):
        drive.parse_file_id("https://drive.google.com/drive/folders/1AbC_dEfGh")


def test_fetch_video_rejects_non_video_mime(monkeypatch, tmp_path):
    class FakeFiles:
        def get(self, **kwargs):
            class R:
                def execute(self_inner):
                    return {"id": "x", "name": "brief.pdf",
                            "mimeType": "application/pdf", "parents": ["p1"]}
            return R()

    class FakeService:
        def files(self):
            return FakeFiles()

    monkeypatch.setattr(drive, "_service", lambda creds: FakeService())
    with pytest.raises(drive.DriveError, match="not a video"):
        drive.fetch_video("x", creds=object(), dest_dir=tmp_path)


def test_fetch_video_permission_error_is_human_readable(monkeypatch, tmp_path):
    class FakeFiles:
        def get(self, **kwargs):
            class R:
                def execute(self_inner):
                    raise drive.HttpError(
                        resp=type("R", (), {"status": 404, "reason": "Not Found"})(),
                        content=b"{}",
                    )
            return R()

    class FakeService:
        def files(self):
            return FakeFiles()

    monkeypatch.setattr(drive, "_service", lambda creds: FakeService())
    with pytest.raises(drive.DriveError, match="access"):
        drive.fetch_video("x", creds=object(), dest_dir=tmp_path)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/pytest tests/test_drive.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.drive'`

- [ ] **Step 3: Write `src/drive.py`**

```python
"""Google Drive access: parse a link, download the ad, upload the results."""

from __future__ import annotations

import io
import re
from pathlib import Path

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload


class DriveError(RuntimeError):
    """Any Drive failure, phrased for a marketer rather than a developer."""


class DriveLinkError(DriveError):
    """The pasted text is not a usable Drive file link."""


_ID = r"([A-Za-z0-9_-]{10,})"
_PATTERNS = [
    re.compile(rf"/file/d/{_ID}"),
    re.compile(rf"[?&]id={_ID}"),
]
_BARE_ID = re.compile(r"^[A-Za-z0-9_-]{25,}$")


def parse_file_id(url: str) -> str:
    text = (url or "").strip()
    if not text:
        raise DriveLinkError("Paste a Google Drive link to the ad first.")

    if "/drive/folders/" in text:
        raise DriveLinkError(
            "That's a link to a folder, not a video. Open the ad itself and "
            "copy its link."
        )

    if _BARE_ID.match(text):
        return text

    if "drive.google.com" not in text and "docs.google.com" not in text:
        raise DriveLinkError(
            "That doesn't look like a Google Drive link."
        )

    for pattern in _PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)

    raise DriveLinkError(
        "Couldn't find a file ID in that link. Use the 'Copy link' option in "
        "Drive."
    )


def _service(creds):
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def fetch_video(file_id: str, creds, dest_dir: Path) -> tuple[Path, str]:
    """Download the ad. Returns (local path, the folder it lives in)."""
    service = _service(creds)
    try:
        meta = service.files().get(
            fileId=file_id,
            fields="id,name,mimeType,parents,size",
            supportsAllDrives=True,
        ).execute()
    except HttpError as exc:
        raise DriveError(
            "You don't have access to that file, or the link is wrong."
        ) from exc

    mime = meta.get("mimeType", "")
    if not mime.startswith("video/"):
        raise DriveError(
            f"That link points to a {mime.split('/')[-1] or 'file'}, not a "
            "video."
        )

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    local = dest_dir / meta["name"]

    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    with local.open("wb") as handle:
        downloader = MediaIoBaseDownload(handle, request, chunksize=8 * 1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk()

    parents = meta.get("parents") or []
    return local, (parents[0] if parents else "root")


def save_batch(
    files: list[Path], parent_id: str, folder_name: str, creds
) -> str:
    """Create a subfolder beside the ad and upload the thumbnails into it."""
    service = _service(creds)
    try:
        folder = service.files().create(
            body={
                "name": folder_name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [parent_id],
            },
            fields="id,webViewLink",
            supportsAllDrives=True,
        ).execute()

        for path in files:
            mime = "image/png" if path.suffix == ".png" else "image/jpeg"
            service.files().create(
                body={"name": path.name, "parents": [folder["id"]]},
                media_body=MediaFileUpload(str(path), mimetype=mime),
                fields="id",
                supportsAllDrives=True,
            ).execute()
    except HttpError as exc:
        raise DriveError(
            "Couldn't save to Drive — you may not have edit access to that "
            "folder. The thumbnails are still downloadable."
        ) from exc

    return folder.get("webViewLink", "")
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/pytest tests/test_drive.py -v
```

Expected: 14 passed

- [ ] **Step 5: Commit**

```bash
git add src/drive.py tests/test_drive.py
git commit -m "feat: drive link parsing, video download, and batch save-back"
```

---

### Task 5: `postprocess.py` — exactly 1920×1080, under 2 MB

**Files:**
- Create: `src/postprocess.py`, `tests/test_postprocess.py`

**Interfaces:**
- Consumes: `src.config` (`FINAL_W`, `FINAL_H`, `MAX_BYTES`, `JPEG_FALLBACK_QUALITY`).
- Produces: `finalize(image_bytes: bytes, out_path: Path) -> Path` — returns the actual written path, which may have a `.jpg` suffix instead of `.png`.

This is the task that makes the 1920×1080 requirement true, and it needs no API access at all, so it gets thorough real tests.

- [ ] **Step 1: Write the failing test**

`tests/test_postprocess.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/pytest tests/test_postprocess.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.postprocess'`

- [ ] **Step 3: Write `src/postprocess.py`**

```python
"""Turn raw model output into a deliverable YouTube thumbnail file.

Two jobs: land on exactly 1920x1080, and stay under YouTube's 2MB cap.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from .config import FINAL_H, FINAL_W, JPEG_FALLBACK_QUALITY, MAX_BYTES


def finalize(image_bytes: bytes, out_path: Path) -> Path:
    """Downscale to 1920x1080 and write, falling back to JPEG if oversize.

    Returns the path actually written — the suffix may differ from out_path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(io.BytesIO(image_bytes)) as raw:
        image = raw.convert("RGB")
        if image.size != (FINAL_W, FINAL_H):
            # Lanczos downscale from 2048x1152: exact ratio, no crop, and the
            # supersampling visibly sharpens the headline type.
            image = image.resize((FINAL_W, FINAL_H), Image.LANCZOS)

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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/pytest tests/test_postprocess.py -v
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/postprocess.py tests/test_postprocess.py
git commit -m "feat: finalize renders to exactly 1920x1080 under YouTube's 2MB cap"
```

---

### Task 6: `refs.py` and the locked style pack

**Files:**
- Create: `src/refs.py`, `tests/test_refs.py`, `refs/style/` (populated), `refs/winners/.gitkeep`
- Copy from: `sample outcomes/*.png` → `refs/style/`

**Interfaces:**
- Consumes: `src.config` (`REFS_STYLE_DIR`, `REFS_WINNERS_DIR`).
- Produces:
  - `style_refs() -> list[Path]`
  - `winner_refs(treatment: str | None = None) -> list[Path]`
  - `pick_refs(treatment: str, limit: int = 3) -> list[Path]`
  - `save_winner(image_path: Path, treatment: str) -> Path`

- [ ] **Step 1: Populate the style pack**

The four approved thumbnails become the locked reference pack. They are committed, unlike the 86 MB source video.

```bash
cd "/Users/omeryadgar/Desktop/dev_projects/YT Thumbnail creator"
mkdir -p refs/style refs/winners
cp "sample outcomes/"*.png refs/style/
touch refs/winners/.gitkeep
ls -la refs/style/
```

Expected: four PNGs listed.

- [ ] **Step 2: Un-ignore the refs directory**

`.gitignore` currently has no rule against these, but confirm they are trackable and that `sample outcomes/` staying untracked is fine:

```bash
git check-ignore -v refs/style/*.png || echo "refs are trackable"
```

Expected: `refs are trackable`

- [ ] **Step 3: Write the failing test**

`tests/test_refs.py`:

```python
import shutil

import pytest

from src import refs


@pytest.fixture
def isolated_refs(tmp_path, monkeypatch):
    style = tmp_path / "style"
    winners = tmp_path / "winners"
    style.mkdir()
    winners.mkdir()
    for name in ("a.png", "b.png", "c.png", "d.png"):
        (style / name).write_bytes(b"\x89PNG fake")
    monkeypatch.setattr(refs, "REFS_STYLE_DIR", style)
    monkeypatch.setattr(refs, "REFS_WINNERS_DIR", winners)
    return style, winners


def test_style_refs_finds_the_locked_pack(isolated_refs):
    assert len(refs.style_refs()) == 4


def test_real_style_pack_is_populated():
    """The four approved samples must actually be in the repo."""
    assert len(refs.style_refs()) >= 4


def test_pick_refs_always_includes_at_least_one_locked_style_ref(isolated_refs):
    style, winners = isolated_refs
    for i in range(6):
        (winners / f"split_screen__{i}.png").write_bytes(b"\x89PNG fake")
    picked = refs.pick_refs("split_screen", limit=3)
    assert len(picked) == 3
    assert any(p.parent == style for p in picked), (
        "style drift guard: a locked reference must always be in the mix"
    )


def test_pick_refs_prefers_winners_of_the_same_treatment(isolated_refs):
    _, winners = isolated_refs
    (winners / "split_screen__1.png").write_bytes(b"\x89PNG fake")
    (winners / "face_closeup__1.png").write_bytes(b"\x89PNG fake")
    picked = refs.pick_refs("split_screen", limit=3)
    names = [p.name for p in picked]
    assert "split_screen__1.png" in names
    assert "face_closeup__1.png" not in names


def test_pick_refs_respects_the_limit(isolated_refs):
    assert len(refs.pick_refs("full_bleed", limit=2)) == 2


def test_save_winner_tags_the_file_with_its_treatment(isolated_refs, tmp_path):
    src_file = tmp_path / "chosen.png"
    src_file.write_bytes(b"\x89PNG fake")
    saved = refs.save_winner(src_file, "text_dominant")
    assert saved.name.startswith("text_dominant__")
    assert saved.exists()
    assert refs.winner_refs("text_dominant") == [saved]
```

- [ ] **Step 4: Run the test to verify it fails**

```bash
.venv/bin/pytest tests/test_refs.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.refs'`

- [ ] **Step 5: Write `src/refs.py`**

```python
"""The house style pack.

refs/style/ is locked: the four approved thumbnails, always contributing at
least one reference to every render call. refs/winners/ grows as the team
approves outputs, which tightens the tool's taste over time. The guarantee that
a locked ref is always present is what bounds style drift.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from .config import REFS_STYLE_DIR, REFS_WINNERS_DIR

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in _IMAGE_SUFFIXES
    )


def style_refs() -> list[Path]:
    return _images(REFS_STYLE_DIR)


def winner_refs(treatment: str | None = None) -> list[Path]:
    winners = _images(REFS_WINNERS_DIR)
    if treatment is None:
        return winners
    return [p for p in winners if p.name.startswith(f"{treatment}__")]


def pick_refs(treatment: str, limit: int = 3) -> list[Path]:
    """References for one render call: matching winners first, style always."""
    style = style_refs()
    if not style:
        raise FileNotFoundError(
            f"No style references found in {REFS_STYLE_DIR}. Copy the approved "
            "thumbnails there before rendering."
        )

    picked: list[Path] = [style[0]]                    # drift guard
    for candidate in winner_refs(treatment) + style[1:] + winner_refs():
        if len(picked) >= limit:
            break
        if candidate not in picked:
            picked.append(candidate)
    return picked[:limit]


def save_winner(image_path: Path, treatment: str) -> Path:
    REFS_WINNERS_DIR.mkdir(parents=True, exist_ok=True)
    dest = REFS_WINNERS_DIR / f"{treatment}__{uuid.uuid4().hex[:8]}{Path(image_path).suffix}"
    shutil.copy2(image_path, dest)
    return dest
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
.venv/bin/pytest tests/test_refs.py -v
```

Expected: 6 passed

- [ ] **Step 7: Commit**

```bash
git add src/refs.py tests/test_refs.py refs/
git commit -m "feat: locked style pack plus growing winners pack with drift guard"
```

---

### Task 7: The prompt files

**Files:**
- Create: `prompts/planner.md`, `prompts/render.md`, `prompts/qa_legibility.md`, `src/prompts.py`, `tests/test_prompts.py`

**Interfaces:**
- Consumes: `src.config` (`PROMPTS_DIR`), `src.models` (`TREATMENT_BRIEF`).
- Produces:
  - `load(name: str) -> str`
  - `render_prompt(variant: Variant) -> str`
  - `planner_prompt(transcript: str | None, context: str | None, headline_override: str | None) -> str`

Prompts live as files so the creative team can tune copy rules without touching Python. `src/prompts.py` only loads and fills them.

- [ ] **Step 1: Write the failing test**

`tests/test_prompts.py`:

```python
import pytest

from src import prompts
from src.models import MATRIX, Variant


def _variant(treatment="split_screen", headline="SAME AI DIFFERENT ANSWER"):
    return Variant(
        index=1, hook_type="stat", treatment=treatment, headline=headline,
        frame_id="scene_001.jpg", second_frame_id=None,
        scene_direction="hard vertical seam of light between them",
        rationale="the ad contrasts two answers",
    )


def test_all_three_prompt_files_load():
    for name in ("planner", "render", "qa_legibility"):
        assert len(prompts.load(name)) > 200


def test_missing_prompt_file_raises():
    with pytest.raises(FileNotFoundError):
        prompts.load("does_not_exist")


def test_render_prompt_contains_the_exact_headline_in_quotes():
    text = prompts.render_prompt(_variant())
    assert '"SAME AI DIFFERENT ANSWER"' in text


def test_render_prompt_carries_the_treatment_brief():
    text = prompts.render_prompt(_variant(treatment="product_forward"))
    assert "FinanceOS" in text


def test_render_prompt_forbids_extra_text():
    text = prompts.render_prompt(_variant()).lower()
    assert "no other text" in text
    assert "do not add" in text


def test_render_prompt_has_no_unfilled_placeholders():
    text = prompts.render_prompt(_variant())
    assert "{" not in text and "}" not in text


def test_planner_prompt_states_the_headline_word_limit():
    text = prompts.planner_prompt(transcript="hello", context=None,
                                  headline_override=None)
    assert "five words" in text.lower()


def test_planner_prompt_includes_every_matrix_row():
    text = prompts.planner_prompt(transcript=None, context=None,
                                  headline_override=None)
    for _, hook, treatment in MATRIX:
        assert hook in text
        assert treatment in text


def test_planner_prompt_notes_missing_transcript():
    text = prompts.planner_prompt(transcript=None, context=None,
                                  headline_override=None)
    assert "no transcript" in text.lower()


def test_planner_prompt_passes_through_override_and_context():
    text = prompts.planner_prompt(
        transcript="t", context="targeting CFOs", headline_override="ONE TRUTH",
    )
    assert "ONE TRUTH" in text
    assert "targeting CFOs" in text
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/pytest tests/test_prompts.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.prompts'`

- [ ] **Step 3: Write `prompts/planner.md`**

```markdown
You are the creative director for Datarails, a financial planning platform. You
are looking at frames from one of our YouTube video ads, plus what is said in
it. Your job is to plan five thumbnail concepts for that ad.

Thumbnails for YouTube in-feed and Demand Gen placements are the only creative a
viewer evaluates before deciding to click. The headline is the whole game.

## Headline rules

- Five words maximum. Twenty-two characters or fewer where you can manage it.
- Not a sentence. No ending period. No quotation marks.
- Must be readable at 320 pixels wide, the size it actually appears in-feed.
- Curiosity, not clickbait: the first seconds of the ad must deliver whatever the
  thumbnail promises.
- Never invent a claim or a statistic. Use only numbers that appear in the ad
  itself or in the extra context below. If neither has a number, frame the stat
  concept as a comparison without a figure.
- Plain finance language a CFO would use. No jargon, no exclamation marks.

## The five concepts

You must return exactly five variants, one per row, using these exact pairings:

| index | hook_type | treatment | what it is |
|---|---|---|---|
| 1 | stat | split_screen | a number or hard comparison, both actors facing off |
| 2 | question | face_closeup | a question one actor's face is already asking |
| 3 | conflict | full_bleed | the disagreement at the heart of the ad |
| 4 | pain | text_dominant | the frustration the viewer recognizes in themselves |
| 5 | outcome | product_forward | the payoff, with the product visible |

For each variant:

- `headline` — the on-image text, following the rules above.
- `frame_id` — the filename of the frame that best suits this treatment. Pick a
  two-shot for split_screen, a clear single face for face_closeup, a frame where
  the product or screen is visible for product_forward.
- `second_frame_id` — only for split_screen, if a second frame gives a better
  second actor. Otherwise null.
- `scene_direction` — one sentence on what the background and composition should
  do. Describe light, colour, and energy, not text.
- `rationale` — one sentence on why this hook comes out of this ad.

Also return `ad_summary`: one sentence on what the ad is about.
```

- [ ] **Step 4: Write `prompts/render.md`**

```markdown
Create a YouTube ad thumbnail, 16:9, in the exact visual style of the reference
images provided.

## House style, non-negotiable

- Extremely high contrast, built to stop a scroll at thumbnail size.
- Headline set in a heavy condensed sans, all caps, pure white with a thick dark
  outline and a hard drop shadow so it separates from anything behind it.
- The people are cut out cleanly from their background with a subtle light rim
  around them, standing in front of the scene rather than inside it.
- Palette: deep navy blue and vivid orange, with hot white light where they meet.
- Cinematic dramatic lighting. Sparks, light rays, or an energy burst as
  appropriate to the direction below.

## The people

The reference images include frames from the actual ad. The people in those
frames are the people who must appear in this thumbnail. Keep their faces,
hair, facial hair, and clothing recognisably the same. Do not substitute
different people. Do not beautify or change their age.

## Layout

{treatment_brief}

## Scene direction

{scene_direction}

## The text

Render exactly this headline, and nothing else:

"{headline}"

Spell it exactly as written, all capitals, every word legible and fully inside
the frame. Do not add any other text, tagline, caption, watermark, signature,
URL, or logo. Do not add a play button or any interface element. Do not letter
the text across a person's face.
```

- [ ] **Step 5: Write `prompts/qa_legibility.md`**

```markdown
Transcribe every word of text you can read in this image.

Return only the words, in reading order, separated by single spaces. If a word
is cut off, partially hidden, or too blurry to read with confidence, do not
include it. If there is no readable text at all, return the single word NONE.

Do not describe the image. Do not add commentary.
```

- [ ] **Step 6: Write `src/prompts.py`**

```python
"""Load and fill the prompt files. No prompt text lives in Python."""

from __future__ import annotations

from functools import lru_cache

from .config import PROMPTS_DIR
from .models import MATRIX, TREATMENT_BRIEF, Variant


@lru_cache(maxsize=None)
def load(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"No prompt file at {path}")
    return path.read_text(encoding="utf-8")


def render_prompt(variant: Variant) -> str:
    return (
        load("render")
        .replace("{treatment_brief}", TREATMENT_BRIEF[variant.treatment])
        .replace("{scene_direction}", variant.scene_direction)
        .replace("{headline}", variant.headline)
    )


def planner_prompt(
    transcript: str | None,
    context: str | None,
    headline_override: str | None,
) -> str:
    parts = [load("planner")]

    if transcript:
        parts.append(f"\n## What is said in the ad\n\n{transcript.strip()}")
    else:
        parts.append(
            "\n## What is said in the ad\n\nNo transcript is available for this "
            "ad. Plan from the frames alone."
        )

    if context:
        parts.append(f"\n## Extra context from the team\n\n{context.strip()}")

    if headline_override:
        parts.append(
            "\n## Headline is fixed\n\nUse this exact headline, unchanged, for "
            f"all five variants:\n\n{headline_override.strip()}\n\nVary only the "
            "frame choice and scene direction."
        )

    # Frame filenames are appended by plan.py, which knows what it extracted.
    return "\n".join(parts)
```

- [ ] **Step 7: Run the tests to verify they pass**

```bash
.venv/bin/pytest tests/test_prompts.py -v
```

Expected: 10 passed

- [ ] **Step 8: Commit**

```bash
git add prompts/ src/prompts.py tests/test_prompts.py
git commit -m "feat: prompt files for planning, rendering, and legibility QA"
```

---

### Task 8: `plan.py` — transcript plus frames to a validated plan

**Files:**
- Create: `src/plan.py`, `tests/test_plan.py`

**Interfaces:**
- Consumes: `src.config`, `src.models` (`BatchPlan`, `MATRIX`), `src.prompts`.
- Produces:
  - `PlanError(RuntimeError)`
  - `transcribe(audio: Path, client=None) -> str | None` — returns `None` on failure, never raises
  - `build_plan(frames: list[Path], audio: Path | None, headline_override: str | None = None, context: str | None = None, client=None) -> BatchPlan`

Every function takes an optional `client` so tests inject a fake. Transcription failure is deliberately non-fatal: the frames alone still produce a usable plan, and losing the batch over a missing audio track would be worse than a slightly weaker hook.

- [ ] **Step 1: Write the failing test**

`tests/test_plan.py`:

```python
from pathlib import Path

import pytest

from src import plan
from src.models import MATRIX, BatchPlan, Variant


def _valid_plan():
    return BatchPlan(
        ad_summary="two colleagues, same AI, different answers",
        transcript_used=True,
        variants=[
            Variant(
                index=i, hook_type=h, treatment=t,
                headline=f"HOOK {i}", frame_id="scene_001.jpg",
                second_frame_id=None, scene_direction="sparks at the seam",
                rationale="from the dialogue",
            )
            for i, h, t in MATRIX
        ],
    )


class FakeTranscriptions:
    def __init__(self, text="they disagree about the forecast", fail=False):
        self.text_value = text
        self.fail = fail

    def create(self, **kwargs):
        if self.fail:
            raise RuntimeError("transcription service unavailable")
        return type("T", (), {"text": self.text_value})()


class FakeResponses:
    def __init__(self, parsed, fail_times=0):
        self.parsed = parsed
        self.fail_times = fail_times
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("bad json")
        return type("R", (), {"output_parsed": self.parsed})()


class FakeClient:
    def __init__(self, parsed=None, fail_times=0, transcribe_fail=False):
        self.responses = FakeResponses(parsed or _valid_plan(), fail_times)
        self.audio = type(
            "A", (), {"transcriptions": FakeTranscriptions(fail=transcribe_fail)}
        )()


@pytest.fixture
def frames(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"scene_{i:03d}.jpg"
        p.write_bytes(b"\xff\xd8\xff fake jpeg")
        paths.append(p)
    return paths


def test_transcribe_returns_text(tmp_path):
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"fake")
    assert plan.transcribe(audio, client=FakeClient()) == (
        "they disagree about the forecast"
    )


def test_transcribe_failure_returns_none_instead_of_raising(tmp_path):
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"fake")
    assert plan.transcribe(audio, client=FakeClient(transcribe_fail=True)) is None


def test_build_plan_returns_five_matrix_variants(frames, tmp_path):
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"fake")
    result = plan.build_plan(frames, audio, client=FakeClient())
    assert len(result.variants) == 5
    result.validate_matrix()


def test_build_plan_sends_every_frame_as_an_image(frames, tmp_path):
    client = FakeClient()
    plan.build_plan(frames, None, client=client)
    sent = client.responses.calls[0]
    content = sent["input"][0]["content"]
    images = [c for c in content if c["type"] == "input_image"]
    assert len(images) == len(frames)


def test_build_plan_lists_frame_filenames_in_the_prompt(frames):
    client = FakeClient()
    plan.build_plan(frames, None, client=client)
    text = client.responses.calls[0]["input"][0]["content"][0]["text"]
    for f in frames:
        assert f.name in text


def test_build_plan_marks_transcript_unused_when_audio_is_missing(frames):
    result = plan.build_plan(frames, None, client=FakeClient())
    assert result.transcript_used is False


def test_build_plan_retries_once_on_bad_response(frames):
    client = FakeClient(fail_times=1)
    result = plan.build_plan(frames, None, client=client)
    assert len(client.responses.calls) == 2
    assert len(result.variants) == 5


def test_build_plan_raises_plan_error_after_second_failure(frames):
    with pytest.raises(plan.PlanError):
        plan.build_plan(frames, None, client=FakeClient(fail_times=2))


def test_build_plan_rejects_a_plan_that_breaks_the_matrix(frames):
    broken = _valid_plan()
    broken.variants[0].treatment = "full_bleed"
    with pytest.raises(plan.PlanError, match="matrix"):
        plan.build_plan(frames, None, client=FakeClient(parsed=broken))


def test_build_plan_with_no_frames_raises(tmp_path):
    with pytest.raises(plan.PlanError):
        plan.build_plan([], None, client=FakeClient())
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/pytest tests/test_plan.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.plan'`

- [ ] **Step 3: Write `src/plan.py`**

```python
"""Turn frames and audio into five validated thumbnail concepts."""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from .config import PLANNER_MODEL, TRANSCRIBE_MODEL
from .models import BatchPlan
from .prompts import planner_prompt

log = logging.getLogger(__name__)


class PlanError(RuntimeError):
    """The planner could not produce a usable batch plan."""


def _client(client=None):
    if client is not None:
        return client
    from openai import OpenAI

    return OpenAI()


def _data_url(image: Path) -> str:
    encoded = base64.b64encode(Path(image).read_bytes()).decode()
    return f"data:image/jpeg;base64,{encoded}"


def transcribe(audio: Path, client=None) -> str | None:
    """Transcribe the ad's audio. Returns None on failure — never fatal.

    A missing transcript weakens hook quality; losing the whole batch over it
    would be worse.
    """
    try:
        with Path(audio).open("rb") as handle:
            result = _client(client).audio.transcriptions.create(
                model=TRANSCRIBE_MODEL, file=handle,
            )
        text = (getattr(result, "text", "") or "").strip()
        return text or None
    except Exception:
        log.warning("transcription failed; planning from frames alone",
                    exc_info=True)
        return None


def build_plan(
    frames: list[Path],
    audio: Path | None,
    headline_override: str | None = None,
    context: str | None = None,
    client=None,
) -> BatchPlan:
    if not frames:
        raise PlanError("No frames to plan from.")

    active = _client(client)
    transcript = transcribe(audio, client=active) if audio else None

    instructions = planner_prompt(transcript, context, headline_override)
    frame_list = "\n".join(f"- {f.name}" for f in frames)
    instructions += (
        "\n## Available frames\n\nUse one of these exact filenames for "
        f"`frame_id`:\n\n{frame_list}\n"
    )

    content: list[dict] = [{"type": "input_text", "text": instructions}]
    for frame in frames:
        content.append({"type": "input_image", "image_url": _data_url(frame)})

    last_error: Exception | None = None
    for attempt in (1, 2):
        try:
            response = active.responses.parse(
                model=PLANNER_MODEL,
                input=[{"role": "user", "content": content}],
                text_format=BatchPlan,
            )
            result: BatchPlan = response.output_parsed
            result.validate_matrix()
            result.transcript_used = transcript is not None
            return result
        except ValueError as exc:          # includes matrix violations
            raise PlanError(f"The planner broke the matrix: {exc}") from exc
        except Exception as exc:
            last_error = exc
            log.warning("planner attempt %s failed", attempt, exc_info=True)

    raise PlanError(
        f"Couldn't plan thumbnails for that ad. ({last_error})"
    ) from last_error
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/pytest tests/test_plan.py -v
```

Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/plan.py tests/test_plan.py
git commit -m "feat: planner producing five validated hook/treatment variants"
```

---

### Task 9: `render.py` — one variant to image bytes

**Files:**
- Create: `src/render.py`, `tests/test_render.py`

**Interfaces:**
- Consumes: `src.config` (`IMAGE_MODEL`, `GEN_SIZE`, `IMAGE_QUALITY`), `src.models` (`Variant`), `src.prompts` (`render_prompt`), `src.refs` (`pick_refs`).
- Produces:
  - `RenderError(RuntimeError)`, `RenderBlocked(RenderError)`
  - `render_variant(variant: Variant, frames: dict[str, Path], client=None, extra_instruction: str = "", frame_override: Path | None = None) -> bytes`

API facts this task depends on, verified against OpenAI's reference: `images.edit` accepts an **array of up to 16 images**; `gpt-image-2` accepts **custom sizes** up to 3840×2160 so `2048x1152` is valid; `input_fidelity` is **not** a gpt-image-2 parameter (it processes all inputs at high fidelity automatically) and must not be sent; gpt-image models **always** return `b64_json`.

- [ ] **Step 1: Write the failing test**

`tests/test_render.py`:

```python
import base64

import pytest

from src import render
from src.config import GEN_SIZE, IMAGE_MODEL
from src.models import Variant

RAW = b"\x89PNG pretend image"


def _variant(treatment="split_screen", frame="scene_002.jpg", second=None):
    return Variant(
        index=1, hook_type="stat", treatment=treatment, headline="47K OVER",
        frame_id=frame, second_frame_id=second,
        scene_direction="orange versus blue, sparks at the seam",
        rationale="the ad names the overrun",
    )


class FakeImages:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def edit(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        payload = base64.b64encode(RAW).decode()
        return type("R", (), {
            "data": [type("D", (), {"b64_json": payload})()]
        })()


class FakeClient:
    def __init__(self, error=None):
        self.images = FakeImages(error)


@pytest.fixture
def frames(tmp_path):
    out = {}
    for name in ("scene_001.jpg", "scene_002.jpg"):
        p = tmp_path / name
        p.write_bytes(b"\xff\xd8\xff fake")
        out[name] = p
    return out


@pytest.fixture(autouse=True)
def fake_refs(tmp_path, monkeypatch):
    ref = tmp_path / "style_a.png"
    ref.write_bytes(b"\x89PNG fake")
    monkeypatch.setattr(render, "pick_refs", lambda treatment, limit=3: [ref])


def test_render_returns_decoded_bytes(frames):
    assert render.render_variant(_variant(), frames, client=FakeClient()) == RAW


def test_render_sends_the_configured_model_size_and_quality(frames):
    client = FakeClient()
    render.render_variant(_variant(), frames, client=client)
    call = client.images.calls[0]
    assert call["model"] == IMAGE_MODEL
    assert call["size"] == GEN_SIZE
    assert call["quality"] == "high"
    assert call["output_format"] == "png"
    assert call["n"] == 1


def test_render_never_sends_input_fidelity(frames):
    """gpt-image-2 rejects it; it processes inputs at high fidelity already."""
    client = FakeClient()
    render.render_variant(_variant(), frames, client=client)
    assert "input_fidelity" not in client.images.calls[0]


def test_render_sends_the_ad_frame_first_then_style_refs(frames):
    client = FakeClient()
    render.render_variant(_variant(), frames, client=client)
    images = client.images.calls[0]["image"]
    assert len(images) == 2          # one ad frame + one style ref


def test_split_screen_sends_both_frames(frames):
    client = FakeClient()
    render.render_variant(
        _variant(second="scene_001.jpg"), frames, client=client,
    )
    assert len(client.images.calls[0]["image"]) == 3


def test_image_array_never_exceeds_the_api_limit(frames, monkeypatch):
    monkeypatch.setattr(
        render, "pick_refs",
        lambda treatment, limit=3: list(frames.values()) * 20,
    )
    client = FakeClient()
    render.render_variant(_variant(), frames, client=client)
    assert len(client.images.calls[0]["image"]) <= 16


def test_unknown_frame_id_raises(frames):
    with pytest.raises(render.RenderError, match="frame"):
        render.render_variant(_variant(frame="nope.jpg"), frames,
                              client=FakeClient())


def test_extra_instruction_is_appended_to_the_prompt(frames):
    client = FakeClient()
    render.render_variant(
        _variant(), frames, client=client,
        extra_instruction="The previous attempt cut off the last word.",
    )
    assert "cut off the last word" in client.images.calls[0]["prompt"]


def test_moderation_refusal_becomes_render_blocked(frames):
    class Blocked(Exception):
        pass

    client = FakeClient(error=Blocked("moderation_blocked: request rejected"))
    with pytest.raises(render.RenderBlocked):
        render.render_variant(_variant(), frames, client=client)


def test_other_api_errors_become_render_error(frames):
    client = FakeClient(error=RuntimeError("503 service unavailable"))
    with pytest.raises(render.RenderError):
        render.render_variant(_variant(), frames, client=client)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/pytest tests/test_render.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.render'`

- [ ] **Step 3: Write `src/render.py`**

```python
"""One variant in, image bytes out. One gpt-image-2 call per variant."""

from __future__ import annotations

import base64
import contextlib
from pathlib import Path

from .config import GEN_SIZE, IMAGE_MODEL, IMAGE_QUALITY
from .models import Variant
from .prompts import render_prompt
from .refs import pick_refs

# images.edit accepts at most 16 input images for gpt-image models.
MAX_INPUT_IMAGES = 16


class RenderError(RuntimeError):
    """The image call failed."""


class RenderBlocked(RenderError):
    """Refused by the content filter — expected occasionally on real faces."""


def _client(client=None):
    if client is not None:
        return client
    from openai import OpenAI

    return OpenAI()


def _is_moderation(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(
        marker in text
        for marker in ("moderation", "safety", "content_policy", "rejected")
    )


def render_variant(
    variant: Variant,
    frames: dict[str, Path],
    client=None,
    extra_instruction: str = "",
    frame_override: Path | None = None,
) -> bytes:
    """Render one thumbnail at GEN_SIZE. Returns raw PNG bytes."""
    primary = frame_override or frames.get(variant.frame_id)
    if primary is None:
        raise RenderError(
            f"The planner picked a frame we don't have: {variant.frame_id}"
        )

    image_paths: list[Path] = [primary]
    if variant.second_frame_id and variant.second_frame_id in frames:
        image_paths.append(frames[variant.second_frame_id])
    image_paths.extend(pick_refs(variant.treatment, limit=3))
    image_paths = image_paths[:MAX_INPUT_IMAGES]

    prompt = render_prompt(variant)
    if extra_instruction:
        prompt += f"\n\n## Correction for this attempt\n\n{extra_instruction}"

    with contextlib.ExitStack() as stack:
        handles = [stack.enter_context(Path(p).open("rb")) for p in image_paths]
        try:
            result = _client(client).images.edit(
                model=IMAGE_MODEL,
                image=handles,
                prompt=prompt,
                size=GEN_SIZE,
                quality=IMAGE_QUALITY,
                output_format="png",
                n=1,
                # No input_fidelity: gpt-image-2 does not accept it and already
                # processes every input at high fidelity.
            )
        except Exception as exc:
            if _is_moderation(exc):
                raise RenderBlocked(
                    "The content filter refused this render."
                ) from exc
            raise RenderError(f"Image generation failed: {exc}") from exc

    return base64.b64decode(result.data[0].b64_json)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/pytest tests/test_render.py -v
```

Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/render.py tests/test_render.py
git commit -m "feat: gpt-image-2 render call with ad frames and style refs"
```

---

### Task 10: `qa.py` — hard checks and the legibility gate

**Files:**
- Create: `src/qa.py`, `tests/test_qa.py`

**Interfaces:**
- Consumes: `src.config` (`FINAL_W`, `FINAL_H`, `MAX_BYTES`, `QA_MODEL`), `src.prompts` (`load`).
- Produces:
  - `QAResult` dataclass with fields `ok: bool`, `problems: list[str]`, `transcribed: str | None`
  - `normalize(text: str) -> list[str]`
  - `headline_is_legible(intended: str, transcribed: str) -> bool`
  - `hard_checks(path: Path) -> list[str]`
  - `check(path: Path, intended_headline: str, client=None) -> QAResult`

- [ ] **Step 1: Write the failing test**

`tests/test_qa.py`:

```python
import io

from PIL import Image

from src import qa
from src.config import FINAL_H, FINAL_W


def _write(path, w=FINAL_W, h=FINAL_H):
    Image.new("RGB", (w, h), (10, 20, 40)).save(path, "PNG")
    return path


class FakeResponses:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("R", (), {"output_text": self.text})()


class FakeClient:
    def __init__(self, text):
        self.responses = FakeResponses(text)


def test_normalize_strips_case_and_punctuation():
    assert qa.normalize("Same AI, Different Answer!") == [
        "SAME", "AI", "DIFFERENT", "ANSWER",
    ]


def test_normalize_collapses_whitespace():
    assert qa.normalize("  47K   OVER  ") == ["47K", "OVER"]


def test_legible_when_every_word_appears_in_order():
    assert qa.headline_is_legible("SAME AI DIFFERENT ANSWER",
                                  "same ai different answer")


def test_legible_when_the_model_reads_extra_words():
    assert qa.headline_is_legible("47K OVER", "financeos 47K over budget")


def test_not_legible_when_a_word_is_missing():
    assert not qa.headline_is_legible("SAME AI DIFFERENT ANSWER",
                                      "same ai answer")


def test_not_legible_when_words_are_out_of_order():
    assert not qa.headline_is_legible("PROBABLY VS PROVEN",
                                      "proven vs probably")


def test_not_legible_when_nothing_is_readable():
    assert not qa.headline_is_legible("47K OVER", "NONE")


def test_hard_checks_pass_for_a_correct_file(tmp_path):
    assert qa.hard_checks(_write(tmp_path / "ok.png")) == []


def test_hard_checks_flag_wrong_dimensions(tmp_path):
    problems = qa.hard_checks(_write(tmp_path / "small.png", 1280, 720))
    assert any("1920" in p for p in problems)


def test_hard_checks_flag_oversize_files(tmp_path, monkeypatch):
    path = _write(tmp_path / "big.png")
    monkeypatch.setattr(qa, "MAX_BYTES", 10)
    assert any("2 MB" in p or "too large" in p for p in qa.hard_checks(path))


def test_hard_checks_flag_a_corrupt_file(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")
    assert qa.hard_checks(path)


def test_check_passes_when_the_headline_reads_back(tmp_path):
    result = qa.check(_write(tmp_path / "a.png"), "47K OVER",
                      client=FakeClient("47K OVER"))
    assert result.ok
    assert result.problems == []


def test_check_fails_and_names_the_problem_when_text_is_unreadable(tmp_path):
    result = qa.check(_write(tmp_path / "b.png"), "SAME AI DIFFERENT ANSWER",
                      client=FakeClient("same different"))
    assert not result.ok
    assert any("headline" in p.lower() for p in result.problems)
    assert result.transcribed == "same different"


def test_check_sends_a_downscaled_320px_image(tmp_path):
    client = FakeClient("47K OVER")
    qa.check(_write(tmp_path / "c.png"), "47K OVER", client=client)
    sent = client.responses.calls[0]["input"][0]["content"]
    image_entry = next(c for c in sent if c["type"] == "input_image")
    assert image_entry["image_url"].startswith("data:image/png;base64,")


def test_check_survives_a_vision_failure_without_blocking_the_batch(tmp_path):
    class Broken:
        class responses:
            @staticmethod
            def create(**kwargs):
                raise RuntimeError("vision down")

    result = qa.check(_write(tmp_path / "d.png"), "47K OVER", client=Broken())
    assert result.ok            # unverified, but not a failure
    assert result.transcribed is None
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/pytest tests/test_qa.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.qa'`

- [ ] **Step 3: Write `src/qa.py`**

```python
"""Verification. OpenAI's own docs warn gpt-image-2 can still struggle with
precise text placement, and our thumbnails are headline-dominant, so we read
every render back before showing it.
"""

from __future__ import annotations

import base64
import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .config import FINAL_H, FINAL_W, MAX_BYTES, QA_MODEL
from .prompts import load

log = logging.getLogger(__name__)

# The width a thumbnail actually occupies in a YouTube feed.
FEED_WIDTH = 320

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)


@dataclass
class QAResult:
    ok: bool
    problems: list[str] = field(default_factory=list)
    transcribed: str | None = None


def normalize(text: str) -> list[str]:
    return _PUNCT.sub(" ", (text or "")).upper().split()


def headline_is_legible(intended: str, transcribed: str) -> bool:
    """Every intended word must appear, in order, in what the model read back.

    Extra words are tolerated — the model may read a shirt logo. Missing or
    reordered words mean the type warped, got cut off, or wrapped wrongly.
    """
    want = normalize(intended)
    got = normalize(transcribed)
    if not want or got == ["NONE"]:
        return False

    index = 0
    for word in got:
        if index < len(want) and word == want[index]:
            index += 1
    return index == len(want)


def hard_checks(path: Path) -> list[str]:
    problems: list[str] = []
    path = Path(path)

    if not path.exists() or path.stat().st_size == 0:
        return ["the file is missing or empty"]

    try:
        with Image.open(path) as im:
            size = im.size
            im.verify()
    except Exception:
        return ["the file isn't a readable image"]

    if size != (FINAL_W, FINAL_H):
        problems.append(
            f"dimensions are {size[0]}x{size[1]}, must be {FINAL_W}x{FINAL_H}"
        )
    if path.stat().st_size > MAX_BYTES:
        problems.append("the file is too large for YouTube's 2 MB limit")
    return problems


def _feed_size_data_url(path: Path) -> str:
    with Image.open(path) as im:
        thumb = im.convert("RGB")
        height = round(thumb.height * FEED_WIDTH / thumb.width)
        thumb = thumb.resize((FEED_WIDTH, height), Image.LANCZOS)
        buffer = io.BytesIO()
        thumb.save(buffer, "PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


def _client(client=None):
    if client is not None:
        return client
    from openai import OpenAI

    return OpenAI()


def check(path: Path, intended_headline: str, client=None) -> QAResult:
    problems = hard_checks(path)
    if problems:
        return QAResult(ok=False, problems=problems)

    try:
        response = _client(client).responses.create(
            model=QA_MODEL,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": load("qa_legibility")},
                    {"type": "input_image",
                     "image_url": _feed_size_data_url(path)},
                ],
            }],
        )
        transcribed = (getattr(response, "output_text", "") or "").strip()
    except Exception:
        # A QA outage must not cost the user their batch.
        log.warning("legibility check unavailable; passing unverified",
                    exc_info=True)
        return QAResult(ok=True, problems=[], transcribed=None)

    if not headline_is_legible(intended_headline, transcribed):
        return QAResult(
            ok=False,
            problems=[
                "the headline isn't readable at feed size "
                f"(read back as {transcribed!r})"
            ],
            transcribed=transcribed,
        )

    return QAResult(ok=True, problems=[], transcribed=transcribed)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/pytest tests/test_qa.py -v
```

Expected: 15 passed

- [ ] **Step 5: Commit**

```bash
git add src/qa.py tests/test_qa.py
git commit -m "feat: QA gate reading every headline back at feed size"
```

---

### Task 11: `pipeline.py` — orchestration, concurrency, partial failure

**Files:**
- Create: `src/pipeline.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `src.probe`, `src.plan`, `src.render`, `src.postprocess`, `src.qa`, `src.models`.
- Produces:
  - `ThumbResult` dataclass: `variant: Variant`, `path: Path | None`, `flagged: bool`, `note: str`
  - `BatchOutcome` dataclass: `plan: BatchPlan`, `results: list[ThumbResult]`, `warnings: list[str]`
  - `generate_batch(video: Path, work_dir: Path, headline_override: str | None = None, context: str | None = None, client=None, progress: Callable[[str], None] | None = None) -> BatchOutcome`

`pipeline` calls its dependencies through their modules (`render.render_variant`, not a bare import), so tests monkeypatch module attributes. The contract that matters: **`results` always has exactly five entries**, even when renders fail. A user must never see a blank screen.

- [ ] **Step 1: Write the failing test**

`tests/test_pipeline.py`:

```python
from pathlib import Path

import pytest

from src import pipeline, postprocess, probe, qa, render
from src import plan as plan_module
from src.models import MATRIX, BatchPlan, Variant
from src.qa import QAResult


def _plan():
    return BatchPlan(
        ad_summary="same AI, different answers",
        transcript_used=True,
        variants=[
            Variant(
                index=i, hook_type=h, treatment=t, headline=f"HOOK {i}",
                frame_id="scene_001.jpg", second_frame_id=None,
                scene_direction="sparks", rationale="from the ad",
            )
            for i, h, t in MATRIX
        ],
    )


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Replace every external dependency with a deterministic fake."""
    frames = []
    for name in ("scene_001.jpg", "scene_002.jpg"):
        frame = tmp_path / name
        frame.write_bytes(b"\xff\xd8\xff fake")
        frames.append(frame)
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"fake")

    monkeypatch.setattr(probe, "extract_frames",
                        lambda video, out_dir, max_frames=16: frames)
    monkeypatch.setattr(probe, "extract_audio", lambda video, out_dir: audio)
    monkeypatch.setattr(plan_module, "build_plan",
                        lambda *a, **k: _plan())
    monkeypatch.setattr(render, "render_variant",
                        lambda *a, **k: b"\x89PNG bytes")

    written = []

    def fake_finalize(image_bytes, out_path):
        Path(out_path).write_bytes(image_bytes)
        written.append(Path(out_path))
        return Path(out_path)

    monkeypatch.setattr(postprocess, "finalize", fake_finalize)
    monkeypatch.setattr(qa, "check", lambda *a, **k: QAResult(ok=True))
    return written


def test_batch_returns_exactly_five_results(wired, tmp_path):
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    assert len(outcome.results) == 5
    assert all(r.path is not None for r in outcome.results)
    assert not any(r.flagged for r in outcome.results)


def test_every_result_keeps_its_variant_metadata(wired, tmp_path):
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    pairs = {(r.variant.hook_type, r.variant.treatment) for r in outcome.results}
    assert pairs == {(h, t) for _, h, t in MATRIX}


def test_output_filenames_are_ordered_and_descriptive(wired, tmp_path):
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    names = sorted(r.path.name for r in outcome.results)
    assert names[0].startswith("01_stat_split_screen")
    assert names[4].startswith("05_outcome_product_forward")


def test_failed_qa_triggers_exactly_one_reroll(wired, tmp_path, monkeypatch):
    calls = {"render": 0}

    def counting_render(*args, **kwargs):
        calls["render"] += 1
        return b"\x89PNG bytes"

    monkeypatch.setattr(render, "render_variant", counting_render)
    monkeypatch.setattr(
        qa, "check",
        lambda path, headline, **k: QAResult(
            ok=False, problems=["the headline isn't readable"],
        ),
    )
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    assert calls["render"] == 10          # five variants, one reroll each
    assert len(outcome.results) == 5
    assert all(r.flagged for r in outcome.results)
    assert all(r.path is not None for r in outcome.results)


def test_reroll_passes_the_failure_reason_back_into_the_prompt(
    wired, tmp_path, monkeypatch
):
    seen = []

    def capturing_render(variant, frames, client=None, extra_instruction="",
                         frame_override=None):
        seen.append(extra_instruction)
        return b"\x89PNG bytes"

    monkeypatch.setattr(render, "render_variant", capturing_render)
    monkeypatch.setattr(
        qa, "check",
        lambda path, headline, **k: QAResult(ok=False, problems=["cut off"]),
    )
    pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    assert any("cut off" in s for s in seen if s)


def test_a_render_that_always_fails_still_yields_a_result_row(
    wired, tmp_path, monkeypatch
):
    def always_fails(*args, **kwargs):
        raise render.RenderError("503 from the API")

    monkeypatch.setattr(render, "render_variant", always_fails)
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    assert len(outcome.results) == 5
    assert all(r.path is None for r in outcome.results)
    assert all("503" in r.note for r in outcome.results)


def test_blocked_render_retries_with_a_different_frame(
    wired, tmp_path, monkeypatch
):
    attempts = []

    def blocked_then_ok(variant, frames, client=None, extra_instruction="",
                        frame_override=None):
        # Keyed on the argument, not a call counter: five variants render
        # concurrently, so a counter would be racy.
        attempts.append((variant.index, frame_override))
        if frame_override is None:
            raise render.RenderBlocked("filter refused")
        return b"\x89PNG bytes"

    monkeypatch.setattr(render, "render_variant", blocked_then_ok)
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    assert len(outcome.results) == 5
    assert all(r.path is not None for r in outcome.results)
    assert all(override is not None for _, override in attempts[5:]), (
        "the reroll must try a different frame than the blocked one"
    )


def test_missing_audio_is_a_warning_not_a_failure(wired, tmp_path, monkeypatch):
    def no_audio(video, out_dir):
        raise probe.ProbeError("That video has no usable audio track.")

    monkeypatch.setattr(probe, "extract_audio", no_audio)
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    assert len(outcome.results) == 5
    assert any("audio" in w.lower() for w in outcome.warnings)


def test_progress_callback_reports_each_phase(wired, tmp_path):
    messages = []
    pipeline.generate_batch(
        tmp_path / "ad.mp4", tmp_path / "work", progress=messages.append,
    )
    joined = " ".join(messages).lower()
    assert "frame" in joined
    assert "plan" in joined or "reading" in joined
    assert "render" in joined
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/pytest tests/test_pipeline.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.pipeline'`

- [ ] **Step 3: Write `src/pipeline.py`**

```python
"""Orchestration. Knows the order of operations and nothing about the UI.

The hard contract: the caller always receives exactly five result rows. A failed
render is a row with path=None and a readable note, never a missing tile.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import plan as plan_module
from . import postprocess, probe, qa, render
from .models import BatchPlan, Variant

log = logging.getLogger(__name__)

MAX_WORKERS = 5


@dataclass
class ThumbResult:
    variant: Variant
    path: Path | None
    flagged: bool = False
    note: str = ""


@dataclass
class BatchOutcome:
    plan: BatchPlan
    results: list[ThumbResult]
    warnings: list[str] = field(default_factory=list)


def _slug(variant: Variant) -> str:
    return f"{variant.index:02d}_{variant.hook_type}_{variant.treatment}"


def _other_frame(frames: dict[str, Path], used: str) -> Path | None:
    for name, path in frames.items():
        if name != used:
            return path
    return None


def _one_variant(
    variant: Variant,
    frames: dict[str, Path],
    out_dir: Path,
    client,
) -> ThumbResult:
    """Render, finalize, verify. One reroll on failure, then flag and move on."""
    extra_instruction = ""
    frame_override: Path | None = None
    last_note = ""

    for attempt in (1, 2):
        try:
            raw = render.render_variant(
                variant, frames, client=client,
                extra_instruction=extra_instruction,
                frame_override=frame_override,
            )
        except render.RenderBlocked as exc:
            last_note = f"blocked by content filter: {exc}"
            frame_override = _other_frame(frames, variant.frame_id)
            continue
        except render.RenderError as exc:
            last_note = str(exc)
            continue

        path = postprocess.finalize(raw, out_dir / f"{_slug(variant)}.png")
        result = qa.check(path, variant.headline, client=client)
        if result.ok:
            return ThumbResult(variant=variant, path=path)

        last_note = "; ".join(result.problems)
        if attempt == 1:
            extra_instruction = (
                f"The previous attempt failed verification: {last_note}. "
                "Render the headline larger, fully inside the frame, with more "
                "space around it, and make every word unmistakably legible."
            )
            continue

        # Second failure: still hand it over, flagged. The user decides.
        return ThumbResult(
            variant=variant, path=path, flagged=True,
            note=f"text may be unreadable — {last_note}",
        )

    return ThumbResult(variant=variant, path=None, flagged=True, note=last_note)


def generate_batch(
    video: Path,
    work_dir: Path,
    headline_override: str | None = None,
    context: str | None = None,
    client=None,
    progress: Callable[[str], None] | None = None,
) -> BatchOutcome:
    def say(message: str) -> None:
        log.info(message)
        if progress:
            progress(message)

    work_dir = Path(work_dir)
    frames_dir = work_dir / "frames"
    out_dir = work_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    say("Pulling frames out of the ad…")
    frame_paths = probe.extract_frames(video, frames_dir)
    frames = {p.name: p for p in frame_paths}

    audio: Path | None = None
    try:
        audio = probe.extract_audio(video, work_dir)
    except probe.ProbeError as exc:
        warnings.append(f"Couldn't read the audio, so hooks come from the "
                        f"picture alone. ({exc})")

    say("Reading the ad and planning five hooks…")
    batch_plan = plan_module.build_plan(
        frame_paths, audio, headline_override, context, client=client,
    )
    if not batch_plan.transcript_used and not warnings:
        warnings.append("No transcript was available; hooks come from the "
                        "frames alone.")

    say("Rendering 5 thumbnails…")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(
            lambda v: _one_variant(v, frames, out_dir, client),
            batch_plan.variants,
        ))

    say("Done.")
    return BatchOutcome(plan=batch_plan, results=results, warnings=warnings)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/pytest tests/test_pipeline.py -v
```

Expected: 9 passed

- [ ] **Step 5: Run the whole suite**

```bash
.venv/bin/pytest -v
```

Expected: all tests from Tasks 1–11 pass.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline.py tests/test_pipeline.py
git commit -m "feat: batch orchestration with concurrent renders and reroll-once QA"
```

---

### Task 12: `auth.py` — Google sign-in and the domain gate

**Files:**
- Create: `src/auth.py`, `tests/test_auth.py`

**Interfaces:**
- Consumes: `src.config` (`ALLOWED_EMAIL_DOMAIN`, `GOOGLE_SCOPES`).
- Produces:
  - `AuthError(RuntimeError)`
  - `is_allowed_email(email: str) -> bool`
  - `build_flow(client_id: str, client_secret: str, redirect_uri: str)`
  - `authorization_url(flow) -> tuple[str, str]`
  - `exchange_code(flow, code: str) -> tuple[object, str]` returning `(credentials, email)`

The OAuth consent screen is configured **Internal** in a free Google Cloud project. Internal apps may use sensitive scopes with no Google verification review, and access is automatically restricted to the Workspace — which is where the `@datarails.com` gate really comes from. `is_allowed_email` is defence in depth.

- [ ] **Step 1: Write the failing test**

`tests/test_auth.py`:

```python
import pytest

from src import auth
from src.config import GOOGLE_SCOPES


@pytest.mark.parametrize("email", [
    "omer.y@datarails.com",
    "Someone.Else@Datarails.com",
    "  omer.y@datarails.com  ",
])
def test_datarails_emails_are_allowed(email):
    assert auth.is_allowed_email(email)


@pytest.mark.parametrize("email", [
    "someone@gmail.com",
    "attacker@datarails.com.evil.com",
    "datarails.com",
    "",
    None,
])
def test_everything_else_is_rejected(email):
    assert not auth.is_allowed_email(email)


def test_build_flow_requests_the_drive_scope():
    flow = auth.build_flow("id.apps.googleusercontent.com", "secret",
                           "https://example.streamlit.app")
    assert set(GOOGLE_SCOPES).issubset(set(flow.oauth2session.scope))


def test_authorization_url_forces_the_datarails_domain_hint():
    flow = auth.build_flow("id.apps.googleusercontent.com", "secret",
                           "https://example.streamlit.app")
    url, state = auth.authorization_url(flow)
    assert "hd=datarails.com" in url
    assert state


def test_exchange_code_rejects_a_non_datarails_account(monkeypatch):
    class FakeFlow:
        credentials = object()

        def fetch_token(self, code=None):
            return None

    monkeypatch.setattr(auth, "_email_from_credentials",
                        lambda creds: "someone@gmail.com")
    with pytest.raises(auth.AuthError, match="datarails.com"):
        auth.exchange_code(FakeFlow(), "code123")


def test_exchange_code_returns_credentials_and_email(monkeypatch):
    sentinel = object()

    class FakeFlow:
        credentials = sentinel

        def fetch_token(self, code=None):
            return None

    monkeypatch.setattr(auth, "_email_from_credentials",
                        lambda creds: "omer.y@datarails.com")
    creds, email = auth.exchange_code(FakeFlow(), "code123")
    assert creds is sentinel
    assert email == "omer.y@datarails.com"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/pytest tests/test_auth.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.auth'`

- [ ] **Step 3: Write `src/auth.py`**

```python
"""Google sign-in. One consent covers identity and Drive access."""

from __future__ import annotations

from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

from .config import ALLOWED_EMAIL_DOMAIN, GOOGLE_SCOPES


class AuthError(RuntimeError):
    """Sign-in failed or the account isn't allowed."""


def is_allowed_email(email: str | None) -> bool:
    if not email:
        return False
    cleaned = email.strip().lower()
    return cleaned.endswith(f"@{ALLOWED_EMAIL_DOMAIN}")


def build_flow(client_id: str, client_secret: str, redirect_uri: str) -> Flow:
    return Flow.from_client_config(
        {
            "web": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [redirect_uri],
            }
        },
        scopes=GOOGLE_SCOPES,
        redirect_uri=redirect_uri,
    )


def authorization_url(flow: Flow) -> tuple[str, str]:
    return flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
        hd=ALLOWED_EMAIL_DOMAIN,     # send users straight to their work account
    )


def _email_from_credentials(credentials) -> str:
    service = build("oauth2", "v2", credentials=credentials,
                    cache_discovery=False)
    return service.userinfo().get().execute().get("email", "")


def exchange_code(flow, code: str) -> tuple[object, str]:
    try:
        flow.fetch_token(code=code)
    except Exception as exc:
        raise AuthError(f"Sign-in didn't complete: {exc}") from exc

    credentials = flow.credentials
    email = _email_from_credentials(credentials)
    if not is_allowed_email(email):
        raise AuthError(
            f"{email or 'That account'} isn't a {ALLOWED_EMAIL_DOMAIN} account. "
            "Sign in with your Datarails email."
        )
    return credentials, email
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/pytest tests/test_auth.py -v
```

Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add src/auth.py tests/test_auth.py
git commit -m "feat: google sign-in with datarails.com domain gate"
```

---

### Task 13: `app.py` — the Streamlit interface

**Files:**
- Create: `app.py`, `.streamlit/config.toml`, `.streamlit/secrets.toml.example`, `tests/test_app_helpers.py`
- Modify: `.gitignore` (confirm `.streamlit/secrets.toml` is ignored — it already is)

**Interfaces:**
- Consumes: `src.auth`, `src.drive`, `src.pipeline`, `src.refs`, `src.config`.
- Produces: `zip_bytes(results) -> bytes`, `batch_folder_name(video_name: str, stamp: str) -> str` (both pure, both tested); everything else is Streamlit callbacks.

Streamlit UI itself is not unit-tested — the two pure helpers are, because a broken zip or a bad folder name is a real bug and both are trivially checkable.

- [ ] **Step 1: Write the failing test**

`tests/test_app_helpers.py`:

```python
import io
import zipfile
from pathlib import Path

from PIL import Image

import app
from src.models import MATRIX, Variant
from src.pipeline import ThumbResult


def _result(tmp_path, index, hook, treatment, missing=False):
    variant = Variant(
        index=index, hook_type=hook, treatment=treatment,
        headline=f"HOOK {index}", frame_id="scene_001.jpg",
        second_frame_id=None, scene_direction="sparks", rationale="from the ad",
    )
    if missing:
        return ThumbResult(variant=variant, path=None, flagged=True,
                           note="render failed")
    path = tmp_path / f"{index:02d}_{hook}_{treatment}.png"
    Image.new("RGB", (1920, 1080), (10, 20, 40)).save(path, "PNG")
    return ThumbResult(variant=variant, path=path)


def test_zip_contains_every_successful_render(tmp_path):
    results = [_result(tmp_path, i, h, t) for i, h, t in MATRIX]
    with zipfile.ZipFile(io.BytesIO(app.zip_bytes(results))) as archive:
        assert len(archive.namelist()) == 5


def test_zip_skips_failed_renders(tmp_path):
    results = [_result(tmp_path, i, h, t) for i, h, t in MATRIX]
    results[2] = _result(tmp_path, 3, "conflict", "full_bleed", missing=True)
    with zipfile.ZipFile(io.BytesIO(app.zip_bytes(results))) as archive:
        assert len(archive.namelist()) == 4


def test_zip_entries_keep_their_descriptive_names(tmp_path):
    results = [_result(tmp_path, 1, "stat", "split_screen")]
    with zipfile.ZipFile(io.BytesIO(app.zip_bytes(results))) as archive:
        assert archive.namelist() == ["01_stat_split_screen.png"]


def test_batch_folder_name_strips_the_extension_and_dates_it():
    name = app.batch_folder_name("claude-vs-claude-v2.mp4", "2026-08-18")
    assert name == "thumbnails — claude-vs-claude-v2 — 2026-08-18"


def test_batch_folder_name_survives_a_name_with_no_extension():
    assert app.batch_folder_name("ad", "2026-08-18") == (
        "thumbnails — ad — 2026-08-18"
    )
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/pytest tests/test_app_helpers.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'app'`

- [ ] **Step 3: Write `.streamlit/config.toml`**

```toml
[server]
maxUploadSize = 200

[theme]
base = "dark"
primaryColor = "#F97316"
```

- [ ] **Step 4: Write `.streamlit/secrets.toml.example`**

```toml
# Copy to .streamlit/secrets.toml locally, or paste into Streamlit Cloud's
# secrets manager. Never commit the real file.
OPENAI_API_KEY = "sk-..."
GOOGLE_CLIENT_ID = "....apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "..."
REDIRECT_URI = "http://localhost:8501"
```

- [ ] **Step 5: Write `app.py`**

```python
"""YT Thumbnail Creator — Streamlit interface. No API logic lives here."""

from __future__ import annotations

import io
import os
import tempfile
import zipfile
from datetime import date
from pathlib import Path

import streamlit as st

from src import auth, drive, refs
from src.pipeline import ThumbResult, generate_batch

# --- pure helpers (tested) --------------------------------------------------
# Everything below the helpers runs inside main(). Streamlit executes this file
# with __name__ == "__main__", so the guard keeps `import app` importable from
# the test suite without firing the sign-in flow and st.stop().
def zip_bytes(results: list[ThumbResult]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for result in results:
            if result.path is not None:
                archive.write(result.path, arcname=result.path.name)
    return buffer.getvalue()


def batch_folder_name(video_name: str, stamp: str) -> str:
    return f"thumbnails — {Path(video_name).stem} — {stamp}"


# --- secrets ---------------------------------------------------------------
def _secret(name: str) -> str:
    value = st.secrets.get(name, os.environ.get(name, ""))
    if not value:
        st.error(
            f"`{name}` isn't configured. Add it in Streamlit's secrets manager."
        )
        st.stop()
    return value


# --- sign-in ---------------------------------------------------------------
def require_sign_in():
    if "credentials" in st.session_state:
        return st.session_state["credentials"]

    flow = auth.build_flow(
        _secret("GOOGLE_CLIENT_ID"),
        _secret("GOOGLE_CLIENT_SECRET"),
        _secret("REDIRECT_URI"),
    )

    code = st.query_params.get("code")
    if code:
        try:
            credentials, email = auth.exchange_code(flow, code)
        except auth.AuthError as exc:
            st.error(str(exc))
            st.query_params.clear()
            st.stop()
        st.session_state["credentials"] = credentials
        st.session_state["email"] = email
        st.query_params.clear()
        st.rerun()

    url, _state = auth.authorization_url(flow)
    st.title("🎬 YT Thumbnail Creator")
    st.write("Five YouTube ad thumbnails from one Drive link.")
    st.link_button("Sign in with your Datarails Google account", url,
                   type="primary")
    st.stop()


# --- main ------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="YT Thumbnail Creator", page_icon="🎬",
                       layout="wide")
    credentials = require_sign_in()

    st.title("🎬 YT Thumbnail Creator")
    st.caption(f"Signed in as {st.session_state.get('email', '')}")

    link = st.text_input(
        "Google Drive link to the ad",
        placeholder="https://drive.google.com/file/d/…/view",
    )

    with st.expander("Advanced"):
        headline_override = st.text_input(
            "Headline override",
            help="Leave empty to let the tool write five different hooks. If filled, "
                 "all five use this exact line and only the visuals vary.",
        )
        context = st.text_area(
            "Extra context",
            help="Campaign goal, who it's for, anything to avoid.",
            height=90,
        )

    if st.button("Generate 5 thumbnails", type="primary", disabled=not link):
        try:
            file_id = drive.parse_file_id(link)
        except drive.DriveLinkError as exc:
            st.error(str(exc))
            st.stop()

        status = st.status("Starting…", expanded=True)
        work_dir = Path(tempfile.mkdtemp(prefix="ytthumb_"))
        try:
            status.update(label="Fetching the ad from Drive…")
            video, parent_id = drive.fetch_video(file_id, credentials, work_dir)

            outcome = generate_batch(
                video, work_dir,
                headline_override=headline_override or None,
                context=context or None,
                progress=lambda message: status.update(label=message),
            )
            status.update(label="Done.", state="complete")

            st.session_state["outcome"] = outcome
            st.session_state["video_name"] = video.name
            st.session_state["parent_id"] = parent_id
            # Free the 86MB source immediately; renders live in work_dir/out.
            video.unlink(missing_ok=True)
        except (drive.DriveError, RuntimeError) as exc:
            status.update(label="Failed.", state="error")
            st.error(str(exc))
            st.stop()

    outcome = st.session_state.get("outcome")
    if outcome:
        for warning in outcome.warnings:
            st.warning(warning)
        st.caption(f"**What the ad is about:** {outcome.plan.ad_summary}")

        columns = st.columns(3)
        for position, result in enumerate(outcome.results):
            with columns[position % 3]:
                label = f"{result.variant.hook_type} · {result.variant.treatment}"
                if result.path is None:
                    st.error(f"**{label}** — couldn't render. {result.note}")
                    continue
                st.image(str(result.path), caption=label)
                if result.flagged:
                    st.warning(f"⚠️ {result.note}")
                st.download_button(
                    "Download", result.path.read_bytes(),
                    file_name=result.path.name,
                    mime="image/png" if result.path.suffix == ".png" else "image/jpeg",
                    key=f"dl_{result.variant.index}",
                )
                if st.button("⭐ Save as reference",
                             key=f"ref_{result.variant.index}"):
                    refs.save_winner(result.path, result.variant.treatment)
                    st.success("Added to the house style pack.")

        successful = [r for r in outcome.results if r.path]
        if successful:
            st.download_button(
                "Download all (.zip)", zip_bytes(outcome.results),
                file_name="thumbnails.zip", mime="application/zip",
            )
            if st.button("Save to Drive"):
                try:
                    url = drive.save_batch(
                        [r.path for r in successful],
                        st.session_state["parent_id"],
                        batch_folder_name(st.session_state["video_name"],
                                          date.today().isoformat()),
                        credentials,
                    )
                    st.success(f"Saved to Drive. [Open the folder]({url})")
                except drive.DriveError as exc:
                    st.error(str(exc))

if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
.venv/bin/pytest tests/test_app_helpers.py -v
```

Expected: 5 passed

- [ ] **Step 7: Verify the app boots without errors**

```bash
.venv/bin/python -c "import ast, pathlib; ast.parse(pathlib.Path('app.py').read_text()); print('app.py parses')"
.venv/bin/streamlit run app.py --server.headless true &
sleep 8 && curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8501 && kill %1
```

Expected: `app.py parses` then `200`. It will show the sign-in screen or a missing-secrets error — both prove it boots.

- [ ] **Step 8: Commit**

```bash
git add app.py .streamlit/config.toml .streamlit/secrets.toml.example tests/test_app_helpers.py
git commit -m "feat: streamlit interface with sign-in, grid, downloads, and save-back"
```

---

### Task 14: First real batch — `scripts/live_run.py`

**Files:**
- Create: `scripts/live_run.py` (the `out/` directory is created at runtime and stays gitignored)

**Interfaces:**
- Consumes: `src.pipeline`, `src.config`.
- Produces: a CLI that runs the real pipeline against a local video file, bypassing Drive and Streamlit entirely.

This is the task where the design meets reality. Everything before it was mocked. The point is to find out whether `gpt-image-2` actually renders the headline legibly — the one real risk the spec names.

- [ ] **Step 1: Write `scripts/live_run.py`**

```python
"""Run the real pipeline against a local video. Costs real money (~$1).

Usage:
    OPENAI_API_KEY=sk-... .venv/bin/python scripts/live_run.py \
        "Sample input ad/claude-vs-claude-with-fos-v2_....mp4"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT          # noqa: E402
from src.pipeline import generate_batch       # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        sample = next((PROJECT_ROOT / "Sample input ad").glob("*.mp4"), None)
        if sample is None:
            print("Pass a video path.")
            return 1
        video = sample
    else:
        video = Path(sys.argv[1])

    work_dir = PROJECT_ROOT / "out" / "live"
    work_dir.mkdir(parents=True, exist_ok=True)

    outcome = generate_batch(video, work_dir, progress=lambda m: print(f"  {m}"))

    print(f"\nAd: {outcome.plan.ad_summary}")
    for warning in outcome.warnings:
        print(f"  warning: {warning}")

    print("\nResults:")
    for result in outcome.results:
        variant = result.variant
        status = "ok"
        if result.path is None:
            status = f"FAILED — {result.note}"
        elif result.flagged:
            status = f"FLAGGED — {result.note}"
        print(f"  {variant.index}. [{variant.hook_type}/{variant.treatment}] "
              f"{variant.headline!r} → {status}")
        if result.path:
            print(f"     {result.path}")

    flagged = sum(1 for r in outcome.results if r.flagged or r.path is None)
    print(f"\n{5 - flagged}/5 clean. Files in {work_dir / 'out'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Confirm the OpenAI account can use gpt-image models**

gpt-image models require API organization verification before the first call.

```bash
export OPENAI_API_KEY=sk-...
.venv/bin/python -c "
from openai import OpenAI
c = OpenAI()
print([m.id for m in c.models.list() if 'image' in m.id])
"
```

Expected: a list including `gpt-image-2`. If it 403s with a verification message, complete organization verification in the OpenAI dashboard before continuing.

- [ ] **Step 3: Run the first real batch**

```bash
.venv/bin/python scripts/live_run.py
```

Expected: five headlines printed and five files in `out/live/out/`. Takes a few minutes and costs roughly $1.

- [ ] **Step 4: Look at all five and judge them against the samples**

```bash
open out/live/out/*.png "sample outcomes/Claude-Vs-ClaudeFOS1.png"
```

Check honestly, and write down the answers — they decide what happens next:
1. Is every headline fully legible, spelled correctly, and inside the frame?
2. Do the actors still look like the actors?
3. Does it read as the same house style as the samples?
4. How many of the five needed a reroll or came back flagged?

- [ ] **Step 5: Record the outcome in the spec's risk section**

Append a short "First live run, 2026-XX-XX" note to §13 of the spec with the flagged count and what the failures looked like. If the flagged rate is above roughly 2 in 5, the spec's escape hatch is now live work: model generates background and subject, headline composited deterministically with HTML and headless Chrome, following the pattern in `Youtube Thumbnail/thumbnails/source/build.js`. Raise it before adding features.

- [ ] **Step 6: Commit**

```bash
git add scripts/live_run.py docs/superpowers/specs/
git commit -m "feat: live run script and first real batch findings"
```

---

### Task 15: Deploy to Streamlit Community Cloud

**Files:**
- Create: `README.md`
- Modify: `.gitignore` (add `out/` if not present — it already is)

- [ ] **Step 1: Create the Google Cloud OAuth client**

In a free Google Cloud project (no billing needed — Drive API and OAuth clients cost nothing):
1. APIs & Services → Library → enable **Google Drive API**.
2. OAuth consent screen → User type **Internal**. Internal apps may use sensitive scopes with no verification review and are automatically limited to `@datarails.com`.
3. Credentials → Create credentials → **OAuth client ID** → Web application.
4. Authorized redirect URIs: add both `http://localhost:8501` and the eventual `https://<app-name>.streamlit.app`.
5. Copy the client ID and secret.

- [ ] **Step 2: Verify locally end to end before deploying**

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# fill in the four values, with REDIRECT_URI = "http://localhost:8501"
.venv/bin/streamlit run app.py
```

Sign in, paste a real Drive link to an ad, generate. Confirm: the grid renders, a download works, and **Save to Drive** creates the folder. This is the last checkpoint before anyone else touches it.

- [ ] **Step 3: Write `README.md`**

```markdown
# YT Thumbnail Creator

Paste a Google Drive link to a video ad, get five YouTube ad thumbnails at
1920×1080. Built for the Datarails marketing team.

## How it works

ffmpeg pulls scene-change frames and the audio out of the ad. A planner model
reads them and writes five hook/treatment concepts. Five concurrent
`gpt-image-2` calls render them, using frames from the ad itself as references
so the actors stay themselves, plus the approved thumbnails in `refs/style/` so
the house style holds. Every render is read back at feed size to confirm the
headline is legible before you see it.

## The five variants

| # | Hook | Treatment |
|---|------|-----------|
| 1 | stat | split screen |
| 2 | question | face close-up |
| 3 | conflict | full bleed |
| 4 | pain | text dominant |
| 5 | outcome | product forward |

## Run it locally

```bash
/opt/homebrew/bin/python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then fill it in
.venv/bin/streamlit run app.py
```

Requires ffmpeg (`brew install ffmpeg`). On Streamlit Cloud it comes from
`packages.txt`.

## Tests

```bash
.venv/bin/pytest
```

Every API call is mocked. For a real batch, which costs about $1:

```bash
.venv/bin/python scripts/live_run.py
```

## Tuning the creative

Copy rules live in `prompts/planner.md`, the house-style description in
`prompts/render.md`. Neither requires touching Python. Approving a thumbnail
with ⭐ adds it to `refs/winners/`, which tightens the style over time; at least
one locked reference from `refs/style/` is always included, which bounds drift.

## Cost

Roughly $0.15 per thumbnail, so about $1 per batch of five.
```

- [ ] **Step 4: Push to GitHub**

```bash
gh repo create datarails-yt-thumbnail-creator --private --source=. --remote=origin
git add README.md && git commit -m "docs: README"
git push -u origin main
```

- [ ] **Step 5: Deploy on Streamlit Community Cloud**

1. share.streamlit.io → New app → pick the repo, branch `main`, main file `app.py`.
2. Advanced settings → Secrets → paste all four values, with `REDIRECT_URI` set to the app's real `https://<name>.streamlit.app` URL.
3. Deploy, then add that same URL to the Google OAuth client's authorized redirect URIs if it changed.

- [ ] **Step 6: Verify the deployed app**

Sign in as yourself, run one batch, download one file, save to Drive. Then have one teammate sign in and run a batch — that confirms the multi-user path, which is the whole reason it's hosted.

- [ ] **Step 7: Commit and push**

```bash
git add -A && git commit -m "chore: deployment configuration" && git push
```

---

## Self-Review

**Spec coverage:** every section of the spec maps to a task — §3 user flow → Tasks 13; §4 module contracts → Tasks 1–12 one-to-one; §5 data model → Task 3; §6 prompts → Task 7; §7 QA gate → Tasks 10 and 11; §8 auth → Task 12; §9 deployment → Task 15; §10 error handling → distributed across the module that owns each failure, with `pipeline` owning partial batches; §11 testing → each task's test step, with the sample ad as fixture in Task 2 and the style pack in Task 6; §12 cost → verified in Task 14; §13 risks → Task 14 Step 5 makes the escape-hatch decision explicit and data-driven.

**Type consistency:** `Variant` and `BatchPlan` field names are identical everywhere they appear (Tasks 3, 8, 9, 11, 13). `finalize` returns a `Path` that may differ in suffix from its argument, and Task 11 uses the return value rather than the argument. `render_variant`'s signature in Task 9 matches every call site in Task 11, including `extra_instruction` and `frame_override`. `qa.check(path, intended_headline, client=None)` matches its call in `pipeline`. `pick_refs(treatment, limit)` matches both its definition and the monkeypatch in Task 9's tests.

**Known gap, deliberate:** no automated test asserts that a real `gpt-image-2` render is legible — that is unknowable without spending money, which is exactly what Task 14 does, and Step 5 turns the result into a decision rather than a shrug.
