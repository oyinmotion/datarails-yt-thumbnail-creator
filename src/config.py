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
