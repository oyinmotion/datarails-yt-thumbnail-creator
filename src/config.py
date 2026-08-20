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

# --- OpenAI client ---------------------------------------------------------
# The SDK default is 600s. A stalled image render would hold one of the five
# workers for ten minutes and the user only sees results once all five land, so
# cap it well below that: a real gpt-image-2 render at 2048x1152 lands in
# roughly 30-90s, and 240s leaves generous headroom without hanging the batch.
OPENAI_TIMEOUT_SECONDS = 240.0
# SDK-level retries (connection errors, 429s) on top of our own reroll logic.
OPENAI_MAX_RETRIES = 2

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

# --- Input limits ----------------------------------------------------------
# The container this runs in has a few GB of disk and has to hold the download,
# the extracted frames and five 2048x1152 renders at once. Ads are normally
# under 100MB; anything past 500MB is a mistake worth catching before the
# download rather than after it fills the disk.
MAX_VIDEO_BYTES = 500 * 1024 * 1024

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

# --- Branding --------------------------------------------------------------
# Two variants so the mark stays legible on any background: the light one on
# dark art, the dark one on light art, chosen by measuring the corner.
LOGO_LIGHT = PROJECT_ROOT / "assets" / "logo_light.png"
LOGO_DARK = PROJECT_ROOT / "assets" / "logo_dark.png"
# Fraction of the image's WIDTH the logo spans. Judged at 320px feed size: much
# under this and the wordmark stops being readable.
LOGO_WIDTH_FRACTION = 0.16
LOGO_MARGIN_FRACTION = 0.035
# Preference order for where the logo goes. The flattest candidate wins; this
# order only breaks ties. bottom-right is last because YouTube stamps its own
# duration badge there.
LOGO_CORNERS = (
    "bottom-left", "top-left", "top-right", "bottom-centre", "top-centre",
    "bottom-right",
)
LOGO_CORNER = LOGO_CORNERS[0]        # kept for callers that want the default

# Above this "busy" score, no candidate is clean enough to sit the logo on bare,
# so a solid plate is drawn behind it. The score combines luminance spread with
# edge energy, so it catches texture, type and graphics — not just contrast.
LOGO_BUSY_THRESHOLD = 14.0
LOGO_PLATE_PAD_FRACTION = 0.45       # of the logo's height, per side
# Plate colours, from the Datarails design system. Navy behind the light logo,
# cream behind the dark one.
LOGO_PLATE_DARK = (12, 20, 43)       # --dr-navy  #0C142B
LOGO_PLATE_LIGHT = (255, 248, 238)   # --dr-cream #FFF8EE
LOGO_PLATE_ALPHA = 255               # fully opaque: "solid colour" means solid

# --- Output ratios ---------------------------------------------------------
# (generation size, final size). Generation edges must be multiples of 16 for
# gpt-image-2; the final sizes are the platform-native ones.
RATIOS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "16x9": ((2048, 1152), (1920, 1080)),   # YouTube thumbnail
    "1x1": ((1088, 1088), (1080, 1080)),    # square placements
    "9x16": ((1152, 2048), (1080, 1920)),   # Shorts / vertical
}
PRIMARY_RATIO = "16x9"
