# Text as a layer — design

**Date:** 2026-09-08 · **Status:** approved in review, awaiting implementation plan
**Piece 3 of 4** in the September improvement plan (after reliability, PR #3, and
speed/cost, PR #4; before output quality).

## Why

Today one `gpt-image-2` call produces a finished thumbnail, headline included.
That makes every change expensive and every headline a gamble:

- Editing a headline, trying a second line of copy, or fixing one weak tile all
  cost a full re-render (~$0.20 per image, ~1 minute) — so in practice nobody
  iterates; a batch is take-it-or-leave-it.
- OpenAI's own docs warn the model "can still struggle with precise text
  placement", so every render is read back by a vision model at feed size, and
  roughly one tile in a batch comes back flagged or re-rolled for type alone.

This design moves the headline out of the model and into a deterministic
typesetting layer. The model renders **art only**, with a reserved text zone;
Pillow sets the headline. Text becomes free to change, always legible, and
identical in quality across every tile.

## Decisions taken in review

| Question | Decision |
|---|---|
| Headline typeface | **Poppins, full family** (OFL, ships in repo). Black for the headline. Not Impact / Helvetica Neue (used in the approved refs; not redistributable), not free look-alikes. Keeps thumbnails inside the Datarails design system. |
| What iteration must allow | **All four:** edit headline in place · re-roll the art · swap the look · 2–3 headline variants on one art. |
| Persistence | **Session only**, as today. Nothing new is stored; download or Save to Drive when done. |
| Output quality | **Never pixelated.** Art cached at native generation size; text rendered at 2×; exactly one downscale to delivery size. |
| Approach | **A — text as a layer** (over B, hybrid: model text by default, composite only on QA failure — rejected because it keeps every iteration action expensive). |

Streamlit Cloud cannot run headless Chrome (installing it needs `apt`, which is
what broke the deploy in September), so compositing is Pillow, not HTML.

## 1. Architecture and data flow

Today: `render_variant → finalize → qa.check → stamp_logo`, one artifact.

New: **two artifacts per tile per ratio**, kept separately for the session.

```
art    = render_art(variant, frames, style, ratio)          # gpt-image-2. NO text. Zone reserved.
final  = compose(art, headline, style, treatment, ratio)    # typeset → stamp_logo → downscale
```

- **`art`** is the expensive, cached thing: `work_dir/art/{slug}_{ratio}.png`,
  stored at the model's native generation size (2048×1152, 1088×1088,
  1152×2048). Produced once per tile; replaced only by *re-roll* or *swap look*.
- **`final`** is cheap and disposable: `work_dir/out/{slug}_{ratio}.png`,
  rebuilt in ~100 ms whenever the headline or the style's type treatment
  changes. Downloads, the zip and Save-to-Drive read `final`, unchanged.

Module ownership:

- **`src/typeset.py`** (new) owns fonts, zones, type treatments, fitting and the
  scrim. Nothing else touches text.
- **`src/render.py`**: `render_variant` becomes `render_art`; every text
  instruction leaves the prompt and the zone-reservation instruction enters.
- **`src/pipeline.py`**: `_one_render` = `render_art` (with the likeness gate)
  + `compose`. `generate_batch` keeps its hard contract — one row per concept,
  every ratio — so the grid code is untouched.
- **`prompts/render.md`** rewritten (no text, reserved zone, actor placement per
  treatment). **`prompts/qa_legibility.md`** retired. **`prompts/qa_likeness.md`**
  unchanged.
- **`assets/fonts/poppins/`**: the full family plus `OFL.txt`; loaded only by
  `typeset.py`.

Session model as today: `st.session_state["work_dir"]` holds both trees;
per-tile edits mutate `final` in place and rerun.

## 2. The typesetting engine (`src/typeset.py`)

**Fonts.** Poppins Black for the headline. Only Black is used in this piece. The
approved refs' small caption pill ("Who's right?") is **out of scope**: the
planner produces one headline, and a caption is a plan-schema change that
belongs to piece 4 (output quality).

**Zones** are fixed per treatment × ratio, as canvas fractions. Deterministic
beats face detection; the render prompt places the actor so the zone stays clear.

| Treatment | 16:9 and 1:1 zone (x, y as % of canvas) | 9:16 zone | Actor placement |
|---|---|---|---|
| split_screen | centre column x 30–70, y 12–60 | centre band y 8–36 | one actor each side |
| face_closeup | left half x 6–52, y 10–70 | top y 8–40 | right half |
| full_bleed | top-centre band x 15–85, y 8–44 | top y 8–38 | lower two-thirds |
| text_dominant | left x 6–62, y 10–72 | top y 8–50 | offset right, still large |
| product_forward | top band x 10–90, y 8–30 | top y 8–30 | below the band |

No zone enters the bottom 14 % of the canvas: that is where `stamp_logo`
prefers to land, so logo and headline never compete.

**Fitting.** Uppercase. Greedy wrap into 1–3 lines. Choose the largest font
size at which every line fits the zone's width and the block fits its height.
**Floor:** cap height ≥ 7 % of canvas height, so at 320 px feed width the type
is still ≥ 14 px tall. If the headline cannot fit at the floor, typeset at the
floor anyway and flag the tile *"headline too long for this layout"* — a
deterministic flag replacing today's vision flag. `MAX_HEADLINE_WORDS` stays 5.

**Type treatment per style**, taken from `STYLE_BRIEF` so the two cannot drift:

| Style | Fill | Stroke | Shadow | Alignment |
|---|---|---|---|---|
| house_energy | white | navy `#0C142B`, ~7 % of size | hard, offset ~6 % of size, 65 % alpha | centre |
| dark_cinematic | cream `#FFF8EE` | none | none | centre |
| flat_graphic | navy on a light zone, cream on a dark zone (sample zone luminance) | none | none | left, hard to the layout |
| clean_corporate | navy `#0C142B` | none | none | centre |

Alignment follows treatment where it matters: `face_closeup` and
`text_dominant` are left-aligned; `split_screen`, `full_bleed`,
`product_forward` centred.

**Rendering.** The text layer is drawn at 2× the art's size and composited down
onto the native-size art; the assembled canvas is then downscaled once
(Lanczos) to delivery size. Text edges are anti-aliased; the art is resampled
exactly once.

**If the model paints into the zone anyway.** Score the zone's busyness with
`branding`'s edge-energy measure before typesetting. Busy → one re-roll with a
sharper reservation instruction. Still busy → typeset over a soft, style-matched
scrim (dark for light type, light for dark type) so legibility is guaranteed.
Never a hard failure.

## 3. Pipeline and QA

**Resolution.** `art` is cached at generation size; typesetting happens on that
canvas; one Lanczos downscale yields 1920×1080 / 1080×1080 / 1080×1920. Every
edit, variant, re-roll and swap goes through the same path, so an edited tile
is never a scaled copy of a scaled copy. Output is PNG; the JPEG-92 fallback
remains only for files over YouTube's 2 MB cap. `IMAGE_QUALITY` stays `"high"`.

**`render_art`** replaces `render_variant` inside `_one_render`. Same inputs;
the prompt reserves the zone and forbids text of any kind. Its success gate is
the **likeness check only** — `SAME` / `DIFFERENT` / `NOBODY` handling and
`people_in_ad` logic unchanged. A likeness failure triggers the existing single
re-roll with the invented-person correction; a blocked render still swaps frame
and retries. The legibility vision call is removed.

**`compose`** = zone-busy check → typeset → `stamp_logo` → downscale →
size/2 MB guard. Pure Pillow, no API, ~100 ms.

**Per-tile actions** are thin wrappers:

| Action | Calls | Cost |
|---|---|---|
| Edit headline | `compose` × 3 ratios | $0 |
| Headline variants | one text-only planner call, then `compose` on pick | $0 in images |
| Re-roll art | `render_art` + `compose` × 3 | 3 images |
| Swap look | `render_art` (other `STYLE_BRIEF`) + `compose` (other type treatment) × 3 | 3 images |

Each is a `RenderOutcome`, so the hard contract and the cost counters
(`attempts`, `billed`) keep working; the results footer shows the running actual
cost for the session.

**Flags.** Deterministic notes a tile can carry: *"headline too long for this
layout"*, *"set over a scrim — art ignored the text zone"*, and the existing
likeness note. `unverified` now means only "likeness check unavailable".

## 4. Results UI and session model

The grid is unchanged: three ratio tabs, three columns, one card per concept.
Each card gains a compact action row under the image:

- **Headline field** — prefilled with the planned headline. ⏎ re-composes the
  concept **at all three ratios** (a headline belongs to the concept, not to one
  size). No API call.
- **"3 more lines"** — one text-only planner call for three alternative
  headlines *for this hook type*, shown as pills; clicking one applies it.
  Nothing renders until you pick.
- **↻ Re-roll art** — 3 image calls, same hook/layout/style, headline re-applied.
  Price shown on the button; footer updates after.
- **Look ▾** — select one of the other three styles; re-renders the art and
  re-typesets with that style's treatment (3 image calls). Caption shows the new
  style; the row is marked *off-matrix*.

Every action runs under a per-card `st.status`; the rest of the grid stays
usable. Downloads, **Download all** and **Save to Drive** export the *current*
state of every tile, with an `_edited` suffix where the headline differs from
the plan.

Per-tile state (current headline, style, cost) lives in `st.session_state`
keyed by concept index. A new Generate wipes `work_dir/art` and `work_dir/out`,
as today.

## 5. Error handling and testing

One bad thing costs one tile, never the batch. `render_art` and `compose` both
run inside `_one_render`'s existing catch-everything boundary.

| Failure | Behaviour | Tile note |
|---|---|---|
| Model painted into the text zone | one re-roll with a sharper instruction, then scrim | "set over a scrim — art ignored the text zone" |
| Headline can't fit at the floor | typeset at the floor | "headline too long for this layout" |
| Font missing or unreadable | fall back to Poppins SemiBold if present, else `TypesetError`: tile fails, art kept | "couldn't set the headline" (art still downloadable) |
| Re-roll / swap API failure | existing `RenderError` path, backoff, note | unchanged |
| "3 more lines" planner failure | pills don't appear, field still editable | inline warning |
| Two actions on one tile | per-tile lock in session state; second is disabled | "working…" |

Cost counters count through every path; a failed call is `attempts += 1`,
`billed += 0`.

**Tests** follow the suite's conventions: every API mocked, Pillow real, ffmpeg
real against the sample ad.

- `tests/test_typeset.py` (new): every treatment × ratio zone lies inside the
  canvas and above the bottom 14 %; fitting picks the largest size that fits and
  never below the floor; 1/2/3-line wrapping; per-style fill/stroke/shadow; the
  luminance-driven flat-graphic colour; scrim when the zone is busy; **the
  anti-pixelation test** — text edges in the delivered PNG show intermediate
  values, not pure steps, and the art has been resampled once.
- `tests/test_render.py`: the prompt contains the zone reservation and no
  headline or text instruction.
- `tests/test_pipeline.py`: `render_art` + `compose` satisfy the hard contract;
  likeness re-roll still fires; no legibility vision call is ever made; art is
  cached at generation size and the final is exactly delivery size; re-roll,
  swap and edit wrappers count cost correctly.
- `tests/test_app_helpers.py`: pure helpers for the action row (price label,
  off-matrix marker, `_edited` naming).
- One integration test: synthetic 2048×1152 art → real Pillow → 1920×1080 PNG
  under 2 MB with the headline present in the zone (pixel sampling, not vision).

The retired `qa_legibility` tests are deleted with the prompt file.

## Out of scope (deliberately)

- The caption pill and any second text element — piece 4.
- Realigning `STYLE_BRIEF` prose with the approved refs (sticker outline, "VS"
  mark) — piece 4, after this changes the render prompt.
- Persisting an editable project to Drive — can be added later without redoing
  this design.
- Re-enabling `SEND_STYLE_REFS` — stays off.

## Success criteria

- Editing a headline on a finished batch takes under a second and costs $0.
- No tile in a batch is ever flagged for type legibility by a model; the only
  model-driven flag left is likeness.
- The delivered PNG is produced by exactly one downscale of native-resolution
  art; text is rendered at 2× — verified by test, not by eye.
- Re-roll, swap and variants each work on one tile without touching the others.
- All existing tests still pass; the suite grows to cover `typeset.py` fully.
