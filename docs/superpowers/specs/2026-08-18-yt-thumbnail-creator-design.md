# YT Thumbnail Creator — Design

Date: 2026-08-18
Owner: Omer Yadgar (omer.y@datarails.com)
Status: Approved design, pending implementation plan

## 1. Purpose

A Datarails-internal web tool that turns a video ad in Google Drive into five
YouTube ad thumbnails in one batch. A user pastes a Drive link, presses one
button, and gets five finished 16:9 PNGs that follow the house style already
proven in `sample outcomes/`.

Thumbnails matter most for YouTube in-feed and Demand Gen placements, where the
thumbnail is the only creative a viewer evaluates before clicking. The tool
exists so that testing five hooks against one video costs a minute instead of a
designer's day.

**Users:** the Datarails marketing and creative team. Multi-user from v1, not a
personal script.

## 2. Non-goals

- No video editing, trimming, or re-encoding of the ad itself.
- No 9:16 or 1:1 output in v1. YouTube in-feed is 16:9; Shorts placements are a
  later phase if asked for.
- No thumbnail editor. Output is final PNGs; refinement happens by re-running or
  in a design tool.
- No performance data ingestion. The tool does not read CTR from Google Ads.
- No deterministic HTML/Chrome compositing. The previous manual pipeline in
  `Youtube Thumbnail/thumbnails/source/` is superseded; the image model renders
  the whole frame including type. (Kept in mind as a fallback — see §13.)

## 3. User flow

1. User opens the app URL, signs in with Google. Access is limited to
   `@datarails.com`.
2. User pastes a Drive link to a video ad.
3. Optionally opens **Advanced** and fills either or both of:
   - **Headline override** — exact text to use on all five.
   - **Extra context** — campaign goal, persona, words to avoid.
4. Presses **Generate 5 thumbnails**. Progress reads: fetching ad → reading it →
   rendering 5.
5. A grid of five appears, each labeled with its hook type and treatment. Per
   tile: **Download**. Below the grid: **Download all (.zip)**, **Save to
   Drive**, and per tile **Save as reference**.

## 4. Architecture

```
Drive link ──► drive.py ──► ad.mp4 (temp, deleted after run)
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
   probe.py: audio.m4a            probe.py: ~16 scene frames
              │                               │
              ▼                               │
   plan.py: transcribe ──► transcript ──► plan.py: GPT-5.6 Sol vision
                                              │
                                        BatchPlan (5 variants)
                                              │
                              ┌───────────────┴──────────────┐
                              ▼    5 concurrent renders      ▼
                        render.py (gpt-image-2)        render.py
                              │
                              ▼
                          qa.py ──► app.py grid
```

### Module contracts

Each module has one job, a typed interface, and no knowledge of the UI.

**`drive.py`**
- `fetch_video(url: str, creds) -> Path` — resolves any Drive URL form
  (`/file/d/<id>/view`, `?id=`, Shared Drive links), verifies the MIME type is a
  video, streams the file to a temp path.
- `save_batch(folder_name: str, files: list[Path], parent_id: str, creds) -> str`
  — creates a dated subfolder beside the source ad, uploads the PNGs, returns the
  folder URL.
- Depends on: `google-api-python-client`. Raises `DriveError` with a
  human-readable message on 403/404.

**`probe.py`** — pure ffmpeg, no AI, no network.
- `extract_frames(video: Path, max_frames: int = 16) -> list[Path]` — scene-change
  selection (`select='gt(scene,0.3)'`), scaled to 1280px wide, JPEG. If scene
  detection yields fewer than 6 frames, falls back to even-interval sampling.
- `extract_audio(video: Path) -> Path` — mono 16 kHz m4a, for transcription.

**`plan.py`** — the only place creative strategy lives.
- `build_plan(frames: list[Path], audio: Path, headline_override: str | None,
  context: str | None) -> BatchPlan`
- Transcribes audio with `gpt-transcribe`, then sends the transcript plus all
  frames to `gpt-5.6-sol` with a strict JSON schema response. Transcription failure
  is non-fatal: falls back to frames-only planning and notes it in the plan.

**`render.py`**
- `render_variant(variant: Variant, frames: dict[str, Path], refs: list[Path])
  -> Path` — one `images.edit` call to `gpt-image-2`.
- Called five times concurrently via `concurrent.futures.ThreadPoolExecutor`.
- Params: `size="2048x1152"` (native 16:9, within gpt-image-2's constraints:
  multiples of 16, max edge 3840), `quality="high"`, `output_format="png"`.
- Reference images passed to the edits endpoint: the variant's chosen frame from
  the ad, plus two or three house-style thumbnails. gpt-image-2 processes every
  input at high fidelity automatically — there is no `input_fidelity` parameter
  to set — which is what preserves the actors' likeness.

**`refs.py`**
- `style_refs() -> list[Path]` — the four locked samples in `refs/style/`.
- `winner_refs() -> list[Path]` — approved thumbnails in `refs/winners/`.
- `pick_refs(treatment: str) -> list[Path]` — returns two or three references for
  a call, preferring winners that share the treatment, falling back to style.
- `save_winner(png: Path, treatment: str) -> None`

**`qa.py`**
- `check(png: Path, variant: Variant) -> QAResult` — see §7.

**`app.py`** — Streamlit only. Auth, form, progress, grid, download, save. No API
logic, no prompt text.

## 5. Data model

`BatchPlan` and `Variant` are Pydantic models, and the same schema is sent to the
planner as a strict JSON schema so the model cannot return a shape we can't read.

```python
class Variant(BaseModel):
    index: int                 # 1-5
    hook_type: Literal["stat", "question", "conflict", "pain", "outcome"]
    treatment: Literal["split_screen", "face_closeup", "full_bleed",
                       "text_dominant", "product_forward"]
    headline: str              # <= 5 words
    frame_id: str              # filename of the chosen frame
    second_frame_id: str | None # for split_screen
    scene_direction: str       # what the background/composition should do
    rationale: str             # why this hook, from the ad's content

class BatchPlan(BaseModel):
    ad_summary: str
    transcript_used: bool
    variants: list[Variant]    # exactly 5
```

The hook/treatment pairing is **fixed**, not chosen by the model, so every batch
spans the space rather than clustering:

| # | hook_type | treatment       | intent |
|---|-----------|-----------------|--------|
| 1 | stat      | split_screen    | a number, two actors facing off |
| 2 | question  | face_closeup    | one face large, headline beside |
| 3 | conflict  | full_bleed      | energy burst, centered type |
| 4 | pain      | text_dominant   | type carries it, actor small |
| 5 | outcome   | product_forward | UI or FinanceOS mark visible |

The model chooses the headline, the frame, and the scene direction within each
row. If `headline_override` is set, all five use it verbatim and only the
treatment varies.

**Simplification, deliberate:** the plan does not carry actor bounding boxes.
GPT vision bounding boxes are unreliable, and gpt-image-2's automatic
high-fidelity input handling means a full frame works as a reference. We pass
whole frames.

## 6. Prompt design

**Planner prompt** carries the headline rules, because copy is where CTR is won:
- Five words maximum, twenty-two characters where possible.
- No full sentences, no ending period.
- Must be readable at 320px wide — the size a thumbnail actually appears in-feed.
- Draw the hook from what is actually said or shown in the ad. No invented
  claims, no invented statistics. If a number appears in the ad, it may be used;
  otherwise the stat variant uses a number from the extra-context field, or
  frames the same claim as a comparison without a figure. It stays hook_type
  `stat`; no sixth hook type exists.
- Curiosity, not clickbait: the first seconds of the ad must deliver what the
  thumbnail promises.

**Render prompt** per variant, assembled from a template:
- The house-style description (high-contrast, heavy condensed sans headline with
  a dark outline, dramatic lit background, subject cut out cleanly with a light
  rim, orange and deep-blue brand palette, optional handwritten annotation).
- The variant's `treatment` layout instruction.
- The exact `headline` text, quoted, with an instruction to render it exactly and
  legibly with no extra words, no watermark, no logo invention.
- The `scene_direction` from the planner.
- A statement that the reference images are the actors from this ad (preserve
  their faces and clothing) and the house style to match.

**Prompt text lives in `prompts/` as files**, not inline strings, so the creative
team can tune copy rules without touching Python.

## 7. QA gate

OpenAI's documentation states gpt-image-2 "can still struggle with precise text
placement and clarity." Our thumbnails are headline-dominant, so verification is
not optional.

1. **Hard checks** — valid PNG, dimensions ≥ 1280×720, file ≤ 2 MB (YouTube's
   thumbnail cap). Oversize files are re-encoded once at lower compression before
   being called a failure.
2. **Legibility check** — the render is downscaled to 320px wide and sent to
   `gpt-5.6-terra` vision with: *"Transcribe every word of text you can read in this
   image."* The transcription is compared to the intended headline, normalized
   for case and punctuation. A mismatch means the text warped, got cut off, or
   the model added words.
3. **One auto-reroll** per variant on failure, with the specific failure appended
   to the render prompt. A second failure still shows the tile, flagged `⚠️ text
   may be unreadable`, so the user always receives five tiles and decides.

**Content-filter refusals** get a separate path: caught by exception type,
labeled `blocked by content filter`, and rerolled once with a different frame as
the actor reference. This is expected occasionally on real faces even though the
footage is our own.

## 8. Auth and secrets

**One Google OAuth consent** covers identity and Drive:
- Scopes: `openid`, `email`, and `https://www.googleapis.com/auth/drive`.
- The OAuth consent screen is configured **Internal** in a free Google Cloud
  project. Internal apps may use sensitive scopes without Google's verification
  review, and access is automatically restricted to the Workspace — which gives
  us the `@datarails.com` gate for free. The app additionally checks the email
  domain after sign-in.
- No GCP billing is required. Drive API and OAuth clients are free; billing on
  the GCP project stays blocked and is irrelevant to this tool.
- Tokens live in Streamlit session state only. Nothing persisted server-side.

The full `drive` scope (rather than `drive.file`) is needed because saving results
means creating a subfolder inside a folder the app did not create.

**OpenAI key** is a single organization key in `st.secrets["OPENAI_API_KEY"]`,
server-side, shared by all users. Note that gpt-image models require API
organization verification on the OpenAI account before first use.

## 9. Deployment

- Streamlit Community Cloud, deployed from a GitHub repo, public URL gated by the
  Google sign-in.
- `requirements.txt`: `streamlit`, `openai`, `google-api-python-client`,
  `google-auth-oauthlib`, `pydantic`, `pillow`.
- `packages.txt`: `ffmpeg`.
- Secrets via Streamlit's secrets manager: `OPENAI_API_KEY`,
  `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `REDIRECT_URI`.
- Temp files under `tempfile.mkdtemp()`, removed in a `finally` block. The
  container is small: the ad is downloaded, probed, and deleted before rendering
  begins.

## 10. Error handling

Every failure produces a sentence a marketer can act on.

| Failure | Behavior |
|---------|----------|
| Link not a Drive URL | Inline validation before any network call |
| No permission / not found | "You don't have access to this file, or the link is wrong." |
| Not a video MIME type | "That link points to a document, not a video." |
| ffmpeg failure | "Couldn't read that video file." Logs stderr. |
| Transcription failure | Non-fatal. Plans from frames alone, notes it in the UI. |
| Planner returns invalid JSON | One retry, then a clear error. Schema-strict responses make this rare. |
| OpenAI rate limit / 5xx | Exponential backoff, three attempts per render |
| Some renders fail | Partial batch shown, failures named. Never a blank screen. |
| Drive save fails | Thumbnails stay downloadable; only the save is reported failed |

## 11. Testing

Real tests, on the parts where correctness is checkable:
- `probe.py` — frame extraction on the sample ad: correct count, real JPEGs,
  fallback path triggers on a video with no scene changes.
- `drive.py` — link parsing across every URL form, including malformed input.
- `plan.py` — schema validation: a valid plan parses, a plan with four variants or
  a bad treatment string is rejected. Model call mocked.
- `qa.py` — hard checks against fixture PNGs (oversize, undersize, valid);
  headline comparison normalization (case, punctuation, whitespace).
- End-to-end with mocked API calls, using the sample ad as fixture.
- One manual script, `scripts/live_run.py`, for real generations against the
  sample ad. Not part of the automated suite.

Fixtures come from what is already in the folder: the 47s skit in
`Sample input ad/` is the video fixture; the four PNGs in `sample outcomes/` are
copied into `refs/style/` as the locked style pack.

## 12. Cost

gpt-image-2 is priced per token, not per image: $8.00/1M image input, $30.00/1M
image output. A 2048×1152 high-quality render plus three reference images works
out to roughly $0.15, so a five-thumbnail batch costs approximately $0.75–$1.00,
plus a few cents for planning and QA calls. These are estimates from token
pricing and should be confirmed against real runs and OpenAI's cost calculator
once the first batches land.

## 13. Risks

**Text rendering is the main one.** If real renders show headlines warping often
enough that the QA gate rerolls constantly, the fallback is the pipeline that
already worked manually: let the model generate background and subject only, and
composite the headline deterministically with HTML and headless Chrome, as in
`Youtube Thumbnail/thumbnails/source/build.js`. This is not built in v1 — it is
the known escape hatch, and the QA gate's failure rate is the signal to take it.

**Likeness fidelity.** References keep the actors close, but not perfect. If
faces drift noticeably, tighter crops around the subject become the reference
instead of full frames.

**Content filters** on real faces may refuse intermittently. Handled, not
eliminated.

**Style drift as `refs/winners/` grows.** If winners accumulate unevenly, the
tool's taste skews. `refs/style/` stays locked and always contributes at least
one reference to every call, which bounds the drift.

## 14. Repo layout

```
YT Thumbnail creator/
├── app.py
├── requirements.txt
├── packages.txt
├── src/
│   ├── drive.py
│   ├── probe.py
│   ├── plan.py
│   ├── render.py
│   ├── refs.py
│   ├── qa.py
│   └── models.py
├── prompts/
│   ├── planner.md
│   └── render.md
├── refs/
│   ├── style/          # the four locked samples
│   └── winners/
├── tests/
├── scripts/live_run.py
├── Sample input ad/
├── sample outcomes/
└── docs/superpowers/specs/
```

## 15. Decisions locked

- Engine: OpenAI. `gpt-image-2` for rendering, `gpt-5.6-sol` for planning,
  `gpt-5.6-terra` for the QA legibility read, `gpt-transcribe` for audio.
  Model IDs live in one `config.py` constant block so a model refresh is a
  one-line change.
- Output: five PNGs at 2048×1152, 16:9, one batch per run.
- Variants: mixed matrix — five hooks paired with five treatments, fixed pairing.
- Ingestion: frames plus audio track, not full-video upload.
- Controls: Drive link, optional headline override, optional context box. No
  aspect-ratio picker in v1.
- Style: locked four-sample pack plus a growing winners pack.
- Delivery: preview grid, per-tile and zip download, optional Drive save-back.
- Hosting: Streamlit Community Cloud, Google sign-in restricted to
  `@datarails.com`.
