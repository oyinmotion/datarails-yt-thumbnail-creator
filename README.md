# YT Thumbnail Creator

Paste a Google Drive link to a video ad, get five YouTube ad thumbnails at
1920×1080. Built for the Datarails marketing team.

## For the people using it

1. Open the app URL and sign in with your **@datarails.com** Google account.
2. Paste a Drive link to the ad. Any link you can open, the tool can fetch.
3. Press **Generate 5 thumbnails**. It takes a few minutes.
4. Download the ones you like, or **Save to Drive** to drop the batch into a
   dated folder beside the source ad.

Optional, under **Advanced**:

- **Headline override** — if you already know the line you want to test, all five
  use it and only the visuals vary.
- **Extra context** — campaign goal, who it's for, words to avoid.

### What you get

Five variants, deliberately different. The pairings are fixed so every batch
spans the space instead of clustering on one idea:

| # | Hook | Layout | Look |
|---|------|--------|------|
| 1 | a number or hard comparison | split screen | house energy — orange/blue, sparks |
| 2 | a question | face close-up | dark cinematic — near-black, one hard light |
| 3 | the disagreement in the ad | full bleed | house energy |
| 4 | the pain the viewer recognises | type-dominant | flat graphic — solid colour fields |
| 5 | the payoff | product forward | clean corporate — light, calm, credible |

Two are the proven house style, so you always have something safe to ship. Three
are genuine alternatives, so you learn what actually wins.

If a tile is marked ⚠️, the tool rendered it but couldn't verify it — read the
note. It checks two things on every render: that the headline is still readable
at the size a thumbnail actually appears in a feed, and that the person in it is
really the actor from the ad rather than someone the model invented.

**⭐ Save as reference** teaches the tool your taste — but only for the current
session. Making a reference permanent means committing the file to
`refs/winners/` in this repo, because the deployed filesystem is wiped on every
restart.

## How it works

ffmpeg pulls scene-change frames and the audio out of the ad. A planner model
reads them and writes five hook/layout concepts. Five concurrent `gpt-image-2`
renders use frames from the ad itself as references — which is what keeps the
actors looking like themselves — plus the approved thumbnails in `refs/style/`
for the house-style slots. Every render is then read back by a vision model
before you see it.

Renders are generated at 2048×1152 and downscaled to exactly 1920×1080:
`gpt-image-2` requires both edges to be multiples of 16, and 1080 is not
(1080 ÷ 16 = 67.5), so 1920×1080 cannot be requested directly. Files over
YouTube's 2 MB cap fall back to JPEG quality 92.

Roughly $1 per batch of five.

## Running it locally

```bash
/opt/homebrew/bin/python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then fill it in
.venv/bin/streamlit run app.py
```

Needs ffmpeg (`brew install ffmpeg`). On Streamlit Cloud it comes from
`packages.txt`.

To test the pipeline without the UI or Drive — this is the cheapest way to check
a prompt change:

```bash
OPENAI_API_KEY=sk-... .venv/bin/python scripts/live_run.py
```

## Tests

```bash
.venv/bin/pytest
```

Every API call is mocked. ffmpeg runs for real against the sample ad, because
shelling out correctly is that module's entire job.

## Deploying it

Do these in order. The order matters: the OAuth redirect URI has to match the
deployed URL, and you only learn that URL in step 2.

**1. Enable the Drive API and prepare a Google Cloud project.**
Any free project works — no billing needed, Drive API and OAuth clients cost
nothing. APIs & Services → Library → enable **Google Drive API**.

**2. Create the Streamlit app.** share.streamlit.io → New app → this repo,
branch `master`, main file `app.py`. It is a private repo, so grant Streamlit
read access to private repositories when it asks. Deploy. It will fail to start
until step 4 — that is expected. Note the URL it gives you, e.g.
`https://dr-yt-thumbnails.streamlit.app`.

**3. Create the OAuth client.**
OAuth consent screen → User type **Internal**. This matters: an Internal app may
use the sensitive Drive scope with no Google verification review, and access is
automatically restricted to the Datarails Workspace — which is where the
`@datarails.com` gate actually comes from. Then Credentials → Create credentials
→ **OAuth client ID** → Web application. Under authorised redirect URIs add
both:

- `http://localhost:8501`
- the URL from step 2

Copy the client ID and secret.

**4. Add the secrets.** In the Streamlit app: Settings → Secrets, paste:

```toml
OPENAI_API_KEY = "sk-..."
GOOGLE_CLIENT_ID = "....apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "..."
REDIRECT_URI = "https://your-app.streamlit.app"
```

`REDIRECT_URI` must match the URL in step 3 exactly — no trailing slash. Reboot
the app.

**Optional but recommended:** add `ALLOWED_EMAILS` to restrict the tool to named
people instead of everyone in the Workspace. Every batch spends from one shared
OpenAI key, so this is a budget control as much as an access control:

```toml
ALLOWED_EMAILS = "omer.y@datarails.com, someone@datarails.com"
```

Commas, spaces or newlines all work. Leave it out to allow any @datarails.com
account. Changing it takes effect on the next sign-in — no redeploy needed. The
domain check still applies on top, so a typo cannot let an outsider in.

**5. Check it yourself, then have one teammate check it.** Sign in, run one
batch, download a file, save to Drive. Then ask someone else to do the same —
that is the only way to confirm the multi-user path, which is the whole reason
it is hosted.

### Prerequisite

The OpenAI account needs **API organization verification** completed before any
`gpt-image` call succeeds. Without it the models are invisible and every render
fails.

## Tuning the creative

No Python required for any of this:

- `prompts/planner.md` — hook strategy, the headline rules, how a frame is chosen.
- `prompts/render.md` — the rules that hold whatever the style, including likeness
  and the thumbnail fundamentals.
- `src/models.py` → `STYLE_BRIEF` — the four looks, each split into background,
  subject and type.

Model IDs live in one place, `src/config.py`, so a model refresh is a one-line
change.

## Known limits

- The OAuth `state` check is inert in practice: Streamlit starts a fresh session
  after Google's redirect, so the stored state is usually absent and the callback
  is allowed. The Internal-app Workspace restriction is what actually bounds
  access. Closing this properly needs a signed cookie or a server-side store.
- `refs/winners/` is written to the app's filesystem, which is ephemeral on
  Streamlit Community Cloud. Starred references last for the session only.
- Renders get one reroll each. A variant that fails verification twice is still
  shown, flagged, so you always receive five tiles and decide for yourself.
