# YT Thumbnail Creator

Paste a Google Drive link to a video ad, get five thumbnail concepts, each
rendered at 1920×1080, 1080×1080 and 1080×1920. Built for the Datarails
marketing team.

## For the people using it

1. Open the app URL and sign in with your **@datarails.com** Google account.
2. Paste a Drive link to the ad. Any link you can open, the tool can fetch.
3. Choose **how many concepts** you want — anywhere from 1 to 5, five by
   default. The caption under the slider tells you how many images that is and
   roughly what it costs before you commit.
4. Press **Generate**. It takes a few minutes.
5. The results are headed with the ad's name, and split into three tabs —
   **16:9 · YouTube**, **1:1 · Square**, **9:16 · Shorts**. Every concept exists
   in all three; switch tabs to see and download them.
6. Download the ones you like, or **Save to Drive** to drop the whole batch,
   every size, into a dated folder beside the source ad.

Optional, under **Advanced**:

- **Headline override** — if you already know the line you want to test, all five
  use it and only the visuals vary.
- **Extra context** — campaign goal, who it's for, words to avoid.

### What you get

Up to five variants, deliberately different. The pairings are fixed so every
batch spans the space instead of clustering on one idea, and the rows are
ordered — asking for two gives you the top two, not a random two:

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
reads them and writes the hook/layout concepts. Concurrent `gpt-image-2` renders
use frames from the ad itself as references — which is what keeps the actors
looking like themselves — and nothing else. Every render is then read back by a
vision model before you see it.

The four approved thumbnails in `refs/style/` are **not** sent to the model
(`SEND_STYLE_REFS` in `src/config.py`, off). Every one of them is a finished
thumbnail showing the two Claude-vs-Claude actors, and handing the model
photographs of two specific men put those men into other ads' thumbnails. The
house look now travels as prose in `STYLE_BRIEF`, which describes palette,
lighting, subject treatment and type per style. Turn the switch back on only
with reference images that contain no people.

Renders are generated at 2048×1152 and downscaled to exactly 1920×1080:
`gpt-image-2` requires both edges to be multiples of 16, and 1080 is not
(1080 ÷ 16 = 67.5), so 1920×1080 cannot be requested directly. Files over
YouTube's 2 MB cap fall back to JPEG quality 92.

Every render carries the Datarails logo, composited from the real asset rather
than drawn by the model. It always sits on flat colour — never over texture,
type or graphics. Each candidate corner is scored for how busy it is, using
luminance spread plus edge energy so that a wall of small text scores as busy
even though it is only two colours. The calmest corner wins; if nothing on the
image is calm enough, a solid brand-coloured plate is drawn to create a flat
area, navy behind the light logo and cream behind the dark one. Preference runs
bottom-left first and bottom-right last, because YouTube stamps its own duration
badge in that corner.

If the ad contains no people — a motion graphic, a screen recording — the
thumbnails are built with no person at all and the house-style references are
withheld, because those references are finished thumbnails of a different ad and
the model will otherwise lift an actor out of them.

Roughly $0.60 per concept — three renders each — so about $3 for a full batch
of five, or $1.20 for two.

Five is the ceiling because every row is a distinct hook, layout and look;
a sixth would have to repeat a pairing and would give you two near-identical
concepts. Raising it means adding rows to `MATRIX` in `src/models.py`.

## Running it locally

```bash
/opt/homebrew/bin/python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then fill it in
.venv/bin/streamlit run app.py
```

Needs ffmpeg and ffprobe. Locally `brew install ffmpeg` is simplest; if they are
not on PATH the app falls back to the `static-ffmpeg` pip package, which is also
how Streamlit Cloud gets them. (There is deliberately no `packages.txt`: the
Cloud image's apt sources include an expired Debian 11 feed, and any apt step
fails before the app can start.)

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
`https://dt-yt-thumbnail-creator.streamlit.app`.

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

## Keeping it awake

Streamlit Community Cloud puts an app to sleep after roughly a week without
traffic, and whoever opens it next gets a "wake this app up" screen followed by
a cold start. `.github/workflows/keep-awake.yml` visits the URL every six hours
to stop the timer ever reaching that point.

Two things to know about it. GitHub disables scheduled workflows in a repository
with no commits for 60 days, so if this project goes quiet the pings stop too —
the Actions tab will say so, and re-enabling is one click. And the ping is a
reachability check, not proof the app is awake: a sleeping app still answers, so
the workflow passing means the URL responded, nothing more.

If it ever needs to move off GitHub, any uptime monitor pointed at the URL on a
6-hourly schedule does the same job.

## Known limits

- Sign-in state is held in the app process, not a cookie, so if the app restarts
  or sleeps while you are on Google's consent screen, the callback is rejected
  and you simply sign in again.
- `refs/winners/` is written to the app's filesystem, which is ephemeral on
  Streamlit Community Cloud. Starred references last for the session only.
- Sign-in is remembered for a week in a signed cookie, but the Drive
  credentials live in the app process. If Streamlit restarts or sleeps, the next
  visit asks for one more click — the app says so rather than looking broken.
- Sign-in is remembered for a week in a signed cookie. The credentials stay
  server-side and are lost when the app restarts, so an occasional re-sign-in is
  expected. Removing someone from `ALLOWED_EMAILS` takes effect on their next
  page load, not when the cookie expires.
- Style references are disabled, so the look is carried by prose rather than by
  example images. If the house style drifts over time, that is the trade for
  never importing a stranger's face.
- Renders get one reroll each. A variant that fails verification twice is still
  shown, flagged, so you always receive five tiles and decide for yourself.
