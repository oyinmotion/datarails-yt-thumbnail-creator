"""YT Thumbnail Creator — Streamlit interface. No API logic lives here."""

from __future__ import annotations

import io
import logging
import os
import shutil
import tempfile
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import extra_streamlit_components as stx
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from src import auth, drive, session as dr_session
from src import plan as plan_module
from src.config import IMAGE_COST_USD
from src.models import DEFAULT_VARIANTS, MAX_VARIANTS, MIN_VARIANTS, STYLE_BRIEF
from src.pipeline import BatchOutcome, ThumbResult, generate_batch, reroll, retitle

log = logging.getLogger(__name__)

# --- brand ------------------------------------------------------------------
# Tokens from the Datarails design system (Design System_Datarails skill).
# The palette and font live in .streamlit/config.toml; this is the polish the
# theme cannot express — surface treatment, the pink accent rule, tile cards.
BRAND_CSS = """
<style>
  .block-container { max-width: 1180px; padding-top: 2.2rem; }
  h1, h2, h3 { letter-spacing: -0.01em; font-weight: 600; }
  h1 { font-size: 2.1rem; }
  /* The signature pink rule under the page title. */
  h1::after {
    content: ""; display: block; width: 3.25rem; height: 4px;
    background: #FA3576; border-radius: 2px; margin-top: .6rem;
  }
  /* Cards: paper on cream, hairline in the warm neutral. */
  div[data-testid="stImageContainer"] img,
  div[data-testid="stImage"] img {
    border-radius: .75rem; border: 1px solid #FFEFD9;
    box-shadow: 0 1px 2px rgba(12,20,43,.06), 0 8px 24px rgba(12,20,43,.05);
  }
  div[data-testid="stImageCaption"] {
    font-size: .8rem; font-weight: 600; color: #0C142B;
    text-transform: uppercase; letter-spacing: .06em;
  }
  /* Gold, not red, for the "couldn't verify" state — it is a caution. */
  div[data-testid="stAlertContainer"] { border-radius: .75rem; }
  .stButton > button, .stDownloadButton > button {
    border-radius: .625rem; font-weight: 600;
  }
  .stDownloadButton > button {
    border: 1px solid #FFEFD9; background: #FFFFFF; color: #0C142B;
  }
  .stDownloadButton > button:hover {
    border-color: #FA3576; color: #C81E5C;
  }
  div[data-testid="stTextInput"] input { background: #FFFFFF; }
  footer, #MainMenu { visibility: hidden; }
</style>
"""

LOGO = Path(__file__).parent / "assets" / "logo_dark.png"


def _apply_brand() -> None:
    """Datarails look: cream surface, navy ink, pink accent, Poppins."""
    st.markdown(BRAND_CSS, unsafe_allow_html=True)
    if LOGO.exists():
        st.logo(str(LOGO), size="large")


# --- pure helpers (tested) --------------------------------------------------
# Everything below the helpers runs inside main(). Streamlit executes this file
# with __name__ == "__main__", so the guard keeps `import app` importable from
# the test suite without firing the sign-in flow and st.stop().
def zip_bytes(results: list[ThumbResult]) -> bytes:
    """Every successful render, at every ratio, in one archive."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for result in results:
            for ratio, path in sorted(result.paths.items()):
                if path is not None:
                    archive.write(path, arcname=f"{ratio}/{download_name(result, ratio)}")
    return buffer.getvalue()


def batch_folder_name(video_name: str, stamp: str) -> str:
    return f"thumbnails — {Path(video_name).stem} — {stamp}"


def oauth_state_matches(expected: str | None, returned: str | None) -> bool:
    """Guard the OAuth callback against a forged `code` (login CSRF).

    Streamlit starts a fresh session on the page load that follows the Google
    redirect, so `expected` can legitimately be gone by the time the callback
    runs. Treat that as unverifiable rather than as an attack — refusing it
    would lock every user out — but refuse outright when a state IS on record
    and the callback carries a different one (or none).
    """
    if expected is None:
        return True
    return bool(returned) and returned == expected


def ad_display_name(video_name: str) -> str:
    """A readable name for the ad, from its Drive filename.

    Datarails ad files carry long encoded tails —
    "..._vid_16x9_47s_high_bus_awa_skit_aic_hall..." — which are noise in a
    header. Keep everything up to the first encoding marker, and tidy the
    separators.
    """
    stem = Path(video_name or "").stem
    for marker in ("_vid_", "_16x9_", "_9x16_", "_1x1_"):
        if marker in stem:
            stem = stem.split(marker)[0]
            break
    cleaned = " ".join(stem.replace("_", " ").replace("-", " ").split())
    return cleaned or "this ad"


def cost_line(outcome: BatchOutcome, planned_images: int) -> str:
    """What the batch actually cost, against what the caption promised.

    The pre-run caption assumes one image call per image. Rerolls (a QA retry,
    a blocked render tried again on another frame) add calls, and only calls
    that returned an image are billed. Say so, in one line.
    """
    billed = outcome.images_billed
    rerolls = max(0, outcome.render_calls - planned_images)
    line = f"Actual: {billed} images billed ≈ **${billed * IMAGE_COST_USD:.2f}**"
    if rerolls:
        line += f" — {rerolls} re-roll{'s' if rerolls != 1 else ''} on top of the {planned_images} planned"
    return line


def price_label(n_images: int) -> str:
    return f"↻ Re-roll art · {n_images} images ≈ ${n_images * IMAGE_COST_USD:.2f}"


def style_caption(result: ThumbResult) -> str:
    """hook · treatment · style, plus a marker when the tile left the matrix."""
    style = result.style or result.variant.style
    label = f"{result.variant.hook_type} · {result.variant.treatment} · {style}"
    if result.style and result.style != result.variant.style:
        label += " · off-matrix"
    return label


def same_headline(a: str, b: str) -> bool:
    """Case- and spacing-insensitive, so a retype of the same line is a no-op."""
    return " ".join((a or "").upper().split()) == " ".join((b or "").upper().split())


def download_name(result: ThumbResult, ratio: str) -> str:
    path = result.paths[ratio]
    headline_changed = result.headline and not same_headline(result.headline, result.variant.headline)
    caption_changed = (result.caption or None) != (result.variant.caption or None)
    if headline_changed or caption_changed:
        return f"{path.stem}_edited{path.suffix}"
    return path.name


def replace_result(outcome: BatchOutcome, new: ThumbResult) -> None:
    for i, row in enumerate(outcome.results):
        if row.variant.index == new.variant.index:
            outcome.results[i] = new
            return
    outcome.results.append(new)


def should_show_outcome(stored_link: str | None, current_link: str) -> bool:
    """The grid belongs to the link that produced it, not whatever is typed now."""
    if not stored_link:
        return False
    return stored_link.strip() == current_link.strip()


# --- secrets ---------------------------------------------------------------
def _read_secret(name: str) -> str:
    """st.secrets first, then the environment. Never raises for a missing file.

    When there is no secrets file at all — a local run without one, or a Cloud
    app whose secrets manager is empty — st.secrets.get() does not return its
    default: Streamlit raises from inside the lookup. On the Cloud that raise
    was the bare "Oh no" crash screen, with the real message ("add X in the
    secrets manager") never reached. Treat "no file" as "no value" instead.
    """
    try:
        value = st.secrets.get(name, "")
    except StreamlitSecretNotFoundError:
        value = ""
    return value or os.environ.get(name, "")


def _allowlist() -> set[str]:
    """Optional ALLOWED_EMAILS secret. Absent means every @datarails.com account."""
    return auth.parse_allowlist(_read_secret("ALLOWED_EMAILS"))


def _secret(name: str) -> str:
    value = _read_secret(name)
    if not value:
        st.error(
            f"`{name}` isn't configured. Add it in Streamlit's secrets manager."
        )
        st.stop()
    return value


# --- sign-in ---------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _pending_signins() -> dict[str, str]:
    """state -> code_verifier, for sign-ins that have left for Google.

    Cannot be session_state: returning from Google is a fresh page load with a
    fresh session. This lives in the app process, so it survives the round trip
    and lets the state check actually mean something — an unrecognised state is
    now a rejection rather than a shrug.
    """
    return {}


@st.cache_resource(show_spinner=False, ttl=600, max_entries=64)
def _exchange_once(
    code: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    allowlist: tuple[str, ...],
    code_verifier: str | None = None,
):
    """Exchange an authorization code exactly once.

    An authorization code is single-use: Google rejects a second attempt with
    invalid_grant. session_state cannot dedupe this, because arriving back from
    Google is a fresh page load with empty session_state, and Streamlit may run
    the script more than once per load. A resource cache keyed on the code is
    shared across runs and sessions, so the exchange happens once and every
    later run reads the result.
    """
    try:
        flow = auth.build_flow(client_id, client_secret, redirect_uri)
        return auth.exchange_code(
            flow, code, allowlist=set(allowlist), code_verifier=code_verifier
        )
    except auth.AuthError:
        raise
    except Exception as exc:
        # Anything unconverted would surface to the user as a redacted Streamlit
        # traceback with the cause only in the server logs. Name the class so the
        # next failure is diagnosable from the screen.
        log.exception("sign-in failed before or during the token exchange")
        raise auth.AuthError(
            "Sign-in didn't complete. Please try again. "
            f"(technical reason: {type(exc).__name__})"
        ) from exc


def dr_session_still_allowed(email: str) -> bool:
    """Re-check the allowlist on every returning visit.

    A cookie must not outlive someone's access: removing them from
    ALLOWED_EMAILS has to take effect on their next page load, not in a week.
    """
    return auth.is_allowed_email(email, _allowlist())


def _signing_secret() -> str:
    """Secret for session cookies. Falls back to the OAuth client secret.

    A dedicated SESSION_SECRET is better — rotating it signs everyone out — but
    falling back means persistence works with no extra configuration, and the
    fallback is already a server-side secret of the same sensitivity.
    """
    return _read_secret("SESSION_SECRET") or _secret("GOOGLE_CLIENT_SECRET")


@st.cache_resource(show_spinner=False)
def _live_sessions() -> dict[str, object]:
    """email -> Google credentials, for browsers holding a valid cookie.

    The cookie only proves who you are; the Drive credentials stay server-side.
    Lost on restart, which is why a returning user whose credentials are gone
    is sent through Google again rather than half-signed-in.
    """
    return {}


def _cookies():
    # Keyed so the component instance is stable across reruns.
    return stx.CookieManager(key="dr_yt_cookies")


def remember_browser() -> None:
    """Write the session cookie once we are safely past any rerun.

    Called from main(), not from the sign-in branch: a cookie written
    immediately before st.rerun() never survives the round trip.
    """
    if not st.session_state.pop("needs_cookie", False):
        return
    email = st.session_state.get("email", "")
    if not email:
        return
    try:
        _cookies().set(
            dr_session.COOKIE_NAME,
            dr_session.mint_token(email, _signing_secret()),
            expires_at=datetime.now() + timedelta(
                seconds=dr_session.DEFAULT_TTL_SECONDS
            ),
            key="set_session_cookie",
        )
    except Exception:
        # Losing persistence is a nuisance, not a reason to fail the page.
        log.warning("could not write the session cookie", exc_info=True)


def require_sign_in():
    if "credentials" in st.session_state:
        return st.session_state["credentials"]

    cookies = _cookies()
    token = cookies.get(dr_session.COOKIE_NAME)
    remembered = dr_session.read_token(token, _signing_secret())
    if remembered:
        credentials = _live_sessions().get(remembered)
        if credentials is not None and dr_session_still_allowed(remembered):
            st.session_state["credentials"] = credentials
            st.session_state["email"] = remembered
            return credentials
        if dr_session_still_allowed(remembered):
            # The cookie is good but the server forgot the Drive credentials —
            # the app restarted or went to sleep. Say so, because "sign in
            # again" with no explanation reads as a bug.
            st.session_state["session_expired_for"] = remembered

    code = st.query_params.get("code")
    if code:
        # A state mismatch is only meaningful when a state was recorded. The
        # redirect back from Google is a fresh page load, so session_state is
        # usually empty here and there is nothing to compare — see README.
        returned_state = st.query_params.get("state")
        pending = _pending_signins()
        code_verifier = pending.pop(returned_state, None) if returned_state else None
        if code_verifier is None:
            st.error(
                "That sign-in link is no longer valid — it may have been used "
                "already, or the app restarted while you were signing in. "
                "Please sign in again."
            )
            st.query_params.clear()
            st.stop()
        try:
            credentials, email = _exchange_once(
                code,
                _secret("GOOGLE_CLIENT_ID"),
                _secret("GOOGLE_CLIENT_SECRET"),
                _secret("REDIRECT_URI"),
                tuple(sorted(_allowlist())),
                code_verifier,
            )
        except auth.AuthError as exc:
            st.error(str(exc))
            st.query_params.clear()
            st.stop()
        st.session_state["credentials"] = credentials
        st.session_state["email"] = email
        _live_sessions()[email.strip().lower()] = credentials
        # The cookie is NOT written here. CookieManager persists it through a
        # frontend round trip, and the st.rerun() below aborts that round trip,
        # so the cookie silently never lands — which is why people were being
        # signed out mid-batch. remember_browser() writes it from main(), where
        # no rerun follows.
        st.session_state["needs_cookie"] = True
        st.query_params.clear()
        st.rerun()

    flow = auth.build_flow(
        _secret("GOOGLE_CLIENT_ID"),
        _secret("GOOGLE_CLIENT_SECRET"),
        _secret("REDIRECT_URI"),
    )
    url, state = auth.authorization_url(flow)
    pending = _pending_signins()
    if len(pending) > 256:          # bounded: abandoned sign-ins must not pile up
        pending.clear()
    pending[state] = flow.code_verifier
    st.session_state["oauth_state"] = state
    _apply_brand()
    st.title("YT Thumbnail Creator")
    st.write("Five thumbnail concepts from one Drive link, in three sizes.")
    if st.session_state.get("session_expired_for"):
        st.info(
            "The app restarted, so it needs you to sign in once more. "
            "One click — Google already knows you."
        )
    st.link_button("Sign in with your Datarails Google account", url,
                   type="primary")
    st.stop()


# --- results card ------------------------------------------------------------
def _render_card(outcome: BatchOutcome, result: ThumbResult, ratio: str, tab_label: str) -> None:
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

    # The action row is per CONCEPT: drawn once, on the 16:9 tab, acting on all
    # three sizes. Nothing to act on if no art was cached.
    if ratio != "16x9" or not result.art_paths:
        return
    busy = st.session_state.get(f"busy_{idx}", False)

    # Streamlit forbids writing a widget's state after the widget is drawn, so
    # anything that wants to change the field (a picked suggestion) leaves a
    # pending value here and we apply it BEFORE the widget is created.
    pending = st.session_state.pop(f"hl_pending_{idx}", None)
    if pending is not None:
        st.session_state[f"hl_{idx}"] = pending
    st.session_state.setdefault(f"hl_{idx}", result.headline)
    new_headline = st.text_input(
        "Headline", key=f"hl_{idx}", disabled=busy,
        help="Press Enter to re-set it on all three sizes. Free — no render.",
    )
    if new_headline.strip() and not same_headline(new_headline, result.headline) and not busy:
        with st.status("Re-setting the headline…", expanded=False):
            replace_result(outcome, retitle(result, headline=new_headline))
        st.rerun()

    st.session_state.setdefault(f"cap_{idx}", result.caption or "")
    new_caption = st.text_input(
        "Caption", key=f"cap_{idx}", disabled=busy, placeholder="optional, up to 3 words",
        help="A small pill under the headline, like \"Who's right?\". Leave empty for none.",
    )
    if " ".join(new_caption.split()) != (result.caption or "") and not busy:
        if len(new_caption.split()) > 3:
            st.warning("Captions are three words at most.")
        else:
            with st.status("Re-setting the caption…", expanded=False):
                replace_result(outcome, retitle(result, caption=new_caption))
            st.rerun()

    c1, c2, c3 = st.columns([1.0, 1.4, 1.0])
    if c1.button("3 more lines", key=f"more_{idx}", disabled=busy):
        got = plan_module.suggest_headlines(result.variant, outcome.plan.ad_summary, result.headline)
        if got:
            st.session_state[f"sugg_{idx}"] = got
        else:
            st.session_state[f"sugg_{idx}"] = []
            st.warning("Couldn't get suggestions just now — the field above still works.")
        st.rerun()
    suggestions = st.session_state.get(f"sugg_{idx}", [])
    if suggestions:
        pick = st.pills("Try one", suggestions, key=f"pick_{idx}", disabled=busy)
        if pick and not same_headline(pick, result.headline):
            st.session_state[f"hl_pending_{idx}"] = pick
            st.session_state.pop(f"pick_{idx}", None)
            replace_result(outcome, retitle(result, headline=pick))
            st.rerun()

    if c2.button(price_label(3), key=f"reroll_{idx}", disabled=busy):
        st.session_state[f"busy_{idx}"] = True
        try:
            with st.status(f"Re-rolling concept {idx}…", expanded=False):
                replace_result(outcome, reroll(result, outcome))
        finally:
            st.session_state[f"busy_{idx}"] = False
        st.rerun()

    others = [style for style in STYLE_BRIEF if style != (result.style or result.variant.style)]
    look_reset = st.session_state.pop(f"look_pending_reset_{idx}", False)
    if look_reset:
        st.session_state[f"look_{idx}"] = "(keep look)"
    look = c3.selectbox("Look", ["(keep look)"] + others, key=f"look_{idx}",
                        disabled=busy, label_visibility="collapsed")
    if look != "(keep look)":
        st.session_state[f"busy_{idx}"] = True
        st.session_state[f"look_pending_reset_{idx}"] = True
        try:
            with st.status(f"Re-rendering concept {idx} as {look}…", expanded=False):
                replace_result(outcome, reroll(result, outcome, style=look))
        finally:
            st.session_state[f"busy_{idx}"] = False
        st.rerun()


# --- main ------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="YT Thumbnail Creator", page_icon="🎬",
                       layout="wide")
    # src/openai_client.py constructs OpenAI() without an api_key argument,
    # which reads os.environ only — export the key before anything reaches the
    # pipeline so it's there regardless of how secrets are wired.
    os.environ.setdefault("OPENAI_API_KEY", _secret("OPENAI_API_KEY"))
    # Google returns a wider scope set than we request — it adds `openid` — and
    # oauthlib treats that as a tampering error and refuses the token. Relaxing
    # the check is the documented way to accept Google's own expansion.
    os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")
    credentials = require_sign_in()
    remember_browser()

    _apply_brand()
    st.title("YT Thumbnail Creator")
    st.caption(f"Signed in as {st.session_state.get('email', '')}")

    link = st.text_input(
        "Google Drive link to the ad",
        placeholder="https://drive.google.com/file/d/…/view",
    )

    variant_count = st.slider(
        "How many concepts?",
        min_value=MIN_VARIANTS, max_value=MAX_VARIANTS, value=DEFAULT_VARIANTS,
        help="Each concept is a different hook and a different look, and is "
             "rendered in all three sizes. Fewer concepts means a shorter wait "
             "and a smaller bill.",
    )
    st.caption(
        f"{variant_count} concept{'s' if variant_count != 1 else ''} × 3 sizes = "
        f"**{variant_count * 3} images**, roughly "
        f"**${variant_count * 3 * IMAGE_COST_USD:.2f}**"
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

    button_label = (
        "Generate 1 thumbnail" if variant_count == 1
        else f"Generate {variant_count} thumbnails"
    )
    if st.button(button_label, type="primary", disabled=not link):
        # A fresh attempt invalidates whatever the previous attempt left
        # behind — both the stale grid and the disk space it was using.
        previous_work_dir = st.session_state.pop("work_dir", None)
        if previous_work_dir:
            shutil.rmtree(previous_work_dir, ignore_errors=True)
        for key in ("outcome", "outcome_link", "parent_id", "video_name"):
            st.session_state.pop(key, None)

        try:
            file_id = drive.parse_file_id(link)
        except drive.DriveLinkError as exc:
            st.error(str(exc))
            st.stop()

        status = st.status("Starting…", expanded=True)
        work_dir = Path(tempfile.mkdtemp(prefix="ytthumb_"))
        video: Path | None = None
        succeeded = False
        try:
            status.update(label="Fetching the ad from Drive…")
            video, parent_id = drive.fetch_video(file_id, credentials, work_dir)

            # Name the ad in the progress label as soon as we know it: a batch
            # runs for minutes and this is the window where you forget which
            # one you started.
            ad_name = ad_display_name(video.name)
            st.session_state["video_name"] = video.name
            status.update(label=f"Working on “{ad_name}” — reading the ad…")

            outcome = generate_batch(
                video, work_dir,
                headline_override=headline_override or None,
                context=context or None,
                variant_count=variant_count,
                progress=lambda message: status.update(
                    label=f"“{ad_name}” — {message}"
                ),
            )
            status.update(label="Done.", state="complete")

            st.session_state["outcome"] = outcome
            st.session_state["outcome_link"] = link
            st.session_state["video_name"] = video.name
            st.session_state["parent_id"] = parent_id
            # Renders live in work_dir/out and the download buttons read them
            # back later, so keep the tree — just not the 86MB source video.
            st.session_state["work_dir"] = work_dir
            succeeded = True
        except Exception as exc:
            # Deliberately broad. openai.OpenAIError is not a RuntimeError, an
            # empty refs/style/ raises FileNotFoundError and a bad render
            # raises PIL's UnidentifiedImageError (an OSError) — none of which
            # were caught before, so each one showed the user a raw traceback
            # and leaked the work tree.
            log.exception("batch failed")
            status.update(label="Failed.", state="error")
            st.error(f"That didn't work: {exc}")
        finally:
            # The video is only needed while generate_batch runs — free it on
            # both the success and failure paths.
            if video is not None:
                video.unlink(missing_ok=True)
            # Every failure path frees the whole ~86MB work tree. The success
            # path must keep it: the download buttons read work_dir/out.
            if not succeeded:
                shutil.rmtree(work_dir, ignore_errors=True)
                st.stop()

    outcome = st.session_state.get("outcome")
    if outcome and should_show_outcome(st.session_state.get("outcome_link"), link):
        # A batch takes minutes, so by the time it lands you have forgotten
        # which ad you asked for. Name it.
        st.subheader(f"Thumbnails for {ad_display_name(st.session_state.get('video_name', ''))}")
        for warning in outcome.warnings:
            st.warning(warning)
        st.caption(f"**What the ad is about:** {outcome.plan.ad_summary}")
        st.caption(cost_line(outcome, planned_images=len(outcome.results) * 3))
        st.caption(
            "Every concept is rendered in **three sizes** — switch tabs to see "
            "the square and vertical versions."
        )

        ratio_tabs = {
            "16x9": "16:9 · YouTube",
            "1x1": "1:1 · Square",
            "9x16": "9:16 · Shorts",
        }
        for tab, (ratio, tab_label) in zip(
            st.tabs(list(ratio_tabs.values())), ratio_tabs.items()
        ):
            with tab:
                columns = st.columns(3)
                for position, result in enumerate(outcome.results):
                    with columns[position % 3]:
                        _render_card(outcome, result, ratio, tab_label)

        successful = [r for r in outcome.results if r.path]
        if successful:
            st.download_button(
                "Download all (.zip)", zip_bytes(outcome.results),
                file_name="thumbnails.zip", mime="application/zip",
            )
            if st.button("Save to Drive"):
                try:
                    url = drive.save_batch(
                        [p for r in successful for p in r.paths.values()],
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
