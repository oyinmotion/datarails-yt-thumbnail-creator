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

from src import auth, drive, refs, session as dr_session
from src.pipeline import ThumbResult, generate_batch

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
                    archive.write(path, arcname=f"{ratio}/{path.name}")
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


def should_show_outcome(stored_link: str | None, current_link: str) -> bool:
    """The grid belongs to the link that produced it, not whatever is typed now."""
    if not stored_link:
        return False
    return stored_link.strip() == current_link.strip()


# --- secrets ---------------------------------------------------------------
def _allowlist() -> set[str]:
    """Optional ALLOWED_EMAILS secret. Absent means every @datarails.com account."""
    raw = st.secrets.get("ALLOWED_EMAILS", os.environ.get("ALLOWED_EMAILS", ""))
    return auth.parse_allowlist(raw)


def _secret(name: str) -> str:
    value = st.secrets.get(name, os.environ.get(name, ""))
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
    return st.secrets.get(
        "SESSION_SECRET", os.environ.get("SESSION_SECRET", "")
    ) or _secret("GOOGLE_CLIENT_SECRET")


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
        # Remember this browser so the next visit skips Google entirely.
        _live_sessions()[email.strip().lower()] = credentials
        cookies.set(
            dr_session.COOKIE_NAME,
            dr_session.mint_token(email, _signing_secret()),
            expires_at=datetime.now() + timedelta(
                seconds=dr_session.DEFAULT_TTL_SECONDS
            ),
            key="set_session_cookie",
        )
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
    st.write("Five thumbnail concepts from one Drive link, in three ratios.")
    st.link_button("Sign in with your Datarails Google account", url,
                   type="primary")
    st.stop()


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

    _apply_brand()
    st.title("YT Thumbnail Creator")
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

            outcome = generate_batch(
                video, work_dir,
                headline_override=headline_override or None,
                context=context or None,
                progress=lambda message: status.update(label=message),
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
                    refs.save_winner(
                        result.path,
                        result.variant.style,
                        result.variant.treatment,
                    )
                    st.success(
                        "Saved as a reference for this session only. The "
                        "server's file system is wiped on every redeploy, so "
                        "making it permanent means committing the file to "
                        "refs/winners/ in the repo."
                    )

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
