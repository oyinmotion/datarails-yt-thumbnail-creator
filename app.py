"""YT Thumbnail Creator — Streamlit interface. No API logic lives here."""

from __future__ import annotations

import io
import os
import shutil
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


def should_show_outcome(stored_link: str | None, current_link: str) -> bool:
    """The grid belongs to the link that produced it, not whatever is typed now."""
    if not stored_link:
        return False
    return stored_link.strip() == current_link.strip()


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
    # src/plan.py, src/render.py and src/qa.py construct OpenAI() with no
    # arguments, which reads os.environ only — export the key before anything
    # reaches the pipeline so it's there regardless of how secrets are wired.
    os.environ.setdefault("OPENAI_API_KEY", _secret("OPENAI_API_KEY"))
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
        except (drive.DriveError, RuntimeError) as exc:
            status.update(label="Failed.", state="error")
            st.error(str(exc))
            # Nothing in work_dir is retained when the batch failed.
            shutil.rmtree(work_dir, ignore_errors=True)
            st.stop()
        finally:
            # The video is only needed while generate_batch runs — free it on
            # both the success and failure paths.
            if video is not None:
                video.unlink(missing_ok=True)

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
