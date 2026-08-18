"""Google Drive access: parse a link, download the ad, upload the results."""

from __future__ import annotations

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
    try:
        with local.open("wb") as handle:
            downloader = MediaIoBaseDownload(handle, request, chunksize=8 * 1024 * 1024)
            done = False
            while not done:
                _, done = downloader.next_chunk()
    except Exception as exc:
        local.unlink(missing_ok=True)
        raise DriveError("The download was interrupted. Try again.") from exc

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
