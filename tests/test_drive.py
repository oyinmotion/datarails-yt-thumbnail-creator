import pytest

from src import drive
from src.config import MAX_VIDEO_BYTES


def _fake_service(meta, downloader_bytes=b"video data"):
    class FakeFiles:
        def get(self, **kwargs):
            class R:
                def execute(self_inner):
                    return meta
            return R()

        def get_media(self, **kwargs):
            return object()

    class FakeService:
        def files(self):
            return FakeFiles()

    return FakeService()


class OneShotDownloader:
    def __init__(self, handle, request, chunksize=None):
        self.handle = handle

    def next_chunk(self):
        self.handle.write(b"video data")
        return None, True


@pytest.mark.parametrize("url,expected", [
    ("https://drive.google.com/file/d/1AbC_dEfGhIjKlMnOpQr/view?usp=sharing",
     "1AbC_dEfGhIjKlMnOpQr"),
    ("https://drive.google.com/file/d/1AbC_dEfGhIjKlMnOpQr/view",
     "1AbC_dEfGhIjKlMnOpQr"),
    ("https://drive.google.com/open?id=1AbC_dEfGhIjKlMnOpQr",
     "1AbC_dEfGhIjKlMnOpQr"),
    ("https://drive.google.com/uc?export=download&id=1AbC_dEfGhIjKlMnOpQr",
     "1AbC_dEfGhIjKlMnOpQr"),
    ("https://drive.google.com/drive/u/0/file/d/1AbC_dEfGhIjKlMnOpQr/view",
     "1AbC_dEfGhIjKlMnOpQr"),
    ("  https://drive.google.com/file/d/1AbC_dEfGhIjKlMnOpQr/view  ",
     "1AbC_dEfGhIjKlMnOpQr"),
    ("1AbC_dEfGhIjKlMnOpQrStUvWxYz123456", "1AbC_dEfGhIjKlMnOpQrStUvWxYz123456"),
])
def test_parse_file_id_handles_every_drive_link_form(url, expected):
    assert drive.parse_file_id(url) == expected


@pytest.mark.parametrize("bad", [
    "",
    "   ",
    "not a url at all",
    "https://example.com/file/d/1AbC_dEfGhIjKlMnOpQr/view",
    "https://drive.google.com/drive/folders/1AbC_dEfGhIjKlMnOpQr",
])
def test_parse_file_id_rejects_bad_input(bad):
    with pytest.raises(drive.DriveLinkError):
        drive.parse_file_id(bad)


def test_folder_link_error_names_the_problem():
    with pytest.raises(drive.DriveLinkError, match="folder"):
        drive.parse_file_id("https://drive.google.com/drive/folders/1AbC_dEfGh")


def test_fetch_video_rejects_non_video_mime(monkeypatch, tmp_path):
    class FakeFiles:
        def get(self, **kwargs):
            class R:
                def execute(self_inner):
                    return {"id": "x", "name": "brief.pdf",
                            "mimeType": "application/pdf", "parents": ["p1"]}
            return R()

    class FakeService:
        def files(self):
            return FakeFiles()

    monkeypatch.setattr(drive, "_service", lambda creds: FakeService())
    with pytest.raises(drive.DriveError, match="not a video"):
        drive.fetch_video("x", creds=object(), dest_dir=tmp_path)


def test_fetch_video_permission_error_is_human_readable(monkeypatch, tmp_path):
    class FakeFiles:
        def get(self, **kwargs):
            class R:
                def execute(self_inner):
                    raise drive.HttpError(
                        resp=type("R", (), {"status": 404, "reason": "Not Found"})(),
                        content=b"{}",
                    )
            return R()

    class FakeService:
        def files(self):
            return FakeFiles()

    monkeypatch.setattr(drive, "_service", lambda creds: FakeService())
    with pytest.raises(drive.DriveError, match="access"):
        drive.fetch_video("x", creds=object(), dest_dir=tmp_path)


def test_fetch_video_download_failure_cleans_up_partial_file(monkeypatch, tmp_path):
    class FakeFiles:
        def get(self, **kwargs):
            class R:
                def execute(self_inner):
                    return {"id": "x", "name": "video.mp4",
                            "mimeType": "video/mp4", "parents": ["p1"]}
            return R()

        def get_media(self, **kwargs):
            return object()

    class FakeService:
        def files(self):
            return FakeFiles()

    class FakeDownloader:
        def __init__(self, *args, **kwargs):
            pass

        def next_chunk(self):
            raise IOError("Network connection lost")

    monkeypatch.setattr(drive, "_service", lambda creds: FakeService())
    monkeypatch.setattr(drive, "MediaIoBaseDownload", FakeDownloader)

    with pytest.raises(drive.DriveError, match="interrupted"):
        drive.fetch_video("x", creds=object(), dest_dir=tmp_path)

    # Verify the partial file was cleaned up
    partial_file = tmp_path / "video.mp4"
    assert not partial_file.exists()


def test_fetch_video_refuses_an_oversize_ad_before_downloading(
    monkeypatch, tmp_path
):
    """A 2GB ad used to fill the disk and then report an interrupted download,
    which invites a retry that fails the same way."""
    size = MAX_VIDEO_BYTES * 4
    meta = {"id": "x", "name": "huge.mp4", "mimeType": "video/mp4",
            "parents": ["p1"], "size": str(size)}
    monkeypatch.setattr(drive, "_service", lambda creds: _fake_service(meta))

    def must_not_download(*args, **kwargs):
        raise AssertionError("the download must not start")

    monkeypatch.setattr(drive, "MediaIoBaseDownload", must_not_download)

    with pytest.raises(drive.DriveError) as exc_info:
        drive.fetch_video("x", creds=object(), dest_dir=tmp_path)
    message = str(exc_info.value)
    assert "2000MB" in message           # names the actual size
    assert "500MB" in message             # and the limit
    assert not list(tmp_path.iterdir())


def test_fetch_video_accepts_a_file_inside_the_size_limit(monkeypatch, tmp_path):
    meta = {"id": "x", "name": "ok.mp4", "mimeType": "video/mp4",
            "parents": ["p1"], "size": str(MAX_VIDEO_BYTES - 1)}
    monkeypatch.setattr(drive, "_service", lambda creds: _fake_service(meta))
    monkeypatch.setattr(drive, "MediaIoBaseDownload", OneShotDownloader)
    local, parent = drive.fetch_video("x", creds=object(), dest_dir=tmp_path)
    assert local == tmp_path / "ok.mp4"
    assert parent == "p1"


def test_fetch_video_tolerates_a_missing_size_field(monkeypatch, tmp_path):
    """Some Drive items report no size; that must not block the download."""
    meta = {"id": "x", "name": "ok.mp4", "mimeType": "video/mp4",
            "parents": ["p1"]}
    monkeypatch.setattr(drive, "_service", lambda creds: _fake_service(meta))
    monkeypatch.setattr(drive, "MediaIoBaseDownload", OneShotDownloader)
    local, _ = drive.fetch_video("x", creds=object(), dest_dir=tmp_path)
    assert local.exists()


def test_fetch_video_sanitizes_a_name_containing_a_path_separator(
    monkeypatch, tmp_path
):
    """A Drive name is user-controlled text, not a path component."""
    meta = {"id": "x", "name": "../../evil.mp4", "mimeType": "video/mp4",
            "parents": ["p1"], "size": "1024"}
    monkeypatch.setattr(drive, "_service", lambda creds: _fake_service(meta))
    monkeypatch.setattr(drive, "MediaIoBaseDownload", OneShotDownloader)
    local, _ = drive.fetch_video("x", creds=object(), dest_dir=tmp_path)
    assert local == tmp_path / "evil.mp4"
    assert local.parent == tmp_path


@pytest.mark.parametrize("name,expected", [
    ("video.mp4", "video.mp4"),
    ("a/b/c.mp4", "c.mp4"),
    ("../../etc/passwd", "passwd"),
    ("folder\\ad.mp4", "ad.mp4"),
    ("  padded.mp4  ", "padded.mp4"),
    ("", "ad.mp4"),
    (".", "ad.mp4"),
    ("..", "ad.mp4"),
    ("/", "ad.mp4"),
])
def test_safe_filename_reduces_a_drive_name_to_a_basename(name, expected):
    assert drive.safe_filename(name) == expected
