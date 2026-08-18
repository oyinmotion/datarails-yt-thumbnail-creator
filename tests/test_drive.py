import pytest

from src import drive


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
