import base64

import pytest

from src import render
from src.config import GEN_SIZE, IMAGE_MODEL
from src.models import Variant

RAW = b"\x89PNG pretend image"


def _variant(treatment="split_screen", frame="scene_002.jpg", second=None):
    return Variant(
        index=1, hook_type="stat", treatment=treatment, headline="47K OVER",
        frame_id=frame, second_frame_id=second,
        scene_direction="orange versus blue, sparks at the seam",
        rationale="the ad names the overrun",
    )


class FakeImages:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def edit(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        payload = base64.b64encode(RAW).decode()
        return type("R", (), {
            "data": [type("D", (), {"b64_json": payload})()]
        })()


class FakeClient:
    def __init__(self, error=None):
        self.images = FakeImages(error)


@pytest.fixture
def frames(tmp_path):
    out = {}
    for name in ("scene_001.jpg", "scene_002.jpg"):
        p = tmp_path / name
        p.write_bytes(b"\xff\xd8\xff fake")
        out[name] = p
    return out


@pytest.fixture(autouse=True)
def fake_refs(tmp_path, monkeypatch):
    ref = tmp_path / "style_a.png"
    ref.write_bytes(b"\x89PNG fake")
    monkeypatch.setattr(render, "pick_refs", lambda treatment, limit=3: [ref])


def test_render_returns_decoded_bytes(frames):
    assert render.render_variant(_variant(), frames, client=FakeClient()) == RAW


def test_render_sends_the_configured_model_size_and_quality(frames):
    client = FakeClient()
    render.render_variant(_variant(), frames, client=client)
    call = client.images.calls[0]
    assert call["model"] == IMAGE_MODEL
    assert call["size"] == GEN_SIZE
    assert call["quality"] == "high"
    assert call["output_format"] == "png"
    assert call["n"] == 1


def test_render_never_sends_input_fidelity(frames):
    """gpt-image-2 rejects it; it processes inputs at high fidelity already."""
    client = FakeClient()
    render.render_variant(_variant(), frames, client=client)
    assert "input_fidelity" not in client.images.calls[0]


def test_render_sends_the_ad_frame_first_then_style_refs(frames):
    client = FakeClient()
    render.render_variant(_variant(), frames, client=client)
    images = client.images.calls[0]["image"]
    assert len(images) == 2          # one ad frame + one style ref


def test_split_screen_sends_both_frames(frames):
    client = FakeClient()
    render.render_variant(
        _variant(second="scene_001.jpg"), frames, client=client,
    )
    assert len(client.images.calls[0]["image"]) == 3


def test_image_array_never_exceeds_the_api_limit(frames, monkeypatch):
    monkeypatch.setattr(
        render, "pick_refs",
        lambda treatment, limit=3: list(frames.values()) * 20,
    )
    client = FakeClient()
    render.render_variant(_variant(), frames, client=client)
    assert len(client.images.calls[0]["image"]) <= 16


def test_unknown_frame_id_raises(frames):
    with pytest.raises(render.RenderError, match="frame"):
        render.render_variant(_variant(frame="nope.jpg"), frames,
                              client=FakeClient())


def test_extra_instruction_is_appended_to_the_prompt(frames):
    client = FakeClient()
    render.render_variant(
        _variant(), frames, client=client,
        extra_instruction="The previous attempt cut off the last word.",
    )
    assert "cut off the last word" in client.images.calls[0]["prompt"]


def test_moderation_refusal_becomes_render_blocked(frames):
    class Blocked(Exception):
        pass

    client = FakeClient(error=Blocked("moderation_blocked: request rejected"))
    with pytest.raises(render.RenderBlocked):
        render.render_variant(_variant(), frames, client=client)


def test_other_api_errors_become_render_error(frames):
    client = FakeClient(error=RuntimeError("503 service unavailable"))
    with pytest.raises(render.RenderError):
        render.render_variant(_variant(), frames, client=client)


def test_rate_limit_with_rejected_is_not_content_blocked(frames):
    """Rate-limit errors contain 'rejected' but are NOT content refusals."""
    client = FakeClient(error=RuntimeError("429 request rejected: rate limit exceeded"))
    with pytest.raises(render.RenderError) as exc_info:
        render.render_variant(_variant(), frames, client=client)
    assert not isinstance(exc_info.value, render.RenderBlocked)


def test_genuine_content_policy_violation_is_blocked(frames):
    """Content policy violations must raise RenderBlocked, not plain RenderError."""
    client = FakeClient(error=RuntimeError("content_policy_violation: unsafe image"))
    with pytest.raises(render.RenderBlocked):
        render.render_variant(_variant(), frames, client=client)
