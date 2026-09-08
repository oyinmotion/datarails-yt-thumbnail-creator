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
    monkeypatch.setattr(
        render, "pick_refs", lambda style, treatment, limit=3, people_in_ad=True: [ref]
    )


def test_render_returns_decoded_bytes(frames):
    assert render.render_art(_variant(), frames, client=FakeClient()) == RAW


def test_render_sends_the_configured_model_size_and_quality(frames):
    client = FakeClient()
    render.render_art(_variant(), frames, client=client)
    call = client.images.calls[0]
    assert call["model"] == IMAGE_MODEL
    assert call["size"] == GEN_SIZE            # 16:9 is the default ratio
    assert call["quality"] == "high"
    assert call["output_format"] == "png"
    assert call["n"] == 1


def test_each_ratio_asks_for_its_own_generation_size(frames):
    from src.config import RATIOS
    for ratio, ((gw, gh), _final) in RATIOS.items():
        client = FakeClient()
        render.render_art(_variant(), frames, client=client, ratio=ratio)
        assert client.images.calls[0]["size"] == f"{gw}x{gh}"


def test_render_never_sends_input_fidelity(frames):
    """gpt-image-2 rejects it; it processes inputs at high fidelity already."""
    client = FakeClient()
    render.render_art(_variant(), frames, client=client)
    assert "input_fidelity" not in client.images.calls[0]


def test_render_sends_the_ad_frame_first_then_style_refs(frames):
    client = FakeClient()
    render.render_art(_variant(), frames, client=client)
    images = client.images.calls[0]["image"]
    assert len(images) == 2          # one ad frame + one style ref


def test_split_screen_sends_both_frames(frames):
    client = FakeClient()
    render.render_art(
        _variant(second="scene_001.jpg"), frames, client=client,
    )
    assert len(client.images.calls[0]["image"]) == 3


def test_image_array_never_exceeds_the_api_limit(frames, monkeypatch):
    monkeypatch.setattr(
        render, "pick_refs",
        lambda style, treatment, limit=3, people_in_ad=True: list(frames.values()) * 20,
    )
    client = FakeClient()
    render.render_art(_variant(), frames, client=client)
    assert len(client.images.calls[0]["image"]) <= 16


def test_unknown_frame_id_raises(frames):
    with pytest.raises(render.RenderError, match="frame"):
        render.render_art(_variant(frame="nope.jpg"), frames,
                              client=FakeClient())


def test_extra_instruction_is_appended_to_the_prompt(frames):
    client = FakeClient()
    render.render_art(
        _variant(), frames, client=client,
        extra_instruction="The previous attempt painted sparks into the reserved area.",
    )
    assert "painted sparks into the reserved area" in client.images.calls[0]["prompt"]


def test_the_prompt_carries_no_headline_and_reserves_the_zone(frames):
    client = FakeClient()
    render.render_art(_variant(), frames, client=client)
    prompt = client.images.calls[0]["prompt"]
    assert "47K OVER" not in prompt
    assert "Reserve a calm area" in prompt


def test_a_style_override_reaches_the_prompt(frames):
    client = FakeClient()
    render.render_art(_variant(), frames, client=client, style="dark_cinematic")
    assert "near-black" in client.images.calls[0]["prompt"].lower()


def test_moderation_refusal_becomes_render_blocked(frames):
    class Blocked(Exception):
        pass

    client = FakeClient(error=Blocked("moderation_blocked: request rejected"))
    with pytest.raises(render.RenderBlocked):
        render.render_art(_variant(), frames, client=client)


def test_other_api_errors_become_render_error(frames):
    client = FakeClient(error=RuntimeError("503 service unavailable"))
    with pytest.raises(render.RenderError):
        render.render_art(_variant(), frames, client=client)


def test_rate_limit_with_rejected_is_not_content_blocked(frames):
    """Rate-limit errors contain 'rejected' but are NOT content refusals."""
    client = FakeClient(error=RuntimeError("429 request rejected: rate limit exceeded"))
    with pytest.raises(render.RenderError) as exc_info:
        render.render_art(_variant(), frames, client=client)
    assert not isinstance(exc_info.value, render.RenderBlocked)


def test_genuine_content_policy_violation_is_blocked(frames):
    """Content policy violations must raise RenderBlocked, not plain RenderError."""
    client = FakeClient(error=RuntimeError("content_policy_violation: unsafe image"))
    with pytest.raises(render.RenderBlocked):
        render.render_art(_variant(), frames, client=client)


class FakeImagesReturning:
    """A response whose shape is not what we expect."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def edit(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _client_returning(response):
    client = FakeClient()
    client.images = FakeImagesReturning(response)
    return client


def test_an_empty_data_list_becomes_a_render_error(frames):
    """Decoding used to sit outside the try, so this raised IndexError — which
    pool.map re-raises, killing all five variants."""
    response = type("R", (), {"data": []})()
    with pytest.raises(render.RenderError):
        render.render_art(_variant(), frames,
                              client=_client_returning(response))


def test_a_missing_data_attribute_becomes_a_render_error(frames):
    response = type("R", (), {})()
    with pytest.raises(render.RenderError):
        render.render_art(_variant(), frames,
                              client=_client_returning(response))


def test_a_null_b64_json_becomes_a_render_error(frames):
    response = type("R", (), {
        "data": [type("D", (), {"b64_json": None})()],
    })()
    with pytest.raises(render.RenderError):
        render.render_art(_variant(), frames,
                              client=_client_returning(response))


def test_a_malformed_payload_is_not_reported_as_a_content_block(frames):
    """A bad payload must not be mistaken for a moderation refusal — the
    pipeline reacts to those by rerolling with a different frame."""
    response = type("R", (), {"data": []})()
    with pytest.raises(render.RenderError) as exc_info:
        render.render_art(_variant(), frames,
                              client=_client_returning(response))
    assert not isinstance(exc_info.value, render.RenderBlocked)


def test_the_shared_client_factory_sets_a_timeout():
    """No module may build a bare OpenAI() with the SDK's 600s default."""
    from src import config, openai_client
    import inspect

    source = inspect.getsource(openai_client.get_client)
    assert "timeout" in source and "max_retries" in source
    assert config.OPENAI_TIMEOUT_SECONDS < 600


def test_a_people_free_ad_gets_no_style_references(frames, monkeypatch):
    """The locked pack is thumbnails OF PEOPLE — sending one to a motion-graphic
    ad is how a stranger from another shoot appears in the output."""
    seen = {}
    monkeypatch.setattr(
        render, "pick_refs",
        lambda style, treatment, limit=3, people_in_ad=True: (
            seen.update(people_in_ad=people_in_ad) or []
        ),
    )
    client = FakeClient()
    render.render_art(_variant(), frames, client=client, people_in_ad=False)
    assert seen["people_in_ad"] is False
    # Only the ad's own frame goes up — nothing else.
    assert len(client.images.calls[0]["image"]) == 1


def test_a_people_free_ad_is_told_not_to_draw_a_person(frames):
    client = FakeClient()
    render.render_art(_variant(), frames, client=client, people_in_ad=False)
    prompt = client.images.calls[0]["prompt"]
    assert "must contain NO person" in prompt
    # The prose wraps, so normalise whitespace before matching a phrase.
    flat = " ".join(prompt.split())
    assert "Never copy a person out of a reference image" in flat


def test_only_the_ads_own_frames_are_sent_by_default(frames, monkeypatch):
    """No style references, so nothing can carry a stranger's face in.

    The autouse fake_refs fixture forces a reference in; this test removes it to
    check the real default.
    """
    monkeypatch.setattr(
        render, "pick_refs",
        lambda style, treatment, limit=3, people_in_ad=True: [],
    )
    client = FakeClient()
    render.render_art(_variant(), frames, client=client)
    assert len(client.images.calls[0]["image"]) == 1


def test_every_render_is_told_not_to_copy_a_face_from_a_style_image(frames):
    client = FakeClient()
    render.render_art(_variant(), frames, client=client)
    flat = " ".join(client.images.calls[0]["prompt"].split())
    assert "Never copy, trace or imitate a face from a style image" in flat


def test_every_render_is_told_to_count_the_people(frames):
    client = FakeClient()
    render.render_art(_variant(), frames, client=client)
    flat = " ".join(client.images.calls[0]["prompt"].split())
    assert "Count the people in the reference frames" in flat
    assert "If they show nobody, the thumbnail contains nobody" in flat
