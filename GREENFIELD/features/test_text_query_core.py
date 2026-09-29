"""Focused unit coverage for bounded Text Query and VACE primitives."""

from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

from GREENFIELD.app_core.media import write_video
from GREENFIELD.features import text_media, text_models, text_retrieval, text_segmentation, video_edit
from GREENFIELD.features.text_models import GroundedBox


def _frames(count: int, height: int = 24, width: int = 32) -> list[np.ndarray]:
    """Produce small distinct RGB frames suitable for deterministic media tests."""
    return [np.full((height, width, 3), index * 10, dtype=np.uint8) for index in range(count)]


def test_interval_boundaries_and_non_overlapping_ranked_windows() -> None:
    """Clip endpoints clamp safely and retrieval NMS never duplicates a moment."""
    assert text_media.interval_start(0, 4, 10) == 0
    assert text_media.interval_start(9.9, 4, 10) == 6
    assert text_media.interval_start(5, 4, 10) == 3

    scores = np.asarray([0.1, 0.2, 1.0, 0.9, 0.1, 0.8], dtype=np.float32)
    timestamps = [0, 2, 4, 6, 8, 10]
    windows = text_retrieval.ranked_windows(scores, timestamps, 12, 4, 5)
    assert windows
    assert len(windows) < 5  # Six retrieval anchors collapse into separate clips.
    for left, right in zip(windows, windows[1:]):
        assert left.start_seconds >= 0
        assert left.end_seconds <= 12
        assert left.end_seconds <= right.start_seconds or right.end_seconds <= left.start_seconds


def test_smoothed_score_boundaries_average_only_the_represented_window() -> None:
    """A window score includes precisely the samples in its 4-second interval."""
    smoothed = text_retrieval.smoothed_window_scores(
        np.asarray([0.0, 1.0, 0.0]), [0.0, 2.0, 4.0], 4,
    )
    np.testing.assert_allclose(smoothed, [0.5, 1 / 3, 0.5])


def test_query_and_edit_duration_limits_are_actionable(monkeypatch) -> None:
    """Long source uploads fail before a decoder attempts a large read."""
    monkeypatch.setattr(
        text_media, "probe_video",
        lambda _path: text_media.VideoInfo(12, 3_601, 300 + 1 / 12, 32, 24),
    )
    with pytest.raises(ValueError, match="up to 300 seconds"):
        text_media.read_uniform_samples("too-long.mp4", 2, 300)

    monkeypatch.setattr(
        text_media, "probe_video",
        lambda _path: text_media.VideoInfo(12, 82, 82 / 12, 32, 24),
    )
    with pytest.raises(ValueError, match="81 frames"):
        text_media.read_editing_episode("too-long-episode.mp4", 12, 81)


def test_clip_decode_and_highlight_preserve_full_frame_geometry(tmp_path: Path) -> None:
    """Highlights overlay source pixels instead of cropping or resizing a clip."""
    source = tmp_path / "source.mp4"
    write_video(_frames(12), source, 12)
    decoded, fps = text_media.read_interval(str(source), 0, 1, 12)
    assert fps == 12
    assert len(decoded) == 12
    assert decoded[0].shape == (24, 32, 3)

    masks = np.zeros((len(decoded), 24, 32), dtype=bool)
    masks[:, 8:16, 10:22] = True
    highlighted = text_segmentation.highlighted_frames(decoded, masks)
    assert [frame.shape for frame in highlighted] == [frame.shape for frame in decoded]
    # A pixel well outside the contour is retained byte-for-byte.
    np.testing.assert_array_equal(highlighted[0][0, 0], decoded[0][0, 0])
    np.testing.assert_array_equal(highlighted[0][~masks[0]], decoded[0][~masks[0]])
    assert not np.array_equal(highlighted[0][10, 12], decoded[0][10, 12])


def test_grounding_success_and_no_match_are_explicit(monkeypatch) -> None:
    """The tracker returns a real target mask or no result, never a fabricated one."""
    frames = _frames(3)
    box = GroundedBox((4, 4, 20, 18), 0.9)
    tracked = np.zeros((3, 24, 32), dtype=bool)
    tracked[:, 4:18, 4:20] = True
    monkeypatch.setattr(text_segmentation, "ground_query", lambda _frame, _query, _threshold=0.25: box)

    def fake_track_steps(_frames, _box, _anchor, **_kwargs):
        if False:
            yield
        return tracked

    monkeypatch.setattr(text_segmentation, "track_box_steps", fake_track_steps)

    success = text_segmentation.ground_and_track(frames, "tree")
    assert success is not None
    assert success.box == box
    assert success.masks.any()

    monkeypatch.setattr(text_segmentation, "ground_query", lambda _frame, _query, _threshold=0.25: None)
    assert text_segmentation.ground_and_track(frames, "not present") is None


def test_missing_text_snapshot_names_the_owner_setup_command(tmp_path: Path) -> None:
    """Browser-side model access does not silently attempt an online download."""
    with pytest.raises(RuntimeError, match=r"GREENFIELD\.features\.text_setup"):
        text_models.require_text_checkpoint(tmp_path / "missing", "SigLIP")


def test_siglip_loader_passes_local_files_only(monkeypatch, tmp_path: Path) -> None:
    """A fake Transformers module records the offline flag without loading weights."""
    checkpoint = tmp_path / "siglip"
    checkpoint.mkdir()
    (checkpoint / "model_manifest.json").touch()
    calls: list[tuple[str, dict]] = []

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, _path, **kwargs):
            calls.append(("processor", kwargs))
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, _path, **kwargs):
            calls.append(("model", kwargs))
            return cls()

        def to(self, _device):
            return self

        def eval(self):
            return self

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoProcessor = FakeProcessor
    fake_transformers.AutoModel = FakeModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(text_models, "require_package", lambda *_args: None)
    monkeypatch.setattr(text_models, "load_text_settings", lambda: SimpleNamespace(semantic_checkpoint=checkpoint))
    monkeypatch.setattr(text_models, "_torch_dtype", lambda: "test-dtype")
    monkeypatch.setattr(text_models, "device", lambda: "cpu")
    text_models.siglip_model.cache_clear()
    try:
        text_models.siglip_model()
    finally:
        text_models.siglip_model.cache_clear()

    assert calls == [
        ("processor", {"local_files_only": True}),
        ("model", {"local_files_only": True, "dtype": "test-dtype"}),
    ]


def test_vace_mask_polarity_uses_white_for_generated_pixels(monkeypatch) -> None:
    """The mask passed to Wan VACE uses its documented black-preserve polarity."""
    captured: dict[str, object] = {}

    class FakePipeline:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(frames=[kwargs["video"]])

    monkeypatch.setattr(
        video_edit, "load_text_settings",
        lambda: SimpleNamespace(edit_max_side=64, edit_max_height=64, vace_steps=1, vace_guidance_scale=5.0),
    )
    monkeypatch.setattr(video_edit, "vace_pipeline", lambda: FakePipeline())
    monkeypatch.setattr(video_edit, "device", lambda: "cpu")
    frames = _frames(5)
    masks = np.zeros((5, 24, 32), dtype=bool)
    masks[:, :, :16] = True
    result = video_edit.edit_video(frames, masks, "make it rainy", 9)

    encoded_mask = np.asarray(captured["mask"][0])
    assert encoded_mask[4, 4] == 255
    assert encoded_mask[4, -4] == 0
    assert all(frame.shape == (24, 32, 3) for frame in result)
    assert video_edit.vace_frame_count(81) == 81
    assert video_edit.vace_frame_count(48) == 45



def test_vace_loader_passes_local_files_only(monkeypatch, tmp_path: Path) -> None:
    """Wan VACE follows the same offline-only checkpoint contract as retrieval."""
    checkpoint = tmp_path / "wan-vace"
    checkpoint.mkdir()
    (checkpoint / "model_manifest.json").touch()
    captured: dict[str, object] = {}

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            captured["path"] = path
            captured["kwargs"] = kwargs
            return cls()

        def to(self, _device):
            return self

    fake_diffusers = ModuleType("diffusers")
    fake_diffusers.WanVACEPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setattr(video_edit, "require_package", lambda *_args: None)
    monkeypatch.setattr(video_edit, "load_text_settings", lambda: SimpleNamespace(edit_checkpoint=checkpoint))
    monkeypatch.setattr(video_edit, "device", lambda: "cpu")
    video_edit.vace_pipeline.cache_clear()
    try:
        video_edit.vace_pipeline()
    finally:
        video_edit.vace_pipeline.cache_clear()

    assert captured["path"] == checkpoint
    assert captured["kwargs"]["local_files_only"] is True
