"""Boundary validation and propagation coverage for editable Text Query settings.

These tests exercise the page-boundary parser (``_text_effective_settings``), then
prove the resulting effective ``TextSettings`` actually reaches retrieval, grounding,
the low-specificity decision, and the scene-cache key.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from GREENFIELD.app_core.contracts import RunArtifacts, StageEvent
from GREENFIELD.features import pages, scene_cache, selection, text_segmentation
from GREENFIELD.features.scene_masks import PrimitiveMask
from GREENFIELD.features.text_media import VideoInfo
from GREENFIELD.features.text_models import GroundedBox
from GREENFIELD.features.text_retrieval import CandidateWindow
from GREENFIELD.features.text_segmentation import MaskTrack
from GREENFIELD.features.text_settings import load_text_settings


def _values_with(base, **overrides):
    """Return parser-order values with selected knobs replaced."""
    values = [getattr(base, name) for name in pages._EDITABLE_TEXT_FIELDS]
    for name, value in overrides.items():
        values[pages._EDITABLE_TEXT_FIELDS.index(name)] = value
    return values


def _tiny_frames(count: int = 3) -> list[np.ndarray]:
    """Make a tiny stationary RGB episode for deterministic adapter tests."""
    return [np.zeros((16, 20, 3), dtype=np.uint8) for _ in range(count)]


def test_text_effective_settings_default_parity() -> None:
    """Untouched controls rebuild the tracked settings exactly, including the key."""
    base = load_text_settings()
    effective = pages._text_effective_settings(base, *_values_with(base))
    assert effective == base
    assert scene_cache.scene_cache_key("a" * 64, 0, 8, 12, effective) == scene_cache.scene_cache_key(
        "a" * 64, 0, 8, 12, base
    )


def test_text_effective_settings_accepts_in_range_overrides() -> None:
    """Every knob is replaceable and keeps the base field's own type."""
    base = load_text_settings()
    effective = pages._text_effective_settings(
        base,
        *_values_with(
            base,
            query_max_source_seconds=600, query_sample_fps=5, query_clip_seconds=8,
            query_result_fps=15, query_max_results=3, query_explicit_candidate_pool=12,
            grounding_threshold=0.4, low_specificity_coverage=0.5,
            source_fov_degrees=95, scene_cache_version="epoch-2",
        ),
    )
    assert effective.query_max_source_seconds == 600
    assert effective.query_sample_fps == 5
    assert effective.query_clip_seconds == 8
    assert effective.query_result_fps == 15
    assert effective.query_max_results == 3
    assert effective.query_explicit_candidate_pool == 12
    assert effective.grounding_threshold == 0.4
    assert effective.low_specificity_coverage == 0.5
    assert effective.source_fov_degrees == 95
    assert effective.scene_cache_version == "epoch-2"
    assert isinstance(effective.query_max_results, int)
    assert isinstance(effective.query_max_source_seconds, float)


def test_text_effective_settings_rejects_bad_values_by_field_name() -> None:
    """Out-of-range, bool, non-numeric, NaN, empty, and wrong-count inputs fail loudly."""
    base = load_text_settings()

    with pytest.raises(ValueError, match="Text Query query_sample_fps"):
        pages._text_effective_settings(base, *_values_with(base, query_sample_fps=0))
    with pytest.raises(ValueError, match="Text Query query_max_source_seconds"):
        pages._text_effective_settings(base, *_values_with(base, query_max_source_seconds=3601))
    with pytest.raises(ValueError, match="Text Query query_max_results"):
        pages._text_effective_settings(base, *_values_with(base, query_max_results=51))
    with pytest.raises(ValueError, match="Text Query query_explicit_candidate_pool"):
        pages._text_effective_settings(base, *_values_with(base, query_explicit_candidate_pool=65))
    with pytest.raises(ValueError, match="Text Query grounding_threshold"):
        pages._text_effective_settings(base, *_values_with(base, grounding_threshold=1.5))
    with pytest.raises(ValueError, match="Text Query low_specificity_coverage"):
        pages._text_effective_settings(base, *_values_with(base, low_specificity_coverage=-0.1))
    with pytest.raises(ValueError, match="Text Query source_fov_degrees"):
        pages._text_effective_settings(base, *_values_with(base, source_fov_degrees=120))

    with pytest.raises(ValueError, match="Text Query query_sample_fps must be a number"):
        pages._text_effective_settings(base, *_values_with(base, query_sample_fps=True))
    with pytest.raises(ValueError, match="Text Query query_sample_fps must be a number"):
        pages._text_effective_settings(base, *_values_with(base, query_sample_fps="abc"))
    with pytest.raises(ValueError, match="Text Query grounding_threshold must be a finite number"):
        pages._text_effective_settings(base, *_values_with(base, grounding_threshold=float("nan")))
    with pytest.raises(ValueError, match="Text Query query_max_results must be a whole number"):
        pages._text_effective_settings(base, *_values_with(base, query_max_results=2.5))

    with pytest.raises(ValueError, match="Text Query scene_cache_version must be a non-empty string"):
        pages._text_effective_settings(base, *_values_with(base, scene_cache_version="   "))
    with pytest.raises(ValueError, match="Text Query scene_cache_version must be a non-empty string"):
        pages._text_effective_settings(base, *_values_with(base, scene_cache_version=5))
    with pytest.raises(ValueError, match="Text Query query_max_results must not be empty"):
        pages._text_effective_settings(base, *_values_with(base, query_max_results=None))
    with pytest.raises(ValueError, match="10 values"):
        pages._text_effective_settings(base)


def test_scene_cache_key_tracks_source_fov_and_cache_version() -> None:
    """Changing either the FOV or the cache epoch yields a fresh scene key."""
    base = load_text_settings()
    baseline = scene_cache.scene_cache_key("a" * 64, 0, 8, 12, base)
    assert scene_cache.scene_cache_key("a" * 64, 0, 8, 12, replace(base, source_fov_degrees=75)) != baseline
    assert scene_cache.scene_cache_key("a" * 64, 0, 8, 12, replace(base, scene_cache_version="next-epoch")) != baseline

    # The value the UI parser builds is the value that keys the cache.
    tuned = pages._text_effective_settings(
        base, *_values_with(base, source_fov_degrees=95, scene_cache_version="epoch-2"),
    )
    assert scene_cache.scene_cache_key("a" * 64, 0, 8, 12, tuned) != baseline


def _patch_query_pipeline(monkeypatch, frames, captured: dict, masks: np.ndarray | None = None) -> None:
    """Replace Text Query workers with deterministic spies capturing their arguments."""
    masks = masks if masks is not None else np.ones((len(frames), 16, 20), dtype=bool)
    monkeypatch.setattr(selection, "read_uniform_samples", lambda _source, sample_fps, maximum_seconds: (
        captured.update(sample_fps=sample_fps, max_source_seconds=maximum_seconds) or
        ([frames[0]], [0.0], VideoInfo(12, 120, 10, 20, 16))
    ))

    def fake_scores(_frames, _query, **_kwargs):
        if False:
            yield
        return np.asarray([0.9], dtype=np.float32)

    monkeypatch.setattr(selection, "semantic_scores_steps", fake_scores)
    monkeypatch.setattr(selection, "ranked_windows", lambda _scores, _times, _source, clip_seconds, limit: (
        captured.update(clip_seconds=clip_seconds, candidate_pool=limit) or
        [CandidateWindow(0, 0, min(clip_seconds, 4.0), 0.9)]
    ))
    monkeypatch.setattr(selection, "read_interval", lambda _source, _start, _duration, result_fps: (
        captured.update(result_fps=result_fps) or (frames, 12.0)
    ))
    monkeypatch.setattr(selection, "source_digest", lambda _source: "source-hash")
    monkeypatch.setattr(selection, "write_video", lambda _frames, path, _fps: (path.touch(), path)[1])

    def fake_ground_steps(_frames, _query, anchor_index=None, threshold=0.25, **_kwargs):
        captured["grounding_threshold"] = threshold
        if False:
            yield
        return MaskTrack(masks, GroundedBox((5, 4, 15, 12), 0.8), 2)

    monkeypatch.setattr(selection, "ground_and_track_steps", fake_ground_steps)


def test_effective_values_reach_retrieval_and_grounding(monkeypatch, tmp_path) -> None:
    """Retrieval, decode, candidate count, and the grounding threshold use overrides."""
    base = load_text_settings()
    effective = pages._text_effective_settings(
        base,
        *_values_with(
            base,
            query_sample_fps=5, query_max_source_seconds=120, query_clip_seconds=6,
            query_result_fps=15, query_max_results=3, query_explicit_candidate_pool=4,
            grounding_threshold=0.55,
        ),
    )
    captured: dict = {}
    frames = _tiny_frames()
    _patch_query_pipeline(monkeypatch, frames, captured)

    list(selection.run("unused.mp4", selection.SelectionRequest("implicit", "tree"), RunArtifacts(tmp_path / "prop"), effective))

    assert captured["sample_fps"] == effective.query_sample_fps
    assert captured["max_source_seconds"] == effective.query_max_source_seconds
    assert captured["clip_seconds"] == effective.query_clip_seconds
    assert captured["candidate_pool"] == effective.query_explicit_candidate_pool
    assert captured["result_fps"] == effective.query_result_fps
    assert captured["grounding_threshold"] == effective.grounding_threshold


def test_ground_and_track_forwards_threshold_to_the_detector(monkeypatch) -> None:
    """A tuned grounding threshold is passed through to Grounding DINO unchanged."""
    seen: dict = {}
    monkeypatch.setattr(text_segmentation, "ground_query", lambda _frame, _query, threshold=0.25: (
        seen.update(threshold=threshold) or None
    ))
    assert text_segmentation.ground_and_track(_tiny_frames(), "tree", threshold=0.6) is None
    assert seen["threshold"] == 0.6


def test_effective_scene_settings_reach_the_scene_cache(monkeypatch, tmp_path) -> None:
    """Explicit mode hands the effective FOV/epoch to the reusable scene build."""
    base = load_text_settings()
    effective = pages._text_effective_settings(
        base, *_values_with(base, source_fov_degrees=88, scene_cache_version="epoch-99"),
    )
    frames = _tiny_frames()
    masks = np.ones((len(frames), 16, 20), dtype=bool)
    captured: dict = {}
    _patch_query_pipeline(monkeypatch, frames, captured, masks)

    def fake_scene(_frames, _fps, _digest, _start, settings):
        captured["scene_settings"] = settings
        if False:
            yield StageEvent("unused", "running", "unused")
        return SimpleNamespace(key="cached-scene")

    monkeypatch.setattr(selection, "_consume_scene", fake_scene)
    monkeypatch.setattr(selection, "lift_masks", lambda _scene, _masks: PrimitiveMask(np.asarray([True]), 0.5))
    monkeypatch.setattr(selection, "project_primitive_mask", lambda _scene, _primitive: masks)

    list(selection.run("unused.mp4", selection.SelectionRequest("explicit", "tree"), RunArtifacts(tmp_path / "scene"), effective))

    assert captured["scene_settings"].source_fov_degrees == 88
    assert captured["scene_settings"].scene_cache_version == "epoch-99"


def test_effective_coverage_threshold_changes_the_low_specificity_flag(monkeypatch, tmp_path) -> None:
    """The projected-mask coverage flag is decided by the run's own threshold."""
    base = load_text_settings()
    frames = _tiny_frames()
    # Half the frame is selected, so projected coverage is exactly 0.5.
    masks = np.zeros((len(frames), 16, 20), dtype=bool)
    masks[:, :8, :] = True

    def run_with(coverage_threshold: float) -> bool:
        effective = pages._text_effective_settings(
            base, *_values_with(base, low_specificity_coverage=coverage_threshold),
        )
        captured: dict = {}
        _patch_query_pipeline(monkeypatch, frames, captured, masks)
        monkeypatch.setattr(selection, "_consume_scene", lambda *_args: _scene_generator(SimpleNamespace(key="cached-scene")))
        monkeypatch.setattr(selection, "lift_masks", lambda _scene, _masks: PrimitiveMask(np.asarray([True]), 0.5))
        monkeypatch.setattr(selection, "project_primitive_mask", lambda _scene, _primitive: masks)
        result = list(selection.run("unused.mp4", selection.SelectionRequest("explicit", "tree"), RunArtifacts(tmp_path / f"cov-{coverage_threshold}"), effective))[-1]
        return result.clips[0].projected_low_specificity

    assert run_with(0.4) is True
    assert run_with(0.9) is False


def _scene_generator(scene):
    """Return an exhausted generator that yields ``scene`` as its StopIteration value."""
    if False:
        yield
    return scene


def test_selection_config_visibility_gates_the_explicit_scene_group() -> None:
    """Text Query hides the Explicit scene group only for Implicit 3D runs.

    Group order: Source, Results, Matching, Explicit scene. Only the last group is
    methodology-gated; the rest stay visible so shared controls never disappear.
    """
    explicit = pages._selection_config_visibility("explicit")
    implicit = pages._selection_config_visibility("implicit")
    assert [update["visible"] for update in explicit] == [True, True, True, True]
    assert [update["visible"] for update in implicit] == [True, True, True, False]


def test_selection_config_visibility_never_rewrites_control_values() -> None:
    """A visibility update must carry no ``value``, or a mode switch would reset knobs."""
    for mode in ("explicit", "implicit"):
        updates = pages._selection_config_visibility(mode)
        assert len(updates) == 4
        assert all("value" not in update for update in updates), mode


def test_selection_page_starts_with_the_explicit_scene_group_visible() -> None:
    """The built page's default Explicit 3D mode shows every configuration group."""
    page = pages.build_selection()
    registration = next(
        fn for fn in page.fns.values()
        if getattr(fn.fn, "__name__", "") == "_selection_config_visibility"
    )
    assert [component.visible for component in registration.outputs] == [True, True, True, True]
