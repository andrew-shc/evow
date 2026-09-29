"""Unit coverage for reusable Gaussian scenes and text-mode dispatch."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import json
import numpy as np
import pytest
import torch

from GREENFIELD.app_core.contracts import RunArtifacts, StageEvent
from GREENFIELD.features import editing, scene_cache, scene_masks, selection
from GREENFIELD.features.scene_cache import CachedScene
from GREENFIELD.features.scene_masks import PrimitiveMask
from GREENFIELD.features.text_media import VideoInfo
from GREENFIELD.features.text_models import GroundedBox
from GREENFIELD.features.text_retrieval import CandidateWindow
from GREENFIELD.features.text_segmentation import MaskTrack
from GREENFIELD.features.text_settings import TextSettings
from GREENFIELD.replay.overrides import settings_json
from GREENFIELD.replay.settings import load_settings as load_replay_settings


def _settings(root: Path) -> TextSettings:
    """Create test-only text settings without relying on local model weights."""
    checkpoints = root / "ASSETS" / "checkpoints"
    return TextSettings(
        root=root,
        semantic_model_id="siglip", grounding_model_id="dino", segmentation_model_id="sam2", edit_model_id="vace",
        semantic_checkpoint=checkpoints / "siglip", grounding_checkpoint=checkpoints / "dino",
        segmentation_checkpoint=checkpoints / "sam2", edit_checkpoint=checkpoints / "vace",
        query_max_source_seconds=300, query_sample_fps=2, query_clip_seconds=4, query_result_fps=12,
        query_max_results=5, query_explicit_candidate_pool=8, grounding_threshold=0.25,
        low_specificity_coverage=0.65, episode_fps=12, episode_max_frames=81,
        source_fov_degrees=70, scene_cache_version="dynamic-gaussian-v1", edit_max_side=64, edit_max_height=64,
        vace_steps=1, vace_guidance_scale=5.0, edit_mask_expand_px=5,
    )


def _frames(count: int = 5) -> list[np.ndarray]:
    """Make a tiny stationary RGB episode for deterministic adapter tests."""
    return [np.zeros((16, 20, 3), dtype=np.uint8) for _ in range(count)]


def _scene(root: Path, name: str, frame_count: int = 5) -> CachedScene:
    """Build tiny file-backed scene metadata for orchestration tests only."""
    scene_dir = root / name
    scene_dir.mkdir(parents=True)
    checkpoint = scene_dir / "scene.pt"
    rendered = scene_dir / "rendered.mp4"
    checkpoint.touch()
    rendered.write_bytes(b"rendered")
    splats = []
    for index in range(frame_count):
        path = scene_dir / f"splat_{index:03d}.splat"
        path.write_bytes(b"splat")
        splats.append(path)
    return CachedScene(name, scene_dir.parent, scene_dir, rendered, checkpoint, tuple(splats), 20, 16, 12, 70)


def _patch_ground(monkeypatch, track) -> None:
    """Patch editing's grounding relay with a deterministic DINO/SAM2 tick stream."""
    def fake_steps(_frames, _prompt, anchor_index=None, threshold=0.25, *, candidate_index=1, total_candidates=1, stage_ids):
        if track is None:
            yield StageEvent("Ground target with Grounding DINO", "running", "No match.", metrics={"matched": False}, stage_id=stage_ids["dino_detect"])
            return None
        yield StageEvent("Ground target with Grounding DINO", "running", "Matched.", metrics={"confidence": track.box.confidence}, stage_id=stage_ids["dino_detect"])
        yield StageEvent("Propagate SAM2 masks", "running", "Forward.", metrics={"reversed": False, "propagated_frames": len(track.masks), "total_frames": len(track.masks)}, stage_id=stage_ids["sam2_propagate"])
        return track
    monkeypatch.setattr(editing, "ground_and_track_steps", fake_steps)


def _patch_vace(monkeypatch, captured: dict | None = None) -> None:
    """Patch editing's Wan VACE relay with a deterministic streaming stub."""
    def fake_steps(frames, masks, _prompt, _seed, settings=None, *, stage_ids=None):
        ids = stage_ids or {}
        if captured is not None:
            captured["vace"] = captured.get("vace", 0) + 1
            captured["mask"] = masks
            captured["settings"] = settings
        yield StageEvent("Prepare VACE conditioning", "running", "prep", metrics={}, stage_id=ids.get("vace_prepare", ""))
        yield StageEvent("Denoise edit proposal", "running", "denoise", metrics={"frames": len(frames)}, stage_id=ids.get("vace_denoise", ""))
        yield StageEvent("Decode edit proposal", "running", "decode", metrics={"frames": len(frames)}, stage_id=ids.get("vace_decode", ""))
        return list(frames)
    monkeypatch.setattr(editing, "edit_video_steps", fake_steps)


def test_scene_cache_key_and_complete_hit_do_not_require_cuda(tmp_path: Path) -> None:
    """Cache identity tracks source/configuration and complete entries reuse offline."""
    settings = _settings(tmp_path)
    first = scene_cache.scene_cache_key("a" * 64, 0, 2, 12, settings)
    assert first != scene_cache.scene_cache_key("a" * 64, 4, 2, 12, settings)
    assert first != scene_cache.scene_cache_key("b" * 64, 0, 2, 12, settings)
    assert first != scene_cache.scene_cache_key("a" * 64, 0, 2, 12, replace(settings, source_fov_degrees=60))
    assert first != scene_cache.scene_cache_key("a" * 64, 0, 2, 12, replace(settings, scene_cache_version="dynamic-gaussian-v2"))

    target = settings.root / "ASSETS" / "scenes" / first
    _scene(target, "scene", frame_count=2)
    (target / "manifest.json").write_text(json.dumps({
        "key": first, "frame_count": 2, "fps": 12, "width": 20, "height": 16, "source_fov_degrees": 70,
    }))
    steps = scene_cache.ensure_scene_steps(_frames(2), 12, "a" * 64, 0, settings, "Resolve")
    # A cache hit is tick-free and caller-terminated: the generator returns the
    # cached scene without emitting any running/complete events.
    try:
        next(steps)
    except StopIteration as complete:
        assert complete.value.key == first
        assert complete.value.cache_hit is True
    else:
        raise AssertionError("cache-hit generator must not emit a tick")


def test_scene_cache_build_sends_and_keys_the_same_reconstruction_settings(monkeypatch, tmp_path: Path) -> None:
    """The worker payload and the cache key must come from one effective settings value."""
    settings = _settings(tmp_path)
    captured: list[list[str]] = []

    class _StopAfterGaussian(Exception):
        """End the build right after the Gaussian command is captured."""

    def fake_worker(command, *_args, **_kwargs):
        captured.append(command)
        if any("depth_worker" in part for part in command):
            # Satisfy the post-depth existence check so the Gaussian call runs.
            depths = Path(command[command.index("--out") + 1])
            depths.parent.mkdir(parents=True, exist_ok=True)
            depths.write_bytes(b"depths")
            return iter(())
        raise _StopAfterGaussian()

    monkeypatch.setattr(scene_cache, "require_checkpoint", lambda *_args: None)
    monkeypatch.setattr(scene_cache, "require_package", lambda *_args: None)
    monkeypatch.setattr(scene_cache, "_worker", fake_worker)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    steps = scene_cache.ensure_scene_steps(_frames(2), 12, "a" * 64, 0, settings, "Resolve")
    with pytest.raises(_StopAfterGaussian):
        list(steps)

    assert len(captured) == 2
    assert all("--settings-json" in command for command in captured)
    expected = json.loads(settings_json(load_replay_settings()))
    assert all(json.loads(command[command.index("--settings-json") + 1]) == expected for command in captured)

    # Recover the key from the temporary build directory named ".<key>_<uuid>.building".
    frames_path = Path(captured[0][captured[0].index("--frames") + 1])
    built_key = frames_path.parent.name.removeprefix(".").rsplit("_", 1)[0]
    assert built_key == scene_cache.scene_cache_key(
        "a" * 64, 0, 2, 12, settings, load_replay_settings(),
    )

    # A changed reconstruction knob must change the key that names the cache.
    tracked = load_replay_settings()
    tuned = replace(tracked, gaussian_steps=tracked.gaussian_steps + 1)
    assert scene_cache.scene_cache_key("a" * 64, 0, 2, 12, settings, tracked) != scene_cache.scene_cache_key(
        "a" * 64, 0, 2, 12, settings, tuned,
    )


def test_lift_and_project_primitive_masks(monkeypatch, tmp_path: Path) -> None:
    """Persistent primitive voting produces one source-view binary mask per frame."""
    scene = _scene(tmp_path, "primitive", frame_count=2)
    means = np.asarray([[0.0, 0.0, 2.0], [0.8, 0.0, 2.0]], dtype=np.float32)
    motion = np.zeros((2, 2, 3), dtype=np.float32)
    scales = np.full((2, 3), 0.1, dtype=np.float32)
    monkeypatch.setattr(scene_masks, "_scene_arrays", lambda _scene: (means, motion, scales))
    masks = np.zeros((2, 16, 20), dtype=bool)
    masks[:, 8, 10] = True

    primitives = scene_masks.lift_masks(scene, masks)
    assert primitives.selected.tolist() == [True, False]
    assert primitives.support == 0.5
    projected = scene_masks.project_primitive_mask(scene, primitives)
    assert projected.shape == (2, 16, 20)
    assert projected[:, 8, 10].all()


@pytest.mark.parametrize("mode", ["implicit", "explicit"])
def test_query_dispatches_implicit_masks_or_explicit_primitive_masks(monkeypatch, tmp_path: Path, mode: str) -> None:
    """Both query modes share retrieval but use their intended mask representation."""
    frames = _frames()
    masks = np.zeros((len(frames), 16, 20), dtype=bool)
    masks[:, 4:12, 5:15] = True
    settings = SimpleNamespace(
        query_sample_fps=2, query_max_source_seconds=300, query_clip_seconds=4,
        query_explicit_candidate_pool=8, query_result_fps=12, query_max_results=5,
        grounding_threshold=0.25, low_specificity_coverage=0.65,
    )
    calls = {"scene": 0, "lift": 0, "project": 0}
    monkeypatch.setattr(selection, "load_text_settings", lambda: settings)
    monkeypatch.setattr(selection, "read_uniform_samples", lambda *_args: (
        [frames[0]], [2.0], VideoInfo(12, 120, 10, 20, 16),
    ))

    def fake_scores(_frames, _query, **_kwargs):
        if False:
            yield
        return np.asarray([0.9], dtype=np.float32)

    def fake_ground_steps(*_args, **_kwargs):
        if False:
            yield
        return MaskTrack(masks, GroundedBox((5, 4, 15, 12), 0.8), 2)

    monkeypatch.setattr(selection, "semantic_scores_steps", fake_scores)
    monkeypatch.setattr(selection, "ranked_windows", lambda *_args: [CandidateWindow(0, 0, 4, 0.9)])
    monkeypatch.setattr(selection, "read_interval", lambda *_args: (frames, 12.0))
    monkeypatch.setattr(selection, "ground_and_track_steps", fake_ground_steps)
    monkeypatch.setattr(selection, "source_digest", lambda _path: "source-hash")
    monkeypatch.setattr(selection, "write_video", lambda _frames, path, _fps: (path.touch(), path)[1])

    if mode == "explicit":
        marker = SimpleNamespace(key="cached-scene")

        def fake_scene(*_args):
            calls["scene"] += 1
            if False:
                yield StageEvent("unused", "running", "unused")
            return marker

        def fake_lift(_scene, _masks):
            calls["lift"] += 1
            return PrimitiveMask(np.asarray([True]), 0.5)

        def fake_project(_scene, _primitive):
            calls["project"] += 1
            return masks

        monkeypatch.setattr(selection, "_consume_scene", fake_scene)
        monkeypatch.setattr(selection, "lift_masks", fake_lift)
        monkeypatch.setattr(selection, "project_primitive_mask", fake_project)
    else:
        monkeypatch.setattr(selection, "_consume_scene", lambda *_args: (_ for _ in ()).throw(AssertionError("implicit query must not build a scene")))

    result = list(selection.run("unused.mp4", selection.SelectionRequest(mode, "tree"), RunArtifacts(tmp_path / mode)))[-1]
    assert result.clips[0].method == mode
    expected = {"scene": 0, "lift": 0, "project": 0} if mode == "implicit" else {"scene": 1, "lift": 1, "project": 1}
    assert calls == expected


@pytest.mark.parametrize("mode", ["implicit", "explicit"])
def test_edit_no_match_fallback_and_explicit_splats(monkeypatch, tmp_path: Path, mode: str) -> None:
    """No grounded object yields a full-scene mask; only explicit mode rebuilds splats."""
    frames = _frames()
    monkeypatch.setattr(editing, "load_text_settings", lambda: SimpleNamespace(
        episode_fps=12, episode_max_frames=81, grounding_threshold=0.25, edit_mask_expand_px=5,
    ))
    monkeypatch.setattr(editing, "read_editing_episode", lambda *_args: (frames, 12.0))
    _patch_ground(monkeypatch, None)
    monkeypatch.setattr(editing, "source_digest", lambda _path: "source-hash")
    _patch_vace(monkeypatch)
    monkeypatch.setattr(editing, "write_video", lambda _frames, path, _fps: (path.write_bytes(b"video"), path)[1])

    calls = {"scene": 0}
    if mode == "explicit":
        base = _scene(tmp_path, "base")
        edited = _scene(tmp_path, "edited")
        returned = iter((base, edited))

        def fake_scene(*_args):
            calls["scene"] += 1
            if False:
                yield StageEvent("unused", "running", "unused")
            return next(returned)

        monkeypatch.setattr(editing, "_consume_scene", fake_scene)
        monkeypatch.setattr(editing, "lift_masks", lambda *_args: PrimitiveMask(np.asarray([True]), 1.0))
        monkeypatch.setattr(editing, "project_primitive_mask", lambda *_args: np.ones((5, 16, 20), dtype=bool))
    else:
        monkeypatch.setattr(editing, "_consume_scene", lambda *_args: (_ for _ in ()).throw(AssertionError("implicit edit must not build a scene")))

    result = list(editing.run("unused.mp4", editing.EditingRequest(mode, "make the scene brighter", 3), RunArtifacts(tmp_path / f"edit-{mode}")))[-1]
    assert result.metadata["scope"] == "full_scene_fallback"
    assert result.metadata["generated"] is True
    if mode == "explicit":
        assert calls["scene"] == 2
        assert len(result.metadata["viewer"]["splat_paths"]) == 5
        assert all(path.startswith("explicit_splats/") for path in result.metadata["viewer"]["splat_paths"])
    else:
        assert calls["scene"] == 0
        assert "viewer" not in result.metadata


def test_explicit_query_reranks_every_candidate_before_selecting_results(monkeypatch, tmp_path: Path) -> None:
    """3D primitive support can promote a later candidate into the final gallery."""
    frames = _frames()
    masks = np.ones((len(frames), 16, 20), dtype=bool)
    settings = SimpleNamespace(
        query_sample_fps=2, query_max_source_seconds=300, query_clip_seconds=4,
        query_explicit_candidate_pool=8, query_result_fps=12, query_max_results=1,
        grounding_threshold=0.25, low_specificity_coverage=0.65,
    )
    calls = {"scene": 0}
    monkeypatch.setattr(selection, "load_text_settings", lambda: settings)
    monkeypatch.setattr(selection, "read_uniform_samples", lambda *_args: (
        [frames[0]], [0.0], VideoInfo(12, 120, 10, 20, 16),
    ))

    def fake_scores(_frames, _query, **_kwargs):
        if False:
            yield
        return np.asarray([0.5], dtype=np.float32)

    def fake_ground_steps(*_args, **_kwargs):
        if False:
            yield
        return MaskTrack(masks, GroundedBox((0, 0, 20, 16), 0.9), 2)

    monkeypatch.setattr(selection, "semantic_scores_steps", fake_scores)
    monkeypatch.setattr(selection, "ranked_windows", lambda *_args: [
        CandidateWindow(0, 0, 4, 0.50), CandidateWindow(1, 4, 8, 0.49),
    ])
    monkeypatch.setattr(selection, "read_interval", lambda *_args: (frames, 12.0))
    monkeypatch.setattr(selection, "ground_and_track_steps", fake_ground_steps)
    monkeypatch.setattr(selection, "source_digest", lambda _path: "source-hash")
    monkeypatch.setattr(selection, "write_video", lambda _frames, path, _fps: (path.touch(), path)[1])

    def fake_scene(*_args):
        calls["scene"] += 1
        if False:
            yield StageEvent("unused", "running", "unused")
        return SimpleNamespace(key=f"scene-{calls['scene']}")

    def fake_lift(_scene, _masks):
        support = 0.0 if calls["scene"] == 1 else 1.0
        return PrimitiveMask(np.asarray([True]), support)

    monkeypatch.setattr(selection, "_consume_scene", fake_scene)
    monkeypatch.setattr(selection, "lift_masks", fake_lift)
    monkeypatch.setattr(selection, "project_primitive_mask", lambda *_args: masks)
    result = list(selection.run("unused.mp4", selection.SelectionRequest("explicit", "tree"), RunArtifacts(tmp_path / "rerank")))[-1]

    assert calls["scene"] == 2
    assert len(result.clips) == 1
    assert result.clips[0].start_seconds == 4


def test_environmental_edit_overrides_a_partial_grounded_target(monkeypatch, tmp_path: Path) -> None:
    """Flooding a forest is scene-wide even if a detector finds a partial target."""
    frames = _frames()
    partial = np.zeros((len(frames), 16, 20), dtype=bool)
    partial[:, 4:12, 5:15] = True
    captured: dict[str, np.ndarray] = {}
    monkeypatch.setattr(editing, "load_text_settings", lambda: SimpleNamespace(
        episode_fps=12, episode_max_frames=81, grounding_threshold=0.25, edit_mask_expand_px=5,
    ))
    monkeypatch.setattr(editing, "read_editing_episode", lambda *_args: (frames, 12.0))
    _patch_ground(monkeypatch, MaskTrack(partial, GroundedBox((5, 4, 15, 12), 0.9), 2))
    monkeypatch.setattr(editing, "source_digest", lambda _path: "source-hash")
    _patch_vace(monkeypatch, captured)
    monkeypatch.setattr(editing, "write_video", lambda _frames, path, _fps: (path.write_bytes(b"video"), path)[1])
    result = list(editing.run("unused.mp4", editing.EditingRequest("implicit", "flood the forest", 0), RunArtifacts(tmp_path / "environmental")))[-1]

    assert result.metadata["scope"] == "full_scene_environmental"
    assert captured["mask"].all()


def test_edit_run_generates_in_a_single_pass(monkeypatch, tmp_path: Path) -> None:
    """Text Manipulation resolves the scope and generates without a separate approval gate."""
    frames = _frames()
    masks = np.zeros((len(frames), 16, 20), dtype=bool)
    masks[:, :8, :] = True
    calls = {"vace": 0}
    monkeypatch.setattr(editing, "load_text_settings", lambda: SimpleNamespace(
        episode_fps=12, episode_max_frames=81, grounding_threshold=0.25, edit_mask_expand_px=5,
    ))
    monkeypatch.setattr(editing, "read_editing_episode", lambda *_args: (frames, 12.0))
    _patch_ground(monkeypatch, MaskTrack(masks, GroundedBox((0, 0, 20, 8), 0.9), 2))
    monkeypatch.setattr(editing, "source_digest", lambda _path: "source-hash")
    monkeypatch.setattr(editing, "write_video", lambda _frames, path, _fps: (path.write_bytes(b"video"), path)[1])
    _patch_vace(monkeypatch, calls)
    artifacts = RunArtifacts(tmp_path / "single-pass")
    result = list(editing.run("unused.mp4", editing.EditingRequest("implicit", "make the sky red", 0), artifacts))[-1]
    assert calls["vace"] == 1
    assert (artifacts.run_dir / "edit_mask.npy").is_file()
    assert not (artifacts.run_dir / "scope_preview.mp4").exists()
    assert "scope_preview" not in result.metadata
    assert result.metadata["generated"] is True


def test_edit_run_flags_an_implausibly_broad_named_target(monkeypatch, tmp_path: Path) -> None:
    """A sky-like target mask covering nearly everything is recorded as a broad named target."""
    frames = _frames()
    masks = np.ones((len(frames), 16, 20), dtype=bool)
    monkeypatch.setattr(editing, "load_text_settings", lambda: SimpleNamespace(
        episode_fps=12, episode_max_frames=81, grounding_threshold=0.25, edit_mask_expand_px=5,
    ))
    monkeypatch.setattr(editing, "read_editing_episode", lambda *_args: (frames, 12.0))
    _patch_ground(monkeypatch, MaskTrack(masks, GroundedBox((0, 0, 20, 16), 0.9), 2))
    monkeypatch.setattr(editing, "source_digest", lambda _path: "source-hash")
    monkeypatch.setattr(editing, "write_video", lambda _frames, path, _fps: (path.write_bytes(b"video"), path)[1])
    _patch_vace(monkeypatch)

    result = list(editing.run("unused.mp4", editing.EditingRequest("implicit", "make the sky red", 0), RunArtifacts(tmp_path / "broad")))[-1]
    assert result.metadata["scope"] == "grounded_target"
    assert result.metadata["mask_coverage"] == 1.0
    assert result.metadata["broad_target_warning"] is True
