"""Text Manipulation's atomic 19-row union flow, per-mode applicability, releases.

These tests pin the declaration (ids, groups, modes, frozen v1 labels, per-version
recorded sets), the render-derived per-mode skips, the aggregate-row lifecycle
(real throttled ``vace_denoise`` ticks, exactly one terminal per row), the
truthful zero-candidate paths, the scene-cache single-terminal regression, the
mode-scoped mask expansion order (implicit expands the 2D mask; explicit lifts
the unexpanded mask then expands the projected one), and the feature-scoped
legacy mapping.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from GREENFIELD.app_core.contracts import FeatureResult, RunArtifacts, StageEvent
from GREENFIELD.app_core.flow import CURRENT_FLOW_VERSION, LEGACY_FLOW_VERSION, stage_applies
from GREENFIELD.features import editing, pages, video_edit
from GREENFIELD.features.scene_masks import PrimitiveMask
from GREENFIELD.features.text_models import GroundedBox
from GREENFIELD.features.text_segmentation import MaskTrack, expand_masks


EDITING_STAGE_IDS = (
    "episode_decode", "vace_trim", "dino_detect", "sam2_propagate", "scope_classify",
    "mask_select", "mask_expand", "prepare_mask", "scene_cache", "mask_project",
    "mask_expand_projected", "vace_prepare", "vace_denoise", "vace_decode",
    "write_proposal", "finalize", "edited_scene", "splat_copy", "publish_artifacts",
)
_IMPLICIT_ONLY = ("mask_expand", "prepare_mask", "finalize")
_EXPLICIT_ONLY = ("scene_cache", "mask_project", "mask_expand_projected", "edited_scene", "splat_copy")
_GROUPS = 5  # editing's five parent accordions precede the stage headers
_OLD_STAGE_IDS = ("inspect_episode", "resolve_scope", "prepare_method", "generate_edit", "save_result")


def _frames(count: int = 5) -> list[np.ndarray]:
    return [np.zeros((16, 20, 3), dtype=np.uint8) for _ in range(count)]


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        episode_fps=12, episode_max_frames=81, grounding_threshold=0.25,
        edit_mask_expand_px=5, vace_steps=4, vace_guidance_scale=5.0,
        edit_max_side=64, edit_max_height=64, source_fov_degrees=70,
    )


def _patch_ground(monkeypatch, track) -> None:
    """Patch editing's grounding relay with a deterministic DINO/SAM2 tick stream."""
    def fake_steps(_frames, _prompt, anchor_index=None, threshold=0.25, *, candidate_index=1, total_candidates=1, stage_ids):
        if track is None:
            yield StageEvent("Ground target with Grounding DINO", "running", "No match.", metrics={"matched": False}, stage_id=stage_ids["dino_detect"])
            return None
        yield StageEvent("Ground target with Grounding DINO", "running", "Matched.", metrics={"confidence": float(track.box.confidence)}, stage_id=stage_ids["dino_detect"])
        yield StageEvent("Propagate SAM2 masks", "running", "Forward.", metrics={"reversed": False, "propagated_frames": len(track.masks), "total_frames": len(track.masks)}, stage_id=stage_ids["sam2_propagate"])
        return track
    monkeypatch.setattr(editing, "ground_and_track_steps", fake_steps)


class _FakeVacePipeline:
    """A minimal Wan VACE double that streams real denoise callbacks when asked."""

    def __init__(self, total_steps: int, exposes_total: bool = True):
        self.scheduler = SimpleNamespace(timesteps=list(range(total_steps)) if exposes_total else [])

    def __call__(self, callback_on_step_end=None, callback_on_step_end_tensor_inputs=None, video=None, **kwargs):
        if callback_on_step_end is not None:
            for step in range(len(self.scheduler.timesteps)):
                callback_on_step_end(self, step, float(step), {"latents": SimpleNamespace(shape=(1, 16, 4, 8, 8))})
        return SimpleNamespace(frames=[video])


def _patch_vace(monkeypatch, total_steps: int = 4, exposes_total: bool = True) -> _FakeVacePipeline:
    pipeline = _FakeVacePipeline(total_steps, exposes_total)
    monkeypatch.setattr(video_edit, "vace_pipeline", lambda: pipeline)
    monkeypatch.setattr(video_edit, "device", lambda: "cpu")
    return pipeline


class _FakeScene:
    """A minimal ``CachedScene`` double with the file-backed fields the run reads."""

    def __init__(self, root: Path, name: str, frame_count: int, cache_hit: bool):
        scene_dir = root / name
        scene_dir.mkdir(parents=True, exist_ok=True)
        self.key = name
        self.rendered = scene_dir / "rendered.mp4"
        self.rendered.write_bytes(b"rendered")
        self.splats = tuple(scene_dir / f"splat_{index:03d}.splat" for index in range(frame_count))
        for splat in self.splats:
            splat.write_bytes(b"splat")
        self.cache_hit = cache_hit


def _patch_scene(monkeypatch, root: Path, miss: bool = True):
    """Patch editing's scene relay to yield a tick (miss) or return silently (hit)."""
    def fake_scene(frames, _fps, _digest, stage_id, _settings):
        if not miss:
            return _FakeScene(root, f"{stage_id}-hit", len(frames), cache_hit=True)
        yield StageEvent("Check scene cache", "running", "building", metrics={"cache": "miss", "scene_key": stage_id}, stage_id=stage_id)
        return _FakeScene(root, f"{stage_id}-miss", len(frames), cache_hit=False)
    monkeypatch.setattr(editing, "_consume_scene", fake_scene)


def _run(mode: str, monkeypatch, tmp_path: Path, track, *, miss: bool = True, total_steps: int = 4, exposes_total: bool = True):
    frames = _frames()
    _patch_ground(monkeypatch, track)
    _patch_scene(monkeypatch, tmp_path, miss=miss)
    _patch_vace(monkeypatch, total_steps=total_steps, exposes_total=exposes_total)
    monkeypatch.setattr(editing, "read_editing_episode", lambda *_args: (frames, 12.0))
    monkeypatch.setattr(editing, "source_digest", lambda _path: "source-hash")
    monkeypatch.setattr(editing, "write_video", lambda _frames, path, _fps: (path.write_bytes(b"video"), path)[1])
    monkeypatch.setattr(editing, "lift_masks", lambda _scene, _masks: PrimitiveMask(np.asarray([True]), 0.5))
    monkeypatch.setattr(editing, "project_primitive_mask", lambda _scene, _primitive: np.ones((len(frames), 16, 20), dtype=bool))
    return [
        item for item in editing.run(
            "unused.mp4", editing.EditingRequest(mode, "make the sky red", 3), RunArtifacts(tmp_path / mode), _settings(),
        )
        if isinstance(item, StageEvent)
    ]


def _headers(trace: dict) -> list[dict]:
    return list(pages._updates("editing", trace)[_GROUPS:_GROUPS + len(EDITING_STAGE_IDS)])


def _header(trace: dict, stage_id: str) -> dict:
    return _headers(trace)[EDITING_STAGE_IDS.index(stage_id)]


def _terminal_counts(events: list[StageEvent]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        if event.status in {"complete", "error", "skipped"}:
            counts[event.stage_id or event.stage] = counts.get(event.stage_id or event.stage, 0) + 1
    return counts


def _running_after_terminal(events: list[StageEvent]) -> list[str]:
    terminated: set[str] = set()
    violations: list[str] = []
    for event in events:
        stage_id = event.stage_id or event.stage
        if event.status == "running":
            if stage_id in terminated:
                violations.append(stage_id)
        elif event.status in {"complete", "error", "skipped"}:
            terminated.add(stage_id)
    return violations


def _first_seen(events: list[StageEvent]) -> list[str]:
    seen: list[str] = []
    for event in events:
        if event.stage_id not in seen:
            seen.append(event.stage_id)
    return seen


def test_editing_flow_declares_ordered_ids_groups_and_frozen_v1_labels() -> None:
    flow = editing.flow()

    assert tuple(stage.stage_id for stage in flow.stages) == EDITING_STAGE_IDS
    assert [
        (group.label, group.stage_ids) for group in flow.groups
    ] == [
        ("Input", ("episode_decode", "vace_trim")),
        ("Resolve scope", ("dino_detect", "sam2_propagate", "scope_classify", "mask_select", "mask_expand")),
        ("Prepare method", ("prepare_mask", "scene_cache", "mask_project", "mask_expand_projected")),
        ("Generate", ("vace_prepare", "vace_denoise", "vace_decode", "write_proposal")),
        ("Save", ("finalize", "edited_scene", "splat_copy", "publish_artifacts")),
    ]
    grouped = [stage_id for group in flow.groups for stage_id in group.stage_ids]
    assert sorted(grouped) == sorted(EDITING_STAGE_IDS)
    assert flow.legacy_stage_ids == (
        "Inspect source episode", "Resolve edit scope", "Prepare selected method",
        "Generate edit proposal", "Save and render result",
    )
    # Every pre-v5 profile wrote the same five coarse bars; v4 is explicitly
    # retained so a saved v4 trace is never mislabeled "Not recorded".
    for version in (LEGACY_FLOW_VERSION, 2, 3, 4):
        assert flow.legacy_profiles[version] == _OLD_STAGE_IDS
    assert {stage.stage_id: stage.modes for stage in flow.stages if stage.modes} == {
        **{stage_id: ("implicit",) for stage_id in _IMPLICIT_ONLY},
        **{stage_id: ("explicit",) for stage_id in _EXPLICIT_ONLY},
    }


@pytest.mark.parametrize("mode", ["implicit", "explicit"])
def test_editing_flow_marks_only_the_other_method_inapplicable(mode: str) -> None:
    flow = editing.flow()
    applies = {stage.stage_id for stage in flow.stages if stage_applies(stage, mode)}
    if mode == "implicit":
        assert set(_EXPLICIT_ONLY) <= set(EDITING_STAGE_IDS) - applies
        assert set(_IMPLICIT_ONLY) <= applies
    else:
        assert set(_IMPLICIT_ONLY) <= set(EDITING_STAGE_IDS) - applies
        assert set(_EXPLICIT_ONLY) <= applies


def test_editing_per_mode_skips_are_render_derived() -> None:
    implicit = _headers({"mode": "implicit", "flow_version": CURRENT_FLOW_VERSION, "events": []})
    for stage_id in _EXPLICIT_ONLY:
        assert "evow-stage-skipped" in implicit[EDITING_STAGE_IDS.index(stage_id)]["elem_classes"], stage_id
    for stage_id in _IMPLICIT_ONLY:
        assert "evow-stage-skipped" not in implicit[EDITING_STAGE_IDS.index(stage_id)]["elem_classes"], stage_id

    explicit = _headers({"mode": "explicit", "flow_version": CURRENT_FLOW_VERSION, "events": []})
    for stage_id in _IMPLICIT_ONLY:
        assert "evow-stage-skipped" in explicit[EDITING_STAGE_IDS.index(stage_id)]["elem_classes"], stage_id
    for stage_id in _EXPLICIT_ONLY:
        assert "evow-stage-skipped" not in explicit[EDITING_STAGE_IDS.index(stage_id)]["elem_classes"], stage_id


def test_implicit_run_terminates_every_applicable_row_once_in_declared_order(monkeypatch, tmp_path: Path) -> None:
    track = MaskTrack(np.ones((5, 16, 20), dtype=bool), GroundedBox((0, 0, 20, 16), 0.9), 2)
    events = _run("implicit", monkeypatch, tmp_path, track)

    expected = [stage_id for stage_id in EDITING_STAGE_IDS if stage_id not in _EXPLICIT_ONLY and stage_id != "publish_artifacts"]
    assert _first_seen(events) == expected
    terminals = _terminal_counts(events)
    for stage_id in expected:
        assert terminals.get(stage_id) == 1, (stage_id, terminals)
    assert _running_after_terminal(events) == []
    for stage_id in _EXPLICIT_ONLY:
        assert stage_id not in terminals
    # The aggregate denoise row is one terminal after its real running ticks.
    denoise = [event for event in events if event.stage_id == "vace_denoise"]
    assert denoise[0].status == "running" and denoise[-1].status == "complete"
    assert [event.status for event in denoise].count("complete") == 1


def test_explicit_run_terminates_every_applicable_row_once_in_declared_order(monkeypatch, tmp_path: Path) -> None:
    track = MaskTrack(np.ones((5, 16, 20), dtype=bool), GroundedBox((0, 0, 20, 16), 0.9), 2)
    events = _run("explicit", monkeypatch, tmp_path, track)

    expected = [stage_id for stage_id in EDITING_STAGE_IDS if stage_id not in _IMPLICIT_ONLY and stage_id != "publish_artifacts"]
    assert _first_seen(events) == expected
    terminals = _terminal_counts(events)
    for stage_id in expected:
        assert terminals.get(stage_id) == 1, (stage_id, terminals)
    assert _running_after_terminal(events) == []
    for stage_id in _IMPLICIT_ONLY:
        assert stage_id not in terminals
    # The old double-terminal regression: each scene row terminates exactly once.
    assert terminals["scene_cache"] == 1
    assert terminals["edited_scene"] == 1


def test_scene_cache_hit_completes_without_a_running_tick(monkeypatch, tmp_path: Path) -> None:
    track = MaskTrack(np.ones((5, 16, 20), dtype=bool), GroundedBox((0, 0, 20, 16), 0.9), 2)
    events = _run("explicit", monkeypatch, tmp_path, track, miss=False)

    for stage_id in ("scene_cache", "edited_scene"):
        row = [event for event in events if event.stage_id == stage_id]
        assert [event.status for event in row] == ["complete"], (stage_id, row)
        assert row[0].metrics["cache"] == "hit"


def test_vace_denoise_streams_real_throttled_ticks_with_real_total(monkeypatch, tmp_path: Path) -> None:
    track = MaskTrack(np.ones((5, 16, 20), dtype=bool), GroundedBox((0, 0, 20, 16), 0.9), 2)
    events = _run("implicit", monkeypatch, tmp_path, track, total_steps=20)

    denoise = [event for event in events if event.stage_id == "vace_denoise" and event.status == "running"]
    # The throttle keeps the first, the last, and every third intermediate step.
    assert [event.metrics["step"] for event in denoise] == [0, 3, 6, 9, 12, 15, 18, 19]
    assert all(event.metrics["total"] == 20 for event in denoise)
    assert all(isinstance(event.metrics["timestep"], float) for event in denoise)
    assert all(event.metrics["latent_shape"] == [1, 16, 4, 8, 8] for event in denoise)
    # ``total`` is the scheduler's real count; ``index + 1`` is never claimed as one.
    assert all(event.metrics["step"] < event.metrics["total"] for event in denoise)
    assert _terminal_counts(events)["vace_denoise"] == 1


def test_vace_denoise_falls_back_to_one_coarse_tick_without_a_real_total(monkeypatch, tmp_path: Path) -> None:
    track = MaskTrack(np.ones((5, 16, 20), dtype=bool), GroundedBox((0, 0, 20, 16), 0.9), 2)
    events = _run("implicit", monkeypatch, tmp_path, track, exposes_total=False)

    denoise = [event for event in events if event.stage_id == "vace_denoise" and event.status == "running"]
    assert len(denoise) == 1
    assert denoise[0].metrics == {"frames": 5}
    assert "step" not in denoise[0].metrics and "total" not in denoise[0].metrics
    assert _terminal_counts(events)["vace_denoise"] == 1


def test_no_grounded_target_falls_back_to_full_scene_with_truthful_counts(monkeypatch, tmp_path: Path) -> None:
    events = _run("implicit", monkeypatch, tmp_path, None)

    select = next(event for event in events if event.stage_id == "mask_select" and event.status == "complete")
    assert select.metrics["scope"] == "full_scene_fallback"
    assert select.metrics["mask_coverage"] == 1.0
    assert "grounding_confidence" not in select.metrics
    assert _terminal_counts(events)["mask_select"] == 1

    dino = next(event for event in events if event.stage_id == "dino_detect" and event.status == "complete")
    assert dino.metrics == {"matched": False}
    propagate = next(event for event in events if event.stage_id == "sam2_propagate" and event.status == "complete")
    assert propagate.metrics["propagated_frames"] == 0
    assert propagate.metrics["total_frames"] == 5


def test_explicit_empty_projection_falls_back_to_full_scene_3d(monkeypatch, tmp_path: Path) -> None:
    frames = _frames()
    track = MaskTrack(np.ones((5, 16, 20), dtype=bool), GroundedBox((0, 0, 20, 16), 0.9), 2)
    _patch_ground(monkeypatch, track)
    _patch_scene(monkeypatch, tmp_path)
    _patch_vace(monkeypatch)
    monkeypatch.setattr(editing, "read_editing_episode", lambda *_args: (frames, 12.0))
    monkeypatch.setattr(editing, "source_digest", lambda _path: "source-hash")
    monkeypatch.setattr(editing, "write_video", lambda _f, path, _fps: (path.write_bytes(b"video"), path)[1])
    monkeypatch.setattr(editing, "lift_masks", lambda _scene, _masks: PrimitiveMask(np.asarray([True]), 0.1))
    monkeypatch.setattr(editing, "project_primitive_mask", lambda _scene, _primitive: np.zeros((len(frames), 16, 20), dtype=bool))

    events = [
        item for item in editing.run(
            "unused.mp4", editing.EditingRequest("explicit", "make the sky red", 0), RunArtifacts(tmp_path / "empty-projection"), _settings(),
        )
        if isinstance(item, StageEvent)
    ]
    project = next(event for event in events if event.stage_id == "mask_project" and event.status == "complete")
    assert project.metrics == {"scope": "full_scene_3d_fallback"}
    assert _terminal_counts(events)["mask_project"] == 1


def test_implicit_saves_the_expanded_direct_mask_and_never_writes_a_primitive_mask(monkeypatch, tmp_path: Path) -> None:
    """Implicit expands the 2D tracked mask before saving ``edit_mask.npy``.

    The implicit expansion row stays before ``prepare_mask``; it must not lift or
    project anything, and an implicit run has no primitive selection to save.
    """
    frames = _frames()
    direct = np.zeros((5, 16, 20), dtype=bool)
    direct[:, 6:9, 8:11] = True
    track = MaskTrack(direct, GroundedBox((8, 6, 11, 9), 0.9), 2)
    _patch_ground(monkeypatch, track)
    _patch_scene(monkeypatch, tmp_path)
    _patch_vace(monkeypatch)
    monkeypatch.setattr(editing, "read_editing_episode", lambda *_args: (frames, 12.0))
    monkeypatch.setattr(editing, "source_digest", lambda _path: "source-hash")
    monkeypatch.setattr(editing, "write_video", lambda _f, path, _fps: (path.write_bytes(b"video"), path)[1])
    monkeypatch.setattr(editing, "lift_masks", lambda *_a: (_ for _ in ()).throw(AssertionError("implicit must not lift")))
    monkeypatch.setattr(editing, "project_primitive_mask", lambda *_a: (_ for _ in ()).throw(AssertionError("implicit must not project")))

    artifacts = RunArtifacts(tmp_path / "implicit-order")
    events = [
        item for item in editing.run(
            "unused.mp4", editing.EditingRequest("implicit", "make the sky red", 3), artifacts, _settings(),
        )
        if isinstance(item, StageEvent)
    ]

    radius = _settings().edit_mask_expand_px
    saved_edit_mask = np.load(artifacts.file("edit_mask.npy")).astype(bool)
    assert np.array_equal(saved_edit_mask, expand_masks(direct, radius=radius))
    # The raw target is smaller than its dilation, so this asserts real expansion.
    assert not np.array_equal(saved_edit_mask, direct)
    assert not (artifacts.run_dir / "base_primitive_mask.npy").exists()
    terminals = _terminal_counts(events)
    assert terminals["mask_expand"] == 1
    assert "mask_expand_projected" not in terminals


def test_explicit_lifts_the_unexpanded_mask_then_expands_the_projected_mask(monkeypatch, tmp_path: Path) -> None:
    """Explicit saves the raw primitive selection and expands only the projection.

    This is the order regression: the old code expanded ``direct_masks`` before
    lifting, which could select extra Gaussians and change
    ``base_primitive_mask.npy``. Here ``lift_masks`` must receive the tracked mask
    untouched, the saved primitive membership must be its unexpanded output, and
    ``edit_mask.npy`` must be the dilation of the projection, not the projection
    of a dilation.
    """
    frames = _frames()
    direct = np.zeros((5, 16, 20), dtype=bool)
    direct[:, 6:9, 8:11] = True
    track = MaskTrack(direct, GroundedBox((8, 6, 11, 9), 0.9), 2)
    _patch_ground(monkeypatch, track)
    _patch_scene(monkeypatch, tmp_path)
    _patch_vace(monkeypatch)
    monkeypatch.setattr(editing, "read_editing_episode", lambda *_args: (frames, 12.0))
    monkeypatch.setattr(editing, "source_digest", lambda _path: "source-hash")
    monkeypatch.setattr(editing, "write_video", lambda _f, path, _fps: (path.write_bytes(b"video"), path)[1])

    lifted: dict = {}
    selection = np.asarray([True, False, True, False], dtype=bool)
    projection = np.zeros((len(frames), 16, 20), dtype=bool)
    projection[:, 5, 7] = True

    def fake_lift(_scene, masks):
        lifted["masks"] = np.array(masks)
        return PrimitiveMask(selection, 0.75)

    monkeypatch.setattr(editing, "lift_masks", fake_lift)
    monkeypatch.setattr(editing, "project_primitive_mask", lambda _scene, _primitive: projection)

    artifacts = RunArtifacts(tmp_path / "explicit-order")
    events = [
        item for item in editing.run(
            "unused.mp4", editing.EditingRequest("explicit", "make the sky red", 3), artifacts, _settings(),
        )
        if isinstance(item, StageEvent)
    ]

    radius = _settings().edit_mask_expand_px
    assert np.array_equal(lifted["masks"], direct)
    assert not np.array_equal(lifted["masks"], expand_masks(direct, radius=radius))
    assert np.array_equal(np.load(artifacts.file("base_primitive_mask.npy")), selection)
    saved_edit_mask = np.load(artifacts.file("edit_mask.npy")).astype(bool)
    assert np.array_equal(saved_edit_mask, expand_masks(projection, radius=radius))
    # The projection is a single pixel, so its dilation is strictly larger: the
    # saved mask can only match if the expansion ran after the projection.
    assert saved_edit_mask.sum() > projection.sum()
    terminals = _terminal_counts(events)
    assert terminals["mask_expand_projected"] == 1
    assert "mask_expand" not in terminals


def test_legacy_v4_editing_trace_maps_old_ids_onto_the_atomic_rows() -> None:
    """A saved v4 trace's five coarse bars land on the rows that now own each operation."""
    events = [
        {"stage": "old", "stage_id": old_id, "status": "complete", "elapsed_seconds": index}
        for index, old_id in enumerate(_OLD_STAGE_IDS, start=1)
    ]
    trace = {"mode": "implicit", "flow_version": 4, "events": events}
    for new_id in ("episode_decode", "scope_classify", "prepare_mask", "vace_prepare", "publish_artifacts"):
        assert "evow-stage-complete" in _header(trace, new_id)["elem_classes"], new_id
    # A union row v4 genuinely lacked is marked not recorded, never left waiting.
    assert "Not recorded by this legacy run." in pages._updates("editing", trace)[_GROUPS + len(EDITING_STAGE_IDS) + EDITING_STAGE_IDS.index("finalize")]["value"]
    assert "Legacy workflow version v4" in pages._updates("editing", trace)[-1]["value"]


def test_legacy_v4_prepare_method_maps_to_editing_not_the_global_future_alias() -> None:
    """The feature-scoped alias keeps ``prepare_method`` on editing's own row."""
    trace = {
        "mode": "explicit", "flow_version": 4,
        "events": [{"stage": "Old prepare label", "stage_id": "prepare_method", "status": "complete", "elapsed_seconds": 1}],
    }
    assert "evow-stage-complete" in _header(trace, "prepare_mask")["elem_classes"]


def test_unknown_pre_current_editing_version_shows_the_legacy_notice() -> None:
    trace = {"mode": "implicit", "flow_version": 0, "events": []}
    assert "Legacy workflow version v0" in pages._updates("editing", trace)[-1]["value"]


def test_editing_handler_wraps_trace_finish_in_publish_artifacts(monkeypatch, tmp_path: Path) -> None:
    """The page emits ``publish_artifacts`` running then complete around ``finish``."""
    import json

    assets = tmp_path / "ASSETS"
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(pages, "ASSETS", assets)

    def fake_run(_source, _request, artifacts, settings=None):
        yield StageEvent("Decode source episode", "running", "decoding", stage_id="episode_decode")
        yield StageEvent("Decode source episode", "complete", "decoded", metrics={"frames": 5}, stage_id="episode_decode")
        # The generated frame count lives on the run's own stage metric, never in
        # the result metadata; the handler must source the publish summary here.
        yield StageEvent("Write edit proposal", "complete", "proposed", metrics={"output_frames": 5}, stage_id="write_proposal")
        primary = artifacts.file("edited.mp4")
        primary.write_bytes(b"video")
        yield FeatureResult(primary=primary, metadata={"mode": "implicit", "scope": "grounded_target"})

    monkeypatch.setattr(pages.editing, "run", fake_run)

    from GREENFIELD.features.text_settings import load_text_settings

    settings = load_text_settings()
    config_values = [getattr(settings, name) for name in pages._EDITABLE_EDITING_FIELDS]
    page = pages.build_editing()
    handler = next(fn.fn for fn in page.fns.values() if getattr(fn.fn, "__name__", "") == "handle")
    stream = handler(str(source), "make the sky red", "implicit", 0, *config_values)
    while True:
        try:
            next(stream)
        except StopIteration:
            break

    trace_path = next((assets / "editing" / "runs").glob("*/trace.json"))
    events = json.loads(trace_path.read_text())["events"]
    statuses = [event["status"] for event in events if event.get("stage_id") == "publish_artifacts"]
    assert statuses == ["running", "complete"]
    complete = next(event for event in events if event.get("stage_id") == "publish_artifacts" and event["status"] == "complete")
    assert complete["metrics"] == {"output_frames": 5, "scope": "grounded_target", "mode": "implicit"}
