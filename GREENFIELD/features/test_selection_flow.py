"""Text Query's atomic 22-row union flow, per-mode applicability, and emissions.

These tests pin the declaration (ids, groups, modes, frozen v1 labels, per-version
recorded sets), the render-derived per-mode skips, the aggregate-row lifecycle,
and the truthful edge cases (cache hit without a running tick, zero candidates,
zero frames, degenerate mask rejection).
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from GREENFIELD.app_core.contracts import RunArtifacts, StageEvent
from GREENFIELD.app_core.flow import CURRENT_FLOW_VERSION, LEGACY_FLOW_VERSION, stage_applies
from GREENFIELD.features import pages, selection, text_models, text_segmentation
from GREENFIELD.features.scene_masks import PrimitiveMask
from GREENFIELD.features.text_media import VideoInfo
from GREENFIELD.features.text_models import GroundedBox
from GREENFIELD.features.text_retrieval import CandidateWindow
from GREENFIELD.features.text_segmentation import MaskTrack


SELECTION_STAGE_IDS = (
    "inspect_source", "siglip_load", "text_embed", "frame_embed", "normalize",
    "faiss_index", "faiss_search", "window_rank", "candidate_decode", "dino_detect",
    "sam2_setup", "sam2_propagate", "mask_validate", "resolve_method", "scene_cache",
    "scene_reconstruct", "mask_lift", "mask_project", "splat_export", "overlay",
    "write_clips", "final_rerank",
)
_IMPLICIT_ONLY = ("resolve_method",)
_EXPLICIT_ONLY = ("scene_cache", "scene_reconstruct", "mask_lift", "mask_project", "splat_export")
_GROUPS = 5  # selection's five parent accordions precede the stage headers

_AGGREGATE_ROWS = (
    "candidate_decode", "dino_detect", "sam2_setup", "sam2_propagate", "mask_validate",
    "resolve_method", "overlay", "write_clips",
)


def _headers(trace: dict) -> list[dict]:
    """Return the 22 stage-header updates for a selection render."""
    updates = pages._updates("selection", trace)
    return list(updates[_GROUPS:_GROUPS + len(SELECTION_STAGE_IDS)])


def _card(trace: dict, stage_id: str) -> str:
    updates = pages._updates("selection", trace)
    start = _GROUPS + len(SELECTION_STAGE_IDS)
    return updates[start + SELECTION_STAGE_IDS.index(stage_id)]["value"]


def _frames(count: int = 3) -> list[np.ndarray]:
    return [np.zeros((16, 20, 3), dtype=np.uint8) for _ in range(count)]


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        query_sample_fps=2, query_max_source_seconds=300, query_clip_seconds=4,
        query_explicit_candidate_pool=8, query_result_fps=12, query_max_results=5,
        grounding_threshold=0.25, low_specificity_coverage=0.65,
    )


def _candidate(start: float = 0.0, score: float = 0.9) -> CandidateWindow:
    return CandidateWindow(0, start, start + 4, score)


def _patch_retrieval(monkeypatch, frames, candidates) -> None:
    """Replace retrieval workers with deterministic spies for a query run."""
    monkeypatch.setattr(selection, "load_text_settings", lambda: _settings())
    monkeypatch.setattr(selection, "read_uniform_samples", lambda *_args: (
        [frames[0]], [0.0], VideoInfo(12, 120, 10, 20, 16),
    ))

    def fake_scores(_frames, _query, **_kwargs):
        if False:
            yield
        return np.asarray([score for score in [0.9]], dtype=np.float32)

    monkeypatch.setattr(selection, "semantic_scores_steps", fake_scores)
    monkeypatch.setattr(selection, "ranked_windows", lambda *_args: list(candidates))
    monkeypatch.setattr(selection, "read_interval", lambda *_args: (frames, 12.0))
    monkeypatch.setattr(selection, "source_digest", lambda _path: "source-hash")
    monkeypatch.setattr(selection, "write_video", lambda _frames, path, _fps: (path.touch(), path)[1])


def _patch_ground(monkeypatch, masks) -> None:
    def fake_ground_steps(_frames, _query, anchor_index=None, threshold=0.25, *, candidate_index=1, total_candidates=1, stage_ids):
        yield StageEvent("Ground target with Grounding DINO", "running", "Detected the target.", metrics={"candidate_index": candidate_index, "confidence": 0.8}, stage_id=stage_ids["dino_detect"])
        yield StageEvent("Prepare SAM2 tracking", "running", "Initializing SAM2.", stage_id=stage_ids["sam2_setup"])
        yield StageEvent("Propagate SAM2 masks", "running", "Forward.", metrics={"reversed": False, "propagated_frames": 1, "total_frames": len(masks)}, stage_id=stage_ids["sam2_propagate"])
        yield StageEvent("Propagate SAM2 masks", "running", "Backward.", metrics={"reversed": True, "propagated_frames": 1, "total_frames": len(masks)}, stage_id=stage_ids["sam2_propagate"])
        yield StageEvent("Validate tracked masks", "running", "Mask is valid.", metrics={"mask_coverage": float(masks.mean())}, stage_id=stage_ids["mask_validate"])
        return MaskTrack(masks, GroundedBox((5, 4, 15, 12), 0.8), 2)

    monkeypatch.setattr(selection, "ground_and_track_steps", fake_ground_steps)


def _terminal_counts(events: list[StageEvent]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        if isinstance(event, StageEvent) and event.status in {"complete", "error", "skipped"}:
            counts[event.stage_id or event.stage] = counts.get(event.stage_id or event.stage, 0) + 1
    return counts


def _running_after_terminal(events: list[StageEvent]) -> list[str]:
    terminated: set[str] = set()
    violations: list[str] = []
    for event in events:
        if not isinstance(event, StageEvent):
            continue
        stage_id = event.stage_id or event.stage
        if event.status == "running":
            if stage_id in terminated:
                violations.append(stage_id)
        elif event.status in {"complete", "error", "skipped"}:
            terminated.add(stage_id)
    return violations


def test_selection_flow_declares_ordered_ids_groups_and_frozen_v1_labels() -> None:
    flow = selection.flow()

    assert tuple(stage.stage_id for stage in flow.stages) == SELECTION_STAGE_IDS
    assert [
        (group.label, group.stage_ids) for group in flow.groups
    ] == [
        ("Input", ("inspect_source",)),
        ("Retrieve", ("siglip_load", "text_embed", "frame_embed", "normalize", "faiss_index", "faiss_search", "window_rank")),
        ("Ground + track", ("candidate_decode", "dino_detect", "sam2_setup", "sam2_propagate", "mask_validate")),
        ("Resolve 3D method", ("resolve_method", "scene_cache", "scene_reconstruct", "mask_lift", "mask_project", "splat_export")),
        ("Save", ("overlay", "write_clips", "final_rerank")),
    ]
    grouped = [stage_id for group in flow.groups for stage_id in group.stage_ids]
    assert sorted(grouped) == sorted(SELECTION_STAGE_IDS)
    assert flow.legacy_stage_ids == (
        "Inspect source video", "Retrieve relevant intervals",
        "Ground and track binary target", "Resolve selected method", "Save highlighted clips",
    )
    # The coarse bars map onto the atomic rows that now own each operation.
    recorded = ("inspect_source", "window_rank", "mask_validate", "resolve_method", "write_clips")
    assert flow.legacy_profiles[LEGACY_FLOW_VERSION] == recorded
    assert flow.legacy_profiles[2] == recorded
    # v3 predates the v4 atomization and wrote the same five coarse bars; its
    # profile entry is what keeps its valid rows from rendering "Not recorded".
    assert flow.legacy_profiles[3] == recorded
    assert {stage.stage_id: stage.modes for stage in flow.stages if stage.modes} == {
        "resolve_method": ("implicit",),
        **{stage_id: ("explicit",) for stage_id in _EXPLICIT_ONLY},
    }


@pytest.mark.parametrize("mode", ["implicit", "explicit"])
def test_selection_flow_marks_only_the_other_method_inapplicable(mode: str) -> None:
    flow = selection.flow()
    applies = {stage.stage_id for stage in flow.stages if stage_applies(stage, mode)}
    if mode == "implicit":
        assert set(_EXPLICIT_ONLY) <= set(SELECTION_STAGE_IDS) - applies
        assert set(_IMPLICIT_ONLY) <= applies
    else:
        assert set(_IMPLICIT_ONLY) <= set(SELECTION_STAGE_IDS) - applies
        assert set(_EXPLICIT_ONLY) <= applies


def test_selection_per_mode_skips_are_render_derived() -> None:
    """Inapplicable rows render Skipped without a producer hand-emitting a skip."""
    implicit = _headers({"mode": "implicit", "flow_version": CURRENT_FLOW_VERSION, "events": []})
    for stage_id in _EXPLICIT_ONLY:
        header = implicit[SELECTION_STAGE_IDS.index(stage_id)]
        assert "evow-stage-skipped" in header["elem_classes"], stage_id
    assert "skipped" not in " ".join(implicit[SELECTION_STAGE_IDS.index("resolve_method")]["elem_classes"])

    explicit = _headers({"mode": "explicit", "flow_version": CURRENT_FLOW_VERSION, "events": []})
    assert "evow-stage-skipped" in explicit[SELECTION_STAGE_IDS.index("resolve_method")]["elem_classes"]
    for stage_id in _EXPLICIT_ONLY:
        assert "evow-stage-skipped" not in explicit[SELECTION_STAGE_IDS.index(stage_id)]["elem_classes"], stage_id


def test_selection_aggregate_rows_emit_ticks_and_exactly_one_terminal(monkeypatch, tmp_path: Path) -> None:
    """Every looping row weaves running... and exactly one owner terminal."""
    frames = _frames()
    masks = np.ones((len(frames), 16, 20), dtype=bool)
    _patch_retrieval(monkeypatch, frames, [_candidate()])
    _patch_ground(monkeypatch, masks)
    monkeypatch.setattr(selection, "_consume_scene", lambda *_args: (_ for _ in ()).throw(AssertionError("implicit query must not build a scene")))

    events = [item for item in selection.run("unused.mp4", selection.SelectionRequest("implicit", "tree"), RunArtifacts(tmp_path / "implicit")) if isinstance(item, StageEvent)]

    terminals = _terminal_counts(events)
    for stage_id in _AGGREGATE_ROWS:
        assert terminals.get(stage_id) == 1, (stage_id, terminals)
    assert terminals.get("final_rerank") == 1
    assert _running_after_terminal(events) == []
    # The propagation row aggregates both directions behind one row id.
    propagate = [e for e in events if e.stage_id == "sam2_propagate" and e.status == "running"]
    assert {e.metrics["reversed"] for e in propagate} == {False, True}


def test_cache_hit_completes_without_a_running_tick(monkeypatch, tmp_path: Path) -> None:
    """A reused scene reports complete on both scene rows with no reconstruction tick."""
    frames = _frames()
    masks = np.ones((len(frames), 16, 20), dtype=bool)
    _patch_retrieval(monkeypatch, frames, [_candidate()])
    _patch_ground(monkeypatch, masks)
    scene = SimpleNamespace(key="scene-key", splats=(), cache_hit=True)
    monkeypatch.setattr(selection, "_consume_scene", lambda *_args: _scene_generator(scene))
    monkeypatch.setattr(selection, "lift_masks", lambda _scene, _masks: PrimitiveMask(np.asarray([True]), 0.5))
    monkeypatch.setattr(selection, "project_primitive_mask", lambda *_args: masks)

    events = [item for item in selection.run("unused.mp4", selection.SelectionRequest("explicit", "tree"), RunArtifacts(tmp_path / "cache-hit")) if isinstance(item, StageEvent)]

    cache_events = [e for e in events if e.stage_id == "scene_cache"]
    assert [e.status for e in cache_events] == ["complete"]
    assert cache_events[0].metrics["cache"] == "hit"
    reconstruct = [e for e in events if e.stage_id == "scene_reconstruct"]
    assert [e.status for e in reconstruct] == ["complete"]
    assert reconstruct[0].metrics["cache"] == "hit"


def test_zero_candidates_completes_aggregate_rows(monkeypatch, tmp_path: Path) -> None:
    """An empty candidate pool completes the aggregate rows with a real zero."""
    frames = _frames()
    _patch_retrieval(monkeypatch, frames, [])
    events: list[StageEvent] = []
    with pytest.raises(ValueError, match="No reliable text-grounded regions"):
        for item in selection.run("unused.mp4", selection.SelectionRequest("implicit", "tree"), RunArtifacts(tmp_path / "zero")):
            if isinstance(item, StageEvent):
                events.append(item)

    decode = next(e for e in events if e.stage_id == "candidate_decode" and e.status == "complete")
    assert decode.metrics["total_candidates"] == 0
    propagate = next(e for e in events if e.stage_id == "sam2_propagate" and e.status == "complete")
    assert propagate.metrics["tracked_candidates"] == 0
    assert _terminal_counts(events).get("candidate_decode") == 1


def test_zero_frame_embedding_still_completes(monkeypatch) -> None:
    """An empty frame list still emits running then complete for frame_embed."""
    class FakeInputs(dict):
        def to(self, _device):
            return self

    class FakeProcessor:
        def __call__(self, **_kwargs):
            return FakeInputs()

    class FakeModel:
        def parameters(self):
            yield torch.zeros(1)

        def get_text_features(self, **_kwargs):
            return torch.ones(2, 4)

        def get_image_features(self, **_kwargs):
            return torch.ones(1, 4)

    monkeypatch.setattr(text_models, "require_package", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(text_models, "siglip_model", lambda: (FakeProcessor(), FakeModel()))
    monkeypatch.setattr(text_models, "device", lambda: "cpu")
    monkeypatch.setattr(text_models, "load_text_settings", lambda: SimpleNamespace(semantic_checkpoint=Path("/checkpoints/siglip")))

    events: list[StageEvent] = []
    steps = text_models.semantic_scores_steps([], "tree", stage_ids=selection._RETRIEVAL_STAGE_IDS)
    try:
        while True:
            events.append(next(steps))
    except StopIteration as complete:
        scores = complete.value

    frame_events = [e for e in events if e.stage_id == "frame_embed"]
    assert [e.status for e in frame_events] == ["running", "complete"]
    assert all(e.metrics["total_frames"] == 0 for e in frame_events)
    assert scores.shape == (0,)


def test_mask_validate_rejects_a_degenerate_track(monkeypatch) -> None:
    """A zero-coverage SAM2 mask is reported and rejected, never highlighted."""
    frames = _frames()
    degenerate = np.zeros((len(frames), 16, 20), dtype=bool)
    monkeypatch.setattr(text_segmentation, "ground_query", lambda *_args, **_kwargs: GroundedBox((2, 2, 14, 14), 0.9))

    def fake_track_steps(*_args, **_kwargs):
        if False:
            yield
        return degenerate

    monkeypatch.setattr(text_segmentation, "track_box_steps", fake_track_steps)

    events: list[StageEvent] = []
    steps = text_segmentation.ground_and_track_steps(frames, "tree", stage_ids=selection._GROUND_STAGE_IDS)
    try:
        while True:
            events.append(next(steps))
    except StopIteration as complete:
        result = complete.value

    assert result is None
    validate = [e for e in events if e.stage_id == "mask_validate"]
    assert validate[-1].metrics["mask_coverage"] == 0.0


def test_legacy_v1_name_only_events_map_onto_the_atomic_rows() -> None:
    """A v1 trace's id-less old labels land on the rows that now own each operation."""
    events = [
        {"stage": name, "status": "complete", "detail": "done", "elapsed_seconds": index}
        for index, name in enumerate(
            ("Inspect source video", "Retrieve relevant intervals", "Ground and track binary target", "Save highlighted clips"),
            start=1,
        )
    ]
    trace = {"mode": "explicit", "flow_version": LEGACY_FLOW_VERSION, "events": events}
    headers = _headers(trace)
    for stage_id in ("inspect_source", "window_rank", "mask_validate", "write_clips"):
        header = headers[SELECTION_STAGE_IDS.index(stage_id)]
        assert "evow-stage-complete" in header["elem_classes"], stage_id
    # A union row v1 genuinely lacked is marked not recorded, never left waiting.
    siglip = headers[SELECTION_STAGE_IDS.index("siglip_load")]
    assert "evow-stage-skipped" in siglip["elem_classes"]
    assert "Not recorded by this legacy run." in _card(trace, "siglip_load")


def test_legacy_v2_retired_ids_map_by_alias_onto_the_atomic_rows() -> None:
    """A v2 trace's retired aggregate ids land on the atomic rows by alias."""
    events = [
        {"stage": "old", "stage_id": old_id, "status": "complete", "elapsed_seconds": index}
        for index, old_id in enumerate(("inspect_source", "retrieve_intervals", "ground_track", "save_clips"), start=1)
    ]
    trace = {"mode": "explicit", "flow_version": 2, "events": events}
    headers = _headers(trace)
    for stage_id in ("inspect_source", "window_rank", "mask_validate", "write_clips"):
        assert "evow-stage-complete" in headers[SELECTION_STAGE_IDS.index(stage_id)]["elem_classes"], stage_id
    assert "Legacy workflow version v2" in pages._updates("selection", trace)[-1]["value"]


def test_legacy_explicit_resolve_method_stays_representable() -> None:
    """A legacy explicit run's projected 3D step is not rewritten as method-skipped."""
    trace = {
        "mode": "explicit", "flow_version": 2,
        "events": [{"stage": "old", "stage_id": "resolve_method", "status": "complete", "elapsed_seconds": 1}],
    }
    header = _headers(trace)[SELECTION_STAGE_IDS.index("resolve_method")]
    assert "evow-stage-complete" in header["elem_classes"]


def test_legacy_v3_trace_resolves_recorded_rows_instead_of_not_recorded() -> None:
    """A v3 trace's valid rows are never mislabeled "Not recorded" by v4's profile.

    v3 predates the v4 atomization and wrote the same five coarse bars. Once the
    global flow version is bumped to 4, a persisted v3 trace must resolve against
    its own recorded set rather than falling through to the v1 name fallback
    (whose labels no longer match the atomic rows).
    """
    trace = {"mode": "explicit", "flow_version": 3, "events": []}
    headers = _headers(trace)
    for stage_id in ("inspect_source", "window_rank", "mask_validate", "resolve_method", "write_clips"):
        header = headers[SELECTION_STAGE_IDS.index(stage_id)]
        assert "evow-stage-skipped" not in header["elem_classes"], stage_id
        assert "Not recorded by this legacy run." not in _card(trace, stage_id), stage_id
    # A row v3 genuinely lacked still reports "not recorded".
    assert "Not recorded by this legacy run." in _card(trace, "scene_cache")
    # An explicit v3 event on ``resolve_method`` stays a real completed row.
    completed = {
        "mode": "explicit", "flow_version": 3,
        "events": [{"stage": "old", "stage_id": "resolve_method", "status": "complete", "elapsed_seconds": 1}],
    }
    assert "evow-stage-complete" in _headers(completed)[SELECTION_STAGE_IDS.index("resolve_method")]["elem_classes"]


@pytest.mark.parametrize("flow_version", [LEGACY_FLOW_VERSION, 2, 3])
def test_legacy_explicit_resolve_method_without_an_event_is_historical(flow_version: int) -> None:
    """A legacy explicit trace's resolve row waits; it is never method-skipped.

    The current explicit flow marks ``resolve_method`` implicit-only, but a v1/v2/v3
    explicit run could record it. Applicability must come from that version's
    recorded profile before the current mode rules, so the row is historical
    (Waiting/Interrupted) rather than "Not used by this method".
    """
    trace = {"mode": "explicit", "flow_version": flow_version, "events": []}
    header = _headers(trace)[SELECTION_STAGE_IDS.index("resolve_method")]
    assert "evow-stage-waiting" in header["elem_classes"]
    assert "evow-stage-skipped" not in header["elem_classes"]
    card = _card(trace, "resolve_method")
    assert "Not used by this method." not in card
    assert "Not recorded by this legacy run." not in card


def test_zero_frame_retrieval_terminates_downstream_rows_before_raising(monkeypatch, tmp_path: Path) -> None:
    """A retrieval sample with no frames still terminates every downstream row.

    ``ranked_windows`` rejects empty scores, so the empty sample must be handled
    as an empty candidate pool: the aggregate, resolve, overlay, write, and rerank
    rows reach exactly one terminal each before the user-facing "no matches" error.
    """
    monkeypatch.setattr(selection, "load_text_settings", lambda: _settings())
    monkeypatch.setattr(selection, "read_uniform_samples", lambda *_args: (
        [], [], VideoInfo(12, 120, 10, 20, 16),
    ))

    def fake_scores(_frames, _query, **_kwargs):
        if False:
            yield
        return np.asarray([], dtype=np.float32)

    monkeypatch.setattr(selection, "semantic_scores_steps", fake_scores)
    monkeypatch.setattr(selection, "source_digest", lambda _path: "source-hash")
    # ``ranked_windows`` is deliberately not patched: the real function rejects
    # empty scores, proving the caller treats zero frames as an empty pool.

    events: list[StageEvent] = []
    with pytest.raises(ValueError, match="No reliable text-grounded regions"):
        for item in selection.run("unused.mp4", selection.SelectionRequest("implicit", "tree"), RunArtifacts(tmp_path / "no-frames")):
            if isinstance(item, StageEvent):
                events.append(item)

    terminals = _terminal_counts(events)
    for stage_id in (*_AGGREGATE_ROWS, "final_rerank"):
        assert terminals.get(stage_id) == 1, (stage_id, terminals)
    assert _running_after_terminal(events) == []
    decode = next(e for e in events if e.stage_id == "candidate_decode" and e.status == "complete")
    assert decode.metrics["total_candidates"] == 0
    propagate = next(e for e in events if e.stage_id == "sam2_propagate" and e.status == "complete")
    assert propagate.metrics["tracked_candidates"] == 0
    rerank = next(e for e in events if e.stage_id == "final_rerank" and e.status == "complete")
    assert rerank.metrics["kept"] == 0


def test_explicit_zero_candidates_reports_no_scene_and_no_reconstruction(monkeypatch, tmp_path: Path) -> None:
    """No candidate reaching scene caching reports "none", never a fabricated scene."""
    frames = _frames()
    _patch_retrieval(monkeypatch, frames, [])

    events: list[StageEvent] = []
    with pytest.raises(ValueError, match="No reliable text-grounded regions"):
        for item in selection.run("unused.mp4", selection.SelectionRequest("explicit", "tree"), RunArtifacts(tmp_path / "explicit-zero")):
            if isinstance(item, StageEvent):
                events.append(item)

    cache = next(e for e in events if e.stage_id == "scene_cache" and e.status == "complete")
    assert cache.metrics["cache"] == "none"
    assert cache.metrics["checked_candidates"] == 0
    assert cache.metrics["hit_candidates"] == 0
    assert cache.metrics["miss_candidates"] == 0
    reconstruct = next(e for e in events if e.stage_id == "scene_reconstruct" and e.status == "complete")
    assert "splats" not in reconstruct.metrics
    assert reconstruct.metrics["scene_key"] is None


def test_explicit_mixed_cache_outcomes_report_truthful_aggregates(monkeypatch, tmp_path: Path) -> None:
    """A run that rebuilds one scene and reuses another reports "mixed" with counts."""
    frames = _frames()
    masks = np.ones((len(frames), 16, 20), dtype=bool)
    _patch_retrieval(monkeypatch, frames, [_candidate(start=0.0), _candidate(start=4.0)])
    _patch_ground(monkeypatch, masks)
    outcomes = iter((
        SimpleNamespace(key="scene-a", splats=(Path("a1"), Path("a2")), cache_hit=False),
        SimpleNamespace(key="scene-b", splats=(Path("b1"),), cache_hit=True),
    ))
    monkeypatch.setattr(selection, "_consume_scene", lambda *_args: _scene_generator(next(outcomes)))
    monkeypatch.setattr(selection, "lift_masks", lambda _scene, _masks: PrimitiveMask(np.asarray([True]), 0.5))
    monkeypatch.setattr(selection, "project_primitive_mask", lambda *_args: masks)

    events = [
        item for item in selection.run("unused.mp4", selection.SelectionRequest("explicit", "tree"), RunArtifacts(tmp_path / "explicit-mixed"))
        if isinstance(item, StageEvent)
    ]

    cache = next(e for e in events if e.stage_id == "scene_cache" and e.status == "complete")
    assert cache.metrics["cache"] == "mixed"
    assert cache.metrics["checked_candidates"] == 2
    assert cache.metrics["hit_candidates"] == 1
    assert cache.metrics["miss_candidates"] == 1
    reconstruct = next(e for e in events if e.stage_id == "scene_reconstruct" and e.status == "complete")
    assert reconstruct.metrics["cache"] == "miss"
    # The freshly rebuilt scene's splats, never the later reused one.
    assert reconstruct.metrics["splats"] == 2


def _scene_generator(scene):
    """Return an exhausted generator that yields ``scene`` as its StopIteration value."""
    if False:
        yield
    return scene
