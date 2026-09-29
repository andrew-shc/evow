"""Future View's declared union flow, per-mode applicability, and stage emissions.

The Future View page renders one static 15-row union layout; a methodology only
runs a subset, and every row it does not run must be terminal or clearly waiting.
These tests pin the declaration (ids, groups, modes, frozen v1 labels) and the
real adapter emissions that drive those rows.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import gradio as gr
import pytest

from GREENFIELD.app_core.contracts import FeatureResult, RunArtifacts, StageEvent
from GREENFIELD.app_core.flow import CURRENT_FLOW_VERSION, stage_applies
from GREENFIELD.features import pages
from GREENFIELD.features import future as future_module
from GREENFIELD.features import future_explicit
from GREENFIELD.features.future import FutureRequest
from GREENFIELD.replay.splat import timeline_keyframe_indices
from GREENFIELD.replay.worker_events import WorkerEventWriter


# Declaration order, so the position of every header in a rendered update tuple
# is unambiguous.
FUTURE_STAGE_IDS = (
    "source_tail", "training_cache", "training_pairs", "train_lora", "train_artifact",
    "svd_load", "depth_infer", "gaussian_fit", "motion_extrapolate", "splat_render_export",
    "svd_generate", "encode_video", "write_trace", "validate_artifacts", "save_forecast",
)
_EXPLICIT_ONLY = (
    "depth_infer", "gaussian_fit", "motion_extrapolate", "splat_render_export", "validate_artifacts",
)
_IMPLICIT_ONLY = ("svd_load", "svd_generate")
_TRAINING = ("training_cache", "training_pairs", "train_lora", "train_artifact")


def _future_headers(mode: str) -> list[dict]:
    """Return the stage-header updates for the current Future View flow."""
    updates = pages._updates("future", {"mode": mode, "flow_version": CURRENT_FLOW_VERSION, "events": []})
    # Four parent groups precede the fifteen stage headers.
    return list(updates[4:4 + len(FUTURE_STAGE_IDS)])


def _status_by_id(events: list[StageEvent]) -> dict[str, list[str]]:
    statuses: dict[str, list[str]] = {}
    for event in events:
        if not isinstance(event, StageEvent):
            continue
        statuses.setdefault(event.stage_id, []).append(event.status)
    return statuses


def _assert_single_terminal_lifecycle(events: list[StageEvent]) -> None:
    """Every touched stage weaves running... one terminal, before the next runs."""
    seen_running: set[str] = set()
    terminal: set[str] = set()
    current: str | None = None
    for event in events:
        if not isinstance(event, StageEvent):
            continue
        stage_id = event.stage_id or event.stage
        if event.status == "running":
            if stage_id != current:
                abandoned = seen_running - terminal
                assert not abandoned, f"{abandoned} left running before {stage_id} ran"
                current = stage_id
            assert stage_id not in terminal, f"{stage_id} ran after terminating"
            seen_running.add(stage_id)
        else:
            assert stage_id not in terminal, f"{stage_id} terminated twice"
            terminal.add(stage_id)
    assert seen_running <= terminal, f"{seen_running - terminal} never reached a terminal"


def _implicit_events(monkeypatch, tmp_path) -> list[StageEvent]:
    settings = SimpleNamespace(
        checkpoint=Path("/models/svd"), output_seconds=2, output_frames=3, output_fps=1,
        chunk_frames=2, max_side=64,
    )

    def fake_steps(frame, count, seed, adapter=None, settings=None):
        yield 0, None
        yield 1, [np.zeros((2, 2, 3), dtype=np.uint8)]
        yield 3, [np.zeros((2, 2, 3), dtype=np.uint8)]

    monkeypatch.setattr(future_module, "generate_continuation_steps", fake_steps)
    monkeypatch.setattr(future_module, "uses_cpu_offload", lambda: True)
    monkeypatch.setattr(future_module, "write_video", lambda frames, path, fps: path)
    return list(future_module.run(
        [np.zeros((2, 2, 3), dtype=np.uint8)], 12,
        FutureRequest("implicit", 0), RunArtifacts(tmp_path), settings,
    ))


def test_future_flow_declares_ordered_ids_groups_and_frozen_v1_labels() -> None:
    flow = future_module.flow()

    assert tuple(stage.stage_id for stage in flow.stages) == FUTURE_STAGE_IDS
    assert [
        (group.label, group.stage_ids) for group in flow.groups
    ] == [
        ("Input", ("source_tail",)),
        ("Prepare method", tuple(FUTURE_STAGE_IDS[1:5]) + ("svd_load", "depth_infer")),
        ("Generate", ("gaussian_fit", "motion_extrapolate", "splat_render_export", "svd_generate")),
        ("Save", ("encode_video", "write_trace", "validate_artifacts", "save_forecast")),
    ]
    # Every declared stage is owned by exactly one group.
    grouped = [stage_id for group in flow.groups for stage_id in group.stage_ids]
    assert sorted(grouped) == sorted(FUTURE_STAGE_IDS)
    # The original v1 identity was the six display names; they stay frozen.
    assert flow.legacy_stage_ids == (
        "Sample input tail", "Prepare training data", "Train selected method",
        "Prepare selected method", "Generate future scenario", "Save forecast",
    )
    display = {stage.stage_id: stage.spec.name for stage in flow.stages}
    assert display["svd_load"] == "Prepare selected method"
    assert display["train_lora"] == "Train selected method"
    assert display["svd_generate"] == "Generate future scenario"


def test_future_flow_declares_a_distinct_recorded_set_per_legacy_version() -> None:
    """v1 recorded the six original rows; v2 recorded the full union."""
    flow = future_module.flow()

    assert flow.legacy_profiles[1] == (
        "source_tail", "training_pairs", "train_lora", "svd_load", "svd_generate", "save_forecast",
    )
    assert flow.legacy_profiles[2] == FUTURE_STAGE_IDS


def test_future_v2_retired_ids_map_by_alias_onto_their_current_rows() -> None:
    """A v2 Future trace's retired ids land on their renamed current rows."""
    # Labels that match no current display name, so only the alias can map them.
    retired = (
        ("train_method", "Old train label"),
        ("prepare_method", "Old prepare label"),
        ("generate_future", "Old generate label"),
    )
    events = [
        {"stage": name, "stage_id": old_id, "status": "complete", "elapsed_seconds": index}
        for index, (old_id, name) in enumerate(retired, start=1)
    ]
    headers = list(pages._updates("future", {
        "mode": "self_trained", "flow_version": 2, "events": events,
    })[4:4 + len(FUTURE_STAGE_IDS)])
    for current_id in ("train_lora", "svd_load", "svd_generate"):
        header = headers[FUTURE_STAGE_IDS.index(current_id)]
        assert "evow-stage-complete" in header["elem_classes"], current_id


def test_future_v2_rows_are_recorded_but_v1_only_marks_its_six() -> None:
    """A v2 trace must not mark a union row it recorded as "not recorded"."""
    v2_headers = list(pages._updates("future", {
        "mode": "self_trained", "flow_version": 2, "events": [],
    })[4:4 + len(FUTURE_STAGE_IDS)])
    for stage_id in ("training_cache", "train_artifact", "write_trace"):
        header = v2_headers[FUTURE_STAGE_IDS.index(stage_id)]
        assert "evow-stage-skipped" not in header["elem_classes"], stage_id

    # A v1 trace genuinely lacked those rows and must mark them not recorded.
    v1_headers = list(pages._updates("future", {
        "mode": "self_trained", "flow_version": 1, "events": [],
    })[4:4 + len(FUTURE_STAGE_IDS)])
    for stage_id in ("training_cache", "train_artifact", "write_trace"):
        header = v1_headers[FUTURE_STAGE_IDS.index(stage_id)]
        assert "evow-stage-skipped" in header["elem_classes"], stage_id


@pytest.mark.parametrize("mode", ["explicit", "implicit", "self_trained"])
def test_future_flow_marks_only_the_other_method_inapplicable(mode: str) -> None:
    flow = future_module.flow()
    applies = {stage.stage_id for stage in flow.stages if stage_applies(stage, mode)}
    if mode == "explicit":
        assert set(_IMPLICIT_ONLY) | set(_TRAINING) <= set(FUTURE_STAGE_IDS) - applies
        assert set(_EXPLICIT_ONLY) <= applies
    else:
        assert set(_EXPLICIT_ONLY) <= set(FUTURE_STAGE_IDS) - applies
        assert set(_IMPLICIT_ONLY) <= applies
        assert (set(_TRAINING) <= applies) is (mode == "self_trained")


def test_future_render_derives_skipped_for_inapplicable_modes() -> None:
    headers = _future_headers("explicit")
    for stage_id in _IMPLICIT_ONLY:
        header = headers[FUTURE_STAGE_IDS.index(stage_id)]
        assert "evow-stage-skipped" in header["elem_classes"], stage_id
    # The explicit producer rows are applicable, so they stay waiting, not skipped.
    assert "Waiting" in headers[FUTURE_STAGE_IDS.index("motion_extrapolate")]["label"]

    headers = _future_headers("implicit")
    for stage_id in _EXPLICIT_ONLY:
        header = headers[FUTURE_STAGE_IDS.index(stage_id)]
        assert "evow-stage-skipped" in header["elem_classes"], stage_id
    for stage_id in _TRAINING:
        header = headers[FUTURE_STAGE_IDS.index(stage_id)]
        assert "evow-stage-skipped" in header["elem_classes"], stage_id


def test_explicit_rows_have_producers_or_a_modes_derived_skip(monkeypatch) -> None:
    """Every explicit-applicable row is produced or explicitly skipped, never faked.

    Phase 3B-i gave ``motion_extrapolate`` and ``splat_render_export`` real
    producers, so a complete explicit lifecycle now terminates both rather than
    leaving them waiting. With no events at all, an explicit-applicable row still
    has a producer and must wait rather than be marked skipped.
    """
    produced = [
        {"stage": "Infer temporal depth", "stage_id": "depth_infer", "status": "complete", "detail": "", "elapsed_seconds": 1},
        {"stage": "Fit dynamic Gaussians", "stage_id": "gaussian_fit", "status": "complete", "detail": "", "elapsed_seconds": 2},
        {"stage": "Extrapolate scene motion", "stage_id": "motion_extrapolate", "status": "complete", "detail": "", "elapsed_seconds": 3},
        {"stage": "Render and export splats", "stage_id": "splat_render_export", "status": "complete", "detail": "", "elapsed_seconds": 4},
        {"stage": "Encode forecast video", "stage_id": "encode_video", "status": "complete", "detail": "", "elapsed_seconds": 5},
        {"stage": "Validate forecast artifacts", "stage_id": "validate_artifacts", "status": "complete", "detail": "", "elapsed_seconds": 6},
    ]
    headers = list(pages._updates("future", {"mode": "explicit", "flow_version": CURRENT_FLOW_VERSION, "events": produced})[4:19])
    for stage_id in _EXPLICIT_ONLY:
        header = headers[FUTURE_STAGE_IDS.index(stage_id)]
        assert "evow-stage-complete" in header["elem_classes"], stage_id

    # A dormant explicit layout leaves every explicit-applicable row waiting (it
    # has a producer); only the other method's rows are skipped.
    dormant = _future_headers("explicit")
    for stage_id in _EXPLICIT_ONLY:
        header = dormant[FUTURE_STAGE_IDS.index(stage_id)]
        assert "Waiting" in header["label"], stage_id
    for stage_id in _IMPLICIT_ONLY:
        header = dormant[FUTURE_STAGE_IDS.index(stage_id)]
        assert "evow-stage-skipped" in header["elem_classes"], stage_id


def test_every_declared_row_has_a_producer_or_a_modes_derived_skip() -> None:
    """Each declared row is either emitted by a producer or skipped by its modes.

    This is the forward guard for the union layout: a newly declared row must not
    sit permanently at Waiting. Explicit rows are exercised with a real lifecycle
    in ``test_explicit_rows_have_producers_or_a_modes_derived_skip``; here we pin
    the weaker structural invariant that every row has *some* producer on at
    least one selectable method or is scoped to a method (and so is skipped when
    that method is not chosen).
    """
    flow = future_module.flow()
    method_modes = ("explicit", "implicit", "self_trained")
    # Rows the adapters actually emit on their own path. ``encode_video`` and
    # ``save_forecast`` are shared tails produced by both ``future.run`` and the
    # explicit reconstruction, and ``write_trace`` is produced by the page.
    module_produced = (
        set(_EXPLICIT_ONLY) | set(_IMPLICIT_ONLY) | set(_TRAINING)
        | {"source_tail", "encode_video", "write_trace", "save_forecast"}
    )
    for stage_id in FUTURE_STAGE_IDS:
        scoped = not all(
            stage_applies(next(stage for stage in flow.stages if stage.stage_id == stage_id), mode)
            for mode in method_modes
        )
        assert stage_id in module_produced or scoped, (
            f"{stage_id} has no producer and belongs to every method, so it can only wait"
        )


def test_implicit_run_emits_svd_and_encoding_terminals(monkeypatch, tmp_path) -> None:
    events = _implicit_events(monkeypatch, tmp_path)
    statuses = _status_by_id(events)

    assert statuses["svd_load"] == ["running", "complete"]
    assert statuses["svd_generate"][0] == "running"
    assert statuses["svd_generate"][-1] == "complete"
    assert statuses["encode_video"] == ["running", "complete"]
    assert statuses["save_forecast"] == ["running", "complete"]
    _assert_single_terminal_lifecycle(events)

    load = next(event for event in events if event.stage_id == "svd_load" and event.status == "complete")
    assert load.metrics == {"checkpoint": "/models/svd", "adapter": False, "offload": True}
    chunk = next(event for event in events if event.stage_id == "svd_generate" and "chunk" in event.metrics)
    assert chunk.metrics == {"chunk": 1, "total_chunks": 2, "generated_frames": 1, "total_frames": 3}
    encode = next(event for event in events if event.stage_id == "encode_video" and event.status == "complete")
    assert encode.metrics == {"frames": 3, "fps": 1, "path": "forecast.mp4"}


def test_self_trained_cache_hit_completes_every_training_row(monkeypatch, tmp_path) -> None:
    from GREENFIELD.features import future_training

    cached = {"complete": True, "mode": "implicit", "model_artifact": "lora", "settings_key": "k"}
    monkeypatch.setattr(future_training, "training_dir", lambda *_: tmp_path / "training")
    monkeypatch.setattr(future_training, "training_manifest", lambda *_: cached)
    monkeypatch.setattr(future_training, "artifact_file", lambda *_: tmp_path / "training" / "lora")
    events = list(future_training.train("source.mp4", "implicit", SimpleNamespace()))

    assert _status_by_id(events) == {
        "training_cache": ["complete"],
        "training_pairs": ["skipped"],
        "train_lora": ["complete"],
        "train_artifact": ["complete"],
    }
    _assert_single_terminal_lifecycle(events)
    # A cache hit is terminal for every training row with no running tick.
    headers = list(pages._updates("future", {
        "mode": "self_trained", "flow_version": CURRENT_FLOW_VERSION,
        "events": [
            {"stage": event.stage, "stage_id": event.stage_id, "status": event.status, "detail": event.detail, "elapsed_seconds": 1}
            for event in events
        ],
    })[4:19])
    for stage_id in _TRAINING:
        header = headers[FUTURE_STAGE_IDS.index(stage_id)]
        assert "Waiting" not in header["label"], stage_id


# Distinct worker details make it unambiguous whether a worker terminal was
# downgraded to a parent tick instead of being re-emitted as a parent terminal.
_DEPTH_SIDECAR = (
    ("load_depth_model", "running", "Loading depth model.", {}),
    ("load_depth_model", "complete", "DEPTH MODEL DONE", {"device": "cuda"}),
    ("infer_temporal_depth", "running", "Estimating depth.", {"frames": 1}),
    ("infer_temporal_depth", "complete", "DEPTH INFER DONE", {"frames": 1}),
    ("write_depth_prior", "running", "Writing depth prior.", {}),
    ("write_depth_prior", "complete", "DEPTH PRIOR DONE", {"path": "depths.npz"}),
)
_GAUSSIAN_SIDECAR = (
    ("load_gaussian_inputs", "running", "Loading frames and depth prior.", {"frames": 1}),
    ("load_gaussian_inputs", "complete", "GAUSSIAN INPUTS DONE", {"frames": 1}),
    ("initialize_gaussians", "running", "Seeding Gaussian primitives.", {}),
    ("initialize_gaussians", "complete", "GAUSSIAN INIT DONE", {"gaussians": 9}),
    ("fit_4d_scene", "running", "Optimizing the scene.", {"total": 5}),
    ("fit_4d_scene", "complete", "GAUSSIAN FIT DONE", {"steps": 5, "gaussians": 9}),
    ("motion_extrapolate", "running", "Fitting a motion prior.", {"fit_frames": 3, "forecast_frames": 2}),
    ("motion_extrapolate", "complete", "MOTION DONE", {"fit_frames": 3, "forecast_frames": 2}),
    ("render_export_splats", "running", "Rendering observed-to-future.", {"total_frames": 3}),
    ("render_export_splats", "complete", "RENDER DONE", {"observed_frames": 1, "forecast_frames": 2}),
    ("encode_rendered_video", "running", "Encoding videos.", {}),
    ("encode_rendered_video", "complete", "ENCODE DONE", {"observed_frames": 1, "forecast_frames": 2, "fps": 6}),
    ("persist_scene_artifacts", "running", "Persisting the scene.", {}),
    ("persist_scene_artifacts", "complete", "PERSIST DONE", {"gaussians": 9}),
)


def _explicit_settings(**overrides) -> SimpleNamespace:
    values = dict(
        history_fps=12, observed_splat_timeline_fps=1, output_frames=2, output_fps=6,
        forecast_splat_timeline_fps=1, output_seconds=1, motion_fit_frames=3,
        render_yaw=0.0, render_shift=0.0, render_fov=60.0, source_fov_degrees=70.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _prepare_explicit_run(monkeypatch, tmp_path, frame_count, settings):
    """Create the depth checkout and every artifact ``run_explicit`` verifies."""
    depth_repo = tmp_path / "depth_repo"
    (depth_repo / "video_depth_anything").mkdir(parents=True)
    (depth_repo / "video_depth_anything" / "video_depth.py").write_text("")
    replay = SimpleNamespace(depth_checkpoint=tmp_path / "depth.pth", depth_repo=depth_repo)
    monkeypatch.setattr(future_explicit, "load_settings", lambda: replay)
    monkeypatch.setattr(future_explicit, "require_checkpoint", lambda *_: None)
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    (tmp_path / "depths.npz").write_bytes(b"depth")
    output = tmp_path / "explicit"
    output.mkdir()
    (output / "forecast.mp4").write_bytes(b"mp4")
    observed = timeline_keyframe_indices(frame_count, settings.history_fps, settings.observed_splat_timeline_fps)
    forecast = timeline_keyframe_indices(settings.output_frames, settings.output_fps, settings.forecast_splat_timeline_fps)
    for index in observed:
        (output / f"splat_{index:03d}.splat").touch()
    for index in forecast:
        (output / f"forecast_splat_{index:03d}.splat").touch()
    return observed, forecast


def _install_fake_workers(monkeypatch, *, depth_events=(), gaussian_events=()) -> None:
    """Replace ``subprocess.Popen`` with a worker that writes an event sidecar."""

    class FinishedProcess:
        returncode = 0

        def poll(self):
            return 0

    def fake_popen(command, stdout=None, stderr=None):
        events_path = Path(command[command.index("--events") + 1])
        module = command[command.index("-m") + 1]
        writer = WorkerEventWriter(events_path)
        for stage_id, status, detail, metrics in (depth_events if module.endswith("depth_worker") else gaussian_events):
            writer.emit(stage_id, status, detail, metrics=metrics)
        return FinishedProcess()

    monkeypatch.setattr(future_explicit.subprocess, "Popen", fake_popen)


def test_explicit_run_emits_depth_encoding_and_validation_terminals(monkeypatch, tmp_path) -> None:
    frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
    settings = _explicit_settings()
    observed, forecast = _prepare_explicit_run(monkeypatch, tmp_path, len(frames), settings)
    # No worker sidecar at all: every Gaussian row must fall back to one honest
    # running/complete boundary rather than being fabricated or left running.
    _install_fake_workers(monkeypatch)

    events = list(future_explicit.run_explicit(frames, tmp_path, settings))
    statuses = _status_by_id(events)
    assert statuses["depth_infer"][0] == "running" and statuses["depth_infer"][-1] == "complete"
    assert statuses["gaussian_fit"][-1] == "complete"
    assert statuses["encode_video"] == ["running", "complete"]
    assert statuses["validate_artifacts"] == ["running", "complete"]
    _assert_single_terminal_lifecycle(events)
    motion = [event for event in events if event.stage_id == "motion_extrapolate"]
    assert motion[-1].status == "complete" and motion[-1].metrics == {}
    validation = [event for event in events if event.stage_id == "validate_artifacts"][-1]
    assert validation.metrics == {"splats": len(observed) + len(forecast), "forecast": "forecast.mp4"}


def test_explicit_run_relays_depth_substeps_under_depth_infer(monkeypatch, tmp_path) -> None:
    frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
    settings = _explicit_settings()
    _prepare_explicit_run(monkeypatch, tmp_path, len(frames), settings)
    _install_fake_workers(monkeypatch, depth_events=_DEPTH_SIDECAR)

    events = list(future_explicit.run_explicit(frames, tmp_path, settings))
    depth = [event for event in events if event.stage_id == "depth_infer"]
    for detail in ("Loading depth model.", "Estimating depth.", "Writing depth prior."):
        assert any(event.status == "running" and event.detail == detail for event in depth), detail
    # The worker terminal became a tick, so the row has exactly one terminal.
    assert any(event.status == "running" and event.detail == "DEPTH MODEL DONE" for event in depth)
    assert not any(event.status == "complete" and event.detail == "DEPTH MODEL DONE" for event in depth)
    assert [event.status for event in depth].count("complete") == 1
    assert depth[-1].status == "complete"


def test_explicit_run_relays_gaussian_substeps_and_terminates_the_row_once(monkeypatch, tmp_path) -> None:
    frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
    settings = _explicit_settings()
    _prepare_explicit_run(monkeypatch, tmp_path, len(frames), settings)
    _install_fake_workers(monkeypatch, gaussian_events=_GAUSSIAN_SIDECAR)

    events = list(future_explicit.run_explicit(frames, tmp_path, settings))
    gaussian = [event for event in events if event.stage_id == "gaussian_fit"]
    for detail in ("Loading frames and depth prior.", "Seeding Gaussian primitives.", "Optimizing the scene."):
        assert any(event.status == "running" and event.detail == detail for event in gaussian), detail
    # A worker terminal is downgraded to a tick; only the parent emits a terminal.
    assert any(event.status == "running" and event.detail == "GAUSSIAN FIT DONE" for event in gaussian)
    assert not any(event.status == "complete" and event.detail == "GAUSSIAN FIT DONE" for event in gaussian)
    assert [event.status for event in gaussian].count("complete") == 1
    assert gaussian[-1].metrics["steps"] == 5


def test_explicit_run_emits_motion_extrapolate_with_fit_and_forecast_metrics(monkeypatch, tmp_path) -> None:
    frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
    settings = _explicit_settings()
    _prepare_explicit_run(monkeypatch, tmp_path, len(frames), settings)
    _install_fake_workers(monkeypatch, gaussian_events=_GAUSSIAN_SIDECAR)

    events = list(future_explicit.run_explicit(frames, tmp_path, settings))
    motion = [event for event in events if event.stage_id == "motion_extrapolate"]
    assert motion[0].status == "running"
    assert motion[-1].status == "complete"
    assert [event.status for event in motion].count("complete") == 1
    assert motion[-1].metrics["fit_frames"] == 3
    assert motion[-1].metrics["forecast_frames"] == 2
    assert any(event.status == "running" and event.detail == "MOTION DONE" for event in motion)


def test_explicit_run_terminates_each_row_once_in_declared_order(monkeypatch, tmp_path) -> None:
    frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
    settings = _explicit_settings()
    _prepare_explicit_run(monkeypatch, tmp_path, len(frames), settings)
    _install_fake_workers(monkeypatch, depth_events=_DEPTH_SIDECAR, gaussian_events=_GAUSSIAN_SIDECAR)

    events = list(future_explicit.run_explicit(frames, tmp_path, settings))
    first_seen: list[str] = []
    for event in events:
        if event.stage_id not in first_seen:
            first_seen.append(event.stage_id)
    assert first_seen == [
        "depth_infer", "gaussian_fit", "motion_extrapolate",
        "splat_render_export", "encode_video", "validate_artifacts",
    ]
    for stage_id in first_seen:
        assert [event.status for event in events if event.stage_id == stage_id].count("complete") == 1, stage_id
    gaussian_terminal = next(
        index for index, event in enumerate(events)
        if event.stage_id == "gaussian_fit" and event.status == "complete"
    )
    motion_start = next(
        index for index, event in enumerate(events)
        if event.stage_id == "motion_extrapolate" and event.status == "running"
    )
    assert gaussian_terminal < motion_start
    _assert_single_terminal_lifecycle(events)


def test_relay_worker_tolerates_missing_and_partial_sidecars(monkeypatch, tmp_path) -> None:
    class FinishedProcess:
        returncode = 0

        def poll(self):
            return 0

    monkeypatch.setattr(future_explicit.subprocess, "Popen", lambda *args, **kwargs: FinishedProcess())

    def drive(events_path: Path):
        stream = future_explicit._relay_worker(
            ["worker"], tmp_path / "worker.log", events_path, future_explicit._GAUSSIAN_EVENT_ROWS,
            startup_stage="load_gaussian_inputs", startup_detail="Starting the 4D Gaussian worker.",
        )
        updates = []
        while True:
            try:
                updates.append(next(stream))
            except StopIteration as finished:
                return updates, finished.value

    updates, terminated = drive(tmp_path / "never-written.json")
    assert terminated == set()
    assert [(event.stage_id, event.status) for event in updates] == [("gaussian_fit", "running")]

    (tmp_path / "partial.json").write_text("{ this is not valid json")
    updates, terminated = drive(tmp_path / "partial.json")
    assert terminated == set()
    assert [event.stage_id for event in updates] == ["gaussian_fit"]


def test_future_stream_emits_write_trace_terminal(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(pages, "ASSETS", tmp_path / "ASSETS")
    monkeypatch.setattr(pages, "read_recent_window", lambda *_: ([np.zeros((2, 2, 3), dtype=np.uint8)], 12))

    def fake_run(frames, fps, request, artifacts, settings, source_name):
        yield StageEvent("Save forecast", "complete", "saved", stage_id="save_forecast")
        yield FeatureResult(primary=artifacts.file("forecast.mp4"))

    monkeypatch.setattr(pages.future, "run", fake_run)
    settings = SimpleNamespace(history_seconds=1, history_fps=12, max_side=64)
    stream = pages._execute_future_stream(str(source), "implicit", FutureRequest("implicit", 0), settings)
    traces = []
    while True:
        try:
            _result, trace = next(stream)
        except StopIteration:
            break
        traces.append(trace)
    statuses = [event["status"] for event in traces[-1]["events"] if event["stage_id"] == "write_trace"]
    assert statuses == ["running", "complete"]


def test_future_stream_emits_terminal_once_when_a_trailing_step_fails(monkeypatch, tmp_path) -> None:
    """A failure after the active stage completed emits one terminal, not a second."""
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(pages, "ASSETS", tmp_path / "ASSETS")
    monkeypatch.setattr(pages, "read_recent_window", lambda *_: ([np.zeros((2, 2, 3), dtype=np.uint8)], 12))

    def completed_train_then_fail(*_args):
        yield StageEvent("Train selected method", "running", "fitting the adapter", stage_id="train_lora")
        yield StageEvent("Train selected method", "complete", "fitted the adapter", stage_id="train_lora")
        # A later step raises, e.g. a missing adapter artifact.
        raise RuntimeError("missing adapter artifact")

    monkeypatch.setattr(pages.future, "run", completed_train_then_fail)
    settings = SimpleNamespace(history_seconds=1, history_fps=12, max_side=64)
    stream = pages._execute_future_stream(str(source), "self_trained", FutureRequest("self_trained", 0), settings)
    for _ in range(4):  # source running, source complete, train running, train complete
        next(stream)
    _result, failed_trace = next(stream)
    terminals = [
        event for event in failed_trace["events"]
        if event["stage_id"] == "train_lora" and event["status"] in {"complete", "error"}
    ]
    assert len(terminals) == 1 and terminals[0]["status"] == "complete"
    assert not any(
        event["stage_id"] == "train_lora" and event["status"] == "error"
        for event in failed_trace["events"]
    )
    # The generic run-failure terminal still records the actual error.
    assert failed_trace["events"][-1]["stage"] == "Run failed"
    with pytest.raises(gr.Error):
        next(stream)


def test_source_adapter_stream_repaints_declared_stage_on_early_failure(monkeypatch, tmp_path) -> None:
    """A failure before any adapter event must still fail the declared first row."""
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(pages, "ASSETS", tmp_path / "ASSETS")

    def failing(_source, _request, _artifacts):
        raise RuntimeError("immediate failure")
        yield  # pragma: no cover

    stream = pages._execute_source_adapter_stream(
        "selection", "implicit", str(source), SimpleNamespace(mode="implicit"), failing,
    )
    _result, trace = next(stream)
    error = next(event for event in trace["events"] if event["status"] == "error")
    assert error["stage"] == "Inspect source video"
    assert error["stage_id"] == "inspect_source"
    updates = pages._updates("selection", trace)
    assert any("evow-stage-failed" in (update.get("elem_classes") or []) for update in updates)
    with pytest.raises(gr.Error):
        next(stream)


def test_source_adapter_stream_resolves_an_empty_stage_id_by_name(monkeypatch, tmp_path) -> None:
    """A legacy name-only adapter event must resolve to the declared id, not the label."""
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(pages, "ASSETS", tmp_path / "ASSETS")

    def naming(_source, _request, _artifacts):
        yield StageEvent("Inspect source video", "running", "decoding")  # deliberately id-less
        raise RuntimeError("boom")

    stream = pages._execute_source_adapter_stream(
        "selection", "implicit", str(source), SimpleNamespace(mode="implicit"), naming,
    )
    _result, running_trace = next(stream)
    running = next(event for event in running_trace["events"] if event["status"] == "running")
    assert running["stage_id"] == "inspect_source"
    _result, terminal_trace = next(stream)
    error = next(event for event in terminal_trace["events"] if event["status"] == "error")
    assert error["stage_id"] == "inspect_source"
