"""Replay worker-side event protocol and command coverage."""

import json

from GREENFIELD.replay import explicit, implicit
from GREENFIELD.replay.overrides import ReplayOverrides
from GREENFIELD.replay.pose import ViewRequest
from GREENFIELD.replay.settings import load_settings
from GREENFIELD.replay.worker_events import WorkerEventWriter, read_events


def test_worker_event_sidecar_replays_every_terminal_event(tmp_path):
    sidecar = tmp_path / "events.json"
    writer = WorkerEventWriter(sidecar)
    writer.emit("fit_4d_scene", "running", "fit", metrics={"step": 1})
    writer.emit("fit_4d_scene", "complete", "done", metrics={"step": 2})
    events = read_events(sidecar, 0)
    assert [event["status"] for event in events] == ["running", "complete"]
    assert read_events(sidecar, 1)[0]["sequence"] == 2
    assert json.loads(sidecar.read_text())[-1]["stage_id"] == "fit_4d_scene"


def test_read_events_skips_malformed_records_without_raising(tmp_path):
    """A garbage record is dropped, never allowed to break the relay."""
    sidecar = tmp_path / "events.json"
    sidecar.write_text(json.dumps([
        {"sequence": 1, "stage_id": "fit_4d_scene", "status": "running", "detail": "ok", "metrics": {"step": 1}},
        {"sequence": "not-an-int", "stage_id": "fit_4d_scene", "status": "complete", "detail": "bad seq"},
        {"sequence": 3, "stage_id": "fit_4d_scene", "status": "teleported", "detail": "bad status"},
        {"sequence": 4, "stage_id": "fit_4d_scene", "status": "complete", "detail": "bad metrics", "metrics": [1, 2]},
        {"sequence": 5, "stage_id": 7, "status": "complete", "detail": "bad stage id"},
        {"sequence": 6, "stage_id": "fit_4d_scene", "status": "complete", "detail": 9},
        {"sequence": 7, "stage_id": "fit_4d_scene", "status": "complete", "detail": "ok"},
    ]))

    events = read_events(sidecar, 0)

    assert [event.get("sequence") for event in events] == [1, 7]


def test_read_events_returns_valid_records_before_a_truncated_tail(tmp_path):
    """A truncated file yields the complete records written before the cut."""
    sidecar = tmp_path / "events.json"
    sidecar.write_text(
        '[{"sequence": 1, "stage_id": "fit_4d_scene", "status": "running", "detail": "ok"},'
        ' {"sequence": 2, "stage_id": "fit_4d_scene", "status": "complete", "detail": "ok"},'
        ' {"sequence": 3, "stage_id": "fit_4d_scene", "sta'
    )

    events = read_events(sidecar, 0)

    assert [event["sequence"] for event in events] == [1, 2]


def test_read_events_tolerates_garbage_and_a_missing_file(tmp_path):
    """A non-JSON body and an absent sidecar both produce no events."""
    garbage = tmp_path / "garbage.json"
    garbage.write_text("not json at all")

    assert read_events(garbage, 0) == []
    assert read_events(tmp_path / "absent.json", 0) == []


def test_read_events_tolerates_a_non_utf8_sidecar_and_keeps_its_valid_prefix(tmp_path):
    """A corrupt byte must not raise or discard the records written before it."""
    valid_prefix = json.dumps([
        {"sequence": 1, "stage_id": "fit_4d_scene", "status": "running", "detail": "ok"},
    ])
    # A complete record followed by bytes that are not valid UTF-8.
    sidecar = tmp_path / "events.json"
    sidecar.write_bytes(valid_prefix.encode() + b"\xff\xfe")
    assert [event["sequence"] for event in read_events(sidecar, 0)] == [1]

    # A wholly corrupt body still produces no events instead of raising.
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_bytes(b"\xff\xfe\xff")
    assert read_events(corrupt, 0) == []


def test_read_events_honors_after_sequence_over_valid_records(tmp_path):
    sidecar = tmp_path / "events.json"
    sidecar.write_text(json.dumps([
        {"sequence": 1, "stage_id": "fit_4d_scene", "status": "running", "detail": "a"},
        {"sequence": 2, "stage_id": "fit_4d_scene", "status": "complete", "detail": "b"},
    ]))

    assert [event["sequence"] for event in read_events(sidecar, 1)] == [2]
    assert read_events(sidecar, 2) == []


def test_commands_carry_effective_settings_and_optional_event_paths(tmp_path):
    effective = ReplayOverrides(gaussian_steps=77, anyview_steps=11).apply(load_settings())
    event_path = tmp_path / "events.json"
    gaussian = explicit.gaussian_command(tmp_path, tmp_path / "out", tmp_path / "depths.npz", ViewRequest(0.0, 0.0, 60.0), 70.0, effective, event_path)
    depth = explicit.depth_command(tmp_path, tmp_path / "depths.npz", effective, event_path)
    anyview = implicit.anyview_command(tmp_path / "episode", tmp_path / "out", effective, event_path)
    for command in (gaussian, depth, anyview):
        assert "--settings-json" in command
        assert "--events" in command
        assert str(event_path) in command


def test_direct_commands_remain_valid_without_event_sidecars(tmp_path):
    settings = load_settings()
    command = explicit.gaussian_command(tmp_path, tmp_path / "out", tmp_path / "depths.npz", ViewRequest(0.0, 0.0, 60.0), 70.0, settings)
    assert "--events" not in command


def _relay_finished_worker(monkeypatch, tmp_path, records):
    """Relay one finished process over sidecar records of (stage_id, status, detail, metrics)."""
    sidecar = tmp_path / "events.json"
    writer = WorkerEventWriter(sidecar)
    for stage_id, status, detail, metrics in records:
        writer.emit(stage_id, status, detail, metrics=metrics)

    class FinishedProcess:
        returncode = 0

        def poll(self):
            return 0

    monkeypatch.setattr(explicit.subprocess, "Popen", lambda *_args, **_kwargs: FinishedProcess())
    return list(explicit._worker(
        ["worker"], tmp_path / "worker.log", sidecar,
        "fit_4d_scene", "Optimizing the persistent 4D Gaussian scene.",
    ))


def test_worker_streams_optimizer_metrics_and_step_detail(monkeypatch, tmp_path):
    """Optimizer records keep their loss, step, and Gaussian count in the relay."""
    updates = _relay_finished_worker(monkeypatch, tmp_path, [
        ("fit_4d_scene", "running", "Optimizing the persistent 4D Gaussian scene.",
         {"step": 12, "total": 260, "image_loss": 0.12345, "gaussians": 384}),
    ])

    assert updates[-1]["stage_id"] == "fit_4d_scene"
    assert updates[-1]["status"] == "running"
    assert updates[-1]["metrics"]["step"] == 12
    assert updates[-1]["metrics"]["image_loss"] == 0.12345
    assert updates[-1]["metrics"]["gaussians"] == 384
    assert "Optimizing the persistent 4D Gaussian scene." in updates[-1]["detail"]


def test_worker_relays_step_less_export_progress_without_optimizer_metrics(monkeypatch, tmp_path):
    """Step-less export records keep their own row, never rewritten as optimizer steps."""
    updates = _relay_finished_worker(monkeypatch, tmp_path, [
        ("render_export_splats", "running",
         "Rendering requested view and exporting observed-to-future splat keyframes.",
         {"rendered_frames": 1, "total_frames": 13, "gaussians": 384}),
    ])

    assert updates[-1]["stage_id"] == "render_export_splats"
    assert updates[-1]["metrics"]["rendered_frames"] == 1
    assert "step" not in updates[-1]["metrics"]
    assert "image_loss" not in updates[-1]["metrics"]
    assert not updates[-1]["detail"].startswith("Optimizing")


def test_coordinator_drains_events_published_before_worker_exit(monkeypatch, tmp_path):
    sidecar = tmp_path / "events.json"
    writer = WorkerEventWriter(sidecar)
    writer.emit("fit_4d_scene", "running", "fit")
    writer.emit("fit_4d_scene", "complete", "done")

    class FinishedProcess:
        returncode = 0

        def poll(self):
            return 0

    monkeypatch.setattr(explicit.subprocess, "Popen", lambda *_args, **_kwargs: FinishedProcess())
    updates = list(explicit._worker(["worker"], tmp_path / "worker.log", sidecar, "fit_4d_scene", "starting"))
    assert [update["status"] for update in updates] == ["running", "running", "complete"]
    assert updates[-1]["stage_id"] == "fit_4d_scene"
