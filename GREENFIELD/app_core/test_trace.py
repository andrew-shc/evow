"""Focused unit coverage for the cross-feature run record contract."""

import json
from pathlib import Path

from GREENFIELD.app_core.contracts import FeatureResult, RunArtifacts, StageEvent
from GREENFIELD.app_core.trace import FeatureTrace, load_trace
from GREENFIELD.app_core.ui import run_total_update, stage_html, stage_label
from GREENFIELD.app_core.contracts import StageSpec


def test_artifacts_cannot_escape_its_run_directory(tmp_path: Path) -> None:
    artifacts = RunArtifacts(tmp_path / "run")
    assert artifacts.file("preview/output.mp4") == tmp_path / "run" / "preview" / "output.mp4"
    try:
        artifacts.file("../outside.mp4")
    except ValueError:
        pass
    else:
        raise AssertionError("artifact path traversal must be rejected")


def test_trace_is_versioned_atomic_and_loads_legacy_shape(tmp_path: Path) -> None:
    trace = FeatureTrace(tmp_path, "future", "implicit", {"seed": 7})
    trace.add(StageEvent("Read", "complete", "done"))
    output = trace.artifacts.file("forecast.mp4")
    output.touch()
    trace.finish(FeatureResult(primary=output, metadata={"generated": True}))

    record = load_trace(trace.run_dir)
    assert record["manifest_version"] == 1
    assert record["result"]["primary"] == "forecast.mp4"
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "trace.json").write_text(json.dumps({"run_id": "legacy", "mode": "explicit", "events": []}))
    assert load_trace(legacy)["feature"] == "replay"


def test_trace_persists_stage_local_time_and_ui_hides_run_total(monkeypatch, tmp_path: Path) -> None:
    moments = iter((10.0, 11.2344, 13.5794, 17.0))
    monkeypatch.setattr("GREENFIELD.app_core.trace.time.monotonic", lambda: next(moments))
    trace = FeatureTrace(tmp_path, "future", "explicit", {})
    first = trace.add(StageEvent("Sample", "running", "started"))
    second = trace.add(StageEvent("Sample", "complete", "done"))
    next_stage = trace.add(StageEvent("Generate", "running", "started"))

    assert first["elapsed_seconds"] == 1.234
    assert first["stage_elapsed_seconds"] == 0.0
    assert second["elapsed_seconds"] == 3.579
    assert second["stage_elapsed_seconds"] == 2.345
    assert next_stage["stage_elapsed_seconds"] == 0.0
    spec = StageSpec("Generate", "purpose", "", "", "", "", "", "")
    assert stage_label(0, spec, next_stage)[0].endswith("0.000s")
    assert run_total_update({"events": [next_stage]})["value"] == "**Run total:** 7.000s"
    html = stage_html(spec, next_stage, stage_started_at=1.25)
    assert "Stage time</dt><dd>0.000s" in html
    assert "stage_elapsed_seconds" not in html


def test_ui_derives_stage_time_for_legacy_events() -> None:
    spec = StageSpec("Generate", "purpose", "", "", "", "", "", "")
    legacy = {"stage": "Generate", "status": "complete", "detail": "done", "elapsed_seconds": 7.5}
    assert stage_label(0, spec, legacy, stage_started_at=2.0)[0].endswith("5.500s")


def test_skipped_stage_renders_no_fabricated_duration() -> None:
    """A skipped stage shows a dash in the card body and its nested tick, not a made-up time."""
    spec = StageSpec("Generate", "purpose", "", "", "", "", "", "")
    skipped = {"stage": "Generate", "status": "skipped", "stage_elapsed_seconds": 2.25, "elapsed_seconds": 2.25}
    html = stage_html(spec, skipped, stage_started_at=0.0, events=[skipped])
    assert "Stage time</dt><dd>—" in html
    assert "Stage time</dt><dd>2.250s" not in html
    assert "Tick 1 · skipped · —" in html
    assert "Tick 1 · skipped · 2.250s" not in html



def test_trace_normalizes_a_relative_root_for_preview_serialization(monkeypatch, tmp_path: Path) -> None:
    """Dashboard callers may supply a relative ASSETS root without breaking saved previews."""
    monkeypatch.chdir(tmp_path)
    trace = FeatureTrace(Path("relative-assets"), "selection", "implicit", {})
    preview = trace.artifacts.file("previews/frame.png")
    preview.touch()
    trace.add(StageEvent("Save", "running", "saved", preview=preview))

    assert trace.run_dir.is_absolute()


def test_trace_persists_flow_identity_and_per_event_stage_ids(tmp_path: Path) -> None:
    """A saved run carries its ordered profile and each event's stable identity."""
    trace = FeatureTrace(
        tmp_path, "future", "self_trained", {}, flow_version=2, profile_id="future.self_trained.v2"
    )
    trace.add(StageEvent("Sample input tail", "complete", "done", stage_id="source_tail"))
    trace.add(StageEvent("Prepare training data", "complete", "done"))

    record = load_trace(trace.run_dir)
    assert record["flow_version"] == 2
    assert record["profile_id"] == "future.self_trained.v2"
    assert record["events"][0]["stage_id"] == "source_tail"
    # A legacy adapter leaves ``stage_id`` empty; the name becomes the identity.
    assert record["events"][1]["stage_id"] == "Prepare training data"


def test_load_trace_defaults_missing_flow_identity_to_legacy(tmp_path: Path) -> None:
    """Pre-versioning records resolve as the legacy name-only profile."""
    run = tmp_path / "legacy"
    run.mkdir()
    (run / "trace.json").write_text(json.dumps({"run_id": "legacy", "events": [{"stage": "Read"}]}))

    record = load_trace(run)
    assert record["flow_version"] == 1
    assert record["profile_id"] == ""
    # An event saved before stage ids still loads; callers normalize it on read.
    assert "stage_id" not in record["events"][0]


def test_resume_preserves_flow_identity(tmp_path: Path) -> None:
    trace = FeatureTrace(tmp_path, "editing", "explicit", {}, flow_version=2, profile_id="editing.explicit.v2")
    resumed = FeatureTrace.resume(tmp_path, "editing", trace.run_dir.name)
    assert resumed.record["flow_version"] == 2
    assert resumed.record["profile_id"] == "editing.explicit.v2"


def test_resume_defaults_missing_flow_identity_to_legacy(tmp_path: Path) -> None:
    run = tmp_path / "editing" / "runs" / "old"
    run.mkdir(parents=True)
    (run / "trace.json").write_text(json.dumps({"feature": "editing", "result": None, "events": []}))

    resumed = FeatureTrace.resume(tmp_path, "editing", "old")
    assert resumed.record["flow_version"] == 1
    assert resumed.record["profile_id"] == ""
