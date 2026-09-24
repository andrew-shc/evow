"""Focused unit coverage for the cross-feature run record contract."""

import json
from pathlib import Path

from GREENFIELD.app_core.contracts import FeatureResult, RunArtifacts, StageEvent
from GREENFIELD.app_core.trace import FeatureTrace, load_trace
from GREENFIELD.app_core.ui import stage_html, stage_label
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
    moments = iter((10.0, 11.25, 13.5, 17.0))
    monkeypatch.setattr("GREENFIELD.app_core.trace.time.monotonic", lambda: next(moments))
    trace = FeatureTrace(tmp_path, "future", "explicit", {})
    first = trace.add(StageEvent("Sample", "running", "started"))
    second = trace.add(StageEvent("Sample", "complete", "done"))
    next_stage = trace.add(StageEvent("Generate", "running", "started"))

    assert first["stage_elapsed_seconds"] == 0
    assert second["stage_elapsed_seconds"] == 2.25
    assert next_stage["stage_elapsed_seconds"] == 0
    spec = StageSpec("Generate", "purpose", "", "", "", "", "", "")
    assert stage_label(0, spec, next_stage)[0].endswith("0.00s")
    html = stage_html(spec, next_stage, stage_started_at=1.25)
    assert '&quot;stage_elapsed_seconds&quot;: 0.0' in html
    assert "run_elapsed_seconds" not in html


def test_ui_derives_stage_time_for_legacy_events() -> None:
    spec = StageSpec("Generate", "purpose", "", "", "", "", "", "")
    legacy = {"stage": "Generate", "status": "complete", "detail": "done", "elapsed_seconds": 7.5}
    assert stage_label(0, spec, legacy, stage_started_at=2.0)[0].endswith("5.50s")


def test_trace_normalizes_a_relative_root_for_preview_serialization(monkeypatch, tmp_path: Path) -> None:
    """Dashboard callers may supply a relative ASSETS root without breaking saved previews."""
    monkeypatch.chdir(tmp_path)
    trace = FeatureTrace(Path("relative-assets"), "selection", "implicit", {})
    preview = trace.artifacts.file("previews/frame.png")
    preview.touch()
    trace.add(StageEvent("Save", "running", "saved", preview=preview))

    assert trace.run_dir.is_absolute()
