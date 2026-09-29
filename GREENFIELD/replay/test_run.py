"""Failure-path coverage for the atomic Replay run generator."""

import pytest

from GREENFIELD.replay import run as run_module
from GREENFIELD.replay.overrides import ReplayOverrides
from GREENFIELD.replay.settings import load_settings


def _failing_stream(monkeypatch, tmp_path, overrides=None):
    monkeypatch.setattr(run_module, "_run_directory", lambda: tmp_path)
    monkeypatch.setattr(run_module, "sample_episode", lambda *_args: (_ for _ in ()).throw(RuntimeError("decode exploded")))
    return run_module.execute_run("clip.mp4", "implicit", 0.0, 13, 70.0, 0.0, 0.0, 60.0, overrides=overrides)


def test_input_failure_marks_the_exact_active_atomic_stage(monkeypatch, tmp_path):
    stream = _failing_stream(monkeypatch, tmp_path)
    sampling = next(stream)
    assert sampling["message"] == "Sampling input episode."
    failed = next(stream)
    events = failed["trace"]["events"]
    assert failed["message"].startswith("Run failed:")
    assert failed["trace"]["workflow_version"] == 2
    assert events[-1]["stage_id"] == "sample_episode"
    assert events[-1]["status"] == "error"
    assert not any(event["stage"] == "Run failed" for event in events)
    with pytest.raises(StopIteration):
        next(stream)


def test_invalid_overrides_fail_before_a_run_directory_is_created(monkeypatch, tmp_path):
    monkeypatch.setattr(run_module, "_run_directory", lambda: tmp_path)
    stream = run_module.execute_run("clip.mp4", "implicit", 0.0, 13, 70.0, 0.0, 0.0, 60.0, overrides=ReplayOverrides(gaussian_steps=0))
    with pytest.raises(ValueError):
        next(stream)


def test_effective_settings_are_saved_for_default_and_override_runs(monkeypatch, tmp_path):
    override = _failing_stream(monkeypatch, tmp_path, ReplayOverrides(gaussian_steps=123))
    next(override)
    failed = next(override)
    assert failed["trace"]["request"]["effective_settings"]["gaussian_steps"] == 123

    second = tmp_path / "second"
    second.mkdir()
    monkeypatch.setattr(run_module, "_run_directory", lambda: second)
    default = _failing_stream(monkeypatch, second)
    next(default)
    failed = next(default)
    tracked = load_settings()
    assert failed["trace"]["request"]["effective_settings"]["fps"] == tracked.fps


def test_snapshot_accepts_legacy_preview_events_without_stage_ids(tmp_path):
    """Opening a V1 run must not fail merely because its preview lacks stage_id."""
    preview = tmp_path / "source.png"
    preview.touch()

    class LegacyTrace:
        record = {"events": [{"stage": "Persist source", "preview": "source.png"}]}

    snapshot = run_module._snapshot(tmp_path, LegacyTrace(), "Saved run loaded.")
    assert snapshot["previews"] == [(str(preview), "Persist source")]
