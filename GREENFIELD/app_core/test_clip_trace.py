"""Clip artifact persistence coverage for Text Query saved runs."""

import json
from pathlib import Path

from GREENFIELD.app_core.contracts import ClipArtifact, FeatureResult
from GREENFIELD.app_core.trace import FeatureTrace, load_trace


def test_trace_serializes_clip_artifacts_relative_to_its_run(tmp_path: Path) -> None:
    """Raw and highlighted clips remain safe, portable trace references."""
    trace = FeatureTrace(tmp_path, "selection", "implicit", {"query": "tree"})
    raw = trace.artifacts.file("clips/00_source.mp4")
    highlighted = trace.artifacts.file("clips/00_highlighted.mp4")
    raw.touch()
    highlighted.touch()
    trace.finish(FeatureResult(clips=[ClipArtifact(raw, highlighted, 1.0, 5.0, 0.8, "implicit")]))

    saved = load_trace(trace.run_dir)
    clip = saved["result"]["clips"][0]
    assert clip["source"] == "clips/00_source.mp4"
    assert clip["highlighted"] == "clips/00_highlighted.mp4"
    assert clip["projected_highlighted"] is None
    assert clip["projected_low_specificity"] is False


def test_old_trace_result_gets_an_empty_clip_list(tmp_path: Path) -> None:
    """Saved runs from before Text Query's typed clips continue to load."""
    run = tmp_path / "legacy"
    run.mkdir()
    (run / "trace.json").write_text(json.dumps({"result": {"primary": "old.mp4"}}))
    assert load_trace(run)["result"]["clips"] == []



def test_trace_rejects_clip_paths_outside_its_run(tmp_path: Path) -> None:
    """A feature cannot serialize a path that would escape saved-run ownership."""
    trace = FeatureTrace(tmp_path, "selection", "implicit", {})
    outside = tmp_path / "outside.mp4"
    outside.touch()
    inside = trace.artifacts.file("clips/inside.mp4")
    inside.touch()
    try:
        trace.finish(FeatureResult(clips=[ClipArtifact(outside, inside, 0, 4, 1.0, "implicit")]))
    except ValueError as error:
        assert "inside the current run" in str(error)
    else:
        raise AssertionError("outside clip artifact must be rejected")
