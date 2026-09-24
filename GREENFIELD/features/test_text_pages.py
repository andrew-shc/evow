"""Saved Text Query handoff and explicit Text Manipulation viewer coverage."""

import json
from pathlib import Path

import pytest

from GREENFIELD.features import pages


def _write_selection_run(assets: Path, run_id: str = "query-a") -> tuple[Path, Path, Path]:
    """Create a minimal safe saved query run with distinct raw/highlighted clips."""
    run_dir = assets / "selection" / "runs" / run_id
    clips = run_dir / "clips"
    clips.mkdir(parents=True)
    source = run_dir / "source.mp4"
    raw = clips / "00_source.mp4"
    highlighted = clips / "00_highlighted.mp4"
    for path in (source, raw, highlighted):
        path.write_bytes(b"video")
    (run_dir / "trace.json").write_text(json.dumps({
        "feature": "selection", "mode": "implicit", "source_video": "source.mp4", "events": [],
        "result": {"clips": [{
            "source": "clips/00_source.mp4", "highlighted": "clips/00_highlighted.mp4",
            "start_seconds": 8.0, "end_seconds": 12.0, "score": 0.75, "method": "implicit",
        }]},
    }))
    return run_dir, raw, highlighted


def test_saved_query_gallery_and_raw_clip_handoff(monkeypatch, tmp_path: Path) -> None:
    """Manipulation picks the hidden raw clip, never the visible yellow overlay."""
    assets = tmp_path / "ASSETS"
    run_dir, raw, highlighted = _write_selection_run(assets)
    monkeypatch.setattr(pages, "ASSETS", assets)
    record = json.loads((run_dir / "trace.json").read_text())

    gallery = pages._selection_gallery_items(record["result"]["clips"], run_dir)
    assert gallery[0][0] == str(highlighted)
    assert "2D semantic mask" in gallery[0][1]
    choices = pages._selection_clip_choices()
    assert choices == [("8.0–12.0 · query-a", "query-a:0")]
    assert Path(pages._load_selection_clip("query-a:0")) == raw
    assert pages._load_saved("selection", "query-a")[0] == str(run_dir / "source.mp4")


def test_saved_result_loader_rejects_path_traversal(monkeypatch, tmp_path: Path) -> None:
    """A trace cannot make Gradio serve a file outside its own saved run."""
    assets = tmp_path / "ASSETS"
    run_dir, _raw, _highlighted = _write_selection_run(assets)
    monkeypatch.setattr(pages, "ASSETS", assets)
    with pytest.raises(ValueError, match="unavailable"):
        pages._safe_result_file(run_dir, "../outside.mp4")
    with pytest.raises(pages.gr.Error):
        pages._load_saved("selection", "../query-a")


def test_explicit_edit_viewer_uses_only_run_owned_splats(monkeypatch, tmp_path: Path) -> None:
    """Saved explicit edit metadata restores the orbitable splat timeline safely."""
    splat_root = tmp_path / "explicit_splats"
    splat_root.mkdir()
    paths = []
    for index in range(3):
        path = splat_root / f"splat_{index:03d}.splat"
        path.write_bytes(b"splat")
        paths.append(f"explicit_splats/{path.name}")
    captured: dict[str, object] = {}

    def fake_splat_html(actual_paths, durations, observed_frames):
        captured.update(paths=actual_paths, durations=durations, observed_frames=observed_frames)
        return "viewer"

    monkeypatch.setattr(pages, "splat_html", fake_splat_html)
    update = pages._editing_viewer_update({"viewer": {"splat_paths": paths, "durations": [1 / 12] * 3}}, tmp_path)
    assert update["visible"] is True
    assert captured["paths"] == [str(tmp_path / path) for path in paths]
    assert captured["durations"] == [1 / 12] * 3
    assert captured["observed_frames"] is None
