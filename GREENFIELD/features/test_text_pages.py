"""Saved Text Query handoff and explicit Text Manipulation viewer coverage."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import gradio as gr
from GREENFIELD.app_core.contracts import StageEvent
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


def test_text_page_inputs_are_unwrapped_and_seeded_in_configuration() -> None:
    """Keep compact text fields and the reproducibility controls in configuration."""
    selection_page = pages.build_selection()
    editing_page = pages.build_editing()
    future_page = pages.build_future()

    query = next(component for component in selection_page.blocks.values()
                 if isinstance(component, gr.Textbox) and component.label == "Query")
    instruction = next(component for component in editing_page.blocks.values()
                       if isinstance(component, gr.Textbox) and component.label == "Instruction")
    assert query.container is False
    assert instruction.container is False

    for page in (future_page, editing_page):
        seed = next(component for component in page.blocks.values()
                    if isinstance(component, gr.Number) and component.label == "Seed")
        assert seed.value == 0
        # A control's ``parent`` is its immediate layout, which may be a gr.Row
        # inside the accordion, so walk up to the enclosing "Configuration" panel.
        ancestor = getattr(seed, "parent", None)
        while ancestor is not None and not isinstance(ancestor, gr.Accordion):
            ancestor = getattr(ancestor, "parent", None)
        assert isinstance(ancestor, gr.Accordion)
        assert ancestor.label == "Configuration"


    future_markdown = [
        component.value
        for component in future_page.blocks.values()
        if isinstance(component, gr.Markdown)
    ]
    assert any(value.startswith("## 2. Methodology ") for value in future_markdown)
    assert "## 3. Forecast" in future_markdown
    assert "## 4. Output" in future_markdown
    assert "## 4. Generate" not in future_markdown

    future_button = next(
        component for component in future_page.blocks.values()
        if isinstance(component, gr.Button) and component.value == "Forecast"
    )
    assert future_button.value == "Forecast"
    future_video = next(
        component for component in future_page.blocks.values()
        if isinstance(component, gr.Video) and component.label == "Source View"
    )
    assert future_video.label == "Source View"

    # The Future View panel was rebuilt with the config helpers, so every control
    # inside it carries a short hover description and the shared config-field class
    # while dropping Gradio's heavy box; the long explanations now live in each
    # group's ⓘ caption. Scope to the Configuration accordion so unrelated page
    # controls are not misread.
    def _in_configuration(component):
        ancestor = getattr(component, "parent", None)
        while ancestor is not None:
            if isinstance(ancestor, gr.Accordion) and ancestor.label == "Configuration":
                return True
            ancestor = getattr(ancestor, "parent", None)
        return False

    future_config_controls = {
        component.label: component
        for component in future_page.blocks.values()
        if isinstance(component, (gr.Number, gr.Slider)) and _in_configuration(component)
    }
    assert future_config_controls
    for component in future_config_controls.values():
        assert "evow-config-control" in component.elem_classes, component.label
        assert component.container is False
        assert component.show_label is False
        assert "evow-config-row" in component.parent.elem_classes

    editing_markdown = [
        component.value
        for component in editing_page.blocks.values()
        if isinstance(component, gr.Markdown)
    ]
    assert "## 4. Modify" in editing_markdown
    assert "## 4. Generate" not in editing_markdown
    assert any(
        isinstance(component, gr.Button) and component.value == "Modify"
        for component in editing_page.blocks.values()
    )
    assert any(
        isinstance(component, gr.Video) and component.label == "Source View"
        for component in editing_page.blocks.values()
    )


def test_config_groups_match_the_visibility_handlers() -> None:
    """Moved controls live in the exact group their page's visibility handler toggles.

    Because the groups are plain ``gr.Column`` wrappers, each handler's registered
    outputs are those same columns; walking every control's parents to its nearest
    column proves the regrouping did not leave a methodology-specific control in a
    group that is hidden for that methodology.
    """
    def nearest_column(component):
        ancestor = getattr(component, "parent", None)
        while ancestor is not None:
            if isinstance(ancestor, gr.Column):
                return ancestor
            ancestor = getattr(ancestor, "parent", None)
        return None

    def control(page, label):
        return next(
            component for component in page.blocks.values()
            if getattr(component, "label", None) == label
        )

    def groups(page, handler_name):
        registration = next(
            fn for fn in page.fns.values()
            if getattr(fn.fn, "__name__", "") == handler_name
        )
        return registration.outputs

    future_page = pages.build_future()
    history, forecast, reconstruction, explicit_view, implicit_rollout, self_trained = groups(
        future_page, "_future_config_visibility"
    )
    assert nearest_column(control(future_page, "History (s)")) is history
    assert nearest_column(control(future_page, "Output (s)")) is forecast
    assert nearest_column(control(future_page, "Max side")) is reconstruction
    for label in ("Observed FPS", "Forecast FPS", "Motion fit", "Yaw", "Shift", "FOV", "Source FOV"):
        assert nearest_column(control(future_page, label)) is explicit_view, label
    for label in ("Chunk frames", "SVD chunk", "Seed"):
        assert nearest_column(control(future_page, label)) is implicit_rollout, label
    for label in ("Training side", "LoRA steps", "LoRA rank", "Training pairs"):
        assert nearest_column(control(future_page, label)) is self_trained, label

    selection_page = pages.build_selection()
    source, results, matching, explicit_scene = groups(selection_page, "_selection_config_visibility")
    assert nearest_column(control(selection_page, "Max source (s)")) is source
    assert nearest_column(control(selection_page, "Max results")) is results
    assert nearest_column(control(selection_page, "Grounding")) is matching
    for label in ("Coverage", "Source FOV", "Cache version"):
        assert nearest_column(control(selection_page, label)) is explicit_scene, label

    editing_page = pages.build_editing()
    episode, edit, edit_matching, scene, run = groups(editing_page, "_editing_config_visibility")
    assert nearest_column(control(editing_page, "Episode FPS")) is episode
    for label in ("Max side", "Max height", "VACE steps", "Guidance", "Mask expand"):
        assert nearest_column(control(editing_page, label)) is edit, label
    assert nearest_column(control(editing_page, "Grounding")) is edit_matching
    for label in ("Source FOV", "Cache version"):
        assert nearest_column(control(editing_page, label)) is scene, label
    assert nearest_column(control(editing_page, "Seed")) is run


def test_selection_configuration_panel_matches_tracked_settings() -> None:
    """Text Query exposes every run knob with a description and config-field class."""
    from GREENFIELD.features.text_settings import load_text_settings

    settings = load_text_settings()
    page = pages.build_selection()

    def _in_configuration(component):
        ancestor = getattr(component, "parent", None)
        while ancestor is not None:
            if isinstance(ancestor, gr.Accordion) and ancestor.label == "Configuration":
                return True
            ancestor = getattr(ancestor, "parent", None)
        return False

    controls = {
        component.label: component
        for component in page.blocks.values()
        if isinstance(component, (gr.Number, gr.Slider, gr.Textbox)) and _in_configuration(component)
    }
    expected = {
        "Max source (s)": settings.query_max_source_seconds,
        "Sample FPS": settings.query_sample_fps,
        "Clip (s)": settings.query_clip_seconds,
        "Result FPS": settings.query_result_fps,
        "Max results": settings.query_max_results,
        "Candidate pool": settings.query_explicit_candidate_pool,
        "Grounding": settings.grounding_threshold,
        "Coverage": settings.low_specificity_coverage,
        "Source FOV": settings.source_fov_degrees,
        "Cache version": settings.scene_cache_version,
    }
    assert set(controls) == set(expected)
    for label, value in expected.items():
        assert controls[label].value == value
        assert "evow-config-control" in controls[label].elem_classes, label
        assert controls[label].container is False
        assert controls[label].show_label is False
        assert "evow-config-row" in controls[label].parent.elem_classes


def test_selection_button_wiring_matches_editable_field_order() -> None:
    """The Configuration controls reach ``handle`` in ``_EDITABLE_TEXT_FIELDS`` order."""
    page = pages.build_selection()
    handle = next(fn for fn in page.fns.values() if getattr(fn.fn, "__name__", "") == "handle")

    assert [component.label for component in handle.inputs[:3]] == ["Sample video", "Query", None]
    assert [component.label for component in handle.inputs[3:]] == [
        "Max source (s)", "Sample FPS", "Clip (s)",
        "Result FPS", "Max results", "Candidate pool",
        "Grounding", "Coverage", "Source FOV", "Cache version",
    ]
    assert len(handle.inputs) == 3 + len(pages._EDITABLE_TEXT_FIELDS)


def _in_editing_configuration(component) -> bool:
    """Return whether a control lives inside the Text Manipulation Configuration panel."""
    ancestor = getattr(component, "parent", None)
    while ancestor is not None:
        if isinstance(ancestor, gr.Accordion) and ancestor.label == "Configuration":
            return True
        ancestor = getattr(ancestor, "parent", None)
    return False


def test_editing_configuration_panel_matches_tracked_settings() -> None:
    """Text Manipulation exposes every run knob with a description and config-field class."""
    from GREENFIELD.features.text_settings import load_text_settings

    settings = load_text_settings()
    page = pages.build_editing()
    controls = {
        component.label: component
        for component in page.blocks.values()
        if isinstance(component, (gr.Number, gr.Slider, gr.Textbox)) and _in_editing_configuration(component)
    }
    expected = {
        "Episode FPS": settings.episode_fps,
        "Episode frames": settings.episode_max_frames,
        "Max side": settings.edit_max_side,
        "Max height": settings.edit_max_height,
        "VACE steps": settings.vace_steps,
        "Guidance": settings.vace_guidance_scale,
        "Mask expand": settings.edit_mask_expand_px,
        "Grounding": settings.grounding_threshold,
        "Source FOV": settings.source_fov_degrees,
        "Cache version": settings.scene_cache_version,
        "Seed": 0,
    }
    assert set(controls) == set(expected)
    for label, value in expected.items():
        assert controls[label].value == value
        assert "evow-config-control" in controls[label].elem_classes, label
        assert controls[label].container is False
        assert controls[label].show_label is False
        assert "evow-config-row" in controls[label].parent.elem_classes


def test_editing_button_wiring_matches_editable_field_order() -> None:
    """The Configuration controls reach ``handle`` in ``_EDITABLE_EDITING_FIELDS`` order."""
    page = pages.build_editing()
    handle = next(fn for fn in page.fns.values() if getattr(fn.fn, "__name__", "") == "handle")

    assert [component.label for component in handle.inputs[:4]] == ["Sample video", "Instruction", None, "Seed"]
    assert [component.label for component in handle.inputs[4:]] == [
        "Episode FPS", "Episode frames",
        "Max side", "Max height", "VACE steps", "Guidance", "Mask expand",
        "Grounding", "Source FOV", "Cache version",
    ]
    assert len(handle.inputs) == 4 + len(pages._EDITABLE_EDITING_FIELDS)


def test_text_feature_explicit_3d_viewers_stay_visible_before_scene_metadata(tmp_path: Path) -> None:
    """Explicit 3D reserves an informative result panel while generation is pending."""
    selection_update = pages._selection_method_change("explicit")
    assert selection_update["visible"] is True
    assert "An Explicit 3D scene will appear here" in selection_update["value"]
    assert pages._selection_method_change("implicit")["visible"] is False
    editing_update = pages._editing_method_change("explicit")
    assert editing_update["visible"] is True
    assert "An Explicit 3D scene will appear here" in editing_update["value"]
    assert pages._editing_method_change("implicit")["visible"] is False
    assert pages._selection_viewer_update({}, tmp_path)["visible"] is True
    assert pages._editing_viewer_update({}, tmp_path)["visible"] is True

    selection_page = pages.build_selection()
    editing_page = pages.build_editing()
    future_page = pages.build_future()
    for page, viewer_id in (
        (selection_page, "selection-splat-viewer"),
        (editing_page, "editing-splat-viewer-output"),
        (future_page, "future-splat-viewer-output"),
    ):
        viewer = next(
            component
            for component in page.blocks.values()
            if isinstance(component, gr.HTML) and component.elem_id == viewer_id
        )
        assert viewer.visible is True
        assert viewer.label == "3D View"
        assert viewer.show_label is True
        assert "evow-splat-viewer" in viewer.elem_classes
        assert viewer.padding is False
        assert "An Explicit 3D scene will appear here" in viewer.value


def test_shared_feature_streams_yield_terminal_error_traces(monkeypatch, tmp_path: Path) -> None:
    """Text Query and generic feature streams repaint a failed stage before a toast."""
    assets = tmp_path / "ASSETS"
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(pages, "ASSETS", assets)
    request = SimpleNamespace(mode="implicit")

    def failing_source_adapter(*_args):
        yield StageEvent("Search clips", "running", "started")
        raise RuntimeError("source-stream failure")

    source_stream = pages._execute_source_adapter_stream(
        "selection", "implicit", str(source), request, failing_source_adapter,
    )
    next(source_stream)
    _result, terminal_source_trace = next(source_stream)
    assert terminal_source_trace["events"][-2]["stage"] == "Search clips"
    assert terminal_source_trace["events"][-2]["status"] == "error"
    with pytest.raises(gr.Error, match="source-stream failure"):
        next(source_stream)

    monkeypatch.setattr(pages, "read_video", lambda _source: ([np.zeros((2, 2, 3), dtype=np.uint8)], 12))

    def failing_generic_adapter(*_args):
        yield StageEvent("Generate", "running", "started")
        raise RuntimeError("generic-stream failure")

    generic_stream = pages._execute_stream(
        "selection", "implicit", str(source), request, failing_generic_adapter,
    )
    next(generic_stream)
    _result, terminal_generic_trace = next(generic_stream)
    assert terminal_generic_trace["events"][-2]["stage"] == "Generate"
    assert terminal_generic_trace["events"][-2]["status"] == "error"
    with pytest.raises(gr.Error, match="generic-stream failure"):
        next(generic_stream)


def test_editing_inline_handler_renders_failed_stage_before_gradio_error(monkeypatch, tmp_path: Path) -> None:
    """The inline Text Manipulation handler must repaint its failed stage before a toast.

    Unlike the shared ``_execute_*_stream`` helpers, ``build_editing``'s ``handle`` is a
    closure that yields Gradio updates rather than the raw trace. Reach the registered
    callback through the built page and assert the terminal yield carries the dashboard's
    ``evow-stage-failed`` class, then that the next step raises the user-facing error.
    """
    assets = tmp_path / "ASSETS"
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(pages, "ASSETS", assets)

    def failing_run(*_args, **_kwargs):
        # Every emitted stage must carry its declared id; the collision-safe
        # renderer no longer resolves an unknown display name.
        yield StageEvent("Decode source episode", "running", "decoding", stage_id="episode_decode")
        raise RuntimeError("editing exploded")

    monkeypatch.setattr(pages.editing, "run", failing_run)

    from GREENFIELD.features.text_settings import load_text_settings

    settings = load_text_settings()
    config_values = [getattr(settings, name) for name in pages._EDITABLE_EDITING_FIELDS]

    page = pages.build_editing()
    handler = next(fn.fn for fn in page.fns.values() if getattr(fn.fn, "__name__", "") == "handle")
    stream = handler(str(source), "flood the forest", "implicit", 0, *config_values)
    next(stream)  # initial result-surface clear
    next(stream)  # first running stage
    terminal = next(stream)  # terminal error repaint
    assert any(
        isinstance(item, dict) and "evow-stage-failed" in (item.get("elem_classes") or [])
        for item in terminal
    ), f"terminal yield did not render a failed stage: {terminal!r}"
    with pytest.raises(gr.Error, match="editing exploded"):
        next(stream)
