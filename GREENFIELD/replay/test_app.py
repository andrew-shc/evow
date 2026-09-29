"""Regression coverage for Replay's compact configuration and flow structure."""

import gradio as gr

from GREENFIELD.dashboard import FLOW_HEADER_CSS
from GREENFIELD.replay.app import _workflow_group_updates, build_app
from GREENFIELD.replay.settings import load_settings
from GREENFIELD.replay.workflow import MAX_FLOW_STAGES


def _components(page, component_type):
    return [component for component in page.blocks.values() if isinstance(component, component_type)]


def _control(page, component_type, label):
    return next(component for component in page.blocks.values() if isinstance(component, component_type) and component.label == label)


def _in_configuration(component):
    ancestor = getattr(component, "parent", None)
    while ancestor is not None:
        if isinstance(ancestor, gr.Accordion) and ancestor.label == "Configuration":
            return True
        ancestor = getattr(ancestor, "parent", None)
    return False


def test_reconstruct_heading_and_button_replace_record():
    page = build_app()
    markdown_values = [component.value for component in _components(page, gr.Markdown)]
    headings = set(markdown_values)
    assert "## 4. Reconstruct" in headings
    assert "## 4. Record" not in headings
    assert any(component.value == "Reconstruct" for component in _components(page, gr.Button))
    # help_icon appends an inline HTML span, so the section 3 heading is matched
    # by prefix rather than equality; requiring "evow-help" keeps the info icon
    # from silently disappearing.
    section_three = next(
        (value for value in markdown_values if value.startswith("## 3. Change Camera View Pose")),
        "",
    )
    assert "evow-help" in section_three


def test_pose_sliders_use_precise_axis_labels():
    """Section 3 sliders name the exact camera axis they change."""
    page = build_app()
    assert _control(page, gr.Slider, "Rotate about the vertical axis (°)")
    assert _control(page, gr.Slider, "Lateral translation (sideways)")
    assert _control(page, gr.Slider, "Output FOV")
    # The former ambiguous labels must not linger as duplicates.
    slider_labels = {component.label for component in _components(page, gr.Slider)}
    assert slider_labels.isdisjoint({"Turn", "Sideways", "Field of view"})


def test_configuration_is_one_flat_list_with_every_method_knob_visible():
    page = build_app()
    labels = [component.label for component in _components(page, gr.Tab)]
    assert "Model" not in labels
    assert "Advanced" not in labels
    assert _control(page, gr.Number, "Start (s)")
    assert _control(page, gr.Number, "Gaussian steps")
    assert _control(page, gr.Number, "AnyView steps")
    rows = [component for component in _components(page, gr.Row) if "evow-config-row" in (component.elem_classes or [])]
    assert len(rows) == 22


def test_configuration_defaults_and_tooltips_remain_bound_to_tracked_settings():
    page = build_app()
    tracked = load_settings()
    assert _control(page, gr.Number, "FPS").value == tracked.fps
    assert _control(page, gr.Number, "Gaussian steps").value == tracked.gaussian_steps
    assert _control(page, gr.Number, "AnyView steps").value == tracked.anyview_steps
    controls = [component for component in page.blocks.values() if isinstance(component, (gr.Number, gr.Slider, gr.Radio)) and _in_configuration(component)]
    assert controls
    # Replay owns a label/help cell for every field, so its editable controls
    # are intentionally bare rather than Gradio's card-like config containers.
    assert all(component.container is False for component in controls)
    assert all(component.show_label is False for component in controls)
    assert all("evow-config-control" in component.elem_classes for component in controls)
    labels = [component.value for component in _components(page, gr.Markdown) if "evow-config-label" in component.elem_classes]
    assert len(labels) == len(controls)
    assert all("evow-help" in label for label in labels)


def test_configuration_controls_are_never_mode_gated_and_flow_is_preallocated():
    page = build_app()
    controls = [component for component in page.blocks.values() if isinstance(component, (gr.Number, gr.Slider, gr.Radio)) and _in_configuration(component)]
    assert all(component.visible for component in controls)
    stage_rows = [component for component in _components(page, gr.Accordion) if "evow-stage" in component.elem_classes]
    assert len(stage_rows) == MAX_FLOW_STAGES
    assert build_app() is not None



def test_replay_configuration_controls_remain_editable_in_dashboard():
    """Guard the CSS boundary where Gradio labels wrap Replay's actual inputs."""
    page = build_app()
    labels = {
        "FPS", "Max side", "AnyView steps", "Guidance", "Depth input",
        "Gaussian stride", "Gaussian steps", "LR · position", "LR · rotation",
        "LR · scale", "LR · color", "LR · opacity", "LR · motion",
        "Depth weight", "Smooth weight", "Init opacity", "BG · R", "BG · G",
        "BG · B",
    }
    controls = [
        component for component in page.blocks.values()
        if getattr(component, "label", None) in labels
    ]
    assert {component.label for component in controls} == labels
    # Gradio uses None for its default editable state; only False disables a control.
    assert all(component.visible is not False and component.interactive is not False for component in controls)
    # A Gradio Number/Radio's native label encloses its real input. Hiding the
    # label to remove duplicate text would hide every control in this list.
    assert ".evow-config-control > label" not in FLOW_HEADER_CSS
    assert ".evow-config-control [data-testid=\"block-label\"]" not in FLOW_HEADER_CSS


def test_replay_saved_runs_are_folded_into_the_source_menu():
    page = build_app()
    saved = next(component for component in _components(page, gr.Dropdown) if component.elem_id == "evow-replay-saved-runs")
    assert saved.label == "Choose a saved run"
    source_selector = next(
        component for component in _components(page, gr.Dropdown)
        if component is not saved and component.label is None
    )
    assert ("Saved run", "saved") in source_selector.choices


def test_atomic_stages_are_nested_under_stable_flow_groups():
    page = build_app()
    groups = [
        component for component in _components(page, gr.Accordion)
        if "evow-flow-group" in component.elem_classes
    ]
    assert [group.label for group in groups] == [
        "Input · Waiting · —", "Depth prior · Waiting · —",
        "Build 4D scene · Waiting · —", "Render + export · Waiting · —",
        "Finish · Waiting · —",
    ]
    # The group accordions own the stage component trees; no stage is left as a
    # flat direct child of the outer execution-flow accordion.
    stage_rows = [component for component in _components(page, gr.Accordion) if "evow-stage" in component.elem_classes]
    assert all(any(ancestor is group for group in groups for ancestor in _ancestors(component)) for component in stage_rows)


def _ancestors(component):
    ancestor = getattr(component, "parent", None)
    while ancestor is not None:
        yield ancestor
        ancestor = getattr(ancestor, "parent", None)


def test_flow_group_titles_follow_the_selected_method_and_preserve_legacy_trace():
    explicit = _workflow_group_updates({"workflow_version": 2, "mode": "explicit", "events": []})
    implicit = _workflow_group_updates({"workflow_version": 2, "mode": "implicit", "events": []})
    legacy = _workflow_group_updates({"workflow_version": 1, "mode": "explicit", "events": []})
    assert [update["label"] for update in explicit] == [
        "Input · Waiting · —", "Depth prior · Waiting · —",
        "Build 4D scene · Waiting · —", "Render + export · Waiting · —",
        "Finish · Waiting · —",
    ]
    assert [update["label"] for update in implicit] == [
        "Input · Waiting · —", "Episode + models · Waiting · —",
        "Encode inputs · Waiting · —", "Condition + generate · Waiting · —",
        "Decode + finish · Waiting · —",
    ]
    assert legacy[0]["label"] == "Legacy trace · Recorded · —"
    assert all(update["visible"] is False for update in legacy[1:])


def test_flow_group_rolls_up_atomic_status_and_accumulated_time():
    trace = {
        "workflow_version": 2,
        "mode": "explicit",
        "events": [
            {"stage_id": "load_depth_model", "status": "complete", "stage_elapsed_seconds": 1.0},
            {"stage_id": "infer_temporal_depth", "status": "complete", "stage_elapsed_seconds": 2.0},
            {"stage_id": "write_depth_prior", "status": "complete", "stage_elapsed_seconds": 3.0},
        ],
    }
    group = _workflow_group_updates(trace)[1]
    assert group["label"] == "Depth prior · Completed · 6.000s"
    assert group["elem_classes"] == ["evow-flow-group", "evow-flow-group-complete"]
