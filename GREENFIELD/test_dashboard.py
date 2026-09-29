"""Regression guard for the manually composed dashboard.

``build_dashboard()`` assembles its root ``gr.Blocks`` by hand: a persistent top
bar plus manually created ``gr.Tabs()`` whose tabs ``render()`` already-built
child pages. Because that composition is manual rather than a single
``gr.TabbedInterface`` call, a Gradio upgrade can silently drop a tab or
miscount/leak the dashboard-wide handlers without raising anything. This test
pins the structural contract the UI depends on: the four feature tabs in order,
and exactly one dashboard-wide "Clear GPU memory" button (child pages must never
grow their own copy).
"""

import re

import gradio as gr

from GREENFIELD.app_core.ui import config_caption, config_number, config_slider, config_textbox
from GREENFIELD.dashboard import DASHBOARD_SUBTITLE, DASHBOARD_TITLE, FLOW_HEADER_CSS, FLOW_HEADER_JS, build_dashboard
from GREENFIELD.features.pages import PROJECT_SAMPLE_VIDEOS, _sample_videos
from GREENFIELD.replay.app import REPLAY_SAMPLE_VIDEOS
from GREENFIELD.replay.app import _allowed_paths, settings


def test_dashboard_composes_four_tabs_and_one_global_clear_button() -> None:
    dashboard = build_dashboard()
    components = list(dashboard.blocks.values())

    # Replay now owns nested Model/Advanced configuration tabs. Select the
    # dashboard-level tab contract by its stable feature labels rather than
    # assuming there are no feature-local Tab components.
    feature_labels = {"3D Video", "Future View", "Text Query", "Text Manipulation"}
    tab_labels = [c.label for c in components if isinstance(c, gr.Tab) and c.label in feature_labels]
    assert tab_labels == ["3D Video", "Future View", "Text Query", "Text Manipulation"]

    # Button text is stored in ``.value`` in Gradio 5.50, not ``.label``.
    clear_buttons = [
        c
        for c in components
        if isinstance(c, gr.Button) and getattr(c, "value", None) == "Clear GPU memory"
    ]
    assert len(clear_buttons) == 1


def test_dashboard_uses_one_persistent_execution_flow_accordion_per_tab() -> None:
    """Pin the visible native Accordion title used to reveal flow details."""
    dashboard = build_dashboard()

    flow_accordions = [
        component
        for component in dashboard.blocks.values()
        if isinstance(component, gr.Accordion)
        and component.label == "Internal Execution Flow"
    ]

    assert len(flow_accordions) == 4

def test_dashboard_header_saved_runs_and_unavailable_mesh_choice_are_explicit() -> None:
    """Pin the title stack and the controls whose browser defaults are misleading."""
    dashboard = build_dashboard()
    components = list(dashboard.blocks.values())
    markdown_values = [
        component.value
        for component in components
        if isinstance(component, gr.Markdown)
    ]

    assert f"# {DASHBOARD_TITLE}" in markdown_values
    assert DASHBOARD_SUBTITLE in markdown_values
    assert f"{DASHBOARD_TITLE} · {DASHBOARD_SUBTITLE}" not in markdown_values
    assert "# 3D Video" in markdown_values

    saved = [
        component
        for component in components
        if isinstance(component, gr.Dropdown)
        and str(getattr(component, "elem_id", "")).endswith("-saved-runs")
    ]
    assert len(saved) == 4
    assert all(component.value is None for component in saved)
    replay_saved = next(component for component in saved if component.elem_id == "evow-replay-saved-runs")
    assert replay_saved.label == "Choose a saved run"
    assert all(component.label == "Choose a saved run" for component in saved if component is not replay_saved)

    assert all("evow-saved-run-dropdown" in component.elem_classes for component in saved)

    source_menus = [
        component for component in components
        if isinstance(component, gr.Dropdown)
        and ("Sample Video (Basic)", "sample_0") in component.choices
    ]
    assert len(source_menus) == 4
    assert all(menu.choices[:3] == [
        ("Sample Video (Basic)", "sample_0"),
        ("Sample Video 1", "sample_1"),
        ("Sample Video 2", "sample_2"),
    ] for menu in source_menus)
    assert all(menu.choices[3:] == [
        ("Video File", "video_file"), ("Webcam", "webcam"),
        ("USB Camera", "usb_camera"), ("Saved Run", "saved"),
    ] for menu in source_menus)
    source_previews = [
        component for component in components
        if isinstance(component, gr.Video) and "evow-source-video" in component.elem_classes
    ]
    assert len(source_previews) == 4
    assert all(component.elem_id.endswith("-source-video") for component in source_previews)
    assert all(component.format is None for component in source_previews)
    assert "evow-source-loading::after" in FLOW_HEADER_CSS
    assert "Loading video…" in FLOW_HEADER_CSS
    assert "muteDashboardVideos" in FLOW_HEADER_JS
    assert "finishSourceLoading" in FLOW_HEADER_JS
    assert 'input.setAttribute("placeholder", "Choose a saved run")' in FLOW_HEADER_JS


    replay_method = next(
        component
        for component in components
        if isinstance(component, gr.Radio)
        and getattr(component, "elem_id", None) == "replay-method-choice"
    )
    assert ("Explicit 3D: Animated Mesh", "mesh") in replay_method.choices
    # Gradio Radio only supports group-wide disabling, so root-level JS handles
    # pointer and keyboard activation for this one intentionally retained option.
    assert "control.disabled = true" in FLOW_HEADER_JS
    assert 'choice.addEventListener("keydown", blockChoice, true)' in FLOW_HEADER_JS

    splat_viewers = [
        component
        for component in components
        if isinstance(component, gr.HTML) and component.label == "3D View"
    ]
    assert len(splat_viewers) == 4
    assert all("evow-splat-viewer" in component.elem_classes for component in splat_viewers)
    assert all(component.padding is False for component in splat_viewers)
    assert '.evow-splat-viewer > label[data-testid="block-label"]' in FLOW_HEADER_CSS



def test_project_sample_paths_and_hard_refresh_allowlist() -> None:
    """Each tab owns its SP0X pair and Gradio can serve it after a refresh."""
    assert [path.name for _label, path in REPLAY_SAMPLE_VIDEOS[1:]] == [
        "sp01_app01_nasa_rocket.mp4", "sp01_app02_VIRAT.mp4",
    ]
    expected = {
        "future": ("sp02_app01_ARECIBO_OBS_SHORT.mp4", "sp02_app02_tsunami.mp4"),
        "selection": ("sp03_app01_river_flooding.mp4", "sp03_app02_flood_leakage.avi"),
        "editing": ("sp04_app01_windsock.mp4", "sp04_app02_model_house_timelapse.mp4"),
    }
    for feature, filenames in expected.items():
        assert PROJECT_SAMPLE_VIDEOS[feature] == _sample_videos(feature)[1:]
        samples = _sample_videos(feature)
        assert [path.name for _label, path in samples[1:]] == list(filenames)
        assert all(label.startswith("Sample Video") for label, _path in samples)

    allowed = _allowed_paths()
    assert str(settings.assets / "replay" / "samples") in allowed
    assert str(settings.assets / "samples") in allowed


def test_shared_config_helpers_keep_labels_compact() -> None:
    """Pin the compact config helpers every feature page uses to save width.

    Each control carries its explanation in Gradio's ``info`` prop so the shared
    root CSS can hide the standing paragraph and the shared root JS can mirror it
    onto the entry box as a native ``title`` tooltip, while the short ``label``
    stays visible. The ``evow-config-field`` marker class is the hook both depend
    on, and the CSS/JS must live in ``FLOW_HEADER_CSS``/``FLOW_HEADER_JS`` because
    the serving dashboard mounts only those (child-page assets are dropped by
    ``render()``).
    """
    assert ".evow-config-caption" in FLOW_HEADER_CSS
    assert ".evow-config-panel label" in FLOW_HEADER_CSS
    # One shared row/label/control contract for all four Configuration panels.
    assert ".evow-config-panel .evow-config-row" in FLOW_HEADER_CSS
    assert ".evow-config-panel .evow-config-label" in FLOW_HEADER_CSS
    assert ".evow-config-panel .evow-config-control" in FLOW_HEADER_CSS
    # The Column gap-reset is what stops the later panels' mode-toggle columns
    # from adding Gradio's default layout gap as extra vertical padding.
    assert ".evow-config-panel .column { gap: 0 !important; }" in FLOW_HEADER_CSS
    assert ".evow-flow-group" in FLOW_HEADER_CSS
    # Narrow viewports must keep stacking the nowrap desktop rows.
    assert "@media (max-width: 640px)" in FLOW_HEADER_CSS
    assert ".evow-config-panel .evow-config-row { align-items: stretch !important; }" in FLOW_HEADER_CSS
    assert ".evow-flow-group > button > span:first-child::after" in FLOW_HEADER_CSS
    assert ".evow-flow-group > button > span:first-child" in FLOW_HEADER_JS
    # Only the long info paragraph is hidden; the JS then puts the same text on
    # the inputs as a native title. ``block-info`` is Gradio's *short label*, so
    # hiding it would leave every knob unlabeled -- guard against that regression.
    assert ".evow-config-field .md.prose" in FLOW_HEADER_CSS
    assert ".evow-config-field div:has(> .md.prose)" in FLOW_HEADER_CSS
    # No actual rule may hide ``block-info`` (the comment above may name it).
    css_without_comments = re.sub(r"/\*.*?\*/", "", FLOW_HEADER_CSS, flags=re.DOTALL)
    assert '[data-testid="block-info"]' not in css_without_comments

    caption = config_caption("Capture", "Explains capture")
    assert "Capture" in caption.value
    assert "evow-help" in caption.value

    # ``info`` is the carrier for the hover tooltip. Gradio 5.50 forces the short
    # label into ``sr-only`` whenever ``container=False``, so the helpers keep the
    # container and ``show_label`` on to render visible labels. Guard the
    # attributes with getattr so the contract still holds across an upgrade.
    detail = "Sampled frames per second"
    number = config_number("FPS", 12, detail=detail)
    slider = config_slider("FOV", 70, 35, 110, detail=detail)
    textbox = config_textbox("Name", "value", detail=detail)
    for control in (number, slider, textbox):
        assert getattr(control, "info", None) == detail
        assert "evow-config-field" in control.elem_classes
        assert getattr(control, "container", None) is True
        assert getattr(control, "show_label", None) is True

    # A caller's own classes survive, and the marker never duplicates.
    merged = config_number("FPS", 12, detail=detail, elem_classes=["evow-custom"])
    assert merged.elem_classes == ["evow-custom", "evow-config-field"]

    # The tooltip hook must ship in the root script that actually reaches the page.
    assert "applyConfigFieldTooltips" in FLOW_HEADER_JS
    assert "evow-config-field" in FLOW_HEADER_JS


def test_all_four_configuration_panels_share_the_compact_row_contract() -> None:
    """Every Configuration accordion renders through the one shared row builder.

    Building each panel separately is what let Replay's spacing diverge from the
    later three. This pins that all four carry the shared panel marker and that
    every control and label inside them uses the shared row/label/control classes
    (so the single ``FLOW_HEADER_CSS`` block can style them identically).
    """
    dashboard = build_dashboard()
    components = list(dashboard.blocks.values())
    panels = [
        component for component in components
        if isinstance(component, gr.Accordion) and component.label == "Configuration"
    ]
    assert len(panels) == 4
    assert all("evow-config-panel" in component.elem_classes for component in panels)

    def _within(component, panel) -> bool:
        ancestor = getattr(component, "parent", None)
        while ancestor is not None:
            if ancestor is panel:
                return True
            ancestor = getattr(ancestor, "parent", None)
        return False

    control_types = (gr.Number, gr.Slider, gr.Radio, gr.Textbox)
    for panel in panels:
        controls = [
            component for component in components
            if isinstance(component, control_types) and _within(component, panel)
        ]
        labels = [
            component for component in components
            if isinstance(component, gr.Markdown)
            and "evow-config-label" in (component.elem_classes or [])
            and _within(component, panel)
        ]
        assert controls, "Configuration panel exposed no controls"
        assert len(labels) == len(controls)
        assert all("evow-config-control" in component.elem_classes for component in controls)
        assert all("evow-config-row" in component.parent.elem_classes for component in controls)
