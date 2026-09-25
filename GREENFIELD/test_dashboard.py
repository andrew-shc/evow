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

import gradio as gr

from GREENFIELD.dashboard import build_dashboard


def test_dashboard_composes_four_tabs_and_one_global_clear_button() -> None:
    dashboard = build_dashboard()
    components = list(dashboard.blocks.values())

    tab_labels = [c.label for c in components if isinstance(c, gr.Tab)]
    assert tab_labels == [
        "4D Replay",
        "Future View (+ Training)",
        "Text Query",
        "Text Manipulation",
    ]

    # Button text is stored in ``.value`` in Gradio 5.50, not ``.label``.
    clear_buttons = [
        c
        for c in components
        if isinstance(c, gr.Button) and getattr(c, "value", None) == "Clear GPU memory"
    ]
    assert len(clear_buttons) == 1
