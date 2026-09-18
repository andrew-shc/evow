"""Compose the four camera features into one private LAN dashboard."""

import gradio as gr

from GREENFIELD.features.pages import build_editing, build_future, build_selection
from GREENFIELD.replay.app import build_app as build_replay


def build_dashboard() -> gr.Blocks:
    """Return the eight-mode dashboard grouped by its four user-facing features."""
    return gr.TabbedInterface(
        [build_replay(), build_future(), build_selection(), build_editing()],
        ["4D Replay", "Future View", "Text Query", "Text Manipulation"],
        title="An Eternal View of Our World · Applications of Single-view Stationary Camera",
    )
