"""Compose the four camera features into one private LAN dashboard."""

import gradio as gr

from GREENFIELD.features.pages import build_editing, build_future, build_selection
from GREENFIELD.replay.app import build_app as build_replay


FLOW_HEADER_CSS = """
html, body { overflow-x: hidden !important; }
.internal-flow { overflow-x: hidden !important; }
.internal-flow > button {
  width: 1.5rem !important;
  min-height: 1.5rem !important;
  justify-content: flex-start !important;
}
.internal-flow > button > span:first-child { display: none !important; }
.internal-flow > button > .icon { margin: 0 !important; font-size: 1rem !important; transform: rotate(-90deg) !important; }
.internal-flow > button.open > .icon { transform: rotate(0deg) !important; }
.evow-section-heading h2 { font-size: 1.5rem !important; font-weight: 700 !important; line-height: 1.35 !important; }
.evow-flow-heading h2 { color: #7e22ce !important; }
.evow-media { outline: 2px solid #58748d !important; outline-offset: -2px !important; border-radius: 10px !important; background: #f8fafc !important; overflow: hidden !important; }
.evow-media > div { border: 0 !important; }
#novel-view-output, #splat-viewer-output { position: relative; overflow: hidden; }
.evow-generation-spinner::after { content: ""; position: absolute; inset: 0; z-index: 10; background: rgb(248 250 252 / 72%); pointer-events: all; }
.evow-generation-spinner::before { content: ""; position: absolute; top: 50%; left: 50%; z-index: 11; width: 28px; height: 28px; margin: -14px; border: 3px solid #628197; border-top-color: #102a43; border-radius: 50%; animation: evow-generation-spin .8s linear infinite; }
@keyframes evow-generation-spin { to { transform: rotate(360deg); } }
"""


def build_dashboard() -> gr.Blocks:
    """Return the eight-mode dashboard grouped by its four user-facing features."""
    return gr.TabbedInterface(
        [build_replay(), build_future(), build_selection(), build_editing()],
        ["4D Replay", "Future View (+ Training)", "Text Query", "Text Manipulation"],
        title="An Eternal View of Our World · Applications of Single-view Stationary Camera",
        css=FLOW_HEADER_CSS,
    )
