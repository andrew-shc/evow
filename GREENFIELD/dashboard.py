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
.evow-stage > button { display: flex !important; align-items: center; justify-content: flex-start !important; margin: 0 !important; height: 2.7rem; min-height: 2.7rem; padding: 9px 11px !important; border: 1px solid #cbd5df !important; border-left: 5px solid #94a3b8 !important; border-radius: 7px !important; background: #f8fafc !important; text-align: left !important; white-space: nowrap !important; }
.evow-stage > button > span:first-child { position: relative; display: block !important; flex: 1 1 0 !important; min-width: 0; width: auto !important; height: 1.3rem; overflow: hidden; font-size: 0 !important; text-align: left !important; }
.evow-stage > button > .icon { flex: 0 0 auto; margin: 0 0 0 12px !important; }
.evow-stage > button > span:first-child::before { content: attr(data-evow-stage-name); position: absolute; top: 0; right: 11rem; left: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--section-header-text-size); line-height: 1.3rem; text-align: left; }
.evow-stage > button > span:first-child::after { content: attr(data-evow-stage-meta); position: absolute; top: 0; right: 0; width: 10.5rem; overflow: hidden; white-space: nowrap; font-size: 12px; line-height: 1.3rem; font-variant-numeric: tabular-nums; font-weight: 650; text-align: right; }
.evow-stage-complete > button { border-left-color: #15803d !important; background: #f0fdf4 !important; color: #166534 !important; }
.evow-stage-running > button { border-left-color: #0369a1 !important; background: #f0f9ff !important; color: #075985 !important; }
.evow-stage-waiting > button { border-left-color: #94a3b8 !important; background: #f8fafc !important; color: #475569 !important; }
.evow-stage-failed > button { border-left-color: #b91c1c !important; background: #fff1f2 !important; color: #991b1b !important; }
"""


FLOW_HEADER_JS = """
() => {
  const formatStageHeaders = () => {
    document.querySelectorAll(".evow-stage > button > span:first-child").forEach((label) => {
      const parts = label.textContent.split(" · ");
      if (parts.length !== 3) return;
      label.dataset.evowStageName = parts[0];
      label.dataset.evowStageMeta = `${parts[1]} · ${parts[2]}`;
    });
  };
  formatStageHeaders();
  new MutationObserver(formatStageHeaders).observe(document.body, {
    characterData: true,
    childList: true,
    subtree: true,
  });
}
"""


def build_dashboard() -> gr.Blocks:
    """Return the eight-mode dashboard grouped by its four user-facing features."""
    return gr.TabbedInterface(
        [build_replay(), build_future(), build_selection(), build_editing()],
        ["4D Replay", "Future View (+ Training)", "Text Query", "Text Manipulation"],
        title="An Eternal View of Our World · Applications of Single-view Stationary Camera",
        css=FLOW_HEADER_CSS,
        js=FLOW_HEADER_JS,
    )
