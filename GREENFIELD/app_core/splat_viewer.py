"""Public explicit-scene viewer API shared by all dashboard features."""

from GREENFIELD.replay.splat_viewer import (
    MAX_SCENES_PER_VIEWER,
    _boundary_percent,
    _scene_groups,
    empty_splat_html,
    splat_html,
)


# ``gr.HTML`` normally lays out its BlockLabel above the content. Explicit 3D
# results need the same floating label treatment as Gradio's native Video
# component, so the title is part of the viewer frame rather than a separate row.
SPLAT_VIEWER_FRAME_CSS = """
.evow-splat-viewer { position: relative !important; overflow: hidden !important; }
.evow-splat-viewer > label[data-testid="block-label"] {
  position: absolute !important;
  top: var(--block-label-margin) !important;
  left: var(--block-label-margin) !important;
  z-index: var(--layer-2) !important;
  margin: 0 !important;
}
"""
__all__ = [
    "MAX_SCENES_PER_VIEWER",
    "SPLAT_VIEWER_FRAME_CSS",
    "_boundary_percent",
    "_scene_groups",
    "empty_splat_html",
    "splat_html",
]
