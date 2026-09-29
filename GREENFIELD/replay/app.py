"""LAN Gradio dashboard for fixed-camera 4D replay and its execution flow."""

import os
from pathlib import Path

from .settings import load_settings

# Keep Gradio's upload copies inside the ignored data tree.
settings = load_settings()
os.environ.setdefault("GRADIO_TEMP_DIR", str(settings.temp_root))

import gradio as gr

from .run import execute_run, list_saved_runs, load_saved_run
from .overrides import ReplayOverrides
from GREENFIELD.app_core.splat_viewer import SPLAT_VIEWER_FRAME_CSS, empty_splat_html, splat_html
from .workflow import MAX_FLOW_STAGES, workflow_card_html, workflow_group_summary, workflow_header, workflow_visible

from GREENFIELD.app_core.ui import (
    config_control_kwargs, config_row, flat_config_number, flat_config_slider,
    help_icon, run_total_update, source_and_saved_controls,
)

SAMPLE_VIDEO = settings.assets / "replay" / "samples" / "trees_swaying_pexels_12644693.mp4"
REPLAY_CSS = """
.evow-section-heading h2 { font-size: 1.5rem !important; font-weight: 700 !important; line-height: 1.35 !important; }
.evow-media { border: 2px solid #71869a !important; border-radius: 10px !important; padding: 6px !important; background: #f8fafc !important; }
#novel-view-output, #splat-viewer-output { position: relative; overflow: hidden; }
/* .evow-help icon styling now lives in dashboard.py's FLOW_HEADER_CSS: this
   nested Blocks css is dropped when the dashboard merges pages with render(). */
.evow-generation-spinner::after { content: ""; position: absolute; inset: 0; z-index: 10; background: rgb(248 250 252 / 72%); pointer-events: all; }
.evow-generation-spinner::before { content: ""; position: absolute; top: 50%; left: 50%; z-index: 11; width: 28px; height: 28px; margin: -14px; border: 3px solid #628197; border-top-color: #102a43; border-radius: 50%; animation: evow-generation-spin .8s linear infinite; }
@keyframes evow-generation-spin { to { transform: rotate(360deg); } }
#replay-method-choice .evow-disabled-method { opacity: .52 !important; cursor: not-allowed !important; }
#replay-method-choice .evow-disabled-method input,
#replay-method-choice .evow-disabled-method [role="radio"] { cursor: not-allowed !important; }
/* Standalone Replay has no dashboard root CSS, so it carries the shared compact
   one-setting-per-row contract itself. These selectors mirror the single block
   in dashboard.py's FLOW_HEADER_CSS; keep the two in lockstep. The .column reset
   kills Gradio 5.50's default layout gap on the accordion's content column. */
.evow-config-panel { padding: 2px 6px 4px !important; }
.evow-config-panel .column { gap: 0 !important; }
.evow-config-panel .form,
.evow-config-panel .block,
.evow-config-panel .wrap { gap: 0 !important; }
.evow-config-panel .block { padding: 0 !important; }
.evow-config-panel .evow-config-row {
  align-items: center !important;
  gap: 8px !important;
  margin: 0 !important;
  padding: 2px 0 !important;
  border-bottom: 1px solid #e5edf3 !important;
}
.evow-config-panel .evow-config-row:last-child { border-bottom: 0 !important; }
.evow-config-panel .evow-config-label {
  flex: 0 1 43% !important;
  min-width: 9rem !important;
  margin: 0 !important;
  color: #334e68 !important;
  font-size: .74rem !important;
  font-weight: 600 !important;
  line-height: 1.3 !important;
}
.evow-config-panel .evow-config-label p { margin: 0 !important; }
.evow-config-panel .evow-config-control {
  flex: 1 1 0 !important;
  min-width: 10rem !important;
  margin: 0 !important;
}
.evow-config-panel .evow-config-control .wrap { gap: 2px !important; }
.evow-config-panel .evow-config-control input,
.evow-config-panel .evow-config-control textarea { min-height: 1.65rem !important; }
"""

# This page can run independently of the composed dashboard.
REPLAY_CSS += SPLAT_VIEWER_FRAME_CSS


GENERATION_START_JS = """
(...values) => {
  const isExplicit = values[1] === "explicit";
  document.getElementById("novel-view-output")?.classList.add("evow-generation-spinner");
  document.getElementById("splat-viewer-output")?.classList.toggle("evow-generation-spinner", isExplicit);
  return values;
}
"""


GENERATION_FINISH_JS = """
() => {
  document.querySelectorAll(".evow-generation-spinner").forEach((element) => {
    element.classList.remove("evow-generation-spinner");
  });
}
"""

REPLAY_INIT_JS = """
() => {
  const bindControls = () => {
    document.querySelectorAll(".evow-saved-run-dropdown input").forEach((input) => {
      if (!input.value) input.setAttribute("placeholder", "Choose a saved run");
    });
    document.querySelectorAll("#replay-method-choice label").forEach((choice) => {
      if (!choice.textContent.includes("Explicit 3D: Animated Mesh")) return;
      choice.classList.add("evow-disabled-method");
      choice.setAttribute("aria-disabled", "true");
      choice.setAttribute("title", "Animated Mesh is not available yet.");
      choice.querySelectorAll("input, [role='radio']").forEach((control) => {
        control.setAttribute("aria-disabled", "true");
        control.setAttribute("tabindex", "-1");
        if ("disabled" in control) control.disabled = true;
      });
      if (choice.dataset.evowDisabledChoiceBound) return;
      choice.dataset.evowDisabledChoiceBound = "true";
      const blockChoice = (event) => {
        event.preventDefault();
        event.stopImmediatePropagation();
      };
      choice.addEventListener("click", blockChoice, true);
      choice.addEventListener("keydown", blockChoice, true);
    });
  };
  bindControls();
  new MutationObserver(bindControls).observe(document.body, { childList: true, subtree: true });
}
"""



def _display(state: dict):
    """Map a live or saved run snapshot to the visible result components."""
    models = state["models"]
    is_explicit = state["trace"]["mode"] == "explicit"
    is_splat = bool(models and models[0].endswith(".splat"))
    # Pipeline snapshots arrive before either final artifact exists. Preserve the
    # previous result behind its loading overlay until a completed run supplies a
    # replacement, including when the new request uses a different source clip.
    video_update = (
        gr.update(value=state["video"], visible=True)
        if state["video"] else gr.update()
    )
    workflow_group_updates = _workflow_group_updates(state["trace"])
    workflow_header_updates = _workflow_header_updates(state["trace"])
    workflow_card_updates = _workflow_cards(state["trace"])
    splat_update = (
        gr.update(value=splat_html(models), visible=True)
        if is_splat else
        gr.update(visible=True) if is_explicit else
        gr.update(value="", visible=False)
    )
    return (
        state["source_video"], video_update,
        gr.update(value=models[0] if models and not is_splat else None,
                  visible=bool(models) and not is_splat),
        splat_update,
        gr.update(visible=bool(models) and not is_splat, maximum=max(0, len(models) - 1), value=0),
        models, *workflow_group_updates, *workflow_header_updates, *workflow_card_updates, run_total_update(state["trace"]), state["run_id"],
        (gr.update(choices=list_saved_runs(), value=state["run_id"])
         if state["video"] else gr.update()),
        state["trace"]["mode"],
    )


def _included_sample() -> str:
    """Load the bundled locked-off trees sample into the input component."""
    if not SAMPLE_VIDEO.is_file():
        raise gr.Error("The included sample video is unavailable on this machine.")
    return str(SAMPLE_VIDEO)


def _run(
    video, mode, start, count, source_fov, yaw, shift, fov,
    fps, max_side, anyview_steps, depth_input_size, gaussian_stride, gaussian_steps,
    gaussian_lr_xyz, gaussian_lr_rotation, gaussian_lr_scale, gaussian_lr_color,
    gaussian_lr_opacity, gaussian_lr_motion, gaussian_loss_depth, gaussian_loss_smoothness,
    gaussian_init_opacity_logit, gaussian_background_r, gaussian_background_g,
    gaussian_background_b, anyview_guidance_scale,
):
    """Start one queued run and stream each real pipeline update to the dashboard."""
    if mode == "mesh":
        raise gr.Error("Animated Mesh is reserved for a later implementation.")
    if not video:
        raise gr.Error("Choose a fixed-camera clip, the included sample, or browser camera capture first.")
    # Hand every raw control value to the typed override boundary once. Coercing
    # here would hide a bad value: int(12.5) would silently become 12 before
    # ReplayOverrides could reject it. Invalid values fail inside execute_run
    # (via overrides.validate) before any filesystem work begins.
    overrides = ReplayOverrides(
        fps=fps, max_side=max_side,
        anyview_steps=anyview_steps, depth_input_size=depth_input_size,
        gaussian_stride=gaussian_stride, gaussian_steps=gaussian_steps,
        gaussian_lr_xyz=gaussian_lr_xyz, gaussian_lr_rotation=gaussian_lr_rotation,
        gaussian_lr_scale=gaussian_lr_scale, gaussian_lr_color=gaussian_lr_color,
        gaussian_lr_opacity=gaussian_lr_opacity, gaussian_lr_motion=gaussian_lr_motion,
        gaussian_loss_depth=gaussian_loss_depth,
        gaussian_loss_smoothness=gaussian_loss_smoothness,
        gaussian_init_opacity_logit=gaussian_init_opacity_logit,
        gaussian_background_r=gaussian_background_r,
        gaussian_background_g=gaussian_background_g,
        gaussian_background_b=gaussian_background_b,
        anyview_guidance_scale=anyview_guidance_scale,
    )
    for state in execute_run(video, mode, start, int(count), source_fov, yaw, shift, fov,
                             overrides=overrides):
        yield _display(state)


def _load(run_id):
    """Restore a completed run and make its sampled episode the next input."""
    try:
        state = load_saved_run(run_id)
        state["message"] = (
            "Saved run loaded. The left input now contains its recorded episode. "
            "Change any setup control and generate a new result."
        )
        display = _display(state)
        return state["source_video"], gr.update(value=0), *display
    except (OSError, ValueError, KeyError) as error:
        raise gr.Error(f"Could not load this run: {error}") from error


def _mesh(index, models):
    """Keep old mesh-only saved runs viewable without affecting splat playback."""
    if not models:
        return gr.update(value=None, visible=False), gr.update(value="", visible=False)
    selected = models[max(0, min(int(index), len(models) - 1))]
    if selected.endswith(".splat"):
        return gr.update(value=None, visible=False), gr.update(value=splat_html([selected]), visible=True)
    return gr.update(value=selected, visible=True), gr.update(value="", visible=False)


def _workflow_cards(trace: dict) -> tuple[str, ...]:
    """Render live content for the preallocated atomic workflow accordions."""
    return tuple(workflow_card_html(trace, index) for index in range(MAX_FLOW_STAGES))


def _workflow_header_updates(trace: dict) -> tuple[dict, ...]:
    """Refresh headers only, preserving each stage accordion's open state."""
    return tuple(
        gr.update(label=label, elem_classes=["evow-stage", state_class], visible=workflow_visible(trace, index))
        for index in range(MAX_FLOW_STAGES)
        for label, state_class in [workflow_header(trace, index)]
    )


# Physical ownership ranges never change, so each Accordion keeps a stable
# identity while labels explain the selected methodology's corresponding phase.
_FLOW_GROUP_INDEXES = (
    range(0, 3),
    range(3, 6),
    range(6, 9),
    range(9, 12),
    range(12, MAX_FLOW_STAGES),
)


def _workflow_group_updates(trace: dict) -> tuple[dict, ...]:
    """Split atomic rows into small method-specific phases without relocating them."""
    if trace.get("workflow_version", 1) < 2:
        return (
            gr.update(label="Legacy trace · Recorded · —", elem_classes=["evow-flow-group", "evow-flow-group-waiting"], visible=True),
            *(gr.update(label="Pipeline", elem_classes=["evow-flow-group", "evow-flow-group-waiting"], visible=False) for _ in _FLOW_GROUP_INDEXES[1:]),
        )
    if trace.get("mode", "explicit") == "implicit":
        labels = (
            "Input",
            "Episode + models",
            "Encode inputs",
            "Condition + generate",
            "Decode + finish",
        )
    else:
        labels = (
            "Input",
            "Depth prior",
            "Build 4D scene",
            "Render + export",
            "Finish",
        )
    updates = []
    for label, indexes in zip(labels, _FLOW_GROUP_INDEXES):
        state, elapsed, concurrent = workflow_group_summary(trace, indexes)
        status = {
            "complete": "Completed", "running": "Running", "waiting": "Waiting",
            "failed": "Failed", "skipped": "Skipped",
        }[state]
        if state == "running" and concurrent:
            status = "Running (concurrent/async)"
        updates.append(gr.update(
            label=f"{label} · {status} · {elapsed}",
            elem_classes=["evow-flow-group", f"evow-flow-group-{state}"],
            visible=True,
        ))
    return tuple(updates)


def _empty_workflow_cards(mode: str) -> tuple[str, ...]:
    """Render the selected method before its run begins."""
    return _workflow_cards({"workflow_version": 2, "mode": "explicit" if mode == "mesh" else mode, "events": []})


def _method_change(mode: str):
    """Refresh stage headers/details for a method and reserve its 3D panel."""
    selected = "explicit" if mode == "mesh" else mode
    is_explicit = mode in {"explicit", "mesh"}
    empty_trace = {"workflow_version": 2, "mode": selected, "events": []}
    return (*_workflow_group_updates(empty_trace),
            *_workflow_header_updates(empty_trace),
            *_empty_workflow_cards(mode),
            gr.update(value=empty_splat_html() if is_explicit else "", visible=is_explicit))


def build_app() -> gr.Blocks:
    """Build a guided page with setup on the left and results on the right."""
    with gr.Blocks(title="EVOW · 3D Video", analytics_enabled=False, css=REPLAY_CSS, js=REPLAY_INIT_JS) as app:
        gr.Markdown(
            "# 3D Video"
        )
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 1. Source")
                source, sample_button, saved, load_button = source_and_saved_controls(
                    "Source", list_saved_runs(), "ASSETS/replay/runs/", SAMPLE_VIDEO.is_file(),
                    "evow-replay-saved-runs", sample_path=str(SAMPLE_VIDEO),
                )

                gr.Markdown("## 2. Methodology " + help_icon("Choose an overall 3D representation family. Model-specific pipeline detail appears in Internal Execution Flow.", "Methodology"))
                mode = gr.Radio(
                    [("Explicit 3D: 4D Gaussian", "explicit"),
                     ("Implicit 3D: Video Diffusion", "implicit"),
                     ("Explicit 3D: Animated Mesh", "mesh")],
                    value="explicit", label=None, show_label=False, container=True, elem_classes="evow-method-choice", elem_id="replay-method-choice",
                )
                with gr.Accordion("Configuration", open=False, elem_classes="evow-config-panel"):
                    # All controls stay visible in one label/value list. This is
                    # deliberately long rather than mode-gated: users can scan,
                    # compare, and debug every effective setting in one place.
                    # Every row uses the shared app_core builder so Replay's
                    # spacing matches the other three Configuration panels.
                    start = flat_config_number("Start (s)", 0, minimum=0, detail="Seconds into the source clip where the sampled episode begins.")
                    count = config_row(
                        "Frames", "Number of sampled frames; AnyView accepts 13, 29, or 41.",
                        lambda classes: gr.Radio([13, 29, 41], value=settings.default_frames, label="Frames",
                                                 **config_control_kwargs({}, classes)),
                    )
                    source_fov = flat_config_slider("Source FOV", 70, 35, 110, step=1, detail="Assumed source camera field of view in degrees.")
                    fps = flat_config_number("FPS", settings.fps, minimum=1, maximum=60, detail="Sampling and source-preview frame rate.")
                    max_side = flat_config_number("Max side", settings.max_side, minimum=64, maximum=2160, detail="Longest encoded side in pixels.")
                    anyview_steps = flat_config_number("AnyView steps", settings.anyview_steps, minimum=1, maximum=200, detail="Diffusion denoising steps for the generated view.")
                    anyview_guidance_scale = flat_config_number("Guidance", settings.anyview_guidance_scale, minimum=0.0, maximum=100.0, step=0.1, detail="AnyView classifier-free guidance; current model requires 0.0.")
                    depth_input_size = flat_config_number("Depth input", settings.depth_input_size, minimum=112, maximum=1568, step=14, detail="Depth encoder input resolution; must be divisible by 14.")
                    gaussian_stride = flat_config_number("Gaussian stride", settings.gaussian_stride, minimum=1, maximum=64, detail="Pixel stride between seeded Gaussian centers.")
                    gaussian_steps = flat_config_number("Gaussian steps", settings.gaussian_steps, minimum=1, maximum=100000, detail="Optimizer steps for the explicit 4D fit.")
                    gaussian_lr_xyz = flat_config_number("LR · position", settings.gaussian_lr_xyz, minimum=1e-6, maximum=1.0, step=1e-5, detail="Learning rate for Gaussian center positions.")
                    gaussian_lr_rotation = flat_config_number("LR · rotation", settings.gaussian_lr_rotation, minimum=1e-6, maximum=1.0, step=1e-5, detail="Learning rate for Gaussian orientations.")
                    gaussian_lr_scale = flat_config_number("LR · scale", settings.gaussian_lr_scale, minimum=1e-6, maximum=1.0, step=1e-5, detail="Learning rate for Gaussian scales.")
                    gaussian_lr_color = flat_config_number("LR · color", settings.gaussian_lr_color, minimum=1e-6, maximum=1.0, step=1e-5, detail="Learning rate for Gaussian colors.")
                    gaussian_lr_opacity = flat_config_number("LR · opacity", settings.gaussian_lr_opacity, minimum=1e-6, maximum=1.0, step=1e-5, detail="Learning rate for Gaussian opacities.")
                    gaussian_lr_motion = flat_config_number("LR · motion", settings.gaussian_lr_motion, minimum=1e-6, maximum=1.0, step=1e-5, detail="Learning rate for per-frame deformation offsets.")
                    gaussian_loss_depth = flat_config_number("Depth weight", settings.gaussian_loss_depth, minimum=0.0, maximum=1000.0, step=0.001, detail="Weight of the depth penalty.")
                    gaussian_loss_smoothness = flat_config_number("Smooth weight", settings.gaussian_loss_smoothness, minimum=0.0, maximum=1000.0, step=0.001, detail="Weight of temporal smoothness.")
                    gaussian_init_opacity_logit = flat_config_number("Init opacity", settings.gaussian_init_opacity_logit, minimum=-20.0, maximum=20.0, step=0.1, detail="Initial Gaussian opacity logit.")
                    gaussian_background_r = flat_config_number("BG · R", settings.gaussian_background_r, minimum=0.0, maximum=1.0, step=0.01, detail="Raster background red channel.")
                    gaussian_background_g = flat_config_number("BG · G", settings.gaussian_background_g, minimum=0.0, maximum=1.0, step=0.01, detail="Raster background green channel.")
                    gaussian_background_b = flat_config_number("BG · B", settings.gaussian_background_b, minimum=0.0, maximum=1.0, step=0.01, detail="Raster background blue channel.")

                gr.Markdown("## 3. Change Camera View Pose " + help_icon(
                    "Changing the requested camera pose is worth demonstrating for both solution families, but it is especially telling for implicit methods: an explicit 3D scene can be orbited freely because its geometry is persistent, whereas an implicit video model has no persistent scene to move through, so a requested pose is the only way to reveal how it handles a viewpoint it was not recorded from.",
                    "Change Camera View Pose",
                ))
                yaw = gr.Slider(-15, 15, value=4, step=1, label="Rotate about the vertical axis (°)")
                shift = gr.Slider(-0.5, 0.5, value=0.1, step=0.05, label="Lateral translation (sideways)")
                fov = gr.Slider(35, 110, value=70, step=1, label="Output FOV")
                gr.Markdown("## 4. Reconstruct")
                run_button = gr.Button("Reconstruct", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("## 5. Output", elem_classes="evow-section-heading")
                with gr.Row():
                    sampled = gr.Video(label="Input", interactive=False, elem_classes="evow-media")
                    output = gr.Video(
                        label="Novel View from (§3)", interactive=False, elem_classes=["evow-media", "evow-output-slot"],
                        elem_id="novel-view-output", visible=True, show_download_button=True,
                    )
                model = gr.Model3D(label="Legacy depth mesh", height=440, visible=False)
                splat_viewer = gr.HTML(
                    value=empty_splat_html(), label="3D View", show_label=True, container=True, padding=False,
                    elem_classes=["evow-splat-viewer"], elem_id="splat-viewer-output", visible=True,
                )
                mesh_index = gr.Slider(0, 0, value=0, step=1, label="Legacy mesh time step", visible=False)

        workflow_groups = []
        workflow_headers = []
        workflow_cards = []
        run_total = None
        with gr.Accordion("Internal Execution Flow", open=False, elem_classes="internal-flow"):
            initial_trace = {"workflow_version": 2, "mode": "explicit", "events": []}
            initial_cards = _empty_workflow_cards("explicit")
            # Keep every atomic stage component resident. Small parent phases
            # reduce scanning noise while inner rows retain their open state.
            initial_groups = _workflow_group_updates(initial_trace)
            for group_update, indexes in zip(initial_groups, _FLOW_GROUP_INDEXES):
                group_label = group_update["label"]
                with gr.Accordion(group_label, open=False, elem_classes="evow-flow-group") as workflow_group:
                    workflow_groups.append(workflow_group)
                    for index in indexes:
                        label, state_class = workflow_header(initial_trace, index)
                        with gr.Accordion(label, open=False, visible=workflow_visible(initial_trace, index), elem_classes=["evow-stage", state_class]) as workflow_header_component:
                            workflow_headers.append(workflow_header_component)
                            workflow_cards.append(gr.HTML(value=initial_cards[index]))
            run_total = gr.Markdown("**Run total:** —", elem_classes="evow-run-total")
        model_paths = gr.State([])
        run_id = gr.State("")
        outputs = [
            sampled, output, model, splat_viewer, mesh_index, model_paths, *workflow_groups, *workflow_headers, *workflow_cards, run_total, run_id, saved, mode,
        ]
        sample_button.click(_included_sample, None, source)
        mode.change(
            _method_change, mode, [*workflow_groups, *workflow_headers, *workflow_cards, splat_viewer],
        )
        run_event = run_button.click(
            _run,
            [source, mode, start, count, source_fov, yaw, shift, fov,
             fps, max_side, anyview_steps, depth_input_size, gaussian_stride, gaussian_steps,
             gaussian_lr_xyz, gaussian_lr_rotation, gaussian_lr_scale, gaussian_lr_color,
             gaussian_lr_opacity, gaussian_lr_motion, gaussian_loss_depth, gaussian_loss_smoothness,
             gaussian_init_opacity_logit, gaussian_background_r, gaussian_background_g,
             gaussian_background_b, anyview_guidance_scale],
            outputs,
            concurrency_limit=1,
            # Gradio's generator progress animation is an orange bar between
            # output panels. The frontend callbacks instead show synchronized
            # spinners directly over the generated artifact areas.
            show_progress="hidden",
            js=GENERATION_START_JS,
        )
        run_event.then(
            fn=None, inputs=None, outputs=None, queue=False, show_progress="hidden",
            js=GENERATION_FINISH_JS,
        )
        load_outputs = [source, start, *outputs]
        load_button.change(_load, saved, load_outputs)
        mesh_index.change(_mesh, [mesh_index, model_paths], [model, splat_viewer])
    return app


def _allowed_paths() -> list[str]:
    """Serve Replay and Future View artifacts, but not arbitrary asset data.

    Gradio's file route rejects paths outside this list. Replay worked because
    its run directory was included, while Future View's splats were denied with
    HTTP 403 before the browser renderer ever saw their bytes.
    """
    return [str(settings.run_root), str(settings.assets / "future" / "runs"),
            str(settings.assets / "selection" / "runs"), str(settings.assets / "live_camera" / "captures"),
            str(settings.assets / "editing" / "runs"), str(settings.assets / "scenes"),
            str(settings.root / "GREENFIELD"), str(Path(__file__).with_name("gaussian_viewer.bundle.js"))]


def main() -> None:
    """Launch the private dashboard on the configured LAN address."""
    from GREENFIELD.dashboard import build_dashboard
    app = build_dashboard()
    app.queue(default_concurrency_limit=1)
    app.launch(
        server_name=os.environ.get("EVOW_BIND_HOST", "0.0.0.0"), server_port=7860, share=False,
        allowed_paths=_allowed_paths(), show_error=True,
    )


if __name__ == "__main__":
    main()
