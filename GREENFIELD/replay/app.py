"""LAN Gradio dashboard for fixed-camera 4D replay and its execution flow."""

import os
from pathlib import Path

from .settings import load_settings

# Keep Gradio's upload copies inside the ignored data tree.
settings = load_settings()
os.environ.setdefault("GRADIO_TEMP_DIR", str(settings.temp_root))

import gradio as gr

from .run import execute_run, list_saved_runs, load_saved_run
from .splat_viewer import empty_splat_html, splat_html
from .workflow import workflow_card_html, workflow_labels


SAMPLE_VIDEO = settings.assets / "replay" / "samples" / "trees_swaying_pexels_12644693.mp4"
REPLAY_CSS = """
.evow-section-heading h2 { font-size: 1.5rem !important; font-weight: 700 !important; line-height: 1.35 !important; }
.evow-media { border: 2px solid #71869a !important; border-radius: 10px !important; padding: 6px !important; background: #f8fafc !important; }
#novel-view-output, #splat-viewer-output { position: relative; overflow: hidden; }
.evow-generation-spinner::after { content: ""; position: absolute; inset: 0; z-index: 10; background: rgb(248 250 252 / 72%); pointer-events: all; }
.evow-generation-spinner::before { content: ""; position: absolute; top: 50%; left: 50%; z-index: 11; width: 28px; height: 28px; margin: -14px; border: 3px solid #628197; border-top-color: #102a43; border-radius: 50%; animation: evow-generation-spin .8s linear infinite; }
@keyframes evow-generation-spin { to { transform: rotate(360deg); } }
"""


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
    workflow_updates = _workflow_cards(state["trace"])
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
        models, *workflow_updates, state["run_id"],
        (gr.update(choices=list_saved_runs(), value=state["run_id"])
         if state["video"] else gr.update()),
        state["trace"]["mode"],
    )


def _included_sample() -> str:
    """Load the bundled locked-off trees sample into the input component."""
    if not SAMPLE_VIDEO.is_file():
        raise gr.Error("The included sample video is unavailable on this machine.")
    return str(SAMPLE_VIDEO)


def _run(video, mode, start, count, source_fov, yaw, shift, fov):
    """Start one queued run and stream each real pipeline update to the dashboard."""
    if mode == "mesh":
        raise gr.Error("Animated Mesh is reserved for a later implementation.")
    if not video:
        raise gr.Error("Choose a fixed-camera clip, the included sample, or browser camera capture first.")
    for state in execute_run(video, mode, start, int(count), source_fov, yaw, shift, fov):
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
    """Render live content for the seven persistent workflow accordions."""
    return tuple(workflow_card_html(trace, index) for index in range(7))


def _empty_workflow_cards(mode: str) -> tuple[str, ...]:
    """Render the selected method before its run begins."""
    return _workflow_cards({"mode": "explicit" if mode == "mesh" else mode, "events": []})


def _method_change(mode: str):
    """Retitle persistent workflow cards and reserve the explicit 3D panel."""
    selected = "explicit" if mode == "mesh" else mode
    is_explicit = mode in {"explicit", "mesh"}
    return (*(gr.update(label=label) for label in workflow_labels(selected)),
            *_empty_workflow_cards(mode),
            gr.update(value=empty_splat_html() if is_explicit else "", visible=is_explicit))



def build_app() -> gr.Blocks:
    """Build a guided page with setup on the left and results on the right."""
    with gr.Blocks(title="EVOW · 4D replay", analytics_enabled=False, css=REPLAY_CSS) as app:
        gr.Markdown(
            "# 4D Replay"
        )
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 1. Source")
                source = gr.Video(label="Source", sources=["upload", "webcam"], format="mp4", elem_classes="evow-media")
                sample_button = gr.Button("Use Sample", interactive=SAMPLE_VIDEO.is_file())
                with gr.Accordion("Saved", open=False):
                    saved = gr.Dropdown(choices=list_saved_runs(), label="Run")
                    load_button = gr.Button("Load")

                gr.Markdown("## 2. Methodology")
                mode = gr.Radio(
                    [("Explicit 3D: 4D Gaussian", "explicit"),
                     ("Implicit 3D: Video Diffusion", "implicit"),
                     ("Explicit 3D: Animated Mesh", "mesh")],
                    value="explicit", label="Method",
                )
                with gr.Accordion("Clip", open=False):
                    start = gr.Number(value=0, minimum=0, label="Start")
                    count = gr.Radio([13, 29, 41], value=settings.default_frames, label="Frames")
                    source_fov = gr.Slider(35, 110, value=70, step=1, label="Source camera field of view")

                gr.Markdown("## 3. Different Camera View")
                yaw = gr.Slider(-15, 15, value=4, step=1, label="Turn")
                shift = gr.Slider(-0.5, 0.5, value=0.1, step=0.05, label="Sideways")
                fov = gr.Slider(35, 110, value=70, step=1, label="Field of view")
                gr.Markdown("## 4. Generate")
                run_button = gr.Button("Generate", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("## 5. Output", elem_classes="evow-section-heading")
                with gr.Row():
                    sampled = gr.Video(label="Input", interactive=False, elem_classes="evow-media")
                    output = gr.Video(
                        label="Novel View from (§3)", interactive=False, elem_classes="evow-media",
                        elem_id="novel-view-output", visible=True, show_download_button=True,
                    )
                model = gr.Model3D(label="Legacy depth mesh", height=440, visible=False)
                splat_viewer = gr.HTML(
                    value=empty_splat_html(), elem_id="splat-viewer-output", visible=True,
                )
                mesh_index = gr.Slider(0, 0, value=0, step=1, label="Legacy mesh time step", visible=False)
                gr.Markdown("## Internal Execution Flow", elem_classes="evow-section-heading evow-flow-heading")
                workflow_headers = []
                workflow_cards = []
                with gr.Accordion("", open=False, elem_classes="internal-flow") as internals:
                    for label, value in zip(workflow_labels("explicit"), _empty_workflow_cards("explicit")):
                        with gr.Accordion(label, open=False) as workflow_header:
                            workflow_headers.append(workflow_header)
                            workflow_cards.append(gr.HTML(value=value))

        model_paths = gr.State([])
        run_id = gr.State("")
        outputs = [
            sampled, output, model, splat_viewer, mesh_index, model_paths, *workflow_cards, run_id, saved, mode,
        ]
        sample_button.click(_included_sample, None, source)
        mode.change(
            _method_change, mode, [*workflow_headers, *workflow_cards, splat_viewer],
        )
        run_event = run_button.click(
            _run, [source, mode, start, count, source_fov, yaw, shift, fov], outputs,
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
        load_button.click(_load, saved, load_outputs)
        mesh_index.change(_mesh, [mesh_index, model_paths], [model, splat_viewer])
    return app


def main() -> None:
    """Launch the private dashboard on the configured LAN address."""
    from GREENFIELD.dashboard import build_dashboard
    app = build_dashboard()
    app.queue(default_concurrency_limit=1)
    app.launch(
        server_name=os.environ.get("EVOW_BIND_HOST", "0.0.0.0"), server_port=7860, share=False,
        allowed_paths=[str(settings.run_root), str(settings.root / "GREENFIELD"), str(Path(__file__).with_name("gaussian_viewer.bundle.js"))],
        show_error=True,
    )


if __name__ == "__main__":
    main()
