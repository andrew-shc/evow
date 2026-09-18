"""LAN Gradio dashboard for fixed-camera 4D replay and its execution flow."""

import os
from pathlib import Path

from .settings import load_settings

# Keep Gradio's upload copies inside the ignored data tree.
settings = load_settings()
os.environ.setdefault("GRADIO_TEMP_DIR", str(settings.temp_root))

import gradio as gr

from .run import execute_run, list_saved_runs, load_saved_run
from .splat_viewer import splat_html
from .workflow import workflow_html


SAMPLE_VIDEO = settings.assets / "replay" / "samples" / "trees_swaying_pexels_12644693.mp4"


def _display(state: dict):
    """Map a live or saved run snapshot to the visible result components."""
    models = state["models"]
    is_splat = bool(models and models[0].endswith(".splat"))
    return (
        state["source_video"], state["video"],
        gr.update(value=models[0] if models and not is_splat else None,
                  visible=bool(models) and not is_splat),
        gr.update(value=splat_html(models) if is_splat else "", visible=is_splat),
        gr.update(visible=bool(models) and not is_splat, maximum=max(0, len(models) - 1), value=0),
        models, state["message"], workflow_html(state["trace"]), state["run_id"],
        (gr.update(choices=list_saved_runs(), value=state["run_id"])
         if state["video"] else gr.update()),
        state["trace"]["mode"],
    )


def _included_sample() -> str:
    """Load the bundled locked-off trees sample into the input component."""
    if not SAMPLE_VIDEO.is_file():
        raise gr.Error("The included sample video is unavailable on this machine.")
    return str(SAMPLE_VIDEO)


def _run(video, mode, start, count, yaw, shift, fov):
    """Start one queued run and stream each real pipeline update to the dashboard."""
    if not video:
        raise gr.Error("Choose a fixed-camera clip, the included sample, or browser camera capture first.")
    for state in execute_run(video, mode, start, int(count), yaw, shift, fov):
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


def _empty_flow(mode: str) -> str:
    """Show the selected method's full plan before the first live event arrives."""
    return workflow_html({"mode": mode, "events": []})


def build_app() -> gr.Blocks:
    """Build a guided page with setup on the left and results on the right."""
    with gr.Blocks(title="EVOW · 4D replay", analytics_enabled=False) as app:
        gr.Markdown(
            "# 4D replay\n"
            "Use one stationary camera view of a changing scene. **1. Choose the source. "
            "2. Choose how to create the view. 3. Choose a nearby viewpoint. 4. Generate and inspect it.**"
        )
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 1. Source episode\nUse a recording, browser camera capture, or the included trees clip. The application assumes the physical view stays stationary.")
                source = gr.Video(label="Stationary-view source", sources=["upload", "webcam"], format="mp4")
                sample_button = gr.Button("Use included stationary trees sample", interactive=SAMPLE_VIDEO.is_file())
                with gr.Accordion("Open a saved comparison", open=False):
                    saved = gr.Dropdown(choices=list_saved_runs(), label="Saved result")
                    load_button = gr.Button("Load selected result")

                gr.Markdown("## 2. Build settings\nBoth methods receive the same short episode, so their outputs can be compared fairly.")
                mode = gr.Radio(
                    [("Explicit: native 4D Gaussian scene", "explicit"),
                     ("Implicit: video diffusion", "implicit")],
                    value="explicit", label="Representation method",
                )
                with gr.Accordion("Episode selection", open=False):
                    start = gr.Number(value=0, minimum=0, label="Clip start: seconds into source recording")
                    count = gr.Radio([13, 29, 41], value=settings.default_frames,
                                     label="Episode length: frames at 12 fps")
                    gr.Markdown("More frames show more scene motion and take longer to process.")

                gr.Markdown("## 3. Requested virtual viewpoint\nThese controls change the result camera, not the physical input camera.")
                yaw = gr.Slider(-15, 15, value=4, step=1, label="Turn view left or right (degrees)")
                shift = gr.Slider(-0.5, 0.5, value=0.1, step=0.05, label="Move view sideways (scene units)")
                fov = gr.Slider(35, 110, value=70, step=1, label="View width / field of view (degrees)")
                gr.Markdown("Keep the requested change small: a single fixed camera cannot observe hidden surfaces.")
                internals_toggle = gr.Checkbox(value=True, label="Show internal execution flow")
                run_button = gr.Button("4. Generate nearby view", variant="primary")

            with gr.Column(scale=2):
                status = gr.Markdown("Choose a source and settings, then select **Generate nearby view**. GPU jobs run one at a time.")
                with gr.Row():
                    sampled = gr.Video(label="Input episode used by both methods", interactive=False)
                    output = gr.Video(label="Generated or rendered nearby view", interactive=False)
                model = gr.Model3D(label="Legacy depth mesh", height=440, visible=False)
                splat_viewer = gr.HTML(visible=False)
                mesh_index = gr.Slider(0, 0, value=0, step=1, label="Legacy mesh time step", visible=False)
                with gr.Column(visible=True) as internals:
                    gr.Markdown("## Internal execution flow\nThis is the live route from your source to the final result. Expand a row for its real code, data, editable setting, and current record.")
                    workflow = gr.HTML(value=_empty_flow("explicit"))

        model_paths = gr.State([])
        run_id = gr.State("")
        outputs = [
            sampled, output, model, splat_viewer, mesh_index, model_paths, status, workflow, run_id, saved, mode,
        ]
        sample_button.click(_included_sample, None, source)
        internals_toggle.change(lambda enabled: gr.update(visible=enabled), internals_toggle, internals)
        mode.change(_empty_flow, mode, workflow)
        run_button.click(_run, [source, mode, start, count, yaw, shift, fov], outputs, concurrency_limit=1)
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
        server_name=os.environ.get("EVOW_BIND_HOST", "192.168.4.254"), server_port=7860, share=False,
        allowed_paths=[str(settings.run_root), str(Path(__file__).with_name("gaussian_viewer.bundle.js"))],
        show_error=True,
    )


if __name__ == "__main__":
    main()
