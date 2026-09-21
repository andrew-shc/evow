"""Runnable Gradio pages for future view, selection, and text editing baselines."""

import cv2
import gradio as gr
import numpy as np

from .common import flow_html, read_video, run_dir, write_video


def _future(video, mode, seconds):
    frames, fps = read_video(video)
    output = []
    last = frames[-1]
    for index in range(int(seconds * fps)):
        if mode == "implicit":
            shift = int(np.sin(index / 8) * 2)
            generated = np.roll(last, shift, axis=1)
        else:
            generated = last.copy()
            cv2.putText(generated, "explicit scene-state baseline", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
        output.append(generated)
    directory = run_dir("future")
    result = write_video(output, directory / "future.mp4", fps)
    stages = [("Read recent episode", "OpenCV decodes the supplied stationary-view recording."),
              ("Advance future", f"{mode} baseline produces {seconds} seconds at {fps:.1f} fps."),
              ("Save result", result)]
    return result, flow_html("Future view", mode, stages)


def build_future() -> gr.Blocks:
    with gr.Blocks(css=".internal-flow button,.internal-flow summary{font-size:1.2rem!important;font-weight:700!important}.evow-media{border:2px solid #71869a!important;border-radius:10px!important;padding:6px!important;background:#f8fafc!important}") as page:
        gr.Markdown("# Future View")
        with gr.Row():
            gr.Markdown("## 1. Source")
            video = gr.Video(label="Source", elem_classes="evow-media")
            with gr.Column():
                gr.Markdown("## 2. Mode")
                mode = gr.Radio([("Implicit video path", "implicit"), ("Explicit scene-state path", "explicit")], value="implicit", label="Mode")
                gr.Markdown("## 3. Duration")
                seconds = gr.Slider(1, 30, value=5, step=1, label="Seconds")
                gr.Markdown("## 4. Generate")
                run = gr.Button("Generate", variant="primary")
        gr.Markdown("## 5. Output")
        result = gr.Video(label="Result", elem_classes="evow-media")
        gr.Markdown("## Internal Execution Flow")
        with gr.Accordion("Show / hide", open=False, elem_classes="internal-flow"):
            flow = gr.HTML(value=flow_html("Future view", "implicit", [("Read recent episode", "Waiting for input."), ("Advance future", "Waiting."), ("Save result", "Waiting.")]))
        run.click(_future, [video, mode, seconds], [result, flow])
    return page


def _select(video, query, mode):
    frames, fps = read_video(video)
    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
    motion = [0.0] + [float(np.mean(cv2.absdiff(gray[i], gray[i - 1]))) for i in range(1, len(gray))]
    brightness = [float(frame.mean()) for frame in gray]
    text = query.lower()
    score = motion if any(word in text for word in ("move", "wind", "tree", "motion")) else [255 - value if "dark" in text else value for value in brightness]
    ranked = sorted(range(len(frames)), key=lambda i: score[i], reverse=True)[:5]
    rows = [[f"{index / fps:.1f}s", f"{min((index + 1) / fps, len(frames) / fps):.1f}s", round(score[index], 2), mode] for index in ranked]
    stages = [("Decode archive window", f"OpenCV decoded {len(frames)} frames."), ("Score query", f"Baseline visual descriptor score for: {query}"), ("Rank intervals", f"Returned {len(rows)} intervals.")]
    return rows, flow_html("Text selection", mode, stages)


def build_selection() -> gr.Blocks:
    with gr.Blocks(css=".internal-flow button,.internal-flow summary{font-size:1.2rem!important;font-weight:700!important}.evow-media{border:2px solid #71869a!important;border-radius:10px!important;padding:6px!important;background:#f8fafc!important}") as page:
        gr.Markdown("# Text Query")
        with gr.Row():
            gr.Markdown("## 1. Source")
            video = gr.Video(label="Video", elem_classes="evow-media")
            with gr.Column():
                gr.Markdown("## 2. Text")
                query = gr.Textbox(label="Text", placeholder="dark clouds above moving branches")
                gr.Markdown("## 3. Mode")
                mode = gr.Radio([("Implicit video descriptors", "implicit"), ("Explicit scene-time descriptors", "explicit")], value="implicit", label="Mode")
                gr.Markdown("## 4. Search")
                run = gr.Button("Search", variant="primary")
        gr.Markdown("## 5. Output")
        matches = gr.Dataframe(headers=["Start", "End", "Score", "Method"], label="Matches")
        gr.Markdown("## Internal Execution Flow")
        with gr.Accordion("Show / hide", open=False, elem_classes="internal-flow"):
            flow = gr.HTML(value=flow_html("Text selection", "implicit", [("Decode archive window", "Waiting."), ("Score query", "Waiting."), ("Rank intervals", "Waiting.")]))
        run.click(_select, [video, query, mode], [matches, flow])
    return page


def _edit(video, prompt, mode):
    frames, fps = read_video(video)
    words = prompt.lower()
    edited = []
    for frame in frames:
        image = frame.astype(np.float32)
        if "bright" in words or "sun" in words:
            image = np.clip(image * 1.25, 0, 255)
        elif "dark" in words or "night" in words:
            image = np.clip(image * .55, 0, 255)
        if "warm" in words or "sunset" in words:
            image[..., 0] = np.clip(image[..., 0] * 1.2, 0, 255)
        if mode == "explicit":
            image[..., 1] = np.clip(image[..., 1] * .94, 0, 255)
        edited.append(image.astype(np.uint8))
    directory = run_dir("editing")
    result = write_video(edited, directory / "edited.mp4", fps)
    stages = [("Decode source", f"Loaded {len(frames)} RGB frames."), ("Apply text-directed edit", f"OpenCV color transform for: {prompt}"), ("Render edited view", result)]
    return result, flow_html("Text editing", mode, stages)


def build_editing() -> gr.Blocks:
    with gr.Blocks(css=".internal-flow button,.internal-flow summary{font-size:1.2rem!important;font-weight:700!important}.evow-media{border:2px solid #71869a!important;border-radius:10px!important;padding:6px!important;background:#f8fafc!important}") as page:
        gr.Markdown("# Text Manipulation")
        with gr.Row():
            gr.Markdown("## 1. Source")
            video = gr.Video(label="Video", elem_classes="evow-media")
            with gr.Column():
                gr.Markdown("## 2. Instruction")
                prompt = gr.Textbox(label="Instruction", placeholder="make the scene warmer and brighter")
                gr.Markdown("## 3. Mode")
                mode = gr.Radio([("Implicit video edit", "implicit"), ("Explicit scene render edit", "explicit")], value="implicit", label="Mode")
                gr.Markdown("## 4. Apply")
                run = gr.Button("Apply", variant="primary")
        gr.Markdown("## 5. Output")
        result = gr.Video(label="Result", elem_classes="evow-media")
        gr.Markdown("## Internal Execution Flow")
        with gr.Accordion("Show / hide", open=False, elem_classes="internal-flow"):
            flow = gr.HTML(value=flow_html("Text editing", "implicit", [("Decode source", "Waiting."), ("Apply text-directed edit", "Waiting."), ("Render edited view", "Waiting.")]))
        run.click(_edit, [video, prompt, mode], [result, flow])
    return page
