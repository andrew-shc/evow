"""Replay-derived visual tokens and reusable dashboard controls."""

from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import quote

import gradio as gr

from .live_camera import LiveCamera
from .contracts import StageSpec
from .splat_viewer import SPLAT_VIEWER_FRAME_CSS
from .timing import format_stopwatch
_LIVE_CAMERA = LiveCamera(Path(__file__).resolve().parents[2] / "ASSETS")
from .model_catalog import models_for


APP_CSS = """
html, body { overflow-x: hidden !important; }.evow-section-heading h2 { font-size: 1.5rem !important; font-weight: 700 !important; line-height: 1.35 !important; }.evow-flow-heading h2 { color: #7e22ce !important; }.evow-method-choice .wrap { flex-direction: column !important; align-items: flex-start !important; gap: 6px !important; }.evow-method-choice label { width: 100%; box-shadow: none !important; border: 0 !important; border-radius: 0 !important; background: transparent !important; padding: 4px 0 !important; }.evow-method-choice label:hover, .evow-method-choice label.selected { background: transparent !important; }.evow-media { outline: 2px solid #58748d !important; outline-offset: -2px !important; border-radius: 10px !important; background: #f8fafc !important; overflow: hidden !important; }.evow-media > div { border: 0 !important; }.internal-flow > div > .column { gap: 6px !important; }.evow-stage { box-shadow: none !important; border: 0 !important; border-radius: 0 !important; background: transparent !important; }.evow-stage > button { min-height: 2.7rem; padding: 9px 11px !important; border: 1px solid #cbd5df !important; border-left: 5px solid #94a3b8 !important; border-radius: 7px !important; background: #f8fafc !important; text-align: left !important; }.evow-stage-complete > button { border-left-color: #15803d !important; background: #f0fdf4 !important; color: #166534 !important; }.evow-stage-running > button { border-left-color: #0369a1 !important; background: #f0f9ff !important; color: #075985 !important; }.evow-stage-error > button { border-left-color: #b91c1c !important; background: #fff1f2 !important; color: #991b1b !important; }.evow-stage-skipped > button { border-left-color: #b45309 !important; background: #fffbeb !important; color: #92400e !important; }

.evow-media { min-width:0 !important; }.evow-media video, .evow-media img, .evow-media canvas, .evow-media iframe { display:block !important; width:100% !important; max-width:100% !important; height:auto !important; object-fit:contain !important; }.evow-output-slot { min-height:240px; contain:layout paint; }.evow-output-slot iframe { min-height:480px !important; }.evow-help-component { display:inline-flex !important; width:auto !important; min-height:0 !important; margin:0 !important; padding:0 !important; }/* .evow-help icon styling now lives in dashboard.py's FLOW_HEADER_CSS: this nested Blocks css is dropped when the dashboard merges pages with render(). */.evow-run-total { margin:10px 0 2px; color:#425466; font-variant-numeric:tabular-nums; }
"""

# Standalone feature pages need the shared floating 3D frame label as well.
APP_CSS += SPLAT_VIEWER_FRAME_CSS

SAVED_RUN_PLACEHOLDER_JS = """
() => {
  const setSavedRunPlaceholders = () => {
    document.querySelectorAll(".evow-saved-run-dropdown input").forEach((input) => {
      if (!input.value) input.setAttribute("placeholder", "Choose a saved run");
    });
  };
  setSavedRunPlaceholders();
  new MutationObserver(setSavedRunPlaceholders).observe(document.body, {
    childList: true,
    subtree: true,
  });
}
"""

def help_icon(text: str, label: str = "More information") -> str:
    """Return an inline icon whose long-form text is pinned by a dashboard click popover."""
    return "<span class=\"evow-help\" tabindex=\"0\" title=\"{}\" aria-label=\"{}\" role=\"button\">ⓘ</span>".format(escape(text, quote=True), escape(label, quote=True))


def config_caption(title: str, detail: str):
    """Compact config-group heading.

    Renders a small caption with the group name and an ⓘ tooltip holding the
    verbose explanation. This keeps control labels short (the reason this
    helper exists) while still documenting what each group does.
    """
    # The help icon rides inside the caption so the explanation stays in the
    # shared popover instead of becoming a standing paragraph per group.
    return gr.HTML(
        '<div class="evow-config-caption">{} {}</div>'.format(escape(title), help_icon(detail)),
        container=False,
    )


# Marks a config control so the shared root CSS can hide Gradio's standing info
# paragraph and the shared root JS can copy that text onto the entry box as a
# hover tooltip. The name is part of the contract with dashboard.py; keep it.
CONFIG_FIELD_CLASS = "evow-config-field"


def _config_field_kwargs(kwargs: dict, detail: str) -> dict:
    """Apply the shared tooltip contract to a control factory's kwargs.

    The description rides on Gradio's ``info`` prop so it reaches the browser,
    where dashboard.py hides the visible paragraph and mirrors it onto the real
    input as a native ``title`` tooltip.

    Gradio 5.50 ignores ``show_label`` whenever ``container=False`` and renders
    the short label with an ``sr-only hide`` class, which is exactly how the knobs
    silently became unlabeled boxes. The compact look therefore cannot come from
    ``container=False``; instead we keep ``container=True`` so the label is drawn
    and rely on the ``.evow-config-panel`` CSS to tighten padding and type. The
    short label must stay visible, so the root CSS hides only the long info
    paragraph (``.md.prose``), never Gradio's ``block-info`` label.

    Caller ``elem_classes`` are preserved; the marker class is appended once,
    never duplicated.
    """
    supplied = kwargs.pop("elem_classes", None) or []
    if isinstance(supplied, str):
        supplied = [supplied]
    classes = list(supplied)
    if CONFIG_FIELD_CLASS not in classes:
        classes.append(CONFIG_FIELD_CLASS)
    # The factory owns these four; drop caller copies so **kwargs cannot collide.
    kwargs.pop("info", None)
    kwargs.pop("container", None)
    kwargs.pop("show_label", None)
    return {**kwargs, "info": detail or None, "container": True, "show_label": True, "elem_classes": classes}


def config_number(label: str, value, minimum=None, maximum=None, step=1, detail: str = "", **kwargs):
    """Short-labelled numeric config control whose ``detail`` becomes a tooltip."""
    return gr.Number(
        label=label, value=value, minimum=minimum, maximum=maximum,
        step=step, **_config_field_kwargs(kwargs, detail),
    )


def config_slider(label: str, value, minimum, maximum, step=1, detail: str = "", **kwargs):
    """Short-labelled slider config control whose ``detail`` becomes a tooltip."""
    return gr.Slider(
        label=label, value=value, minimum=minimum, maximum=maximum,
        step=step, **_config_field_kwargs(kwargs, detail),
    )


def config_textbox(label: str, value, detail: str = "", **kwargs):
    """Short-labelled text config control whose ``detail`` becomes a tooltip."""
    return gr.Textbox(
        label=label, value=value, **_config_field_kwargs(kwargs, detail),
    )



# The one compact "one setting per row" contract every Configuration accordion
# (Replay, Future View, Text Query, Text Manipulation) shares. The same three
# class names are the contract dashboard.py's FLOW_HEADER_CSS styles and the
# tests assert, so all four panels render from identical selectors.
CONFIG_ROW_CLASS = "evow-config-row"
CONFIG_LABEL_CLASS = "evow-config-label"
CONFIG_CONTROL_CLASS = "evow-config-control"


def flat_config_label(label: str, detail: str, *, elem_classes: str = CONFIG_LABEL_CLASS) -> gr.Markdown:
    """Render the visible half of one compact label/control configuration row."""
    return gr.Markdown(
        f"{escape(label)} {help_icon(detail, label)}",
        container=False,
        elem_classes=elem_classes,
    )


def config_control_kwargs(kwargs: dict, classes: tuple[str, ...]) -> dict:
    """Return kwargs that make a row's control bare but still self-contained.

    ``config_row`` calls the control factory with the merged marker classes, so
    this helper folds the caller's own ``elem_classes`` in and pins the compact
    contract: ``container=False``/``show_label=False`` keeps Gradio from drawing
    a second visible label (the adjacent Markdown label already owns the name
    and ⓘ popover). The native wrapper stays in the DOM, keeping the input
    keyboard- and pointer-reachable; CSS never hides it.
    """
    supplied = kwargs.pop("elem_classes", None) or []
    if isinstance(supplied, str):
        supplied = [supplied]
    kwargs.pop("container", None)
    kwargs.pop("show_label", None)
    kwargs.pop("info", None)
    return {
        **kwargs,
        "container": False,
        "show_label": False,
        "scale": 2,
        "min_width": 0,
        "elem_classes": [*classes, *supplied],
    }


def config_row(label: str, detail: str, build_control, *, row_classes=(), control_classes=()):
    """Build one compact label/control row and return the control it created.

    ``build_control`` receives the merged control classes so a non-numeric
    control (e.g. ``gr.Radio``) participates in the exact same row contract as
    the numeric helpers: it is called inside the row and its component returned.
    """
    with gr.Row(elem_classes=[CONFIG_ROW_CLASS, *row_classes]):
        flat_config_label(label, detail, elem_classes=CONFIG_LABEL_CLASS)
        return build_control((CONFIG_CONTROL_CLASS, *control_classes))


def flat_config_number(label: str, value, minimum=None, maximum=None, step=1, detail: str = "", **kwargs):
    """Add one editable numeric label/value row with its explanation in ⓘ."""
    return config_row(
        label, detail,
        lambda classes: gr.Number(
            label=label, value=value, minimum=minimum, maximum=maximum, step=step,
            **config_control_kwargs(kwargs, classes),
        ),
    )


def flat_config_slider(label: str, value, minimum, maximum, step=1, detail: str = "", **kwargs):
    """Add one editable slider label/value row with its explanation in ⓘ."""
    return config_row(
        label, detail,
        lambda classes: gr.Slider(
            label=label, value=value, minimum=minimum, maximum=maximum, step=step,
            **config_control_kwargs(kwargs, classes),
        ),
    )


def flat_config_textbox(label: str, value, detail: str = "", **kwargs):
    """Add one editable text label/value row with its explanation in ⓘ."""
    return config_row(
        label, detail,
        lambda classes: gr.Textbox(
            label=label, value=value, **config_control_kwargs(kwargs, classes),
        ),
    )

def run_total_update(trace: dict, notice: str = "") -> dict:
    """Render one page-level total from the latest persisted trace event.

    ``notice`` prefixes the markdown when a saved run's workflow identity could
    not be resolved cleanly (for example an unknown future version), so the
    warning travels with the render without changing any call-site arity.
    """
    events = trace.get("events", [])
    elapsed = float(events[-1].get("elapsed_seconds", 0)) if events else None
    value = "—" if elapsed is None else format_stopwatch(elapsed)
    markdown = f"**Run total:** {value}"
    return gr.update(value=f"**{notice}**\n\n{markdown}" if notice else markdown)


def methodology_html(feature: str, mode: str) -> str:
    """Render selected model provenance before a run starts, including primary links."""
    title = "Explicit 3D" if mode == "explicit" else "Implicit video"
    rows = "".join(
        "<li><a href=\"{}\" target=\"_blank\" rel=\"noopener\">{}</a> <code>{}</code> — {}</li>".format(
            escape(model.url, quote=True), escape(model.name), escape(model.identifier), escape(model.role))
        for model in models_for(feature, mode)
    )
    return ("<section class=\"evow-method-models\"><strong>Selected methodology: {}</strong>"
            "<p>Models used by this path (links open their primary project/model pages):</p><ul>{}</ul></section>"
            .format(title, rows))


def _source_line(spec: StageSpec) -> int | None:
    path = Path(__file__).resolve().parents[2] / spec.source_file
    if not spec.source_file or not spec.source_location or not path.is_file():
        return None
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if spec.source_location in line:
            return number
    return None


def _live_visual(events: list[dict]) -> str:
    """Draw a truthful metric bar when an adapter supplies a completed/total pair."""
    metrics = (events[-1].get("metrics") or {}) if events else {}
    pairs = (("generated_frames", "total_frames"), ("embedded_frames", "total_frames"), ("step", "total"))
    for completed_key, total_key in pairs:
        if completed_key in metrics and total_key in metrics and float(metrics[total_key]) > 0:
            completed, total = float(metrics[completed_key]), float(metrics[total_key])
            percent = max(0, min(100, completed * 100 / total))
            return (f'<div class="evow-live-visual" aria-label="{escape(completed_key)} progress">'
                    f'<div class="evow-live-bar"><i style="width:{percent:.2f}%"></i></div>'
                    f'<span>{completed:g} / {total:g} · {percent:.1f}%</span></div>')
    return '<div class="evow-live-visual evow-live-pulse"><i></i><span>Working — awaiting the next recorded metric tick</span></div>' if events and events[-1].get("status") == "running" else ""



def _readable_value(value: object, key: str | None = None) -> str:
    """Render persisted scalar values without exposing their JSON representation."""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    # Worker elapsed metrics remain numeric in the trace for downstream
    # consumers, but they need the same fixed-width execution-flow display.
    if (
        key
        and (key == "elapsed_seconds" or (key.startswith("elapsed_") and key.endswith("_seconds")))
        and isinstance(value, (int, float))
    ):
        return format_stopwatch(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(_readable_value(item) for item in value)
    return str(value)


def detail_rows(values: dict[str, object] | tuple[tuple[str, object], ...]) -> str:
    """Present implementation facts and runtime metrics as compact labeled rows."""
    items = values.items() if isinstance(values, dict) else values
    rows = "".join(
        f"<div><dt>{escape(key.replace('_', ' ').title())}</dt><dd>{escape(_readable_value(value, key))}</dd></div>"
        for key, value in items if value is not None
    )
    return f'<dl class="evow-kv">{rows}</dl>' if rows else ""
def _nested_ticks(events: list[dict]) -> str:
    """Expose every adapter emission as a second-level, independently expandable step."""
    if not events:
        return ""
    rows = []
    for number, item in enumerate(events, 1):
        elapsed = float(item.get("stage_elapsed_seconds", item.get("elapsed_seconds", 0)))
        # A skipped tick did no work, so show a dash instead of a fabricated duration.
        elapsed_label = "—" if item.get("status") == "skipped" else format_stopwatch(elapsed)
        rows.append('<details class="evow-tick"{}><summary>Tick {} · {} · {}</summary><p>{}</p>{}</details>'.format(
            " open" if number == len(events) else "", number, escape(str(item.get("status", "waiting"))), elapsed_label,
            escape(str(item.get("detail", ""))), detail_rows(item.get("metrics") or {})))
    return '<h4>Nested execution ticks</h4><div class="evow-ticks">' + "".join(rows) + '</div>'

def stage_html(spec: StageSpec, event: dict | None = None, stage_started_at: float | None = None,
               events: list[dict] | None = None) -> str:
    """Render Replay's Program / Live run record layout for any feature stage."""
    line = _source_line(spec)
    source = f"{spec.source_file}:{line}" if line else spec.source_file or "Not recorded"
    source_url = "/gradio_api/file=" + quote(str(Path(__file__).resolve().parents[2] / spec.source_file), safe="/") + "#:~:text=" + quote(spec.source_location, safe="") if line else "#"
    total_elapsed = float(event.get("elapsed_seconds", 0)) if event else 0
    # Old saved runs do not contain the persisted stage clock, so continue to
    # derive it from their first event. New traces never show total job time in
    # every stage card: it belongs in the page-level run summary instead.
    stage_elapsed = (float(event["stage_elapsed_seconds"]) if event and "stage_elapsed_seconds" in event
                     else max(0, total_elapsed - stage_started_at) if stage_started_at is not None else total_elapsed)
    detail = event.get("detail", "Waiting for this operation.") if event else "Waiting for this operation."
    worker_log = ((event.get("metrics") or {}).get("log_tail")) if event else None
    # Every persisted emission is visible; progress polls are no longer hidden
    # behind the latest completion event.
    ticks = events or ([] if event is None else [event])
    tick_html = _live_visual(ticks) + _nested_ticks(ticks)
    worker_log_html = "<h4>Worker log</h4><pre>" + escape(str(worker_log)) + "</pre>" if worker_log else ""
    models_html = "".join(
        f'<li><a href="{escape(model.url, quote=True)}" target="_blank" rel="noopener">{escape(model.name)}</a> '
        f'<code>{escape(model.identifier)}</code> — {escape(model.role)}</li>' for model in spec.model_refs)
    program_rows = detail_rows((("Inputs", spec.inputs), ("Outputs", spec.outputs), ("Editable control", spec.controls)))
    metrics = (event.get("metrics") or {}) if event else {}
    live_rows = detail_rows({key: value for key, value in metrics.items() if key != "log_tail"})
    if event is None:
        elapsed_label = "Not started"
    elif event.get("status") == "skipped":
        # A skipped stage did no work, so show a dash instead of a fabricated duration.
        elapsed_label = "—"
    else:
        elapsed_label = format_stopwatch(stage_elapsed)
    return f'''<article class="evow-stage-card"><style>.evow-stage-card{{color:#253646;line-height:1.45}}.evow-stage-purpose{{margin:0 0 13px;color:#526575;font-size:12px}}.evow-stage-detail{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.evow-stage-detail section{{min-width:0}}.evow-stage-detail h4{{margin:0 0 7px;font-size:12px;color:#425466;text-transform:uppercase;letter-spacing:.05em}}.evow-stage-detail p{{margin:0 0 8px}}.evow-stage-detail code{{color:#173f59;white-space:normal;overflow-wrap:anywhere;font-weight:600}}.evow-stage-detail pre{{max-height:260px;margin:0;overflow:auto;padding:9px;border:1px solid #d5dfe7;border-radius:5px;background:#f6f9fb;color:#17212b;white-space:pre-wrap;font:12px ui-monospace,SFMono-Regular,monospace}}.evow-code-link{{color:#0b5e96;text-decoration:underline;font:12px ui-monospace,SFMono-Regular,monospace}}.evow-kv{{display:grid;gap:6px;margin:0}}.evow-kv div{{display:grid;grid-template-columns:minmax(110px,35%) 1fr;gap:8px}}.evow-kv dt{{color:#526575;font-size:12px}}.evow-kv dd{{margin:0;overflow-wrap:anywhere}}.evow-live-visual{{margin:8px 0;padding:8px;border:1px solid #bfdbfe;border-radius:6px;background:#f0f9ff;font-size:12px}}.evow-live-bar{{height:7px;overflow:hidden;border-radius:99px;background:#dbeafe}}.evow-live-bar i{{display:block;height:100%;background:linear-gradient(90deg,#0284c7,#22c55e);transition:width .25s ease}}.evow-live-visual span{{display:block;margin-top:5px}}.evow-live-pulse i{{display:inline-block;width:8px;height:8px;margin-right:6px;border-radius:50%;background:#0284c7;animation:evow-pulse 1s infinite}}.evow-tick{{margin:5px 0;border-left:2px solid #93c5fd;padding-left:8px;font-size:12px}}.evow-tick summary{{cursor:pointer}}@keyframes evow-pulse{{50%{{opacity:.25}}}}@media(max-width:760px){{.evow-stage-detail{{grid-template-columns:1fr}}.evow-kv div{{grid-template-columns:1fr}}}}</style><p class="evow-stage-purpose">Stage purpose {help_icon(spec.purpose, "Stage purpose")}</p><div class="evow-stage-detail"><section><h4>Program</h4><p><a class="evow-code-link" href="{escape(source_url, quote=True)}" target="_blank" rel="noopener">{escape(source)}</a></p><p><code>{escape(spec.function_chain)}</code></p>{'<h4>Selected models</h4><ul>' + models_html + '</ul>' if models_html else ''}{program_rows}</section><section><h4>Live run record</h4><p>{escape(str(detail))}</p><dl class="evow-kv"><div><dt>Stage time</dt><dd>{elapsed_label}</dd></div></dl>{live_rows}{worker_log_html}{tick_html}</section></div></article>'''


def stage_label(index: int, spec: StageSpec, event: dict | None = None, stage_started_at: float | None = None, *, display_override: str | None = None) -> tuple[str, str]:
    status = "waiting" if event is None else str(event.get("status", "waiting"))
    # ``display_override`` lets a renderer derive a truthful label (e.g.
    # "Interrupted") while still driving CSS from the event's own status.
    display = display_override or {"complete": "Completed", "running": "Running", "error": "Error", "waiting": "Waiting", "skipped": "Skipped"}.get(status, "Waiting")
    elapsed_seconds = (float(event["stage_elapsed_seconds"])
                       if event and "stage_elapsed_seconds" in event
                       else max(0, float(event.get("elapsed_seconds", 0)) - (stage_started_at or 0)) if event else None)
    # A skipped stage did no work, so it shows a dash instead of a fabricated duration.
    elapsed = "—" if status == "skipped" or elapsed_seconds is None else format_stopwatch(elapsed_seconds)
    css_status = "failed" if status == "error" else status
    return f"{index + 1}. {spec.name} · {display} · {elapsed}", f"evow-stage-{css_status}"



def source_and_saved_controls(
    source_label: str,
    saved_choices: list[tuple[str, str]],
    registry: str,
    sample_available: bool,
    saved_id: str | None = None,
    sample_path: str | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Offer one source menu, including a saved-run route that opens its picker.

    The saved picker remains subordinate to the main source choice, but its
    native menu is focused and opened as soon as that route becomes visible.
    This prevents a user from reaching an empty, read-only source field first.
    """
    default_mode = "sample" if sample_available and sample_path else "video_file"
    selector = gr.Dropdown(
        [("Sample video", "sample"), ("Video file", "video_file"), ("Webcam", "webcam"), ("USB camera", "usb_camera"), ("Saved run", "saved")],
        value=default_mode, label=None, show_label=False, container=False,
    )
    source = gr.Video(value=sample_path if default_mode == "sample" else None, label="Sample video" if default_mode == "sample" else "Video file", sources=["upload"], format="mp4", elem_classes="evow-media")
    sample = gr.Button("Use Sample", interactive=sample_available, visible=False)
    webcam = gr.Video(label="Webcam", sources=["webcam"], format="mp4", visible=False, elem_classes="evow-media")
    webcam.change(lambda path: path, webcam, source, queue=False, show_progress="hidden")
    with gr.Column(visible=False) as live_panel:
        with gr.Row(elem_classes="evow-inline-help"):
            device = gr.Dropdown(choices=_LIVE_CAMERA.choices(), label=None, show_label=False, container=False, scale=12, min_width=0)
            gr.Markdown(help_icon("Locally attached cameras appear here automatically.", "Camera"), container=False, padding=False, elem_classes="evow-inline-help-icon")
        refresh_devices = gr.Button("Refresh cameras", size="sm")
        preview = gr.Image(label="Live preview", type="numpy", interactive=False, elem_classes="evow-media")
        duration = gr.Slider(2, 30, value=8, step=1, label="Capture seconds")
        capture = gr.Button("Capture and use this clip")
        usb_clip = gr.Video(label="USB camera clip", interactive=False, visible=False, elem_classes="evow-media")
        status = gr.Markdown("Select a camera to start its preview.", elem_classes="evow-clip-caption")
        refresh_devices.click(lambda: gr.update(choices=_LIVE_CAMERA.choices()), None, device, queue=False, show_progress="hidden")
        device.change(_LIVE_CAMERA.preview, device, [preview, status], queue=False, show_progress="hidden")
        def capture_camera(selected_device: str | None, seconds: float):
            path, message = _LIVE_CAMERA.capture(selected_device, seconds)
            return gr.update(value=path, visible=False), gr.update(value=path, visible=bool(path)), message
        recording = capture.click(lambda _device, _seconds: gr.update(value="**Recording camera clip…**"), [device, duration], status, queue=False, show_progress="hidden")
        recording.then(capture_camera, [device, duration], [source, usb_clip, status], show_progress="hidden")
        timer = gr.Timer(1.0)
        timer.tick(lambda selected_device: _LIVE_CAMERA.preview(selected_device)[0], device, preview, queue=False, show_progress="hidden")
    with gr.Column(visible=False) as saved_panel:
        with gr.Row(elem_classes="evow-inline-help"):
            # Gradio otherwise selects the first (newest) choice implicitly,
            # making it impossible to trigger a load by choosing that run first.
            saved = gr.Dropdown(
                choices=saved_choices,
                value=None,
                # Gradio's installed Dropdown has no placeholder argument.
                # The root UI hook supplies this exact visible empty-state text.
                label="Choose a saved run",
                show_label=False,
                container=False,
                elem_id=saved_id,
                elem_classes="evow-saved-run-dropdown",
                scale=12,
                min_width=0,
            )
            gr.Markdown(help_icon(f"Choose a completed run from `{registry}`. Its source remains immutable.", "Saved run"), container=False, padding=False, elem_classes="evow-inline-help-icon")
    def choose_source(mode: str):
        if mode == "sample":
            source_update = gr.update(value=sample_path if sample_available else None, visible=True, label="Sample video", interactive=False)
        elif mode == "video_file":
            source_update = gr.update(value=None, visible=True, label="Video file", interactive=True)
        elif mode == "saved":
            source_update = gr.update(value=None, visible=True, label="Saved source", interactive=False)
        else:
            source_update = gr.update(value=None, visible=False)
        return source_update, gr.update(visible=mode == "webcam"), gr.update(visible=mode == "usb_camera"), gr.update(visible=mode == "saved")

    # The client hook starts before the server has applied the visibility update,
    # so retry briefly across render frames until the newly shown picker exists.
    open_saved_picker_js = None
    if saved_id:
        open_saved_picker_js = f"""(mode) => {{
          if (mode === "saved") {{
            let attempts = 0;
            const openPicker = () => {{
              const input = document.getElementById({saved_id!r})?.querySelector("input");
              if (input && input.offsetParent) {{ input.focus(); input.click(); return; }}
              if (++attempts < 20) window.setTimeout(openPicker, 25);
            }};
            window.setTimeout(openPicker, 0);
          }}
          return [mode];
        }}"""
    selector.change(
        choose_source, selector, [source, webcam, live_panel, saved_panel],
        queue=False, show_progress="hidden", js=open_saved_picker_js,
    )
    return source, sample, saved, saved
