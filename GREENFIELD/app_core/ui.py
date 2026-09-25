"""Replay-derived visual tokens and reusable dashboard controls."""

from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import quote

import gradio as gr

from .contracts import StageSpec
from .model_catalog import models_for


APP_CSS = """
html, body { overflow-x: hidden !important; }.evow-section-heading h2 { font-size: 1.5rem !important; font-weight: 700 !important; line-height: 1.35 !important; }.evow-flow-heading h2 { color: #7e22ce !important; }.evow-method-choice .wrap { flex-direction: column !important; align-items: flex-start !important; gap: 6px !important; }.evow-method-choice label { width: 100%; box-shadow: none !important; border: 0 !important; border-radius: 0 !important; background: transparent !important; padding: 4px 0 !important; }.evow-method-choice label:hover, .evow-method-choice label.selected { background: transparent !important; }.evow-media { outline: 2px solid #58748d !important; outline-offset: -2px !important; border-radius: 10px !important; background: #f8fafc !important; overflow: hidden !important; }.evow-media > div { border: 0 !important; }.internal-flow > div > .column { gap: 6px !important; }.evow-stage { box-shadow: none !important; border: 0 !important; border-radius: 0 !important; background: transparent !important; }.evow-stage > button { min-height: 2.7rem; padding: 9px 11px !important; border: 1px solid #cbd5df !important; border-left: 5px solid #94a3b8 !important; border-radius: 7px !important; background: #f8fafc !important; text-align: left !important; }.evow-stage-complete > button { border-left-color: #15803d !important; background: #f0fdf4 !important; color: #166534 !important; }.evow-stage-running > button { border-left-color: #0369a1 !important; background: #f0f9ff !important; color: #075985 !important; }.evow-stage-error > button { border-left-color: #b91c1c !important; background: #fff1f2 !important; color: #991b1b !important; }
.evow-media { min-width:0 !important; }.evow-media video, .evow-media img, .evow-media canvas, .evow-media iframe { display:block !important; width:100% !important; max-width:100% !important; height:auto !important; object-fit:contain !important; }.evow-output-slot { min-height:240px; contain:layout paint; }.evow-output-slot iframe { min-height:480px !important; }.evow-help-component { display:inline-flex !important; width:auto !important; min-height:0 !important; margin:0 !important; padding:0 !important; }/* .evow-help icon styling now lives in dashboard.py's FLOW_HEADER_CSS: this nested Blocks css is dropped when the dashboard merges pages with render(). */.evow-run-total { margin:10px 0 2px; color:#425466; font-variant-numeric:tabular-nums; }
"""

def help_icon(text: str, label: str = "More information") -> str:
    """Return an inline icon whose long-form text is pinned by a dashboard click popover."""
    return "<span class=\"evow-help\" tabindex=\"0\" title=\"{}\" aria-label=\"{}\" role=\"button\">ⓘ</span>".format(escape(text, quote=True), escape(label, quote=True))

def run_total_update(trace: dict) -> dict:
    """Render one page-level total from the latest persisted trace event."""
    events = trace.get("events", [])
    elapsed = float(events[-1].get("elapsed_seconds", 0)) if events else None
    value = "—" if elapsed is None else f"{elapsed:.2f}s"
    return gr.update(value=f"**Run total:** {value}")


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



def _readable_value(value: object) -> str:
    """Render persisted scalar values without exposing their JSON representation."""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (list, tuple)):
        return ", ".join(_readable_value(item) for item in value)
    return str(value)


def detail_rows(values: dict[str, object] | tuple[tuple[str, object], ...]) -> str:
    """Present implementation facts and runtime metrics as compact labeled rows."""
    items = values.items() if isinstance(values, dict) else values
    rows = "".join(
        f"<div><dt>{escape(key.replace('_', ' ').title())}</dt><dd>{escape(_readable_value(value))}</dd></div>"
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
        rows.append('<details class="evow-tick"{}><summary>Tick {} · {} · {:.2f}s</summary><p>{}</p>{}</details>'.format(
            " open" if number == len(events) else "", number, escape(str(item.get("status", "waiting"))), elapsed,
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
    elapsed_label = f"{stage_elapsed:.2f}s" if event else "Not started"
    return f'''<article class="evow-stage-card"><style>.evow-stage-card{{color:#253646;line-height:1.45}}.evow-stage-purpose{{margin:0 0 13px;color:#526575;font-size:12px}}.evow-stage-detail{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.evow-stage-detail section{{min-width:0}}.evow-stage-detail h4{{margin:0 0 7px;font-size:12px;color:#425466;text-transform:uppercase;letter-spacing:.05em}}.evow-stage-detail p{{margin:0 0 8px}}.evow-stage-detail code{{color:#173f59;white-space:normal;overflow-wrap:anywhere;font-weight:600}}.evow-stage-detail pre{{max-height:260px;margin:0;overflow:auto;padding:9px;border:1px solid #d5dfe7;border-radius:5px;background:#f6f9fb;color:#17212b;white-space:pre-wrap;font:12px ui-monospace,SFMono-Regular,monospace}}.evow-code-link{{color:#0b5e96;text-decoration:underline;font:12px ui-monospace,SFMono-Regular,monospace}}.evow-kv{{display:grid;gap:6px;margin:0}}.evow-kv div{{display:grid;grid-template-columns:minmax(110px,35%) 1fr;gap:8px}}.evow-kv dt{{color:#526575;font-size:12px}}.evow-kv dd{{margin:0;overflow-wrap:anywhere}}.evow-live-visual{{margin:8px 0;padding:8px;border:1px solid #bfdbfe;border-radius:6px;background:#f0f9ff;font-size:12px}}.evow-live-bar{{height:7px;overflow:hidden;border-radius:99px;background:#dbeafe}}.evow-live-bar i{{display:block;height:100%;background:linear-gradient(90deg,#0284c7,#22c55e);transition:width .25s ease}}.evow-live-visual span{{display:block;margin-top:5px}}.evow-live-pulse i{{display:inline-block;width:8px;height:8px;margin-right:6px;border-radius:50%;background:#0284c7;animation:evow-pulse 1s infinite}}.evow-tick{{margin:5px 0;border-left:2px solid #93c5fd;padding-left:8px;font-size:12px}}.evow-tick summary{{cursor:pointer}}@keyframes evow-pulse{{50%{{opacity:.25}}}}@media(max-width:760px){{.evow-stage-detail{{grid-template-columns:1fr}}.evow-kv div{{grid-template-columns:1fr}}}}</style><p class="evow-stage-purpose">{escape(spec.purpose)}</p><div class="evow-stage-detail"><section><h4>Program</h4><p><a class="evow-code-link" href="{escape(source_url, quote=True)}" target="_blank" rel="noopener">{escape(source)}</a></p><p><code>{escape(spec.function_chain)}</code></p>{'<h4>Selected models</h4><ul>' + models_html + '</ul>' if models_html else ''}{program_rows}</section><section><h4>Live run record</h4><p>{escape(str(detail))}</p><dl class="evow-kv"><div><dt>Stage time</dt><dd>{elapsed_label}</dd></div></dl>{live_rows}{worker_log_html}{tick_html}</section></div></article>'''


def stage_label(index: int, spec: StageSpec, event: dict | None = None, stage_started_at: float | None = None) -> tuple[str, str]:
    status = "waiting" if event is None else str(event.get("status", "waiting"))
    display = {"complete": "Completed", "running": "Running", "error": "Failed", "waiting": "Waiting"}.get(status, "Waiting")
    elapsed_seconds = (float(event["stage_elapsed_seconds"])
                       if event and "stage_elapsed_seconds" in event
                       else max(0, float(event.get("elapsed_seconds", 0)) - (stage_started_at or 0)) if event else None)
    elapsed = "—" if elapsed_seconds is None else f"{elapsed_seconds:.2f}s"
    return f"{index + 1}. {spec.name} · {display} · {elapsed}", f"evow-stage-{status}"


def source_and_saved_controls(source_label: str, saved_choices: list[tuple[str, str]], registry: str, sample_available: bool, saved_id: str | None = None) -> tuple[Any, Any, Any, Any]:
    """Build the identical Replay-derived Source/Saved block for every feature."""
    source = gr.Video(label=source_label, sources=["upload", "webcam"], format="mp4", elem_classes="evow-media")
    sample = gr.Button("Use Sample", interactive=sample_available)
    with gr.Accordion("Saved", open=False):
        saved = gr.Dropdown(choices=saved_choices, label=None, show_label=False, container=False, elem_id=saved_id)
        load = gr.Button("Load")
        gr.Markdown(f"Registry: `{registry}`", elem_classes="evow-clip-caption")
    return source, sample, saved, load
