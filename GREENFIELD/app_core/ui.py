"""Replay-derived visual tokens and reusable dashboard controls."""

from html import escape
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

import gradio as gr

from .contracts import StageSpec


APP_CSS = """
html, body { overflow-x: hidden !important; }.evow-section-heading h2 { font-size: 1.5rem !important; font-weight: 700 !important; line-height: 1.35 !important; }.evow-flow-heading h2 { color: #7e22ce !important; }.evow-method-choice .wrap { flex-direction: column !important; align-items: flex-start !important; gap: 6px !important; }.evow-method-choice label { width: 100%; box-shadow: none !important; border: 0 !important; border-radius: 0 !important; background: transparent !important; padding: 4px 0 !important; }.evow-method-choice label:hover, .evow-method-choice label.selected { background: transparent !important; }.evow-media { outline: 2px solid #58748d !important; outline-offset: -2px !important; border-radius: 10px !important; background: #f8fafc !important; overflow: hidden !important; }.evow-media > div { border: 0 !important; }.internal-flow > div > .column { gap: 6px !important; }.evow-stage { box-shadow: none !important; border: 0 !important; border-radius: 0 !important; background: transparent !important; }.evow-stage > button { min-height: 2.7rem; padding: 9px 11px !important; border: 1px solid #cbd5df !important; border-left: 5px solid #94a3b8 !important; border-radius: 7px !important; background: #f8fafc !important; text-align: left !important; }.evow-stage-complete > button { border-left-color: #15803d !important; background: #f0fdf4 !important; color: #166534 !important; }.evow-stage-running > button { border-left-color: #0369a1 !important; background: #f0f9ff !important; color: #075985 !important; }.evow-stage-error > button { border-left-color: #b91c1c !important; background: #fff1f2 !important; color: #991b1b !important; }
"""


def _source_line(spec: StageSpec) -> int | None:
    path = Path(__file__).resolve().parents[2] / spec.source_file
    if not spec.source_file or not spec.source_location or not path.is_file():
        return None
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if spec.source_location in line:
            return number
    return None


def stage_html(spec: StageSpec, event: dict | None = None, stage_started_at: float | None = None) -> str:
    """Render Replay's Program / Live run record layout for any feature stage."""
    line = _source_line(spec)
    source = f"{spec.source_file}:{line}" if line else spec.source_file or "Not recorded"
    source_url = "/gradio_api/file=" + quote(str(Path(__file__).resolve().parents[2] / spec.source_file), safe="/") + "#:~:text=" + quote(spec.source_location, safe="") if line else "#"
    contract = {"source_file": source, "source_location": spec.source_location, "function_chain": spec.function_chain, "inputs": spec.inputs, "outputs": spec.outputs, "editable_control": spec.controls}
    total_elapsed = float(event.get("elapsed_seconds", 0)) if event else 0
    # Old saved runs do not contain the persisted stage clock, so continue to
    # derive it from their first event. New traces never show total job time in
    # every stage card: it belongs in the page-level run summary instead.
    stage_elapsed = (float(event["stage_elapsed_seconds"]) if event and "stage_elapsed_seconds" in event
                     else max(0, total_elapsed - stage_started_at) if stage_started_at is not None else total_elapsed)
    runtime = {"status": event.get("status", "waiting") if event else "waiting", "stage_elapsed_seconds": round(stage_elapsed, 2), "metrics": event.get("metrics") if event else None, "event": event}
    detail = event.get("detail", "Waiting for this operation.") if event else "Waiting for this operation."
    worker_log = ((event.get("metrics") or {}).get("log_tail")) if event else None
    worker_log_html = "<h4>Worker log</h4><pre>" + escape(str(worker_log)) + "</pre>" if worker_log else ""
    return f'''<article class="evow-stage-card"><style>.evow-stage-card{{color:#253646;line-height:1.45}}.evow-stage-purpose{{margin:0 0 13px;color:#526575;font-size:12px}}.evow-stage-detail{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.evow-stage-detail section{{min-width:0}}.evow-stage-detail h4{{margin:0 0 7px;font-size:12px;color:#425466;text-transform:uppercase;letter-spacing:.05em}}.evow-stage-detail p{{margin:0 0 8px}}.evow-stage-detail code{{color:#173f59;white-space:normal;overflow-wrap:anywhere;font-weight:600}}.evow-stage-detail pre{{max-height:260px;margin:0;overflow:auto;padding:9px;border:1px solid #d5dfe7;border-radius:5px;background:#f6f9fb;color:#17212b;white-space:pre-wrap;font:12px ui-monospace,SFMono-Regular,monospace}}.evow-code-link{{color:#0b5e96;text-decoration:underline;font:12px ui-monospace,SFMono-Regular,monospace}}@media(max-width:760px){{.evow-stage-detail{{grid-template-columns:1fr}}}}</style><p class="evow-stage-purpose">{escape(spec.purpose)}</p><div class="evow-stage-detail"><section><h4>Program</h4><p><a class="evow-code-link" href="{escape(source_url, quote=True)}" target="_blank" rel="noopener">{escape(source)}</a></p><p><code>{escape(spec.function_chain)}</code></p><pre>{escape(json.dumps(contract, indent=2))}</pre></section><section><h4>Live run record</h4><p>{escape(str(detail))}</p>{worker_log_html}<pre>{escape(json.dumps(runtime, indent=2))}</pre></section></div></article>'''


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
