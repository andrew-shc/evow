"""Compose the four camera features into one private LAN dashboard."""

import gradio as gr

from GREENFIELD.features.pages import build_editing, build_future, build_selection
from GREENFIELD.replay.app import build_app as build_replay


DASHBOARD_TITLE = "An Eternal View of Our World · Applications of Single-view Stationary Camera"


FLOW_HEADER_CSS = """
html, body { overflow-x: hidden !important; }
.internal-flow { overflow-x: hidden !important; }
/* The execution flow is a compact scan list, not a card stack. */
.internal-flow > div > .column { gap: 6px !important; }
/* Keep the section panels, but make method choices read as one aligned group. */
.evow-method-choice .wrap { flex-direction: column !important; align-items: flex-start !important; gap: 6px !important; }
.evow-method-choice label { width: 100%; box-shadow: none !important; border: 0 !important; border-radius: 0 !important; background: transparent !important; padding: 4px 0 !important; }
.evow-method-choice label:hover, .evow-method-choice label.selected { background: transparent !important; }
/* Captions keep Clip labels visible without recreating component containers. */
.evow-clip-panel > div > .column { gap: 6px !important; }
.evow-clip-caption { min-height: 0 !important; margin: 4px 0 -2px !important; padding: 0 !important; color: #425466; font-size: 12px; }
.evow-clip-caption p { margin: 0 !important; }
.internal-flow > button {
  width: 1.5rem !important;
  min-height: 1.5rem !important;
  justify-content: flex-start !important;
}
.internal-flow > button > span:first-child { display: none !important; }
.internal-flow > button > .icon { margin: 0 !important; font-size: 1rem !important; transform: rotate(-90deg) !important; }
.internal-flow > button.open > .icon { transform: rotate(0deg) !important; }
.evow-section-heading h2 { font-size: 1.5rem !important; font-weight: 700 !important; line-height: 1.35 !important; }
.evow-flow-heading h2 { color: #7e22ce !important; text-align: center !important; }
.evow-media { outline: 2px solid #58748d !important; outline-offset: -2px !important; border-radius: 10px !important; background: #f8fafc !important; overflow: hidden !important; }
.evow-media > div { border: 0 !important; }
.evow-media { min-width: 0 !important; }
.evow-media video, .evow-media img, .evow-media canvas, .evow-media iframe { display: block !important; width: 100% !important; max-width: 100% !important; height: auto !important; object-fit: contain !important; }
/* Reserve result geometry before a browser has decoded its first video frame. */
.evow-output-slot { min-height: 240px; contain: layout paint; }
.evow-output-slot iframe { min-height: 480px !important; }
#splat-viewer-output, #future-splat-viewer-output, #selection-splat-viewer, #editing-splat-viewer-output { min-height: 480px; contain: layout paint; }
#novel-view-output, #splat-viewer-output { position: relative; overflow: hidden; }
.evow-generation-spinner::after { content: ""; position: absolute; inset: 0; z-index: 10; background: rgb(248 250 252 / 72%); pointer-events: all; }
.evow-generation-spinner::before { content: ""; position: absolute; top: 50%; left: 50%; z-index: 11; width: 28px; height: 28px; margin: -14px; border: 3px solid #628197; border-top-color: #102a43; border-radius: 50%; animation: evow-generation-spin .8s linear infinite; }
@keyframes evow-generation-spin { to { transform: rotate(360deg); } }
.evow-stage { box-shadow: none !important; border: 0 !important; border-radius: 0 !important; background: transparent !important; overflow: visible !important; }
.evow-stage > button { display: flex !important; align-items: center; justify-content: flex-start !important; margin: 0 !important; height: 2.7rem; min-height: 2.7rem; padding: 9px 11px !important; border: 1px solid #cbd5df !important; border-left: 5px solid #94a3b8 !important; border-radius: 7px !important; background: #f8fafc !important; text-align: left !important; white-space: nowrap !important; }
.evow-stage > button > span:first-child { position: relative; display: block !important; flex: 1 1 0 !important; min-width: 0; width: auto !important; height: 1.3rem; overflow: hidden; font-size: 0 !important; text-align: left !important; }
.evow-stage > button > .icon { flex: 0 0 auto; margin: 0 0 0 12px !important; }
.evow-stage > button > span:first-child::before { content: attr(data-evow-stage-name); position: absolute; top: 0; right: 11rem; left: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--section-header-text-size); line-height: 1.3rem; text-align: left; }
.evow-stage > button > span:first-child::after { content: attr(data-evow-stage-meta); position: absolute; top: 0; right: 0; width: 10.5rem; overflow: hidden; white-space: nowrap; font-size: 12px; line-height: 1.3rem; font-variant-numeric: tabular-nums; font-weight: 650; text-align: right; }
.evow-stage-complete > button { border-left-color: #15803d !important; background: #f0fdf4 !important; color: #166534 !important; }
.evow-stage-running > button { border-left-color: #0369a1 !important; background: #f0f9ff !important; color: #075985 !important; }
.evow-stage-waiting > button { border-left-color: #94a3b8 !important; background: #f8fafc !important; color: #475569 !important; }
.evow-stage-failed > button { border-left-color: #b91c1c !important; background: #fff1f2 !important; color: #991b1b !important; }
/* Root header bar. Sticky positioning keeps the title and the compact GPU
   control visible on every tab and while any page scrolls. The controls stack
   vertically under the title, with the title left-aligned and the compact GPU
   controls right-aligned beneath it. The direct children are
   forced to min-width:0 so the title can shrink rather than push the control
   wider. */
.evow-topbar { position: sticky; top: 0; z-index: 60; width: 100%; box-sizing: border-box; flex-direction: row !important; flex-wrap: wrap !important; align-items: center !important; gap: 10px !important; padding: 10px 18px !important; background: #ffffff; }
.evow-topbar > * { min-width: 0 !important; }
.evow-topbar-title { flex: 1 1 0 !important; width: auto !important; }
.evow-topbar-title h1 { margin: 0 !important; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 1.55rem !important; line-height: 1.25 !important; }
/* Gradio 5.50 puts elem_classes directly on the <button>, alongside its
   generated size class (e.g. "sm"), so this selector can override the default
   full-width, large-text button. align-self + width keep it compact and
   right-aligned under the title instead of stretching across the bar. */
.evow-global-gpu { align-self: center !important; width: auto !important; max-width: max-content !important; min-width: 0 !important; min-height: 1.8rem !important; padding: 3px 10px !important; font-size: 11px !important; font-weight: 500 !important; line-height: 1.4 !important; }
.evow-global-status { flex-basis: 100%; margin: 0 !important; color: #64748b; font-size: 11px; text-align: right; }
.evow-global-status p { margin: 0 !important; }
/* Shared dashboard CSS must live here: the child feature pages are merged into
   this root Blocks with .render(), which drops their css=/js=, so REPLAY_CSS /
   APP_CSS rules never reach the served page. The icon glyph ⓘ (U+24D8) is already
   circled, so this only fixes its box and vertical alignment; inline-flex +
   overflow:visible prevent the top of the glyph from being cropped by the
   surrounding Markdown line box. */
.evow-help { display: inline-flex; align-items: center; justify-content: center; vertical-align: middle; line-height: 1; overflow: visible; margin-left: 0.35em; cursor: help; }
.evow-help-popover { position:fixed; z-index:10000; display:none; width:min(460px, calc(100vw - 32px)); max-height:min(70vh, 520px); overflow:auto; padding:12px 14px; border:1px solid #9fb1c0; border-radius:8px; background:#102a43; color:#f8fafc; box-shadow:0 8px 24px rgb(15 23 42 / .28); font:13px/1.5 system-ui; }
"""


FLOW_HEADER_JS = """
() => {
  const formatStageHeaders = () => {
    document.querySelectorAll(".evow-stage > button > span:first-child").forEach((label) => {
      const parts = label.textContent.split(" · ");
      if (parts.length !== 3) return;
      label.dataset.evowStageName = parts[0];
      label.dataset.evowStageServerMeta = `${parts[1]} · ${parts[2]}`;
      label.dataset.evowStageMeta = label.dataset.evowStageServerMeta;
    });
  };
  formatStageHeaders();
  new MutationObserver(formatStageHeaders).observe(document.body, {
    characterData: true,

    childList: true,
    subtree: true,
  });
  // Backend events remain the source of truth. Between events, this browser
  // clock advances the *display only* for active stages at millisecond precision.
  const stageClocks = new WeakMap();
  const tickStageClocks = () => {
    document.querySelectorAll(".evow-stage > button > span:first-child").forEach((label) => {
      const meta = label.dataset.evowStageServerMeta || "";
      const match = meta.match(/^Running · ([0-9.]+)s$/);
      if (!match) { stageClocks.delete(label); return; }
      const serverSeconds = Number(match[1]);
      const clock = stageClocks.get(label);
      if (!clock || clock.serverMeta !== meta) {
        stageClocks.set(label, { serverMeta: meta, serverSeconds, startedAt: performance.now() });
      }
      const active = stageClocks.get(label);
      label.dataset.evowStageMeta = `Running · ${(active.serverSeconds + (performance.now() - active.startedAt) / 1000).toFixed(3)}s`;
    });
    requestAnimationFrame(tickStageClocks);
  };
  requestAnimationFrame(tickStageClocks);
  const helpPopover = document.createElement("div");
  helpPopover.className = "evow-help-popover";
  document.body.appendChild(helpPopover);
  let openHelpIcon = null;
  const hideHelp = () => { helpPopover.style.display = "none"; openHelpIcon = null; };
  const showHelp = (icon) => {
    helpPopover.textContent = icon._evowHelpText || icon.getAttribute("title") || "";
    const box = icon.getBoundingClientRect();
    helpPopover.style.display = "block";
    helpPopover.style.left = `${Math.max(16, Math.min(box.right - helpPopover.offsetWidth, window.innerWidth - helpPopover.offsetWidth - 16))}px`;
    helpPopover.style.top = `${Math.max(16, box.bottom + 4)}px`;
    openHelpIcon = icon;
  };
  const bindHelp = () => document.querySelectorAll(".evow-help").forEach((icon) => {
    if (icon.dataset.evowBound) return;
    icon.dataset.evowBound = "true";
    icon._evowHelpText = icon.getAttribute("title") || "";
    icon.removeAttribute("title");
    icon.addEventListener("click", (event) => { event.stopPropagation(); openHelpIcon === icon ? hideHelp() : showHelp(icon); });
    icon.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); openHelpIcon === icon ? hideHelp() : showHelp(icon); } });
  });
  document.addEventListener("click", (event) => { if (!helpPopover.contains(event.target) && !event.target.closest(".evow-help")) hideHelp(); });
  bindHelp();
  // CSS owns media geometry. Synthetic resize events raced Gradio's first layout.
  new MutationObserver(() => { bindHelp(); }).observe(document.body, { childList: true, subtree: true });

}
"""


def _clear_gpu_memory() -> str:
    """Release this dashboard process's cached models after an out-of-memory error.

    This is a dashboard-wide recovery action: it frees cached GPU memory for every
    project tab at once, and only for THIS dashboard process. It cannot free GPU
    memory held by another process.
    """
    from GREENFIELD.features.model_adapters import clear_gpu_memory
    result = clear_gpu_memory()
    before = result["before_bytes"] / 1024**3
    after = result["after_bytes"] / 1024**3
    released = ", ".join(result["released"]) or "cached CUDA allocations"
    return f"**GPU memory cleared:** {released}. App allocation: {before:.2f} → {after:.2f} GiB."


def build_dashboard() -> gr.Blocks:
    """Compose the four feature pages under one dashboard-wide control bar.

    ``gr.TabbedInterface`` cannot host arbitrary content above its tabs, so the
    dashboard is assembled manually: a top-level ``gr.Blocks`` holds a persistent
    header stack with the title above a compact dashboard-wide "Clear GPU memory"
    control, followed by ``gr.Tabs()`` whose tabs each ``render()`` an already-built
    feature page.

    ``Blocks.render()`` transfers a child page's components and event listeners,
    but NOT its ``css=``/``js=``. Shared styling therefore has to stay in this
    module's root ``FLOW_HEADER_CSS`` (see the known issue in GREENFIELD/AGENTS.md).
    """
    pages = [
        ("4D Replay", build_replay()),
        ("Future View (+ Training)", build_future()),
        ("Text Query", build_selection()),
        ("Text Manipulation", build_editing()),
    ]
    with gr.Blocks(title=DASHBOARD_TITLE, css=FLOW_HEADER_CSS, js=FLOW_HEADER_JS) as dashboard:
        with gr.Column(elem_classes="evow-topbar"):
            gr.Markdown(f"# {DASHBOARD_TITLE}", elem_classes="evow-topbar-title")
            clear_gpu = gr.Button(
                "Clear GPU memory",
                variant="secondary",
                size="sm",
                elem_classes="evow-global-gpu",
            )
            gpu_status = gr.Markdown("", elem_classes="evow-global-status")
        clear_gpu.click(_clear_gpu_memory, None, gpu_status, queue=False, show_progress="hidden")
        with gr.Tabs():
            for name, page in pages:
                with gr.Tab(name):
                    page.render()
    return dashboard
