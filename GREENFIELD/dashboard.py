"""Compose the four camera features into one private LAN dashboard."""

import gradio as gr

from GREENFIELD.features.pages import build_editing, build_future, build_selection
from GREENFIELD.app_core.splat_viewer import SPLAT_VIEWER_FRAME_CSS
from GREENFIELD.replay.app import build_app as build_replay


DASHBOARD_TITLE = "An Eternal View of Our World"
DASHBOARD_SUBTITLE = "Applications of Single-view Stationary Camera"


FLOW_HEADER_CSS = """
html, body { overflow-x: hidden !important; }
/* A subtle tint distinguishes the expandable advanced execution record. */
.internal-flow { overflow-x: hidden !important; padding: 4px 8px !important; border: 1px solid #e9d5ff !important; border-radius: 10px !important; background: #faf5ff !important; }
/* The execution flow is a compact scan list, not a card stack. */
.internal-flow > div > .column { gap: 6px !important; }
/* Keep the section panels, but make method choices read as one aligned group. */
.evow-method-choice .wrap { flex-direction: column !important; align-items: flex-start !important; gap: 6px !important; }
.evow-method-choice label { width: 100%; box-shadow: none !important; border: 0 !important; border-radius: 0 !important; background: transparent !important; padding: 4px 0 !important; }
.evow-method-choice label:hover, .evow-method-choice label.selected { background: transparent !important; }
/* Gradio Radio has no per-choice disabled API. The root script marks the
   unavailable Animated Mesh choice semantically disabled after render. */
#replay-method-choice .evow-disabled-method { opacity: .52 !important; cursor: not-allowed !important; }
#replay-method-choice .evow-disabled-method input,
#replay-method-choice .evow-disabled-method [role="radio"] { cursor: not-allowed !important; }
/* Captions keep Clip labels visible without recreating component containers. */
.evow-clip-panel > div > .column { gap: 6px !important; }
.evow-clip-caption { min-height: 0 !important; margin: 4px 0 -2px !important; padding: 0 !important; color: #425466; font-size: 12px; }
.evow-clip-caption p { margin: 0 !important; }
.evow-inline-help { align-items: center !important; gap: 6px !important; }
.evow-inline-help-icon { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; margin: 0 !important; padding: 0 !important; }
.evow-inline-help-icon p { margin: 0 !important; }
.internal-flow > button {
  position: relative !important;
  width: 100% !important;
  min-height: 2.7rem !important;
  justify-content: center !important;
  color: #7e22ce !important;
  font-size: 1.5rem !important;
  font-weight: 700 !important;
  line-height: 1.35 !important;
}
/* This is the native Accordion summary, so it remains visible while its
   stages are collapsed. The absolute chevron keeps the title centered. */
.internal-flow > button > span:first-child {
  display: block !important;
  flex: 0 1 auto !important;
  width: auto !important;
  overflow: visible !important;
  font: inherit !important;
  text-align: center !important;
}
.internal-flow > button > .icon { position: absolute !important; right: .75rem !important; margin: 0 !important; font-size: 1rem !important; transform: rotate(-90deg) !important; }
.internal-flow > button.open > .icon { transform: rotate(0deg) !important; }
.evow-section-heading h2 { font-size: 1.5rem !important; font-weight: 700 !important; line-height: 1.35 !important; }
.evow-media { outline: 2px solid #58748d !important; outline-offset: -2px !important; border-radius: 10px !important; background: #f8fafc !important; overflow: hidden !important; }
.evow-video-list { display:grid; gap:18px; }
.evow-clip-result { display:grid; gap:8px; min-width:0; }
.evow-clip-result video { display:block; width:100%; max-width:100%; height:auto; aspect-ratio:16/9; border-radius:8px; background:#101b27; object-fit:contain; }
.evow-clip-interval { min-width:0; padding:9px 10px; border:1px solid #cbd5df; border-radius:8px; background:#f8fafc; }
.evow-clip-interval strong { display:block; color:#243b53; font-size:12px; }
.evow-clip-interval p { margin:4px 0 0; color:#526575; font-size:11px; line-height:1.4; overflow-wrap:anywhere; }
.evow-empty-clips { margin:0; color:#425466; font-size:12px; line-height:1.4; }
.evow-trim-range { --trim-start:0%; --trim-end:100%; position:relative; height:34px; margin:2px 0; }
.evow-trim-track, .evow-trim-fill { position:absolute; top:14px; height:5px; border-radius:999px; pointer-events:none; }
.evow-trim-track { right:8px; left:8px; background:#cbd5df; }
.evow-trim-fill { left:calc(8px + (100% - 16px) * var(--trim-start)); right:calc(8px + (100% - 16px) * (1 - var(--trim-end))); background:#0f766e; }
.evow-trim-range input[type=range] { position:absolute; inset:0; width:100%; margin:0; appearance:none; background:transparent; pointer-events:none; }
.evow-trim-range input[type=range]::-webkit-slider-thumb { width:16px; height:16px; appearance:none; border:2px solid #0f766e; border-radius:50%; background:#fff; box-shadow:0 1px 3px rgb(15 23 42 / .28); pointer-events:auto; cursor:grab; }
.evow-trim-range input[type=range]::-moz-range-thumb { width:13px; height:13px; border:2px solid #0f766e; border-radius:50%; background:#fff; pointer-events:auto; cursor:grab; }
.evow-trim-range output { position:absolute; top:25px; left:0; color:#526575; font-size:10px; font-variant-numeric:tabular-nums; }
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
.evow-stage > button > span:first-child::after { content: attr(data-evow-stage-meta); position: absolute; top: 0; right: 0; width: 10.5rem; overflow: hidden; white-space: nowrap; font-size: 12px; line-height: 1.3rem; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-variant-numeric: tabular-nums; font-weight: 650; text-align: right; }
.evow-stage-complete > button { border-left-color: #15803d !important; background: #f0fdf4 !important; color: #166534 !important; }
.evow-stage-running > button { border-left-color: #0369a1 !important; background: #f0f9ff !important; color: #075985 !important; }
.evow-stage-waiting > button { border-left-color: #94a3b8 !important; background: #f8fafc !important; color: #475569 !important; }
.evow-stage-failed > button { border-left-color: #b91c1c !important; background: #fff1f2 !important; color: #991b1b !important; }
.evow-stage-skipped > button { border-left-color: #b45309 !important; background: #fffbeb !important; color: #92400e !important; }
/* Root header bar. Sticky positioning keeps the title and the compact GPU
   control visible on every tab and while any page scrolls. The controls stack
   vertically under the title, with the title left-aligned and the compact GPU
   controls right-aligned beneath it. The direct children are
   forced to min-width:0 so the title can shrink rather than push the control
   wider. */
.evow-topbar { position: sticky; top: 0; z-index: 60; width: 100%; box-sizing: border-box; flex-direction: row !important; flex-wrap: wrap !important; align-items: center !important; gap: 10px !important; padding: 10px 18px !important; background: #ffffff; }
.evow-topbar > * { min-width: 0 !important; }
.evow-topbar-title { flex: 1 1 0 !important; width: auto !important; gap: 0 !important; }
.evow-topbar-title h1 { margin: 0 !important; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 1.55rem !important; line-height: 1.25 !important; }
.evow-topbar-subtitle { margin: 2px 0 0 !important; color: #526575; font-size: .84rem; line-height: 1.3; }
.evow-topbar-subtitle p { margin: 0 !important; }
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
/* Compact config labels: the group heading is a single small line with the
   explanation in its ⓘ popover, and control labels never wrap. This keeps the
   dense feature-page option panels from spending width on verbose text. */
.evow-config-caption {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  margin: 0.4rem 0 0.1rem;
  font-size: 0.76rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--body-text-color-subdued);
}
.evow-config-caption .evow-help { font-size: 0.8rem; }
.evow-config-panel label { white-space: nowrap; font-size: 0.78rem; }
/* Hide ONLY Gradio's standing info paragraph: FLOW_HEADER_JS copies the same
   description onto the entry box as a native ``title`` tooltip, so leaving it
   visible would double the text and widen every configured row. In Gradio 5.50
   the long detail lives in an Info wrapper div around a ``.md.prose`` markdown
   span, so those two selectors are all we hide. Do NOT hide
   ``[data-testid="block-info"]``: that testid is the component's *short label*
   (the visible ``label`` text), and hiding it is what made every knob an
   unlabeled box. */
.evow-config-field .md.prose,
.evow-config-field div:has(> .md.prose) { display: none !important; }
/* One compact, one-setting-per-row contract shared by all four Configuration
   accordions (Replay, Future View, Text Query, Text Manipulation). A single
   block replaces the former separate replay and later-panel blocks so the four
   panels cannot drift apart again. Only gaps, padding, and label type shrink;
   input min-height stays ~1.65rem so every entry remains a comfortable target. */
.evow-config-panel { padding: 2px 6px 4px !important; }
/* Gradio 5.50 renders every gr.Column with gap: var(--layout-gap). The later
   panels wrap their rows in mode visibility-toggle gr.Column groups, which are
   plain `column`s rather than `form`s, so the default gap survived the old
   `.form`-only reset and inflated each row's vertical padding; Replay escaped
   only because its single wrapper column was given an explicit gap:0. Reset the
   stacked layout gaps on every config-panel descendant instead. This stays
   scoped under .evow-config-panel so .evow-flow-group and unrelated layout
   columns keep their own spacing. */
.evow-config-panel .column { gap: 0 !important; }
.evow-config-panel .form,
.evow-config-panel .block,
.evow-config-panel .wrap { gap: 0 !important; }
.evow-config-panel .block { padding: 0 !important; }
.evow-config-panel .evow-config-caption { margin: 0.3rem 0 0 !important; }
/* Replay's settings list, and the later panels' toggle groups, now share this
   exact row/label/control contract. The Markdown label owns the ⓘ explanation;
   the control keeps its native Gradio wrapper (hiding it would make the input
   unreachable), rendered as screen-reader-only by the bare container. */
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
/* One nested layer makes atomic trace rows scannable without coalescing or
   relocating them; children keep their stable Gradio component identity. */
.evow-flow-group {
  margin: 4px 0 !important;
  padding: 0 6px 6px !important;
  border: 1px solid #d7e0e8 !important;
  border-radius: 8px !important;
  background: #ffffff !important;
  box-shadow: none !important;
}
.evow-flow-group > button {
  display: flex !important;
  align-items: center !important;
  min-height: 2.2rem !important;
  padding: 6px 4px !important;
  border-left: 4px solid #94a3b8 !important;
  background: transparent !important;
  color: #334e68 !important;
  font-size: .82rem !important;
  font-weight: 700 !important;
  letter-spacing: .02em !important;
  text-align: left !important;
}
/* Parent phase headers use the same two-column summary as their atomic rows:
   name on the left; the rolled-up lifecycle state and stopwatch on the right. */
.evow-flow-group > button > span:first-child {
  position: relative;
  display: block !important;
  flex: 1 1 0 !important;
  min-width: 0;
  width: auto !important;
  height: 1.3rem;
  overflow: hidden;
  font-size: 0 !important;
  text-align: left !important;
}
.evow-flow-group > button > .icon { flex: 0 0 auto; margin: 0 0 0 10px !important; }
.evow-flow-group > button > span:first-child::before {
  content: attr(data-evow-stage-name);
  position: absolute;
  top: 0;
  right: 11rem;
  left: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: .82rem;
  line-height: 1.3rem;
  text-align: left;
}
.evow-flow-group > button > span:first-child::after {
  content: attr(data-evow-stage-meta);
  position: absolute;
  top: 0;
  right: 0;
  width: 10.5rem;
  overflow: hidden;
  white-space: nowrap;
  font: 650 12px/1.3rem ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-variant-numeric: tabular-nums;
  text-align: right;
}
.evow-flow-group-complete > button { border-left-color: #15803d !important; color: #166534 !important; }
.evow-flow-group-running > button { border-left-color: #0369a1 !important; color: #075985 !important; }
.evow-flow-group-waiting > button { border-left-color: #94a3b8 !important; color: #475569 !important; }
.evow-flow-group-failed > button { border-left-color: #b91c1c !important; color: #991b1b !important; }
.evow-flow-group-skipped > button { border-left-color: #b45309 !important; color: #92400e !important; }
.evow-flow-group > div > .column { gap: 5px !important; }
.evow-config-panel input[type="number"],
.evow-config-panel input[type="text"],
.evow-config-panel textarea { min-height: 1.65rem !important; padding: 3px 5px !important; }
/* Narrow viewports stack the config columns, so the desktop single-line labels
   no longer have room and would overflow sideways. Let labels wrap and break
   long tokens there; desktop keeps the dense nowrap layout above. */
@media (max-width: 640px) {
  .evow-config-panel label { white-space: normal !important; }
  .evow-config-panel .evow-config-field { overflow-wrap: anywhere; }
  .evow-config-panel .evow-config-row { align-items: stretch !important; }
  .evow-config-panel .evow-config-label,
  .evow-config-panel .evow-config-control { min-width: 0 !important; }
}
/* A sample selection can take time to reach the browser, especially for the
   longer bundled videos. Preserve the old frame as a muted, grey thumbnail and
   show an overlay until the replacement video fires a playable media event. */
.evow-source-video { position: relative !important; }
.evow-source-video.evow-source-loading video { filter: grayscale(1) opacity(.45) !important; }
.evow-source-video.evow-source-loading::before { content: ""; position: absolute; z-index: 12; top: 50%; left: 50%; width: 26px; height: 26px; margin: -30px 0 0 -13px; border: 3px solid #9fb1c0; border-top-color: #102a43; border-radius: 50%; animation: evow-generation-spin .8s linear infinite; }
.evow-source-video.evow-source-loading::after { content: "Loading video…"; position: absolute; z-index: 12; top: calc(50% + 8px); left: 0; right: 0; color: #102a43; font-size: 13px; font-weight: 650; text-align: center; text-shadow: 0 1px #fff; pointer-events: none; }
"""

# Child-page CSS is omitted by Gradio's ``render()``, so retain the shared
# embedded 3D frame label at the composed dashboard root.
FLOW_HEADER_CSS += SPLAT_VIEWER_FRAME_CSS


FLOW_HEADER_JS = """
() => {
  const formatStageHeaders = () => {
    document.querySelectorAll(".evow-stage > button > span:first-child, .evow-flow-group > button > span:first-child").forEach((label) => {
      const parts = label.textContent.split(" · ");
      if (parts.length !== 3) return;
      const serverMeta = `${parts[1]} · ${parts[2]}`;
      if (label.dataset.evowStageServerMeta === serverMeta) return;
      label.dataset.evowStageName = parts[0];
      label.dataset.evowStageServerMeta = serverMeta;
      label.dataset.evowStageMeta = serverMeta;
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
    document.querySelectorAll(".evow-stage > button > span:first-child, .evow-flow-group > button > span:first-child").forEach((label) => {
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
  // Videos are demonstrations, not an audio channel. Set both properties so
  // newly rendered Gradio <video> elements cannot resume audio after a refresh.
  const muteDashboardVideos = () => document.querySelectorAll("video").forEach((video) => {
    video.muted = true;
    video.defaultMuted = true;
    video.volume = 0;
    video.setAttribute("muted", "");
  });
  const finishSourceLoading = (event) => {
    const source = event.target.closest?.(".evow-source-video");
    if (!source) return;
    source.classList.remove("evow-source-loading");
    source.removeAttribute("aria-busy");
  };
  document.addEventListener("loadeddata", finishSourceLoading, true);
  document.addEventListener("canplay", finishSourceLoading, true);
  muteDashboardVideos();
  // Gradio 5.50 has no range-slider component. These paired native inputs act
  // as one accessible control and mirror a stable "start,end" value into the
  // hidden Gradio Textbox that the Python handler validates.
  const syncTrimRange = (range, changed) => {
    const inputs = range.querySelectorAll("input[type=range]");
    if (inputs.length !== 2) return;
    let start = Number(inputs[0].value), end = Number(inputs[1].value);
    if (changed === inputs[0] && start > end) end = start;
    if (changed === inputs[1] && end < start) start = end;
    inputs[0].value = String(start); inputs[1].value = String(end);
    const maximum = Math.max(Number(inputs[0].max) || 1, 1);
    range.style.setProperty("--trim-start", String(start / maximum));
    range.style.setProperty("--trim-end", String(end / maximum));
    const output = range.querySelector("output");
    if (output) output.textContent = `${start.toFixed(1)}s – ${end.toFixed(1)}s`;
    if (!changed) return;
    const hidden = document.getElementById(range.dataset.evowTrimTarget || "")?.querySelector("textarea, input");
    if (!hidden) return;
    hidden.value = `${start},${end}`;
    hidden.dispatchEvent(new Event("input", { bubbles:true }));
    hidden.dispatchEvent(new Event("change", { bubbles:true }));
  };
  const bindTrimRanges = () => document.querySelectorAll(".evow-trim-range").forEach((range) => {
    if (range.dataset.evowTrimBound) return;
    range.dataset.evowTrimBound = "true";
    syncTrimRange(range, null);
    range.querySelectorAll("input[type=range]").forEach((input) => input.addEventListener("input", () => syncTrimRange(range, input)));
  });
  bindTrimRanges();
  // Child page JavaScript is dropped by Blocks.render(), so this root-level
  // binder makes just the unavailable Radio choice inert after every render.
  const disableAnimatedMeshChoice = () => document.querySelectorAll("#replay-method-choice label").forEach((choice) => {
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
  disableAnimatedMeshChoice();
  new MutationObserver(disableAnimatedMeshChoice).observe(document.body, {
    childList: true,
    subtree: true,
  });
  // Every config factory carries its description in Gradio's ``info`` prop, but
  // the CSS above hides the standing paragraph. This copies that text onto the
  // real controls as a native ``title`` so hovering an entry box explains it
  // without widening the panel. ``root`` defaults to the whole document; the
  // final observer re-runs it after Gradio renders or switches tabs.
  const applyConfigFieldTooltips = (root = document) => {
    root.querySelectorAll(".evow-config-field").forEach((field) => {
      if (field.dataset.evowTip === "1") return;
      // Gradio 5.50 renders no ``.info``/``block-label``: the description is the
      // Info markdown (``.md.prose``) and the short label is ``block-info``.
      // ``.md.prose`` is hidden by CSS but still in the DOM, so its text reads out.
      const infoNode = field.querySelector(".md.prose") || field.querySelector(".info");
      const detail = infoNode ? infoNode.textContent.trim() : "";
      const labelNode =
        field.querySelector("label[data-testid='block-label']") ||
        field.querySelector("[data-testid='block-info']") ||
        field.querySelector("label");
      const label = labelNode ? labelNode.textContent.trim() : "";
      // The tooltip exists to surface the hidden description; a label-only title
      // would merely repeat the visible label. Wait for real detail text, so a
      // field whose ``.md.prose`` arrives late stays unmarked and retryable.
      const title = detail ? (label ? `${label} — ${detail}` : detail) : "";
      if (!title) return;
      // Only mark the field complete once a target control actually received the
      // title. Gradio can insert the field shell before it fills ``.md.prose`` or
      // the input, and since we mark nothing here, the next observer pass retries.
      // Marking eagerly would strand those rows without a hover title forever.
      let titledControls = 0;
      field.querySelectorAll("input, textarea, select, [role='slider']").forEach((control) => {
        control.setAttribute("title", title);
        titledControls += 1;
      });
      if (titledControls === 0) return;
      field.dataset.evowTip = "1";
    });
  };
  applyConfigFieldTooltips();
  // CSS owns media geometry. Synthetic resize events raced Gradio's first layout.
  const setSavedRunPlaceholders = () => {
    document.querySelectorAll(".evow-saved-run-dropdown input").forEach((input) => {
      if (!input.value) input.setAttribute("placeholder", "Choose a saved run");
    });
  };
  setSavedRunPlaceholders();

  // ``characterData`` matters: Gradio may set ``.md.prose``'s text after the
  // field shell exists, and only a text-node mutation re-triggers the tooltip
  // binder for those rows (the field stays unmarked until it has real text).
  new MutationObserver(() => { bindHelp(); setSavedRunPlaceholders(); applyConfigFieldTooltips(); muteDashboardVideos(); bindTrimRanges(); }).observe(document.body, { childList: true, subtree: true, characterData: true });

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
        ("3D Video", build_replay()),
        ("Future View", build_future()),
        ("Text Query", build_selection()),
        ("Text Manipulation", build_editing()),
    ]
    with gr.Blocks(title=DASHBOARD_TITLE, css=FLOW_HEADER_CSS, js=FLOW_HEADER_JS) as dashboard:
        with gr.Column(elem_classes="evow-topbar"):
            with gr.Column(elem_classes="evow-topbar-title"):
                gr.Markdown(f"# {DASHBOARD_TITLE}", container=False)
                gr.Markdown(DASHBOARD_SUBTITLE, container=False, elem_classes="evow-topbar-subtitle")
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
