"""Embed an orbitable, fully preloaded 4D Gaussian viewer inside Gradio."""

from html import escape
import json
from pathlib import Path
from urllib.parse import quote


# GaussianSplats3D stores per-scene transforms and visibility in shader uniform
# arrays. The bundled version has a 32-scene ceiling, so one 4D timeline needs
# several independently preloaded renderers rather than one oversized scene.
MAX_SCENES_PER_VIEWER = 32


def _scene_groups(sources: list[str]) -> list[list[str]]:
    """Split a fully resident timeline into shader-safe renderer groups."""
    return [sources[index : index + MAX_SCENES_PER_VIEWER] for index in range(0, len(sources), MAX_SCENES_PER_VIEWER)]


def _boundary_percent(frame_durations: list[float], observed_frame_count: int | None) -> float | None:
    """Locate the recorded-to-predicted divider by elapsed time, not frame count."""
    if observed_frame_count is None:
        return None
    return sum(frame_durations[:observed_frame_count]) / sum(frame_durations) * 100


def splat_html(
    paths: list[str], frame_durations: list[float] | None = None, observed_frame_count: int | None = None
) -> str:
    """Build a viewer that fully preloads a temporal Gaussian timeline.

    Every supplied splat is downloaded and built before the player can be
    controlled. Each renderer owns no more than ``MAX_SCENES_PER_VIEWER`` static
    scenes, avoiding the bundled renderer's shader limit. Playback only changes
    the previous and current scene visibility, and switches among already-ready
    renderers while carrying the orbit camera across a group boundary.
    """
    if not paths or any(Path(path).suffix != ".splat" for path in paths):
        raise ValueError("The viewer accepts one or more Gaussian .splat frames.")
    if frame_durations is None:
        frame_durations = [1 / 12] * len(paths)
    if len(frame_durations) != len(paths) or any(duration <= 0 for duration in frame_durations):
        raise ValueError("Each splat frame needs one positive playback duration.")
    if observed_frame_count is not None and not 0 < observed_frame_count < len(paths):
        raise ValueError("The observed segment must end within the supplied timeline.")

    sources = ["/gradio_api/file=" + quote(path, safe="/") for path in paths]
    source_groups = _scene_groups(sources)
    boundary_percent = _boundary_percent(frame_durations, observed_frame_count)
    module = "/gradio_api/file=" + quote(str(Path(__file__).with_name("gaussian_viewer.bundle.js")), safe="/")
    document = f"""<!doctype html>
<html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
html,body{{margin:0;width:100%;height:100%;overflow:hidden;background:#101b27;color:#eef4fa;font:14px system-ui,sans-serif}}
#scene-host{{position:absolute;inset:0;overflow:hidden}}
.scene-group{{position:absolute;inset:0;visibility:hidden;opacity:0;pointer-events:none}}
.scene-group.active{{visibility:visible;opacity:1;pointer-events:auto}}
#loading{{position:absolute;inset:0;display:grid;place-items:center;z-index:3;pointer-events:none}}
#loading span{{width:22px;height:22px;border:3px solid #628197;border-top-color:#eef4fa;border-radius:50%;animation:spin .8s linear infinite}}
#loading.error{{padding:24px;text-align:center;pointer-events:auto;color:#f3c4c4;background:#101b27e8}}
@keyframes spin{{to{{transform:rotate(360deg)}}}}
#controls{{position:absolute;display:flex;align-items:center;gap:8px;left:12px;right:12px;bottom:12px;z-index:4;background:#101b27dd;padding:8px;border-radius:7px}}
button{{color:#eef4fa;background:#28455a;border:1px solid #628197;border-radius:5px;padding:4px 9px;cursor:pointer}}
button:disabled{{cursor:wait;opacity:.65}}
#timeline{{--timeline-start:#55758c;--boundary:100%;flex:1;min-width:0;appearance:none;background:transparent;cursor:pointer}}
#timeline:disabled{{cursor:wait}}
#timeline::-webkit-slider-runnable-track{{height:5px;border-radius:999px;background:linear-gradient(to right,var(--timeline-start) 0,var(--timeline-start) var(--boundary),#8e6a95 var(--boundary),#8e6a95 100%)}}
#timeline::-moz-range-track{{height:5px;border-radius:999px;background:linear-gradient(to right,var(--timeline-start) 0,var(--timeline-start) var(--boundary),#8e6a95 var(--boundary),#8e6a95 100%)}}
#timeline:not(.segmented)::-webkit-slider-runnable-track{{background:#55758c}}
#timeline:not(.segmented)::-moz-range-track{{background:#55758c}}
#timeline::-webkit-slider-thumb{{appearance:none;width:14px;height:14px;margin-top:-4.5px;border:1px solid #d7e4ee;border-radius:50%;background:#eef4fa}}
#timeline::-moz-range-thumb{{width:12px;height:12px;border:1px solid #d7e4ee;border-radius:50%;background:#eef4fa}}
</style></head><body><div id="scene-host"></div><div id="loading" aria-label="Preparing 3D replay"><span></span></div>
<div id="controls"><button id="play" disabled>Play</button><input id="timeline" aria-label="3D scene timeline" type="range" min="0" max="0" value="0" step="0.01" disabled></div>
<script type="module">
import * as Splats from {json.dumps(module)};
const sourceGroups = {json.dumps(source_groups)};
const frameDurations = {json.dumps(frame_durations)};
const observedBoundaryPercent = {json.dumps(boundary_percent)};
const sceneHost = document.getElementById('scene-host');
const loading = document.getElementById('loading'), timeline = document.getElementById('timeline');
const play = document.getElementById('play');
const groupStarts = [], frameStarts = [];
let totalDuration = 0, groups = [], activeGroup = -1, currentTime = 0, playing = false, timer;

for (const duration of frameDurations) {{ frameStarts.push(totalDuration); totalDuration += duration; }}
for (let index = 0; index < sourceGroups.length; index++) groupStarts.push(index * {MAX_SCENES_PER_VIEWER});

function frameIndexAt(time) {{
  if (time >= totalDuration) return frameDurations.length - 1;
  let low = 0, high = frameStarts.length - 1;
  while (low <= high) {{
    const middle = Math.floor((low + high) / 2);
    if (frameStarts[middle] <= time) low = middle + 1; else high = middle - 1;
  }}
  return Math.max(0, high);
}}

function cameraState(viewer) {{
  return {{ position: viewer.camera.position.toArray(), quaternion: viewer.camera.quaternion.toArray(),
            up: viewer.camera.up.toArray(), zoom: viewer.camera.zoom,
            target: viewer.controls ? viewer.controls.target.toArray() : null }};
}}

function applyCameraState(viewer, state) {{
  viewer.camera.position.fromArray(state.position);
  viewer.camera.quaternion.fromArray(state.quaternion);
  viewer.camera.up.fromArray(state.up);
  viewer.camera.zoom = state.zoom;
  viewer.camera.updateProjectionMatrix();
  if (viewer.controls && state.target) {{ viewer.controls.target.fromArray(state.target); viewer.controls.update(); }}
  viewer.forceRenderNextFrame();
}}

function setLocalFrame(group, localFrame) {{
  if (group.visibleFrame === localFrame) return;
  if (group.visibleFrame >= 0) group.viewer.getSplatScene(group.visibleFrame).visible = false;
  group.viewer.getSplatScene(localFrame).visible = true;
  group.visibleFrame = localFrame;
  group.viewer.forceRenderNextFrame();
}}

function activateGroup(index) {{
  if (activeGroup === index) return;
  const previous = activeGroup >= 0 ? groups[activeGroup] : null;
  const state = previous ? cameraState(previous.viewer) : null;
  if (previous) {{ previous.viewer.stop(); previous.element.classList.remove('active'); }}
  const next = groups[index];
  next.element.classList.add('active');
  if (state) applyCameraState(next.viewer, state);
  next.viewer.start();
  activeGroup = index;
}}

function setTime(time) {{
  currentTime = Math.max(0, Math.min(totalDuration, Number(time) || 0));
  const frameIndex = frameIndexAt(currentTime);
  const groupIndex = Math.floor(frameIndex / {MAX_SCENES_PER_VIEWER});
  activateGroup(groupIndex);
  setLocalFrame(groups[groupIndex], frameIndex - groupStarts[groupIndex]);
  timeline.value = String(currentTime);
}}

function stop() {{ playing = false; clearTimeout(timer); play.textContent = 'Play'; }}

function scheduleNextFrame() {{
  const index = frameIndexAt(currentTime);
  const edge = frameStarts[index] + frameDurations[index];
  timer = setTimeout(() => {{
    setTime(edge >= totalDuration ? 0 : edge);
    if (playing) scheduleNextFrame();
  }}, Math.max(1, (edge - currentTime) * 1000));
}}

async function preloadGroup(paths) {{
  const element = document.createElement('div');
  element.className = 'scene-group';
  sceneHost.appendChild(element);
  const viewer = new Splats.Viewer({{
    rootElement: element, cameraUp: [0, -1, 0], initialCameraPosition: [0, 0, 0], initialCameraLookAt: [0, 0, 4],
    sharedMemoryForWorkers: false, gpuAcceleratedSort: false, ignoreDevicePixelRatio: true,
    dynamicScene: false, enableOptionalEffects: true
  }});
  await viewer.addSplatScenes(paths.map((path, index) => ({{
    path, format: Splats.SceneFormat.Splat, splatAlphaRemovalThreshold: 1, visible: index === 0
  }})), false);
  // The active canvas alone receives input and render frames; inactive groups
  // stay resident without spending CPU/GPU time until their temporal window.
  if (viewer.controls) {{
    viewer.controls.enableRotate = true; viewer.controls.enablePan = true; viewer.controls.enableZoom = true;
    viewer.controls.screenSpacePanning = true; viewer.controls.update();
  }}
  return {{ viewer, element, visibleFrame: 0 }};
}}

timeline.addEventListener('input', () => {{ stop(); setTime(timeline.value); }});
play.addEventListener('click', () => {{
  if (playing) {{ stop(); return; }}
  playing = true; play.textContent = 'Pause'; scheduleNextFrame();
}});

(async () => {{
  try {{
    // Promise.all starts every bounded group at once. Controls remain disabled
    // until every temporal Gaussian is decoded and its GPU scene buffer exists.
    groups = await Promise.all(sourceGroups.map(preloadGroup));
    timeline.max = String(totalDuration);
    if (observedBoundaryPercent !== null) {{
      timeline.classList.add('segmented');
      timeline.style.setProperty('--boundary', observedBoundaryPercent + '%');
    }}
    setTime(0);
    timeline.disabled = false; play.disabled = false;
    loading.remove();
  }} catch (error) {{
    loading.classList.add('error'); loading.textContent = 'Unable to prepare 3D replay.';
  }}
}})();
</script></body></html>"""
    return (
        '<iframe title="Interactive 4D Gaussian scene" sandbox="allow-scripts allow-same-origin" '
        'style="width:100%;height:480px;border:0;border-radius:8px;pointer-events:auto;touch-action:none" '
        f'srcdoc="{escape(document, quote=True)}"></iframe>'
    )


def empty_splat_html() -> str:
    """Keep the 3D panel visibly reserved before an explicit run creates a scene."""
    return (
        '<div style="height:480px;display:grid;place-items:center;border:2px solid #58748d;'
        'border-radius:10px;background:#101b27;color:#e8f0f5;font:15px system-ui">'
        "</div>"
    )
