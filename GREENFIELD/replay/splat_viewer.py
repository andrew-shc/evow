"""Embed an orbitable, preloaded 4D Gaussian viewer inside the Gradio page."""

from html import escape
import json
from pathlib import Path
from urllib.parse import quote


def splat_html(paths: list[str], frame_durations: list[float] | None = None,
               timeline_description: str | None = None) -> str:
    """Build one viewer that preloads selected temporal scenes once.

    GaussianSplats3D can switch visibility cheaply after its scene buffer is
    built, but adding or removing a scene rebuilds that buffer. This deliberately
    follows 4D Replay's successful pattern: load frame zero for immediate use,
    then batch the already-bounded timeline in the background without replacing
    the iframe or the orbit camera.
    """
    if not paths or any(Path(path).suffix != ".splat" for path in paths):
        raise ValueError("The viewer accepts one or more Gaussian .splat frames.")
    if frame_durations is None:
        frame_durations = [1 / 12] * len(paths)
    if len(frame_durations) != len(paths) or any(duration <= 0 for duration in frame_durations):
        raise ValueError("Each splat frame needs one positive playback duration.")
    sources = ["/gradio_api/file=" + quote(path, safe="/") for path in paths]
    module = "/gradio_api/file=" + quote(str(Path(__file__).with_name("gaussian_viewer.bundle.js")), safe="/")
    document = f"""<!doctype html>
<html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
html,body{{margin:0;width:100%;height:100%;overflow:hidden;background:#101b27;color:#eef4fa;font:14px system-ui,sans-serif}}
#loading{{position:absolute;inset:0;display:grid;place-items:center;z-index:2;pointer-events:none}}#loading span{{width:22px;height:22px;border:3px solid #628197;border-top-color:#eef4fa;border-radius:50%;animation:spin .8s linear infinite}}@keyframes spin{{to{{transform:rotate(360deg)}}}}
#controls{{position:absolute;display:flex;align-items:center;gap:8px;left:12px;right:12px;bottom:12px;z-index:2;background:#101b27dd;padding:8px;border-radius:7px}}
#timeline{{flex:1}} button{{color:#eef4fa;background:#28455a;border:1px solid #628197;border-radius:5px;padding:4px 9px;cursor:pointer}}#context{{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
</style></head><body><div id="loading" aria-label="Loading 3D scene"><span></span></div>
<div id="controls"><button id="play" disabled>Play</button><button id="reset" disabled>Reset view</button><input id="timeline" type="range" min="0" max="{len(paths) - 1}" value="0" step="1" disabled><output id="time">Frame 1 / {len(paths)}</output><span id="context">{escape(timeline_description or "")}</span></div>
<script type="module">
import * as Splats from {json.dumps(module)};
const sources = {json.dumps(sources)};
const frameDurations = {json.dumps(frame_durations)};
const loading = document.getElementById('loading'), timeline = document.getElementById('timeline');
const play = document.getElementById('play'), reset = document.getElementById('reset'), time = document.getElementById('time'), context = document.getElementById('context');
let viewer, playing = false, timer, loadedScenes = 0;
function setFrame(frame) {{
  const index = Math.max(0, Math.min(loadedScenes - 1, Number(frame)));
  viewer.splatMesh.scenes.forEach((scene, sceneIndex) => {{ scene.visible = sceneIndex === index; }});
  viewer.splatMesh.updateTransforms();
  timeline.value = String(index); time.textContent = "Frame " + (index + 1) + " / " + sources.length + " · loaded " + loadedScenes;
}}
function stop() {{ playing = false; clearTimeout(timer); play.textContent = 'Play'; }}
function advance() {{
  const next = (Number(timeline.value) + 1) % loadedScenes;
  setFrame(next);
  if (playing) timer = setTimeout(advance, frameDurations[next] * 1000);
}}
try {{
  viewer = new Splats.Viewer({{
    cameraUp: [0, -1, 0], initialCameraPosition: [0, 0, 0], initialCameraLookAt: [0, 0, 4],
    sharedMemoryForWorkers: false, gpuAcceleratedSort: false, ignoreDevicePixelRatio: true,
    dynamicScene: true, enableOptionalEffects: true
  }});
  await viewer.addSplatScene(sources[0], {{
    format: Splats.SceneFormat.Splat, splatAlphaRemovalThreshold: 1, visible: true, showLoadingUI: false
  }});
  loadedScenes = 1;
  timeline.max = "0";
  viewer.start();
  // Keep interaction explicit even if the upstream viewer changes defaults.
  if (viewer.controls) {{
    viewer.controls.enableRotate = true; viewer.controls.enablePan = true; viewer.controls.enableZoom = true;
    viewer.controls.screenSpacePanning = true; viewer.controls.update();
  }}
  setFrame(0); timeline.disabled = false; play.disabled = false; reset.disabled = false;
  loading.remove();
  (async () => {{
    try {{
      context.textContent += " · loading selected 3D timeline";
      await viewer.addSplatScenes(sources.slice(1).map((path) => ({{
        path, format: Splats.SceneFormat.Splat, splatAlphaRemovalThreshold: 1, visible: false
      }})), false);
      loadedScenes = sources.length;
      timeline.max = String(sources.length - 1);
      setFrame(timeline.value);
      context.textContent += " · 3D timeline ready";
    }} catch (error) {{
      context.textContent += " · 3D timeline failed to load: " + (error.message || error);
    }}
  }})();
  timeline.addEventListener('input', () => {{ stop(); setFrame(timeline.value); }});
  reset.addEventListener('click', () => {{ if (viewer.controls) viewer.controls.reset(); }});
  play.addEventListener('click', () => {{
    if (playing) {{ stop(); return; }} playing = true; play.textContent = 'Pause';
    timer = setTimeout(advance, frameDurations[Number(timeline.value)] * 1000);
  }});
}} catch (error) {{ loading.remove(); context.textContent += " · unable to start 3D player: " + (error.message || error); }}
</script></body></html>"""
    return (
        '<iframe title="Interactive 4D Gaussian scene" sandbox="allow-scripts allow-same-origin" '
        'style="width:100%;height:480px;border:0;border-radius:8px;pointer-events:auto;touch-action:none" '
        f'srcdoc="{escape(document, quote=True)}"></iframe>'
    )

def empty_splat_html() -> str:
    """Keep the 3D panel visibly reserved before an explicit run creates a scene."""
    return ('<div style="height:480px;display:grid;place-items:center;border:2px solid #58748d;'
            'border-radius:10px;background:#101b27;color:#e8f0f5;font:15px system-ui">'
            '</div>')
