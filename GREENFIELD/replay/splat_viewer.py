"""Embed an orbitable, preloaded 4D Gaussian viewer inside the Gradio page."""

from html import escape
import json
from pathlib import Path
from urllib.parse import quote


def splat_html(paths: list[str]) -> str:
    """Build one viewer that preloads every time step before local playback.

    Replacing an iframe for every Gradio slider change forced the browser to
    download and initialize a Gaussian renderer repeatedly. This viewer loads all
    time-indexed scenes once, then toggles GPU scene visibility for each frame.
    """
    if not paths or any(Path(path).suffix != ".splat" for path in paths):
        raise ValueError("The viewer accepts one or more Gaussian .splat frames.")
    sources = ["/gradio_api/file=" + quote(path, safe="/") for path in paths]
    module = "/gradio_api/file=" + quote(str(Path(__file__).with_name("gaussian_viewer.bundle.js")), safe="/")
    document = f"""<!doctype html>
<html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
html,body{{margin:0;width:100%;height:100%;overflow:hidden;background:#101b27;color:#eef4fa;font:14px system-ui,sans-serif}}
#loading{{position:absolute;inset:0;display:grid;place-items:center;z-index:2;pointer-events:none}}#loading span{{width:22px;height:22px;border:3px solid #628197;border-top-color:#eef4fa;border-radius:50%;animation:spin .8s linear infinite}}@keyframes spin{{to{{transform:rotate(360deg)}}}}
#controls{{position:absolute;display:flex;align-items:center;gap:8px;left:12px;right:12px;bottom:12px;z-index:2;background:#101b27dd;padding:8px;border-radius:7px}}
#timeline{{flex:1}} button{{color:#eef4fa;background:#28455a;border:1px solid #628197;border-radius:5px;padding:4px 9px;cursor:pointer}}
</style></head><body><div id="loading" aria-label="Loading 3D scene"><span></span></div>
<div id="controls"><button id="play" disabled>Play</button><input id="timeline" type="range" min="0" max="{len(paths) - 1}" value="0" step="1" disabled><output id="time">Frame 1 / {len(paths)}</output></div>
<script type="module">
import * as Splats from {json.dumps(module)};
const sources = {json.dumps(sources)};
const loading = document.getElementById('loading'), timeline = document.getElementById('timeline');
const play = document.getElementById('play'), time = document.getElementById('time');
let viewer, playing = false, timer;
function setFrame(frame) {{
  const index = Math.max(0, Math.min(sources.length - 1, Number(frame)));
  viewer.splatMesh.scenes.forEach((scene, sceneIndex) => {{ scene.visible = sceneIndex === index; }});
  viewer.splatMesh.updateTransforms();
  timeline.value = String(index); time.textContent = `Frame ${{index + 1}} / ${{sources.length}}`;
}}
function stop() {{ playing = false; clearInterval(timer); play.textContent = 'Play'; }}
try {{
  viewer = new Splats.Viewer({{
    cameraUp: [0, -1, 0], initialCameraPosition: [0, 0, 0], initialCameraLookAt: [0, 0, 4],
    sharedMemoryForWorkers: false, gpuAcceleratedSort: false, ignoreDevicePixelRatio: true,
    dynamicScene: true, enableOptionalEffects: true
  }});
  await viewer.addSplatScenes(sources.map((path, index) => ({{
    path, format: Splats.SceneFormat.Splat, splatAlphaRemovalThreshold: 1, visible: index === 0
  }})), false);
  viewer.start(); setFrame(0); timeline.disabled = false; play.disabled = false;
  loading.remove();
  timeline.addEventListener('input', () => {{ stop(); setFrame(timeline.value); }});
  play.addEventListener('click', () => {{
    if (playing) {{ stop(); return; }} playing = true; play.textContent = 'Pause';
    timer = setInterval(() => setFrame((Number(timeline.value) + 1) % sources.length), 1000 / 12);
  }});
}} catch (error) {{ loading.remove(); }}
</script></body></html>"""
    return (
        '<iframe title="Interactive 4D Gaussian scene" sandbox="allow-scripts allow-same-origin" '
        'style="width:100%;height:480px;border:0;border-radius:8px" '
        f'srcdoc="{escape(document, quote=True)}"></iframe>'
    )


def empty_splat_html() -> str:
    """Keep the 3D panel visibly reserved before an explicit run creates a scene."""
    return ('<div style="height:480px;display:grid;place-items:center;border:2px solid #58748d;'
            'border-radius:10px;background:#101b27;color:#e8f0f5;font:15px system-ui">'
            '</div>')
