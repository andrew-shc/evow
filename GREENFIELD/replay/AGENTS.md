# GREENFIELD/replay

This directory owns the first application feature: replay a short monocular video from a nearby virtual viewpoint. The LAN Gradio app exposes implicit video generation and explicit time-varying 3D Gaussian scenes.

## Files

- app.py builds the LAN interface and streams stage updates.
- run.py coordinates input preparation, a selected backend, and replayable traces.
- clip.py extracts the same short episode for both backends. The application records the stationary-view assumption but does not validate camera motion.
- pose.py defines the shared virtual camera request.
- overrides.py is the per-run override boundary: `ReplayOverrides` validates and applies every exposed knob, `settings_json` feeds the isolated workers, and `reconstruction_fingerprint` keys explicit scene caches. Keep its field names identical to `Settings` field names so `dataclasses.replace` and worker JSON stay in lockstep.
- implicit.py and anyview_worker.py adapt the external AnyView model.
- explicit.py, depth_worker.py, depth_prior.py, gaussian_worker.py, and splat.py initialize and fit a native 4D Gaussian scene. splat_viewer.py embeds an orbitable Three.js viewer; gaussian_viewer.bundle.js is the locally bundled upstream viewer library.
- splat_viewer.py fully preloads every supplied .splat frame before enabling its controls. It partitions the timeline into 32-scene static renderer groups, swaps only the active group, and preserves the orbit camera at group boundaries; do not reconnect it to a Gradio change handler, because that rebuilds the iframe and interrupts animation.
- workflow.py turns each V2 run trace into a script-free, expandable execution flow. Every row maps to one stable `stage_id`; worker subprocesses publish ordered events through an atomic JSON sidecar, and every serial stage must emit a terminal state before the next starts. The expanded card exposes source excerpt, data contract, previews, bounded worker-log tail, and raw event JSON. Old traces stay raw legacy evidence rather than receiving fabricated atomic states. A `skipped` status renders as "Skipped" with no duration. Do not reintroduce an iframe or JavaScript here: Gradio streams this component during execution.
- trace.py and settings.py hold the shared run record and paths.

## Gotchas

- AnyView accepts only 1 + 4k frames, up to 41. Use 13, 29, or 41 frames in the UI.
- The application assumes a stationary physical viewpoint and deliberately does not validate it. Keep the source view stationary when using either reconstruction method.
- A fixed camera does not observe hidden surfaces. Both virtual views are estimates; explicit initialization depth is relative, not metric; the fitted scene cannot recover unseen surfaces from a fixed view.
- Run GPU models in separate processes so one backend releases GPU memory before the next.
- Put every uploaded clip, checkpoint, depth map, model file, render, trace, and sample video under root ASSETS/. The external checkouts are code only.
- The external AnyView weights and code have noncommercial license terms; review them before any public or commercial deployment.
- Critical Future View constraint: gaussian_viewer.bundle.js declares `Constants.MaxScenes = 32` and encodes scene transforms/visibility as shader uniform arrays. A normal 189-frame run cannot be one renderer scene list; raising a Python cap (such as 240) does not change that limit. Keep every frame resident by splitting it into <=32-scene static viewers, disable controls until every group loads, and hand off the orbit camera at group boundaries. System RAM does not remove this WebGL/GPU shader limit.
- Do not mutate browser scene buffers during playback. The bundled renderer supports only 32 scenes per shader, so the shared viewer keeps all timeline frames resident across bounded groups and runs only the active group.
