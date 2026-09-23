# GREENFIELD/replay

This directory owns the first application feature: replay a short monocular video from a nearby virtual viewpoint. The LAN Gradio app exposes implicit video generation and explicit time-varying 3D Gaussian scenes.

## Files

- app.py builds the LAN interface and streams stage updates.
- run.py coordinates input preparation, a selected backend, and replayable traces.
- clip.py extracts the same short episode for both backends. The application records the stationary-view assumption but does not validate camera motion.
- pose.py defines the shared virtual camera request.
- implicit.py and anyview_worker.py adapt the external AnyView model.
- explicit.py, depth_worker.py, depth_prior.py, gaussian_worker.py, and splat.py initialize and fit a native 4D Gaussian scene. splat_viewer.py embeds an orbitable Three.js viewer; gaussian_viewer.bundle.js is the locally bundled upstream viewer library.
- splat_viewer.py starts with one .splat frame, then preloads the already-bounded selected timeline in one batch. Its own timeline and Play control swap scene visibility; do not reconnect it to a Gradio change handler, because that rebuilds the iframe and interrupts animation.
- workflow.py turns each recorded run trace into a script-free, expandable execution flow. Each stage summary opens once into a program contract and live run record, including source file, function chain, data contract, linked left-side control, preview, metrics, request settings, and raw event JSON. Do not reintroduce an iframe or JavaScript here: Gradio streams this component during execution.
- trace.py and settings.py hold the shared run record and paths.

## Gotchas

- AnyView accepts only 1 + 4k frames, up to 41. Use 13, 29, or 41 frames in the UI.
- The application assumes a stationary physical viewpoint and deliberately does not validate it. Keep the source view stationary when using either reconstruction method.
- A fixed camera does not observe hidden surfaces. Both virtual views are estimates; explicit initialization depth is relative, not metric; the fitted scene cannot recover unseen surfaces from a fixed view.
- Run GPU models in separate processes so one backend releases GPU memory before the next.
- Put every uploaded clip, checkpoint, depth map, model file, render, trace, and sample video under root ASSETS/. The external checkouts are code only.
- The external AnyView weights and code have noncommercial license terms; review them before any public or commercial deployment.
- Do not mutate browser scene buffers during playback. Long Future View timelines are downsampled server-side before the viewer preloads its selected scenes in one batch.
