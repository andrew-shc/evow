# GREENFIELD/app_core

Shared, framework-facing building blocks for every private dashboard feature.

- `contracts.py` holds small typed request, artifact, event, and result contracts.
- `trace.py` persists versioned, replayable feature runs. Do not put feature-specific
  pipeline logic here.
- `media.py` owns bounded video decoding and browser-playable artifact writing.
- `models.py` owns optional dependency and checkpoint diagnostics; model adapters
  must not download weights during a dashboard request.
- `splat_viewer.py` is the public reusable explicit-3D viewer API. Keep its browser
  controls minimal: timeline and Play/Pause only; never add a camera-pose reset.
- `ui.py` contains the visual language shared by all Gradio pages.

All run data belongs below `ASSETS/`. This directory must remain light: it is the
stable seam between UI controllers and feature adapters, not a second pipeline.
