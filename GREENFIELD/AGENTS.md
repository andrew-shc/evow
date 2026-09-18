# GREENFIELD/AGENTS.md

This file records the camera application's purpose and domain intent. See the root AGENTS.md for repository conventions and [ROADMAP.md](ROADMAP.md) for the planned experience and build order.

## Motivation

Make a view observed by one camera around the clock explorable: revisit recorded moments from nearby virtual viewpoints, see possible futures, find moments with text, and change recorded scenes with text. The first camera is locked to sky and trees; scene elements may move, while the physical camera stays fixed.

## Domain-specific intent

- The application has four features: 3D video / 4D replay, future view, text selection from the archive, and text editing of 3D video.
- Every feature has two usable modes, for eight modes total. The implicit mode works through video-based generation or understanding. The explicit mode works through a persistent 3D scene representation, such as a mesh or Gaussian splats.
- A feature page shows one mode at a time. Each mode has a **Show internals** toggle for its input, visual processing stages, progress, timings, and result.
- The owner starts runs on demand and sees their stages live. Public visitors explore curated, interactive recordings of completed runs through the same feature pages. Public visitors do not start processing jobs or see the live camera.
- The camera supplies the only visual input. Forecasts may also use date and time. Show a 30-second continuation and a possible view two hours ahead. Text editing covers scene content as well as weather and lighting.
- Identify recorded frames and generated results clearly. A single fixed view cannot directly reveal hidden surfaces, so nearby-view results are estimates.

## Current status and next steps

The first feature, 4D replay, now has a LAN Gradio application in `replay/` with implicit AnyView generation and explicit dynamic Gaussian scenes. It accepts an uploaded or browser-recorded clip, runs one GPU job at a time, saves stage traces and results, and can replay completed runs. Continuous capture, the month archive, the remaining six modes, and the curated public showcase are still planned in [ROADMAP.md](ROADMAP.md).

## Pipeline map

Camera → timestamped archive → video features and persistent 3D scene → queued mode run → stage trace and result → private live dashboard or curated public replay.

Keep recordings, model outputs, traces, and public replay assets under root ASSETS/. Keep declarative parameters under CONFIGS/ and new executable code under GREENFIELD/.

## Known issues

- The fixed camera provides no direct observation of geometry behind visible surfaces. The interface must not present inferred geometry as captured fact.
- Processing all eight modes on one workstation requires a queue that lets capture continue during expensive runs.

## Results

Replay runs and output assets are saved under `ASSETS/replay/runs/` and excluded from git. See `replay/README.md` for result structure and limitations.

## Run Protocols

Run the private replay app with `conda activate evow` and `python -m GREENFIELD.replay.app` from the repo root. It listens on the current LAN address `192.168.4.254:7860` by default; `EVOW_BIND_HOST` overrides the bind address. Successful runs save their sampled input, ordered stage updates, timings, meshes or generated video, and final result. Curate completed traces for the later public application.
