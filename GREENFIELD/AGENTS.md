# GREENFIELD/AGENTS.md

This file records the camera application's purpose and domain intent. See the root AGENTS.md for repository conventions and [ROADMAP.md](ROADMAP.md) for the planned experience and build order.

## Motivation

Make a view observed by one camera around the clock explorable: revisit recorded moments from nearby virtual viewpoints, see possible futures, find moments with text, and change recorded scenes with text. The first camera is locked to sky and trees; scene elements may move, while the physical camera stays fixed.

## Domain-specific intent

- The application has four features: 3D video / 4D replay, future view, text selection from the archive, and text editing of 3D video.
- Every feature has two usable modes, for eight modes total. The implicit mode works through video-based generation or understanding. The explicit mode works through a persistent 3D scene representation, such as a mesh or Gaussian splats.
- A feature page shows one mode at a time. Each mode has a **Show internals** toggle for its input, visual processing stages, progress, timings, and result.
- The owner starts runs on demand and sees their stages live. Public visitors explore curated, interactive recordings of completed runs through the same feature pages. Public visitors do not start processing jobs or see the live camera.
- **Methodology invariant:** every feature presents the same two solution families in this order: **Explicit 3D** first, then **Implicit 3D**. Feature-specific words follow the colon; never replace the shared 3D terminology with a different primary label.
- The camera supplies the only visual input. Forecasts may also use date and time. Show a 30-second continuation and a possible view two hours ahead. Text editing covers scene content as well as weather and lighting.
- Identify recorded frames and generated results clearly. A single fixed view cannot directly reveal hidden surfaces, so nearby-view results are estimates.

## Current status and next steps

All four feature pages now have a private LAN implementation: 4D Replay, Future View, Text Query, and Text Manipulation. Replay uses implicit AnyView and explicit dynamic Gaussian scenes; the text modes reuse those scene workers alongside local SigLIP, Grounding DINO, SAM2, and Wan VACE adapters. Each page saves a replayable trace and run-owned artifacts. Continuous capture, the month archive, and the curated public showcase remain planned in [ROADMAP.md](ROADMAP.md).

## Pipeline map

Camera → timestamped archive → video features and persistent 3D scene → queued mode run → stage trace and result → private live dashboard or curated public replay.

Keep recordings, model outputs, traces, and public replay assets under root ASSETS/. Keep declarative parameters under CONFIGS/ and new executable code under GREENFIELD/.

## Known issues

- The fixed camera provides no direct observation of geometry behind visible surfaces. The interface must not present inferred geometry as captured fact.
- Processing all eight modes on one workstation requires a queue that lets capture continue during expensive runs.
- **Dashboard styling:** the dashboard is a top-level `gr.Blocks` that hosts a persistent top bar above manually built `gr.Tabs()`, merging each feature page with `Blocks.render()`. `render()` drops every child page's `css=`/`js=`, so only the root `FLOW_HEADER_CSS` in `dashboard.py` is mounted. Shared dashboard CSS — including `.evow-help` and `.evow-topbar` styling — must live there; rules in `replay/app.py`'s `REPLAY_CSS` or `app_core/ui.py`'s `APP_CSS` never reach the served page. After editing any UI code, restart the Gradio process: LAN clients otherwise keep seeing the stale build.

## Results

Replay runs and output assets are saved under `ASSETS/replay/runs/` and excluded from git. See `replay/README.md` for result structure and limitations.

## Run Protocols

Run the private replay app with `conda activate evow` and `python -m GREENFIELD.replay.app` from the repo root. It listens on the current LAN address `192.168.4.254:7860` by default; `EVOW_BIND_HOST` overrides the bind address. Successful runs save their sampled input, ordered stage updates, timings, meshes or generated video, and final result. Curate completed traces for the later public application.

## Text Query and Text Manipulation

Text Query searches one fixed-camera source up to five minutes and writes up to five non-overlapping, four-second raw and highlighted clips under `ASSETS/selection/runs/`. Its yellow binary overlay is presentation-only; the raw clip is the only query handoff accepted by Text Manipulation. Text Manipulation accepts an upload or raw query clip up to 81 frames at 12 fps, writes generated outputs under `ASSETS/editing/runs/`, and labels them as generated. Explicit text runs cache complete episode-scale Gaussian scenes atomically under `ASSETS/scenes/`, keyed by source content and reconstruction settings.

The owner installs text-model snapshots with `conda run -n evow python -m GREENFIELD.features.text_setup`; browser requests use local files only and report that command when a snapshot is absent or incomplete. All text-mode geometry is estimated from the single fixed camera and must never be presented as observed hidden structure.
