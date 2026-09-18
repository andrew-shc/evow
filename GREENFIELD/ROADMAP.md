# Eight-mode camera application roadmap

## Experience

One fixed camera records a sky-and-trees view continuously. The application makes a one-month archive available for text selection, 3D replay, future views, and text editing. Each feature has an implicit video-based mode and an explicit mode that uses a persistent 3D scene. These are eight working modes of one application.

Each feature page displays one mode at a time. A mode switch selects **Implicit** or **Explicit 3D**. A separate **Show internals** toggle on each mode reveals the source material, visual stages, progress, stage timings, and final result. Switching modes keeps the selected clip, query, prompt, viewpoint, or forecast horizon so the two results can be viewed against the same request.

The owner's private dashboard starts runs on demand and updates their stages while they execute. Runs share one workstation and enter a queue; recording continues independently. Every run saves a trace that the owner can replay. The public application offers the same feature and internals controls over curated recorded traces. Its controls scrub saved stages and results; they do not trigger a run or reveal the live camera.

Label camera recordings, generated frames, and reconstructed renders in both dashboards. Nearby viewpoints and geometry inferred from one fixed view are estimates. The two-hour view is one possible future, rather than a guaranteed weather prediction.

## Eight modes and working examples

| Feature and mode | Working example | Internals shown |
| --- | --- | --- |
| 3D video / 4D replay: Implicit | Select a short recorded clip and request a nearby virtual viewpoint; generate a view-shifted video. | Source frames, requested viewpoint, generation previews, completed video, progress and timings. |
| 3D video / 4D replay: Explicit 3D | Use the same clip and viewpoint; build or load a time-varying 3D scene and render the shifted video. | Source frames, estimated geometry and motion, scene preview, rendered frames, progress and timings. |
| Future view: Implicit | Use recent camera frames and clock time to generate a 30-second continuation and a possible view two hours ahead. | Input history, generation previews, future frames, progress and timings. |
| Future view: Explicit 3D | Use the same history and horizons; advance the 3D scene state and render possible future views. | Current geometry and state, projected motion or conditions, rendered views, progress and timings. |
| Text selection: Implicit | Search the month-long archive for “dark clouds above moving tree branches” and return matching time intervals. | Query, searched clip windows, match scores, ranked intervals, progress and timings. |
| Text selection: Explicit 3D | Use the same query to search time-indexed, text-addressable features attached to a persistent 3D scene. | Query, matched 3D regions and time windows, ranked intervals, progress and timings. |
| Text editing: Implicit | On a chosen clip, demonstrate a scene-content edit and a weather or lighting edit using text. | Source clip and prompt, affected video regions, edit previews, completed video, progress and timings. |
| Text editing: Explicit 3D | Apply the same prompts to the 3D scene and render the edited clip. | Source scene and prompt, selected 3D regions or conditions, updated scene, rendered frames, progress and timings. |

The explicit text-selection mode uses a persistent 3D scene with time-indexed changes, so it can search the archive without reconstructing a separate full scene for every frame. All eight modes expose visual stages and timings; the table names the stages each mode must make visible.

## Shared capture, processing, and replay

1. Record timestamped footage continuously. Preserve source clip boundaries and recording gaps so selections and traces refer to exact archive intervals. Size the archive and text index for at least one month.
2. Maintain the video features needed by implicit modes and a reusable 3D scene with time-indexed state or features for explicit modes. Only the camera provides visual observations; forecast modes may additionally use date and time.
3. Queue owner-requested jobs on the single workstation. A run identifies its feature, mode, source interval, and request parameters, then emits ordered stage updates with status, progress, preview assets, timings, and a final result. The dashboard renders these updates as they arrive and from the saved trace afterward.
4. Store raw recordings, derived scene data, outputs, and traces under root ASSETS/. Put declarative settings under CONFIGS/; put new executable code under GREENFIELD/ and add a directory AGENTS.md when creating a code subdirectory.
5. Curate at least one completed trace for each mode for public replay. The public application reads published trace assets and the searchable recorded archive. It has no control that starts a job or opens the live camera.

## Build order

1. **Capture and archive:** Install the fixed camera, verify continuous recording and timestamps, and build a searchable one-month archive.
2. **Dashboard and traces:** Build the four feature pages, method switch, internals toggle, owner job queue, live stage updates, saved trace replay, and public replay from curated traces. Exercise the interface with sample traces before models are connected.
3. **Scene and mode adapters:** Build reusable video features and a persistent 3D scene, then connect the eight modes. Complete text selection and future view first, followed by 4D replay and text editing.
4. **Public showcase:** Curate the eight working examples, check that recorded and generated material is labeled, and publish the recorded experience without exposing private run controls.

## Verification

- Each of the eight modes completes its working example and produces a result plus a replayable trace containing the stages listed above.
- All four feature pages preserve the request when switching methods; the internals toggle shows and hides the currently selected mode's stages.
- A private run updates stages live, then replays the same ordered stages after completion. Queued and running jobs do not stop capture.
- Text selection returns playable intervals from the month-long archive. Forecasts use camera history and date/time only. Every explicit mode reads or updates the persistent 3D scene.
- The public application contains curated interactive traces for all eight modes. Its controls only read recorded assets, and it shows neither live camera footage nor private job controls.
