# GREENFIELD/features

Replay-shell dashboard pages and narrowly scoped adapters for Future View, Text
Query, and Text Manipulation. Every page keeps its own feature semantics while
sharing typed traces, artifact handling, and visual language from `app_core`.

## Files

- `pages.py` composes the Gradio controllers; do not put model work in callbacks.
- `future.py`, `selection.py`, and `editing.py` validate requests and emit typed
  stage events/results.
- `model_adapters.py` owns Future View’s lazy, local-only model loading. It must
  never download weights from a browser request.

## Gotchas

All outputs belong in `ASSETS/<feature>/runs/`. Explicit scenes are per-run or
episode-cache assets, never a persistent month-scale scene. Missing weights must
produce an actionable error rather than silently returning a heuristic result.

## Future View

Future View samples up to the latest 7.5 seconds at 12 fps and always writes a
30-second, 6 fps generated scenario. `future_setup.py` is an owner-only model
download command; request handlers always use local model files. Explicit mode
uses a per-run Gaussian scene and a constant-velocity motion prior, which must
never be described as a reliable physical or weather prediction. The browser scene
viewer preloads every exported temporal frame before enabling playback. Because the bundled
renderer has a 32-scene shader ceiling, it holds all frames in bounded static renderer
groups and swaps only fully resident groups at playback boundaries. Do not stream add/remove
operations while playback runs: this renderer rebuilds its scene buffer for those mutations.


## CUDA recovery

Implicit Future View uses Diffusers CPU offload and 14-frame chunks. The owner-facing **Clear GPU memory** control releases only this dashboard process’s cached models after an OOM; it cannot free GPU memory held by another process.

## Text modes

- `text_media.py` owns duration checks and exact, non-cropping RGB interval decoding. Do not route long uploads through generic `read_video()`.
- `text_models.py` loads SigLIP, Grounding DINO, and SAM2 only from owner-installed snapshots; `text_setup.py` pins their Hugging Face revisions and Wan VACE under `ASSETS/checkpoints/`.
- `text_retrieval.py` performs window smoothing and temporal NMS. `text_segmentation.py` turns grounded boxes into SAM2 masks and applies only overlay rendering.
- `scene_cache.py` is the common fixed-camera dynamic-Gaussian facade. A cache entry is valid only after its manifest appears atomically; never save feature-specific output inside the shared cache.
- `scene_masks.py` moves a tracked region between 2D masks and Gaussian primitive identities. It is a camera-view estimate, not semantic ground truth.
- `video_edit.py` is the only Text Manipulation generator. Wan VACE receives white generated / black preserved masks; do not reintroduce frame-by-frame InstructPix2Pix.

## Text-mode gotchas

Text Query accepts one source of at most five minutes, samples retrieval at 2 fps, and returns up to five non-overlapping four-second clips. Text Manipulation accepts at most 81 frames / 6.75 seconds at 12 fps. The explicit scene cache is episode-scale by design, not a month-scale reconstruction. Query handoff must use `ClipArtifact.source`, never `ClipArtifact.highlighted`. Explicit saved edit runs copy their splat timeline out of the shared cache so saved viewers remain valid if a cache is pruned.
