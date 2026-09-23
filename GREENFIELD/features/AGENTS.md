# GREENFIELD/features

Replay-shell dashboard pages and narrowly scoped adapters for Future View, Text
Query, and Text Manipulation. Every page keeps its own feature semantics while
sharing typed traces, artifact handling, and visual language from `app_core`.

## Files

- `pages.py` composes the Gradio controllers; do not put model work in callbacks.
- `future.py`, `selection.py`, and `editing.py` validate requests and emit typed
  stage events/results.
- `model_adapters.py` owns lazy, local-only third-party model loading. It must
  never download weights from a browser request.

## Gotchas

All outputs belong in `ASSETS/<feature>/runs/`. Explicit mode currently means a
per-run reconstruction/projection, never a persistent month-scale scene. Missing
weights must produce an actionable error rather than silently returning a
heuristic result.

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
