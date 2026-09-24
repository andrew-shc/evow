# BROWNFIELD.md

Tracks vendored checkouts that the application calls. Their source is left intact; all generated data and weights live under the ignored root `ASSETS/` tree.

Status legend: not started | partial | migrated

| Folder | Status | Notes |
|---|---|---|
| `AnyView-DVS/` | partial | Official AnyView checkout used by the implicit 4D replay worker. Weights and generated episodes/results are outside the checkout in `ASSETS/checkpoints/anyview/` and `ASSETS/replay/runs/`. Upstream examples and their default output conventions are untouched. |
| `Video-Depth-Anything/` | partial | Official Video Depth Anything checkout used by the explicit depth worker. The checkpoint, inferred depths, meshes, and renders are in `ASSETS/`. Upstream examples and output conventions are untouched. |

## Change log

- 2026-09-16: Added the two upstream checkouts to drive the first replay feature without modifying their source. Greenfield wrappers redirect checkpoints and outputs to `ASSETS/`; model and capture parameters live in `CONFIGS/replay.yaml`. Both folders remain **partial** because the upstream example commands have not been migrated.

- 2026-09-22: Added declarative local Diffusers, Transformers/SigLIP, and FAISS feature adapters. Model setup remains an explicit owner action and all generated data stays under `ASSETS/`.
- 2026-09-22: Future View explicit forecasting reuses the Video Depth Anything and gsplat replay workers with all per-run scene data redirected to `ASSETS/future/runs/`; both vendored folders remain **partial**.
- 2026-09-23: Text Query and Text Manipulation explicit modes call the existing Video Depth Anything/gsplat Greenfield workers through an episode-scoped atomic cache at `ASSETS/scenes/`. No vendored source was changed; raw/highlighted clips, generated edits, and copied viewer splats remain under their owning `ASSETS/<feature>/runs/` directories. `Video-Depth-Anything/` remains **partial**.
