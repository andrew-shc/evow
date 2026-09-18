# BROWNFIELD.md

Tracks vendored checkouts that the application calls. Their source is left intact; all generated data and weights live under the ignored root `ASSETS/` tree.

Status legend: not started | partial | migrated

| Folder | Status | Notes |
|---|---|---|
| `AnyView-DVS/` | partial | Official AnyView checkout used by the implicit 4D replay worker. Weights and generated episodes/results are outside the checkout in `ASSETS/checkpoints/anyview/` and `ASSETS/replay/runs/`. Upstream examples and their default output conventions are untouched. |
| `Video-Depth-Anything/` | partial | Official Video Depth Anything checkout used by the explicit depth worker. The checkpoint, inferred depths, meshes, and renders are in `ASSETS/`. Upstream examples and output conventions are untouched. |

## Change log

- 2026-09-16: Added the two upstream checkouts to drive the first replay feature without modifying their source. Greenfield wrappers redirect checkpoints and outputs to `ASSETS/`; model and capture parameters live in `CONFIGS/replay.yaml`. Both folders remain **partial** because the upstream example commands have not been migrated.
