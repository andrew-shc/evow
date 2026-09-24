# Text Query and Text Manipulation

These two private LAN dashboard pages operate on a stationary single-camera video.

Text Query accepts one source of up to five minutes. It samples the video at 2 fps for local SigLIP retrieval, ranks non-overlapping four-second windows, grounds the requested subject with Grounding DINO, and propagates a binary mask with SAM2. Results retain the full source frame: yellow fill/contour is an overlay, not a crop. Each saved result stores both the raw source clip and the highlighted clip under `ASSETS/selection/runs/`.

Text Manipulation accepts one upload or raw Text Query clip of at most 81 frames / 6.75 seconds at 12 fps. It tries to ground the target named by the instruction; when that fails, it intentionally uses an all-white mask for a scene-wide environmental or style edit. Wan VACE treats white mask pixels as generated and black pixels as preserved. All output is labeled generated, and fixed-camera geometry is an estimate.

## Owner setup

Install the reproducible environment first, then download snapshots once as the owner:

```bash
conda run -n evow python -m pip install -r CONFIGS/replay_requirements.txt
conda run -n evow python -m GREENFIELD.features.text_setup
```

The setup command loads `HF_TOKEN` from the environment or `.env` without printing it. It pins and records the revisions of SigLIP, Grounding DINO Tiny, SAM2 Hiera Tiny, and Wan VACE 1.3B in `ASSETS/checkpoints/`. The VACE snapshot is large (roughly 19 GB). Browser requests always use local files only; if a snapshot is missing or incomplete, the page reports this same owner command instead of downloading it.

## Explicit 3D cache

The explicit modes reuse the replay depth and dynamic-Gaussian workers for only the ranked/query-selected episode. Complete scenes are atomically cached under `ASSETS/scenes/`, keyed by source SHA-256, temporal bounds, cadence, source FOV, and reconstruction settings. Query highlights are rerendered from selected Gaussian primitive IDs. Explicit edits use that projected mask for VACE, then fit a generated descendant scene; copied `.splat` timeline files live in the edit run so saved viewers remain replayable if the shared cache is later pruned.
