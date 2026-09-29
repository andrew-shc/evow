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


## Internal execution flow

Feature generators must yield the terminal error trace once before raising `gr.Error`; otherwise the browser retains the previous running stage. Every emitted stage must carry a `stage_id` declared on the feature's `STAGES` (or the page renders an orphan row): fold artifact-saving and similar tail work into a declared stage rather than inventing a new name, and pass the matching `stage_id` through any worker wrapper that emits under a stage label. Each feature declares its parent groups as `SuperStage` objects owning those **stage ids**, plus a frozen `LEGACY_STAGE_IDS` tuple of the v1 name-only identities and a per-version `LEGACY_PROFILES` map (each pre-current `flow_version` to the exact current-id set it could record), so a saved run is never resolved against a newer profile and a v2 union row is never mislabeled "not recorded". Feature stages are serial unless a feature explicitly documents concurrency. The shared renderer permits only the newest configured stage to appear running and maps `error` to the dashboard’s red `evow-stage-failed` class; a saved, supported trace (current *or* legacy) whose last tick is still `running` is derived as "Interrupted" (never mutated on disk), while an *earlier* pre-current running tick whose stage was superseded keeps the serial "completed before the next stage" repair. Whether a trace was loaded from disk must be threaded into `pages._updates` as an explicit `is_saved` flag from the load-saved callbacks, never inferred from `result`: a failed or partial saved run legitimately has `result=None`, and a live trace has no `result` until its final yield, so `result` cannot tell the two apart. Method-inapplicable stages and a reused self-trained adapter's "Prepare training data" stage emit a terminal `skipped` status rendered as "Skipped", never a misleading "Waiting". A failure after the active stage already reached a terminal emits only the generic run-failure event, never a second terminal for that stage.

## Text Query flow invariants

Text Query declares one 22-row union with stable snake_case `stage_id`s and five explicit `SuperStage` parents owning those ids (never positional indexes): Inspect source; the seven Retrieve rows; the five Ground + track rows; the six Resolve 3D method rows; and the three Save rows. Keep the ids stable: persisted traces, the per-version `LEGACY_PROFILES`, and the adapter stage-id maps all key off them.

Applicability is declared on the flow, not the renderer. `resolve_method` is implicit-only; `scene_cache`, `scene_reconstruct`, `mask_lift`, `mask_project`, and `splat_export` are explicit-only. The renderer derives "Skipped" for those mode-inapplicable rows on the **current (v4)** profile only. For a legacy trace, resolve historical applicability from that version's `legacy_profiles` entry *before* the current mode rules: a row the version could record but has no event for is historical Waiting/Interrupted, never "Not used by this method"; only a row genuinely absent from that version's profile is "Not recorded". `LEGACY_PROFILES` must map every pre-v4 version (1, 2, and 3) to its five recorded ids (`inspect_source`, `window_rank`, `mask_validate`, `resolve_method`, `write_clips`); a missing entry makes v3 traces mislabel valid rows.

`selection.run` is an aggregate emitter: every looping row weaves `running` ticks and exactly one caller-owned terminal after the loop, so a producer must never emit its own terminal for an aggregate row. Zero candidates and zero retrieval frames both flow through as an empty pool (skip `ranked_windows`, which rejects empty scores) and still terminate every downstream row before raising the user-facing "no matches" error; the page's duplicate-terminal guard then suppresses a second terminal. For explicit mode the `scene_cache` terminal aggregates `checked_candidates`/`hit_candidates`/`miss_candidates` and reports `cache` as one of `none`/`hit`/`miss`/`mixed`; `scene_reconstruct` claims no scene (no `splats`) when nothing was checked.

`ensure_scene_steps` is tick-only: a cache hit returns immediately, and a build yields progress but never a terminal, so the caller owns the single `scene_reconstruct` terminal and reads `CachedScene.cache_hit` to report a reuse truthfully. Pass each feature's stage-id map into the shared adapters (`semantic_scores_steps`, `ground_and_track_steps`, `scene_cache.ensure_scene_steps`) rather than hardcoding ids in the adapters.

## Text Manipulation flow invariants

Text Manipulation declares one 19-row union with stable snake_case `stage_id`s and five explicit `SuperStage` parents owning those ids (never positional indexes): Input (`episode_decode`, `vace_trim`); Resolve scope (`dino_detect`, `sam2_propagate`, `scope_classify`, `mask_select`, `mask_expand`); Prepare method (`prepare_mask`, `scene_cache`, `mask_project`, `mask_expand_projected`); Generate (`vace_prepare`, `vace_denoise`, `vace_decode`, `write_proposal`); Save (`finalize`, `edited_scene`, `splat_copy`, `publish_artifacts`). Keep the ids stable: persisted traces, the per-version `LEGACY_PROFILES`, and the alias map all key off them. Declared order **is** execution order, so a parent never rolls up a child that runs after a later group.

Mask expansion is deliberately split into two mode-scoped rows because the two methods expand on opposite sides of `mask_project`: `mask_expand` (Resolve scope) dilates the tracked 2D mask *before* `prepare_mask` for Implicit 3D, while `mask_expand_projected` (Prepare method, after `mask_project`) dilates the projected mask for Explicit 3D. Explicit must lift the **unexpanded** tracked mask, so `base_primitive_mask.npy` stays the raw primitive selection and only the projected result is dilated into `edit_mask.npy`; do not collapse the two rows into one or expand `direct_masks` before `lift_masks`, which would change the selected Gaussians, the saved primitive mask, fallback behavior, and the final conditioning.

Applicability is declared on the flow, not the renderer. `mask_expand`, `prepare_mask`, and `finalize` are implicit-only; `scene_cache`, `mask_project`, `mask_expand_projected`, `edited_scene`, and `splat_copy` are explicit-only. The renderer derives "Skipped" for those mode-inapplicable rows on the current (v5) profile only, so producers must never hand-emit a duplicate skip. For a legacy trace, historical applicability comes from that version's `legacy_profiles` entry *before* the current mode rules.

`LEGACY_PROFILES` maps every pre-v5 version (1, 2, 3, and 4) to the same five coarse bars (`inspect_episode`, `resolve_scope`, `prepare_method`, `generate_edit`, `save_result`); v4's atomization landed with the v5 global bump, so a missing `4` entry would mislabel a valid v4 trace's rows "Not recorded". `prepare_method` is Text Manipulation's own retired id here, not Future View's global `svd_load` alias, which is why the retired identities live in the feature-scoped `RETIRED_STAGE_ALIASES_BY_FEATURE` (consulted before `RETIRED_STAGE_ALIASES`).

Grounding and tracking relay `text_segmentation.ground_and_track_steps` but declare only `dino_detect` and `sam2_propagate`; the helper's `sam2_setup`/`mask_validate` rows map to blank ids and are dropped, and `editing.run` owns both terminals. `vace_denoise` is an aggregate: `edit_video_steps` streams its throttled running ticks live from a worker thread (first/last/every-Nth, real `step`/`total`/`timestep`/`latent_shape`) and `editing.run` owns its single terminal just before `vace_decode`. `total` is the scheduler's real timestep count; when the pipeline exposes no honest total, exactly one coarse tick carrying only `frames` is emitted and `step + 1` is never fabricated.

`scene_cache` and `edited_scene` each relay the tick-only `ensure_scene_steps` and own exactly one terminal via `_relay_scene_ticks`/`_scene_terminal`: a cache hit completes with no running tick, a miss relays the build's ticks then completes. `edited_scene` copies `scene.rendered` before emitting its terminal, and `splat_copy` copies run-owned splats. `publish_artifacts` is owned by the editing page handler around `trace.finish`, mirroring Future View's `write_trace`.

## CUDA recovery

The self-trained implicit Future View method stores source-hash and settings-bound LoRA artifacts under `ASSETS/future/training/`. Never use the final inference-history tail to train or validate an adapter; only a complete manifest may be loaded into that method’s forecast. Its training-only 512px long-edge ceiling is part of the fingerprint: it exists because a 640×480, 14-frame LoRA backward pass exhausts the 24 GB owner GPU, while forecast resolution remains independent. Every training-affecting knob (optimizer steps, LoRA rank, temporal-pair count, and the training long-edge ceiling) must stay in `training_key`; adding a field there intentionally invalidates previously cached artifacts once so a changed setting can never reuse a stale adapter.

The SVD checkpoint uses a continuous Euler `v_prediction` scheduler. A training worker must call `set_timesteps`, sample values from `scheduler.timesteps` (not integer indices), use stateless Euler input scaling for randomly ordered batches, and train against `get_velocity`; otherwise the worker either crashes or learns a mismatched target.

Implicit Future View uses Diffusers CPU offload and 14-frame chunks. A PEFT adapter must be installed only after removing the base UNet's Accelerate hooks, then receive freshly rebuilt hooks; restore that sequence before returning the cached pipeline to base SVD. Wrapping an already hooked UNet causes the misleading `PeftModel._hf_hook` failure after otherwise successful denoising. The owner-facing **Clear GPU memory** control lives in the dashboard-wide top bar (shown on every tab, not inside any one project page) and releases only this dashboard process’s cached models after an OOM; it cannot free GPU memory held by another process. Because that button is wired with `queue=False`, it bypasses the generation queue and can run concurrently with an in-flight generation, so `model_adapters._MODEL_LIFECYCLE_LOCK` (a module-level **ownership-free semaphore**, not an `RLock`) is held for the generator’s whole lifetime and for all of `clear_gpu_memory()`; do not remove it, and do not make it thread-owned. Gradio may advance the sync generator for each `next()` on a different threadpool thread and close it from the event-loop thread, so a reentrant lock would raise on the cross-thread release in the generator’s outer `finally` and deadlock every later generation and Clear-GPU call. Because the semaphore is not reentrant, generator teardown must call the unlocked `_clear_gpu_memory_unlocked()` helper directly rather than the locking public `clear_gpu_memory()` wrapper. Teardown failures must never replace an already-propagating inference error (they are logged when one is in flight, and re-raised only on clean completion).

## Text modes

- `text_media.py` owns duration checks and exact, non-cropping RGB interval decoding. Do not route long uploads through generic `read_video()`.
- `text_models.py` loads SigLIP, Grounding DINO, and SAM2 only from owner-installed snapshots; `text_setup.py` pins their Hugging Face revisions and Wan VACE under `ASSETS/checkpoints/`.
- `text_retrieval.py` performs window smoothing and temporal NMS. `text_segmentation.py` turns grounded boxes into SAM2 masks and applies only overlay rendering.
- `scene_cache.py` is the common fixed-camera dynamic-Gaussian facade. A cache entry is valid only after its manifest appears atomically; never save feature-specific output inside the shared cache.
- `scene_masks.py` moves a tracked region between 2D masks and Gaussian primitive identities. It is a camera-view estimate, not semantic ground truth.
- `video_edit.py` is the only Text Manipulation generator. Wan VACE receives white generated / black preserved masks; do not reintroduce frame-by-frame InstructPix2Pix. `edit_video` is the positional-compatible list wrapper; `edit_video_steps` is the streaming adapter that yields real `vace_prepare`/`vace_denoise`/`vace_decode` rows.

## Text-mode gotchas

Text Query accepts one source of at most five minutes, samples retrieval at 2 fps, and returns up to five non-overlapping four-second clips. Text Manipulation accepts at most 81 frames / 6.75 seconds at 12 fps. The explicit scene cache is episode-scale by design, not a month-scale reconstruction. Query handoff must use `ClipArtifact.source`, never `ClipArtifact.highlighted`. Explicit saved edit runs copy their splat timeline out of the shared cache so saved viewers remain valid if a cache is pruned.
