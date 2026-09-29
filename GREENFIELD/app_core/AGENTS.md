# GREENFIELD/app_core

Shared, framework-facing building blocks for every private dashboard feature.

- `contracts.py` holds small typed request, artifact, event, and result contracts.
- `trace.py` persists versioned, replayable feature runs. Do not put feature-specific
  pipeline logic here.
- `media.py` owns bounded video decoding and browser-playable artifact writing.
- `models.py` owns optional dependency and checkpoint diagnostics; model adapters
  must not download weights during a dashboard request.
- `live_camera.py` detects locally attached Linux V4L2 devices, reads preview frames, and writes bounded source clips below `ASSETS/live_camera/captures/`. Keep browser-webcam capture separate: it belongs to the client browser, while this module owns camera hardware attached to the dashboard host.
- `splat_viewer.py` is the public reusable explicit-3D viewer API. Keep its browser
  controls minimal: timeline and Play/Pause only; never add a camera-pose reset.
- `timing.py` is the single source of truth for persisted and rendered
  stopwatch precision. Keep elapsed values numeric and rounded to three decimal
  places for trace compatibility.
- `flow.py` owns the stage-identity schema: it splits a stage's stable `stage_id`
  from its human `name`, models `FlowStage`/`FeatureFlow`, versions profiles
  (`flow_version`), and normalizes legacy name-only events. `FeatureFlow` also
  carries the frozen v1 `legacy_stage_ids` display names and a per-version
  `legacy_profiles` map (each pre-current version to the exact current-id set it
  could have recorded); `resolve_flow` maps a saved run's persisted identity onto
  the registry's current flow (`FlowResolution`: current flow, the ids that
  exact version recorded, and a human-readable notice). A trace older than
  `CURRENT_FLOW_VERSION` maps best-effort — an exact current `stage_id` wins,
  then a feature-scoped retired alias
  (`RETIRED_STAGE_ALIASES_BY_FEATURE`), then the global `RETIRED_STAGE_ALIASES`,
  then the unique display-name match — and is named as a legacy version in the
  notice; an unknown pre-current version exposes no legacy ids so the renderer
  falls back to the v1 name set. A newer version is uninterpretable (all-waiting,
  no events mapped). A stage id that looks like a number is a real id and must
  never be reinterpreted as a positional index. `CURRENT_FLOW_VERSION` is 5: v4
  atomized Text Query's five coarse bars into 22 rows and v5 atomized Text
  Manipulation's five bars into 19 rows, so **every** pre-current version each
  feature could have written needs its own `legacy_profiles` entry (Future View
  1–4, Text Query 1–4, Text Manipulation 1–4). The feature-scoped alias map
  exists because a retired identity is otherwise ambiguous: Future View retired
  `prepare_method` onto `svd_load` globally, while Text Manipulation owned
  `prepare_method` as a current id until v4 and now retires it onto its own
  `prepare_mask`.
  Keep it pure (no feature imports) so `trace.py` can import it without a cycle.
- `super_stages.py` owns parent-stage group declarations, lifecycle rollups, and
  rendered status/timing updates. Feature pages declare only their named groups
  and child **stage IDs** — never positional indexes, which would silently
  regroup a saved run whose flow changed. Never duplicate status precedence or
  group CSS locally.
- Saved-run dropdowns in `ui.py` must explicitly use `value=None`; Gradio
  otherwise preselects the newest run and prevents its first selection callback.
- Config knob helpers (`config_number`/`config_slider`/`config_textbox`) must keep
  `container=True` and `show_label=True`. In Gradio 5.50 `container=False` silently
  forces `show_label=False` and renders the short label as `sr-only hide`, so the
  knobs show up as unlabeled boxes. The dashboard CSS hides only the long info
  paragraph (`.md.prose` and its wrapper), never `[data-testid="block-info"]`,
  which is the short label (not the info text).
- `ui.py` contains the visual language shared by all Gradio pages.

All run data belongs below `ASSETS/`. This directory must remain light: it is the
stable seam between UI controllers and feature adapters, not a second pipeline.
