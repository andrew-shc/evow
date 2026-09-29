"""Future View: explicit, base implicit, and source-adapted implicit forecasts."""
from dataclasses import dataclass
from typing import Iterator

from GREENFIELD.app_core.contracts import FeatureResult, RunArtifacts, StageEvent, StageSpec
from GREENFIELD.app_core.flow import CURRENT_FLOW_VERSION, LEGACY_FLOW_VERSION, FeatureFlow, FlowStage
from GREENFIELD.app_core.media import write_video
from GREENFIELD.app_core.model_catalog import GSPLAT, SVD, VIDEO_DEPTH_ANYTHING
from GREENFIELD.app_core.super_stages import SuperStage
from .future_explicit import run_explicit
from .future_settings import load_future_settings
from .future_training import artifact_file, train, training_dir
from .model_adapters import generate_continuation_steps, uses_cpu_offload


# The stage ids that belong to exactly one methodology family. Applicability
# lives on the flow (``FlowStage.modes``) so a row a selected method never emits
# still renders "Skipped" instead of waiting forever; the renderer trusts this
# instead of every producer hand-emitting a skip.
_SELF_TRAINED_MODES = ("self_trained",)
_IMPLICIT_MODES = ("implicit", "self_trained")
_EXPLICIT_MODES = ("explicit",)


STAGES = (
    StageSpec(
        "Sample input tail",
        "Sample the final window from the supplied stationary-camera video.",
        source_file="GREENFIELD/app_core/media.py",
        source_location="read_recent_window",
        function_chain="read_recent_window → observed frames",
        inputs="source video",
        outputs="observed RGB frames",
        controls="History (s) · History FPS",
        stage_id="source_tail",
    ),
    StageSpec(
        "Check training cache",
        "For self-trained implicit forecasting, check for a complete source-bound LoRA artifact before sampling any data.",
        source_file="GREENFIELD/features/future_training.py",
        source_location="training_manifest",
        function_chain="training_key → training_manifest",
        inputs="source video · training settings",
        outputs="cache hit or miss",
        controls="Self-trained",
        model_refs=(SVD,),
        stage_id="training_cache",
    ),
    StageSpec(
        "Prepare training data",
        "For self-trained implicit forecasting, sample only pre-holdout source windows.",
        source_file="GREENFIELD/features/future_training.py",
        source_location="prepare_implicit_pairs",
        function_chain="prepare_implicit_pairs",
        inputs="source video",
        outputs="bounded temporal pairs",
        controls="Self-trained",
        model_refs=(SVD,),
        stage_id="training_pairs",
    ),
    StageSpec(
        "Train selected method",
        "Fit the source-specific implicit LoRA before forecasting the held-out tail.",
        source_file="GREENFIELD/features/future_training.py",
        source_location="def train",
        function_chain="future_implicit_train (isolated worker)",
        inputs="temporal pairs",
        outputs="completed LoRA adapter",
        controls="Self-trained",
        model_refs=(SVD,),
        stage_id="train_lora",
    ),
    StageSpec(
        "Save training artifact",
        "Publish the complete LoRA adapter manifest atomically so a later run can reuse it.",
        source_file="GREENFIELD/features/future_training.py",
        source_location="manifest.json",
        function_chain="atomic manifest.json replace",
        inputs="fitted LoRA",
        outputs="reusable training artifact manifest",
        controls="Self-trained",
        model_refs=(SVD,),
        stage_id="train_artifact",
    ),
    StageSpec(
        "Prepare selected method",
        "Load Stable Video Diffusion, optionally with the source-specific LoRA, before sequential rollout.",
        source_file="GREENFIELD/features/future.py",
        source_location="generate_continuation_steps",
        function_chain="stable_video_pipeline → PeftModel.from_pretrained",
        inputs="history · methodology · seed",
        outputs="ready local SVD pipeline",
        controls="Implicit rollout",
        model_refs=(SVD,),
        stage_id="svd_load",
    ),
    StageSpec(
        "Infer temporal depth",
        "Estimate a temporally aligned depth prior and optical-flow motion from the observed history.",
        source_file="GREENFIELD/replay/depth_worker.py",
        source_location="def main",
        function_chain="VideoDepthAnything.infer_video_depth",
        inputs="observed frames",
        outputs="relative depth prior",
        controls="Explicit view",
        model_refs=(VIDEO_DEPTH_ANYTHING,),
        stage_id="depth_infer",
    ),
    StageSpec(
        "Fit dynamic Gaussians",
        "Fit the persistent, time-varying 3D Gaussian scene to the observed history.",
        source_file="GREENFIELD/replay/gaussian_worker.py",
        source_location="def main",
        function_chain="gsplat.rasterization → Adam",
        inputs="frames · depth prior · render pose",
        outputs="fitted dynamic Gaussian scene",
        controls="Explicit view",
        model_refs=(GSPLAT,),
        stage_id="gaussian_fit",
    ),
    StageSpec(
        "Extrapolate scene motion",
        "Extrapolate the learned deformation with a constant-velocity motion prior into the future window.",
        source_file="GREENFIELD/replay/gaussian_worker.py",
        source_location="def main",
        function_chain="constant-velocity motion prior",
        inputs="fitted scene motion",
        outputs="extrapolated future motion",
        controls="Explicit view",
        model_refs=(GSPLAT,),
        stage_id="motion_extrapolate",
    ),
    StageSpec(
        "Render and export splats",
        "Render the requested virtual view and export every observed-to-future splat keyframe.",
        source_file="GREENFIELD/replay/gaussian_worker.py",
        source_location="def main",
        function_chain="rasterization → save_splat",
        inputs="fitted scene · render pose",
        outputs="splat timeline",
        controls="Observed FPS · Forecast FPS",
        model_refs=(GSPLAT,),
        stage_id="splat_render_export",
    ),
    StageSpec(
        "Generate future scenario",
        "Roll out one possible future; it is never observed footage.",
        source_file="GREENFIELD/features/future.py",
        source_location="generate_continuation_steps",
        function_chain="StableVideoDiffusionPipeline (chunked)",
        inputs="prepared SVD state",
        outputs="generated RGB frames",
        controls="Chunk frames · SVD chunk · Seed",
        model_refs=(SVD,),
        stage_id="svd_generate",
    ),
    StageSpec(
        "Encode forecast video",
        "Encode the generated frames into the forecast video.",
        source_file="GREENFIELD/app_core/media.py",
        source_location="def write_video",
        function_chain="write_video",
        inputs="generated RGB frames",
        outputs="forecast.mp4",
        controls="Output (s) · Output FPS",
        stage_id="encode_video",
    ),
    StageSpec(
        "Write run trace",
        "Write the replayable run trace once the forecast is complete.",
        source_file="GREENFIELD/app_core/trace.py",
        source_location="def finish",
        function_chain="FeatureTrace.finish",
        inputs="generated scenario · events",
        outputs="trace.json",
        controls="None",
        stage_id="write_trace",
    ),
    StageSpec(
        "Validate forecast artifacts",
        "Check that the explicit forecast video and every timeline splat frame exist.",
        source_file="GREENFIELD/features/future_explicit.py",
        source_location="ordered_future_splats",
        function_chain="ordered_future_splats",
        inputs="exported scene frames · forecast video",
        outputs="verified artifact set",
        controls="Explicit view",
        model_refs=(GSPLAT,),
        stage_id="validate_artifacts",
    ),
    StageSpec(
        "Save forecast",
        "Write the final forecast video and replayable trace.",
        source_file="GREENFIELD/app_core/media.py",
        source_location="def write_video",
        function_chain="write_video",
        inputs="generated scenario",
        outputs="forecast.mp4 · trace.json",
        controls="Forecast",
        stage_id="save_forecast",
    ),
)

# The named parents own explicit stage IDs, so reordering ``STAGES`` can never
# silently regroup a saved run.
FLOW_GROUPS = (
    SuperStage("Input", ("source_tail",)),
    SuperStage("Prepare method", ("training_cache", "training_pairs", "train_lora", "train_artifact", "svd_load", "depth_infer")),
    SuperStage("Generate", ("gaussian_fit", "motion_extrapolate", "splat_render_export", "svd_generate")),
    SuperStage("Save", ("encode_video", "write_trace", "validate_artifacts", "save_forecast")),
)

# Frozen v1 identities (v1 had no ids, so these are the original display names).
# Kept literal so a later (currently v3) stage is never mistaken for one a v1 run
# could have recorded. Only the union rows that already existed in v1 keep a
# matching display name; every newer row is correctly marked "not recorded" for
# a pre-current trace.
LEGACY_STAGE_IDS = (
    "Sample input tail",
    "Prepare training data",
    "Train selected method",
    "Prepare selected method",
    "Generate future scenario",
    "Save forecast",
)

# The exact current-id sets each pre-current Future View profile could record.
# v1 persisted display names only; those six labels map onto the six current ids
# that share the name. v2 already persisted ids for the full 15-row union, so
# every current id is recordable — a v2 trace must never mark a union row (e.g.
# ``training_cache`` or ``write_trace``) as "not recorded".
LEGACY_PROFILES = {
    LEGACY_FLOW_VERSION: (
        "source_tail", "training_pairs", "train_lora", "svd_load", "svd_generate", "save_forecast",
    ),
    2: (
        "source_tail", "training_cache", "training_pairs", "train_lora", "train_artifact",
        "svd_load", "depth_infer", "gaussian_fit", "motion_extrapolate", "splat_render_export",
        "svd_generate", "encode_video", "write_trace", "validate_artifacts", "save_forecast",
    ),
    # v3 was the full 15-row union (unchanged by the v4 global bump), so a v3
    # trace recorded every current id and must never mark a union row "not
    # recorded". Kept in sync with ``STAGES`` so a future rename cannot drift.
    3: tuple(spec.stage_id for spec in STAGES),
    # v4 was the same full union; the v5 bump (Text Manipulation's atomization)
    # left Future View untouched, so its saved traces still record every row.
    4: tuple(spec.stage_id for spec in STAGES),
}

# Per-row methodology applicability. Empty means every mode; the first entry in
# the list is the canonical branch used by a mode that never emits the stage.
_STAGE_MODES = {
    "source_tail": (),
    "training_cache": _SELF_TRAINED_MODES,
    "training_pairs": _SELF_TRAINED_MODES,
    "train_lora": _SELF_TRAINED_MODES,
    "train_artifact": _SELF_TRAINED_MODES,
    "svd_load": _IMPLICIT_MODES,
    "depth_infer": _EXPLICIT_MODES,
    "gaussian_fit": _EXPLICIT_MODES,
    "motion_extrapolate": _EXPLICIT_MODES,
    "splat_render_export": _EXPLICIT_MODES,
    "svd_generate": _IMPLICIT_MODES,
    "encode_video": (),
    "write_trace": (),
    "validate_artifacts": _EXPLICIT_MODES,
    "save_forecast": (),
}


def flow() -> FeatureFlow:
    """Return Future View's current ordered stage profile."""
    stages = tuple(FlowStage(spec.stage_id, spec, modes=_STAGE_MODES[spec.stage_id]) for spec in STAGES)
    return FeatureFlow(
        "future", CURRENT_FLOW_VERSION, stages, FLOW_GROUPS,
        legacy_stage_ids=LEGACY_STAGE_IDS, legacy_profiles=LEGACY_PROFILES,
    )


@dataclass(frozen=True)
class FutureRequest:
    mode: str
    seed: int

    def validate(self):
        if self.mode not in {"explicit", "implicit", "self_trained"}:
            raise ValueError("Choose Explicit 3D, Implicit 3D [prior], or Implicit 3D [self-trained].")


def _svd_load_metrics(adapter, settings) -> dict:
    """Report only the SVD-load facts that are actually true for this run.

    ``offload`` mirrors the cached pipeline's placement: model CPU offload is
    enabled exactly when CUDA is available. ``checkpoint`` is included only when
    the settings expose one, so a stripped-down test settings object never claims
    a checkpoint it does not have.
    """
    metrics = {}
    checkpoint = getattr(settings, "checkpoint", None)
    if checkpoint:
        metrics["checkpoint"] = str(checkpoint)
    metrics["adapter"] = bool(adapter)
    metrics["offload"] = uses_cpu_offload()
    return metrics


def run(frames, fps: float, request: FutureRequest, artifacts: RunArtifacts, settings=None, source: str | None = None) -> Iterator[StageEvent | FeatureResult]:
    """Generate one forecast, adapting only the self-trained implicit method first."""
    request.validate()
    settings = settings or load_future_settings()
    adapter = None
    # The four training rows belong only to Implicit 3D [self-trained]. The flow's
    # modes would render them skipped on their own; emitting the data/train skips
    # keeps a v1-style serial reading and preserves the existing "terminal before
    # the first running stage" contract.
    if request.mode != "self_trained":
        yield StageEvent("Prepare training data", "skipped", "Not used by this method; only Implicit 3D [self-trained] samples pre-holdout source windows.", stage_id="training_pairs")
        yield StageEvent("Train selected method", "skipped", "Not used by this method; only Implicit 3D [self-trained] fits a source-specific LoRA adapter.", stage_id="train_lora")
    if request.mode == "self_trained":
        if not source:
            raise ValueError("Implicit 3D [self-trained] needs the selected source video.")
        # This is deliberately part of the method run, not a separate dashboard action.
        yield from train(source, "implicit", settings)
        adapter = artifact_file(source, "implicit", settings)
        if not adapter:
            raise RuntimeError("Self-trained implicit forecasting finished without a reusable LoRA adapter.")

    if request.mode in {"implicit", "self_trained"}:
        steps = generate_continuation_steps(frames[-1], settings.output_frames, request.seed, str(adapter) if adapter else None, settings)
        # ``svd_load`` wraps the first ``next()``; the generator's first yield is
        # exactly the model-load/offload boundary.
        yield StageEvent("Prepare selected method", "running", "Loading Stable Video Diffusion" + (" with the source-specific LoRA." if adapter else " prior."), stage_id="svd_load")
        next(steps)
        yield StageEvent("Prepare selected method", "complete", "Stable Video Diffusion is ready for sequential local rollout.", metrics=_svd_load_metrics(adapter, settings), stage_id="svd_load")
        chunk_frames = max(1, int(getattr(settings, "chunk_frames", settings.output_frames) or 1))
        total_chunks = max(1, (settings.output_frames + chunk_frames - 1) // chunk_frames)
        yield StageEvent("Generate future scenario", "running", f"Generating a {settings.output_seconds}-second future at {settings.output_fps:g} fps · 0/{settings.output_frames} frames.", stage_id="svd_generate")
        generated = []
        chunk = 0
        ticked_chunks: set[int] = set()
        for item in steps:
            # The adapter interleaves real per-denoise-step ticks with the completed
            # chunk tuple. Forward each tick unchanged; a chunk that already streamed
            # tick(s) must not also emit the coarse fallback, so the row never shows
            # duplicate progress for one chunk.
            if isinstance(item, StageEvent):
                reported_chunk = item.metrics.get("chunk")
                if isinstance(reported_chunk, int):
                    ticked_chunks.add(reported_chunk)
                yield item
                continue
            completed, generated = item
            chunk += 1
            if chunk not in ticked_chunks:
                yield StageEvent("Generate future scenario", "running", f"Generated local video continuation · {completed}/{settings.output_frames} frames ({completed / settings.output_fps:.1f}/{settings.output_seconds}s).", metrics={"chunk": chunk, "total_chunks": total_chunks, "generated_frames": completed, "total_frames": settings.output_frames}, stage_id="svd_generate")
        yield StageEvent("Generate future scenario", "complete", "Stable Video Diffusion completed the plausible future scenario.", metrics={"output_frames": settings.output_frames, "output_seconds": settings.output_seconds, "output_fps": settings.output_fps}, stage_id="svd_generate")
        yield StageEvent("Encode forecast video", "running", "Encoding the generated future into the forecast video.", stage_id="encode_video")
        output = write_video(generated, artifacts.file("forecast.mp4"), settings.output_fps)
        yield StageEvent("Encode forecast video", "complete", f"Encoded {settings.output_frames} frames at {settings.output_fps:g} fps.", metrics={"frames": settings.output_frames, "fps": settings.output_fps, "path": output.name}, stage_id="encode_video")
        explicit = None
    else:
        # The explicit worker owns fitting, motion extrapolation, splat export,
        # encoding, and artifact validation; its wrapper emits those stage ids.
        explicit_stream = run_explicit(frames, artifacts.run_dir, settings)
        while True:
            try:
                yield next(explicit_stream)
            except StopIteration as finished:
                explicit = finished.value
                break
        output = explicit.forecast

    yield StageEvent("Save forecast", "running", "Writing the forecast video and replayable trace.", stage_id="save_forecast")
    yield StageEvent("Save forecast", "complete", f"Saved a {settings.output_seconds}-second generated future at {settings.output_fps:g} fps; it is not camera footage.", stage_id="save_forecast")
    metadata = {"generated": True, "mode": request.mode, "seed": request.seed, "training_artifact": str(training_dir(source, "implicit", settings)) if adapter else None, "seconds": settings.output_seconds, "fps": settings.output_fps, "input_seconds": round(len(frames) / fps, 2), "input_fps": fps, "input_segment": "tail end of source video", "output_seconds": settings.output_seconds, "output_fps": settings.output_fps, "output_frames": settings.output_frames}
    if request.mode == "explicit":
        metadata["viewer"] = {"splat_paths": [str(path.relative_to(artifacts.run_dir)) for path in explicit.splats], "observed_frames": len(explicit.observed_indices), "observed_fps": settings.observed_splat_timeline_fps, "forecast_frames": len(explicit.forecast_indices), "forecast_fps": settings.forecast_splat_timeline_fps, "splat_durations": list(explicit.observed_durations + explicit.forecast_durations), "keyframe_fps": settings.forecast_splat_timeline_fps}
    yield FeatureResult(primary=output, metadata=metadata)
