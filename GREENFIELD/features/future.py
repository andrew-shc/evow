"""Two local Future View methods with fixed, clearly labeled 30-second output."""
from dataclasses import dataclass
from typing import Iterator
from GREENFIELD.app_core.contracts import FeatureResult, RunArtifacts, StageEvent, StageSpec
from GREENFIELD.app_core.model_catalog import GSPLAT, SVD, VIDEO_DEPTH_ANYTHING
from GREENFIELD.app_core.media import write_video
from .future_settings import load_future_settings
from .model_adapters import generate_continuation_steps
from .future_explicit import run_explicit

STAGES = (
    StageSpec("Sample input tail", "Sample the final window from the supplied stationary-camera video.",
              "GREENFIELD/app_core/media.py", "read_recent_window", "source video", "observed RGB frames", "Source"),
    StageSpec("Prepare selected method", "Load Stable Video Diffusion for implicit rollout or Video Depth Anything reconstruction inputs for explicit 3D.",
              "GREENFIELD/features/future.py", "run", "history · methodology · seed", "ready local model or geometry prior", "Methodology", model_refs=(SVD, VIDEO_DEPTH_ANYTHING)),
    StageSpec("Generate future scenario", "Roll out one possible future; it is never observed footage.",
              "GREENFIELD/features/future.py", "run", "prepared method state", "generated RGB frames or future 3D splats", "Generate", model_refs=(SVD, GSPLAT)),
    StageSpec("Save forecast", "Write the video, trace, and explicit scene artifacts when available.",
              "GREENFIELD/app_core/media.py", "write_video", "generated scenario", "forecast.mp4 · trace.json", "Generate"),
)


@dataclass(frozen=True)
class FutureRequest:
    mode: str; seed: int
    def validate(self):
        if self.mode not in {"implicit", "explicit"}: raise ValueError("Choose Implicit video or Explicit 3D.")

def run(frames, fps: float, request: FutureRequest, artifacts: RunArtifacts, settings=None) -> Iterator[StageEvent | FeatureResult]:
    """Emit truthful stage updates for either local Future View method."""
    request.validate()
    settings = settings or load_future_settings()
    if request.mode == "implicit":
        steps = generate_continuation_steps(frames[-1], settings.output_frames, request.seed)
        yield StageEvent("Prepare selected method", "running",
                         "Loading the local Stable Video Diffusion continuation model.")
        next(steps)
        yield StageEvent("Prepare selected method", "complete",
                         "Stable Video Diffusion is ready for sequential local rollout.")
        yield StageEvent("Generate future scenario", "running",
                         f"Generating a {settings.output_seconds}-second future at {settings.output_fps:g} fps · 0/{settings.output_frames} frames.")
        generated = []
        for completed, generated in steps:
            yield StageEvent("Generate future scenario", "running",
                             f"Generated local video continuation · {completed}/{settings.output_frames} frames ({completed / settings.output_fps:.1f}/{settings.output_seconds}s).",
                             metrics={"generated_frames": completed, "total_frames": settings.output_frames})
        yield StageEvent("Generate future scenario", "complete",
                         "Stable Video Diffusion completed the 30-second plausible scenario.",
                         metrics={"output_frames": settings.output_frames, "output_seconds": settings.output_seconds, "output_fps": settings.output_fps})
        yield StageEvent("Save forecast", "running", "Encoding the generated video and replayable trace.")
        output = write_video(generated, artifacts.file("forecast.mp4"), settings.output_fps)
        explicit = None
    else:
        explicit_stream = run_explicit(frames, artifacts.run_dir, settings)
        while True:
            try:
                yield next(explicit_stream)
            except StopIteration as finished:
                explicit = finished.value
                break
        output = explicit.forecast
        yield StageEvent("Generate future scenario", "complete",
                         "Dynamic 3D Gaussians were fitted, extrapolated, and exported as a future scene timeline.",
                         metrics={"output_frames": settings.output_frames, "output_seconds": settings.output_seconds, "output_fps": settings.output_fps})
        yield StageEvent("Save forecast", "running", "Saving the forecast video and replayable 3D-scene trace.")
    yield StageEvent("Save forecast", "complete", f"Saved a {settings.output_seconds}-second generated future at {settings.output_fps:g} fps; it is not camera footage.")
    metadata = {"generated": True, "mode": request.mode, "seed": request.seed,
                "seconds": settings.output_seconds, "fps": settings.output_fps,
                "input_seconds": round(len(frames) / fps, 2), "input_fps": fps, "input_segment": "tail end of source video",
                "output_seconds": settings.output_seconds, "output_fps": settings.output_fps,
                "output_frames": settings.output_frames}
    if request.mode == "explicit":
        # Relative paths let the saved-run loader reconstruct the local viewer.
        metadata["viewer"] = {
            "splat_paths": [str(path.relative_to(artifacts.run_dir)) for path in explicit.splats],
            "observed_frames": len(explicit.observed_indices), "observed_fps": settings.observed_splat_timeline_fps,
            "forecast_frames": len(explicit.forecast_indices), "forecast_fps": settings.forecast_splat_timeline_fps,
            "splat_durations": list(explicit.observed_durations + explicit.forecast_durations),
            "keyframe_fps": settings.forecast_splat_timeline_fps,
        }
    yield FeatureResult(primary=output, metadata=metadata)
