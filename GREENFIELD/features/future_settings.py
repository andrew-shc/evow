"""Declarative Future View settings and local checkpoint locations."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import yaml

@dataclass(frozen=True)
class FutureSettings:
    root: Path; model_id: str; checkpoint: Path; history_seconds: float; history_fps: int
    output_seconds: int; output_fps: int; observed_splat_timeline_fps: int; forecast_splat_timeline_fps: int
    chunk_frames: int; motion_fit_frames: int; max_side: int
    training_max_side: int; training_max_pairs: int; training_lora_steps: int; training_lora_rank: int
    # Explicit-3D render pose and the SVD VAE decode batch. Defaults are the
    # values the workers previously hardcoded, so an untouched run is unchanged.
    render_yaw: float; render_shift: float; render_fov: float; source_fov_degrees: float
    svd_decode_chunk_size: int
    @property
    def output_frames(self): return self.output_seconds * self.output_fps

@lru_cache(maxsize=1)
def load_future_settings() -> FutureSettings:
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "CONFIGS" / "features.yaml").read_text())
    runtime, models = config["runtime"], config["models"]
    return FutureSettings(root, models["future_video"], root / "ASSETS" / "checkpoints" / "stable-video-diffusion", float(runtime["future_history_seconds"]), int(runtime["future_history_fps"]), int(runtime["future_output_seconds"]), int(runtime["future_output_fps"]), int(runtime["future_observed_splat_timeline_fps"]), int(runtime["future_forecast_splat_timeline_fps"]), int(runtime["future_svd_chunk_frames"]), int(runtime["future_motion_fit_frames"]), int(runtime["future_max_side"]), int(runtime["future_training_max_side"]), int(runtime["future_training_max_pairs"]), int(runtime["future_training_lora_steps"]), int(runtime["future_training_lora_rank"]), float(runtime["future_render_yaw"]), float(runtime["future_render_shift"]), float(runtime["future_render_fov"]), float(runtime["future_source_fov_degrees"]), int(runtime["future_svd_decode_chunk_size"]))
