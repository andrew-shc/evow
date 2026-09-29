"""Resolve local paths and declarative replay settings."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    root: Path
    assets: Path
    run_root: Path
    temp_root: Path
    anyview_repo: Path
    depth_repo: Path
    anyview_checkpoint: Path
    anyview_tokenizer: Path
    anyview_text_embedding: Path
    depth_checkpoint: Path
    fps: int
    max_side: int
    default_frames: int
    anyview_steps: int
    depth_input_size: int
    gaussian_stride: int
    gaussian_steps: int
    # Gaussian training and rendering knobs. The defaults reproduce the values
    # that were previously hardcoded in gaussian_worker.py exactly, so an
    # untouched CONFIGS/replay.yaml keeps the fitted scene bit-for-bit the same.
    gaussian_lr_xyz: float
    gaussian_lr_rotation: float
    gaussian_lr_scale: float
    gaussian_lr_color: float
    gaussian_lr_opacity: float
    gaussian_lr_motion: float
    gaussian_loss_depth: float
    gaussian_loss_smoothness: float
    gaussian_init_opacity_logit: float
    gaussian_background_r: float
    gaussian_background_g: float
    gaussian_background_b: float
    anyview_guidance_scale: float


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    root = Path(__file__).resolve().parents[2]
    # Load secrets into the process environment without inspecting or relaying the file.
    load_dotenv(root / ".env", override=False)
    config = yaml.safe_load((root / "CONFIGS" / "replay.yaml").read_text())
    assets = root / "ASSETS"
    run_root = assets / "replay" / "runs"
    temp_root = assets / "replay" / "gradio_tmp"
    run_root.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    return Settings(
        root=root,
        assets=assets,
        run_root=run_root,
        temp_root=temp_root,
        anyview_repo=root / "AnyView-DVS",
        depth_repo=root / "Video-Depth-Anything",
        anyview_checkpoint=assets / "checkpoints" / "anyview" / "anyview_dvs_2b.pt",
        anyview_tokenizer=assets / "checkpoints" / "anyview" / "tokenizer.pth",
        anyview_text_embedding=assets / "checkpoints" / "anyview" / "default_text_emb.pt",
        depth_checkpoint=assets / "checkpoints" / "video_depth_anything" / "video_depth_anything_vits.pth",
        fps=int(config["capture"]["fps"]),
        max_side=int(config["capture"]["max_side"]),
        default_frames=int(config["capture"]["default_frames"]),
        anyview_steps=int(config["model"]["anyview_steps"]),
        depth_input_size=int(config["model"]["depth_input_size"]),
        gaussian_stride=int(config["model"]["gaussian_stride"]),
        gaussian_steps=int(config["model"]["gaussian_steps"]),
        gaussian_lr_xyz=float(config["model"]["gaussian_lr_xyz"]),
        gaussian_lr_rotation=float(config["model"]["gaussian_lr_rotation"]),
        gaussian_lr_scale=float(config["model"]["gaussian_lr_scale"]),
        gaussian_lr_color=float(config["model"]["gaussian_lr_color"]),
        gaussian_lr_opacity=float(config["model"]["gaussian_lr_opacity"]),
        gaussian_lr_motion=float(config["model"]["gaussian_lr_motion"]),
        gaussian_loss_depth=float(config["model"]["gaussian_loss_depth"]),
        gaussian_loss_smoothness=float(config["model"]["gaussian_loss_smoothness"]),
        gaussian_init_opacity_logit=float(config["model"]["gaussian_init_opacity_logit"]),
        gaussian_background_r=float(config["model"]["gaussian_background_r"]),
        gaussian_background_g=float(config["model"]["gaussian_background_g"]),
        gaussian_background_b=float(config["model"]["gaussian_background_b"]),
        anyview_guidance_scale=float(config["model"]["anyview_guidance_scale"]),
    )
