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
    )
