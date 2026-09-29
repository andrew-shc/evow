"""Declarative limits and local checkpoint locations for both text features."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TextSettings:
    """Values shared by text retrieval, segmentation, scene, and edit adapters."""

    root: Path
    semantic_model_id: str
    grounding_model_id: str
    segmentation_model_id: str
    edit_model_id: str
    semantic_checkpoint: Path
    grounding_checkpoint: Path
    segmentation_checkpoint: Path
    edit_checkpoint: Path
    query_max_source_seconds: float
    query_sample_fps: float
    query_clip_seconds: float
    query_result_fps: float
    query_max_results: int
    query_explicit_candidate_pool: int
    # Grounding DINO's confidence threshold and the projected-mask coverage that
    # marks a grounded region as low-spatial-specificity. Both were previously
    # module constants; moving them here lets Text Query override them per run
    # while these defaults reproduce the old hardcoded values exactly.
    grounding_threshold: float
    low_specificity_coverage: float
    episode_fps: int
    episode_max_frames: int
    source_fov_degrees: float
    scene_cache_version: str
    edit_max_side: int
    edit_max_height: int
    vace_steps: int
    # Wan VACE's classifier-free guidance and the dilation applied to the
    # resolved edit mask before conditioning. Exposing both lets an owner trade
    # prompt adherence and boundary blending while the defaults (5.0 and 5)
    # reproduce the previously hardcoded edit path exactly.
    vace_guidance_scale: float
    edit_mask_expand_px: int

    @property
    def episode_seconds(self) -> float:
        """Return the maximum source duration accepted by Text Manipulation."""
        return self.episode_max_frames / self.episode_fps

    @property
    def scene_config_token(self) -> str:
        """Version the values that make a cached Gaussian scene incompatible."""
        return f"version={self.scene_cache_version};fov={self.source_fov_degrees:g};fps={self.episode_fps};frames={self.episode_max_frames}"


@lru_cache(maxsize=1)
def load_text_settings() -> TextSettings:
    """Load tracked text-feature settings without reading local secrets."""
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "CONFIGS" / "features.yaml").read_text())
    models = config["models"]
    runtime = config["runtime"]
    checkpoint_root = root / "ASSETS" / "checkpoints"
    return TextSettings(
        root=root,
        semantic_model_id=str(models["semantic_retrieval"]),
        grounding_model_id=str(models["text_grounding"]),
        segmentation_model_id=str(models["video_segmentation"]),
        edit_model_id=str(models["video_edit"]),
        semantic_checkpoint=checkpoint_root / "siglip",
        grounding_checkpoint=checkpoint_root / "grounding-dino",
        segmentation_checkpoint=checkpoint_root / "sam2",
        edit_checkpoint=checkpoint_root / "wan-vace",
        query_max_source_seconds=float(runtime["query_max_source_seconds"]),
        query_sample_fps=float(runtime["query_sample_fps"]),
        query_clip_seconds=float(runtime["query_clip_seconds"]),
        query_result_fps=float(runtime["query_result_fps"]),
        query_max_results=int(runtime["query_max_results"]),
        query_explicit_candidate_pool=int(runtime["query_explicit_candidate_pool"]),
        grounding_threshold=float(runtime["text_grounding_threshold"]),
        low_specificity_coverage=float(runtime["text_low_specificity_coverage"]),
        episode_fps=int(runtime["text_episode_fps"]),
        episode_max_frames=int(runtime["text_episode_max_frames"]),
        source_fov_degrees=float(runtime["text_source_fov_degrees"]),
        scene_cache_version=str(runtime["text_scene_cache_version"]),
        edit_max_side=int(runtime["text_edit_max_side"]),
        edit_max_height=int(runtime["text_edit_max_height"]),
        vace_steps=int(runtime["text_vace_steps"]),
        vace_guidance_scale=float(runtime["text_vace_guidance_scale"]),
        edit_mask_expand_px=int(runtime["text_edit_mask_expand_px"]),
    )
