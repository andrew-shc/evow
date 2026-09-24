"""Local Wan VACE source-conditioned video editing adapter."""

from functools import lru_cache

import cv2
import numpy as np

from GREENFIELD.app_core.models import require_package
from .text_models import device, require_text_checkpoint
from .text_settings import load_text_settings


@lru_cache(maxsize=1)
def vace_pipeline():
    """Load the owner-installed VACE pipeline without a dashboard download."""
    require_package("diffusers", "models")
    require_package("torch", "models")
    import torch
    from diffusers import WanVACEPipeline

    settings = load_text_settings()
    require_text_checkpoint(settings.edit_checkpoint, "Wan VACE")
    pipe = WanVACEPipeline.from_pretrained(
        settings.edit_checkpoint,
        local_files_only=True,
        torch_dtype=torch.float16 if device() == "cuda" else torch.float32,
    )
    if device() == "cuda":
        # Keep the 1.3B model practical on the shared 24 GB workstation.
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cpu")
    return pipe


def vace_frame_count(frame_count: int) -> int:
    """Wan temporal VAE accepts 1 + 4k frames; trim only an incomplete tail."""
    valid = 1 + 4 * max(0, (frame_count - 1) // 4)
    if valid < 5:
        raise ValueError("Text Manipulation needs at least five source frames.")
    return valid


def _dimensions(width: int, height: int) -> tuple[int, int]:
    """Fit an episode into VACE's 480p envelope without changing its aspect ratio."""
    settings = load_text_settings()
    scale = min(1.0, settings.edit_max_side / max(width, height), settings.edit_max_height / height)
    resized_width = max(16, int(round(width * scale / 16)) * 16)
    resized_height = max(16, int(round(height * scale / 16)) * 16)
    return resized_width, resized_height


def _uint8_frame(value) -> np.ndarray:
    """Normalize Diffusers/PIL output to browser-video RGB pixels."""
    array = np.asarray(value.convert("RGB") if hasattr(value, "convert") else value)
    if array.dtype != np.uint8:
        # Some pipeline adapters return floats in [0, 1], while others use
        # 0--255. Avoid ndarray ``initial`` compatibility differences here.
        maximum = float(array.max()) if array.size else 0.0
        array = (array * 255 if maximum <= 1.01 else array).clip(0, 255).astype(np.uint8)
    return array


def edit_video(frames: list[np.ndarray], masks: np.ndarray, prompt: str, seed: int) -> list[np.ndarray]:
    """Apply one text instruction while preserving masked-off source context."""
    if not frames or masks.shape[0] != len(frames):
        raise ValueError("VACE needs aligned source frames and binary masks.")
    count = vace_frame_count(len(frames))
    frames = frames[:count]
    masks = masks[:count]
    if any(frame.shape[:2] != mask.shape for frame, mask in zip(frames, masks)):
        raise ValueError("VACE masks must match their source frame resolutions.")
    require_package("PIL", "models")
    import torch
    from PIL import Image

    height, width = frames[0].shape[:2]
    target_width, target_height = _dimensions(width, height)
    source = [Image.fromarray(cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)) for frame in frames]
    # Wan VACE treats white as generated content and black as preserved context.
    generation_masks = [Image.fromarray(cv2.resize(mask.astype(np.uint8) * 255, (target_width, target_height), interpolation=cv2.INTER_NEAREST)) for mask in masks]
    generator = torch.Generator(device=device()).manual_seed(int(seed))
    settings = load_text_settings()
    result = vace_pipeline()(
        prompt=f"Preserve the fixed camera viewpoint and scene layout. {prompt.strip()}",
        negative_prompt="camera movement, text, watermark, blurry, distorted geometry",
        video=source,
        mask=generation_masks,
        height=target_height,
        width=target_width,
        num_frames=count,
        num_inference_steps=settings.vace_steps,
        guidance_scale=5.0,
        generator=generator,
        output_type="np",
    ).frames[0]
    return [cv2.resize(_uint8_frame(frame), (width, height), interpolation=cv2.INTER_LANCZOS4) for frame in result]


def release_video_edit_model() -> list[str]:
    """Release VACE's local cached pipeline after a request or owner recovery."""
    if not vace_pipeline.cache_info().currsize:
        return []
    pipe = vace_pipeline()
    if hasattr(pipe, "remove_all_hooks"):
        pipe.remove_all_hooks()
    if hasattr(pipe, "to"):
        # CPU offload leaves the final component on CUDA; move it back before
        # dropping the cache. VACE intentionally retains fp16 weights, so this
        # explicit cleanup should not warn about CPU fp16 inference.
        pipe.to("cpu", silence_dtype_warnings=True)
    vace_pipeline.cache_clear()
    return ["Wan VACE"]
