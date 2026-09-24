"""Lazy local-only inference adapters; browser requests never download weights."""
from functools import lru_cache
import numpy as np
from GREENFIELD.app_core.models import require_checkpoint, require_package
from .future_settings import load_future_settings

def _device():
    require_package("torch", "models")
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"

@lru_cache(maxsize=1)
def stable_video_pipeline():
    require_package("diffusers", "models")
    settings = load_future_settings(); require_checkpoint(settings.checkpoint, "Stable Video Diffusion")
    import torch
    from diffusers import StableVideoDiffusionPipeline
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        settings.checkpoint, local_files_only=True, variant="fp16",
        torch_dtype=torch.float16 if _device() == "cuda" else torch.float32,
    )
    if _device() == "cuda":
        # Keeping the whole SVD pipeline on a 24 GB card consumes roughly 15 GB
        # before denoising activations exist. Accelerate moves one component at a
        # time instead, leaving headroom for inference and other dashboard work.
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cpu")
    return pipe

def generate_continuation_steps(frame: np.ndarray, count: int, seed: int):
    """Yield completed local SVD chunks, releasing all CUDA state on completion."""
    require_package("PIL", "models")
    import torch
    from PIL import Image
    settings = load_future_settings()
    result = []
    current = Image.fromarray(frame)
    pipe = stable_video_pipeline()
    try:
        # The first yield separates model loading from the generated-video stage.
        yield 0, None
        for chunk in range((count + settings.chunk_frames - 1) // settings.chunk_frames):
            generator = torch.Generator(device=_device()).manual_seed(seed + chunk)
            images = pipe(current, num_frames=settings.chunk_frames, generator=generator,
                          decode_chunk_size=1).frames[0]
            result.extend(np.asarray(image.convert("RGB")) for image in images)
            current = images[-1]
            yield min(len(result), count), result
    finally:
        # A LAN request must not reserve the workstation GPU after its result is saved.
        del pipe
        clear_gpu_memory()


def generate_continuation(frame: np.ndarray, count: int, seed: int) -> list[np.ndarray]:
    """Consume the streamed SVD adapter for callers that only need its result."""
    steps = generate_continuation_steps(frame, count, seed)
    _loaded, _ = next(steps)
    generated = []
    for _completed, generated in steps:
        pass
    return generated[:count]


def _cached_model_memory_release(loader, label: str) -> str | None:
    """Move one lazily cached model off CUDA without creating a new one."""
    if not loader.cache_info().currsize:
        return None
    value = loader()
    model = value[-1] if isinstance(value, tuple) else value
    if hasattr(model, "to"):
        model.to("cpu")
    if hasattr(model, "remove_all_hooks"):
        model.remove_all_hooks()
    loader.cache_clear()
    return label


def clear_gpu_memory() -> dict[str, int | list[str]]:
    """Release cached feature models and return before/after CUDA allocator state.

    This is intentionally callable from the dashboard after an OOM. It never
    touches another process's CUDA context; it only releases this app's models.
    """
    import gc
    import torch
    before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    released = []
    for loader, label in ((stable_video_pipeline, "Stable Video Diffusion"),):
        name = _cached_model_memory_release(loader, label)
        if name:
            released.append(name)
    gc.collect()
    # Text Query and Wan VACE keep separate, local-only caches so their
    # optional dependencies do not affect Future View imports.
    from .text_models import release_text_models
    from .video_edit import release_video_edit_model
    released.extend(release_text_models())
    released.extend(release_video_edit_model())
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    return {"before_bytes": before, "after_bytes": after, "released": released}
