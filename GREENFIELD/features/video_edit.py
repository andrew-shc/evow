"""Local Wan VACE source-conditioned video editing adapter."""

from functools import lru_cache
import inspect
import queue
import threading
from typing import Mapping

import cv2
import numpy as np

from GREENFIELD.app_core.contracts import StageEvent
from GREENFIELD.app_core.models import require_package
from .text_models import device, require_text_checkpoint
from .text_settings import load_text_settings


# Per-denoise-step progress. Wan VACE reports each diffusion step through
# ``callback_on_step_end``; we forward a bounded, honest sample as
# ``vace_denoise`` ticks. ``_MAX_DENOISE_TICKS`` caps how many steps one run can
# report: the first step, the last step, and every ``ceil(total / cap)``-th
# intermediate step are kept, while ``step``/``total`` always stay the real
# indices so truthful progress survives the thinning.
_MAX_DENOISE_TICKS = 8
# How long the streaming generator waits on the worker's tick queue before
# checking whether the pipeline thread has finished.
_QUEUE_POLL_SECONDS = 0.05

# Blank ids for callers (e.g. the plain ``edit_video`` wrapper) that only need
# frames and render no Wan VACE rows.
_NO_VACE_STAGE_IDS: Mapping[str, str] = {
    "vace_prepare": "", "vace_denoise": "", "vace_decode": "",
}


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


def _dimensions(width: int, height: int, settings) -> tuple[int, int]:
    """Fit an episode into VACE's configured envelope without changing its aspect ratio.

    ``settings`` is the effective per-run ``TextSettings`` from the page boundary,
    so an override to either resolution ceiling changes this run's output rather
    than the worker's tracked defaults.
    """
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


def _accepts_step_callback(pipe) -> bool:
    """Whether a pipeline's ``__call__`` can report per-denoise-step progress.

    Signature inspection is preferred over a trial call so an unsupported
    pipeline is never run twice. A test double that accepts ``**kwargs`` can
    absorb the callback too.
    """
    try:
        parameters = inspect.signature(pipe.__call__).parameters
    except (TypeError, ValueError):
        return False
    if "callback_on_step_end" in parameters:
        return True
    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())


def _total_scheduler_steps(pipe) -> int | None:
    """Read the pipeline's real timestep count, or ``None`` when it exposes none.

    The scheduler's ``timesteps`` length is the authoritative count of denoise
    steps actually run; falling back to a pipeline ``num_timesteps`` attribute
    keeps an alternate Diffusers layout honest rather than fabricating a total.
    """
    timesteps = getattr(getattr(pipe, "scheduler", None), "timesteps", None)
    if timesteps is not None:
        try:
            total = len(timesteps)
        except TypeError:
            total = 0
        if total:
            return int(total)
    for attribute in ("num_timesteps", "_num_timesteps"):
        total = getattr(pipe, attribute, None)
        if total:
            return int(total)
    return None


def _denoise_step_is_reported(step: int, total: int) -> bool:
    """Keep the first, last, and evenly spaced intermediate steps within the tick cap."""
    if step == 0 or step == total - 1:
        return True
    stride = max(1, -(-total // _MAX_DENOISE_TICKS))  # ceil(total / cap)
    return step % stride == 0


def _latent_shape(latents) -> list[int]:
    """Return a JSON-safe shape without copying the tensor."""
    shape = getattr(latents, "shape", None)
    return [int(dimension) for dimension in shape] if shape is not None else []


def edit_video_steps(frames: list[np.ndarray], masks: np.ndarray, prompt: str, seed: int, settings=None, *, stage_ids: Mapping[str, str] | None = None):
    """Yield Wan VACE prepare/denoise/decode ticks and return the generated frames.

    ``vace_prepare`` and ``vace_decode`` are simple rows and emit their own
    terminal here; ``vace_denoise`` is an aggregate whose running ticks stream
    live from a worker thread while its single terminal is left to the caller.
    When the pipeline exposes no real step total, exactly one coarse denoise tick
    is emitted rather than fabricating a ``step + 1`` index. ``stage_ids`` maps
    the row names onto the owning feature's stable ids; blank ids mean the caller
    renders no row.
    """
    stage_ids = stage_ids or _NO_VACE_STAGE_IDS
    settings = settings or load_text_settings()
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
    target_width, target_height = _dimensions(width, height, settings)
    source = [
        Image.fromarray(cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA))
        for frame in frames
    ]
    # Wan VACE treats white as generated content and black as preserved context.
    generation_masks = [
        Image.fromarray(
            cv2.resize(mask.astype(np.uint8) * 255, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        )
        for mask in masks
    ]
    generator = torch.Generator(device=device()).manual_seed(int(seed))
    pipe = vace_pipeline()
    prepare_metrics = {"frames": count, "height": target_height, "width": target_width}
    yield StageEvent(
        "Prepare VACE conditioning", "running",
        f"Resizing {count} frames and their masks into Wan VACE's envelope.",
        metrics=prepare_metrics, stage_id=stage_ids["vace_prepare"],
    )
    yield StageEvent(
        "Prepare VACE conditioning", "complete",
        f"Prepared {count} frames at {target_width}×{target_height} with 0/255 generation masks.",
        metrics=prepare_metrics, stage_id=stage_ids["vace_prepare"],
    )

    tick_queue: queue.Queue = queue.Queue()

    def report_step(_pipe, step, timestep, callback_kwargs):
        total = _total_scheduler_steps(_pipe)
        if total is None or not _denoise_step_is_reported(step, total):
            return {}
        tick_queue.put(("tick", {
            "step": int(step),
            "total": int(total),
            "timestep": float(timestep),
            "latent_shape": _latent_shape(callback_kwargs.get("latents")),
        }))
        return {}

    call_kwargs = dict(
        prompt=f"Preserve the fixed camera viewpoint and scene layout. {prompt.strip()}",
        negative_prompt="camera movement, text, watermark, blurry, distorted geometry",
        video=source,
        mask=generation_masks,
        height=target_height,
        width=target_width,
        num_frames=count,
        num_inference_steps=settings.vace_steps,
        guidance_scale=settings.vace_guidance_scale,
        generator=generator,
        output_type="np",
    )
    if _accepts_step_callback(pipe):
        call_kwargs["callback_on_step_end"] = report_step
        call_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

    def denoise():
        try:
            pipeline_result = pipe(**call_kwargs)
        except BaseException as error:  # Surface any worker failure to the caller.
            tick_queue.put(("error", error))
        else:
            tick_queue.put(("result", pipeline_result.frames[0]))

    streamed = False
    raw_frames = None
    worker = threading.Thread(target=denoise, name="vace-denoise", daemon=True)
    worker.start()
    try:
        while True:
            try:
                kind, payload = tick_queue.get(timeout=_QUEUE_POLL_SECONDS)
            except queue.Empty:
                # Only a dead worker with a drained queue means no result was ever
                # produced; a worker that finished in the gap still has its result
                # waiting, so keep draining rather than failing the run.
                if not worker.is_alive() and tick_queue.empty():
                    raise RuntimeError("VACE denoise worker exited without producing a result.")
                continue
            if kind == "tick":
                streamed = True
                yield StageEvent(
                    "Denoise edit proposal", "running",
                    f"Denoising step {payload['step'] + 1}/{payload['total']} · t={payload['timestep']:g}",
                    metrics=payload, stage_id=stage_ids["vace_denoise"],
                )
            elif kind == "result":
                raw_frames = payload
                break
            else:
                raise payload
    finally:
        worker.join()

    if not streamed:
        # The pipeline exposed no real step total, so report one honest coarse
        # tick with only a frame count instead of inventing ``step + 1``.
        yield StageEvent(
            "Denoise edit proposal", "running",
            f"Denoised {count} frames with Wan VACE.",
            metrics={"frames": count}, stage_id=stage_ids["vace_denoise"],
        )

    yield StageEvent(
        "Decode edit proposal", "running",
        "VAE-decoding and Lanczos-upscaling the proposal to source resolution.",
        metrics={"frames": count}, stage_id=stage_ids["vace_decode"],
    )
    result = [cv2.resize(_uint8_frame(frame), (width, height), interpolation=cv2.INTER_LANCZOS4) for frame in raw_frames]
    yield StageEvent(
        "Decode edit proposal", "complete",
        f"Decoded {len(result)} frames back to {width}×{height}.",
        metrics={"frames": len(result), "height": height, "width": width}, stage_id=stage_ids["vace_decode"],
    )
    return result


def edit_video(frames: list[np.ndarray], masks: np.ndarray, prompt: str, seed: int, settings=None) -> list[np.ndarray]:
    """Apply one text instruction while preserving masked-off source context.

    ``settings`` is the validated effective per-run configuration; when omitted
    the tracked defaults are loaded so existing callers keep their behavior. This
    positional-compatible wrapper consumes the streaming adapter and returns only
    its frames; callers that render progress use ``edit_video_steps`` directly.
    """
    steps = edit_video_steps(frames, masks, prompt, seed, settings)
    while True:
        try:
            next(steps)
        except StopIteration as finished:
            return finished.value


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
