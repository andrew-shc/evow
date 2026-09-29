"""Lazy local-only inference adapters; browser requests never download weights."""
from functools import lru_cache
import inspect
import logging
import queue
import sys
import threading

import numpy as np

from GREENFIELD.app_core.contracts import StageEvent
from GREENFIELD.app_core.models import require_checkpoint, require_package
from .future_settings import load_future_settings

logger = logging.getLogger(__name__)

# Per-denoise-step progress. The SVD pipeline reports each diffusion step through
# ``callback_on_step_end``; we forward a bounded, honest sample as ``svd_generate``
# ticks. ``_MAX_STEP_TICKS_PER_CHUNK`` caps how many steps one chunk can report so
# a slow consumer cannot accumulate an unbounded tick backlog: the first step,
# the last step, and every ``ceil(total / cap)``-th intermediate step are kept.
# ``step`` and ``total`` in each tick always stay the real indices, never the
# sampled ones, so the browser shows truthful progress even when ticks are thinned.
_MAX_STEP_TICKS_PER_CHUNK = 8
# How long the streaming generator waits on the worker's tick queue before
# checking whether the pipeline thread has finished. Short enough to feel live,
# long enough not to spin a core while a denoise step is simply slow.
_CHUNK_QUEUE_POLL_SECONDS = 0.05

# The dashboard-wide "Clear GPU memory" button is wired with ``queue=False``, so it
# bypasses the ``concurrency_limit=1`` generation queue and runs ``clear_gpu_memory()``
# on a worker thread even while a generation is mid-denoise or rebuilding PEFT/CPU-
# offload hooks. This ownership-free semaphore serialises model lifecycle work: the
# Clear button simply waits for an in-flight generation instead of tearing the cached
# pipeline down underneath it.
#
# It must NOT be a thread-owned ``RLock``: Gradio 5.50.0 advances sync generators via
# ``anyio.to_thread.run_sync`` (each ``next()`` can run on a different threadpool
# thread) and closes them from the event-loop thread. A reentrant lock acquired on one
# thread and released on another raises ``RuntimeError: cannot release un-acquired
# lock`` in the generator's outer ``finally``, which leaves the lock permanently held so
# every later generation and Clear-GPU call blocks forever. A semaphore's release is
# explicitly ownership-free. Because it is not reentrant, generator teardown must call
# ``_clear_gpu_memory_unlocked()`` directly rather than re-acquiring.
_MODEL_LIFECYCLE_LOCK = threading.Semaphore(1)


def _device():
    require_package("torch", "models")
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def uses_cpu_offload() -> bool:
    """Whether the cached SVD pipeline runs with model CPU offload.

    ``stable_video_pipeline`` calls ``enable_model_cpu_offload`` exactly when a
    CUDA device is available and otherwise keeps the pipeline on CPU, so this is
    a truthful, side-effect-free report of the loaded pipeline's placement. It
    never imports torch unless the package is installed, so a model-free test
    run simply reports no offload instead of failing on a missing dependency.
    """
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())

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


def _load_lora_with_offload_hooks(pipe, adapter: str):
    """Wrap SVD's UNet only after its Accelerate hooks have been detached.

    A PEFT wrapper proxies the base UNet's ``_hf_hook`` attribute, but Accelerate
    must delete that attribute from the module it originally attached to. Rebuilding
    offload hooks around the wrapper keeps both PEFT and CPU offloading valid.
    """
    from peft import PeftModel

    base_unet = pipe.unet
    offload_enabled = bool(getattr(pipe, "_all_hooks", ()))
    if offload_enabled:
        pipe.remove_all_hooks()
    try:
        pipe.unet = PeftModel.from_pretrained(base_unet, adapter)
        if offload_enabled:
            pipe.enable_model_cpu_offload()
    except Exception:
        # Keep the cached base pipeline usable if an adapter cannot be loaded.
        pipe.unet = base_unet
        if offload_enabled:
            pipe.enable_model_cpu_offload()
        raise
    return base_unet, offload_enabled


def _restore_unet_with_offload_hooks(pipe, base_unet, offload_enabled: bool) -> None:
    """Unload the wrapper's PEFT adapter, then rebuild hooks on the returned base SVD.

    ``PeftModel.from_pretrained`` mutates ``base_unet`` in place, replacing its
    target ``nn.Linear`` submodules with LoRA layers. Pointing ``pipe.unet`` back
    at ``base_unet`` alone would therefore leave the cached pipeline adapter-
    contaminated. Asking the wrapper to unload reverts those submodules to their
    original base layers in place; only then does the cached pipeline genuinely
    return to base SVD.
    """
    wrapper = pipe.unet
    if offload_enabled:
        pipe.remove_all_hooks()
    unload_wrapper = getattr(wrapper, "unload", None)
    if callable(unload_wrapper):
        unload_wrapper()
    pipe.unet = base_unet
    if offload_enabled:
        pipe.enable_model_cpu_offload()


def _accepts_step_callback(pipe) -> bool:
    """Whether a pipeline's ``__call__`` can report per-denoise-step progress.

    Newer Diffusers SVD exposes ``callback_on_step_end`` explicitly; a test double
    that accepts ``**kwargs`` can absorb it too. An older pipeline offering neither
    falls back to one coarse per-chunk tick instead of raising on an unknown
    keyword. Signature inspection is preferred over a trial call so an unsupported
    pipeline is never run twice.
    """
    try:
        parameters = inspect.signature(pipe.__call__).parameters
    except (TypeError, ValueError):
        return False
    if "callback_on_step_end" in parameters:
        return True
    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())


def _total_denoise_steps(pipe) -> int | None:
    """Read the pipeline's real timestep count, or ``None`` when it does not expose one."""
    for attribute in ("num_timesteps", "_num_timesteps"):
        total = getattr(pipe, attribute, None)
        if total:
            return int(total)
    return None


def _step_is_reported(step: int, total: int) -> bool:
    """Keep the first, last, and evenly spaced intermediate steps within the tick cap."""
    if step == 0 or step == total - 1:
        return True
    stride = max(1, -(-total // _MAX_STEP_TICKS_PER_CHUNK))  # ceil(total / cap)
    return step % stride == 0


def _latent_shape(latents) -> list[int]:
    """Return a JSON-safe shape without copying the tensor."""
    shape = getattr(latents, "shape", None)
    return [int(dimension) for dimension in shape] if shape is not None else []


def _denoise_tick(record: dict, total_chunks: int) -> StageEvent:
    """Build one honest ``svd_generate`` tick from a real diffusers step."""
    detail = (f"Denoising chunk {record['chunk']}/{total_chunks} · "
              f"step {record['step'] + 1}/{record['total']} · t={record['timestep']:g}")
    return StageEvent("Generate future scenario", "running", detail, metrics=record, stage_id="svd_generate")


def _chunk_progress_event(chunk: int, total_chunks: int, generated_frames: int, total_frames: int) -> StageEvent:
    """The coarse fallback tick for a pipeline that cannot report denoise steps."""
    detail = f"Generated local video continuation · {generated_frames}/{total_frames} frames."
    metrics = {"chunk": chunk, "total_chunks": total_chunks, "generated_frames": generated_frames, "total_frames": total_frames}
    return StageEvent("Generate future scenario", "running", detail, metrics=metrics, stage_id="svd_generate")


def _stream_svd_chunk(pipe, initial_image, settings, generator, chunk_number: int, total_chunks: int, on_step):
    """Run one SVD chunk on a worker thread, yielding each real denoise tick live.

    The worker owns the ``pipe(...)`` call so the callback can push a tick the
    moment a denoise step finishes, which lets this generator surface progress
    while the GPU is still working instead of only after the chunk returns. The
    worker is always joined in ``finally`` so no denoise thread outlives the
    generator, and an inference failure is re-raised here after that join.

    Returns ``(images, streamed)`` where ``streamed`` is whether any per-step
    tick was emitted for this chunk. A pipeline that accepts the callback but
    reports no honest step total streams nothing, and the caller then emits its
    documented single coarse chunk tick.
    """
    tick_queue: queue.Queue = queue.Queue()

    def report_step(_pipe, step, timestep, callback_kwargs):
        total = _total_denoise_steps(_pipe)
        if total is None:
            # The pipeline invoked a step callback but exposes no real timestep
            # count. Fabricating ``step + 1`` would make every step look like the
            # last; instead report nothing and let the caller emit one honest
            # coarse chunk tick.
            return {}
        if not _step_is_reported(step, total):
            return {}
        record = {
            "step": int(step),
            "total": int(total),
            "timestep": float(timestep),
            "chunk": chunk_number,
            "latent_shape": _latent_shape(callback_kwargs.get("latents")),
        }
        tick_queue.put(("tick", _denoise_tick(record, total_chunks), record))
        return {}

    def denoise():
        try:
            images = pipe(
                initial_image, num_frames=settings.chunk_frames, generator=generator,
                decode_chunk_size=settings.svd_decode_chunk_size,
                callback_on_step_end=report_step,
                callback_on_step_end_tensor_inputs=["latents"],
            ).frames[0]
        except BaseException as error:  # Surface any worker failure to the caller.
            tick_queue.put(("error", error, None))
        else:
            tick_queue.put(("result", images, None))

    streamed = False
    worker = threading.Thread(target=denoise, name="svd-denoise", daemon=True)
    worker.start()
    try:
        while True:
            try:
                kind, payload, record = tick_queue.get(timeout=_CHUNK_QUEUE_POLL_SECONDS)
            except queue.Empty:
                # Only a dead worker with a drained queue means no result was ever produced;
                # a worker that finished between the timeout and this check will have its
                # result waiting, so keep draining instead of failing the run.
                if not worker.is_alive() and tick_queue.empty():
                    raise RuntimeError("SVD denoise worker exited without producing a result.")
                continue
            if kind == "tick":
                streamed = True
                if on_step is not None:
                    on_step(record)
                yield payload
            elif kind == "result":
                return payload, streamed
            else:
                raise payload
    finally:
        worker.join()


def generate_continuation_steps(frame: np.ndarray, count: int, seed: int, adapter: str | None = None, settings=None, on_step=None):
    """Yield completed local SVD chunks, releasing all CUDA state on completion.

    ``settings`` carries the per-run effective configuration so the editable VAE
    decode batch reaches the pipeline; omitting it reproduces tracked defaults.

    While a chunk denoises, this generator also yields ``svd_generate`` tick
    events carrying the real ``{step, total, timestep, chunk, latent_shape}``
    from Diffusers' step callback, so the dashboard advances during inference
    rather than only at chunk boundaries. ``on_step`` is an optional
    ``Callable[[dict], None]`` push observer that receives the same raw record;
    the tick is still yielded for generator consumers, so passing a callback is
    never required to stream progress.
    """
    require_package("PIL", "models")
    import torch
    from PIL import Image
    settings = settings or load_future_settings()
    result = []
    current = Image.fromarray(frame)
    # Hold the lifecycle lock for as long as this generator lives, releasing only in the
    # outer ``finally``. The Clear-GPU button above runs with ``queue=False``, so without
    # this it could call ``clear_gpu_memory()`` concurrently and tear down the exact
    # pipeline this generator is denoising with or rebuilding hooks on.
    _MODEL_LIFECYCLE_LOCK.acquire()
    try:
        pipe = stable_video_pipeline()
        base_unet = None
        offload_enabled = False
        adapter_loaded = False
        try:
            if adapter:
                # SVD lacks Diffusers' pipeline LoRA mixin; PEFT wraps only its temporal UNet.
                base_unet, offload_enabled = _load_lora_with_offload_hooks(pipe, adapter)
                adapter_loaded = True
            # The first yield separates model loading from the generated-video stage.
            yield 0, None
            total_chunks = (count + settings.chunk_frames - 1) // settings.chunk_frames
            step_callback_supported = _accepts_step_callback(pipe)
            for chunk_offset in range(total_chunks):
                chunk_number = chunk_offset + 1
                generator = torch.Generator(device=_device()).manual_seed(seed + chunk_offset)
                if step_callback_supported:
                    images, streamed = yield from _stream_svd_chunk(pipe, current, settings, generator, chunk_number, total_chunks, on_step)
                    if not streamed:
                        # The pipeline accepted the callback but exposed no honest
                        # step total, so fall back to the documented single coarse
                        # chunk tick instead of inventing per-step progress.
                        completed = min(len(result) + len(images), count)
                        yield _chunk_progress_event(chunk_number, total_chunks, completed, count)
                else:
                    images = pipe(current, num_frames=settings.chunk_frames, generator=generator,
                                  decode_chunk_size=settings.svd_decode_chunk_size).frames[0]
                    completed = min(len(result) + len(images), count)
                    yield _chunk_progress_event(chunk_number, total_chunks, completed, count)
                result.extend(np.asarray(image.convert("RGB")) for image in images)
                current = images[-1]
                yield min(len(result), count), result
        finally:
            # Capture whatever error is already propagating before teardown runs; a hook
            # or allocator failure here must never replace the real inference error, or
            # the owner sees a misleading cause instead of the actual failure.
            inference_error = sys.exc_info()[1]
            # A LAN request must never leave the workstation's VRAM or model cache dirty,
            # even if hook teardown fails, so disposal runs in its own finally block.
            try:
                if adapter_loaded:
                    _restore_unet_with_offload_hooks(pipe, base_unet, offload_enabled)
            except Exception:
                if inference_error is None:
                    raise
                logger.exception("Future View adapter teardown failed after an inference error.")
            finally:
                del pipe
                try:
                    # The generator already holds ``_MODEL_LIFECYCLE_LOCK`` for its whole
                    # lifetime, so it must call the unlocked helper; the public wrapper
                    # would try to re-acquire the non-reentrant semaphore and deadlock.
                    _clear_gpu_memory_unlocked()
                except Exception:
                    if inference_error is None:
                        raise
                    logger.exception("Future View GPU cleanup failed after an inference error.")
    finally:
        _MODEL_LIFECYCLE_LOCK.release()


def generate_continuation(frame: np.ndarray, count: int, seed: int, adapter: str | None = None, settings=None) -> list[np.ndarray]:
    """Consume the streamed SVD adapter for callers that only need its result."""
    steps = generate_continuation_steps(frame, count, seed, adapter, settings)
    _loaded, _ = next(steps)
    generated = []
    for item in steps:
        if isinstance(item, StageEvent):
            continue  # Progress ticks carry no frames; only chunk tuples do.
        _completed, generated = item
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


def _clear_gpu_memory_unlocked() -> dict[str, int | list[str]]:
    """Release cached feature models and return before/after CUDA allocator state.

    Callers must already hold ``_MODEL_LIFECYCLE_LOCK``; this is the work half of
    ``clear_gpu_memory`` split out so the generation generator's teardown can invoke it
    while it still owns the semaphore (a semaphore is not reentrant). It never touches
    another process's CUDA context; it only releases this app's models.
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


def clear_gpu_memory() -> dict[str, int | list[str]]:
    """Release cached feature models and return before/after CUDA allocator state.

    This is intentionally callable from the dashboard after an OOM. It never
    touches another process's CUDA context; it only releases this app's models.
    The lifecycle lock makes it wait for any in-flight generation before it
    mutates or clears the cached pipeline.
    """
    with _MODEL_LIFECYCLE_LOCK:
        return _clear_gpu_memory_unlocked()
