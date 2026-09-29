"""Local-only Transformers adapters used by both text feature pipelines."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator, Mapping

import numpy as np

from GREENFIELD.app_core.contracts import StageEvent
from GREENFIELD.app_core.models import require_checkpoint, require_package
from .text_settings import load_text_settings


TEXT_SETUP_COMMAND = "conda run -n evow python -m GREENFIELD.features.text_setup"


@dataclass(frozen=True)
class GroundedBox:
    """A text-conditioned image region ready to seed SAM2 tracking."""

    xyxy: tuple[float, float, float, float]
    confidence: float


def require_text_checkpoint(path, label: str) -> None:
    """Reject missing or incomplete owner-installed local model snapshots."""
    require_checkpoint(path, label, TEXT_SETUP_COMMAND)
    if not (path / "model_manifest.json").is_file():
        raise RuntimeError(f"{label} weights at {path} are incomplete. Install them with: {TEXT_SETUP_COMMAND}")


def device() -> str:
    """Use CUDA when present while preserving diagnostic behavior on CPU hosts."""
    require_package("torch", "models")
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _torch_dtype():
    """Use half precision only where the local CUDA device can execute it."""
    import torch

    return torch.float16 if device() == "cuda" else torch.float32


def _inputs_for_model(inputs, model):
    """Move processor tensors to a model and match its floating-point dtype."""
    parameter = next(model.parameters())
    prepared = inputs.to(parameter.device)
    for name, value in prepared.items():
        if hasattr(value, "is_floating_point") and value.is_floating_point():
            prepared[name] = value.to(dtype=parameter.dtype)
    return prepared


@lru_cache(maxsize=1)
def siglip_model():
    """Load local SigLIP weights without allowing a dashboard download."""
    require_package("transformers", "models")
    from transformers import AutoModel, AutoProcessor

    settings = load_text_settings()
    require_text_checkpoint(settings.semantic_checkpoint, "SigLIP")
    processor = AutoProcessor.from_pretrained(settings.semantic_checkpoint, local_files_only=True)
    model = AutoModel.from_pretrained(
        settings.semantic_checkpoint, local_files_only=True, dtype=_torch_dtype()
    ).to(device()).eval()
    return processor, model


def semantic_scores_steps(frames: list[np.ndarray], query: str, *, stage_ids: Mapping[str, str]) -> Iterator[StageEvent]:
    """Yield Text Query's atomic retrieval stages and return one score per frame.

    ``stage_ids`` maps this feature's stable flow identities onto the six
    retrieval sub-stages (``siglip_load``, ``text_embed``, ``frame_embed``,
    ``normalize``, ``faiss_index``, ``faiss_search``), so the caller's declared
    flow owns the ids and this model adapter stays feature-agnostic. Every metric
    is a fact measured here: the checkpoint/device actually loaded, the embedding
    width, the real frame count, and the FAISS match count/best score. An empty
    frame list still emits ``running`` then ``complete`` for ``frame_embed`` with
    ``total_frames=0`` so the row is never left dangling.
    """
    require_package("faiss", "models")
    import faiss
    import torch

    settings = load_text_settings()
    checkpoint = str(settings.semantic_checkpoint)
    yield StageEvent("Load local SigLIP", "running", "Loading the owner-installed SigLIP processor and weights.", stage_id=stage_ids["siglip_load"])
    processor, model = siglip_model()
    inference_device = device()
    yield StageEvent(
        "Load local SigLIP", "complete", f"Loaded local SigLIP on {inference_device}.",
        metrics={"checkpoint": checkpoint, "device": inference_device}, stage_id=stage_ids["siglip_load"],
    )
    with torch.inference_mode():
        yield StageEvent("Embed query text", "running", "Encoding the query text with SigLIP.", stage_id=stage_ids["text_embed"])
        text = model.get_text_features(
            **_inputs_for_model(processor(text=[query], padding="max_length", return_tensors="pt"), model)
        ).cpu().numpy().astype(np.float32)
        yield StageEvent(
            "Embed query text", "complete", "Encoded the query into one normalized SigLIP embedding.",
            metrics={"dimensions": int(text.shape[1])}, stage_id=stage_ids["text_embed"],
        )
        yield StageEvent(
            "Embed source frames", "running", f"Embedding sampled source frames · 0/{len(frames)}.",
            metrics={"embedded_frames": 0, "total_frames": len(frames)}, stage_id=stage_ids["frame_embed"],
        )
        vectors = []
        for completed, frame in enumerate(frames, 1):
            vector = model.get_image_features(
                **_inputs_for_model(processor(images=frame, return_tensors="pt"), model)
            ).cpu().numpy()[0]
            vectors.append(vector)
            yield StageEvent(
                "Embed source frames", "running", f"Embedding sampled source frames · {completed}/{len(frames)}.",
                metrics={"embedded_frames": completed, "total_frames": len(frames)}, stage_id=stage_ids["frame_embed"],
            )
        yield StageEvent(
            "Embed source frames", "complete", f"Embedded all {len(frames)} sampled source frames.",
            metrics={"embedded_frames": len(frames), "total_frames": len(frames)}, stage_id=stage_ids["frame_embed"],
        )
    dimensions = int(text.shape[1])
    matrix = np.asarray(vectors, dtype=np.float32) if vectors else np.empty((0, dimensions), dtype=np.float32)
    yield StageEvent("Normalize embeddings", "running", "L2-normalizing the image and text embeddings.", stage_id=stage_ids["normalize"])
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    text /= np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-12)
    yield StageEvent(
        "Normalize embeddings", "complete", f"Normalized {matrix.shape[0]} image vectors and the text vector.",
        metrics={"vectors": int(matrix.shape[0]), "dimensions": dimensions}, stage_id=stage_ids["normalize"],
    )
    # Keep FAISS as the local query index even for small single-video searches;
    # it makes the same contract scale to later archive ingestion.
    yield StageEvent("Build local FAISS index", "running", "Building an inner-product index over the frame embeddings.", stage_id=stage_ids["faiss_index"])
    index = faiss.IndexFlatIP(dimensions)
    index.add(matrix)
    yield StageEvent(
        "Build local FAISS index", "complete", f"Indexed {matrix.shape[0]} frame embeddings.",
        metrics={"vectors": int(matrix.shape[0]), "dimensions": dimensions}, stage_id=stage_ids["faiss_index"],
    )
    yield StageEvent("Search FAISS index", "running", "Scoring every frame against the query embedding.", stage_id=stage_ids["faiss_search"])
    if len(matrix):
        ranked_scores, ranked_indices = index.search(text, len(matrix))
        scores = np.empty(len(matrix), dtype=np.float32)
        scores[ranked_indices[0]] = ranked_scores[0]
        matches, best_score = int(len(matrix)), float(ranked_scores[0][0])
    else:
        # No sampled frames: report the honest zero-match result instead of
        # fabricating a score, and let ranked_windows reject the empty input.
        scores = np.empty(0, dtype=np.float32)
        matches, best_score = 0, None
    yield StageEvent(
        "Search FAISS index", "complete", f"Scored {matches} frame embeddings against the query.",
        metrics={"matches": matches, "best_score": best_score}, stage_id=stage_ids["faiss_search"],
    )
    return scores


@lru_cache(maxsize=1)
def grounding_dino_model():
    """Load the local open-vocabulary detector used to seed SAM2."""
    require_package("transformers", "models")
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    settings = load_text_settings()
    require_text_checkpoint(settings.grounding_checkpoint, "Grounding DINO")
    processor = AutoProcessor.from_pretrained(settings.grounding_checkpoint, local_files_only=True)
    # The Transformer's Grounding DINO text enhancer currently leaves a few
    # intermediate tensors in float32. Keeping this compact detector in
    # float32 avoids a Float/Half matmul while VACE and SigLIP remain in half
    # precision on CUDA, where their memory savings matter much more.
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        settings.grounding_checkpoint, local_files_only=True, dtype=torch.float32
    ).to(device()).eval()
    return processor, model


def ground_query(frame: np.ndarray, query: str, threshold: float = 0.25) -> GroundedBox | None:
    """Return the strongest local text-grounded box or no invented target."""
    if not query.strip():
        return None
    require_package("PIL", "models")
    import torch
    from PIL import Image

    processor, model = grounding_dino_model()
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    with torch.inference_mode():
        inputs = _inputs_for_model(processor(images=image, text=[[query]], return_tensors="pt"), model)
        output = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        output, inputs.input_ids, threshold=threshold, text_threshold=threshold, target_sizes=[image.size[::-1]]
    )[0]
    if len(result["boxes"]) == 0:
        return None
    scores = result["scores"].detach().float().cpu().numpy()
    index = int(scores.argmax())
    box = result["boxes"][index].detach().float().cpu().numpy().tolist()
    return GroundedBox(tuple(float(value) for value in box), float(scores[index]))


@lru_cache(maxsize=1)
def sam2_video_model():
    """Load local SAM2 video tracking weights without a browser network call."""
    require_package("transformers", "models")
    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    settings = load_text_settings()
    require_text_checkpoint(settings.segmentation_checkpoint, "SAM2")
    processor = Sam2VideoProcessor.from_pretrained(settings.segmentation_checkpoint, local_files_only=True)
    # SAM2's video-memory attention similarly mixes float32 state with model
    # weights when loaded in half precision in the current Transformers build.
    # This Hiera Tiny checkpoint remains modest in float32 and is more reliable
    # than failing after a successful grounding pass.
    model = Sam2VideoModel.from_pretrained(
        settings.segmentation_checkpoint, local_files_only=True, dtype=torch.float32
    ).to(device()).eval()
    return processor, model


def release_text_models() -> list[str]:
    """Move cached text models off CUDA before clearing their local caches."""
    released: list[str] = []
    for loader, label in ((siglip_model, "SigLIP"), (grounding_dino_model, "Grounding DINO"), (sam2_video_model, "SAM2")):
        if not loader.cache_info().currsize:
            continue
        value = loader()
        model = value[-1] if isinstance(value, tuple) else value
        if hasattr(model, "to"):
            model.to("cpu")
        loader.cache_clear()
        released.append(label)
    return released
