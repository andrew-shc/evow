"""Selected-method model provenance shown by each dashboard methodology control."""

from .contracts import ModelReference


VIDEO_DEPTH_ANYTHING = ModelReference("Video Depth Anything", "depth-anything/Video-Depth-Anything", "temporally consistent relative depth prior", "https://github.com/DepthAnything/Video-Depth-Anything")
GSPLAT = ModelReference("gsplat", "nerfstudio-project/gsplat", "differentiable Gaussian rasterizer and 4D scene fitting", "https://github.com/nerfstudio-project/gsplat")
ANYVIEW = ModelReference("AnyView-DVS", "AnyView-DVS", "camera-conditioned novel-view video diffusion", "https://github.com/NVlabs/AnyView")
SVD = ModelReference("Stable Video Diffusion", "stabilityai/stable-video-diffusion-img2vid-xt", "image-conditioned future-video rollout", "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt")
SIGLIP = ModelReference("SigLIP", "google/siglip-base-patch16-224", "text/video-frame semantic retrieval", "https://huggingface.co/google/siglip-base-patch16-224")
# FAISS is a local library, not a downloaded checkpoint, but it is a real
# retrieval dependency: the query path builds an IndexFlatIP over the SigLIP
# frame vectors. Listing it keeps the atomic retrieval rows' provenance truthful.
FAISS = ModelReference("FAISS", "facebookresearch/faiss", "local inner-product index for text/video similarity search", "https://github.com/facebookresearch/faiss")
GROUNDING_DINO = ModelReference("Grounding DINO", "IDEA-Research/grounding-dino-tiny", "open-vocabulary target detection", "https://github.com/IDEA-Research/GroundingDINO")
SAM2 = ModelReference("SAM 2", "facebook/sam2.1-hiera-tiny", "video target segmentation and tracking", "https://github.com/facebookresearch/sam2")
WAN_VACE = ModelReference("Wan VACE", "Wan-AI/Wan2.1-VACE-1.3B-diffusers", "masked, text-conditioned video editing", "https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers")


def models_for(feature: str, mode: str) -> tuple[ModelReference, ...]:
    """Return only the models actually selected by a feature/method family."""
    if feature == "replay":
        return (VIDEO_DEPTH_ANYTHING, GSPLAT) if mode == "explicit" else (ANYVIEW,)
    if feature == "future":
        return (VIDEO_DEPTH_ANYTHING, GSPLAT) if mode == "explicit" else (SVD,)
    if feature == "selection":
        shared = (SIGLIP, FAISS, GROUNDING_DINO, SAM2)
        return shared + ((VIDEO_DEPTH_ANYTHING, GSPLAT) if mode == "explicit" else ())
    if feature == "editing":
        shared = (GROUNDING_DINO, SAM2, WAN_VACE)
        return shared + ((VIDEO_DEPTH_ANYTHING, GSPLAT) if mode == "explicit" else ())
    return ()
