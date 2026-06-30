"""FEATHER slide encoders (MahmoodLab MIL-Lab, ICML 2025) - gated, HF-only.

FEATHER is a lightweight supervised ABMIL slide aggregator: it pools a slide's patch
features into a single 512-d slide vector. The released variants differ only in the
patch encoder they consume - CONCH v1.5 (768-d), UNI2-h (1536-d), or UNI (1024-d) - and
all output 512-d. Loaded purely from HuggingFace via transformers ``trust_remote_code``.

Forward, confirmed by reading the model code (the checkpoints ship NO classifier head,
so ``forward()`` / ``return_slide_feats`` hit a missing ``classifier``; the slide vector
is the pre-head pooled feature from ``forward_features``):

    from transformers import AutoModel
    model = AutoModel.from_pretrained(path, trust_remote_code=True)
    wsi_feats, _log = model.forward_features(h)   # h [1, N, patch_dim] -> [1, 512]

Gated: needs a HF token with accepted access to the ``MahmoodLab/abmil.*`` repo.

Reference:  https://huggingface.co/MahmoodLab/abmil.base.conch_v15.pc108-24k
Paper:      Lenz et al., "Do Multiple Instance Learning Models Transfer?", ICML 2025 -
            arXiv:2506.09022
"""

from __future__ import annotations

import numpy as np

from raw2features.core.plugins import register

from .base import SlideEmbedder, SlideModelSpec

# Default spec for direct instantiation; build_slide_embedder() overwrites .spec with
# the registry-sourced one (the per-variant source / patch_encoder / revision).
_DEFAULT = SlideModelSpec(
    name="feather_conch_v15",
    family="feather",
    source="hf-hub:MahmoodLab/abmil.base.conch_v15.pc108-24k",
    embedding_dim=512,
    patch_encoder="conch_v1_5",
    patch_dim=768,
    gated=True,
    license="CC-BY-NC-ND-4.0 (FEATHER, Mahmood Lab; non-commercial)",
    transform_source_url="https://huggingface.co/MahmoodLab/abmil.base.conch_v15.pc108-24k",
    doi="10.48550/arXiv.2506.09022",
)


# One class, three registry names (one per backbone). The slide builder resolves by
# model name, so each name needs its own registration / entry point pointing here.
@register("slide_embedders", "feather_conch_v15")
@register("slide_embedders", "feather_uni_v2")
@register("slide_embedders", "feather_uni")
class FeatherSlideEmbedder(SlideEmbedder):
    """FEATHER: supervised ABMIL slide foundation models (Mahmood Lab, ICML 2025)."""

    def __init__(self) -> None:
        super().__init__(_DEFAULT)
        self._model = None
        self._device = "cpu"

    def load(self, device: str = "cuda", dtype=None) -> FeatherSlideEmbedder:
        from huggingface_hub import snapshot_download
        from transformers import AutoModel

        repo = self.spec.source.split(":", 1)[-1]  # strip the "hf-hub:" prefix
        path = snapshot_download(repo, revision=self.spec.weights_revision)
        model = AutoModel.from_pretrained(path, trust_remote_code=True)
        model.eval().to(device)
        self._model = model
        self._device = device
        return self

    def encode(
        self,
        features: np.ndarray,
        coords: np.ndarray | None = None,  # noqa: ARG002 - FEATHER needs no coords
        patch_size_lv0: int | None = None,  # noqa: ARG002 - nor patch spacing
    ) -> np.ndarray:
        import torch

        if self._model is None:
            raise RuntimeError("call load() before encode()")
        feat = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        feat = feat.unsqueeze(0).to(self._device)  # [1, N, patch_dim]
        with torch.inference_mode():
            wsi_feats, _ = self._model.forward_features(feat)  # [1, 512]
        return wsi_feats.reshape(-1).float().cpu().numpy()

    def unload(self) -> None:
        import torch

        if self._model is not None:
            self._model.cpu()
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
