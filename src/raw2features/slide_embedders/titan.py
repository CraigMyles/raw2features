"""TITAN slide encoder (MahmoodLab/TITAN, CC-BY-NC-ND-4.0).

TITAN encodes a slide's CONCH v1.5 patch features into a single slide vector. It
builds a spatial feature grid from the patch ``coords`` and the level-0 patch
spacing (``patch_size_lv0``), so it needs both - not just the feature matrix. It is
gated: a HuggingFace token with accepted access to ``MahmoodLab/TITAN`` is required.

The load and forward are transcribed from the official model card:

    from transformers import AutoModel
    model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    with torch.autocast("cuda", torch.float16), torch.inference_mode():
        emb = model.encode_slide_from_patch_features(features, coords, patch_size_lv0)

The expected patch encoder is **CONCH v1.5** (``-f conch_v1_5``, 768-d, 512 px @ 20x),
not UNI; ``patch_size_lv0`` is our store's ``patching.level0_patch`` (512 for a 20x
slide, 1024 for 40x), which matches TITAN's definition exactly.

Reference:  https://huggingface.co/MahmoodLab/TITAN
Paper:      Ding et al., Nature Medicine 2025 - doi:10.1038/s41591-025-03982-3
"""

from __future__ import annotations

import numpy as np

from raw2features.core.plugins import register

from .base import SlideEmbedder, SlideModelSpec

_SPEC = SlideModelSpec(
    name="titan",
    family="titan",
    source="hf-hub:MahmoodLab/TITAN",
    embedding_dim=768,
    patch_encoder="conch_v1_5",
    patch_dim=768,
    gated=True,
    license="CC-BY-NC-ND-4.0 (non-commercial)",
    transform_source_url="https://huggingface.co/MahmoodLab/TITAN",
    doi="10.1038/s41591-025-03982-3",
    notes=(
        "Takes CONCH v1.5 (768-d) patch features + level-0 coords. Gated - HF token "
        "with accepted MahmoodLab/TITAN access required."
    ),
)


@register("slide_embedders", "titan")
class TITANSlideEmbedder(SlideEmbedder):
    """TITAN: multimodal whole-slide foundation model (Mahmood Lab, 2024)."""

    def __init__(self) -> None:
        super().__init__(_SPEC)
        self._model = None
        self._device = "cpu"

    def load(self, device: str = "cuda", dtype=None) -> TITANSlideEmbedder:
        from transformers import AutoModel

        # TITAN ships its model class as remote code on the hub (gated). AutoModel
        # with trust_remote_code is the card's documented load path.
        model = AutoModel.from_pretrained(
            "MahmoodLab/TITAN",
            trust_remote_code=True,
            revision=self.spec.weights_revision,  # pin the immutable HF commit
        )
        model.eval().to(device)
        self._model = model
        self._device = device
        return self

    def encode(
        self,
        features: np.ndarray,
        coords: np.ndarray | None = None,
        patch_size_lv0: int | None = None,
    ) -> np.ndarray:
        import torch

        if self._model is None:
            raise RuntimeError("call load() before encode()")
        if coords is None:
            raise ValueError(
                "TITAN needs patch coords (level-0 x,y) to build its feature grid"
            )
        if patch_size_lv0 is None:
            raise ValueError(
                "TITAN needs patch_size_lv0 (the store's patching.level0_patch): the "
                "level-0 spacing between adjacent patches (512 for 20x, 1024 for 40x)"
            )

        feat = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        coords_t = torch.from_numpy(np.ascontiguousarray(coords)).to(torch.int64)
        feat = feat.to(self._device)
        coords_t = coords_t.to(self._device)

        # The card runs the forward under fp16 autocast on CUDA; keep fp32 on CPU.
        use_amp = self._device.startswith("cuda")
        autocast = (
            torch.autocast("cuda", torch.float16)
            if use_amp
            else torch.autocast("cpu", enabled=False)
        )
        with autocast, torch.inference_mode():
            vec = self._model.encode_slide_from_patch_features(
                feat, coords_t, int(patch_size_lv0)
            )
        return vec.reshape(-1).float().cpu().numpy()

    def unload(self) -> None:
        import torch

        if self._model is not None:
            self._model.cpu()
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
