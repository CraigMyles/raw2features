"""PRISM slide encoder (paige-ai/Prism, CC-BY-NC-ND-4.0).

PRISM is a Perceiver-based vision aggregator (paired with a BioGPT text decoder for its
vision-language objective; we use only the image path) that pools a slide's **Virchow
v1** patch features into a single 1280-d slide vector. Unlike TITAN it needs **no
coordinates** and no patch spacing - just the ``(N, 2560)`` feature matrix. It is gated:
a HuggingFace token with accepted access to ``paige-ai/Prism`` is required.

The load and forward are transcribed from the official model card:

    from transformers import AutoModel
    model = AutoModel.from_pretrained("paige-ai/Prism", trust_remote_code=True)
    with torch.autocast("cuda", torch.float16), torch.inference_mode():
        reprs = model.slide_representations(tile_embeddings)   # [1, N, 2560]
        slide_vec = reprs["image_embedding"]                   # [1, 1280]

The expected patch encoder is **Virchow v1** (``-f virchow``, 2560-d = concat of the
class token and the mean of the patch tokens) - ``perceiver_config.context_dim`` is
hard-fixed to 2560, so Virchow2 or a class-token-only 1280-d feature is the wrong input.

PRISM's remote modeling code imports ``environs`` and ``sacremoses`` at module load (for
the BioGPT decoder path, unused here), so they ship in the ``[prism]`` extra.

Reference:  https://huggingface.co/paige-ai/Prism
Paper:      Shaikovski et al., arXiv:2405.10254
"""

from __future__ import annotations

import numpy as np

from raw2features.core.plugins import register

from .base import SlideEmbedder, SlideModelSpec

_SPEC = SlideModelSpec(
    name="prism",
    family="prism",
    source="hf-hub:paige-ai/Prism",
    embedding_dim=1280,
    patch_encoder="virchow",
    patch_dim=2560,
    gated=True,
    license="CC-BY-NC-ND-4.0 (PRISM, Paige + Microsoft Research; non-commercial)",
    transform_source_url="https://huggingface.co/paige-ai/Prism",
    doi="10.48550/arXiv.2405.10254",
    weights_sha256="01a0f7bcfd1559de31794e08a99e9fe4f9bd758c3b9f1482fe9cb3f5e1e7c5e1",
    weights_revision="b5ef311e78a0811c71ecf618f6f7140e9ebb5fc6",
    notes=(
        "Perceiver aggregator over Virchow v1 (2560-d) patch features -> 1280-d slide "
        "vector. No coords needed. Gated - needs an accepted paige-ai/Prism gate and "
        "the [prism] extra (environs, sacremoses)."
    ),
)


@register("slide_embedders", "prism")
class PrismSlideEmbedder(SlideEmbedder):
    """PRISM: vision-language slide foundation model (Paige, 2024)."""

    def __init__(self) -> None:
        super().__init__(_SPEC)
        self._model = None
        self._device = "cpu"

    def load(self, device: str = "cuda", dtype=None) -> PrismSlideEmbedder:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            "paige-ai/Prism",
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
        coords: np.ndarray | None = None,  # noqa: ARG002 - PRISM needs no coords
        patch_size_lv0: int | None = None,  # noqa: ARG002 - nor patch spacing
    ) -> np.ndarray:
        import torch

        if self._model is None:
            raise RuntimeError("call load() before encode()")

        feat = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        feat = feat.unsqueeze(0).to(self._device)  # [1, N, 2560]

        # The card runs the forward under fp16 autocast on CUDA; keep fp32 on CPU.
        use_amp = self._device.startswith("cuda")
        autocast = (
            torch.autocast("cuda", torch.float16)
            if use_amp
            else torch.autocast("cpu", enabled=False)
        )
        with autocast, torch.inference_mode():
            reprs = self._model.slide_representations(feat)
            vec = reprs["image_embedding"]  # [1, 1280]
        return vec.reshape(-1).float().cpu().numpy()

    def unload(self) -> None:
        import torch

        if self._model is not None:
            self._model.cpu()
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
