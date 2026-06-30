"""MADELEINE slide encoder (MahmoodLab/madeleine) - optional, needs ``madeleine``.

MADELEINE is a multi-stain ABMIL aggregator (ECCV 2024) that pools a slide's CONCH v1
patch features into one 512-d slide vector. It loads through the authors' own
``madeleine`` package (not ``transformers``), so it gets its own family with the import
deferred to :meth:`load`.

Load + forward, confirmed by introspecting the real model:

    from madeleine.models.factory import create_model_from_pretrained
    model, _dtype = create_model_from_pretrained(local_dir)  # downloads the gated repo
    model = model.to(device).eval()
    with torch.inference_mode():
        slide_vec = model.encode_he(feats, device)           # [1, N, 512] -> [1, 512]

The expected patch encoder is **CONCH v1** (``-f conch``, 512-d). It is gated (a HF
token with accepted ``MahmoodLab/madeleine`` access is required).

Licence note: the HF card tags ``mit`` but the GitHub repo's LICENSE says
CC-BY-NC-ND-4.0 - a conflict we record, not resolve (see registry.yaml). Install::

    pip install "raw2features[madeleine]"
    pip install git+https://github.com/mahmoodlab/MADELEINE.git

Reference:  https://huggingface.co/MahmoodLab/madeleine
Paper:      Jaume et al., ECCV 2024 - arXiv:2408.02859
"""

from __future__ import annotations

import os

import numpy as np

from raw2features.core.plugins import register

from .base import SlideEmbedder, SlideModelSpec

_SPEC = SlideModelSpec(
    name="madeleine",
    family="madeleine",
    source="hf-hub:MahmoodLab/madeleine",
    embedding_dim=512,
    patch_encoder="conch",
    patch_dim=512,
    gated=True,
    license="MIT (HF card) / CC-BY-NC-ND-4.0 (GitHub LICENSE) - CONFLICT, verify",
    transform_source_url="https://huggingface.co/MahmoodLab/madeleine",
    doi="10.48550/arXiv.2408.02859",
    weights_sha256="34437fe7cf6e1d9b6fb41ef592416ef890dc07c599ca1cc8d1ff00c40ce23496",
    weights_revision="a5eca29194526644eaa725cbad62c0b5023007db",
    notes=(
        "Multi-stain ABMIL over CONCH v1 (512-d) patch features -> 512-d slide vector. "
        "Loads via the madeleine package's create_model_from_pretrained; gated - needs "
        "an accepted MahmoodLab/madeleine gate + the [madeleine] extra and git package."
    ),
)


@register("slide_embedders", "madeleine")
class MadeleineSlideEmbedder(SlideEmbedder):
    """MADELEINE: multi-stain slide representation learning (Mahmood Lab, ECCV 2024)."""

    def __init__(self) -> None:
        super().__init__(_SPEC)
        self._model = None
        self._device = "cpu"

    def load(self, device: str = "cuda", dtype=None) -> MadeleineSlideEmbedder:
        try:
            from madeleine.models.factory import create_model_from_pretrained
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                "MADELEINE needs the optional `madeleine` package:\n"
                '  pip install "raw2features[madeleine]"\n'
                "  pip install git+https://github.com/mahmoodlab/MADELEINE.git"
            ) from exc

        # The factory downloads the (gated) repo into local_dir and loads it. Point it
        # at a stable cache dir so it fetches once; it returns (model, dtype).
        local_dir = os.environ.get("RAW2FEATURES_MADELEINE_DIR") or os.path.join(
            os.path.expanduser("~/.cache/raw2features"), "madeleine"
        )
        os.makedirs(local_dir, exist_ok=True)
        model, _ = create_model_from_pretrained(local_dir)
        model.eval().to(device)
        self._model = model
        self._device = device
        return self

    def encode(
        self,
        features: np.ndarray,
        coords: np.ndarray | None = None,  # noqa: ARG002 - MADELEINE needs no coords
        patch_size_lv0: int | None = None,  # noqa: ARG002 - nor patch spacing
    ) -> np.ndarray:
        import torch

        if self._model is None:
            raise RuntimeError("call load() before encode()")

        feat = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        feat = feat.unsqueeze(0).to(self._device)  # [1, N, 512]
        with torch.inference_mode():
            vec = self._model.encode_he(feat, self._device)  # [1, 512]
        return vec.reshape(-1).float().cpu().numpy()

    def unload(self) -> None:
        import torch

        if self._model is not None:
            self._model.cpu()
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
