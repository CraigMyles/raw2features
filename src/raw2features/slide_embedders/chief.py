"""CHIEF slide encoder (Wang et al., Nature 2024) - gated-attention MIL over CTransPath.

CHIEF aggregates a slide's CTransPath (768-d) patch features into one 768-d slide
vector with a gated-attention MIL head. The slide vector is the attention-weighted
sum of the input features (``WSI_feature`` in the authors' forward), which is
**independent of CHIEF's anatomical-site text branch** (in ``CHIEF.forward`` it is
``softmax(attention).T @ features``, not a function of the site input) - so we need
no site input, and we drop the ``organ_embedding`` buffer (it depends on a second
weight file, ``Text_emdding.pth``, that we never load).

The aggregator below is our own minimal re-implementation, with submodule names
mirroring the checkpoint so the pretrained weights load; we vendor no model code.
Output dim = 768 (= the CTransPath patch dim; the head preserves it).

Weights are not on Hugging Face. The WSI aggregator checkpoint
(``CHIEF_pretraining.pth``) is in the project's public Google Drive; we fetch it by
its file id and verify it against the pinned SHA-256 before loading (Drive has no
immutable revision, so the SHA is the pin). Needs the ``[chief]`` extra (``gdown``).

The expected patch encoder is **CTransPath** (``-f ctranspath``, 768-d). The model
code is MIT (ours); the **weights are GPL-3.0** (non-commercial academic) - see
docs/MODELS.md.

Reference:  https://github.com/hms-dbmi/CHIEF
Paper:      Wang et al., "A pathology foundation model for cancer diagnosis and
            prognosis prediction", Nature 2024 - doi:10.1038/s41586-024-07894-z
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

from raw2features.core.plugins import register

from .base import SlideEmbedder, SlideModelSpec

# Public Google Drive file id + SHA-256 for the WSI aggregator checkpoint
# (CHIEF_pretraining.pth). Drive gives no immutable revision, so the SHA is the pin;
# we verify it on download.
_CHIEF_GDRIVE_ID = "10bJq_ayX97_1w95omN8_mESrYAGIBAPb"
_CHIEF_SHA256 = "6a46d200b32a65e5ce4774611b889b5f1bbf7a39f9111321a2a1b5dbdb9996b8"
_CHIEF_FILENAME = "CHIEF_pretraining.pth"

_SPEC = SlideModelSpec(
    name="chief",
    family="chief",
    source=f"gdrive:{_CHIEF_GDRIVE_ID}",
    embedding_dim=768,
    patch_encoder="ctranspath",
    patch_dim=768,
    gated=False,
    license="GPL-3.0 (weights; non-commercial academic)",
    transform_source_url="https://github.com/hms-dbmi/CHIEF",
    doi="10.1038/s41586-024-07894-z",
    weights_sha256=_CHIEF_SHA256,
    notes=(
        "Gated-attention MIL slide encoder over CTransPath (768-d) -> 768-d. Weights "
        "fetched by their public Drive file id + verified SHA-256 (no HF mirror). "
        "Needs the [chief] extra (gdown). Site-agnostic WSI_feature; no coords. "
        "`-f ctranspath -s chief`."
    ),
)


def _fetch_chief_weights() -> str:
    """Download CHIEF_pretraining.pth from Drive (cached) and verify the pinned SHA."""
    import gdown

    cache_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "raw2features", "chief"
    )
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, _CHIEF_FILENAME)

    if os.path.exists(dest) and _sha256(dest) == _CHIEF_SHA256:
        return dest

    url = f"https://drive.google.com/uc?id={_CHIEF_GDRIVE_ID}"
    gdown.download(url, dest, quiet=True)
    digest = _sha256(dest)
    if digest != _CHIEF_SHA256:
        raise RuntimeError(
            f"CHIEF weights SHA-256 mismatch: got {digest}, expected {_CHIEF_SHA256}. "
            "The Google Drive file may have changed; not loading."
        )
    return dest


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_attention_net(size: list[int]):
    """CHIEF's pre-attention FC + gated-attention net, submodule names matching the
    checkpoint (``attention_net.0`` linear, ``attention_net.3.attention_{a,b,c}``)."""
    import torch.nn as nn

    class _AttnNetGated(nn.Module):
        def __init__(self, L: int, D: int):
            super().__init__()
            self.attention_a = nn.Sequential(
                nn.Linear(L, D), nn.Tanh(), nn.Dropout(0.25)
            )
            self.attention_b = nn.Sequential(
                nn.Linear(L, D), nn.Sigmoid(), nn.Dropout(0.25)
            )
            self.attention_c = nn.Linear(D, 1)

        def forward(self, x):
            a = self.attention_a(x)
            b = self.attention_b(x)
            return self.attention_c(a.mul(b)), x

    return nn.Sequential(
        nn.Linear(size[0], size[1]),
        nn.ReLU(),
        nn.Dropout(0.25),
        _AttnNetGated(size[1], size[2]),
    )


@register("slide_embedders", "chief")
class CHIEFSlideEmbedder(SlideEmbedder):
    """CHIEF gated-attention MIL slide encoder (Wang et al., Nature 2024)."""

    def __init__(self) -> None:
        super().__init__(_SPEC)
        self._model = None
        self._device = "cpu"

    def load(self, device: str = "cuda", dtype=None) -> CHIEFSlideEmbedder:
        import torch

        try:
            import gdown  # noqa: F401
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                "CHIEF needs the optional [chief] extra to fetch its Drive-hosted "
                'weights:\n  pip install "raw2features[chief]"'
            ) from exc

        ckpt = _fetch_chief_weights()
        # size_arg="small": [in=768, hidden=512, attn=256]. We build only the attention
        # pathway (all we use); the classifier/site branches are ignored on load.
        model = _build_attention_net([768, 512, 256])
        sd = torch.load(ckpt, map_location="cpu")
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        # organ_embedding needs the separate Text_emdding.pth and feeds only the
        # site-dependent branch; drop it (we use the site-agnostic WSI_feature).
        sd.pop("organ_embedding", None)
        # Keep only the attention_net.* keys; the rest (classifiers, att_head,
        # text_to_vision) belong to branches we don't run.
        anet = {
            k[len("attention_net.") :]: v
            for k, v in sd.items()
            if k.startswith("attention_net.")
        }
        missing, _unexpected = model.load_state_dict(anet, strict=False)
        if missing:
            raise RuntimeError(f"CHIEF attention_net keys not loaded: {missing}")
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
        h = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        h = h.to(self._device)  # [N, 768]
        with torch.inference_mode():
            # WSI_feature = softmax(gated-attention).T @ original features
            # (site-agnostic - independent of CHIEF's anatomical-site branch).
            attn, _ = self._model(h)  # [N, 1]
            attn = torch.softmax(attn.transpose(1, 0), dim=1)  # [1, N]
            vec = torch.mm(attn, h)  # [1, 768]
        return vec.reshape(-1).float().cpu().numpy()

    def unload(self) -> None:
        import torch

        if self._model is not None:
            self._model.cpu()
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
