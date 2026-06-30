"""TANGLE slide encoder (Jaume et al., CVPR 2024) - multi-head ABMIL over UNI features.

TANGLE aggregates a slide's UNI (1024-d) patch features into one slide vector with a
multi-head gated-attention MIL head. It was trained by aligning the slide embedding
with the slide's bulk RNA-seq (CLIP-style); at inference only the slide branch runs, so
the gene side is irrelevant here. We ship the **pan-cancer** checkpoint (TANGLE v2,
trained across 27 TCGA cohorts), not the breast-specific ones.

The aggregator below is our own minimal re-implementation, with submodule names
mirroring the checkpoint so the pretrained weights load; we vendor no model code. The
slide vector is ``get_features`` = the multi-head ABMIL forward: a shared pre-attention
MLP, ``n_heads`` gated-attention heads, attention-weighted sum per head, then a
projection back to ``hidden_dim``. Output dim = 512 (the projected ``hidden_dim``).

Weights are not on Hugging Face. The checkpoint (``model.pt``, ~16 MB) is in the
project's public Google Drive; we fetch it by its file id and verify the pinned SHA-256
before loading (Drive has no immutable revision, so the SHA is the pin). Needs the
``[tangle]`` extra (``gdown``).

The expected patch encoder is **UNI** (``-f uni``, 1024-d). Licence: CC-BY-NC-ND-4.0
(non-commercial, no-derivatives).

Reference:  https://github.com/mahmoodlab/TANGLE
Paper:      Jaume et al., "Transcriptomics-guided Slide Representation Learning in
            Computational Pathology", CVPR 2024 - arXiv:2405.11618
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

from raw2features.core.plugins import register

from .base import SlideEmbedder, SlideModelSpec

# Public Google Drive file id + SHA-256 for the pan-cancer (TANGLE v2) checkpoint
# (model.pt). Drive gives no immutable revision, so the SHA is the pin; we verify it on
# download. Architecture (n_heads=4, hidden_dim=512, embedding_dim=1024) comes from the
# checkpoint's config.json and is fixed here.
_TANGLE_GDRIVE_ID = "1f0fNPr5vU6F9Qy0OM3g1TxliBZcVfjmX"
_TANGLE_SHA256 = "786d43c9ce0bd257f954faa0e484e543402eea1e0d014ab71398c3c872b17d55"
_TANGLE_FILENAME = "tangle_pancancer_model.pt"
_N_HEADS = 4
_HIDDEN_DIM = 512
_PATCH_DIM = 1024

_SPEC = SlideModelSpec(
    name="tangle",
    family="tangle",
    source=f"gdrive:{_TANGLE_GDRIVE_ID}",
    embedding_dim=512,
    patch_encoder="uni",
    patch_dim=_PATCH_DIM,
    gated=False,
    license="CC-BY-NC-ND-4.0 (non-commercial)",
    transform_source_url="https://github.com/mahmoodlab/TANGLE",
    doi="10.48550/arXiv.2405.11618",
    weights_sha256=_TANGLE_SHA256,
    notes=(
        "Multi-head ABMIL slide encoder over UNI (1024-d) -> 512-d. Pan-cancer "
        "(27 TCGA cohorts) TANGLE v2 checkpoint, fetched by its public Drive file id + "
        "verified SHA-256 (no HF mirror). Needs the [tangle] extra (gdown). No coords. "
        "`-f uni -s tangle`."
    ),
)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_tangle_weights() -> str:
    """Download the TANGLE checkpoint (cached) and verify the pinned SHA-256."""
    import gdown

    cache_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "raw2features", "tangle"
    )
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, _TANGLE_FILENAME)
    if os.path.exists(dest) and _sha256(dest) == _TANGLE_SHA256:
        return dest

    gdown.download(f"https://drive.google.com/uc?id={_TANGLE_GDRIVE_ID}", dest,
                   quiet=True)
    digest = _sha256(dest)
    if digest != _TANGLE_SHA256:
        raise RuntimeError(
            f"TANGLE weights SHA-256 mismatch: got {digest}, "
            f"expected {_TANGLE_SHA256}. "
            "The Google Drive file may have changed; not loading."
        )
    return dest


def _build_wsi_embedder(patch_dim: int, hidden_dim: int, n_heads: int):
    """TANGLE's multi-head ABMIL ``wsi_embedder``, submodule names matching the
    checkpoint (``pre_attn``, ``attn`` ModuleList of gated heads, ``proj_multihead``).
    """
    import torch.nn as nn

    class _GatedAttn(nn.Module):
        """One BatchedABMIL head: gated attention -> per-patch softmax weight."""

        def __init__(self, dim: int):
            super().__init__()
            self.attention_a = nn.Sequential(
                nn.Linear(dim, dim), nn.Tanh(), nn.Dropout(0.25)
            )
            self.attention_b = nn.Sequential(
                nn.Linear(dim, dim), nn.Sigmoid(), nn.Dropout(0.25)
            )
            self.attention_c = nn.Linear(dim, 1)

        def forward(self, x):  # x: [B, N, dim] -> softmax weights [B, N, 1]
            import torch

            a = self.attention_a(x)
            b = self.attention_b(x)
            return torch.softmax(self.attention_c(a.mul(b)), dim=1)

    class _TangleMH(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_heads = n_heads
            self.hidden_dim = hidden_dim
            self.pre_attn = nn.Sequential(
                nn.Linear(patch_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                nn.GELU(), nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                nn.GELU(), nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim * n_heads),
                nn.LayerNorm(hidden_dim * n_heads), nn.GELU(), nn.Dropout(0.1),
            )
            self.attn = nn.ModuleList(_GatedAttn(hidden_dim) for _ in range(n_heads))
            self.proj_multihead = nn.Linear(hidden_dim * n_heads, hidden_dim)

        def forward(self, bags):  # bags: [B, N, patch_dim] -> [B, hidden_dim]
            import torch

            b, n, _ = bags.shape
            e = self.pre_attn(bags)  # [B, N, hidden*heads]
            # split the last axis into (hidden, heads), heads fastest (einops "(d h)")
            e = e.reshape(b, n, self.hidden_dim, self.n_heads)
            # per-head softmax attention over patches, weighted sum
            attn = torch.stack(
                [self.attn[i](e[:, :, :, i]) for i in range(self.n_heads)], dim=-1
            )  # [B, N, 1, heads]
            slide = torch.sum(e * attn, dim=1)  # [B, hidden, heads]
            slide = slide.reshape(b, self.hidden_dim * self.n_heads)
            return self.proj_multihead(slide)  # [B, hidden]

    return _TangleMH()


@register("slide_embedders", "tangle")
class TangleSlideEmbedder(SlideEmbedder):
    """TANGLE multi-head ABMIL slide encoder (Jaume et al., CVPR 2024)."""

    def __init__(self) -> None:
        super().__init__(_SPEC)
        self._model = None
        self._device = "cpu"

    def load(self, device: str = "cuda", dtype=None) -> TangleSlideEmbedder:
        import torch

        try:
            import gdown  # noqa: F401
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                "TANGLE needs the optional [tangle] extra to fetch its Drive-hosted "
                'weights:\n  pip install "raw2features[tangle]"'
            ) from exc

        ckpt = _fetch_tangle_weights()
        model = _build_wsi_embedder(_PATCH_DIM, _HIDDEN_DIM, _N_HEADS)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        # Keep only the WSI branch (strip the module./wsi_embedder. prefixes); the RNA
        # branch and any projection heads are unused at inference.
        prefix = "wsi_embedder."
        wsi = {}
        for k, v in sd.items():
            kk = k[len("module.") :] if k.startswith("module.") else k
            if kk.startswith(prefix):
                wsi[kk[len(prefix) :]] = v
        missing, _unexpected = model.load_state_dict(wsi, strict=False)
        if missing:
            raise RuntimeError(f"TANGLE wsi_embedder keys not loaded: {missing}")
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
        feat = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        feat = feat.unsqueeze(0).to(self._device)  # [1, N, 1024]
        with torch.inference_mode():
            vec = self._model(feat)  # [1, 512]
        return vec.reshape(-1).float().cpu().numpy()

    def unload(self) -> None:
        import torch

        if self._model is not None:
            self._model.cpu()
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
