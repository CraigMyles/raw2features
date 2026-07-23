"""Prov-GigaPath and GigaPath-Flash slide encoders (LongNet) - gated.

The GigaPath slide encoder needs ``flash-attn``, a hard dependency with no SDPA
fallback. Prebuilt wheels exist for x86 with torch 2.7-2.9; other combinations (torch
2.6, or aarch64 with torch >=2.10) require a source build with ``nvcc``. When flash-attn
is not installed, the slow tests for this encoder skip, as for any gated or optional
model. Needs the ``[gigapath_slide]`` extra + the gigapath git package + ``fairscale`` +
a matching flash-attn.

The paired slide encoders are 12-layer LongNet models (dilated-attention ViTs).
The original maps 1536-d GigaPath tile features to 768 dimensions; Flash maps
384-d Flash tile features to 384 dimensions. Both are position-aware: they index
a fixed 2-D sine-cosine positional grid from patch ``coords``. raw2features stores
coordinates in source level-0 pixels, so it sets LongNet's runtime coordinate divisor
to the level-0 extent of one extracted tile. This normalizes those coordinates into
the tile-index frame of upstream's default 256-pixel tiling. The encoder therefore
needs the coordinates and tile extent as well as the feature matrix. Gated model
access is required for the corresponding Hugging Face repository.

Load and forward follow the authors' own pipeline
(``gigapath.pipeline.run_inference_with_slide_encoder``):

    from gigapath.slide_encoder import create_model
    model = create_model(ckpt, architecture, patch_dim, global_pool=True)
    model.tile_size = patch_size_lv0          # runtime coords_to_pos divisor
    with torch.autocast("cuda", torch.float16), torch.inference_mode():
        out = model(features[1,N,D], coords[1,N,2], all_layer_embed=True)
    slide_vec = out[-1]                        # last_layer_embed -> [1, output_dim]

``global_pool=True`` is the authors' recommended setting: the demo notes the CLS
token is *not* trained during slide pretraining, so the slide vector is the
mean-pooled last layer. ``create_model``'s own weight download forces the repo
HEAD with no revision, so we download the pinned commit ourselves and hand it the
local path (keeps the weights sha-pinnable).

The registry binds each slide model to its matching patch encoder. LongNet is
constructed with its upstream 256-pixel default, which determines its dilated-attention
segment lengths. raw2features then passes level-0 coordinates unchanged and adapts only
the runtime ``coords_to_pos`` divisor to that coordinate frame.

Needs the ``[gigapath_slide]`` extra plus two post-install steps: the gigapath
git package (with ``fairscale``; it vendors torchscale, which ``load`` aliases as
top-level ``torchscale``), and a prebuilt ``flash-attn`` wheel (a hard dep with no
SDPA fallback; needs an Ampere+ GPU at runtime). See pyproject's
``[gigapath_slide]`` comment and docs/MODELS.md.

Reference:  https://huggingface.co/prov-gigapath/prov-gigapath
Paper:      Xu et al., "A whole-slide foundation model for digital pathology from
            real-world data", Nature 2024 - doi:10.1038/s41586-024-07441-w
"""

from __future__ import annotations

import operator

import numpy as np

from raw2features.core.plugins import register
from raw2features.embedders._hub import download_pinned_hf_file, verify_sha256

from .base import SlideEmbedder, SlideModelSpec

_SPEC = SlideModelSpec(
    name="gigapath_slide",
    family="gigapath_slide",
    source="hf-hub:prov-gigapath/prov-gigapath",
    embedding_dim=768,
    patch_encoder="gigapath",
    patch_dim=1536,
    gated=True,
    license="Apache-2.0 (HF card; research-only caveat in prose)",
    transform_source_url="https://huggingface.co/prov-gigapath/prov-gigapath",
    doi="10.1038/s41586-024-07441-w",
    notes=(
        "LongNet slide encoder over GigaPath (1536-d) patch features + level-0 coords. "
        "Gated - HF token with accepted prov-gigapath/prov-gigapath access required."
    ),
    architecture="gigapath_slide_enc12l768d",
)


@register("slide_embedders", "gigapath_flash_slide")
@register("slide_embedders", "gigapath_slide")
class GigapathSlideEmbedder(SlideEmbedder):
    """GigaPath LongNet slide encoder for original and Flash checkpoints."""

    def __init__(self) -> None:
        super().__init__(_SPEC)
        self._model = None
        self._device = "cpu"

    def load(self, device: str = "cuda", dtype=None) -> GigapathSlideEmbedder:
        import importlib
        import sys

        try:
            # gigapath vendors torchscale at `gigapath.torchscale` but imports it
            # absolutely. Alias the vendored package as top-level `torchscale` so those
            # imports resolve to it (Python finds `torchscale.<sub>` via the package
            # __path__) - so no separate PyPI torchscale is needed.
            if "torchscale" not in sys.modules:
                sys.modules["torchscale"] = importlib.import_module(
                    "gigapath.torchscale"
                )
            from gigapath.slide_encoder import create_model
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                "GigaPath's slide encoder needs the optional gigapath stack:\n"
                '  pip install "raw2features[gigapath_slide]"\n'
                "  pip install --no-deps fairscale "
                '"git+https://github.com/prov-gigapath/prov-gigapath.git@'
                '9d42a60babe04359978d5ad2eb94e7b3bcf9ca39"\n'
                "  + a prebuilt flash-attn wheel for your torch+CUDA "
                "(hard dep, no SDPA fallback; Ampere+ GPU at runtime)"
            ) from exc

        # create_model's hf_hub path forces the repo HEAD (no revision); download the
        # pinned commit ourselves and pass the local file so the weights stay pinnable.
        ckpt = download_pinned_hf_file(
            self.spec.source,
            self.spec.weights_filename or "slide_encoder.pth",
            self.spec.weights_revision,
        )
        verify_sha256(ckpt, self.spec.weights_sha256, what=self.spec.name)
        architecture = self.spec.architecture or "gigapath_slide_enc12l768d"
        model = create_model(
            ckpt,
            architecture,
            int(self.spec.patch_dim),
            global_pool=True,
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
                "GigaPath's slide encoder needs patch coords (level-0 x,y) to index "
                "its positional grid"
            )
        if patch_size_lv0 is None:
            raise ValueError(
                "GigaPath's slide encoder needs patch_size_lv0 (the store's "
                "patching.level0_patch): the level-0 extent of one extracted tile "
                "used as its positional-grid divisor"
            )
        try:
            if isinstance(patch_size_lv0, (bool, np.bool_)):
                raise TypeError
            tile_extent_lv0 = int(operator.index(patch_size_lv0))
        except (TypeError, OverflowError) as exc:
            raise ValueError("patch_size_lv0 must be a positive integer") from exc
        if tile_extent_lv0 <= 0:
            raise ValueError("patch_size_lv0 must be a positive integer")

        coords_array = np.ascontiguousarray(coords, dtype=np.float32)
        if coords_array.ndim != 2 or coords_array.shape[1] != 2:
            raise ValueError("GigaPath patch coords must have shape (N, 2)")
        if not np.isfinite(coords_array).all():
            raise ValueError("GigaPath patch coords must contain only finite values")

        slide_ngrids = int(self._model.slide_ngrids)
        cells = np.floor(coords_array / tile_extent_lv0)
        if np.any(cells < 0) or np.any(cells >= slide_ngrids):
            raise ValueError(
                "GigaPath patch coords fall outside its positional grid after "
                f"division by patch_size_lv0={tile_extent_lv0}; each axis must map "
                f"to a cell in [0, {slide_ngrids - 1}]"
            )

        # LongNet maps a coordinate to a grid cell with floor(coord / tile_size).
        # Stored coordinates are source level-0 pixels, so the level-0 extent of one
        # extracted tile is the divisor corresponding to upstream's 256-pixel
        # target-resolution frame. Overlapping tokens may share a positional cell;
        # LongNet still receives every token.
        self._model.tile_size = tile_extent_lv0

        feat = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        coords_t = torch.from_numpy(coords_array)
        feat = feat.unsqueeze(0).to(self._device)  # [1, N, patch_dim]
        coords_t = coords_t.unsqueeze(0).to(self._device)  # [1, N, 2]

        use_amp = self._device.startswith("cuda")
        autocast = (
            torch.autocast("cuda", torch.float16)
            if use_amp
            else torch.autocast("cpu", enabled=False)
        )
        with autocast, torch.inference_mode():
            # all_layer_embed=True + [-1] is the authors' last_layer_embed (the slide
            # vector their downstream tasks use); with global_pool it is the
            # mean-pooled, normed final layer.
            out = self._model(feat, coords_t, all_layer_embed=True)
        return out[-1].reshape(-1).float().cpu().numpy()

    def unload(self) -> None:
        import torch

        if self._model is not None:
            self._model.cpu()
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
