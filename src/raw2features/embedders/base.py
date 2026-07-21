"""``Embedder`` - the foundation-model seam, and the ``ModelSpec`` it consumes.

Design note: the patcher delivers patches at *exactly* the model's
input size and target MPP, so embedders **normalise only** - they do not apply the
model card's eval-time Resize/CenterCrop, which would re-scale the patch and
distort the MPP we worked to hit exactly. The normalisation mean/std are sourced
from each model's card (recorded in the registry with ``transform_source_url``);
we never guess them.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch


@contextlib.contextmanager
def _forward_ctx(device: str, dtype):
    """Inference context shared by every embedder's ``embed_batch``.

    Wraps ``inference_mode`` + ``autocast`` and, for the true-fp32 path on CUDA,
    forces full-precision matmul/conv (TF32 OFF). TF32 is otherwise on by default
    on recent NVIDIA GPUs and rounds fp32 conv inputs to ~10-bit mantissas, so its
    result is both ~1e-2 away from true fp32 *and* sensitive to convolution
    algorithm choice (which varies with tensor memory layout). That makes a
    ``--amp fp32`` run neither reproducible nor genuinely fp32. Forcing TF32 off
    here keeps fp32 meaning fp32, so an on-GPU batched transform is equivalent to
    the per-patch CPU transform within fp tolerance. AMP runs (bf16/fp16) are
    unaffected -- they autocast convs regardless.
    """
    import torch

    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
    use_amp = dev_type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    true_fp32_cuda = dev_type == "cuda" and not use_amp
    if true_fp32_cuda:
        prev_matmul = torch.backends.cuda.matmul.allow_tf32
        prev_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        with (
            torch.inference_mode(),
            torch.autocast(device_type=dev_type, dtype=dtype, enabled=use_amp),
        ):
            yield
    finally:
        if true_fp32_cuda:
            torch.backends.cuda.matmul.allow_tf32 = prev_matmul
            torch.backends.cudnn.allow_tf32 = prev_cudnn


@dataclass(frozen=True)
class ModelSpec:
    """Provenance-complete description of a feature extractor.

    ``transform_source_url`` and the (mean, std, input_size) it documents are
    mandatory and must come from the model's authoritative card/repo.
    """

    name: str
    family: str  # "timm" | "torchvision"
    source: str  # hf-hub id, or "torchvision://<name>?weights=<enum>"
    embedding_dim: int
    input_size: int
    # How the model output is reduced to one vector per patch:
    #   "cls"     - the class token,            output[:, 0]
    #   "pooled"  - the model's own global pool (e.g. resnet50 avg-pool)
    #   "cls_concat_meanpatch" - concat(class token, mean over PATCH tokens),
    #       skipping ``reg_tokens`` register tokens that sit between them; the
    #       Virchow2 recipe, sourced from its card. embedding_dim is then 2*token_dim.
    pooling: str  # "cls" | "pooled" | "cls_concat_meanpatch"
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    transform_source_url: str
    license: str
    gated: bool
    # Number of register tokens between the class token and the patch tokens in the
    # output sequence (output[:, 1 : 1+reg_tokens]). Only consulted by the
    # ``cls_concat_meanpatch`` pooling, which means patch tokens start at
    # ``1 + reg_tokens``. 0 for every other model.
    reg_tokens: int = 0
    # Where the normalisation (mean/std/interpolation) comes from:
    #   "registry"      - the mean/std/interpolation recorded below (the default;
    #                     transcribed from the card, unchanged for every v1 model).
    #   "pretrained_cfg"- resolved from the loaded model's ``pretrained_cfg`` at
    #                     load time (the faithful path the Virchow2 card uses via
    #                     ``resolve_data_config``). The registry values are the
    #                     card-documented fallback used offline (no weights) and are
    #                     asserted to match once the weights are loaded.
    transform_source: str = "registry"
    # Inference precision the model's card uses/recommends ("fp32" | "bf16" | "fp16").
    # Sourced from the card like mean/std. ``--amp auto`` resolves to this per model;
    # an explicit ``--amp`` overrides it. Default fp32 (full precision) when a card
    # gives no guidance.
    inference_amp: str = "fp32"
    interpolation: str = "bilinear"
    # Microns-per-pixel the model was trained at / recommends (e.g. 0.5 == 20x).
    # Paper-sourced -- model cards usually omit it. ``input_size`` is the patch size
    # in pixels, so the field of view is ``input_size * recommended_mpp`` microns.
    # None for the MPP-agnostic ImageNet baselines (resnet50, dinov2). The pipeline
    # warns when ``target_mpp`` is extracted at a different scale.
    recommended_mpp: float | None = None
    # Patch size in PIXELS to extract at, before the model's own resize to
    # ``input_size``. Defaults to ``input_size`` (extract the input verbatim,
    # normalise-only). Set only where the card extracts a DIFFERENT size and
    # resizes -- e.g. conch_v1_5 extracts 512 px @ 20x then resizes to 448.
    # With ``recommended_mpp`` it fixes the field of view (``patch_px * mpp``
    # µm) the model runs at. None -> ``input_size``; read via :attr:`extract_px`.
    recommended_patch_px: int | None = None
    # Imaging modality the model consumes: "brightfield" (RGB H&E; the default) or
    # "multiplex" (a stack of single-marker channels + marker ids, e.g. KRONOS/CODEX).
    # The runner reads RGB patches for brightfield and native N-channel patches +
    # channel_names for multiplex.
    modality: str = "brightfield"
    timm_kwargs: dict[str, Any] = field(default_factory=dict)
    # Optional checkpoint load for timm models whose weights are a bare HF .pth/.bin
    # (no timm pretrained_cfg, so the hf-hub one-liner 404s). When set, the timm family
    # builds ``source`` with pretrained=False and loads this checkpoint. Keys:
    # ``repo``, ``filename`` (required); ``state_dict_key`` (sub-dict to pull, e.g.
    # "teacher"), ``strip_prefixes`` (list), ``strict`` (default True).
    checkpoint: dict[str, Any] | None = None
    # sha256 of the model's weight file + the HuggingFace commit it was pinned at
    # (HF repos are mutable). Both are recorded in every output's provenance, so an
    # embedding is traceable to the exact weights that produced it.
    weights_sha256: str | None = None
    weights_revision: str | None = None
    # Exact artifact whose bytes ``weights_sha256`` identifies (or a stable weight
    # enum for torchvision). Kept explicit so provenance never has to guess which
    # file a repo-level loader selected.
    weights_filename: str | None = None
    # Experimental models may have a deliberately narrower integrity guarantee than
    # the stable registry contract.  The reason must be recorded in ``notes`` and docs.
    experimental: bool = False
    notes: str = ""
    # Resolvable DOI for the model's paper (FAIR findability) - a journal DOI when one
    # exists, else the arXiv DataCite DOI (10.48550/arXiv.*); None only for open-weights
    # releases with no publication. Flows into the output store's model provenance.
    doi: str | None = None
    # Resolved strategy contract for a derived multiplex output. Ordinary registry
    # models leave this unset, preserving their v0.1 fingerprints byte-for-byte. Kept
    # last so adding it does not shift any pre-existing positional constructor fields.
    multiplex: dict[str, Any] | None = None

    @property
    def extract_px(self) -> int:
        """Patch size (px) to extract at, before the model's resize to input_size."""
        return int(self.recommended_patch_px or self.input_size)


class Embedder(ABC):
    """Abstract feature extractor wrapping one model."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._model: Any = None
        self._device: str = "cpu"
        self._dtype: Any = None

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def modality(self) -> str:
        """``"brightfield"`` (RGB) or ``"multiplex"`` (marker stack) - from the spec."""
        return self.spec.modality

    def set_panel(self, channel_names: list[str] | None) -> dict:
        """Bind a slide's marker panel (multiplex). No-op for brightfield models.

        Multiplex embedders (e.g. KRONOS) resolve ``channel_names`` -> kept channels +
        per-marker normalisation + marker ids here, once per slide. Returns a summary
        for provenance (kept/dropped markers); ``{}`` for brightfield.
        """
        return {}

    @property
    def embedding_dim(self) -> int:
        return self.spec.embedding_dim

    @property
    def transform_signature(self) -> tuple:
        """Hashable identity of this model's preprocessing.

        Two embedders with equal signatures produce the *same* transformed tensor
        from the same patches, so :meth:`transform_batch` can be computed once and
        shared across them (the decode-once fan-out also shares the transform). The
        tuple covers every input to :meth:`transform_batch`: the target
        ``input_size`` (which determines whether/how a resize happens), the
        normalisation ``mean``/``std``, and the ``interpolation`` used on resize.
        """
        return (
            self.spec.input_size,
            self.spec.mean,
            self.spec.std,
            self.spec.interpolation,
        )

    @property
    def transform_input_dtype(self) -> str:
        """Host-patch contract accepted by :meth:`transform_batch`.

        Every built-in RGB embedder currently consumes HWC ``uint8`` patches.  A
        plugin whose transform genuinely consumes normalized floating-point input
        may override this with ``"float32_0_1"``.  Multiplex adapters consult this
        explicit seam instead of assuming that quantizing to uint8 is universally
        valid.  It is intentionally not part of ordinary brightfield fingerprints,
        preserving their existing output identity; a multiplex strategy records the
        resolved value in its own contract.
        """

        return "uint8"

    @abstractmethod
    def load(
        self,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        compile: bool = False,
    ) -> Embedder:
        """Construct and move the model to ``device``; return ``self``.

        ``compile`` torch.compiles the model **once** here (off by default). It is a
        speed-only knob: compilation reorders fp ops so the output is not
        bit-identical, but stays within fp tolerance. Compiling at load amortises the
        one-off warmup over a warm-worker shard (like the model load itself).
        """

    def _maybe_compile(self, compile: bool) -> None:
        """torch.compile ``self._model`` in place when ``compile`` is set.

        Called by concrete ``load`` implementations after the model is built, moved to
        the device and put in eval mode. A no-op when ``compile`` is False (the
        default). ``dynamic=True`` marks the batch dim dynamic so the smaller final
        batch of a slide does not trigger a per-batch recompile (one compile per
        process, reused for every batch shape).
        """
        if not compile:
            return
        import torch

        self._model = torch.compile(self._model, dynamic=True)

    @abstractmethod
    def embed_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Embed a (B, 3, H, W) float tensor -> (B, embedding_dim) float32 (CPU)."""

    def transform(self, patch_hwc_uint8: np.ndarray) -> torch.Tensor:
        """HWC uint8 patch -> normalised CHW float tensor (resize only on mismatch)."""
        from .transforms import to_model_tensor

        return to_model_tensor(patch_hwc_uint8, self.spec)

    def transform_batch(
        self, patches_hwc_uint8: list[np.ndarray], device: str
    ) -> torch.Tensor:
        """Stack & normalise a batch of HWC uint8 patches on ``device``.

        Returns a ``(B, 3, input_size, input_size)`` float32 tensor that already
        lives on ``device`` -- the normalisation (uint8 -> /255 -> (x-mean)/std)
        runs as batched GPU kernels after a single host->device copy, freeing the
        CPU and using the otherwise-idle GPU. For the common ``patch_px ==
        input_size`` case this is arithmetically the per-patch :meth:`transform`
        applied to the whole batch at once. The default implementation lives in
        ``transforms.to_model_batch``; subclasses may override.
        """
        from .transforms import to_model_batch

        return to_model_batch(patches_hwc_uint8, self.spec, device)

    def unload(self) -> None:
        self._model = None
