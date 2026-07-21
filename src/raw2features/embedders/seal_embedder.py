"""SEAL patch encoders (MahmoodLab) - spatial-transcriptomics LoRA fine-tunes; gated.

SEAL fine-tunes a frozen pathology backbone (CONCH, UNI2-h, …) with a LoRA adapter
aligned to spatial transcriptomics. The HF checkpoints are LoRA **deltas**, so loading
for inference still means: build the frozen base + merge the LoRA. We use SEAL's own
image-only loader (``ModelMixin.get_img_model``) - the gene/omics model is not loaded.
Output dim = the base encoder's (512 for CONCH, 1536 for UNI2-h).

SEAL is **experimental in raw2features v0.2.0**. The LoRA adapter is revision-pinned,
SHA-256 verified, and loaded with a frozen constructor contract, but SEAL's upstream
factory still fetches the frozen CONCH/UNI2-h base from mutable Hugging Face HEAD. The
persisted composite fingerprint records that limitation explicitly; SEAL is outside the
stable exact-weight pinning guarantee until the base download can be injected locally.

The ``[seal]`` extra installs the ``seal`` package from a **pinned fork commit**
(``cjmielke/SEAL`` @ ``5334490`` - the open PR #1 to ``mahmoodlab/SEAL``). Upstream
``seal`` reads ``conf/config.yaml`` and ``cache/organ_ids.json`` relative to the CWD, so
it only runs from a repo checkout; the fork resolves them from the install dir (its
``find_config_yaml`` / ``find_organ_ids``), which is what lets this run as a library.
Swap the install URL to ``mahmoodlab/SEAL`` once the PR merges.

Needs the ``[seal]`` extra (``peft`` + ``scanpy``) + the pinned ``seal`` fork + the
backbone's own pinned package (for CONCH, revision
``141cc09c7d4ff33d8eda562bd75169b457f71a62``), and an ``HF_TOKEN`` env
var with accepted access to ``MahmoodLab/SEAL`` and the base model's gate (SEAL reads
``HF_TOKEN`` from the environment; raw2features stores no credentials).

v0.0.1 (Feb 2026): CONCH + UNI2-h backbones are on HF now (h0mini/phikonv2/virchow2
coming). The family is backbone-parameterised (``spec.source`` = the backbone key), so a
new backbone is a one-line registry entry once its checkpoint lands.

Reference:  https://huggingface.co/MahmoodLab/SEAL
Paper:      Hemker, Song et al., "Towards Spatial Transcriptomics-driven Pathology
            Foundation Models", arXiv:2602.14177
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING

from raw2features.core.plugins import register

from ._hub import download_pinned_hf_file, verify_sha256
from .base import Embedder
from .fingerprint import SEAL_CONSTRUCTOR_CONTRACT

if TYPE_CHECKING:  # pragma: no cover
    import torch

_SEAL_REPO = "MahmoodLab/SEAL"


@register("embedders", "seal")
class SealEmbedder(Embedder):
    """A SEAL-adapted backbone (frozen base + LoRA); ``model(x) -> (B, embedding_dim)``.

    ``spec.source`` is the SEAL backbone key (``"conch"`` / ``"univ2"``); the LoRA
    checkpoint is ``seal_<backbone>_vision.pth`` in ``MahmoodLab/SEAL``.
    """

    def load(
        self,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        compile: bool = False,
    ) -> SealEmbedder:
        import torch

        try:
            from seal.models.load_model import ModelMixin
            from seal.utils.constants import EMB_DICT
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                "SEAL needs the optional `seal` stack:\n"
                '  pip install "raw2features[seal]"\n'
                "  pip install --no-deps "
                '"git+https://github.com/cjmielke/SEAL.git'
                '@5334490645e8410e7d8ef6978cebc4fd98f9cf9a"\n'
                "  + the backbone's own package (e.g. git+https://github.com/"
                "Mahmoodlab/CONCH.git@141cc09c7d4ff33d8eda562bd75169b457f71a62)"
            ) from exc

        backbone = self.spec.source  # "conch" | "univ2"
        # Freeze every output-affecting constructor input instead of asking SEAL's
        # update_config(), which permits CWD-local config and per-user overrides.
        conf = deepcopy(SEAL_CONSTRUCTOR_CONTRACT)
        conf["encoder"] = backbone
        conf["out_dim"] = int(EMB_DICT[backbone])

        # Resolve and verify the adapter before either deserialising it or paying to
        # construct its large frozen base. The base itself remains upstream-managed
        # and unpinned, which is why SEAL is marked experimental in v0.2.0.
        ckpt_path = download_pinned_hf_file(
            f"hf-hub:{_SEAL_REPO}",
            self.spec.weights_filename or f"seal_{backbone}_vision.pth",
            self.spec.weights_revision,
        )
        verify_sha256(ckpt_path, self.spec.weights_sha256, what=self.spec.name)

        mixin = ModelMixin()
        mixin.conf = conf
        mixin.emb_dict = EMB_DICT
        model, _transform, _precision = mixin.get_img_model(
            backbone,
            partial_blocks=conf["partial_blocks"],
            use_adapter=conf["use_adapter"],
            adapter_bottleneck=conf["adapter_bottleneck"],
            hf_token=os.environ.get("HF_TOKEN"),
        )

        # Merge the SEAL LoRA delta onto the frozen base (strict=False: PEFT key names +
        # unused projection/decoder keys; module./virchow2 prefixes per SEAL's code).
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:  # older torch / narrow test doubles
            ckpt = torch.load(ckpt_path, map_location="cpu")
        except Exception:  # verified legacy checkpoints may contain non-tensor metadata
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        if not isinstance(sd, Mapping) or not all(isinstance(k, str) for k in sd):
            raise ValueError(
                f"{self.spec.name}: SEAL adapter is not a string-keyed state dict"
            )
        sd = dict(sd)
        if any(k.startswith("module.") for k in sd):
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
        if any("encoder.base_model.model.encoder." in k for k in sd):
            sd = {
                k.replace(
                    "encoder.base_model.model.encoder.",
                    "encoder.base_model.model.model.",
                ): v
                for k, v in sd.items()
            }
        # strict=False is necessary for unused SEAL projection/decoder keys, but it
        # must not turn a wrong backbone or broken prefix rewrite into a silent no-op.
        lora_a = {key for key in sd if "lora_A" in key}
        lora_b = {key for key in sd if "lora_B" in key}
        if not lora_a or not lora_b:
            raise ValueError(
                f"{self.spec.name}: checkpoint contains no complete LoRA A/B adapter"
            )
        model_keys = set(model.state_dict())
        checkpoint_lora = lora_a | lora_b
        model_lora = {key for key in model_keys if "lora_A" in key or "lora_B" in key}
        unmatched_lora = sorted(checkpoint_lora - model_keys)
        if unmatched_lora:
            sample = ", ".join(unmatched_lora[:3])
            raise ValueError(
                f"{self.spec.name}: {len(unmatched_lora)} LoRA adapter keys do not "
                f"match the constructed {backbone} base (for example: {sample})"
            )
        missing_lora = sorted(model_lora - checkpoint_lora)
        if missing_lora:
            sample = ", ".join(missing_lora[:3])
            raise ValueError(
                f"{self.spec.name}: checkpoint is missing {len(missing_lora)} LoRA "
                f"adapter keys required by the constructed {backbone} base "
                f"(for example: {sample})"
            )
        if not (set(sd) & model_keys):
            raise ValueError(
                f"{self.spec.name}: checkpoint has no keys in common with the model"
            )
        model.load_state_dict(sd, strict=False)
        model.eval().to(device)
        self._model = model
        self._device = device
        self._dtype = dtype or torch.float32
        self._maybe_compile(compile)
        return self

    def embed_batch(self, batch: torch.Tensor) -> torch.Tensor:
        from .base import _forward_ctx

        with _forward_ctx(self._device, self._dtype):
            out = self._model(batch.to(self._device))
        return out.float().cpu()
