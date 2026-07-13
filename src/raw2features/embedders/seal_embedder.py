"""SEAL patch encoders (MahmoodLab) - spatial-transcriptomics LoRA fine-tunes; gated.

SEAL fine-tunes a frozen pathology backbone (CONCH, UNI2-h, …) with a LoRA adapter
aligned to spatial transcriptomics. The HF checkpoints are LoRA **deltas**, so loading
for inference still means: build the frozen base + merge the LoRA. We use SEAL's own
image-only loader (``ModelMixin.get_img_model``) - the gene/omics model is not loaded.
Output dim = the base encoder's (512 for CONCH, 1536 for UNI2-h).

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
from typing import TYPE_CHECKING

from raw2features.core.plugins import register

from .base import Embedder

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
        import argparse

        import torch

        try:
            from seal.models.encoder_factory import find_config_yaml
            from seal.models.load_model import ModelMixin
            from seal.utils.constants import EMB_DICT
            from seal.utils.exp_utils import update_config
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

        from huggingface_hub import hf_hub_download

        backbone = self.spec.source  # "conch" | "univ2"
        # The fork resolves conf/config.yaml + cache/organ_ids.json from its install
        # dir, so this works from any CWD (no repo checkout / chdir).
        conf = update_config(argparse.Namespace(config=find_config_yaml()))
        conf["encoder"] = backbone
        # Disable SEAL's image-reconstruction decoder - a training-only head we don't
        # use (our forward returns the encoder features). It also avoids loading its
        # weights, which are encoder-dim-specific: univ2's decoder is [512, 1536], so it
        # would size-mismatch a default [512, 512] build.
        conf["lambda_recon_img"] = 0.0

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
        ckpt_path = hf_hub_download(
            _SEAL_REPO,
            f"seal_{backbone}_vision.pth",
            revision=self.spec.weights_revision,
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
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
