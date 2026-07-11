"""open_clip image towers (CLIP-style vision-language pathology models) - optional.

Models like QuiltNet and BiomedCLIP are CLIP image-text models; we use only the image
tower, whose ``encode_image`` returns the projected image embedding (the standard
PLIP/QuiltNet downstream patch feature). They load through the upstream ``open_clip``
package rather than ``timm``/``transformers``, so they get their own family with the
dependency **deferred** to :meth:`load` - the entry-point loader skips this family
cleanly when ``open_clip`` is absent.

The embedding is returned **un-normalised** (raw ``encode_image``): open_clip L2-
normalises only inside its contrastive loss / cosine retrieval, so the faithful feature
for downstream linear probes / MIL is the pre-normalisation projection.

Install::

    pip install "raw2features[open_clip]"
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from raw2features.core.plugins import register

from ._hub import (
    download_pinned_hf_snapshot,
    pinned_model_cache_dir,
    verify_sha256,
)
from .base import Embedder

if TYPE_CHECKING:  # pragma: no cover
    import torch


_BIOMEDCLIP_SOURCE = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
_BIOMEDCLIP_TEXT_SOURCE = "hf-hub:microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
_BIOMEDCLIP_TEXT_REVISION = "d673b8835373c6fa116d6d8006b33d48734e305d"


def _make_biomedclip_config_local(snapshot: Path) -> None:
    """Pin BiomedCLIP's nested PubMedBERT config and rewrite our private config.

    Loading BiomedCLIP's top-level weights from a pinned local directory is not
    sufficient: its ``open_clip_config.json`` names a second mutable HF repository
    for the text tower.  OpenCLIP constructs that tower even when raw2features only
    uses ``encode_image``.  Download the small files needed for construction at an
    immutable commit, then point the app-owned config at that absolute local path.
    """
    nested = Path(
        pinned_model_cache_dir(_BIOMEDCLIP_TEXT_SOURCE, _BIOMEDCLIP_TEXT_REVISION)
    )
    download_pinned_hf_snapshot(
        _BIOMEDCLIP_TEXT_SOURCE,
        _BIOMEDCLIP_TEXT_REVISION,
        allow_patterns=("config.json", "tokenizer_config.json", "vocab.txt"),
        local_dir=str(nested),
    )

    config_path = snapshot / "open_clip_config.json"
    with config_path.open(encoding="utf-8") as fh:
        config = json.load(fh)
    try:
        text_cfg = config["model_cfg"]["text_cfg"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "BiomedCLIP open_clip_config.json has no model_cfg.text_cfg mapping"
        ) from exc
    if not isinstance(text_cfg, dict):
        raise ValueError(
            "BiomedCLIP open_clip_config.json model_cfg.text_cfg is not a mapping"
        )
    nested_path = str(nested.resolve())
    text_cfg["hf_model_name"] = nested_path
    text_cfg["hf_tokenizer_name"] = nested_path

    # snapshot is deliberately raw2features-owned, so it is safe to rewrite. Use
    # replace-on-close so another process can never observe half-written JSON.
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=snapshot, delete=False
        ) as tmp:
            json.dump(config, tmp, indent=2)
            tmp.write("\n")
            tmp_path = tmp.name
        os.replace(tmp_path, config_path)
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@register("embedders", "open_clip")
class OpenClipEmbedder(Embedder):
    """An open_clip image tower; ``encode_image(x)`` -> ``(B, embedding_dim)``.

    ``spec.source`` is the open_clip model name, e.g. ``hf-hub:wisdomik/QuiltNet-B-32``
    (open_clip resolves the ``hf-hub:`` prefix to the Hugging Face checkpoint + its
    ``open_clip_config.json``).
    """

    def load(
        self,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        compile: bool = False,
    ) -> OpenClipEmbedder:
        import torch

        try:
            import open_clip
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                "open_clip models (quiltnet, biomedclip) need the optional "
                '`open_clip_torch` package: pip install "raw2features[open_clip]"'
            ) from exc

        # open_clip's hf-hub path does not accept a revision. Download the exact
        # config + checkpoint into a raw2features-owned, revision-specific directory
        # and verify the checkpoint before the upstream loader deserialises it.
        local_dir = pinned_model_cache_dir(self.spec.source, self.spec.weights_revision)
        download_pinned_hf_snapshot(
            self.spec.source,
            self.spec.weights_revision,
            allow_patterns=("open_clip_config.json", "open_clip_pytorch_model.bin"),
            local_dir=local_dir,
        )
        snapshot = Path(local_dir)
        verify_sha256(
            str(snapshot / "open_clip_pytorch_model.bin"),
            self.spec.weights_sha256,
            what=self.spec.name,
        )
        if self.spec.source == _BIOMEDCLIP_SOURCE:
            _make_biomedclip_config_local(snapshot)
        model, preprocess = open_clip.create_model_from_pretrained(
            f"local-dir:{snapshot}"
        )
        model.eval().to(device)
        self._model = model
        self._device = device
        self._dtype = dtype or torch.float32
        self._assert_transform_matches_preprocess(preprocess)
        self._maybe_compile(compile)
        return self

    def _assert_transform_matches_preprocess(self, preprocess) -> None:
        """Cross-check the registry norm against open_clip's own preprocess Normalize.

        open_clip carries the authoritative mean/std in its ``preprocess`` Compose; we
        source those into the registry and assert here so a drift fails loudly rather
        than silently embedding under the wrong normalisation (same guard as CONCH).
        """
        norm = next(
            (t for t in getattr(preprocess, "transforms", [])
             if t.__class__.__name__ == "Normalize"),
            None,
        )
        if norm is None:  # pragma: no cover - defensive
            return
        got_mean = tuple(float(x) for x in norm.mean)
        got_std = tuple(float(x) for x in norm.std)
        for got, want, field in (
            (got_mean, self.spec.mean, "mean"),
            (got_std, self.spec.std, "std"),
        ):
            if any(abs(a - b) > 1e-6 for a, b in zip(got, want, strict=True)):
                raise ValueError(
                    f"{self.spec.name}: open_clip preprocess {field}={got} disagrees "
                    f"with registry {field}={want}; fix registry.yaml (do not guess)."
                )

    def embed_batch(self, batch: torch.Tensor) -> torch.Tensor:
        from .base import _forward_ctx

        with _forward_ctx(self._device, self._dtype):
            out = self._model.encode_image(batch.to(self._device))
        return out.float().cpu()
