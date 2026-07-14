"""Generic timm / HuggingFace-hub backbone driver (UNI, UNI2-h, DINOv2, ...)."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from raw2features.core.plugins import register

from ._hub import download_pinned_url, pin_source, verify_sha256
from .base import Embedder

if TYPE_CHECKING:  # pragma: no cover
    import torch


def _resolve_callables(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Resolve dotted-string kwargs to objects.

    e.g. 'timm.layers.SwiGLUPacked', 'torch.nn.SiLU', or our own
    'raw2features.embedders.convstem.ConvStem' (a custom embed_layer).
    """
    out: dict[str, Any] = {}
    for key, val in kwargs.items():
        if isinstance(val, str) and val.startswith(
            ("timm.", "torch.", "raw2features.")
        ):
            mod, attr = val.rsplit(".", 1)
            out[key] = getattr(importlib.import_module(mod), attr)
        else:
            out[key] = val
    return out


@register("embedders", "timm")
class TimmEmbedder(Embedder):
    """Loads any timm model (incl. ``hf-hub:`` ids) with ``num_classes=0``."""

    def load(
        self,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        compile: bool = False,
    ) -> TimmEmbedder:
        import timm
        import torch

        kwargs = _resolve_callables(dict(self.spec.timm_kwargs))
        if self.spec.checkpoint:
            # Weights are a bare HF .pth/.bin (no timm pretrained_cfg): build the
            # arch then load the checkpoint ourselves.
            model = timm.create_model(self.spec.source, pretrained=False, **kwargs)
            self._load_checkpoint(model, self.spec.checkpoint)
        else:
            # Pin the download to the recorded immutable HF commit (hf-hub:repo@rev).
            model = timm.create_model(
                pin_source(self.spec.source, self.spec.weights_revision),
                pretrained=True,
                **kwargs,
            )
        model.eval().to(device)
        self._model = model
        self._device = device
        self._dtype = dtype or torch.float32
        if self.spec.transform_source == "pretrained_cfg":
            self._sync_transform_from_pretrained_cfg(model)
        self._maybe_compile(compile)
        return self

    def _load_checkpoint(self, model, ckpt: dict) -> None:
        """Load a bare HF ``.pth``/``.bin`` state_dict into ``model``.

        For timm-arch models published as a raw checkpoint (no pretrained_cfg).
        ``ckpt`` keys: ``repo``, ``filename`` (required); ``state_dict_key`` (pull a
        sub-dict, e.g. "teacher"); ``strip_prefixes`` (list, e.g. ["backbone."]);
        ``strict`` (default True). Prefers a ``weights_only`` load, falling back for
        checkpoints that pickle non-tensor metadata.
        """
        import torch

        # Pin + verify the checkpoint bytes before deserialising, so the
        # weights_only=False fallback only ever unpickles vetted, pinned bytes. The
        # checkpoint is pinned either by HF repo+revision, or by a stable ``url`` (e.g.
        # a GitHub release asset) for weights with no HuggingFace home.
        if ckpt.get("url"):
            path = download_pinned_url(
                ckpt["url"], self.spec.weights_sha256, what=self.spec.name
            )
        else:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                ckpt["repo"], ckpt["filename"], revision=self.spec.weights_revision
            )
            verify_sha256(path, self.spec.weights_sha256, what=self.spec.name)
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:  # noqa: BLE001 - some .pth pickle non-tensor objects
            state = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("state_dict_key") is not None:
            state = state[ckpt["state_dict_key"]]
        prefixes = tuple(ckpt.get("strip_prefixes") or ())
        if prefixes:
            cleaned = {}
            for key, val in state.items():
                for pre in prefixes:
                    if key.startswith(pre):
                        key = key[len(pre):]
                        break
                cleaned[key] = val
            state = cleaned
        model.load_state_dict(state, strict=ckpt.get("strict", True))

    def _sync_transform_from_pretrained_cfg(self, model) -> None:
        """Adopt mean/std/interpolation from the model's ``pretrained_cfg``.

        The faithful path for models (e.g. Virchow2) whose card builds the
        transform via ``resolve_data_config(model.pretrained_cfg)`` rather than
        documenting fixed numbers. The registry still carries the card-documented
        values as an offline (no-weights) fallback; here we overwrite the frozen
        spec with the loaded model's authoritative config and assert the two agree
        within fp tolerance, so a card/weights drift is caught loudly rather than
        silently embedding under the wrong normalisation.
        """
        cfg = getattr(model, "pretrained_cfg", None) or {}
        mean = cfg.get("mean")
        std = cfg.get("std")
        interp = cfg.get("interpolation")
        if mean is None or std is None:
            raise ValueError(
                f"{self.spec.name}: transform_source=pretrained_cfg but the loaded "
                f"model has no mean/std in pretrained_cfg ({sorted(cfg)})."
            )
        mean, std = tuple(float(x) for x in mean), tuple(float(x) for x in std)
        for got, want, field in (
            (mean, self.spec.mean, "mean"),
            (std, self.spec.std, "std"),
        ):
            if any(abs(a - b) > 1e-6 for a, b in zip(got, want, strict=True)):
                raise ValueError(
                    f"{self.spec.name}: pretrained_cfg {field}={got} disagrees with "
                    f"the card-documented registry {field}={want}; update the "
                    f"registry (do not guess)."
                )
        if interp and str(interp).lower() != str(self.spec.interpolation).lower():
            raise ValueError(
                f"{self.spec.name}: pretrained_cfg interpolation={interp!r} disagrees "
                f"with registry interpolation={self.spec.interpolation!r}; update "
                "the registry (do not guess)."
            )
        # Frozen dataclass: mutate via object.__setattr__ (same idiom as tests).
        object.__setattr__(self.spec, "mean", mean)
        object.__setattr__(self.spec, "std", std)
        if interp:
            object.__setattr__(self.spec, "interpolation", interp)

    def embed_batch(self, batch: torch.Tensor) -> torch.Tensor:
        from .base import _forward_ctx

        with _forward_ctx(self._device, self._dtype):
            out = self._model(batch.to(self._device))
        return self._pool(out).float().cpu()

    def _pool(self, out: torch.Tensor) -> torch.Tensor:
        """Reduce the model output to one vector per item, per ``spec.pooling``.

        ``cls_concat_meanpatch`` expects the full token sequence
        ``[B, 1+reg+P, D]`` and returns ``cat([cls, patch_tokens.mean(1)])`` of
        width ``2*D`` -- the Virchow2 recipe, with the ``reg`` register tokens
        between the class token and the patch tokens skipped. Every other pooling
        keeps the prior behaviour: a 2-D output passes through, and a stray token
        tensor falls back to the class token.
        """
        import torch

        if self.spec.pooling == "cls_concat_meanpatch":
            if out.ndim != 3:
                raise ValueError(
                    f"{self.spec.name}: cls_concat_meanpatch needs a token tensor "
                    f"[B, tokens, D], got shape {tuple(out.shape)} -- the model must "
                    f"return tokens (do not pass num_classes=0 / global-pool it)."
                )
            cls = out[:, 0]
            patch_tokens = out[:, 1 + self.spec.reg_tokens :]
            return torch.cat([cls, patch_tokens.mean(dim=1)], dim=-1)
        if out.ndim > 2:  # safety: a token tensor -> take CLS
            return out[:, 0]
        return out
