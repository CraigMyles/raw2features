"""KRONOSv1 multiplex (spatial-proteomics) encoder - optional, needs the `kronos` pkg.

KRONOS embeds MULTIPLEX imaging (CODEX / Phenocycler), not H&E: a patch is a stack of
single-marker channels ``[M, H, W]``, each standardised by its marker's mean/std and
tagged with a marker id (a sinusoidal marker-identity embedding). We resolve the slide's
marker panel (the reader's ``channel_names``) to KRONOS marker ids + per-marker stats
from the model's published ``marker_metadata.csv`` (175 markers; common synonyms and
compound CD-names are resolved - see ``_resolve_marker``); markers outside it (blanks /
empties / genuinely-unknown) are dropped, and the unknowns surfaced - panel-agnostic.

This is the first ``multiplex`` embedder; brightfield (H&E) models are unaffected. Loads
the public KRONOSv1 weights (``hf_hub:MahmoodLab/kronos``, gated). The ``kronos`` import
is deferred to :meth:`load`, so the dependency stays optional (the ``[kronos]`` extra).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

from raw2features.core.plugins import register

from ._hub import download_pinned_hf_file, verify_sha256
from .base import Embedder

if TYPE_CHECKING:  # pragma: no cover
    import torch


def _to_unit_interval(arr: np.ndarray, dtype) -> np.ndarray:
    """Scale a multiplex stack to ``[0, 1]`` by its source dtype.

    KRONOS's per-marker mean/std are calibrated on ``[0, 1]`` intensities, so the input
    must be mapped there first. CODEX/Phenocycler is uint16 (``/65535``), but uint8
    multiplex (``/255``) and already-normalised float sources occur too - scaling by the
    dtype's range rather than a hard-coded ``65535`` keeps those correct instead of
    silently squashing (uint8) or overflowing the assumption.
    """
    if np.issubdtype(dtype, np.unsignedinteger):
        return arr / float(np.iinfo(dtype).max)
    if np.issubdtype(dtype, np.floating):
        return arr  # assume already in [0, 1] (the marker stats' scale)
    raise NotImplementedError(
        f"unsupported multiplex dtype {dtype!r}; expected unsigned int or float [0,1]"
    )


def _norm_marker(name: str) -> str | None:
    """Normalise a marker name for matching KRONOS's vocab; return None to drop it."""
    s = name.strip().lower()
    if s.startswith(("hoechst", "hochst", "dapi")):
        return "DAPI"  # nuclear stain -> KRONOS's DAPI marker
    if s.startswith(("blank", "empty")):
        return None  # background / empty cycles are not markers
    return s.upper().replace("_", "").replace("-", "").replace(" ", "")


# Well-established marker synonyms (normalised) so a panel that names a marker
# differently than KRONOS's vocabulary still matches it. Conservative: each maps a
# common antibody/clone name to KRONOS's canonical name, verified against the published
# KRONOS marker vocabulary - we never map to a different molecular identity.
_MARKER_ALIASES = {
    "PANCYTOKERATIN": "CYTOKERATIN",  # pan-cytokeratin cocktail == cytokeratin
    "PANCK": "CYTOKERATIN",
    "CK": "CYTOKERATIN",
    "GRANZYMEB": "GZMB",  # granzyme B == gene symbol GZMB
    "GRANZYMEA": "GZMA",
    "CLA": "CD162",  # cutaneous lymphocyte antigen == CD162 (PSGL-1)
    "PSGL1": "CD162",
}

_CD_TOKEN = re.compile(r"CD\d+[A-Z]*")


def _resolve_marker(normed: str, vocab: dict) -> str | None:
    """Map a normalised marker name to a KRONOS vocab key, or None if unsupported.

    Tries, in order: exact match; a curated synonym (:data:`_MARKER_ALIASES`); and a
    CD-number token embedded in a compound name (e.g. ``CLACD162`` -> ``CD162``). Only
    well-established equivalences, so KRONOS is never fed a marker under a wrong
    identity; a name that resolves to nothing is genuinely outside KRONOS's vocabulary.
    """
    if not normed:  # blank/empty cycle (``_norm_marker`` returned None)
        return None
    if normed in vocab:
        return normed
    alias = _MARKER_ALIASES.get(normed)
    if alias and alias in vocab:
        return alias
    for tok in _CD_TOKEN.findall(normed):
        if tok in vocab:
            return tok
    return None


def _patch_kronos_attention_to_sdpa() -> None:
    """Replace KRONOS's naive attention fallback with torch SDPA (flash attention).

    KRONOS (DINOv2-derived) materialises the full N×N attention matrix when xFormers is
    absent - enormous for multiplex (M markers → ~8k tokens/patch, ~26 GB at batch 8).
    ``torch.nn.functional.scaled_dot_product_attention`` is numerically identical (it
    applies the ``1/sqrt(head_dim)`` scale internally, == the model's ``self.scale``)
    but dispatches to fused flash / memory-efficient kernels: far less memory + faster,
    with no extra dependency (no CUDA-pinned xFormers build).
    Benchmarked on an A100 (M=41): 20->41 patches/s, ~26 GB -> ~1.3 GB at batch 8.
    Idempotent; only the non-return_attn path is swapped.
    """
    import torch.nn.functional as F
    from kronos.attention import Attention

    if getattr(Attention, "_r2f_sdpa", False):
        return

    def forward(self, x, return_attn=False):
        if return_attn:  # SDPA doesn't expose the matrix; keep the original path
            return Attention._r2f_orig(self, x, return_attn=True)
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(out))

    Attention._r2f_orig = Attention.forward
    Attention.forward = forward
    Attention._r2f_sdpa = True


@register("embedders", "kronos")
class KronosEmbedder(Embedder):
    """KRONOSv1 (ViT-S/16, 384-d) multiplex patch encoder.

    Per-slide usage: :meth:`set_panel` resolves the marker panel once, then
    :meth:`transform_batch` / :meth:`embed_batch` work on multi-channel patches.
    """

    def load(
        self,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        compile: bool = False,
    ) -> KronosEmbedder:
        import csv

        try:
            from kronos import create_model_from_pretrained
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                "KRONOS needs the optional `kronos` package. Install the stack, then "
                "the (non-PyPI, gated) package:\n"
                '  pip install "raw2features[kronos]"\n'
                "  pip install git+https://github.com/mahmoodlab/KRONOS.git@"
                "48979362386c8440c934954be3d88ccfa74d6f36"
            ) from exc

        # Use torch SDPA (flash attention) instead of KRONOS's naive fallback - far
        # less memory + faster, no xFormers dependency. Opt out with sdpa: false.
        if self.spec.timm_kwargs.get("sdpa", True):
            _patch_kronos_attention_to_sdpa()

        cache = self.spec.timm_kwargs.get("cache_dir")
        checkpoint = download_pinned_hf_file(
            self.spec.source,
            "kronos_vits16_model.pt",
            self.spec.weights_revision,
            cache_dir=cache,
        )
        verify_sha256(checkpoint, self.spec.weights_sha256, what=self.spec.name)
        model, precision, _dim = create_model_from_pretrained(
            checkpoint_path=checkpoint,
            cache_dir=cache,
            cfg={"model_type": "vits16", "token_overlap": False},
        )
        model.eval().to(device)
        self._model = model
        self._device = device
        self._dtype = precision  # KRONOS sets its own precision (fp32)

        # marker vocabulary: normalised name -> (marker_id, mean, std, canonical_name),
        # from the model's published marker_metadata.csv. The canonical KRONOS name is
        # kept so the per-slide mapping provenance can record the identity each channel
        # resolved to (not just our normalised key).
        meta_path = download_pinned_hf_file(
            self.spec.source,
            "marker_metadata.csv",
            self.spec.weights_revision,
            cache_dir=cache,
        )
        self._vocab: dict[str, tuple[int, float, float, str]] = {}
        with open(meta_path) as fh:
            for row in csv.DictReader(fh):
                key = _norm_marker(row["marker_name"])
                if key and key not in self._vocab:
                    self._vocab[key] = (
                        int(row["marker_id"]),
                        float(row["marker_mean"]),
                        float(row["marker_std"]),
                        row["marker_name"],
                    )
        self._panel: dict | None = None
        return self

    def set_panel(self, channel_names: list[str] | None) -> dict:
        """Resolve a slide's marker panel -> kept channel indices + ids + stats.

        Each channel name is matched to KRONOS's vocabulary (exact, then synonyms +
        compound CD-names via :func:`_resolve_marker`). Matches are kept with their ids
        + per-marker stats; blank/empty cycles drop silently; a *named* marker that
        doesn't resolve is dropped, warned about, and recorded under ``unmatched`` (it
        may be a synonym to add, or genuinely outside KRONOS's 175-marker vocabulary).
        Returns a summary (``n_markers``, ``kept``, ``dropped``, ``unmatched``) for
        provenance; call once per slide before :meth:`transform_batch`.
        """
        idx, ids, means, stds, kept, dropped, unmatched = [], [], [], [], [], [], []
        mapping = []
        for i, name in enumerate(channel_names or []):
            key = _norm_marker(name)
            vkey = _resolve_marker(key, self._vocab) if key else None
            if vkey is not None:
                mid, mean, std, canonical = self._vocab[vkey]
                idx.append(i)
                ids.append(mid)
                means.append(mean)
                stds.append(std)
                kept.append(name)
                # Explicit, retrievable record of how this source channel was identified
                # to KRONOS: original name + position -> KRONOS marker name + id.
                mapping.append(
                    {
                        "channel": name,
                        "channel_index": i,
                        "kronos_marker": canonical,
                        "marker_id": int(mid),
                    }
                )
            else:
                dropped.append(name)
                if key:  # a named marker (not a blank/empty cycle) missed
                    unmatched.append(name)
        if not idx:
            raise ValueError("no channels matched the KRONOS marker vocabulary")
        if unmatched:
            # KRONOS is panel-agnostic, so unknown markers are dropped - but a *named*
            # marker we couldn't match (vs a blank/empty cycle) is surfaced: it may be a
            # KRONOS marker under a synonym to add, or be
            # genuinely outside its vocabulary. Report coverage + names (also in the
            # store's panel provenance) so the user can decide.
            import warnings

            n_named = len(kept) + len(unmatched)
            warnings.warn(
                f"KRONOS matched {len(kept)}/{n_named} named markers; "
                f"{len(unmatched)} not in its 175-marker vocabulary (dropped): "
                f"{unmatched}. Check whether any are supported under a different name.",
                stacklevel=2,
            )
        self._panel = {
            "idx": np.asarray(idx),
            "ids": ids,
            "mean": np.asarray(means, dtype=np.float32),
            "std": np.asarray(stds, dtype=np.float32),
            "kept": kept,
            "dropped": dropped,
        }
        # `unmatched` = named markers KRONOS doesn't recognise (a subset of `dropped`,
        # excluding the blank/empty cycles) - surfaced separately for the provenance.
        # `mapping` is the retrievable channel -> KRONOS-marker record; `vocabulary`
        # pins where the canonical names/ids came from (the model's own metadata file).
        return {
            "n_markers": len(kept),
            "kept": kept,
            "dropped": dropped,
            "unmatched": unmatched,
            "mapping": mapping,
            "vocabulary": f"{self.spec.source} marker_metadata.csv"
            + (f"@{self.spec.weights_revision}" if self.spec.weights_revision else ""),
        }

    def transform_batch(
        self, patches_hwc: list[np.ndarray], device: str
    ) -> torch.Tensor:
        """[H,W,M] patches -> [B,M',H,W] float (kept markers, per-marker normalised)."""
        import torch

        if self._panel is None:
            raise RuntimeError("call set_panel() before embedding multiplex patches")
        idx = self._panel["idx"]
        src_dtype = patches_hwc[0].dtype
        stack = np.stack([p[:, :, idx].astype(np.float32) for p in patches_hwc])
        stack = np.transpose(stack, (0, 3, 1, 2))  # [B, M', H, W]
        stack = _to_unit_interval(stack, src_dtype)  # -> [0, 1] per the source dtype
        t = torch.from_numpy(np.ascontiguousarray(stack)).to(device)
        mean = torch.from_numpy(self._panel["mean"]).to(device)[None, :, None, None]
        std = torch.from_numpy(self._panel["std"]).to(device)[None, :, None, None]
        return (t - mean) / std

    def embed_batch(self, batch: torch.Tensor) -> torch.Tensor:
        import torch

        marker_ids = [
            torch.tensor(self._panel["ids"], device=self._device)
            for _ in range(batch.shape[0])
        ]
        with torch.no_grad():
            patch_emb, _marker_emb, _token_emb = self._model(
                batch.to(self._device), marker_ids=marker_ids
            )
        return patch_emb.float().cpu()
