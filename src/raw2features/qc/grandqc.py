"""GrandQC tissue + artifact quality control (opt-in ``[grandqc]`` extra).

Reimplements GrandQC's two-stage inference (Weng et al., *Nat Commun* 2024,
10.1038/s41467-024-54769-y) as a generic UNet++/EfficientNet-B0 via
``segmentation_models_pytorch`` -- so only the optional deps and the pinned, CC-BY-NC-SA
weights (Zenodo, fetched on first use, never bundled) are external. The two roles share
one inference:

* ``--segmenter grandqc`` -- the **tissue** stage (MPP ~10) -> a tissue mask.
* ``--qc grandqc`` -- the **artifact** stage (7 classes @ MPP 1.5) -> a per-pixel class
  raster, projected to per-patch ``qc/grandqc/`` scores by
  :func:`raw2features.core.qc.patch_qc_scores`.

GPU-validated end-to-end: the smp build, the per-stage decoder (UNet++ tissue / UNet
artifact), the class counts, and the pinned Zenodo shas were confirmed on real H&E (the
artifact scores fall out as mostly clean tissue on a clean slide).
"""

from __future__ import annotations

import numpy as np

from raw2features.core.geometry import Point, Region, Size
from raw2features.core.mpp import nearest_level

# Pinned weights + shas (Zenodo: 14507273 tissue / 14041538 artifact), all four verified
# on download; verify_sha256 runs every load.
_Z = "https://zenodo.org/records"


def _art(name: str, mpp: float, sha: str | None = None) -> dict:
    return {"url": f"{_Z}/14041538/files/{name}", "sha256": sha, "mpp": mpp}


_TISSUE = {
    "url": f"{_Z}/14507273/files/Tissue_Detection_MPP10.pth",
    "sha256": "0e3577628a553774419a5b87a2947531988e07175b9e1a8048f53b6188f4d76e",
    "mpp": 10.0,
}
_ARTIFACT = {
    "1.0": _art(
        "GrandQC_MPP1.pth", 1.0,
        "cda7506dbb2e21be1410077d4f55632e7073fe068cf318df8fbc185a66806ecf",
    ),
    "1.5": _art(
        "GrandQC_MPP15.pth", 1.5,
        "6e90ec8548d4050734c30e30d5ca7fe67e84b59b1dd8ae338447dad90bae4ff1",
    ),
    "2.0": _art(
        "GrandQC_MPP2.pth", 2.0,
        "703d21cb3cac5b0797b145ca55428f21ab25bad18e3738ed8a3b6d42bbfcb61c",
    ),
}
# Artifact class index -> snake_case name (GrandQC CLASS_MAPPING; background last). The
# per-patch scores cover these labelled classes (raster value 0, if any, is padding).
QC_CLASSES = {
    1: "clean_tissue",
    2: "tissue_fold",
    3: "darkspot_foreign",
    4: "pen_marking",
    5: "airbubble_edge",
    6: "out_of_focus",
    7: "background",
}
_TILE = 512  # GrandQC model patch size
_TISSUE_CLASS = 0  # tissue-stage argmax index meaning "tissue" (1 == background)
_MEAN = np.asarray((0.485, 0.456, 0.406), np.float32)  # ImageNet (timm-efficientnet)
_STD = np.asarray((0.229, 0.224, 0.225), np.float32)


def _build(classes: int, arch: str = "unet"):
    """Build GrandQC's smp decoder: ``unetplusplus`` (tissue) or ``unet`` (artifact).

    The two stages use DIFFERENT decoders -- the architecture is detected from the
    checkpoint (:func:`_arch_of`), not assumed.
    """
    import segmentation_models_pytorch as smp

    cls = smp.UnetPlusPlus if arch == "unetplusplus" else smp.Unet
    return cls(
        encoder_name="timm-efficientnet-b0", encoder_weights=None,
        in_channels=3, classes=classes,
    )


def _arch_of(sd: dict) -> str:
    """smp decoder architecture from the state_dict key pattern."""
    return "unetplusplus" if any(".blocks.x_" in k for k in sd) else "unet"


def _load(spec: dict, device: str):
    """Download + load a GrandQC checkpoint; return ``(model, n_classes)``.

    The architecture (UNet++ tissue / UNet artifact) and class count are read from the
    checkpoint, never assumed. Some GrandQC checkpoints (the artifact stage) are
    full-model pickles (smp 0.3.1 / timm 0.4.12) that cannot load here; convert once to
    a clean ``<name>.statedict.pt`` (recipe in ``docs/SEGMENTATION.md``), preferred.
    """
    import os

    import torch

    from raw2features.embedders._hub import download_pinned_url

    name = spec["url"].rsplit("/", 1)[-1]
    path = download_pinned_url(spec["url"], spec["sha256"], what=f"grandqc[{name}]")
    converted = os.path.splitext(path)[0] + ".statedict.pt"
    try:
        src = converted if os.path.exists(converted) else path
        sd = torch.load(src, map_location="cpu", weights_only=False)
    except (ModuleNotFoundError, AttributeError) as e:
        raise RuntimeError(
            f"grandqc: {name} is a full-model pickle (smp 0.3.1 / timm 0.4.12) and "
            f"cannot load here: {e}. Convert it once to {os.path.basename(converted)} "
            "-- see docs/SEGMENTATION.md."
        ) from e
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    head = next((k for k in sd if k.endswith("segmentation_head.0.weight")), None)
    classes = int(sd[head].shape[0]) if head is not None else len(QC_CLASSES) + 1
    model = _build(classes, _arch_of(sd))
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval(), classes


def _read_at_mpp(reader, model_mpp: float):
    """Whole slide as RGB uint8 at ~``model_mpp``; returns ``(img, level0_per_px)``."""
    import cv2

    level = nearest_level(reader.mpp, reader.level_downsamples(), model_mpp)
    dim = reader.level_dimensions[level]
    img = np.asarray(
        reader.read_region(Region(level, Point(0, 0), Size(dim.width, dim.height)))
    )[..., :3]
    level_mpp = reader.mpp * reader.level_downsamples()[level]
    scale = level_mpp / model_mpp  # resize the level to exactly model_mpp
    if abs(scale - 1.0) > 1e-3:
        h, w = img.shape[:2]
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        img = cv2.resize(img, size, interpolation=interp)
    return img, model_mpp / reader.mpp  # level-0 px per raster px


def _segment(model, img: np.ndarray, device: str, batch: int = 8) -> np.ndarray:
    """Tile ``img`` into 512 px, run it; return the argmax class raster (H, W)."""
    import torch

    h, w = img.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    tiles, locs = [], []
    for y in range(0, h, _TILE):
        for x in range(0, w, _TILE):
            tile = img[y : y + _TILE, x : x + _TILE]
            pad = np.zeros((_TILE, _TILE, 3), np.uint8)
            pad[: tile.shape[0], : tile.shape[1]] = tile
            tiles.append(pad)
            locs.append((y, x))
    with torch.inference_mode():
        for s in range(0, len(tiles), batch):
            arr = np.stack(tiles[s : s + batch]).astype(np.float32) / 255.0
            arr = (arr - _MEAN) / _STD
            t = torch.from_numpy(arr.transpose(0, 3, 1, 2)).to(device)
            pred = model(t).argmax(1).to("cpu").numpy().astype(np.uint8)
            for j, (y, x) in enumerate(locs[s : s + batch]):
                p = pred[j][: min(_TILE, h - y), : min(_TILE, w - x)]
                out[y : y + p.shape[0], x : x + p.shape[1]] = p
    return out


class GrandQC:
    """Lazily-loaded GrandQC stages, reused across one slide's grids.

    ``device`` is the torch device for inference; ``artifact_mpp`` selects the artifact
    flavour (``"1.0"`` 10x / ``"1.5"`` 7x, the default / ``"2.0"`` 5x).
    """

    def __init__(
        self, device: str = "cpu", artifact_mpp: str = "1.5",
        stain_norm: str | None = None,
    ) -> None:
        self.device = device
        self.artifact_mpp = str(artifact_mpp)
        self.stain_norm = stain_norm  # None|"macenko": normalize before inference
        self._tissue = None
        self._artifact = None
        self._raster: tuple[np.ndarray, float] | None = None
        self._artifact_img: tuple[np.ndarray, float] | None = None  # raw (img, ds)

    def _maybe_normalize(self, img: np.ndarray) -> np.ndarray:
        """Stain-normalize ``img`` to canonical H&E when ``stain_norm`` is set.

        ``stain_norm`` is macenko | reinhard | vahadane (fitted from this image); see
        :func:`raw2features.core.stain.make_normalizer`.
        """
        if not self.stain_norm:
            return img
        from raw2features.core.stain import make_normalizer

        norm = make_normalizer(self.stain_norm, img)
        return norm(img) if norm is not None else img

    def _read(self, reader, mpp: float):
        """Read the model input at ``mpp``, optionally stain-normalized."""
        img, ds = _read_at_mpp(reader, mpp)
        return self._maybe_normalize(img), ds

    def tissue_mask(self, reader):
        """A :class:`~raw2features.segmenters.base.TissueMask` from the tissue stage."""
        from raw2features.segmenters.base import TissueMask

        if self._tissue is None:
            self._tissue, _ = _load(_TISSUE, self.device)
        img, ds = self._read(reader, _TISSUE["mpp"])
        raster = _segment(self._tissue, img, self.device)
        mask = (raster == _TISSUE_CLASS).astype(np.float32)
        return TissueMask(mask=mask, level=0, downsample=ds)  # level-0 px per mask px

    def artifact_image(self, reader) -> tuple[np.ndarray, float]:
        """The slide RGB (uint8) at the artifact MPP -- *before* stain normalization --
        plus its level-0-px-per-image-px.

        Lets callers mask scanner canvas before tallying class fractions (GrandQC reads
        black/white padding as artifact; the canvas is a raw-slide property, not the
        normalized model input). Shares the read with :meth:`artifact_raster` (cached).
        """
        mpp = _ARTIFACT[self.artifact_mpp]["mpp"]
        if self._artifact_img is None:
            self._artifact_img = _read_at_mpp(reader, mpp)
        return self._artifact_img

    def artifact_raster(self, reader) -> tuple[np.ndarray, float]:
        """The artifact raster + its level-0-per-px (computed once per slide)."""
        if self._raster is None:
            if self._artifact is None:
                self._artifact, _ = _load(_ARTIFACT[self.artifact_mpp], self.device)
            img, ds = self.artifact_image(reader)            # raw slide image
            raster = _segment(self._artifact, self._maybe_normalize(img), self.device)
            self._raster = (raster, ds)
        return self._raster

    def qc_for_grid(self, reader, coords, level0_patch):
        """``(scores (N, k) float16, classes)`` for a grid, ready for ``write_qc``."""
        from raw2features.core.qc import patch_qc_scores

        raster, ds = self.artifact_raster(reader)
        values = list(QC_CLASSES)
        scores = patch_qc_scores(
            coords, level0_patch, raster, values, raster_downsample=ds
        )
        return scores.astype(np.float16), [QC_CLASSES[v] for v in values]

    def provenance(self) -> dict:
        """The provenance block recorded with the qc layer."""
        return {
            "tool": "grandqc",
            "model_mpp": _ARTIFACT[self.artifact_mpp]["mpp"],
            "tissue_model_mpp": _TISSUE["mpp"],
            "doi": "10.1038/s41467-024-54769-y",
            "license": "CC-BY-NC-SA-4.0",
            "non_commercial": True,
            "derivation": "coverage_fraction",
            "stain_norm": self.stain_norm,
        }
