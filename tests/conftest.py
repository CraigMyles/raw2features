"""Shared test fixtures, incl. a synthetic OME-NGFF v0.4 store.

The synthetic store mimics a ``bioformats2raw``-style 5D image (t, c, z, y, x)
with RGB in the channel axis and a /2 pyramid, so reader/patcher/sink tests run
fully offline without the multi-GB real slides.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from raw2features.embedders.base import Embedder, ModelSpec


class MockEmbedder(Embedder):
    """CPU mock embedder for runner/sink/thumbnail tests (no weights, no GPU).

    ``bias`` shifts every output element, so different tests can assert distinct
    values without each redefining the embedder.
    """

    def __init__(
        self, dim: int = 8, input_size: int = 64, bias: float = 0.0, name: str = "mock"
    ) -> None:
        super().__init__(
            ModelSpec(
                name=name,
                family="mock",
                source="mock://",
                embedding_dim=dim,
                input_size=input_size,
                pooling="cls",
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                transform_source_url="https://example.org/mock",
                license="MIT",
                gated=False,
            )
        )
        self._bias = bias

    def load(self, device="cpu", dtype=None, compile=False):
        self._device = "cpu"
        return self

    def embed_batch(self, batch):
        v = batch.float().mean(dim=(1, 2, 3))
        return v.unsqueeze(1).repeat(1, self.spec.embedding_dim) + self._bias


def build_ngff_v04(
    store_path: str,
    *,
    mpp0: float = 0.5,
    sizes: tuple[tuple[int, int], ...] = ((200, 300), (100, 150), (50, 75)),
) -> str:
    """Write a 5D OME-NGFF v0.4 store. ``sizes`` are (H, W) per level."""
    g = zarr.open_group(str(store_path), mode="w", zarr_format=2)
    for i, (h, w) in enumerate(sizes):
        a = g.create_array(
            str(i),
            shape=(1, 3, 1, h, w),
            chunks=(1, 1, 1, min(64, h), min(64, w)),
            dtype="uint8",
        )
        ramp = np.linspace(0, 255, h, dtype="uint8")[:, None] * np.ones((1, w), "uint8")
        a[0, 0, 0] = ramp  # R: vertical ramp
        a[0, 1, 0] = 255 - ramp  # G: inverse ramp
        a[0, 2, 0] = 128  # B: constant
    axes = [
        {"name": "t", "type": "time", "unit": "second"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]
    datasets = [
        {
            "path": str(i),
            "coordinateTransformations": [
                {
                    "type": "scale",
                    "scale": [1.0, 1.0, 1.0, mpp0 * (2**i), mpp0 * (2**i)],
                }
            ],
        }
        for i in range(len(sizes))
    ]
    g.attrs["multiscales"] = [
        {"version": "0.4", "name": "synthetic", "axes": axes, "datasets": datasets}
    ]
    return str(store_path)


@pytest.fixture
def synthetic_ngff(tmp_path) -> str:
    """A small synthetic OME-NGFF v0.4 store; level-0 MPP 0.5, sizes 200x300 down."""
    return build_ngff_v04(str(tmp_path / "synthetic.zarr"))


def build_multiplex_ngff(
    store_path: str,
    *,
    mpp0: float = 0.5,
    channels: tuple[str, ...] = ("DAPI", "CD3", "CD8", "CD20", "FOXP3"),
    sizes: tuple[tuple[int, int], ...] = ((128, 160), (64, 80), (32, 40)),
) -> str:
    """Write a multi-channel (c,y,x) OME-NGFF store with omero marker names (uint16).

    Models the CODEX/multiplex layout the channel-aware reader + nuclear segmenter +
    KRONOS path consume: N marker channels (incl. a DAPI channel) + a /2 pyramid.
    """
    g = zarr.open_group(str(store_path), mode="w", zarr_format=2)
    nch = len(channels)
    for i, (h, w) in enumerate(sizes):
        a = g.create_array(
            str(i), shape=(nch, h, w),
            chunks=(1, min(64, h), min(64, w)), dtype="uint16",
        )
        base = (np.arange(h * w).reshape(h, w) % 500).astype("uint16")
        for c in range(nch):
            a[c] = base + np.uint16((c + 1) * 1000)
    axes = [
        {"name": "c", "type": "channel"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]
    datasets = [
        {
            "path": str(i),
            "coordinateTransformations": [
                {"type": "scale", "scale": [1.0, mpp0 * (2**i), mpp0 * (2**i)]}
            ],
        }
        for i in range(len(sizes))
    ]
    g.attrs["multiscales"] = [
        {"version": "0.4", "name": "multiplex", "axes": axes, "datasets": datasets}
    ]
    g.attrs["omero"] = {"channels": [{"label": c, "active": True} for c in channels]}
    return str(store_path)


@pytest.fixture
def synthetic_multiplex_ngff(tmp_path) -> str:
    """A small synthetic multiplex OME-NGFF store (5 marker channels incl. DAPI)."""
    return build_multiplex_ngff(str(tmp_path / "multiplex.zarr"))
