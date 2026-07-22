#!/usr/bin/env python
"""Convert a multiplex multi-channel TIFF to a multi-channel OME-Zarr (NGFF).

Marker names (one per channel) are written as ``omero.channels[].label`` so the
raw2features channel-aware reader can surface them to multiplex strategies and native
multiplex models. A spatial pyramid is built so low-resolution normalization and
nuclear-channel masking are efficient.

Usage::

    python scripts/codex_to_omezarr.py INPUT.tif OUTPUT.ome.zarr \\
        --markers markers.json [--mpp 0.5]

``--markers`` is a JSON file: either a bare list of channel names, or an object with a
``raw_markers`` / ``channels`` / ``markers`` key. The channel count must match the TIFF.

Needs the OME-Zarr reader stack plus TIFF codecs:
``pip install "raw2features[zarr]" tifffile imagecodecs``.
"""

from __future__ import annotations

import argparse
import json


def load_markers(path: str) -> list[str]:
    data = json.load(open(path))
    if isinstance(data, list):
        return data
    for key in ("raw_markers", "channels", "markers"):
        if key in data:
            return data[key]
    raise ValueError(
        f"{path}: expected a list, or an object with a raw_markers/channels/markers key"
    )


def convert(
    inp: str, out: str, markers: list[str], mpp: float, scale_factors: list[int]
) -> None:
    import ngff_zarr as nz
    import tifffile
    import zarr

    data = tifffile.imread(inp)
    if data.ndim != 3:
        raise ValueError(f"expected a (C, Y, X) TIFF, got shape {data.shape}")
    if len(markers) != data.shape[0]:
        raise ValueError(f"{len(markers)} markers != {data.shape[0]} channels")
    img = nz.to_ngff_image(
        data, dims=["c", "y", "x"], scale={"c": 1.0, "y": mpp, "x": mpp}, name="codex"
    )
    multiscales = nz.to_multiscales(img, scale_factors=scale_factors)
    nz.to_ngff_zarr(out, multiscales, overwrite=True)
    zarr.open_group(out, mode="a").attrs["omero"] = {
        "channels": [{"label": m, "active": True} for m in markers]
    }
    print(f"wrote {out}: {data.shape[0]} channels @ {mpp} um/px")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="multi-channel CODEX/multiplex TIFF (C,Y,X)")
    ap.add_argument("output", help="output OME-Zarr path (*.ome.zarr)")
    ap.add_argument("--markers", required=True, help="JSON of per-channel marker names")
    ap.add_argument("--mpp", type=float, default=0.5, help="microns/pixel (def 0.5)")
    ap.add_argument("--scale-factors", type=int, nargs="*", default=[2, 4, 8, 16])
    args = ap.parse_args()
    markers = load_markers(args.markers)
    convert(args.input, args.output, markers, args.mpp, args.scale_factors)


if __name__ == "__main__":
    main()
