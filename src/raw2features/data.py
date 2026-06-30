"""Tiny synthetic sample data, so the quickstart runs with no download / token / GPU.

``write_sample_slide`` writes a small multiscale OME-NGFF v0.4 RGB store with a
tissue-like blob on a bright background - enough for the tissue segmenter to find
patches and for the whole pipeline (read → segment → tile → embed) to run end to end
on a laptop CPU. The blob has two H&E-ish compartments (a haematoxylin-rich and an
eosin-rich side) and a few darker gland-like rings, so extracted features carry some
structure rather than uniform noise. It is synthetic, not real histology; use it to
learn the tool.
"""

from __future__ import annotations


def write_sample_slide(
    out_path: str,
    *,
    mpp0: float = 0.5,
    size: int = 1024,
    levels: int = 3,
    seed: int = 0,
) -> str:
    """Write a synthetic OME-NGFF v0.4 slide and return its path.

    ``size`` is the level-0 side in pixels (square); ``levels`` halved pyramid levels;
    ``mpp0`` the level-0 microns/pixel. Needs the ``[zarr]`` extra.
    """
    import numpy as np
    import zarr

    rng = np.random.default_rng(seed)
    h = w = int(size)
    # Level-0 RGB: bright background with an elliptical H&E-ish "tissue" blob, so the
    # default otsu (saturation) segmenter keeps a sensible patch set.
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ry, rx = (yy - h * 0.5) / (h * 0.40), (xx - w * 0.5) / (w * 0.32)
    inside = ry**2 + rx**2 <= 1.0
    # Two tissue compartments blended left -> right (haematoxylin-rich nuclei vs
    # eosin-rich stroma) so features separate into regions, not uniform noise.
    haem = np.array([120, 90, 165], dtype=np.float32)  # purple-ish, nuclei-dense
    eos = np.array([205, 120, 165], dtype=np.float32)  # pink-ish, stroma
    frac = np.clip(xx / max(w - 1, 1), 0.0, 1.0)[..., None]
    base = haem * (1.0 - frac) + eos * frac
    # A few darker gland-like rings for local structure.
    glands = np.zeros((h, w), dtype=np.float32)
    for _ in range(max(6, size // 256)):
        gy, gx = rng.uniform(0.2, 0.8, size=2) * np.array([h, w], dtype=np.float32)
        gr = rng.uniform(0.03, 0.07) * size
        dist = np.hypot(yy - gy, xx - gx)
        glands += np.exp(-((dist - gr) ** 2) / (2.0 * (gr * 0.3) ** 2))
    base -= 70.0 * np.clip(glands, 0.0, 1.0)[..., None]
    noise = rng.integers(-18, 18, size=(h, w, 3))
    blob = np.clip(base + noise, 0, 255).astype(np.uint8)
    img[inside] = blob[inside]

    g = zarr.open_group(str(out_path), mode="w", zarr_format=2)
    sizes: list[tuple[int, int]] = []
    cur = img
    for i in range(int(levels)):
        lh, lw = cur.shape[:2]
        sizes.append((lh, lw))
        a = g.create_array(
            str(i),
            shape=(1, 3, 1, lh, lw),
            chunks=(1, 1, 1, min(256, lh), min(256, lw)),
            dtype="uint8",
        )
        a[0, :, 0] = np.transpose(cur, (2, 0, 1))  # (H,W,3) -> (3,H,W)
        cur = cur[::2, ::2]  # halve for the next pyramid level

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
        {"version": "0.4", "name": "sample", "axes": axes, "datasets": datasets}
    ]
    return str(out_path)
