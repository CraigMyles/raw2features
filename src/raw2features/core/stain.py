"""Stain normalization -- optional H&E colour normalization (Macenko/Reinhard/Vahadane).

Pulls an H&E image toward a fixed canonical stain so a model trained on one lab's slides
meets a familiar colour distribution. Three methods behind :func:`make_normalizer` --
Macenko (numpy), Reinhard (LAB mean/std, needs cv2), Vahadane (NMF stain matrix, needs
scikit-learn). Two uses:

* GrandQC QC -- :func:`macenko_normalize` on the artifact-stage input (single image).
* Patch embedding -- :func:`make_normalizer` fits the slide's stain *once* (from a
  thumbnail) and applies it to every patch, so the whole slide is normalized
  consistently rather than each patch estimating its own (noisy) stain.

Caveats: the per-slide fit cannot correct *intra-slide* stain gradients (one transform
per slide); the Reinhard target is OpenCV-8-bit LAB (not CIELAB) -- self-consistent but
not portable to CIELAB; the Vahadane NMF fit varies slightly across BLAS/threading. For
off-the-shelf foundation-model encoders the downstream benefit of inference-time
normalization is small-to-neutral -- treat it as an experiment knob.

Macenko et al., ISBI 2009; Reinhard et al., 2001; Vahadane et al., IEEE TMI 2016.
"""

from __future__ import annotations

import numpy as np

# Canonical H&E reference (Macenko / Vahadane): stain vectors (RGB OD) + 99th-pct
# concentrations. Normalizing to these makes an image look like "standard" H&E. Verbatim
# from schaugf/HEnorm_python, which uses them with Io=240 -- matching our io default, so
# the constants and our OD scale are consistent (do NOT "fix" io to 255).
_HE_REF = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
_MAXC_REF = np.array([1.9705, 1.0308])

# Canonical Reinhard target: per-channel L*a*b* (OpenCV 8-bit) mean/std of standard H&E.
_LAB_REF_MEAN = np.array([180.0, 146.0, 122.0])
_LAB_REF_STD = np.array([42.0, 12.0, 8.0])


def macenko_fit(
    rgb, *, io: int = 240, alpha: float = 1.0, beta: float = 0.15,
    sample: int = 200_000, rng_seed: int = 0,
):
    """Estimate a source stain from ``rgb``; return ``(he, max_c)`` or ``(None, None)``.

    ``he`` is the 3x2 stain matrix (H&E in optical density), ``max_c`` the per-stain
    99th-pct concentration. ``(None, None)`` on too little tissue or a degenerate
    estimate -- callers then skip normalization.
    """
    flat = np.asarray(rgb)[..., :3].reshape(-1, 3).astype(np.float64)
    od = -np.log((flat + 1.0) / io)
    # Tissue: not bright background and not near-black padding. OME-Zarr canvas
    # fill (0,0,0) is high-OD, so the OD<beta drop misses it -- it would poison the
    # whole-slide fit (the same black-canvas issue handled in the cohort QC scan).
    tissue = od[~np.any(od < beta, axis=1) & (flat.max(1) >= 10)]
    if len(tissue) < 100:
        return None, None
    sub = tissue
    if len(sub) > sample:
        idx = np.random.default_rng(rng_seed).choice(len(sub), sample, replace=False)
        sub = sub[idx]
    try:
        _, vecs = np.linalg.eigh(np.cov(sub.T))
        proj = sub @ vecs[:, 1:3]
        phi = np.arctan2(proj[:, 1], proj[:, 0])
        lo, hi = np.percentile(phi, alpha), np.percentile(phi, 100 - alpha)
        v_lo = vecs[:, 1:3] @ np.array([np.cos(lo), np.sin(lo)])
        v_hi = vecs[:, 1:3] @ np.array([np.cos(hi), np.sin(hi)])
        he = np.array([v_lo, v_hi]).T if v_lo[0] > v_hi[0] else np.array([v_hi, v_lo]).T
        conc = np.linalg.pinv(he) @ od.T
    except np.linalg.LinAlgError:
        return None, None
    max_c = np.percentile(conc, 99, axis=1)
    max_c[max_c == 0] = 1.0
    return he, max_c


def macenko_apply(rgb, he, max_c, *, io: int = 240) -> np.ndarray:
    """Re-express ``rgb`` in the canonical stain via a fitted ``(he, max_c)``."""
    rgb = np.asarray(rgb)[..., :3]
    h, w = rgb.shape[:2]
    od = -np.log((rgb.reshape(-1, 3).astype(np.float64) + 1.0) / io)
    conc = np.linalg.pinv(he) @ od.T
    conc = conc / max_c[:, None] * _MAXC_REF[:, None]
    out = io * np.exp(-_HE_REF @ conc)
    return np.clip(out, 0, 255).T.reshape(h, w, 3).astype(np.uint8)


def macenko_normalize(rgb, **kw) -> np.ndarray:
    """Macenko-normalize a single image (fit its stain, then re-express it canonically).

    Degenerate inputs (too little tissue / non-invertible) are returned unchanged.
    """
    he, max_c = macenko_fit(rgb, **kw)
    if he is None:
        return np.asarray(rgb)[..., :3].astype(np.uint8)
    return macenko_apply(rgb, he, max_c, io=int(kw.get("io", 240)))


def _lab_tissue_stats(rgb):
    """Per-channel L*a*b* mean/std over tissue (drop white glass + black pad)."""
    import cv2

    lab = cv2.cvtColor(np.asarray(rgb)[..., :3].astype(np.uint8),
                       cv2.COLOR_RGB2LAB).astype(np.float64)
    flat = lab.reshape(-1, 3)
    fg = (flat[:, 0] > 20) & (flat[:, 0] < 235)  # not black padding, not white glass
    if fg.sum() < 100:
        fg = np.ones(len(flat), bool)
    return flat[fg].mean(0), flat[fg].std(0) + 1e-6


def reinhard_fit(rgb):
    """Reinhard source stats (a slide's L*a*b* mean, std); apply maps to canonical."""
    return _lab_tissue_stats(rgb)


def reinhard_apply(rgb, src_mean, src_std) -> np.ndarray:
    """Reinhard: shift L*a*b* from the slide's stats to the canonical H&E target."""
    import cv2

    lab = cv2.cvtColor(np.asarray(rgb)[..., :3].astype(np.uint8),
                       cv2.COLOR_RGB2LAB).astype(np.float64)
    out = (lab - src_mean) / src_std * _LAB_REF_STD + _LAB_REF_MEAN
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def vahadane_fit(rgb, *, io: int = 240, beta: float = 0.15,
                 sample: int = 100_000, rng_seed: int = 0):
    """Stain matrix via NMF on optical density; ``(he, max_c)`` or ``(None, None)``.

    A *fast NMF approximation* of Vahadane stain separation, not faithful SPCN: the
    stain matrix is plain NMF and concentrations come from least-squares (via
    :func:`macenko_apply`). True Vahadane sparse-codes the per-pixel concentrations
    (L1/LARS-LASSO) -- measured ~2.3 s/patch vs 0.2 ms (~10^4x), impractical at
    feature-extraction scale -- and sklearn's sparse-NMF collapses a stain vector on
    a 2-component H&E fit, so robust plain NMF is the viable choice. Needs scikit-learn.
    """
    try:
        from sklearn.decomposition import NMF
    except ImportError as e:  # pragma: no cover - optional dep
        raise RuntimeError(
            "vahadane stain norm needs scikit-learn (pip install scikit-learn)"
        ) from e
    flat = np.asarray(rgb)[..., :3].reshape(-1, 3).astype(np.float64)
    od = -np.log((flat + 1.0) / io)
    tissue = od[~np.any(od < beta, axis=1) & (flat.max(1) >= 10)]  # drop bg+black
    if len(tissue) < 100:
        return None, None
    sub = tissue
    if len(sub) > sample:
        idx = np.random.default_rng(rng_seed).choice(len(sub), sample, replace=False)
        sub = sub[idx]
    try:
        model = NMF(n_components=2, init="nndsvda", max_iter=400, random_state=rng_seed)
        model.fit(np.maximum(sub, 0.0))
        he = model.components_.T  # 3 x 2 stain vectors (OD)
        norms = np.linalg.norm(he, axis=0)
        if np.any(norms < 1e-6):  # a collapsed stain vector -> degenerate separation
            return None, None
        he = he / norms
        if he[0, 0] < he[0, 1]:  # order to match Macenko (col 0 = higher-R stain)
            he = he[:, ::-1]
        conc = np.linalg.pinv(he) @ od.T
    except (np.linalg.LinAlgError, ValueError):  # guard the pinv path
        return None, None
    max_c = np.percentile(conc, 99, axis=1)
    max_c[max_c == 0] = 1.0
    return he, max_c


def make_normalizer(method: str | None, ref_rgb):
    """A per-slide patch normalizer: fit ``method`` from ``ref_rgb``; return a callable.

    ``method`` is ``macenko`` | ``reinhard`` | ``vahadane``. The slide's stain is fitted
    once and applied to every patch (``patch -> normalized_patch``); returns ``None`` if
    ``method`` is falsy/unknown or the fit is degenerate -- callers then skip it.
    """
    if method in ("macenko", "vahadane"):
        he, max_c = (macenko_fit if method == "macenko" else vahadane_fit)(ref_rgb)
        if he is None:
            return None
        return lambda img: macenko_apply(img, he, max_c)
    if method == "reinhard":
        mean, std = reinhard_fit(ref_rgb)
        return lambda img: reinhard_apply(img, mean, std)
    return None
