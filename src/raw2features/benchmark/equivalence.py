"""Output-equivalence check for optimization experiments.

A speed-only change must leave the embeddings numerically identical (within fp
tolerance). This compares two ``*.embeddings.zarr`` stores: ``coords`` must match
exactly, and every shared ``features/<model>`` must be ``allclose``. It is the
guarantee that a speed-only change to the pipeline has not altered the
embeddings it produces.
"""

from __future__ import annotations


def compare_stores(
    path_a: str, path_b: str, *, rtol: float = 1e-3, atol: float = 1e-3
) -> dict:
    """Compare two embeddings stores. Returns a report dict with ``ok`` overall.

    ``coords`` must be bit-identical (the spatial contract); each shared model's
    features must be ``np.allclose`` at the given tolerances (default suits the
    float16 storage dtype).
    """
    import numpy as np

    from raw2features.core.store import open_grid

    a = open_grid(path_a)  # the sole grid (one geometry per store)
    b = open_grid(path_b)
    report: dict = {"ok": True, "issues": [], "models": {}}

    ca, cb = np.asarray(a["coords"][:]), np.asarray(b["coords"][:])
    if ca.shape != cb.shape or not np.array_equal(ca, cb):
        report["ok"] = False
        report["issues"].append(f"coords differ ({ca.shape} vs {cb.shape})")

    fa, fb = a["features"], b["features"]
    shared = sorted(set(fa.keys()) & set(fb.keys()))
    if not shared:
        report["ok"] = False
        report["issues"].append("no shared feature models to compare")

    for m in shared:
        xa = np.asarray(fa[m][:]).astype(np.float32)
        xb = np.asarray(fb[m][:]).astype(np.float32)
        if xa.shape != xb.shape:
            report["ok"] = False
            report["models"][m] = {"ok": False, "reason": f"{xa.shape} vs {xb.shape}"}
            continue
        close = bool(np.allclose(xa, xb, rtol=rtol, atol=atol))
        max_abs = float(np.max(np.abs(xa - xb))) if xa.size else 0.0
        report["models"][m] = {"ok": close, "max_abs_diff": max_abs}
        if not close:
            report["ok"] = False

    return report
