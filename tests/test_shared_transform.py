"""Transform-once fan-out: share one transformed batch across same-signature models.

When several feature extractors share a preprocessing signature (input_size, mean,
std, interpolation) the transformed tensor is identical, so the runner computes it
*once per signature* and feeds it to every model in the group via ``embed_batch``.
These tests pin (a) the grouping, (b) that a same-signature pair triggers one
transform per batch (not one per model), (c) that a differing signature is *not*
shared, and (d) that the shared-transform path yields features bit-identical to the
naive per-model path.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="torch not installed")

from conftest import MockEmbedder
from raw2features.pipeline.runner import RunConfig, _group_by_transform, run_slide


class CountingEmbedder(MockEmbedder):
    """MockEmbedder that records every ``transform_batch`` call (a spy).

    ``interp`` overrides the interpolation so two instances can be given the same
    or a differing transform signature without touching the rest of the spec.
    """

    def __init__(self, *, interp: str = "bilinear", **kwargs) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self.spec, "interpolation", interp)
        self.transform_calls: list[int] = []

    def transform_batch(self, patches, device):
        self.transform_calls.append(len(patches))
        return super().transform_batch(patches, device)


def test_group_by_transform_collapses_same_signature():
    a = CountingEmbedder(name="a", interp="bilinear")
    b = CountingEmbedder(name="b", interp="bilinear")
    c = CountingEmbedder(name="c", interp="bicubic")
    # a,b share a signature -> one group; c differs -> its own group.
    groups = _group_by_transform([a, b, c])
    assert [[e.name for e in g] for g in groups] == [["a", "b"], ["c"]]


def test_single_model_is_one_group_of_one():
    a = CountingEmbedder(name="a")
    groups = _group_by_transform([a])
    assert len(groups) == 1 and groups[0] == [a]


def test_same_signature_pair_shares_one_transform_per_batch(synthetic_ngff, tmp_path):
    # Two models, identical preprocessing signature, run together. The transform
    # must be computed once per batch and reused -- so only the first member's
    # transform_batch is called, exactly once per batch.
    common = dict(no_seg=True, target_mpp=0.5, patch_px=64, device="cpu", amp="fp32")
    a = CountingEmbedder(name="a", dim=8, bias=1.0, interp="bilinear")
    b = CountingEmbedder(name="b", dim=5, bias=2.0, interp="bilinear")

    out = str(tmp_path / "out")
    summary = run_slide(
        synthetic_ngff,
        out,
        RunConfig(models=["a", "b"], batch_size=8, **common),
        embedders=[a, b],
    )
    n = summary["n_patches"]
    n_batches = -(-n // 8)  # ceil division

    # Group leader does the (shared) transform once per batch; the other member,
    # never -- its transform_batch is skipped entirely.
    assert len(a.transform_calls) == n_batches
    assert len(b.transform_calls) == 0
    # And every patch was transformed exactly once in total.
    assert sum(a.transform_calls) == n


def test_differing_signature_each_model_transforms(synthetic_ngff, tmp_path):
    # Different interpolation -> different signature -> NOT shared: each model runs
    # its own transform (the pre-change behaviour is preserved when no sharing
    # applies).
    common = dict(no_seg=True, target_mpp=0.5, patch_px=64, device="cpu", amp="fp32")
    a = CountingEmbedder(name="a", dim=8, bias=1.0, interp="bilinear")
    b = CountingEmbedder(name="b", dim=5, bias=2.0, interp="bicubic")

    out = str(tmp_path / "out")
    summary = run_slide(
        synthetic_ngff,
        out,
        RunConfig(models=["a", "b"], batch_size=8, **common),
        embedders=[a, b],
    )
    n_batches = -(-summary["n_patches"] // 8)
    assert len(a.transform_calls) == n_batches
    assert len(b.transform_calls) == n_batches


def test_shared_transform_features_identical_to_per_model(synthetic_ngff, tmp_path):
    """The shared-transform path produces bit-identical features to a per-model run.

    Embedding each same-signature model alone (forcing its own transform) and
    embedding them together (one shared transform) must yield identical feature
    arrays -- sharing the tensor changes only *how often* it is computed, not the
    arithmetic.
    """

    common = dict(no_seg=True, target_mpp=0.5, patch_px=64, device="cpu", amp="fp32")

    def _feats(store_uri, model):
        from raw2features.core.store import open_grid

        g = open_grid(store_uri)  # the sole grid
        return np.asarray(g["features"][model][:])

    # Per-model baselines (each in its own store => each does its own transform).
    sa = run_slide(
        synthetic_ngff,
        str(tmp_path / "a"),
        RunConfig(models=["a"], batch_size=8, **common),
        embedders=[CountingEmbedder(name="a", dim=8, bias=1.0)],
    )
    sb = run_slide(
        synthetic_ngff,
        str(tmp_path / "b"),
        RunConfig(models=["b"], batch_size=8, **common),
        embedders=[CountingEmbedder(name="b", dim=5, bias=2.0)],
    )
    a_alone = _feats(sa["output_uri"], "a")
    b_alone = _feats(sb["output_uri"], "b")

    # Together: same signature -> one shared transform feeding both models.
    st = run_slide(
        synthetic_ngff,
        str(tmp_path / "both"),
        RunConfig(models=["a", "b"], batch_size=8, **common),
        embedders=[
            CountingEmbedder(name="a", dim=8, bias=1.0),
            CountingEmbedder(name="b", dim=5, bias=2.0),
        ],
    )
    assert np.array_equal(_feats(st["output_uri"], "a"), a_alone)
    assert np.array_equal(_feats(st["output_uri"], "b"), b_alone)
