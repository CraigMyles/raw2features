"""Multiplex (spatial-proteomics) modality: channel-aware reader, nuclear segmenter,
KRONOS spec. All weight-free (CPU-safe); the gated KRONOS forward is marked slow.
"""

from __future__ import annotations

import numpy as np
import pytest

from raw2features.core.geometry import Point, Region, Size
from raw2features.readers.omezarr import OmeZarrReader


def test_reader_exposes_channels_and_native_read(synthetic_multiplex_ngff):
    with OmeZarrReader(synthetic_multiplex_ngff) as r:
        assert r.channel_names == ["DAPI", "CD3", "CD8", "CD20", "FOXP3"]
        reg = Region(level=0, location=Point(0, 0), size=Size(32, 32))
        mc = r.read_region_channels(reg)
        assert mc.shape == (32, 32, 5) and mc.dtype == np.uint16
        rgb = r.read_region(reg)  # the H&E path still collapses to 3-channel uint8
        assert rgb.shape == (32, 32, 3) and rgb.dtype == np.uint8


def test_reader_channel_names_none_for_rgb(synthetic_ngff):
    with OmeZarrReader(synthetic_ngff) as r:
        assert r.channel_names is None  # plain RGB -> no marker panel


def test_nuclear_segmenter_thresholds_dapi(synthetic_multiplex_ngff):
    from raw2features.segmenters.nuclear import NuclearSegmenter

    with OmeZarrReader(synthetic_multiplex_ngff) as r:
        tm = NuclearSegmenter(seg_mpp=2.0).segment(r)
        assert tm.mask.ndim == 2 and tm.mask.dtype == np.float32
        assert set(np.unique(tm.mask)).issubset({0.0, 1.0})


def test_nuclear_segmenter_errors_without_a_nuclear_channel(synthetic_multiplex_ngff):
    from raw2features.segmenters.nuclear import NuclearSegmenter

    seg = NuclearSegmenter(nuclear_aliases=("nonexistent",))
    with OmeZarrReader(synthetic_multiplex_ngff) as r, pytest.raises(ValueError):
        seg.segment(r)


def test_kronos_spec_is_multiplex():
    from raw2features.embedders.model_registry import get_spec

    s = get_spec("kronos")
    assert s.family == "kronos"
    assert s.modality == "multiplex"
    assert s.embedding_dim == 384
    assert s.input_size == 224
    assert s.gated is True
    assert "CC-BY-NC-ND" in s.license


def test_kronos_family_resolves_without_the_package():
    from raw2features.core import plugins

    assert plugins.get("embedders", "kronos").__name__ == "KronosEmbedder"


def test_brightfield_models_default_to_brightfield_modality():
    from raw2features.embedders.model_registry import get_spec

    assert get_spec("uni").modality == "brightfield"
    assert get_spec("resnet50").modality == "brightfield"


def test_kronos_marker_resolver_handles_synonyms_and_compound_names():
    # Real CODEX panels name markers differently than KRONOS's vocab; exact-match-only
    # silently drops supported markers, so the resolver handles synonyms/compound names.
    from raw2features.embedders.kronos_embedder import _norm_marker, _resolve_marker

    vocab = {"CD4": 0, "CD162": 0, "CYTOKERATIN": 0, "GZMB": 0, "DAPI": 0}
    cases = {
        "cd4": "CD4",                 # exact
        "pancytokeratin": "CYTOKERATIN",  # synonym (pan- cocktail)
        "granzymeb": "GZMB",          # synonym (gene symbol)
        "cla_cd162": "CD162",         # compound name -> CD-number token
        "cd62l": None,                # genuinely absent -> dropped
        "emptya488_1": None,          # empty cycle -> normed to None
    }
    for raw, expected in cases.items():
        assert _resolve_marker(_norm_marker(raw), vocab) == expected, raw


def test_kronos_set_panel_records_retrievable_mapping():
    # The store must record HOW each channel was identified to KRONOS (channel ->
    # canonical marker + id), so a multiplex embedding is auditable/reproducible.
    from raw2features.embedders.kronos_embedder import KronosEmbedder
    from raw2features.embedders.model_registry import get_spec

    emb = KronosEmbedder(get_spec("kronos"))
    # mock vocab: normalised key -> (id, mean, std, canonical KRONOS name)
    emb._vocab = {
        "DAPI": (4, 0.1, 0.2, "DAPI"),
        "CD4": (248, 0.1, 0.2, "CD4"),
        "CYTOKERATIN": (322, 0.1, 0.2, "Cytokeratin"),
        "CD162": (198, 0.1, 0.2, "CD162"),
    }
    s = emb.set_panel(
        ["hoechst1", "cd4", "pancytokeratin", "cla_cd162", "blank", "cd62l"]
    )
    mp = {m["channel"]: (m["kronos_marker"], m["marker_id"]) for m in s["mapping"]}
    assert mp["hoechst1"] == ("DAPI", 4)
    assert mp["pancytokeratin"] == ("Cytokeratin", 322)  # synonym -> canonical + id
    assert mp["cla_cd162"] == ("CD162", 198)  # compound CD-name resolved
    assert s["n_markers"] == 4
    assert s["unmatched"] == ["cd62l"]  # named-but-unknown surfaced; "blank" silent
    assert "vocabulary" in s  # where the canonical names/ids came from


def test_multiplex_scaling_is_dtype_aware():
    # The KRONOS input must reach [0,1] for any source dtype, not just uint16 - a
    # hard-coded /65535 silently squashes uint8 multiplex. (Pure helper, no weights.)
    from raw2features.embedders.kronos_embedder import _to_unit_interval

    u16 = np.full((1, 1, 2, 2), 65535, dtype=np.uint16)
    assert np.allclose(_to_unit_interval(u16.astype(np.float32), np.uint16), 1.0)
    u8 = np.full((1, 1, 2, 2), 255, dtype=np.uint8)
    assert np.allclose(_to_unit_interval(u8.astype(np.float32), np.uint8), 1.0)
    f = np.full((1, 1, 2, 2), 0.5, dtype=np.float32)
    assert np.allclose(_to_unit_interval(f, np.float32), 0.5)  # already normalised
    with pytest.raises(NotImplementedError):
        _to_unit_interval(np.zeros((1, 1, 2, 2), np.float32), np.int16)


@pytest.mark.slow
def test_kronos_forward_on_synthetic_multiplex(synthetic_multiplex_ngff):
    """Gated + heavy: loads KRONOSv1 and embeds a synthetic multiplex patch."""
    import torch

    pytest.importorskip("kronos")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from raw2features.embedders.model_registry import build_embedder

    emb = build_embedder("kronos").load("cuda")
    with OmeZarrReader(synthetic_multiplex_ngff) as r:
        emb.set_panel(r.channel_names)
        patch = r.read_region_channels(
            Region(level=0, location=Point(0, 0), size=Size(224, 224))
        )
    out = emb.embed_batch(emb.transform_batch([patch], "cuda"))
    assert tuple(out.shape) == (1, 384)
