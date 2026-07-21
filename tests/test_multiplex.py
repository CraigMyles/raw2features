"""Multiplex (spatial-proteomics) modality: channel-aware reader, nuclear segmenter,
KRONOS spec. All weight-free (CPU-safe); the gated KRONOS forward is marked slow.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from raw2features.core.geometry import Point, Region, Size
from raw2features.readers.omezarr import OmeZarrReader


def test_reader_exposes_channels_and_native_read(synthetic_multiplex_ngff):
    with OmeZarrReader(synthetic_multiplex_ngff) as r:
        assert r.channel_names == ["DAPI", "CD3", "CD8", "CD20", "FOXP3"]
        assert r.has_channel_axis is True
        assert r.channel_count == 5
        reg = Region(level=0, location=Point(0, 0), size=Size(32, 32))
        mc = r.read_region_channels(reg)
        assert mc.shape == (32, 32, 5) and mc.dtype == np.uint16
        rgb = r.read_region(reg)  # the H&E path still collapses to 3-channel uint8
        assert rgb.shape == (32, 32, 3) and rgb.dtype == np.uint8


def test_reader_channel_names_none_for_rgb(synthetic_ngff):
    with OmeZarrReader(synthetic_ngff) as r:
        assert r.channel_names is None  # plain RGB -> no marker panel
        assert r.has_channel_axis is True
        assert r.channel_count == 3


def test_reader_preserves_unnamed_omero_channel_positions(
    synthetic_multiplex_ngff,
):
    """An unnamed channel must not shift every later marker to the wrong pixels."""
    import zarr

    root = zarr.open_group(synthetic_multiplex_ngff, mode="r+")
    root.attrs["omero"] = {
        "channels": [
            {"label": "DAPI"},
            {"active": False},
            {"name": "CD8"},
            {"label": "CD20"},
            {"label": "FOXP3"},
        ]
    }

    with OmeZarrReader(synthetic_multiplex_ngff) as reader:
        assert reader.channel_names == ["DAPI", "", "CD8", "CD20", "FOXP3"]
        assert len(reader.channel_names or []) == reader.channel_count == 5
        patch = reader.read_region_channels(
            Region(level=0, location=Point(0, 0), size=Size(1, 1))
        )

    # The label after the unnamed slot is still bound to physical channel index 2.
    assert patch[0, 0, 2] == 3000


def test_reader_surfaces_incomplete_omero_panel_without_padding(
    synthetic_multiplex_ngff,
):
    """The strategy binder can detect metadata-count/C-axis mismatches explicitly."""
    import zarr

    root = zarr.open_group(synthetic_multiplex_ngff, mode="r+")
    root.attrs["omero"] = {"channels": [{"label": "DAPI"}, {"label": "CD3"}]}

    with OmeZarrReader(synthetic_multiplex_ngff) as reader:
        assert reader.channel_names == ["DAPI", "CD3"]
        assert reader.channel_count == 5


def test_channel_name_override_fills_missing_metadata_positionally(
    synthetic_multiplex_ngff,
):
    import zarr

    root = zarr.open_group(synthetic_multiplex_ngff, mode="r+")
    root.attrs["omero"] = {"channels": [{"label": "DAPI"}, {}, {"label": "CD8"}]}
    supplied = ["dapi", "CD3", "CD8", "CD20", "FOXP3"]

    with OmeZarrReader(synthetic_multiplex_ngff) as reader:
        reader.apply_channel_names(supplied)
        assert reader.original_channel_names == ["DAPI", "", "CD8"]
        assert reader.channel_names == supplied
        assert reader.channel_names_source == "channel_names_file"


def test_channel_name_override_rejects_conflicts_counts_and_duplicates(
    synthetic_multiplex_ngff,
):
    with OmeZarrReader(synthetic_multiplex_ngff) as reader:
        with pytest.raises(ValueError, match="C index 1"):
            reader.apply_channel_names(["DAPI", "CK", "CD8", "CD20", "FOXP3"])
        with pytest.raises(ValueError, match="file has 2 names but C=5"):
            reader.apply_channel_names(["DAPI", "CD3"])
        with pytest.raises(ValueError, match="unique name"):
            reader.apply_channel_names(["DAPI", "CD3", "cd3", "CD20", "FOXP3"])


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


def test_nuclear_segmenter_combines_repeated_dapi_channels(
    synthetic_multiplex_ngff,
):
    from raw2features.segmenters.nuclear import NuclearSegmenter

    with OmeZarrReader(synthetic_multiplex_ngff):
        assert NuclearSegmenter()._nuclear_indices(
            ["DAPI1", "CD3", "DAPI2", "CD20", "FOXP3"]
        ) == [0, 2]


def test_nuclear_segmenter_does_not_treat_dna_biomarker_as_nuclear_stain():
    from raw2features.segmenters.nuclear import NuclearSegmenter

    segmenter = NuclearSegmenter()
    assert segmenter._nuclear_index(["DAPI", "DNA-PKcs", "CD3"]) == 0
    for non_nuclear_name in ["DNA3", "DNA12", "DNA-PKcs", "cDNA1", "DNA damage"]:
        with pytest.raises(ValueError, match="no nuclear channel"):
            segmenter._nuclear_index([non_nuclear_name, "CD3"])


@pytest.mark.parametrize(
    "name",
    [
        "DAPI_cycle1",
        "Hoechst 33342",
        "DNA",
        "DNA1",
        "DNA 2",
        "DNA1(Ir191)",
        "191Ir_DNA1",
        "Ir193_DNA2",
    ],
)
def test_nuclear_segmenter_keeps_intended_nuclear_name_variants(name):
    from raw2features.segmenters.nuclear import NuclearSegmenter

    assert NuclearSegmenter()._nuclear_index(["CD3", name]) == 1


def test_nuclear_segmenter_combines_one_dna1_dna2_pair_without_uint16_overflow():
    from raw2features.segmenters.nuclear import NuclearSegmenter

    segmenter = NuclearSegmenter()
    names = ["CD3", "DNA1(Ir191)", "DNA2(Ir193)"]
    assert segmenter._nuclear_indices(names) == [1, 2]
    block = np.zeros((2, 2, 3), dtype=np.uint16)
    block[..., 1] = np.uint16(60_000)
    block[..., 2] = np.uint16(50_000)
    single = segmenter._combine_nuclear_channels(block, [1])
    combined = segmenter._combine_nuclear_channels(block, [1, 2])
    assert single.dtype == np.uint16
    assert np.all(single == 60_000)
    assert combined.dtype == np.float32
    assert np.all(combined == 55_000.0)


def test_nuclear_segmenter_combines_metal_prefixed_dna_pair_and_repeated_dapi():
    from raw2features.segmenters.nuclear import NuclearSegmenter

    segmenter = NuclearSegmenter()
    assert segmenter._nuclear_indices(["CD3", "191Ir_DNA1", "Ir193_DNA2"]) == [
        1,
        2,
    ]
    assert segmenter._nuclear_indices(["DAPI1", "CD3", "DAPI2", "DAPI3"]) == [
        0,
        2,
        3,
    ]


@pytest.mark.parametrize(
    "names",
    [
        ["DAPI", "DNA1", "DNA2"],
        ["DNA", "DNA2"],
        ["DAPI1", "Hoechst1"],
    ],
)
def test_nuclear_segmenter_rejects_noncanonical_multiple_nuclear_matches(names):
    from raw2features.segmenters.nuclear import NuclearSegmenter

    with pytest.raises(ValueError, match="one nuclear stain family"):
        NuclearSegmenter()._nuclear_indices(names)


def test_nuclear_segmenter_combines_repeated_numbered_dna_channels():
    from raw2features.segmenters.nuclear import NuclearSegmenter

    assert NuclearSegmenter()._nuclear_indices(
        ["DNA1", "DNA1(Ir191)", "Ir193_DNA2"]
    ) == [0, 1, 2]


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
        "cd4": "CD4",  # exact
        "pancytokeratin": "CYTOKERATIN",  # synonym (pan- cocktail)
        "granzymeb": "GZMB",  # synonym (gene symbol)
        "cla_cd162": "CD162",  # compound name -> CD-number token
        "cd62l": None,  # genuinely absent -> dropped
        "emptya488_1": None,  # empty cycle -> normed to None
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


def test_kronos_set_panel_does_not_warn_about_an_empty_channel_name():
    from raw2features.embedders.kronos_embedder import KronosEmbedder
    from raw2features.embedders.model_registry import get_spec

    emb = KronosEmbedder(get_spec("kronos"))
    emb._vocab = {"DAPI": (4, 0.1, 0.2, "DAPI")}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary = emb.set_panel(["DAPI", ""])

    assert not caught
    assert summary["unmatched"] == []
    assert summary["dropped"] == [""]


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
