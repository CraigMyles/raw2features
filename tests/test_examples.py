"""The shipped example config + manifest files parse (so they can't silently rot)."""

from __future__ import annotations

from pathlib import Path

from raw2features.core.config import load_extractions, load_manifest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_example_extractions_yaml_parses():
    exts = load_extractions(str(EXAMPLES / "extractions.yaml"))
    assert len(exts) >= 2
    assert all("model" in e for e in exts)
    # the same model repeats at two MPPs (the ablation the file demonstrates)
    assert any(e["model"] == "uni" and e.get("mpp") == 0.5 for e in exts)
    assert any(e["model"] == "uni" and e.get("mpp") == 1.0 for e in exts)


def test_example_cohort_manifest_parses():
    rows = load_manifest(str(EXAMPLES / "cohort.csv"))
    assert len(rows) >= 3
    assert all("path" in r and r["path"] for r in rows)
    # at least one row carries a per-slide source_mpp, at least one omits it
    assert any("source_mpp" in r for r in rows)
    assert any("source_mpp" not in r for r in rows)
