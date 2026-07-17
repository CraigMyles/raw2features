"""The blessed top-level public API and the card-sourced resize interpolation.

The public names must resolve from ``raw2features`` directly (so users couple to a
stable surface, not deep module paths) while ``import raw2features`` stays light enough
for the lean core install. No torch needed.
"""

from __future__ import annotations

import subprocess
import sys
import warnings

import pytest

PUBLIC = (
    "embed_slide", "run_slide", "RunConfig", "register", "available", "get",
    "write_patches_geojson", "validate_store",
)


def test_public_names_resolve():
    import raw2features

    for name in PUBLIC:
        assert getattr(raw2features, name) is not None
    assert set(PUBLIC).issubset(set(raw2features.__all__))


def test_slide_entry_points_state_their_supported_abstraction():
    import raw2features

    assert "high-level public entry point" in raw2features.embed_slide.__doc__
    assert "single-grid primitive" in raw2features.run_slide.__doc__


def test_unknown_attr_raises():
    import raw2features

    with pytest.raises(AttributeError):
        raw2features.does_not_exist  # noqa: B018


def test_bare_import_is_light():
    # A fresh interpreter: importing the package must not pull torch or zarr, even when
    # they are installed (the lazy re-export keeps the core install importable).
    code = (
        "import sys, raw2features; "
        "assert 'torch' not in sys.modules, 'torch pulled at import'; "
        "assert 'zarr' not in sys.modules, 'zarr pulled at import'; "
        "print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_pil_resample_follows_the_card():
    from PIL import Image

    from raw2features.embedders.transforms import _pil_resample

    assert _pil_resample("bicubic") == Image.BICUBIC
    assert _pil_resample("bilinear") == Image.BILINEAR
    assert _pil_resample("lanczos") == Image.LANCZOS
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert _pil_resample("nonsense") == Image.BILINEAR
    assert any("unknown interpolation" in str(w.message) for w in caught)
