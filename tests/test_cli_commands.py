"""Command-level smoke tests for the read-side CLIs: validate-store, export-h5, verify.

The audit flagged these as exercised only via their Python helpers, never through the
actual `app` command surface. These run them as a user would (typer CliRunner).
"""

from __future__ import annotations

import os

import pytest
from typer.testing import CliRunner

from conftest import MockEmbedder, build_ngff_v04
from raw2features.cli.main import app

try:
    import torch as _torch_mod  # noqa: F401

    _TORCH = True
except ImportError:
    _TORCH = False


def _build_store(tmp_path) -> str:
    from raw2features.pipeline.runner import RunConfig, embed_slide

    slide = build_ngff_v04(str(tmp_path / "S.zarr"))
    out = str(tmp_path / "out")
    cfg = RunConfig(models=["mock"], no_seg=True, target_mpp=0.5, patch_px=64,
                    device="cpu", amp="fp32")
    embed_slide(slide, out, cfg,
                embedders=[MockEmbedder(dim=8, input_size=64, name="mock")])
    return os.path.join(out, "S.embeddings.zarr")


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_validate_store_cli_conforms(tmp_path):
    store = _build_store(tmp_path)
    r = CliRunner().invoke(app, ["validate-store", store])
    assert r.exit_code == 0, r.output
    assert "conform" in r.output.lower()


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_export_h5_cli_writes_file(tmp_path):
    pytest.importorskip("h5py")
    store = _build_store(tmp_path)
    out = str(tmp_path / "h5")
    r = CliRunner().invoke(app, ["export-h5", store, out, "-m", "mock"])
    assert r.exit_code == 0, r.output
    assert any(f.endswith(".h5") for f in os.listdir(out))


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_verify_cli_exits_1_when_not_complete(tmp_path):
    slide = build_ngff_v04(str(tmp_path / "S.zarr"))
    receipts = str(tmp_path / "receipts")  # empty -> nothing is complete
    os.makedirs(receipts, exist_ok=True)
    r = CliRunner().invoke(
        app, ["verify", slide, "--receipts-dir", receipts, "-f", "mock",
              "--no-seg", "--mpp", "0.5", "--patch-size", "64", "--quiet"],
    )
    assert r.exit_code == 1


# -- main() error wrapper (torch-free) -----------------------------------------


def _raise(exc):
    def f():
        raise exc

    return f


def test_main_maps_file_not_found_to_exit_2(monkeypatch, capsys):
    import raw2features.cli.main as m

    monkeypatch.setattr(m, "app", _raise(FileNotFoundError("missing.zarr")))
    with pytest.raises(SystemExit) as ei:
        m.main()
    assert ei.value.code == 2
    assert "missing.zarr" in capsys.readouterr().err


def test_main_maps_missing_extra_to_install_hint(monkeypatch, capsys):
    import raw2features.cli.main as m

    mod, extra = next(iter(m._EXTRA_FOR_MODULE.items()))
    monkeypatch.setattr(m, "app", _raise(ModuleNotFoundError("x", name=mod)))
    with pytest.raises(SystemExit) as ei:
        m.main()
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "pip install" in err and extra in err
