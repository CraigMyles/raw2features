"""Command-level smoke tests for the read-side CLIs: validate-store, export-h5, verify.

The audit flagged these as exercised only via their Python helpers, never through the
actual `app` command surface. These run them as a user would (typer CliRunner).
"""

from __future__ import annotations

import os
from importlib import import_module

import pytest
from typer.testing import CliRunner

from conftest import MockEmbedder, build_ngff_v04
from raw2features.cli.main import app

try:
    import torch as _torch_mod  # noqa: F401

    _TORCH = True
except ImportError:
    _TORCH = False


def test_embed_cli_seeds_runconfig_from_resolved_geometry(tmp_path, monkeypatch):
    """The high-level CLI must not encode "unset" as the real 1.0/224 geometry."""
    embed_module = import_module("raw2features.cli.embed")
    captured = {}

    def fake_embed_slide(slide, out_dir, cfg, **kwargs):
        captured.update(cfg=cfg, kwargs=kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(embed_module, "embed_slide", fake_embed_slide)
    result = CliRunner().invoke(
        app,
        [
            "embed",
            str(tmp_path / "unused.zarr"),
            str(tmp_path / "out"),
            "-m",
            "conch_v1_5",
            "--mpp",
            "1.0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (captured["cfg"].target_mpp, captured["cfg"].patch_px) == (1.0, 512)
    assert captured["kwargs"]["requested_mpp"] == 1.0
    assert captured["kwargs"]["requested_patch_px"] is None


def test_verify_cli_hashes_the_source_mpp_override(tmp_path, monkeypatch):
    verify_module = import_module("raw2features.cli.verify")
    captured = {}

    def fake_contracts(cfg):
        captured["source_mpp"] = cfg.source_mpp
        return {}

    monkeypatch.setattr(verify_module, "expected_model_contracts", fake_contracts)
    monkeypatch.setattr(verify_module, "is_complete", lambda *args, **kwargs: True)
    result = CliRunner().invoke(
        app,
        [
            "verify",
            str(tmp_path / "slide.zarr"),
            "--receipts-dir",
            str(tmp_path / "receipts"),
            "--source-mpp",
            "0.37",
            "--device",
            "cpu",
            "--quiet",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["source_mpp"] == pytest.approx(0.37)


@pytest.mark.parametrize(
    "args",
    [
        ["embed", "slide.zarr", "out"],
        ["embed-many", "slides", "out"],
        ["benchmark", "slide.zarr"],
        ["verify", "slide.zarr", "--receipts-dir", "receipts"],
    ],
)
def test_commands_reject_unknown_amp_before_doing_work(args):
    result = CliRunner().invoke(app, [*args, "--amp", "fast16"])

    assert result.exit_code == 2
    assert "--amp" in result.output
    assert "auto, fp32, bf16, fp16" in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["embed", "slide.zarr", "out"],
        ["embed-many", "slides", "out"],
        ["benchmark", "slide.zarr"],
    ],
)
@pytest.mark.parametrize("batch_size", ["0", "-3"])
def test_embedding_commands_reject_nonpositive_batch_size(args, batch_size):
    result = CliRunner().invoke(app, [*args, "--batch-size", batch_size])

    assert result.exit_code == 2
    assert "--batch-size" in result.output
    assert "greater than zero" in result.output


@pytest.mark.parametrize(
    ("command", "option"),
    [
        ("embed", "--mpp"),
        ("embed", "--source-mpp"),
        ("embed", "--patch-size"),
        ("embed", "--step"),
        ("embed-many", "--mpp"),
        ("embed-many", "--source-mpp"),
        ("embed-many", "--patch-size"),
        ("embed-many", "--step"),
        ("verify", "--mpp"),
        ("verify", "--source-mpp"),
        ("verify", "--patch-size"),
        ("verify", "--step"),
        ("benchmark", "--mpp"),
        ("benchmark", "--patch-size"),
    ],
)
def test_geometry_options_reject_zero_before_work(command, option):
    positional = {
        "embed": ["slide.zarr", "out"],
        "embed-many": ["slides", "out"],
        "verify": ["slide.zarr", "--receipts-dir", "receipts"],
        "benchmark": ["slide.zarr"],
    }[command]
    result = CliRunner().invoke(app, [command, *positional, option, "0"])

    assert result.exit_code == 2
    assert option in result.output
    assert "greater than zero" in result.output


def test_mpp_option_rejects_nonfinite_value_before_work():
    result = CliRunner().invoke(
        app, ["embed", "slide.zarr", "out", "--mpp", "nan"]
    )

    assert result.exit_code == 2
    assert "--mpp" in result.output
    assert "finite and greater than zero" in result.output


def _build_store(tmp_path) -> str:
    from raw2features.pipeline.runner import RunConfig, embed_slide

    slide = build_ngff_v04(str(tmp_path / "S.zarr"))
    out = str(tmp_path / "out")
    cfg = RunConfig(
        models=["mock"],
        no_seg=True,
        target_mpp=0.5,
        patch_px=64,
        device="cpu",
        amp="fp32",
    )
    embed_slide(
        slide, out, cfg, embedders=[MockEmbedder(dim=8, input_size=64, name="mock")]
    )
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
        app,
        [
            "verify",
            slide,
            "--receipts-dir",
            receipts,
            "-f",
            "mock",
            "--no-seg",
            "--mpp",
            "0.5",
            "--patch-size",
            "64",
            "--quiet",
        ],
    )
    assert r.exit_code == 1


def test_verify_cli_hides_malformed_source_credentials(tmp_path):
    secret = "DO_NOT_PRINT"
    malformed = f"https://user:{secret}@exa／mple.com/image.zarr"
    result = CliRunner().invoke(
        app,
        [
            "verify",
            malformed,
            "--receipts-dir",
            str(tmp_path / "receipts"),
        ],
    )

    assert result.exit_code == 1
    assert secret not in result.output
    assert "source URI is malformed" in result.output


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_verify_cli_binds_receipt_to_expected_output_directory(tmp_path):
    from raw2features.embedders.model_registry import get_spec
    from raw2features.pipeline.runner import RunConfig, embed_slide

    slide = build_ngff_v04(str(tmp_path / "S.zarr"))
    out = str(tmp_path / "out")
    receipts = str(tmp_path / "receipts")
    cfg = RunConfig(
        models=["resnet50"],
        no_seg=True,
        target_mpp=0.5,
        patch_px=64,
        device="cpu",
        amp="fp32",
    )
    embedder = MockEmbedder(
        dim=get_spec("resnet50").embedding_dim,
        input_size=get_spec("resnet50").input_size,
        name="resnet50",
    )
    embedder.spec = get_spec("resnet50")
    embed_slide(
        slide,
        out,
        cfg,
        receipts_dir=receipts,
        requested_mpp=0.5,
        requested_patch_px=64,
        embedders=[embedder],
    )
    common = [
        "verify",
        slide,
        "--receipts-dir",
        receipts,
        "-f",
        "resnet50",
        "--no-seg",
        "--mpp",
        "0.5",
        "--patch-size",
        "64",
        "--amp",
        "fp32",
        "--quiet",
    ]

    correct = CliRunner().invoke(app, [*common, "--out-dir", out])
    wrong = CliRunner().invoke(
        app, [*common, "--out-dir", str(tmp_path / "different-out")]
    )

    assert correct.exit_code == 0, correct.output
    assert wrong.exit_code == 1


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_verify_cli_fails_closed_for_unregistered_model(tmp_path):
    from raw2features.pipeline.runner import RunConfig, embed_slide

    slide = build_ngff_v04(str(tmp_path / "S.zarr"))
    out = str(tmp_path / "out")
    receipts = str(tmp_path / "receipts")
    cfg = RunConfig(
        models=["custom"],
        no_seg=True,
        target_mpp=0.5,
        patch_px=64,
        device="cpu",
        amp="fp32",
    )
    embed_slide(
        slide,
        out,
        cfg,
        receipts_dir=receipts,
        requested_mpp=0.5,
        requested_patch_px=64,
        embedders=[MockEmbedder(dim=8, input_size=64, name="custom")],
    )

    result = CliRunner().invoke(
        app,
        [
            "verify",
            slide,
            "--receipts-dir",
            receipts,
            "--out-dir",
            out,
            "-f",
            "custom",
            "--no-seg",
            "--mpp",
            "0.5",
            "--patch-size",
            "64",
            "--amp",
            "fp32",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 1
    assert "current output contract for unregistered model(s) custom" in result.output


def test_verify_cli_does_not_mask_internal_contract_keyerror(tmp_path, monkeypatch):
    def fail_contract(_cfg):
        raise KeyError("broken current contract")

    monkeypatch.setattr(
        "raw2features.cli.verify.expected_model_contracts", fail_contract
    )
    result = CliRunner().invoke(
        app,
        [
            "verify",
            str(tmp_path / "S.zarr"),
            "--receipts-dir",
            str(tmp_path / "receipts"),
            "-f",
            "resnet50",
            "--device",
            "cpu",
            "--quiet",
        ],
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, KeyError)
    assert result.exception.args == ("broken current contract",)


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
