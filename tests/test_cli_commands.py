"""Command-level smoke tests for the read-side CLIs: validate-store, export-h5, verify.

The audit flagged these as exercised only via their Python helpers, never through the
actual `app` command surface. These run them as a user would (typer CliRunner).
"""

from __future__ import annotations

import os
from dataclasses import replace
from importlib import import_module

import pytest
import typer
from click.utils import strip_ansi
from typer.testing import CliRunner

from conftest import MockEmbedder, build_ngff_v04
from raw2features.cli._validation import parse_channel_names_file
from raw2features.cli.main import app

try:
    import torch as _torch_mod  # noqa: F401

    _TORCH = True
except ImportError:
    _TORCH = False


def test_channel_names_file_parsers_are_format_and_path_independent(tmp_path):
    txt = tmp_path / "channels.txt"
    csv = tmp_path / "panel.csv"
    tsv = tmp_path / "renamed.tsv"
    txt.write_text("\ufeff# physical order\n CD3 \n\nCK\nDAPI\n")
    csv.write_text('channel_name\nCD3\n"CK"\nDAPI\n')
    tsv.write_text("marker\nCD3\nCK\nDAPI\n")

    assert parse_channel_names_file(str(txt)) == ["CD3", "CK", "DAPI"]
    assert parse_channel_names_file(str(csv)) == ["CD3", "CK", "DAPI"]
    assert parse_channel_names_file(str(tsv)) == ["CD3", "CK", "DAPI"]


@pytest.mark.parametrize(
    ("name", "contents", "message"),
    [
        ("empty.txt", "", "at least one"),
        ("duplicate.txt", "CD3\ncd3\n", "unique"),
        ("columns.csv", "CD3,CK\n", "one column"),
        ("blank.csv", "channel_name\n\n", "at least one"),
    ],
)
def test_channel_names_file_parser_rejects_ambiguous_inputs(
    tmp_path, name, contents, message
):
    path = tmp_path / name
    path.write_text(contents)
    with pytest.raises(typer.BadParameter, match=message):
        parse_channel_names_file(str(path))


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


def test_verify_cli_passes_native_multiplex_legacy_segmenter_guard(
    tmp_path, monkeypatch
):
    verify_module = import_module("raw2features.cli.verify")
    captured = {}

    monkeypatch.setattr(verify_module, "expected_model_contracts", lambda cfg: {})
    monkeypatch.setattr(
        verify_module,
        "resolve_multiplex_source_config",
        lambda slide, cfg: replace(
            cfg,
            resolved_channel_names=["DAPI", "CD3"],
            resolved_nuclear_channel_indices=[0],
            resolved_original_channel_names=["DAPI", "CD3"],
        ),
    )

    def fake_complete(*args, **kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr(verify_module, "is_complete", fake_complete)
    result = CliRunner().invoke(
        app,
        [
            "verify",
            str(tmp_path / "slide.zarr"),
            "--receipts-dir",
            str(tmp_path / "receipts"),
            "-f",
            "kronos",
            "--device",
            "cpu",
            "--quiet",
        ],
    )

    assert result.exit_code == 0, result.output
    requirements = captured["compatible_grid_segmenters"]
    assert requirements
    assert set(next(iter(requirements.values())).values()) == {"nuclear"}


def _multiplex_cli_options(channel_names_file=None) -> list[str]:
    options = [
        "--multiplex-strategy",
        "channelwise",
        "--marker",
        "CD3",
        "--marker",
        "CK",
        "--multiplex-normalization",
        "percentile",
        "--multiplex-percentile-low",
        "2",
        "--multiplex-percentile-high",
        "98",
        "--multiplex-aggregation",
        "concat",
        "--multiplex-normalization-max-side-px",
        "1024",
    ]
    if channel_names_file is not None:
        options.extend(["--channel-names-file", str(channel_names_file)])
    return options


def _assert_multiplex_config(cfg) -> None:
    assert cfg.multiplex_strategy == "channelwise"
    assert cfg.multiplex_markers == ["CD3", "CK"]
    assert cfg.multiplex_normalization == "percentile"
    assert cfg.multiplex_percentile_low == pytest.approx(2.0)
    assert cfg.multiplex_percentile_high == pytest.approx(98.0)
    assert cfg.multiplex_aggregation == "concat"
    assert cfg.multiplex_normalization_max_side_px == 1024
    assert cfg.channel_names_override == ["CD3", "CK", "DAPI"]


def test_embed_cli_threads_multiplex_options(tmp_path, monkeypatch):
    embed_module = import_module("raw2features.cli.embed")
    captured = {}

    def fake_embed_slide(slide, out_dir, cfg, **kwargs):
        captured["cfg"] = cfg
        return {"status": "complete"}

    monkeypatch.setattr(embed_module, "embed_slide", fake_embed_slide)
    names = tmp_path / "channels.txt"
    names.write_text("CD3\nCK\nDAPI\n")
    result = CliRunner().invoke(
        app,
        [
            "embed",
            str(tmp_path / "unused.zarr"),
            str(tmp_path / "out"),
            *_multiplex_cli_options(names),
        ],
    )

    assert result.exit_code == 0, result.output
    _assert_multiplex_config(captured["cfg"])


def test_embed_many_cli_threads_multiplex_options(tmp_path, monkeypatch):
    embed_many_module = import_module("raw2features.cli.embed_many")
    slides = tmp_path / "slides"
    slides.mkdir()
    (slides / "one.zarr").mkdir()
    names = tmp_path / "channels.csv"
    names.write_text("channel_name\nCD3\nCK\nDAPI\n")
    captured = {}

    def fake_embed_shard_serial(shard, out_dir, cfg, *args):
        captured["cfg"] = cfg
        return 1, 0, 0

    monkeypatch.setattr(
        embed_many_module, "_embed_shard_serial", fake_embed_shard_serial
    )
    monkeypatch.setattr(
        embed_many_module,
        "resolve_multiplex_source_config",
        lambda slide, cfg: replace(
            cfg, resolved_channel_names=list(cfg.channel_names_override)
        ),
    )
    result = CliRunner().invoke(
        app,
        [
            "embed-many",
            str(slides),
            str(tmp_path / "out"),
            *_multiplex_cli_options(names),
        ],
    )

    assert result.exit_code == 0, result.output
    _assert_multiplex_config(captured["cfg"])


def test_embed_many_preflights_every_panel_before_starting_worker(
    tmp_path, monkeypatch
):
    embed_many_module = import_module("raw2features.cli.embed_many")
    slides = tmp_path / "slides"
    slides.mkdir()
    (slides / "one.zarr").mkdir()
    (slides / "two.zarr").mkdir()
    checked = []
    worker_started = False

    def fake_resolve(slide, cfg):
        checked.append(os.path.basename(slide))
        if len(checked) == 2:
            raise ValueError("channel count mismatch")
        return cfg

    def fake_embed_shard_serial(*args, **kwargs):
        nonlocal worker_started
        worker_started = True
        return 0, 0, 0

    monkeypatch.setattr(
        embed_many_module, "resolve_multiplex_source_config", fake_resolve
    )
    monkeypatch.setattr(
        embed_many_module, "_embed_shard_serial", fake_embed_shard_serial
    )
    result = CliRunner().invoke(
        app,
        ["embed-many", str(slides), str(tmp_path / "out")],
    )

    assert result.exit_code == 1, result.output
    assert checked == ["one.zarr", "two.zarr"]
    assert not worker_started
    assert "multiplex panel preflight failed" in strip_ansi(result.output)


def test_embed_many_rejects_unknown_strategy_before_starting_worker(
    tmp_path, monkeypatch
):
    embed_many_module = import_module("raw2features.cli.embed_many")
    slides = tmp_path / "slides"
    slides.mkdir()
    (slides / "one.zarr").mkdir()
    worker_started = False

    def fake_embed_shard_serial(*args, **kwargs):
        nonlocal worker_started
        worker_started = True
        return 0, 0, 0

    monkeypatch.setattr(
        embed_many_module, "_embed_shard_serial", fake_embed_shard_serial
    )
    result = CliRunner().invoke(
        app,
        [
            "embed-many",
            str(slides),
            str(tmp_path / "out"),
            "--multiplex-strategy",
            "not-a-strategy",
        ],
    )

    assert result.exit_code == 1, result.output
    assert not worker_started
    assert "multiplex panel preflight failed" in strip_ansi(result.output)


@pytest.mark.parametrize(
    ("extra_options", "message"),
    [
        (["--slide-encoder", "titan"], "model-agnostic slide poolers"),
        (["--devices", "cpu,cpu"], "single-device"),
    ],
)
def test_embed_many_rejects_incompatible_strategy_execution_before_worker(
    tmp_path, monkeypatch, extra_options, message
):
    embed_many_module = import_module("raw2features.cli.embed_many")
    slides = tmp_path / "slides"
    slides.mkdir()
    (slides / "one.zarr").mkdir()
    worker_started = False

    def fake_embed_shard_serial(*args, **kwargs):
        nonlocal worker_started
        worker_started = True
        return 0, 0, 0

    monkeypatch.setattr(
        embed_many_module, "_embed_shard_serial", fake_embed_shard_serial
    )
    result = CliRunner().invoke(
        app,
        [
            "embed-many",
            str(slides),
            str(tmp_path / "out"),
            "--multiplex-strategy",
            "channelwise",
            *extra_options,
        ],
    )

    assert result.exit_code == 1, result.output
    assert not worker_started
    assert message in strip_ansi(result.output)


def test_verify_cli_threads_multiplex_options(tmp_path, monkeypatch):
    verify_module = import_module("raw2features.cli.verify")
    captured = {}

    def fake_resolve(slide, cfg, *, device):
        captured["cfg"] = cfg
        return cfg, {}

    monkeypatch.setattr(
        verify_module, "resolve_multiplex_output_contracts", fake_resolve
    )
    monkeypatch.setattr(
        verify_module, "resolve_multiplex_source_config", lambda slide, cfg: cfg
    )
    monkeypatch.setattr(verify_module, "is_complete", lambda *args, **kwargs: True)
    names = tmp_path / "channels.tsv"
    names.write_text("marker\nCD3\nCK\nDAPI\n")
    result = CliRunner().invoke(
        app,
        [
            "verify",
            str(tmp_path / "slide.zarr"),
            "--receipts-dir",
            str(tmp_path / "receipts"),
            "--device",
            "cpu",
            "--quiet",
            *_multiplex_cli_options(names),
        ],
    )

    assert result.exit_code == 0, result.output
    _assert_multiplex_config(captured["cfg"])


def test_embed_cli_parses_namespaced_strategy_params(tmp_path, monkeypatch):
    embed_module = import_module("raw2features.cli.embed")
    captured = {}

    def fake_embed_slide(slide, out_dir, cfg, **kwargs):
        captured["cfg"] = cfg
        return {"status": "complete"}

    monkeypatch.setattr(embed_module, "embed_slide", fake_embed_slide)
    result = CliRunner().invoke(
        app,
        [
            "embed",
            str(tmp_path / "unused.zarr"),
            str(tmp_path / "out"),
            "--multiplex-strategy",
            "third_party",
            "--multiplex-params",
            '{"adapter":{"width":3},"enabled":true}',
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["cfg"].multiplex_strategy_params == {
        "adapter": {"width": 3},
        "enabled": True,
    }


@pytest.mark.parametrize(
    "args",
    [
        ["embed", "slide.zarr", "out"],
        ["embed-many", "slides", "out"],
        ["verify", "slide.zarr", "--receipts-dir", "receipts"],
    ],
)
def test_commands_reject_invalid_strategy_params_json(args):
    result = CliRunner().invoke(
        app,
        [
            *args,
            "--multiplex-strategy",
            "third_party",
            "--multiplex-params",
            '{"bad": NaN}',
        ],
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 2
    assert "--multiplex-params" in output
    assert "finite JSON object" in output


@pytest.mark.parametrize(
    "args",
    [
        ["embed", "slide.zarr", "out"],
        ["embed-many", "slides", "out"],
        ["verify", "slide.zarr", "--receipts-dir", "receipts"],
    ],
)
@pytest.mark.parametrize(
    "bounds",
    [
        ["--multiplex-percentile-low", "99", "--multiplex-percentile-high", "1"],
        ["--multiplex-percentile-low", "nan"],
        ["--multiplex-percentile-high", "101"],
    ],
)
def test_commands_reject_invalid_multiplex_percentiles(args, bounds):
    result = CliRunner().invoke(app, [*args, *bounds])
    output = strip_ansi(result.output)

    assert result.exit_code == 2
    assert "--multiplex-percentile" in output
    assert "0 <= low < high <= 100" in output


@pytest.mark.parametrize(
    "args",
    [
        ["embed", "slide.zarr", "out"],
        ["embed-many", "slides", "out"],
        ["verify", "slide.zarr", "--receipts-dir", "receipts"],
    ],
)
def test_commands_reject_nonpositive_multiplex_normalization_max_side(args):
    result = CliRunner().invoke(
        app, [*args, "--multiplex-normalization-max-side-px", "0"]
    )
    output = strip_ansi(result.output)

    assert result.exit_code == 2
    assert "--multiplex-normalization-max-side-px" in output


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
    output = strip_ansi(result.output)

    assert result.exit_code == 2
    assert "--amp" in output
    assert "auto, fp32, bf16, fp16" in output


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
    output = strip_ansi(result.output)

    assert result.exit_code == 2
    assert "--batch-size" in output
    assert "greater than zero" in output


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
    output = strip_ansi(result.output)

    assert result.exit_code == 2
    assert option in output
    assert "greater than zero" in output


def test_mpp_option_rejects_nonfinite_value_before_work():
    result = CliRunner().invoke(app, ["embed", "slide.zarr", "out", "--mpp", "nan"])
    output = strip_ansi(result.output)

    assert result.exit_code == 2
    assert "--mpp" in output
    assert "finite and greater than zero" in output


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
