"""``--compile`` (optional torch.compile of the model) wiring + correctness.

``compile`` is a speed-only knob, so it must:
  * be a *runtime* RunConfig field (it does not change the embeddings' identity, so
    it must not enter the content/grid hash -- pinned in test_config_integrity);
  * be threaded through every embed CLI;
  * leave the output finite and within fp tolerance of the non-compiled path
    (compile reorders fp ops, so it is *not* bit-identical -- we assert allclose,
    not equality). The numeric check is gated on torch + CUDA + a working inductor
    and skips cleanly where torch.compile cannot run.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from raw2features.cli.benchmark import benchmark
from raw2features.cli.embed import embed
from raw2features.cli.embed_many import embed_many
from raw2features.pipeline.runner import RunConfig


def test_compile_is_a_runtime_field():
    # Speed-only -> runtime, never content. (test_config_integrity pins that the
    # hashes are unchanged; this just asserts the classification directly.)
    assert "compile" in RunConfig._RUNTIME_FIELDS
    assert "compile" not in RunConfig._CONTENT_FIELDS
    assert RunConfig(models=["resnet50"]).compile is False  # off by default


def test_compile_does_not_change_the_config_hash():
    base = RunConfig(models=["resnet50"])
    compiled = RunConfig(models=["resnet50"], compile=True)
    assert base.content_hash() == compiled.content_hash()
    assert base.grid_hash() == compiled.grid_hash()


@pytest.mark.parametrize("cli", [embed, embed_many, benchmark])
def test_every_embed_cli_exposes_compile(cli):
    assert "compile" in inspect.signature(cli).parameters


# -- gated numeric correctness on GPU -----------------------------------------
# NB: torch is imported *inside* the GPU test below (not at module scope) so the
# three pure-Python tests above still collect and run on a torch-less box.


def _inductor_works(torch, device: str) -> bool:
    """True iff torch.compile actually lowers + runs a tiny model on *device*.

    torch.compile is only meaningful with a functional inductor/triton backend;
    on a box where that is broken we skip rather than fail (the change is still
    correct, it just can't be exercised here).
    """
    try:
        m = torch.nn.Linear(4, 4).to(device).eval()
        cm = torch.compile(m, dynamic=True)
        with torch.inference_mode():
            out = cm(torch.randn(2, 4, device=device))
        return bool(torch.isfinite(out).all())
    except Exception:  # noqa: BLE001 - any compile/backend failure -> skip
        return False


@pytest.mark.slow
def test_compiled_embedder_matches_eager_within_tolerance():
    """A compiled forward is finite and within fp tolerance of the eager forward.

    Uses the open dinov2 ViT-L proxy (no token). Skips cleanly when CUDA or a
    working torch.compile backend is unavailable, or weights can't be fetched.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch.compile speedup path needs CUDA")
    device = "cuda"
    if not _inductor_works(torch, device):
        pytest.skip("torch.compile/inductor not functional on this box")

    from raw2features.embedders.model_registry import build_embedder

    try:
        eager = build_embedder("dinov2").load(
            device=device, dtype=torch.float32, compile=False
        )
        compiled = build_embedder("dinov2").load(
            device=device, dtype=torch.float32, compile=True
        )
    except Exception as exc:  # noqa: BLE001 - weights unavailable / offline
        pytest.skip(f"dinov2 weights unavailable ({type(exc).__name__})")

    rng = np.random.RandomState(0)
    isz = eager.spec.input_size
    patches = [rng.randint(0, 256, (isz, isz, 3), dtype=np.uint8) for _ in range(3)]
    tens = eager.transform_batch(patches, device)

    out_eager = eager.embed_batch(tens).numpy()
    out_compiled = compiled.embed_batch(tens).numpy()

    assert out_eager.shape == out_compiled.shape == (3, eager.spec.embedding_dim)
    assert np.isfinite(out_compiled).all()
    # Not bit-identical (compile reorders fp ops); must stay within tolerance.
    assert np.allclose(out_eager, out_compiled, rtol=1e-3, atol=1e-3), (
        f"max_abs_diff={np.max(np.abs(out_eager - out_compiled))}"
    )
