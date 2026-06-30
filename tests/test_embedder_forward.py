"""Real-weights embedder forward pass (marked slow: downloads weights)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from raw2features.embedders.model_registry import (  # noqa: E402
    build_embedder,
    get_spec,
    load_registry,
)


def _brightfield_models() -> list[str]:
    """Registry models that take an RGB patch. Multiplex models (e.g. kronos) need a
    per-slide marker panel + N-channel input, so their forward is covered by
    test_multiplex.py and they're excluded here rather than parametrized-and-skipped."""
    return [n for n in sorted(load_registry()) if get_spec(n).modality == "brightfield"]


@pytest.mark.slow
@pytest.mark.parametrize("name", _brightfield_models())
def test_forward_output_matches_declared_dim(name):
    """Each brightfield model's forward output must be (B, embedding_dim) and finite.

    This is the registry's dimension contract under real weights. Gated models
    skip cleanly when their weights aren't available (no token / offline).
    """
    spec = get_spec(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        emb = build_embedder(name).load(device=device, dtype=torch.float32)
    except Exception as exc:  # noqa: BLE001 - gated weights / no token / offline
        pytest.skip(f"{name}: weights unavailable ({type(exc).__name__})")
    rng = np.random.RandomState(0)
    patch = rng.randint(0, 256, (spec.input_size, spec.input_size, 3), dtype=np.uint8)
    batch = torch.stack([emb.transform(patch) for _ in range(2)])
    out = emb.embed_batch(batch)
    assert tuple(out.shape) == (2, spec.embedding_dim)
    assert bool(torch.isfinite(out).all())
