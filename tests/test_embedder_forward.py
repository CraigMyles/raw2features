"""Real-weights embedder forward pass (marked slow: downloads weights).

Set ``R2F_TEST_STRICT_WEIGHTS=1`` on an intentionally provisioned runner to
turn model-load errors into test failures.  The default remains skip-on-unavailable
so gated models, offline development, and missing optional dependencies stay usable.
"""

from __future__ import annotations

import os
import sys

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


def _strict_weights_enabled() -> bool:
    """Whether provisioned real-weight runs should fail instead of skip."""
    value = os.environ.get("R2F_TEST_STRICT_WEIGHTS", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_embedder_for_validation(name: str, device: str):
    """Load ``name``, preserving the historical skip policy unless strict."""
    try:
        return build_embedder(name).load(device=device, dtype=torch.float32)
    except Exception as exc:  # noqa: BLE001 - gated weights / no token / offline
        if _strict_weights_enabled():
            raise
        pytest.skip(f"{name}: weights unavailable ({type(exc).__name__})")


@pytest.mark.slow
@pytest.mark.parametrize("name", _brightfield_models())
def test_forward_output_matches_declared_dim(name):
    """Each brightfield model's forward output must be (B, embedding_dim) and finite.

    This is the registry's dimension contract under real weights. Gated models
    skip cleanly when their weights aren't available (no token / offline).
    """
    spec = get_spec(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb = _load_embedder_for_validation(name, device)
    rng = np.random.RandomState(0)
    patch = rng.randint(0, 256, (spec.input_size, spec.input_size, 3), dtype=np.uint8)
    batch = torch.stack([emb.transform(patch) for _ in range(2)])
    out = emb.embed_batch(batch)
    assert tuple(out.shape) == (2, spec.embedding_dim)
    assert bool(torch.isfinite(out).all())


class _BrokenEmbedder:
    def load(self, **_kwargs):
        raise RuntimeError("deliberate load failure")


@pytest.mark.parametrize(
    ("strict_value", "expected_error", "message"),
    [
        (None, pytest.skip.Exception, "weights unavailable"),
        ("1", RuntimeError, "deliberate load failure"),
        ("true", RuntimeError, "deliberate load failure"),
    ],
)
def test_model_load_failure_policy(
    monkeypatch, strict_value, expected_error, message
):
    """Loader failures skip by default and surface when strict mode is enabled."""
    if strict_value is None:
        monkeypatch.delenv("R2F_TEST_STRICT_WEIGHTS", raising=False)
    else:
        monkeypatch.setenv("R2F_TEST_STRICT_WEIGHTS", strict_value)
    monkeypatch.setattr(
        sys.modules[__name__], "build_embedder", lambda _name: _BrokenEmbedder()
    )

    with pytest.raises(expected_error, match=message):
        _load_embedder_for_validation("broken", "cpu")
