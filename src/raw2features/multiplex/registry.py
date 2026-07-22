"""Build multiplex strategies through the public plugin seam."""

from __future__ import annotations

from raw2features.core import plugins

from .base import MultiplexStrategy


def build_strategy(name: str) -> MultiplexStrategy:
    """Instantiate a registered multiplex strategy by name."""

    cls = plugins.get("multiplex_strategies", name)
    strategy = cls()
    if not isinstance(strategy, MultiplexStrategy):
        raise TypeError(
            f"multiplex strategy {name!r} returned {type(strategy).__name__}, "
            "not a MultiplexStrategy"
        )
    return strategy
