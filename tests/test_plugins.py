"""The supported implementation-plugin discovery contract."""

from __future__ import annotations

import pytest

from raw2features.core import plugins


class _EntryPoint:
    def __init__(self, name, value=None, error: Exception | None = None):
        self.name = name
        self._value = value
        self._error = error

    def load(self):
        if self._error is not None:
            raise self._error
        return self._value


@pytest.fixture
def isolated_plugins(monkeypatch):
    registry = {seam: dict(values) for seam, values in plugins._REGISTRY.items()}
    monkeypatch.setattr(plugins, "_REGISTRY", registry)

    def install(*entry_points):
        monkeypatch.setattr(
            plugins,
            "entry_points",
            lambda *, group: list(entry_points),
        )

    return install


def test_importable_entry_point_is_available_and_resolved(isolated_plugins):
    implementation = object()
    isolated_plugins(_EntryPoint("third_party_reader", implementation))

    assert "third_party_reader" in plugins.available("readers")
    assert plugins.get("readers", "third_party_reader") is implementation


def test_entry_point_load_failure_is_skipped(isolated_plugins):
    isolated_plugins(_EntryPoint("broken_reader", error=ImportError("optional dep")))

    assert "broken_reader" not in plugins.available("readers")
    with pytest.raises(KeyError, match="broken_reader.*not found"):
        plugins.get("readers", "broken_reader")


def test_in_process_registration_precedes_same_named_entry_point(isolated_plugins):
    local = object()
    external = object()
    plugins.register("readers", "precedence_probe")(local)
    isolated_plugins(_EntryPoint("precedence_probe", external))

    assert plugins.get("readers", "precedence_probe") is local
