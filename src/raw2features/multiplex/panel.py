"""Deterministic selection of named channels from a positional multiplex panel."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from typing import Any


def marker_name_identity(value: Any) -> str:
    """Return the user-facing marker identity used by ``--marker`` selection.

    This is deliberately less biological than a model vocabulary resolver.  It only
    makes metadata spelling comparisons Unicode-, whitespace-, and case-insensitive;
    each native model remains responsible for mapping the selected source labels to
    its own canonical marker identities.
    """

    return unicodedata.normalize("NFKC", str(value)).strip().casefold()


def resolve_marker_selection(
    channel_names: Sequence[str | None],
    requested_markers: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Resolve an ordered marker request to physical C-axis positions.

    With no explicit request every physical position is retained, including unnamed
    slots.  Keeping those slots is important because model-specific panel binding may
    drop them later, but their positions still define the source array.  An explicit
    request must identify each channel uniquely after :func:`marker_name_identity`.
    The returned order is the requested order, not source order.
    """

    panel = ["" if value is None else str(value) for value in channel_names]
    requested = [str(value).strip() for value in (requested_markers or [])]
    if any(not value for value in requested):
        raise ValueError("selected multiplex marker names must be non-empty")

    if not requested:
        return [
            {"source_index": index, "source_name": source_name}
            for index, source_name in enumerate(panel)
        ]

    positions: dict[str, list[int]] = {}
    for index, source_name in enumerate(panel):
        identity = marker_name_identity(source_name)
        if identity:
            positions.setdefault(identity, []).append(index)

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for requested_name in requested:
        identity = marker_name_identity(requested_name)
        if identity in seen:
            raise ValueError(
                f"multiplex marker {requested_name!r} was requested more than once"
            )
        seen.add(identity)
        matches = positions.get(identity, [])
        if not matches:
            raise ValueError(
                f"requested multiplex marker {requested_name!r} is not present"
            )
        if len(matches) != 1:
            raise ValueError(
                f"requested multiplex marker {requested_name!r} is ambiguous; "
                f"it matches source channel indices {matches}"
            )
        index = matches[0]
        records.append({"source_index": index, "source_name": panel[index]})
    return records
