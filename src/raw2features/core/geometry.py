"""Small geometry primitives shared across seams.

Coordinate conventions (important, and asserted throughout):

* A patch ``location`` is always given in **level-0 pixel coordinates** (the
  OpenSlide / CLAM convention). This makes embeddings losslessly relocatable to
  the source slide regardless of which pyramid level was actually read.
* A ``Region.size`` is in the pixels of ``Region.level`` (the level being read).
"""

from __future__ import annotations

from typing import NamedTuple


class Point(NamedTuple):
    """A 2D point. ``x`` is column, ``y`` is row."""

    x: int
    y: int


class Size(NamedTuple):
    """A 2D size. ``width`` is columns, ``height`` is rows."""

    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height


class Region(NamedTuple):
    """A rectangular read request.

    Mirrors ``openslide.read_region``: ``location`` is in level-0 pixels,
    ``size`` is in ``level`` pixels.
    """

    level: int
    location: Point  # level-0 (x, y) of the top-left corner
    size: Size  # (w, h) in `level` pixels

    @classmethod
    def patch(cls, x: int, y: int, size: int, level: int) -> Region:
        """A square patch at level-0 ``(x, y)`` of ``size`` px on ``level``."""
        return cls(level=level, location=Point(x, y), size=Size(size, size))
