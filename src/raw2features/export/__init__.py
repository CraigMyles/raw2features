"""Post-hoc exporters: convert a raw2features embedding store into interop formats.

Exporters read an existing ``<slide_id>.embeddings.zarr`` and emit a sibling
artifact in another ecosystem's format. They never recompute embeddings and
keep their heavy, ecosystem-specific dependencies optional, so the core install
stays light. The first exporter targets the scverse **SpatialData** format.
"""

from __future__ import annotations
