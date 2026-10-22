"""worldarray: chunked arrays in world coordinates, through tensorstore.

Open zarr v2, zarr v3, N5 and Neuroglancer precomputed arrays and pyramids by path, know
where each sits (voxel size, the centre of voxel 0, units, axes), and read voxel boxes or
world boxes as numpy arrays, padded where the box leaves the data. ``docs/spec.md`` is the
contract; the README is its first five sections.
"""

from __future__ import annotations

from . import store, units
from .array import Pyramid, WorldArray, metadata, open, open_pyramid, open_tensorstore
from .context import capped_context, capped_context_spec, shared_context
from .geometry import Box, index_range
from .grid import Block, Grid, blocks, level_factors
from .levels import select_level
from .placement import ArrayMeta, corner_to_centre, default_axes
from .retry import Retry
from .store import Node, detect, list_chunk_keys

__version__ = "0.1.1"

__all__ = [
    "ArrayMeta",
    "Block",
    "Box",
    "Grid",
    "Node",
    "blocks",
    "corner_to_centre",
    "default_axes",
    "detect",
    "index_range",
    "level_factors",
    "list_chunk_keys",
    "Pyramid",
    "Retry",
    "WorldArray",
    "capped_context",
    "capped_context_spec",
    "metadata",
    "open",
    "open_pyramid",
    "open_tensorstore",
    "select_level",
    "shared_context",
    "store",
    "units",
]
