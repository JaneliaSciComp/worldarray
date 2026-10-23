"""worldarray: chunked arrays in world coordinates, through tensorstore.

Open zarr v2, zarr v3, N5 and Neuroglancer precomputed arrays and pyramids by path, know
where each sits (voxel size, the centre of voxel 0, units, axes), and read voxel boxes or
world boxes as numpy arrays, padded where the box leaves the data; create placed OME-NGFF
levels and write them from any number of jobs; plan an output grid and cut it into blocks
before anything exists. ``docs/spec.md`` is the contract; the README is its first five
sections.
"""

from __future__ import annotations

from . import grid, store, units, write
from .array import Pyramid, WorldArray, create, metadata, open, open_pyramid, open_tensorstore
from .context import capped_context, capped_context_spec, shared_context, thread_budget
from .geometry import Box, index_range
from .grid import Block, Grid, blocks, level_factors
from .levels import select_level
from .placement import ArrayMeta, corner_to_centre, default_axes
from .retry import Retry
from .store import Node, detect, list_chunk_keys
from .write import (
    chunk_key,
    ensure_group_entry,
    level_placement,
    level_shape,
    ome_multiscales,
    parse_chunk_key,
)

__version__ = "0.5.0"

__all__ = [
    "ArrayMeta",
    "Block",
    "Box",
    "Grid",
    "Node",
    "blocks",
    "chunk_key",
    "corner_to_centre",
    "create",
    "default_axes",
    "detect",
    "ensure_group_entry",
    "grid",
    "index_range",
    "level_factors",
    "level_placement",
    "level_shape",
    "list_chunk_keys",
    "ome_multiscales",
    "parse_chunk_key",
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
    "thread_budget",
    "store",
    "units",
    "write",
]
