"""worldarray: chunked arrays in world coordinates, through tensorstore.

Open zarr v2, zarr v3, N5 and Neuroglancer precomputed arrays and pyramids by path, know
where each sits (voxel size, the centre of voxel 0, units, axes), and read voxel boxes or
world boxes as numpy arrays, padded where the box leaves the data. ``docs/spec.md`` is the
contract; the README is its first five sections.
"""

from __future__ import annotations

import warnings

from .array import Pyramid, WorldArray, metadata, open, open_pyramid, open_tensorstore
from .context import capped_context, capped_context_spec, shared_context
from .geometry import Box
from .levels import select_level
from .metadata import ArrayMeta
from .retry import Retry

__version__ = "0.1.0a0"

__all__ = [
    "ArrayMeta",
    "Box",
    "ImageDataInterface",
    "Pyramid",
    "Retry",
    "WorldArray",
    "capped_context",
    "capped_context_spec",
    "metadata",
    "open",
    "open_ds_tensorstore",
    "open_pyramid",
    "open_tensorstore",
    "select_level",
    "shared_context",
]


class ImageDataInterface(WorldArray):
    """``WorldArray`` under the name every migrating repo imports. Goes away in 0.2."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ImageDataInterface is WorldArray; import that name instead (the alias goes away in 0.2)",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


def open_ds_tensorstore(path: str, **kwargs):
    """``open_tensorstore`` under the name mesh-n-bone and cellmap-analyze used."""
    warnings.warn(
        "open_ds_tensorstore is open_tensorstore (the alias goes away in 0.2)",
        DeprecationWarning,
        stacklevel=2,
    )
    return open_tensorstore(path, **kwargs)
