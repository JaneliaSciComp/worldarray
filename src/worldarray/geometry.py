"""Boxes, and the one rule that maps a world box to voxel indices.

A world box ``[begin, end)`` is half-open per axis. A voxel is read iff its **centre**
lies in the box: with voxel size ``s`` and translation ``t`` (the centre of voxel 0), the
voxels are ``ceil((begin - t) / s) <= i < ceil((end - t) / s)``, after float noise is
snapped away. A box aligned to voxel corners (every blockwise block) and a box aligned to
voxel centres (mesh-n-bone's) both map to whole voxels with nothing rounded
(spec, Behaviour 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

SNAP_TOL = 1e-6  # of a voxel: a quotient this close to a whole number is that whole number


def snap_integral(values, tol: float = SNAP_TOL) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    rounded = np.round(arr)
    return np.where(np.abs(arr - rounded) <= tol, rounded, arr)


@dataclass(frozen=True)
class Box:
    """``[begin, end)`` per axis, in world units (floats) or voxels (whole floats)."""

    begin: tuple[float, ...]
    end: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.begin) != len(self.end):
            raise ValueError(f"begin and end differ in length: {self.begin} vs {self.end}")
        object.__setattr__(self, "begin", tuple(float(b) for b in self.begin))
        object.__setattr__(self, "end", tuple(float(e) for e in self.end))

    @property
    def shape(self) -> tuple[float, ...]:
        return tuple(e - b for b, e in zip(self.begin, self.end))

    @property
    def ndim(self) -> int:
        return len(self.begin)

    @classmethod
    def coerce(cls, box: Any) -> Box:
        """A ``Box`` from a ``(begin, end)`` pair, anything with ``.begin``/``.end`` or
        ``.offset``/``.shape`` (a funlib ``Roi``, duck-typed), or a ``Box``."""
        if isinstance(box, Box):
            return box
        if hasattr(box, "begin") and hasattr(box, "end"):
            return cls(tuple(box.begin), tuple(box.end))
        if hasattr(box, "offset") and hasattr(box, "shape"):
            offset, shape = tuple(box.offset), tuple(box.shape)
            return cls(offset, tuple(o + s for o, s in zip(offset, shape)))
        try:
            begin, end = box
            return cls(tuple(begin), tuple(end))
        except (TypeError, ValueError):
            raise TypeError(
                "a box is a (begin, end) pair, an object with .begin/.end or .offset/.shape, or a Box"
            ) from None


def index_range(begin, end, voxel_size, translation) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The half-open voxel index range whose centres lie in ``[begin, end)``."""
    s = np.asarray(voxel_size, dtype=float)
    t = np.asarray(translation, dtype=float)
    lo = np.ceil(snap_integral((np.asarray(begin, dtype=float) - t) / s))
    hi = np.ceil(snap_integral((np.asarray(end, dtype=float) - t) / s))
    hi = np.maximum(hi, lo)
    return tuple(int(v) for v in lo), tuple(int(v) for v in hi)


def is_aligned(begin, end, voxel_size, translation) -> bool:
    """Whether every edge sits on a voxel corner or a voxel centre."""
    s = np.asarray(voxel_size, dtype=float)
    t = np.asarray(translation, dtype=float)
    for edges in (begin, end):
        q = (np.asarray(edges, dtype=float) - t) / s
        twice = snap_integral(2 * q)
        if not np.all(twice == np.round(twice)):
            return False
    return True
