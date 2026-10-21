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

    @property
    def centre(self) -> tuple[float, ...]:
        return tuple((b + e) / 2 for b, e in zip(self.begin, self.end))

    @property
    def empty(self) -> bool:
        return any(e <= b for b, e in zip(self.begin, self.end))

    def intersect(self, other) -> Box:
        """The overlap of two boxes; a box of zero extent where there is none."""
        other = Box.coerce(other)
        begin = tuple(max(a, b) for a, b in zip(self.begin, other.begin))
        end = tuple(max(bg, min(a, b)) for bg, a, b in zip(begin, self.end, other.end))
        return Box(begin, end)

    def union(self, other) -> Box:
        """The smallest box holding both."""
        other = Box.coerce(other)
        return Box(
            tuple(min(a, b) for a, b in zip(self.begin, other.begin)),
            tuple(max(a, b) for a, b in zip(self.end, other.end)),
        )

    def contains(self, other) -> bool:
        """Whether a point (one coordinate per axis) lies in the box, or another box lies
        within it. Half-open, so a point on the far edge is outside."""
        if isinstance(other, Box) or hasattr(other, "begin") or hasattr(other, "offset"):
            o = Box.coerce(other)
            return all(
                b <= ob and oe <= e for b, e, ob, oe in zip(self.begin, self.end, o.begin, o.end)
            )
        return all(b <= p < e for b, e, p in zip(self.begin, self.end, other))

    def grow(self, before, after=None) -> Box:
        """Extended by ``before`` on the low side and ``after`` (default ``before``) on the
        high side, a scalar or one value per axis; negative values shrink."""
        before = self._per_axis(before)
        after = self._per_axis(before if after is None else after)
        return Box(
            tuple(b - x for b, x in zip(self.begin, before)),
            tuple(e + x for e, x in zip(self.end, after)),
        )

    def shift(self, by) -> Box:
        by = self._per_axis(by)
        return Box(
            tuple(b + x for b, x in zip(self.begin, by)), tuple(e + x for e, x in zip(self.end, by))
        )

    def snap(self, voxel_size, translation, mode: str = "grow") -> Box:
        """Aligned to the voxel corners of a grid whose voxel 0 is centred at
        ``translation``: ``grow`` rounds outward, ``shrink`` inward, ``closest`` to the
        nearest corner. After ``grow`` the box reads exactly the voxels it touches."""
        vs = np.asarray(voxel_size, dtype=float)
        corner0 = np.asarray(translation, dtype=float) - vs / 2
        qb = snap_integral((np.asarray(self.begin) - corner0) / vs)
        qe = snap_integral((np.asarray(self.end) - corner0) / vs)
        if mode == "grow":
            qb, qe = np.floor(qb), np.ceil(qe)
        elif mode == "shrink":
            qb, qe = np.ceil(qb), np.floor(qe)
        elif mode == "closest":
            qb, qe = np.round(qb), np.round(qe)
        else:
            raise ValueError("mode is 'grow', 'shrink' or 'closest'")
        return Box(tuple(corner0 + qb * vs), tuple(corner0 + np.maximum(qe, qb) * vs))

    def to_index(self, voxel_size, translation) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """The voxels whose centres lie in the box (the centre rule)."""
        return index_range(self.begin, self.end, voxel_size, translation)

    @classmethod
    def from_index(cls, begin, end, voxel_size, translation) -> Box:
        """The world box exactly covering voxels ``[begin, end)`` of a grid."""
        vs = np.asarray(voxel_size, dtype=float)
        corner0 = np.asarray(translation, dtype=float) - vs / 2
        return cls(tuple(corner0 + np.asarray(begin) * vs), tuple(corner0 + np.asarray(end) * vs))

    def _per_axis(self, value) -> tuple[float, ...]:
        if np.ndim(value) == 0:
            return (float(value),) * self.ndim
        value = tuple(float(v) for v in value)
        if len(value) != self.ndim:
            raise ValueError(f"expected {self.ndim} values, got {value}")
        return value

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
