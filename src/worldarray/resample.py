"""Reading at another voxel size, axis by axis.

The target grid's voxel 0 starts at the source **corner** (or at an ``origin`` the caller
anchors elsewhere) and target voxel ``j`` is centred at ``corner + (j + 1/2) * t``, which is
source index ``p = origin + (j + 1/2) * t / s - 1/2``.

``"nearest"`` takes the source voxel whose extent holds the target centre, nearest neighbour
as ITK defines it; a centre on a voxel boundary (every even whole factor) belongs to the
upper voxel. ``"linear"`` takes the block mean when ``t / s`` is a whole number of 2 or more
and the grids are aligned (a block cut short by the end of the data is the mean of what it
has), else linear interpolation between the two neighbours, the border voxel repeated past
the first and last centre. Integer intensity dtypes are rounded back.

A target voxel outside the data (its centre past the data, or for a block mean a block that
overlaps none of it) is filled: with the padding value, or for ``"edge"`` with the border
voxel (nearest, linear) or the last block that holds data (mean). The source voxels read come from the target centres, grown by
the kernel radius, never from the box edges, so a target voxel's value depends only on its
index and the source array: two adjoining reads agree with one read across both (spec,
Behaviour 4).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .geometry import snap_integral

Method = str  # "nearest" | "linear"


@dataclass(frozen=True)
class AxisPlan:
    """How one spatial axis is resampled for target indices ``[jb, je)``."""

    lo: int  # source read range [lo, hi)
    hi: int
    inside: np.ndarray  # per target voxel: it holds data (see the module docstring)
    kind: str  # "same" | "nearest" | "linear" | "mean"
    index: np.ndarray | None = None  # nearest: source index per target voxel (clamped)
    lower: np.ndarray | None = None  # linear: lower neighbour (clamped), weight on the upper
    weight: np.ndarray | None = None
    start: np.ndarray | None = None  # mean: block [start, stop) per target voxel (clamped)
    stop: np.ndarray | None = None


def _clip_to_inside(j: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Target indices moved onto the nearest inside target voxel (edge padding)."""
    if not inside.any() or inside.all():
        return j
    held = j[inside]
    return np.clip(j, held.min(), held.max())


def plan_axis(
    n: int,
    s: float,
    t: float,
    jb: int,
    je: int,
    method: Method,
    origin: float = 0.0,
    edge: bool = False,
) -> AxisPlan:
    """``origin``: the target grid's voxel 0 corner, in source voxels from the source corner
    (0 when the two grids share a corner). ``edge``: outside target voxels repeat the
    nearest inside one instead of taking the fill."""
    j = np.arange(jb, je, dtype=float)
    ratio = t / s
    if je <= jb:
        return AxisPlan(0, 0, np.zeros(0, dtype=bool), "same")
    whole = snap_integral(ratio)
    aligned = float(snap_integral(origin)).is_integer()
    if method == "linear" and whole == np.round(whole) and whole >= 2 and aligned:
        f, o = int(whole), int(round(float(origin)))
        raw_start, raw_stop = o + j * f, o + (j + 1) * f
        inside = (raw_start < n) & (raw_stop > 0)  # the block overlaps the data at all
        if edge:
            jj = _clip_to_inside(j, inside)
            raw_start, raw_stop = o + jj * f, o + (jj + 1) * f
        start = np.clip(raw_start.astype(int), 0, n)
        stop = np.clip(raw_stop.astype(int), 0, n)
        lo = int(start[inside].min()) if inside.any() else 0
        hi = int(stop[inside].max()) if inside.any() else 0
        return AxisPlan(lo, hi, inside, "mean", start=start, stop=stop)

    centre = snap_integral(origin + (j + 0.5) * ratio)  # in source voxels from the corner
    inside = (centre >= 0) & (centre < n)
    # for nearest and linear, "edge" is the clamped source index below: the border voxel repeated
    if method == "nearest":
        # the source voxel whose extent [i, i + 1) holds the target centre: nearest neighbour as
        # ITK defines it; a centre on a boundary belongs to the upper voxel
        idx = np.clip(np.floor(centre), 0, n - 1).astype(int)
        lo, hi = _bounds(idx[inside] if not edge else idx, n)
        return AxisPlan(lo, hi, inside, "nearest", index=idx)
    if method == "linear":
        p = snap_integral(centre - 0.5)  # source index of the target centre
        pc = np.clip(p, 0, n - 1)  # border repeated past the first and last centre
        lower = np.floor(pc).astype(int)
        weight = pc - lower
        upper_needed = np.minimum(lower + 1, n - 1)
        used = np.ones_like(inside) if edge else inside
        lo, hi = _bounds(np.concatenate([lower[used], upper_needed[used]]), n)
        return AxisPlan(lo, hi, inside, "linear", lower=lower, weight=weight)
    raise ValueError(f"unknown method {method!r}; use 'nearest' or 'linear'")


def _bounds(indices: np.ndarray, n: int) -> tuple[int, int]:
    if indices.size == 0:
        return 0, 0
    return int(indices.min()), int(indices.max()) + 1


def apply_axis(block: np.ndarray, axis: int, plan: AxisPlan) -> np.ndarray:
    """``block`` (read over ``[plan.lo, plan.hi)`` along ``axis``) resampled along it."""
    if plan.kind == "same":
        return block
    if plan.kind == "nearest":
        idx = np.clip(plan.index - plan.lo, 0, max(block.shape[axis] - 1, 0))
        return np.take(block, idx, axis=axis)
    if plan.kind == "linear":
        last = max(block.shape[axis] - 1, 0)
        lower = np.clip(plan.lower - plan.lo, 0, last)
        upper = np.clip(plan.lower + 1 - plan.lo, 0, last)
        w = plan.weight.reshape([-1 if a == axis else 1 for a in range(block.ndim)])
        lo_v = np.take(block, lower, axis=axis).astype(np.float64)
        hi_v = np.take(block, upper, axis=axis).astype(np.float64)
        return lo_v * (1 - w) + hi_v * w
    if plan.kind == "mean":
        start = np.clip(plan.start - plan.lo, 0, block.shape[axis])
        stop = np.clip(plan.stop - plan.lo, 0, block.shape[axis])
        if (
            start.size
            and np.all(stop[:-1] == start[1:])
            and np.all(stop > start)
            and start[0] == 0
            and stop[-1] == block.shape[axis]
        ):  # the blocks tile the read: one pass, no cumulative sum
            sums = np.add.reduceat(block, start, axis=axis, dtype=np.float64)
            count = (stop - start).reshape([-1 if a == axis else 1 for a in range(block.ndim)])
            return sums / count
        cs = np.cumsum(block.astype(np.float64), axis=axis)
        zero = np.zeros_like(np.take(cs, [0], axis=axis))
        cs = np.concatenate([zero, cs], axis=axis)
        start = np.clip(plan.start - plan.lo, 0, block.shape[axis])
        stop = np.clip(plan.stop - plan.lo, 0, block.shape[axis])
        count = np.maximum(stop - start, 1).reshape(
            [-1 if a == axis else 1 for a in range(block.ndim)]
        )
        return (np.take(cs, stop, axis=axis) - np.take(cs, start, axis=axis)) / count
    raise ValueError(plan.kind)


def target_shape(n: int, s: float, t: float, origin: float = 0.0) -> int:
    """How many target voxels of size ``t`` cover ``n`` source voxels of size ``s``, counting
    from a target grid whose voxel 0 corner sits ``origin`` source voxels from the source
    corner: the first and last target voxels that overlap the data, inclusive."""
    ratio = t / s
    first = np.floor(snap_integral(-origin / ratio))
    last = np.ceil(snap_integral((n - origin) / ratio))
    return int(max(last - first, 0))


def resample(
    read_index: Callable[[Sequence[int], Sequence[int], object], np.ndarray],
    shape: Sequence[int],
    dtype: np.dtype,
    spatial: Sequence[int],
    source_voxel_size: Sequence[float],
    target_voxel_size: Sequence[float],
    begin: Sequence[int],
    end: Sequence[int],
    method: Method,
    padding,
    mask: bool,
    origins: Sequence[float] | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """The target voxels ``[begin, end)`` (target indices on the spatial axes, source indices
    elsewhere) of the source read through ``read_index(begin, end, padding)``. ``origins``:
    per spatial axis, the target grid's corner in source voxels from the source corner.
    ``dtype`` is what the output is cast to when the block read comes back as the stored
    dtype; a ``read_index`` that transforms its block decides the dtype instead."""
    ndim = len(shape)
    edge = padding == "edge"
    plans: dict[int, AxisPlan] = {}
    read_begin, read_end = list(begin), list(end)
    for k, axis in enumerate(spatial):
        origin = float(origins[k]) if origins is not None else 0.0
        plan = plan_axis(
            shape[axis],
            source_voxel_size[k],
            target_voxel_size[k],
            begin[axis],
            end[axis],
            method,
            origin,
            edge,
        )
        plans[axis] = plan
        read_begin[axis], read_end[axis] = plan.lo, plan.hi

    out_shape = tuple(int(e - b) for b, e in zip(begin, end))
    inside = np.ones(out_shape, dtype=bool)
    for axis, plan in plans.items():
        inside &= plan.inside.reshape([-1 if a == axis else 1 for a in range(ndim)])

    if not inside.any() or any(read_end[a] <= read_begin[a] for a in plans):
        fill = 0 if edge else padding
        out = np.full(out_shape, fill, dtype=dtype)
        return (out, inside) if mask else out

    # the spatial axes are read exactly; the other axes pad as any index read does
    block = read_index(read_begin, read_end, padding)
    out_dtype = block.dtype if block.dtype != dtype else dtype
    out = block
    for axis in sorted(plans, key=lambda a: plans[a].kind != "mean"):
        out = apply_axis(out, axis, plans[axis])
    if np.issubdtype(out_dtype, np.integer) or np.issubdtype(out_dtype, np.bool_):
        out = np.rint(out) if out.dtype.kind == "f" else out
        if np.issubdtype(out_dtype, np.integer):
            info = np.iinfo(out_dtype)
            out = np.clip(out, info.min, info.max)
    out = np.ascontiguousarray(out.astype(out_dtype, copy=False))
    if not edge and not inside.all():
        out = out.copy()
        out[~inside] = padding
    return (out, inside) if mask else out
