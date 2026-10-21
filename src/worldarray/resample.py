"""Reading at another voxel size, axis by axis.

The target grid's voxel 0 starts at the source **corner** and has ``ceil(n * s / t)`` voxels
on an axis with ``n`` source voxels of size ``s`` and target size ``t``; target voxel ``j``
is centred at ``corner + (j + 1/2) * t``, which is source index ``p = (j + 1/2) * t / s - 1/2``.

``"nearest"`` takes the source voxel whose centre is nearest ``p``; on a tie (every even
whole factor) the lower index. ``"linear"`` takes the block mean when ``t / s`` is a whole
number of 2 or more, else linear interpolation between the two neighbours, the border voxel
repeated past the first and last centre. Integer intensity dtypes are rounded back.

The source voxels read come from the first and last **target centres**, grown by the kernel
radius, never from the box edges, so a target voxel's value depends only on its index and
the source array: two adjoining reads agree with one read across both (spec, Behaviour 4).
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
    inside: np.ndarray  # per target voxel: its centre lies inside the data
    kind: str  # "same" | "nearest" | "linear" | "mean"
    index: np.ndarray | None = None  # nearest: source index per target voxel (clamped)
    lower: np.ndarray | None = None  # linear: lower neighbour (clamped), weight on the upper
    weight: np.ndarray | None = None
    start: np.ndarray | None = None  # mean: block [start, stop) per target voxel (clamped)
    stop: np.ndarray | None = None


def plan_axis(n: int, s: float, t: float, jb: int, je: int, method: Method) -> AxisPlan:
    j = np.arange(jb, je, dtype=float)
    ratio = t / s
    centre = snap_integral((j + 0.5) * ratio)  # target centre, in source voxels from the corner
    inside = (centre >= 0) & (centre < n)
    if je <= jb:
        return AxisPlan(0, 0, inside, "same")
    if np.isclose(ratio, 1.0, rtol=1e-9):
        idx = np.clip(np.round(centre - 0.5), 0, n - 1).astype(int)
        lo, hi = _bounds(idx[inside], n)
        return AxisPlan(lo, hi, inside, "nearest", index=idx)
    whole = snap_integral(ratio)
    if method == "linear" and whole == np.round(whole) and whole >= 2:
        f = int(whole)
        start = np.clip((j * f).astype(int), 0, n)
        stop = np.clip(((j + 1) * f).astype(int), 0, n)
        lo = int(start[inside].min()) if inside.any() else 0
        hi = int(stop[inside].max()) if inside.any() else 0
        return AxisPlan(lo, hi, inside, "mean", start=start, stop=stop)
    p = snap_integral(centre - 0.5)  # source index of the target centre
    if method == "nearest":
        idx = np.clip(np.ceil(snap_integral(p - 0.5)), 0, n - 1).astype(
            int
        )  # ties to the lower index
        lo, hi = _bounds(idx[inside], n)
        return AxisPlan(lo, hi, inside, "nearest", index=idx)
    if method == "linear":
        pc = np.clip(p, 0, n - 1)  # border repeated past the first and last centre
        lower = np.floor(pc).astype(int)
        weight = pc - lower
        upper_needed = np.minimum(lower + 1, n - 1)
        lo, hi = _bounds(np.concatenate([lower[inside], upper_needed[inside]]), n)
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
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """The target voxels ``[begin, end)`` (target indices on the spatial axes, source indices
    elsewhere) of the source read through ``read_index(begin, end, padding)``."""
    ndim = len(shape)
    plans: dict[int, AxisPlan] = {}
    read_begin, read_end = list(begin), list(end)
    for k, axis in enumerate(spatial):
        plan = plan_axis(
            shape[axis], source_voxel_size[k], target_voxel_size[k], begin[axis], end[axis], method
        )
        plans[axis] = plan
        read_begin[axis], read_end[axis] = plan.lo, plan.hi

    out_shape = tuple(int(e - b) for b, e in zip(begin, end))
    inside = np.ones(out_shape, dtype=bool)
    for axis, plan in plans.items():
        inside &= plan.inside.reshape([-1 if a == axis else 1 for a in range(ndim)])

    if not inside.any() or any(read_end[a] <= read_begin[a] for a in plans):
        fill = 0 if padding == "edge" else padding
        out = np.full(out_shape, fill, dtype=dtype)
        return (out, inside) if mask else out

    # the spatial axes are read exactly; the other axes pad as any index read does
    block = read_index(read_begin, read_end, padding if padding != "edge" else "edge")
    out = block
    for axis in sorted(plans, key=lambda a: plans[a].kind != "mean"):
        out = apply_axis(out, axis, plans[axis])
    if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
        out = np.rint(out) if out.dtype.kind == "f" else out
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            out = np.clip(out, info.min, info.max)
    out = np.ascontiguousarray(out.astype(dtype, copy=False))
    if padding != "edge" and not inside.all():
        out = out.copy()
        out[~inside] = padding
    return (out, inside) if mask else out
