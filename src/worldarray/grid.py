"""Grids and blocks: an output planned, and cut into jobs, before anything is stored.

``blocks(shape, block_shape, context, box, clip)`` tiles ``[0, shape)``, or a voxel
``box`` with the block grid anchored at its begin, into write boxes in C order with no
overlap and no gap, the last block on an axis clipped. Each block's read box is its write
box grown by ``context``. ``Grid`` is a ``WorldArray`` without a store: the same ``box``,
``corner``, ``index_box``, ``chunks_covering``, ``write_shape`` and ``blocks`` from the
facts alone. ``level_factors`` is the pyramid schedule mesh-n-bone and cellmap-analyze
share. Nothing here schedules or runs anything (spec, Behaviours 17 to 19).

Choices the spec leaves open, made here:

- ``box`` may reach past the array on any side (a ROI rounded outward to a factor
  multiple). The write boxes tile ``box`` exactly, outside the array too; ``clip=True``
  clips the **read** boxes to the array, ``[0, shape)``, never to ``box`` and never a write
  box. A write box wholly outside the array then has an empty read box at the array's
  nearest face. ``clip=False`` leaves read boxes unclipped, negative begins included, for a
  padded ``read_index``.
- ``context`` is one int, one per axis, or a ``(before, after)`` pair per axis; the per-axis
  entries may mix ints and pairs. Negative context raises.
- ``Blocks`` is a plain frozen dataclass of the five arguments, normalised: ``box`` to a
  pair of int tuples (``[0, shape)`` when none is given), ``context`` to one
  ``(before, after)`` pair per axis. Blocks are computed from their id on demand, so
  ``blocks[id]`` is O(1) and the record pickles to a worker without the blocks. Ids run from
  0 to ``len - 1``; any other id is an ``IndexError``. ``index_of(begin)`` takes a begin on
  the box-anchored grid.
- ``Block.at_start`` and ``Block.at_end`` say, per axis, whether the block is the first or
  last on the block grid (the box's edge, or the array's). ``Block.inner`` is the write box
  inside the read box, ``slice(begin - read_begin, end - read_begin)`` per axis, which stays
  a correct offset into a padded read when ``clip=False`` lets ``read_begin`` go below 0.
- ``Grid`` keeps ``voxel_size`` and ``translation`` with one entry per axis, as
  ``ArrayMeta`` does; given over the spatial axes only, the other axes get 1.0 and 0.0 (the
  short-list rule of Convention 4). ``units`` is kept per axis too: a string is the spatial
  axes' unit and every other axis gets ``""``; a dict by kind gives ``"space"`` to the
  spatial axes and ``"time"`` to an axis named ``t``, ``""`` elsewhere; a sequence is per
  axis. ``shards`` need ``chunks`` and must be a whole number of them on every axis.
- ``Grid.chunks_covering(begin, end, write=False)`` takes ``write`` as ``WorldArray``'s
  does in 0.2: ``write=True`` runs on ``write_shape``. A grid without that shape raises.
- ``Grid.meta(dtype)`` has ``format`` None, ``path`` ``""``, ``parent`` None, empty
  ``attrs`` and ``group_attrs``, ``chunk_shape`` the chunks (the shape when none),
  ``axes_named`` True (a created array names its axes), ``placed_by`` ``"ome"`` and
  ``fill_value`` None: a grid has no store, no documents and no fill.
- ``Grid.from_meta(meta)`` keeps the array's chunks (and its ``shard_shape`` when the meta
  carries one); the target grid at ``voxel_size`` keeps neither, since its chunks are the
  new array's to choose. ``voxel_size`` is in the meta's units.
- ``level_factors`` compares voxel sizes with ``rtol 1e-6``, as Behaviour 3 does. The stop
  rule is checked before each level, so a ``shape`` already within ``stop_shape`` gives no
  levels and the last level returned is the first that fits. ``ratio`` below 1 raises,
  since no axis would ever qualify.

The ``WorldArray`` side of 0.3 (``vol.blocks``, ``vol.write_chunk`` and
``wa.create(path, grid=g)``) is wired in with the write module of 0.2; nothing here opens a
store.
"""

from __future__ import annotations

import itertools
import math
import operator
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from .geometry import Box, index_range
from .placement import ArrayMeta, default_axes, spatial_axes
from .resample import target_shape

RTOL = 1e-6


def _whole(values, what: str) -> tuple[int, ...]:
    values = tuple(values)
    out = tuple(int(v) for v in values)
    if any(o != v for o, v in zip(out, values)):
        raise ValueError(f"{what} must be whole numbers, got {values}")
    return out


def _sized(values, what: str, ndim: int, minimum: int) -> tuple[int, ...]:
    out = _whole(values, what)
    if len(out) != ndim:
        raise ValueError(f"{what} has {len(out)} entries; expected {ndim}")
    if any(v < minimum for v in out):
        raise ValueError(f"{what} entries must be at least {minimum}, got {out}")
    return out


def _context(context, ndim: int) -> tuple[tuple[int, int], ...]:
    """One ``(before, after)`` pair per axis from an int, one int per axis, or a pair per axis."""
    if not isinstance(context, Iterable):
        entries = [context] * ndim
    else:
        entries = list(context)
        if len(entries) != ndim:
            raise ValueError(f"context has {len(entries)} entries; the blocks have {ndim} axes")
    pairs = []
    for e in entries:
        pair = _whole((e, e) if not isinstance(e, Iterable) else e, "context")
        if len(pair) != 2:
            raise ValueError(f"a context entry is an int or a (before, after) pair, got {e!r}")
        if min(pair) < 0:
            raise ValueError(f"context cannot be negative, got {e!r}")
        pairs.append(pair)
    return tuple(pairs)


# ---------------------------------------------------------------------------
# Blocks


@dataclass(frozen=True)
class Block:
    """One block: a plain, picklable record. ``begin``/``end`` is the write box,
    ``read_begin``/``read_end`` the write box grown by the context (clipped to the array
    when the blocks clip)."""

    index: tuple[int, ...]  # position on the block grid
    id: int  # linear C-order id
    begin: tuple[int, ...]
    end: tuple[int, ...]
    read_begin: tuple[int, ...]
    read_end: tuple[int, ...]
    at_start: tuple[bool, ...]  # per axis: the first block on the grid
    at_end: tuple[bool, ...]  # per axis: the last block on the grid

    @property
    def inner(self) -> tuple[slice, ...]:
        """The write box inside the read box: ``data[block.inner]`` of a read of
        ``[read_begin, read_end)`` is the block's own voxels."""
        return tuple(
            slice(b - rb, e - rb) for b, e, rb in zip(self.begin, self.end, self.read_begin)
        )


@dataclass(frozen=True)
class Blocks:
    """The blocks tiling ``[0, shape)``, or ``box``, in C order. A frozen record of its
    arguments that computes each block on demand: ``len``, iteration, ``blocks[id]`` (O(1))
    and ``index_of(begin)``."""

    shape: tuple[int, ...]
    block_shape: tuple[int, ...]
    context: tuple[tuple[int, int], ...] = 0  # normalised to (before, after) per axis
    box: tuple[tuple[int, ...], tuple[int, ...]] | None = None  # normalised; [0, shape) if None
    clip: bool = True

    def __post_init__(self) -> None:
        shape = _whole(self.shape, "shape")
        if any(s < 0 for s in shape):
            raise ValueError(f"shape cannot be negative, got {shape}")
        n = len(shape)
        block_shape = _sized(self.block_shape, "block_shape", n, 1)
        if self.box is None:
            box = ((0,) * n, shape)
        else:
            b = Box.coerce(self.box)
            if b.ndim != n:
                raise ValueError(f"box has {b.ndim} axes; the blocks have {n}")
            box = (_whole(b.begin, "box"), _whole(b.end, "box"))
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "block_shape", block_shape)
        object.__setattr__(self, "context", _context(self.context, n))
        object.__setattr__(self, "box", box)
        object.__setattr__(self, "clip", bool(self.clip))

    @property
    def _counts(self) -> tuple[int, ...]:
        """Blocks per axis."""
        return tuple(-(-max(e - b, 0) // s) for b, e, s in zip(*self.box, self.block_shape))

    @property
    def count(self) -> int:
        return math.prod(self._counts)

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[Block]:
        counts = self._counts
        for i, index in enumerate(itertools.product(*(range(c) for c in counts))):
            yield self._block(index, i, counts)

    def __getitem__(self, block_id: int) -> Block:
        i = operator.index(block_id)
        counts = self._counts
        total = math.prod(counts)
        if not 0 <= i < total:
            raise IndexError(f"block id {i} is outside [0, {total})")
        index, rest = [], i
        for c in reversed(counts):
            rest, k = divmod(rest, c)
            index.append(k)
        return self._block(tuple(reversed(index)), i, counts)

    def index_of(self, begin: Sequence[int]) -> tuple[int, ...]:
        """The block index of the block whose write box starts at ``begin``."""
        begin = tuple(begin)
        if len(begin) != len(self.shape):
            raise ValueError(f"begin has {len(begin)} entries; the blocks have {len(self.shape)}")
        index = []
        for b, b0, s, c in zip(begin, self.box[0], self.block_shape, self._counts):
            k, r = divmod(b - b0, s)
            if r != 0 or not 0 <= k < c:
                raise ValueError(
                    f"{begin} is not a block's first voxel: the blocks start at {self.box[0]} "
                    f"in steps of {self.block_shape}, {self._counts} per axis"
                )
            index.append(int(k))
        return tuple(index)

    def _block(self, index: tuple[int, ...], block_id: int, counts: tuple[int, ...]) -> Block:
        origin, stop = self.box
        begin = tuple(o + k * s for o, k, s in zip(origin, index, self.block_shape))
        end = tuple(min(b + s, e) for b, s, e in zip(begin, self.block_shape, stop))
        read_begin = tuple(b - c[0] for b, c in zip(begin, self.context))
        read_end = tuple(e + c[1] for e, c in zip(end, self.context))
        if self.clip:
            read_begin = tuple(min(max(r, 0), n) for r, n in zip(read_begin, self.shape))
            read_end = tuple(
                max(min(r, n), lo) for r, n, lo in zip(read_end, self.shape, read_begin)
            )
        return Block(
            index=tuple(index),
            id=block_id,
            begin=begin,
            end=end,
            read_begin=read_begin,
            read_end=read_end,
            at_start=tuple(k == 0 for k in index),
            at_end=tuple(k == c - 1 for k, c in zip(index, counts)),
        )


def blocks(shape, block_shape, context=0, box=None, clip: bool = True) -> Blocks:
    """The blocks tiling ``[0, shape)``, or the voxel ``box`` (anchored at its begin), by
    ``block_shape``, each read box grown by ``context`` and clipped to the array when
    ``clip`` (Behaviour 17)."""
    return Blocks(shape, block_shape, context, box, clip)


# ---------------------------------------------------------------------------
# Grids


def _per_axis(values, what: str, ndim: int, spatial: tuple[int, ...], other: float):
    values = tuple(float(v) for v in values)
    if len(values) == ndim:
        return values
    if len(values) == len(spatial):
        out = [other] * ndim
        for k, axis in enumerate(spatial):
            out[axis] = values[k]
        return tuple(out)
    raise ValueError(
        f"{what} has {len(values)} entries; expected {len(spatial)} (spatial) or {ndim}"
    )


def _axis_units(units, axes: tuple[str, ...], spatial: tuple[int, ...]) -> tuple[str, ...]:
    sp = set(spatial)
    if isinstance(units, str):
        return tuple(units if i in sp else "" for i in range(len(axes)))
    if isinstance(units, dict):
        unknown = set(units) - {"space", "time"}
        if unknown:
            raise ValueError(f"units by kind take 'space' and 'time', got {sorted(unknown)}")
        return tuple(
            units.get("space", "") if i in sp else units.get("time", "") if a == "t" else ""
            for i, a in enumerate(axes)
        )
    units = tuple(str(u) for u in units)
    if len(units) != len(axes):
        raise ValueError(f"units has {len(units)} entries; expected {len(axes)}")
    return units


@dataclass(frozen=True)
class Grid:
    """A store-less grid: ``shape`` voxels of ``voxel_size`` with voxel 0 centred at
    ``translation`` (in ``units``), optionally chunked and sharded. Answers what a
    ``WorldArray`` of the same facts answers (Behaviour 18)."""

    shape: tuple[int, ...]
    voxel_size: tuple[float, ...]  # one entry per axis once constructed
    translation: tuple[float, ...]  # the CENTRE of voxel 0, one entry per axis
    axes: tuple[str, ...] | None = None  # default_axes(ndim) when None
    units: str | dict | tuple[str, ...] = "nanometer"  # one string per axis once constructed
    chunks: tuple[int, ...] | None = None
    shards: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        shape = _whole(self.shape, "shape")
        if any(s < 0 for s in shape):
            raise ValueError(f"shape cannot be negative, got {shape}")
        n = len(shape)
        axes = default_axes(n) if self.axes is None else tuple(str(a) for a in self.axes)
        if len(axes) != n:
            raise ValueError(f"axes has {len(axes)} entries; the shape has {n}")
        spatial = spatial_axes(axes)
        voxel_size = _per_axis(self.voxel_size, "voxel_size", n, spatial, 1.0)
        if any(v <= 0 for v in voxel_size):
            raise ValueError(f"voxel_size must be positive, got {voxel_size}")
        translation = _per_axis(self.translation, "translation", n, spatial, 0.0)
        chunks = None if self.chunks is None else _sized(self.chunks, "chunks", n, 1)
        shards = None if self.shards is None else _sized(self.shards, "shards", n, 1)
        if shards is not None:
            if chunks is None:
                raise ValueError("shards need chunks: a shard is a whole number of chunks")
            if any(s % c for s, c in zip(shards, chunks)):
                raise ValueError(f"shards {shards} are not a whole number of chunks {chunks}")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "voxel_size", voxel_size)
        object.__setattr__(self, "translation", translation)
        object.__setattr__(self, "units", _axis_units(self.units, axes, spatial))
        object.__setattr__(self, "chunks", chunks)
        object.__setattr__(self, "shards", shards)

    @classmethod
    def from_meta(cls, meta: ArrayMeta, voxel_size: Sequence[float] | None = None) -> Grid:
        """The grid of an existing array; with ``voxel_size`` (per spatial axis, in the meta's
        units), Behaviour 4's target grid over it: anchored at its corner, ``ceil(n * s / t)``
        voxels per spatial axis, voxel 0 centred at ``corner + t / 2``, so a
        ``read(voxel_size=t)`` of the source fills it exactly."""
        if voxel_size is None:
            return cls(
                meta.shape,
                meta.voxel_size,
                meta.translation,
                meta.axes,
                meta.units,
                meta.chunk_shape,
                getattr(meta, "shard_shape", None),
            )
        target = tuple(float(v) for v in voxel_size)
        spatial = meta.spatial
        if len(target) != len(spatial):
            raise ValueError(
                f"voxel_size has {len(target)} entries; the array has {len(spatial)} spatial axes"
            )
        shape, vs, tr = list(meta.shape), list(meta.voxel_size), list(meta.translation)
        corner = meta.corner
        for k, axis in enumerate(spatial):
            shape[axis] = target_shape(meta.shape[axis], meta.voxel_size[axis], target[k])
            vs[axis] = target[k]
            tr[axis] = corner[axis] + target[k] / 2
        return cls(tuple(shape), tuple(vs), tuple(tr), meta.axes, meta.units)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def spatial(self) -> tuple[int, ...]:
        """Indices of the axes named z, y or x when any are; else the last three."""
        return spatial_axes(self.axes)

    @property
    def corner(self) -> tuple[float, ...]:
        """``translation - voxel_size / 2`` on the spatial axes."""
        return tuple(self.translation[i] - self.voxel_size[i] / 2 for i in self.spatial)

    @property
    def box(self) -> Box:
        """The grid's extent on the spatial axes, ``[corner, corner + shape * voxel_size)``."""
        sp = self.spatial
        return Box.from_index(
            (0,) * len(sp),
            tuple(self.shape[i] for i in sp),
            tuple(self.voxel_size[i] for i in sp),
            tuple(self.translation[i] for i in sp),
        )

    def index_box(self, box) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """The voxel index range a world box selects (Behaviour 2's centre rule), over the
        spatial axes or every axis; the other axes come back whole."""
        box = Box.coerce(box)
        spatial, ndim = self.spatial, self.ndim
        if box.ndim == len(spatial):
            axes = list(spatial)
        elif box.ndim == ndim:
            axes = list(range(ndim))
        else:
            raise ValueError(
                f"box has {box.ndim} axes; the grid has {ndim} ({len(spatial)} spatial)"
            )
        lo, hi = index_range(
            box.begin,
            box.end,
            [self.voxel_size[a] for a in axes],
            [self.translation[a] for a in axes],
        )
        begin, end = [0] * ndim, list(self.shape)
        for a, b, e in zip(axes, lo, hi):
            begin[a], end[a] = b, e
        return tuple(begin), tuple(end)

    @property
    def write_shape(self) -> tuple[int, ...] | None:
        """The unit one writer may own: the shards when given, else the chunks; None when
        the grid has neither."""
        return self.shards if self.shards is not None else self.chunks

    def chunks_covering(
        self, begin: Sequence[int], end: Sequence[int], write: bool = False
    ) -> Iterator[tuple[int, ...]]:
        """The chunk indices a voxel box touches, clipped to the grid; ``write=True`` on the
        write-unit grid."""
        cs = self.write_shape if write else self.chunks
        if cs is None:
            raise ValueError(f"this grid has no {'write_shape' if write else 'chunks'}")
        first = [max(0, int(b)) // c for b, c in zip(begin, cs)]
        last = [(min(int(e), n) - 1) // c for e, n, c in zip(end, self.shape, cs)]
        if any(hi < lo for lo, hi in zip(first, last)):
            return iter(())
        return (
            tuple(int(i + lo) for i, lo in zip(idx, first))
            for idx in itertools.product(*(range(hi - lo + 1) for lo, hi in zip(first, last)))
        )

    def blocks(self, block_shape=None, context=0, box=None, clip: bool = True) -> Blocks:
        """``blocks`` over this grid's shape; ``block_shape`` defaults to ``write_shape``, so
        default blocks are aligned writes."""
        if block_shape is None:
            block_shape = self.write_shape
            if block_shape is None:
                raise ValueError("this grid has no chunks or shards; pass block_shape")
        return Blocks(self.shape, block_shape, context, box, clip)

    def meta(self, dtype) -> ArrayMeta:
        """The ``ArrayMeta`` an array created from this grid reports, placed by OME."""
        return ArrayMeta(
            shape=self.shape,
            dtype=dtype,
            chunk_shape=self.chunks if self.chunks is not None else self.shape,
            axes=self.axes,
            axes_named=True,
            units=self.units,
            voxel_size=self.voxel_size,
            translation=self.translation,
            format=None,
            path="",
            placed_by="ome",
        )


# ---------------------------------------------------------------------------
# Schedules


def level_factors(
    voxel_size: Sequence[float],
    ratio: float = 1.5,
    max_levels: int | None = None,
    stop_shape: Sequence[int] | None = None,
    shape: Sequence[int] | None = None,
) -> list[tuple[int, ...]]:
    """Per level, the per-axis downsampling factors from the level before: 2 on every axis
    whose current voxel size is within ``ratio`` of the smallest current one, 1 elsewhere,
    compounding level over level, until ``max_levels`` levels or until a level's shape
    (``shape`` ceil-divided by the cumulative factors) fits within ``stop_shape`` on every
    axis. Voxel size (30, 8, 8) gives (1, 2, 2), (1, 2, 2), then (2, 2, 2) on: z waits until
    y and x reach 32. Isotropic data gives all 2s (mesh-n-bone's schedule, Behaviour 19)."""
    current = [float(v) for v in voxel_size]
    n = len(current)
    if n == 0 or any(v <= 0 for v in current):
        raise ValueError(f"voxel_size must be positive, got {tuple(voxel_size)}")
    if ratio < 1:
        raise ValueError(f"ratio must be at least 1, got {ratio}")
    if max_levels is None and stop_shape is None:
        raise ValueError("pass max_levels, or stop_shape with shape")
    if max_levels is not None and max_levels < 0:
        raise ValueError(f"max_levels cannot be negative, got {max_levels}")
    if stop_shape is not None:
        if shape is None:
            raise ValueError("stop_shape needs shape")
        stop_shape = _sized(stop_shape, "stop_shape", n, 1)
        shape = _sized(shape, "shape", n, 0)
    cumulative = [1] * n
    out: list[tuple[int, ...]] = []
    while max_levels is None or len(out) < max_levels:
        if stop_shape is not None and all(
            -(-s // c) <= t for s, c, t in zip(shape, cumulative, stop_shape)
        ):
            break
        limit = ratio * min(current) * (1 + RTOL)
        step = tuple(2 if v <= limit else 1 for v in current)
        out.append(step)
        current = [v * f for v, f in zip(current, step)]
        cumulative = [c * f for c, f in zip(cumulative, step)]
    return out
