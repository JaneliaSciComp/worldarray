"""``WorldArray``: a stored array, where it sits, and how to read it.

One class for both kinds of consumer. Index-space readers (chunkmirage, CineMap,
liconn-err) use ``.ts``, ``.meta`` and ``read_index`` and keep native units
(``units=None``). World-box readers (cellmap-flow, cellmap-analyze, mesh-n-bone) use
``read`` with boxes in the units they ask for, nanometres by default (spec, section 2).
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import tensorstore as ts

from . import resample as _resample
from .context import shared_context
from .geometry import Box, index_range, is_aligned
from .levels import LevelRef, Mode, discover, same_voxel_size, select_level
from .metadata import ArrayMeta, read_meta
from .retry import Retry, read_with_retries
from .store import Node, detect, is_remote, open_array

logger = logging.getLogger(__name__)

Units = str | dict[str, str] | None


def _unit_targets(units: Units) -> tuple[str | None, str | None]:
    if units is None:
        return None, None
    if isinstance(units, str):
        return units, None
    return units.get("space"), units.get("time")


class WorldArray:
    """A stored array in world coordinates.

    ``path`` names an array, a level under a multiscale group, or a group (then ``level`` or
    ``voxel_size`` picks the level; a group with neither raises, listing its levels).
    """

    def __init__(
        self,
        path: str,
        *,
        level: str | None = None,
        voxel_size: Sequence[float] | None = None,
        level_mode: Mode = "floor",
        on_mismatch: str = "error",
        padding: Any = 0,
        method: str = "nearest",
        select: dict[str, int] | None = None,
        units: Units = "nanometer",
        context: ts.Context | None = None,
        retries: Retry | None = None,
        _ref: LevelRef | None = None,
        _node: Node | None = None,
    ):
        if on_mismatch not in ("error", "resample", "relabel"):
            raise ValueError("on_mismatch is 'error', 'resample' or 'relabel'")
        self.path = path
        self.padding = padding
        self.method = method
        self.units = units
        self.context = context if context is not None else shared_context(0)
        self.retries = retries if retries is not None else Retry()
        self._select = dict(select or {})
        self._opts = dict(
            level=level, voxel_size=voxel_size, level_mode=level_mode, on_mismatch=on_mismatch
        )

        node = _node if _node is not None else detect(path)
        self._pyramid: Pyramid | None = None
        ref = _ref
        if ref is None:
            ref = self._choose_level(node, level, voxel_size, level_mode, on_mismatch)
        self._ref = ref
        node_for_level = (
            node if (ref.scale_index is not None or ref.path == node.path) else detect(ref.path)
        )
        self._node = node_for_level
        self._store = open_array(node_for_level, context=self.context, scale_index=ref.scale_index)
        self._meta = read_meta(node_for_level, self._store, scale_index=ref.scale_index)
        self._apply_select()
        space, time = _unit_targets(units)
        self._grid = self._meta.in_units(space=space, time=time)

        self._target: tuple[float, ...] | None = None
        if voxel_size is not None:
            requested = tuple(float(v) for v in voxel_size)
            current = self.voxel_size
            if not same_voxel_size(current, requested):
                if on_mismatch == "error":
                    raise ValueError(
                        f"{path}: no level at voxel size {list(requested)}; levels: {self._level_list()}"
                    )
                if on_mismatch == "resample":
                    self._target = requested
                else:  # relabel: read the level voxel for voxel and call its voxels the requested size
                    vs = list(self._grid.voxel_size)
                    for k, axis in enumerate(self._grid.spatial):
                        vs[axis] = requested[k]
                    self._grid = replace(self._grid, voxel_size=tuple(vs))

    # --- construction helpers -------------------------------------------------

    def _choose_level(self, node: Node, level, voxel_size, level_mode, on_mismatch) -> LevelRef:
        if (
            node.format == "precomputed"
            and node.scale_key is not None
            and level is None
            and voxel_size is None
        ):
            keys = [str(s.get("key", i)) for i, s in enumerate(node.docs["info"]["scales"])]
            if node.scale_key not in keys:
                raise ValueError(
                    f"{self.path}: no precomputed scale {node.scale_key!r}; scales: {keys}"
                )
            return LevelRef(node.scale_key, node.path, keys.index(node.scale_key))
        if node.kind == "array" and level is None and voxel_size is None:
            return LevelRef(node.path.rsplit("/", 1)[-1], node.path)
        if node.kind == "array" and level is not None:
            warnings.warn(f"{self.path} is an array; level={level!r} is ignored", stacklevel=3)
            return LevelRef(node.path.rsplit("/", 1)[-1], node.path)
        self._pyramid = Pyramid(
            self.path, context=self.context, units=self.units, select=self._select, _node=node
        )
        if level is not None:
            return self._pyramid._ref(level)
        if voxel_size is not None:
            mode = "resample" if on_mismatch == "resample" else level_mode
            return self._pyramid._select_ref(voxel_size, mode)
        raise ValueError(
            f"{self.path} is a group; pass level=<name> or voxel_size=...; levels: {self._pyramid.names}"
        )

    def _apply_select(self) -> None:
        for name, index in sorted(
            self._select.items(), key=lambda kv: -self._meta.axes.index(kv[0])
        ):
            if name not in self._meta.axes:
                raise ValueError(
                    f"{self.path}: no axis named {name!r} to select; axes are {self._meta.axes}"
                )
            axis = self._meta.axes.index(name)
            self._store = self._store[ts.d[axis][int(index)]]
            self._meta = self._meta.drop_axis(axis)

    def _level_list(self) -> list[tuple[str, tuple[float, ...]]]:
        return self.levels

    # --- what it is -------------------------------------------------------------

    @property
    def ts(self) -> ts.TensorStore:
        """The raw tensorstore: C order, origin 0, selected axes dropped."""
        return self._store

    @property
    def meta(self) -> ArrayMeta:
        """Metadata in the store's own units."""
        return self._meta

    @property
    def grid(self) -> ArrayMeta:
        """Metadata in the requested units (``units``), all axes."""
        return self._grid

    @property
    def shape(self) -> tuple[int, ...]:
        return self._grid.shape

    @property
    def dtype(self) -> np.dtype:
        return self._grid.dtype

    @property
    def axes(self) -> tuple[str, ...]:
        return self._grid.axes

    @property
    def spatial(self) -> tuple[int, ...]:
        return self._grid.spatial

    @property
    def voxel_size(self) -> tuple[float, ...]:
        """Spatial axes, in ``units``; the resample target when one is set."""
        if self._target is not None:
            return self._target
        return tuple(self._grid.voxel_size[i] for i in self._grid.spatial)

    @property
    def translation(self) -> tuple[float, ...]:
        """The centre of voxel 0 on the spatial axes, in ``units``."""
        if self._target is not None:
            return tuple(c + t / 2 for c, t in zip(self.corner, self._target))
        return tuple(self._grid.translation[i] for i in self._grid.spatial)

    @property
    def corner(self) -> tuple[float, ...]:
        """``translation - voxel_size / 2`` on the spatial axes: what funlib, daisy and
        Neuroglancer call the offset. The same corner at every voxel size."""
        c = self._grid.corner
        return tuple(c[i] for i in self._grid.spatial)

    @property
    def levels(self) -> list[tuple[str, tuple[float, ...]]]:
        """``(name, spatial voxel size)`` of every level of the pyramid this array belongs to."""
        if self._pyramid is None:
            self._pyramid = Pyramid(
                self.path, context=self.context, units=self.units, select=self._select
            )
        return [(lvl.level_name, lvl.voxel_size) for lvl in self._pyramid.levels]

    @property
    def level_name(self) -> str:
        return self._ref.name

    @property
    def remote(self) -> bool:
        return is_remote(self._node.path)

    def __repr__(self) -> str:
        return (
            f"WorldArray({self.path!r}, level={self.level_name!r}, shape={self.shape}, dtype={self.dtype}, "
            f"axes={self.axes}, voxel_size={self.voxel_size}, translation={self.translation})"
        )

    def reopen(self, *, context: ts.Context | None = None, **overrides) -> WorldArray:
        """The same array on another context (or with other defaults), opened afresh."""
        opts = {
            **self._opts,
            "padding": self.padding,
            "method": self.method,
            "select": self._select,
            "units": self.units,
            "retries": self.retries,
            **overrides,
        }
        return WorldArray(
            self.path, context=context if context is not None else self.context, **opts
        )

    # --- index space ------------------------------------------------------------

    def chunks_covering(
        self, begin: Sequence[int], end: Sequence[int]
    ) -> Iterator[tuple[int, ...]]:
        """The chunk indices a voxel box touches, clipped to the array."""
        cs = self._meta.chunk_shape
        first = [max(0, int(b)) // c for b, c in zip(begin, cs)]
        last = [(min(int(e), n) - 1) // c for e, n, c in zip(end, self.shape, cs)]
        if any(hi < lo for lo, hi in zip(first, last)):
            return
        for idx in np.ndindex(*[hi - lo + 1 for lo, hi in zip(first, last)]):
            yield tuple(int(i + lo) for i, lo in zip(idx, first))

    def read_index(
        self,
        begin: Sequence[int],
        end: Sequence[int],
        padding: Any = None,
        mask: bool = False,
        retries: Retry | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """The voxels ``[begin, end)`` on every axis, clipped to the array, read, and padded
        to the full box with ``padding`` (a value, or ``"edge"`` for the nearest voxel). A box
        entirely outside the array returns the fill without touching storage. With ``mask``,
        also a boolean array that is True inside the array."""
        padding = self.padding if padding is None else padding
        begin = np.asarray(begin, dtype=np.int64)
        end = np.asarray(end, dtype=np.int64)
        shape = np.asarray(self.shape, dtype=np.int64)
        if begin.shape != shape.shape or end.shape != shape.shape:
            raise ValueError(
                f"begin/end must have {len(shape)} entries, got {begin.tolist()} / {end.tolist()}"
            )
        if np.any(end < begin):
            raise ValueError(f"end {end.tolist()} is before begin {begin.tolist()}")
        out_shape = tuple(int(v) for v in end - begin)
        clip_lo = np.maximum(begin, 0)
        clip_hi = np.minimum(end, shape)
        overlap = bool(np.all(clip_hi > clip_lo)) and all(out_shape)
        inner = tuple(slice(int(a - b), int(c - b)) for a, c, b in zip(clip_lo, clip_hi, begin))
        inside = np.zeros(out_shape, dtype=bool)
        if overlap:
            inside[inner] = True

        if padding == "edge":
            # read the overlap, or the nearest one-voxel slab on an axis with none, then repeat the border
            read_lo, read_hi, before, after = [], [], [], []
            for b, e, a, c, n in zip(begin, end, clip_lo, clip_hi, shape):
                if c > a:
                    read_lo.append(int(a))
                    read_hi.append(int(c))
                    before.append(int(a - b))
                    after.append(int(e - c))
                else:
                    edge = 0 if e <= 0 else int(n - 1)
                    read_lo.append(edge)
                    read_hi.append(edge + 1)
                    before.append(int(e - b) - 1)
                    after.append(0)
            if any(s == 0 for s in out_shape):
                out = np.empty(out_shape, dtype=self.dtype)
            else:
                block = self._read_block(read_lo, read_hi, retries)
                out = np.pad(block, list(zip(before, after)), mode="edge")
        else:
            out = np.full(out_shape, padding, dtype=self.dtype)
            if overlap:
                out[inner] = self._read_block(clip_lo.tolist(), clip_hi.tolist(), retries)
        out = np.ascontiguousarray(out)
        return (out, inside) if mask else out

    def _read_block(
        self, lo: Sequence[int], hi: Sequence[int], retries: Retry | None
    ) -> np.ndarray:
        view = self._store[tuple(slice(int(a), int(c)) for a, c in zip(lo, hi))]
        data = read_with_retries(view, retries or self.retries, remote=self.remote, what=self.path)
        data = np.asarray(data)
        if not data.dtype.isnative:
            data = data.astype(data.dtype.newbyteorder("="))
        return data

    # --- world space ------------------------------------------------------------

    def read(
        self,
        box,
        voxel_size: Sequence[float] | None = None,
        padding: Any = None,
        method: str | None = None,
        mask: bool = False,
        retries: Retry | None = None,
        strict: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """The data in a world box, half-open per axis, in ``units``.

        The box covers the spatial axes (other axes are read whole, or as ``select`` pinned
        them) or every axis. A voxel is read iff its centre lies in the box. ``voxel_size``
        reads at that size instead (resampling by ``method``); ``strict`` raises for a box
        aligned to neither voxel corners nor voxel centres.
        """
        box = Box.coerce(box)
        padding = self.padding if padding is None else padding
        method = self.method if method is None else method
        target = tuple(float(v) for v in voxel_size) if voxel_size is not None else self._target
        spatial = self._grid.spatial
        ndim = self._grid.ndim

        if box.ndim == len(spatial):
            axes = list(spatial)
        elif box.ndim == ndim:
            axes = list(range(ndim))
        else:
            raise ValueError(
                f"box has {box.ndim} axes; the array has {ndim} ({len(spatial)} spatial)"
            )

        source_vs = tuple(self._grid.voxel_size[i] for i in spatial)
        if target is not None and same_voxel_size(target, source_vs):
            target = None

        # the grid the box is measured against: the source, or the target grid from the same corner
        vs = list(self._grid.voxel_size)
        tr = list(self._grid.translation)
        if target is not None:
            corner = self._grid.corner
            for k, axis in enumerate(spatial):
                vs[axis] = target[k]
                tr[axis] = corner[axis] + target[k] / 2
        box_vs = [vs[a] for a in axes]
        box_tr = [tr[a] for a in axes]
        if strict and not is_aligned(box.begin, box.end, box_vs, box_tr):
            raise ValueError(
                f"{self.path}: box {box} is aligned to neither voxel corners nor voxel centres"
            )
        lo, hi = index_range(box.begin, box.end, box_vs, box_tr)

        begin = [0] * ndim
        end = list(self.shape)
        for a, lo_a, hi_a in zip(axes, lo, hi):
            begin[a], end[a] = lo_a, hi_a

        if target is None:
            return self.read_index(begin, end, padding=padding, mask=mask, retries=retries)

        def read_index(b, e, pad):
            return self.read_index(b, e, padding=pad, retries=retries)

        return _resample.resample(
            read_index,
            self.shape,
            self.dtype,
            spatial,
            source_vs,
            target,
            begin,
            end,
            method,
            padding,
            mask,
        )

    def to_ndarray_ts(self, roi, **kwargs) -> np.ndarray:
        """``read(roi)``, under the name every existing copy used."""
        return self.read(roi, **kwargs)


class Pyramid:
    """The levels of a multiscale group (or of the group an array belongs to), each a
    ``WorldArray``, opened on first use."""

    def __init__(
        self,
        path: str,
        *,
        context: ts.Context | None = None,
        units: Units = "nanometer",
        select: dict[str, int] | None = None,
        _node: Node | None = None,
        **kwargs,
    ):
        self.path = path
        self.context = context if context is not None else shared_context(0)
        self.units = units
        self._select = select or {}
        self._kwargs = kwargs
        self._node = _node if _node is not None else detect(path)
        self._refs = discover(self._node)
        if not self._refs:
            raise ValueError(
                f"{path}: no levels found (no multiscales datasets, no s0, no precomputed scales)"
            )
        self._levels: dict[str, WorldArray] = {}

    @property
    def names(self) -> list[str]:
        return [r.name for r in self._refs]

    def _ref(self, name: str) -> LevelRef:
        for ref in self._refs:
            if ref.name == name:
                return ref
        raise ValueError(f"{self.path}: no level named {name!r}; levels: {self.names}")

    def _open(self, ref: LevelRef) -> WorldArray:
        if ref.name not in self._levels:
            node = self._node if ref.scale_index is not None else None
            self._levels[ref.name] = WorldArray(
                self.path if ref.scale_index is not None else ref.path,
                context=self.context,
                units=self.units,
                select=self._select,
                _ref=ref,
                _node=node,
                **self._kwargs,
            )
        return self._levels[ref.name]

    @property
    def levels(self) -> tuple[WorldArray, ...]:
        return tuple(self._open(ref) for ref in self._refs)

    def by_name(self, name: str) -> WorldArray:
        return self._open(self._ref(name))

    def _select_ref(self, voxel_size, mode: Mode = "floor") -> LevelRef:
        pairs = [(lvl.level_name, lvl.grid) for lvl in self.levels]
        name, _ = select_level(pairs, voxel_size, mode)
        return self._ref(name)

    def select(self, voxel_size, mode: Mode = "floor") -> WorldArray:
        return self._open(self._select_ref(voxel_size, mode))

    def __iter__(self) -> Iterator[WorldArray]:
        return iter(self.levels)

    def __len__(self) -> int:
        return len(self._refs)


# ---------------------------------------------------------------------------
# Functions


def open(path: str, **kwargs) -> WorldArray:  # noqa: A001
    return WorldArray(path, **kwargs)


def open_pyramid(path: str, **kwargs) -> Pyramid:
    return Pyramid(path, **kwargs)


def metadata(path: str, *, level: str | None = None, units: Units = None) -> ArrayMeta:
    """The ``ArrayMeta`` of an array (or of ``level`` under a group), in native units unless
    ``units`` says otherwise."""
    return (
        WorldArray(path, level=level, units=units).meta
        if units is None
        else WorldArray(path, level=level, units=units).grid
    )


def open_tensorstore(
    path: str, *, level: str | None = None, context: ts.Context | None = None
) -> ts.TensorStore:
    """The raw tensorstore of an array or level: C order, origin 0, a precomputed single
    channel dropped. The access fallbacks and shared context apply."""
    return WorldArray(path, level=level, units=None, context=context).ts
