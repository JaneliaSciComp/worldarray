"""``WorldArray``: a stored array, where it sits, and how to read it.

One class for both kinds of consumer. Index-space readers (chunkmirage, CineMap,
liconn-err) use ``.ts``, ``.meta`` and ``read_index`` and keep native units
(``units=None``). World-box readers (cellmap-flow, cellmap-analyze, mesh-n-bone) use
``read`` with boxes in the units they ask for, nanometres by default (spec, section 2).

Subclass or compose, as you like: every piece of internal state is underscore-prefixed, and
the public attributes ``path``, ``units``, ``context``, ``padding``, ``method``,
``retries`` and ``transform`` are the defaults the reads use, settable at any time.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import tensorstore as ts

from . import resample as _resample
from . import units as U
from . import write as _write
from .context import from_recipe, recipe_of, shared_context
from .geometry import Box, index_range, is_aligned
from .levels import LevelRef, Mode, discover, discover_group, same_voxel_size, select_level
from .placement import (
    ArrayMeta,
    _dims_from_docs,
    default_axes,
    light_levels,
    read_meta,
    read_meta_light,
)
from .retry import Retry, read_with_retries
from .store import (
    Node,
    detect,
    is_remote,
    join,
    normalize_path,
    open_array,
    precomputed_scale_index,
)

logger = logging.getLogger(__name__)

Units = str | dict[str, str] | None
Transform = Callable[[np.ndarray], np.ndarray]


def _unit_targets(units: Units) -> tuple[str | None, str | None]:
    if units is None:
        return None, None
    if isinstance(units, str):
        return units, None
    return units.get("space"), units.get("time")


class WorldArray:
    """A stored array in world coordinates.

    ``path`` names an array (then it is the array that is read, whatever ``voxel_size``
    asks for), a level under a multiscale group, or a group (then ``level`` or
    ``voxel_size`` picks the level; a group with neither raises, listing its levels).
    """

    def __init__(
        self,
        path: str,
        *,
        level: str | int | None = None,
        voxel_size: Sequence[float] | None = None,
        level_mode: Mode = "floor",
        on_mismatch: str = "error",
        padding: Any = 0,
        method: str = "nearest",
        select: dict[str, int] | None = None,
        units: Units = "nanometer",
        context: ts.Context | None = None,
        retries: Retry | None = None,
        metadata_conflicts: str = "error",
        base_voxel_size: Sequence[float] | None = None,
        base_units: str | None = None,
        anchor: Sequence[float] | None = None,
        transform: Transform | None = None,
        strict_units: bool = True,
        mode: str = "r",
        store_fill: bool = False,
        _ref: LevelRef | None = None,
        _node: Node | None = None,
    ):
        if on_mismatch not in ("error", "resample", "relabel", "keep"):
            raise ValueError("on_mismatch is 'error', 'resample', 'relabel' or 'keep'")
        if mode not in ("r", "r+"):
            raise ValueError("mode is 'r' (read) or 'r+' (read and write)")
        self.path = path
        # where later opens find it (reopen, a pickle, the pyramid): a local path resolved
        # against the working directory now, so a worker that changes directory opens the same store
        self._where = normalize_path(path)
        self._mode = mode
        self._store_fill = bool(store_fill)
        self.padding = padding
        self.method = method
        self.units = units
        self.transform = transform
        self.context = context if context is not None else shared_context(0)
        self.retries = retries if retries is not None else Retry()
        self._select = dict(select or {})
        self._opts = dict(
            level=level,
            voxel_size=voxel_size,
            level_mode=level_mode,
            on_mismatch=on_mismatch,
            metadata_conflicts=metadata_conflicts,
            base_voxel_size=base_voxel_size,
            base_units=base_units,
            anchor=anchor,
            strict_units=strict_units,
            mode=mode,
            store_fill=store_fill,
        )
        self._anchor = tuple(float(a) for a in anchor) if anchor is not None else None

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
        self._store = open_array(
            node_for_level,
            context=self.context,
            scale_index=ref.scale_index,
            write=mode == "r+",
            store_fill=store_fill,
        )
        self._meta = read_meta(
            node_for_level, self._store, scale_index=ref.scale_index, conflicts=metadata_conflicts
        )
        self._apply_select()
        if base_voxel_size is not None:
            self._meta = self._rescaled_to_base(base_voxel_size, base_units)
        space, time = _unit_targets(units)
        self._grid = self._meta.in_units(space=space, time=time, strict=strict_units)
        # the stored level's spatial voxel size, before any relabel renames the grid
        self._stored_voxel_size = tuple(self._grid.voxel_size[i] for i in self._grid.spatial)

        self._target: tuple[float, ...] | None = None
        if voxel_size is not None:
            requested = tuple(float(v) for v in voxel_size)
            if not same_voxel_size(self._source_voxel_size(), requested):
                if on_mismatch == "error":
                    raise ValueError(
                        f"{path}: no level at voxel size {list(requested)}; levels: {self.levels}"
                    )
                if on_mismatch == "resample":
                    self._target = requested
                elif on_mismatch == "keep":
                    pass  # the chosen level, at its own voxel size
                else:
                    # relabel: read the level voxel for voxel and call its voxels the requested
                    # size; voxel i stays voxel i, so every world quantity scales by the ratio
                    # (the corner moves with it: cellmap-flow's rule, which chunkmirage_server uses)
                    vs, tr = list(self._grid.voxel_size), list(self._grid.translation)
                    for k, axis in enumerate(self._grid.spatial):
                        ratio = requested[k] / vs[axis]
                        vs[axis] = requested[k]
                        tr[axis] *= ratio
                    self._grid = replace(self._grid, voxel_size=tuple(vs), translation=tuple(tr))

    # --- construction helpers -------------------------------------------------

    def _choose_level(self, node: Node, level, voxel_size, level_mode, on_mismatch) -> LevelRef:
        if node.format == "precomputed" and node.scale_key is not None:
            # a path naming a scale is that scale, as an array path is that array
            if level not in (None, 0, "first"):
                warnings.warn(
                    f"{self.path} names a precomputed scale; level={level!r} is ignored",
                    stacklevel=3,
                )
            index = precomputed_scale_index(node.docs["info"], node.scale_key)
            return LevelRef(node.scale_key, node.path, index)
        if level == "first":
            level = 0
        if node.kind == "array":
            if level is not None and level != 0:  # "first" (already 0) and 0 mean "this array"
                warnings.warn(f"{self.path} is an array; level={level!r} is ignored", stacklevel=3)
            return LevelRef(node.path.rsplit("/", 1)[-1], node.path)  # the array itself, always
        self._pyramid = Pyramid(
            self.path, context=self.context, units=self.units, select=self._select, _node=node
        )
        if level is not None:
            return self._pyramid._ref(level)
        if voxel_size is not None:
            mode = "resample" if on_mismatch == "resample" else level_mode
            return self._pyramid._select_ref(voxel_size, mode)
        raise ValueError(
            f"{self.path} is a group; pass level=<name or index> or voxel_size=...; levels: {self._pyramid.names}"
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

    def _rescaled_to_base(self, base: Sequence[float], base_units: str | None) -> ArrayMeta:
        """This level's metadata with the pyramid's level 0 declared to have voxels of
        ``base`` (in ``base_units``): every level's spatial voxel size and translation scale
        by the same per-axis factor, so stores whose metadata is in voxels of level 0 (as
        BigStitcher writes) or unitless (a BigDataViewer N5) can be placed correctly."""
        base = tuple(float(v) for v in base)
        native = Pyramid(self.path, context=self.context, units=None, select=self._select)
        level0 = native.levels[0].meta if len(native) > 1 else self._meta
        spatial = self._meta.spatial
        if len(base) != len(spatial):
            raise ValueError(
                f"base_voxel_size has {len(base)} entries; the array has {len(spatial)} spatial axes"
            )
        vs, tr, units = (
            list(self._meta.voxel_size),
            list(self._meta.translation),
            list(self._meta.units),
        )
        for k, axis in enumerate(spatial):
            factor = base[k] / level0.voxel_size[axis]
            vs[axis] *= factor
            tr[axis] *= factor
            if base_units is not None:
                units[axis] = base_units
        return replace(self._meta, voxel_size=tuple(vs), translation=tuple(tr), units=tuple(units))

    def _source_voxel_size(self) -> tuple[float, ...]:
        return tuple(self._grid.voxel_size[i] for i in self._grid.spatial)

    def _origins(self) -> list[float]:
        """The target grid's corner, per spatial axis, in source voxels from the source corner."""
        if self._target is None:
            return [0.0] * len(self._grid.spatial)
        corner = self._grid.corner
        anchors = self.anchor
        return [
            (anchors[k] - corner[axis]) / self._grid.voxel_size[axis]
            for k, axis in enumerate(self._grid.spatial)
        ]

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
        """The stored level's metadata in the requested units (``units``), all axes."""
        return self._grid

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape reads see: the stored shape, or with a resample target the number of
        target voxels covering the data on each spatial axis."""
        if self._target is None:
            return self._grid.shape
        shape = list(self._grid.shape)
        for k, axis in enumerate(self._grid.spatial):
            shape[axis] = _resample.target_shape(
                self._grid.shape[axis],
                self._grid.voxel_size[axis],
                self._target[k],
                self._origins()[k],
            )
        return tuple(shape)

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
        return self._source_voxel_size()

    @property
    def source_voxel_size(self) -> tuple[float, ...]:
        """The stored level's voxel size on the spatial axes, in ``units``: what the level
        is, whether reads resample it or ``on_mismatch="relabel"`` renames its grid."""
        return self._stored_voxel_size

    @property
    def resampled(self) -> bool:
        """Whether reads resample to a target voxel size (``on_mismatch="resample"``)."""
        return self._target is not None

    @property
    def translation(self) -> tuple[float, ...]:
        """The centre of voxel 0 on the spatial axes, in ``units``."""
        if self._target is not None:
            return tuple(c + t / 2 for c, t in zip(self.corner, self._target))
        return tuple(self._grid.translation[i] for i in self._grid.spatial)

    @property
    def anchor(self) -> tuple[float, ...]:
        """Where a target voxel corner sits when resampling: the source corner unless
        ``anchor`` was given, so a volume can be read onto another volume's grid."""
        if self._anchor is not None:
            return self._anchor
        c = self._grid.corner
        return tuple(c[i] for i in self._grid.spatial)

    @property
    def corner(self) -> tuple[float, ...]:
        """``translation - voxel_size / 2`` on the spatial axes: what funlib, daisy and
        Neuroglancer call the offset. The same corner at every voxel size, unless an
        ``anchor`` moved the target grid."""
        if self._target is not None and self._anchor is not None:
            return self._anchor
        c = self._grid.corner
        return tuple(c[i] for i in self._grid.spatial)

    @property
    def box(self) -> Box:
        """The array's extent on the spatial axes as a world box, ``[corner, corner + shape *
        voxel_size)``: what funlib calls the array's roi. ``Box.from_index`` gives the same
        over every axis. With a resample target it is the extent of the target voxels that
        ``shape`` counts, which may reach past the stored data by less than one target voxel,
        so ``read(box)`` returns ``shape``."""
        sp = self._grid.spatial
        if self._target is not None:
            begin, end = [], []
            for k, axis in enumerate(sp):
                first, last = _resample.target_range(
                    self._grid.shape[axis],
                    self._grid.voxel_size[axis],
                    self._target[k],
                    self._origins()[k],
                )
                begin.append(self.anchor[k] + first * self._target[k])
                end.append(self.anchor[k] + last * self._target[k])
            return Box(tuple(begin), tuple(end))
        return Box.from_index(
            (0,) * len(sp),
            tuple(self._grid.shape[i] for i in sp),
            tuple(self._grid.voxel_size[i] for i in sp),
            tuple(self._grid.translation[i] for i in sp),
        )

    @property
    def levels(self) -> list[tuple[str, tuple[float, ...]]]:
        """``(name, spatial voxel size)`` of every level of the pyramid this array belongs to,
        from the group's metadata where it lists them, else by opening each level."""
        return self.levels_pyramid.voxel_sizes()

    @property
    def levels_pyramid(self) -> Pyramid:
        if self._pyramid is None:
            self._pyramid = Pyramid(
                self._where, context=self.context, units=self.units, select=self._select
            )
        return self._pyramid

    @property
    def level_name(self) -> str:
        return self._ref.name

    @property
    def level_path(self) -> str:
        """The path of the array actually opened: the level under a group, or ``path`` itself
        for an array (``path`` stays what the caller passed). Writers and chunk-key code
        want this one."""
        return self._meta.path

    @property
    def remote(self) -> bool:
        return is_remote(self._node.path)

    @property
    def writable(self) -> bool:
        """Opened with ``mode="r+"`` (or by ``create``): ``write_index`` and ``write`` work."""
        return self._mode == "r+"

    @property
    def write_shape(self) -> tuple[int, ...]:
        """The unit one writer may own: the shard shape when sharded, else the chunk shape
        (Behaviour 16)."""
        return self._meta.write_shape

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
            "transform": self.transform,
            **overrides,
        }
        return WorldArray(
            self._where, context=context if context is not None else self.context, **opts
        )

    def __reduce__(self):
        """A handle pickles by its arguments and its context's recipe, and reopens in the
        receiving process (Behaviour 11): a worker opens its own store, cheaply, on a shared
        context when the handle had one, else on a context rebuilt from the same spec, so a
        capped context stays capped. A local path goes as resolved at open, so a worker that
        changed directory still finds the store."""
        opts = {
            **self._opts,
            "padding": self.padding,
            "method": self.method,
            "select": self._select,
            "units": self.units,
            "retries": self.retries,
            "transform": self.transform,
        }
        return (_reopen, (self._where, opts, recipe_of(self.context)))

    # --- index space ------------------------------------------------------------

    def chunks_covering(
        self, begin: Sequence[int], end: Sequence[int], write: bool = False
    ) -> Iterator[tuple[int, ...]]:
        """The chunk indices a voxel box touches, clipped to the array; ``write=True`` on the
        write-unit grid (the shard grid of a sharded array)."""
        cs = self.write_shape if write else self._meta.chunk_shape
        first = [max(0, int(b)) // c for b, c in zip(begin, cs)]
        last = [(min(int(e), n) - 1) // c for e, n, c in zip(end, self._grid.shape, cs)]
        if any(hi < lo for lo, hi in zip(first, last)):
            return
        for idx in np.ndindex(*[hi - lo + 1 for lo, hi in zip(first, last)]):
            yield tuple(int(i + lo) for i, lo in zip(idx, first))

    def _key_params(self) -> dict:
        """The facts ``chunk_key`` and ``parse_chunk_key`` need, from the array's documents:
        the storage grid of the whole array (a sharded array's shards; a selected axis is
        still a key component), the separator and encoding, a precomputed scale's origin."""
        node = self._node
        shape, _dtype, chunks, _fill, shard = _dims_from_docs(
            node, self._ref.scale_index if self._ref.scale_index is not None else None
        )
        params: dict[str, Any] = {
            "format": node.format,
            "chunk_shape": shard or chunks,
            "shape": shape,
        }
        if node.format == "zarr2":
            params["separator"] = node.docs.get(".zarray", {}).get("dimension_separator", ".")
        elif node.format == "zarr3":
            enc = node.docs.get("zarr.json", {}).get("chunk_key_encoding", {}) or {}
            params["encoding"] = enc.get("name", "default")
            params["separator"] = (enc.get("configuration") or {}).get("separator")
        elif node.format == "precomputed":
            scale = node.docs["info"]["scales"][self._ref.scale_index or 0]
            if "sharding" in scale:
                raise NotImplementedError("chunk keys of a sharded precomputed store name shards")
            spatial = tuple(scale["size"])[::-1]
            params["chunk_shape"] = tuple(scale["chunk_sizes"][0])[::-1]
            params["shape"] = spatial
            params["origin"] = tuple(int(o) for o in scale.get("voxel_offset", [0, 0, 0]))[::-1]
        return params

    def chunk_key(self, index: Sequence[int]) -> str:
        """The storage key, relative to the array, of the write unit at ``index`` on the
        storage grid (Behaviour 15). A precomputed array's key names the spatial chunk only."""
        return _write.chunk_key(index, **self._key_params())

    def parse_chunk_key(self, key: str) -> tuple[int, ...] | None:
        """``chunk_key`` inverted; None for a key that is not a chunk."""
        return _write.parse_chunk_key(key, **self._key_params())

    def list_chunks(self) -> list[tuple[int, ...]]:
        """The chunk indices that exist in storage (C order, the storage grid), by listing the
        array's keys with ``parse_chunk_key``: zarr v2 and v3 (a sharded array lists its
        shards), N5, and unsharded precomputed. Works on any kvstore, URLs included."""
        params = self._key_params()
        kv = ts.KvStore.open(self._store.kvstore.spec()).result()
        keys = [k.decode() if isinstance(k, bytes) else str(k) for k in kv.list().result()]
        found = set()
        for k in keys:
            idx = _write.parse_chunk_key(
                k.rsplit("/", 1)[-1] if params["format"] == "precomputed" else k, **params
            )
            if idx is not None:
                found.add(idx)
        return sorted(found)

    # --- writing (Behaviour 13) ----------------------------------------------------

    def write_index(
        self, begin: Sequence[int], end: Sequence[int], data: np.ndarray, partial: bool = False
    ) -> None:
        """Write ``data`` over the voxel box ``[begin, end)`` of the stored grid. The box must
        be a union of whole write units (every bound a multiple of ``write_shape`` on its axis,
        or the array's end), so writers in different processes need no coordination; a
        unit written whole is one put. ``partial=True`` lifts the check and does tensorstore's
        read-modify-write of the touched units, unsafe while anything else writes the same
        unit. ``data`` may be a scalar, which fills the box: it is converted to the array's
        dtype only when the value survives (``0`` into ``uint8``, not ``300`` or ``0.5``).
        No other cast, no padding, no resampling; the write commits before returning."""
        if not self.writable:
            raise ValueError(
                f"{self.path} was opened read-only; open it with mode='r+' or create it"
            )
        if self._target is not None:
            raise ValueError(
                f"{self.path}: no resampling on write; open the array without voxel_size="
            )
        begin, end = tuple(int(b) for b in begin), tuple(int(e) for e in end)
        data = np.asarray(data)
        shape = self._grid.shape
        if len(begin) != len(shape) or len(end) != len(shape):
            raise ValueError(f"{self.path}: begin and end need one entry per axis of {shape}")
        wanted = tuple(e - b for b, e in zip(begin, end))
        if data.ndim == 0:
            data = self._scalar(data)
        elif data.shape != wanted:
            raise ValueError(f"{self.path}: data has shape {data.shape}; the box is {wanted}")
        elif data.dtype != self.dtype:
            raise ValueError(
                f"{self.path}: data is {data.dtype}, the array {self.dtype}; nothing is cast on write"
            )
        for axis, (b, e, n) in enumerate(zip(begin, end, shape)):
            if b < 0 or e > n or e <= b:
                raise ValueError(
                    f"{self.path}: write [{b}, {e}) on axis {axis} ({self.axes[axis]}) lies outside the array of {n}"
                )
        if not partial:
            for axis, (b, e, u, n) in enumerate(zip(begin, end, self.write_shape, shape)):
                if b % u != 0 or (e % u != 0 and e != n):
                    raise ValueError(
                        f"{self.path}: write [{b}, {e}) on axis {axis} ({self.axes[axis]}) is not whole write "
                        f"units of {u}; align the box to write_shape {self.write_shape} or pass partial=True"
                    )
        self._store[tuple(slice(b, e) for b, e in zip(begin, end))].write(data).result()

    def _scalar(self, value: np.ndarray) -> np.ndarray:
        """A 0-d ``value`` in the array's dtype, or ``ValueError`` when converting changes it:
        an integer or bool array takes only whole numbers in its range, a float array any
        number that does not overflow it (rounded to its precision)."""
        dtype = self.dtype
        with np.errstate(invalid="ignore", over="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = value.astype(dtype)
            if dtype.kind in "biu":
                kept = bool(out == value)
            else:
                kept = bool(np.isfinite(out) or not np.isfinite(value))
        if not kept:
            raise ValueError(f"{self.path}: {value.item()!r} does not fit {dtype}")
        return out

    def write(self, box, data: np.ndarray, partial: bool = False) -> None:
        """Write ``data`` over a world box: the voxels Behaviour 2's centre rule selects, then
        ``write_index``. A box reaching outside the array raises; nothing is padded or
        resampled on write."""
        begin, end = self.index_box(box)
        for axis, (b, e, n) in enumerate(zip(begin, end, self._grid.shape)):
            if b < 0 or e > n:
                raise ValueError(
                    f"{self.path}: the box reaches outside the array on axis {axis} ({self.axes[axis]}): "
                    f"voxels [{b}, {e}) of {n}; nothing is padded on write"
                )
        self.write_index(begin, end, data, partial=partial)

    def blocks(self, block_shape=None, context=0, box=None, clip: bool = True):
        """The array cut into blocks (Behaviour 17): write boxes tiling the stored grid (or a
        voxel ``box``) with ``block_shape`` (default ``write_shape``, so the default blocks are
        aligned writes), read boxes grown by ``context`` and clipped to the array when
        ``clip``. Plain records, nothing scheduled."""
        from .grid import blocks as _blocks

        return _blocks(
            self._grid.shape,
            self.write_shape if block_shape is None else block_shape,
            context=context,
            box=box,
            clip=clip,
        )

    def write_chunk(self, index: Sequence[int], data: np.ndarray) -> None:
        """Write the whole write unit at ``index`` on the storage grid (the shard of a sharded
        array); ``data`` has that unit's shape, clipped at the array's end."""
        unit, shape = self.write_shape, self._grid.shape
        index = tuple(int(i) for i in index)
        if len(index) != len(shape):
            raise ValueError(f"{self.path}: index needs one entry per axis of {shape}")
        begin = tuple(i * u for i, u in zip(index, unit))
        if any(i < 0 or b >= n for i, b, n in zip(index, begin, shape)):
            raise ValueError(f"{self.path}: no write unit at {index}")
        end = tuple(min(b + u, n) for b, u, n in zip(begin, unit, shape))
        self.write_index(begin, end, data)

    def read_index(
        self,
        begin: Sequence[int],
        end: Sequence[int],
        padding: Any = None,
        mask: bool = False,
        retries: Retry | None = None,
        select: dict[str, int] | None = None,
        transform: Transform | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """The voxels ``[begin, end)`` on every axis of the stored array, clipped to it, read,
        and padded to the full box with ``padding`` (a value, or ``"edge"`` for the nearest
        voxel). A box entirely outside the array returns the fill without touching storage.
        With ``mask``, also a boolean array that is True inside the array. ``select`` pins
        axes for this read only (``{"c": 1}``): ``begin``/``end`` then cover the remaining
        axes and the result drops the pinned ones. ``transform`` (default ``self.transform``)
        is applied to the stored values before padding, so padding is never transformed."""
        if select:
            return self._read_index_selected(begin, end, select, padding, mask, retries, transform)
        padding = self.padding if padding is None else padding
        transform = self.transform if transform is None else transform
        shape = self._grid.shape
        ndim = len(shape)
        if len(begin) != ndim or len(end) != ndim:
            raise ValueError(f"begin/end must have {ndim} entries, got {list(begin)} / {list(end)}")
        begin = tuple(int(b) for b in begin)
        end = tuple(int(e) for e in end)
        if any(e < b for b, e in zip(begin, end)):
            raise ValueError(f"end {list(end)} is before begin {list(begin)}")
        out_shape = tuple(e - b for b, e in zip(begin, end))
        clip_lo = tuple(max(b, 0) for b in begin)
        clip_hi = tuple(min(e, n) for e, n in zip(end, shape))

        # the hot path: a box inside the array needs no mask, no fill and no copy
        if clip_lo == begin and clip_hi == end and all(out_shape):
            out = self._read_block(begin, end, retries, transform)
            return (out, np.ones(out_shape, dtype=bool)) if mask else out

        overlap = all(h > lo for lo, h in zip(clip_lo, clip_hi)) and all(out_shape)
        inner = tuple(slice(a - b, c - b) for a, c, b in zip(clip_lo, clip_hi, begin))
        if padding == "edge":
            # read the overlap, or the nearest one-voxel slab on an axis with none, then repeat the border
            read_lo, read_hi, before, after = [], [], [], []
            for b, e, a, c, n in zip(begin, end, clip_lo, clip_hi, shape):
                if c > a:
                    read_lo.append(a)
                    read_hi.append(c)
                    before.append(a - b)
                    after.append(e - c)
                else:
                    edge = 0 if e <= 0 else n - 1
                    read_lo.append(edge)
                    read_hi.append(edge + 1)
                    before.append(e - b - 1)
                    after.append(0)
            if not all(out_shape):
                out = np.empty(out_shape, dtype=self._grid.dtype)
            else:
                block = self._read_block(read_lo, read_hi, retries, transform)
                out = np.pad(block, list(zip(before, after)), mode="edge")
        else:
            block = self._read_block(clip_lo, clip_hi, retries, transform) if overlap else None
            dtype = block.dtype if block is not None else self._grid.dtype
            out = np.full(out_shape, padding, dtype=dtype)
            if block is not None:
                out[inner] = block
        if not out.flags.c_contiguous:
            out = np.ascontiguousarray(out)
        if mask:
            inside = np.zeros(out_shape, dtype=bool)
            if overlap:
                inside[inner] = True
            return out, inside
        return out

    def _read_block(
        self,
        lo: Sequence[int],
        hi: Sequence[int],
        retries: Retry | None,
        transform: Transform | None = None,
    ) -> np.ndarray:
        view = self._store[tuple(slice(int(a), int(c)) for a, c in zip(lo, hi))]
        data = read_with_retries(view, retries or self.retries, remote=self.remote, what=self.path)
        data = np.asarray(data)
        if not data.dtype.isnative:
            data = data.astype(data.dtype.newbyteorder("="))
        if transform is not None:
            data = np.asarray(transform(data))
        return data

    def _read_index_selected(self, begin, end, select, padding, mask, retries, transform):
        axes = self._grid.axes
        pinned: dict[int, int] = {}
        for name, index in select.items():
            if name not in axes:
                raise ValueError(f"{self.path}: no axis named {name!r} to select; axes are {axes}")
            pinned[axes.index(name)] = int(index)
        if len(begin) != self._grid.ndim - len(pinned):
            raise ValueError(
                f"begin/end must have {self._grid.ndim - len(pinned)} entries with select={select}"
            )
        full_begin, full_end, it = [], [], iter(zip(begin, end))
        for axis in range(self._grid.ndim):
            if axis in pinned:
                full_begin.append(pinned[axis])
                full_end.append(pinned[axis] + 1)
            else:
                b, e = next(it)
                full_begin.append(int(b))
                full_end.append(int(e))
        out = self.read_index(
            full_begin, full_end, padding=padding, mask=mask, retries=retries, transform=transform
        )
        drop = tuple(sorted(pinned))
        if mask:
            return np.squeeze(out[0], axis=drop), np.squeeze(out[1], axis=drop)
        return np.squeeze(out, axis=drop)

    def index_box(self, box) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """The voxel index range a world box selects on the stored grid (Behaviour 2), over
        the spatial axes or every axis; the other axes come back whole. For snapping a
        caller's box to whole voxels before an index read."""
        box = Box.coerce(box)
        spatial, ndim = self._grid.spatial, self._grid.ndim
        if box.ndim == len(spatial):
            axes = list(spatial)
        elif box.ndim == ndim:
            axes = list(range(ndim))
        else:
            raise ValueError(
                f"box has {box.ndim} axes; the array has {ndim} ({len(spatial)} spatial)"
            )
        lo, hi = index_range(
            box.begin,
            box.end,
            [self._grid.voxel_size[a] for a in axes],
            [self._grid.translation[a] for a in axes],
        )
        begin, end = [0] * ndim, list(self._grid.shape)
        for a, b, e in zip(axes, lo, hi):
            begin[a], end[a] = b, e
        return tuple(begin), tuple(end)

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
        anchor: Sequence[float] | None = None,
        transform: Transform | None = None,
        post_transform: Transform | None = None,
        select: dict[str, int] | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """The data in a world box, half-open per axis, in ``units``. ``select`` pins
        non-spatial axes for this read only, as it does at open, and drops them from the result.

        ``post_transform`` is applied after resampling (after the cast back to the stored
        dtype) and before padding, for a normalisation that wants the resampled block.

        The box covers the spatial axes (other axes are read whole, or as ``select`` pinned
        them) or every axis. A voxel is read iff its centre lies in the box. ``voxel_size``
        reads at that size instead (resampling by ``method``); ``anchor`` says where a
        target voxel corner lies (pass another volume's ``corner`` to read onto its grid;
        the source corner, or the ``anchor`` given at open, otherwise); ``transform`` is
        applied to the stored values before resampling and padding; ``strict`` raises for a
        box aligned to neither voxel corners nor voxel centres.
        """
        box = Box.coerce(box)
        padding = self.padding if padding is None else padding
        method = self.method if method is None else method
        transform = self.transform if transform is None else transform
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

        source_vs = self._source_voxel_size()
        if target is not None and same_voxel_size(target, source_vs):
            target = None

        # the grid the box is measured against: the source, or the target grid from the anchor
        vs = list(self._grid.voxel_size)
        tr = list(self._grid.translation)
        origins: list[float] = []
        if target is not None:
            corner = self._grid.corner
            anchors = tuple(float(a) for a in anchor) if anchor is not None else self.anchor
            if len(anchors) != len(spatial):
                raise ValueError(
                    f"anchor has {len(anchors)} entries; the array has {len(spatial)} spatial axes"
                )
            for k, axis in enumerate(spatial):
                vs[axis] = target[k]
                tr[axis] = anchors[k] + target[k] / 2
                origins.append((anchors[k] - corner[axis]) / source_vs[k])
        box_vs = [vs[a] for a in axes]
        box_tr = [tr[a] for a in axes]
        if strict and not is_aligned(box.begin, box.end, box_vs, box_tr):
            raise ValueError(
                f"{self.path}: box {box} is aligned to neither voxel corners nor voxel centres"
            )
        lo, hi = index_range(box.begin, box.end, box_vs, box_tr)

        begin = [0] * ndim
        end = list(self._grid.shape)
        for a, lo_a, hi_a in zip(axes, lo, hi):
            begin[a], end[a] = lo_a, hi_a
        drop: tuple[int, ...] = ()
        if select:
            names = self._grid.axes
            pinned = []
            for name, index in select.items():
                if name not in names:
                    raise ValueError(
                        f"{self.path}: no axis named {name!r} to select; axes are {names}"
                    )
                a = names.index(name)
                begin[a], end[a] = int(index), int(index) + 1
                pinned.append(a)
            drop = tuple(sorted(pinned))
            if drop:  # the callables see the array as the caller will, without the pinned axes

                def _without_pinned(fn):
                    return lambda a: np.expand_dims(np.asarray(fn(np.squeeze(a, axis=drop))), drop)

                transform = _without_pinned(transform) if transform is not None else None
                post_transform = (
                    _without_pinned(post_transform) if post_transform is not None else None
                )

        if target is None:
            want_mask = mask or post_transform is not None
            out = self.read_index(
                begin, end, padding=padding, mask=want_mask, retries=retries, transform=transform
            )
            if post_transform is None:
                return _squeeze(out, drop)
            data, inside = out
            data = np.asarray(post_transform(data))
            if padding != "edge":
                data = data.copy()
                data[~inside] = padding
            return _squeeze((data, inside) if mask else data, drop)

        def read_index(b, e, pad):
            return self.read_index(b, e, padding=pad, retries=retries, transform=transform)

        out = _resample.resample(
            read_index,
            self._grid.shape,
            self._grid.dtype,
            spatial,
            source_vs,
            target,
            begin,
            end,
            method,
            padding,
            True,
            origins,
        )
        data, inside = out
        if post_transform is not None:
            data = np.asarray(post_transform(data))
            if padding != "edge" and not inside.all():
                data = data.copy()
                data[~inside] = padding
        return _squeeze((data, inside) if mask else data, drop)

    def read_onto(
        self,
        grid,
        begin: Sequence[int] | None = None,
        end: Sequence[int] | None = None,
        padding: Any = None,
        method: str | None = None,
        mask: bool = False,
        retries: Retry | None = None,
        transform: Transform | None = None,
        post_transform: Transform | None = None,
        select: dict[str, int] | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Voxels ``[begin, end)`` of another grid (a ``Grid``), read from this array
        (Behaviour 20): ``begin`` and ``end`` index the grid's spatial axes, all of it when
        None, and the result has exactly that shape there; other axes are read as ``read``
        reads them. At the array's own voxel size nothing is resampled, however the grids are
        offset: each grid voxel takes the stored voxel whose centre lies in it. Otherwise the
        array is resampled onto the grid by ``method``, anchored at the grid's corner. The
        grid's units are converted to ``units``; its spatial axes must be named as this
        array's are."""
        sp, mine = grid.spatial, self._grid.spatial
        names = tuple(grid.axes[i] for i in sp)
        if names != tuple(self._grid.axes[i] for i in mine):
            raise ValueError(
                f"{self.path}: the grid's spatial axes are {names}; the array's "
                f"{tuple(self._grid.axes[i] for i in mine)}"
            )
        vs, tr = [], []
        for i, j in zip(sp, mine):
            have, want = grid.units[i], self._grid.units[j]
            f = 1.0 if have == want else U.factor(have, want)
            vs.append(grid.voxel_size[i] * f)
            tr.append(grid.translation[i] * f)
        begin = (0,) * len(sp) if begin is None else tuple(int(b) for b in begin)
        end = tuple(grid.shape[i] for i in sp) if end is None else tuple(int(e) for e in end)
        if len(begin) != len(sp) or len(end) != len(sp):
            raise ValueError(f"begin and end need one entry per spatial axis of the grid {names}")
        return self.read(
            Box.from_index(begin, end, vs, tr),
            voxel_size=vs,
            anchor=tuple(t - v / 2 for t, v in zip(tr, vs)),
            padding=padding,
            method=method,
            mask=mask,
            retries=retries,
            transform=transform,
            post_transform=post_transform,
            select=select,
        )


def _squeeze(out, drop: tuple[int, ...]):
    if not drop:
        return out
    if isinstance(out, tuple):
        return np.squeeze(out[0], axis=drop), np.squeeze(out[1], axis=drop)
    return np.squeeze(out, axis=drop)


class Pyramid:
    """The levels of a multiscale group (or of the group an array belongs to), each a
    ``WorldArray``, opened on first use. Choosing a level by voxel size reads the group's
    own metadata where it lists the levels' scales (OME-NGFF, COSEM N5, precomputed), and
    opens the levels only when it does not."""

    def __init__(
        self,
        path: str,
        *,
        context: ts.Context | None = None,
        units: Units = "nanometer",
        select: dict[str, int] | None = None,
        mode: str = "r",
        _node: Node | None = None,
        **kwargs,
    ):
        if mode not in ("r", "r+"):
            raise ValueError("mode is 'r' (read) or 'r+' (read and write)")
        self.path = path
        self.context = context if context is not None else shared_context(0)
        self.units = units
        self._select = select or {}
        self._mode = mode
        self._kwargs = {**kwargs, "mode": mode}
        self._node = _node if _node is not None else detect(path)
        self._group, self._refs = discover_group(self._node)  # the node listing the levels
        if not self._refs:
            raise ValueError(
                f"{path}: no levels found (no multiscales datasets, no s0, no precomputed scales)"
            )
        self._levels: dict[str, WorldArray] = {}
        self._light: list[tuple[str, tuple[float, ...], tuple[float, ...]]] | None = None

    @property
    def names(self) -> list[str]:
        return [r.name for r in self._refs]

    def _ref(self, level: str | int) -> LevelRef:
        """A level by name, or by position (0 is the first listed, the finest by convention)."""
        if isinstance(level, int) and not isinstance(level, bool):
            if not -len(self._refs) <= level < len(self._refs):
                raise ValueError(
                    f"{self.path}: no level {level}; there are {len(self._refs)}: {self.names}"
                )
            return self._refs[level]
        for ref in self._refs:
            if ref.name == str(level):
                return ref
        raise ValueError(f"{self.path}: no level named {level!r}; levels: {self.names}")

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

    def by_name(self, level: str | int) -> WorldArray:
        return self._open(self._ref(level))

    def placements(self) -> list[tuple[str, tuple[float, ...], tuple[float, ...]]]:
        """``(name, spatial voxel size, spatial translation)`` per level in the requested
        units, from the group's own document when it lists the levels' scales (OME-NGFF,
        COSEM N5, precomputed), else by opening each level."""
        if self._light is None:
            space, _ = _unit_targets(self.units)
            # the group the levels were discovered from, whichever path opened the pyramid
            group = self._group
            light = light_levels(group, self.names, space) if group is not None else None
            if light is None:
                light = [
                    (lvl.level_name, lvl.source_voxel_size, lvl.translation) for lvl in self.levels
                ]
            self._light = light
        return self._light

    def level_meta(self, level: str | int) -> ArrayMeta:
        """One level's ``ArrayMeta`` in native units from its documents alone, with no
        tensorstore opened (shape, dtype, chunks and placement; a sharded array's inner
        chunk)."""
        ref = self._ref(level)
        conflicts = self._kwargs.get("metadata_conflicts", "error")
        if ref.scale_index is not None:
            return read_meta_light(self._node, scale_index=ref.scale_index, conflicts=conflicts)
        return read_meta_light(detect(ref.path), conflicts=conflicts)

    def metas(self) -> list[ArrayMeta]:
        """``level_meta`` for every level, in order."""
        return [self.level_meta(ref.name) for ref in self._refs]

    def voxel_sizes(self) -> list[tuple[str, tuple[float, ...]]]:
        """``(name, spatial voxel size)`` per level; see ``placements``."""
        return [(name, vs) for name, vs, _ in self.placements()]

    def translations(self) -> list[tuple[str, tuple[float, ...]]]:
        """``(name, spatial translation)`` per level; see ``placements``."""
        return [(name, tr) for name, _, tr in self.placements()]

    def add_level(
        self,
        name: str,
        factors=None,
        origin=0,
        *,
        voxel_size=None,
        translation=None,
        shape=None,
        dtype=None,
        chunks=None,
        shards=None,
        codecs=None,
        fill_value=None,
        attrs: dict | None = None,
        exists: str = "error",
        from_level: str | int | None = None,
    ) -> WorldArray:
        """A new level's dataset entry and its empty array (Behaviour 14). Placement by
        ``level_placement`` over ``from_level`` (the finest listed when None) from ``factors``
        and ``origin``, unless ``voxel_size`` and ``translation`` are given; shape by
        ``level_shape`` unless given; dtype, chunks, shards, codecs and fill value copied from
        that level unless given. Writes no voxel data. One process at a time."""
        if self._mode != "r+":
            raise ValueError(f"{self.path}: open the pyramid with mode='r+' to add a level")
        group = self._group
        if group is None or group.kind != "group" or group.format not in ("zarr2", "zarr3"):
            raise ValueError(f"{self.path}: add_level needs an OME zarr group (v2 or v3)")
        base_ref = self._ref(from_level if from_level is not None else 0)
        base_node = detect(base_ref.path)
        base = read_meta_light(base_node, conflicts=self._kwargs.get("metadata_conflicts", "error"))
        ndim, spatial = base.ndim, base.spatial
        if voxel_size is None or translation is None:
            if factors is None:
                raise ValueError("pass factors (and origin), or voxel_size and translation")
            f = _write.expand_spatial(
                (factors,) * len(spatial) if np.isscalar(factors) else factors, spatial, ndim, 1
            )
            o = _write.expand_spatial(
                (origin,) * len(spatial) if np.isscalar(origin) else origin, spatial, ndim, 0
            )
            vs, tr = _write.level_placement(base.voxel_size, base.translation, f, o)
            if shape is None:
                shape = _write.level_shape(base.shape, f, o)
        else:
            vs = _write.expand_spatial(voxel_size, spatial, ndim, 1.0)
            tr = _write.expand_spatial(translation, spatial, ndim, 0.0)
            if shape is None:
                raise ValueError("pass shape when placing a level by voxel_size and translation")
        base_codecs, base_shards = _write.codecs_of(base_node)
        if codecs is None:
            codecs = base_codecs if base_codecs is not None else "none"
        if shards is None:
            shards = base_shards
        vol = create(
            join(group.path, name),
            shape,
            base.dtype if dtype is None else dtype,
            vs,
            tr,
            axes=base.axes,
            units=base.units,
            chunks=base.chunk_shape if chunks is None else chunks,
            shards=shards,
            fill_value=(base.fill_value if base.fill_value is not None else 0)
            if fill_value is None
            else fill_value,
            codecs=codecs,
            format=group.format,
            attrs=attrs,
            exists=exists,
            context=self.context,
        )
        self._group, self._refs = discover_group(detect(group.path))
        self._light, self._levels = None, {}
        return vol

    def _select_ref(self, voxel_size, mode: Mode = "floor") -> LevelRef:
        name, _ = select_level(self.voxel_sizes(), voxel_size, mode)
        return self._ref(name)

    def select(self, voxel_size, mode: Mode = "floor") -> WorldArray:
        return self._open(self._select_ref(voxel_size, mode))

    def __iter__(self) -> Iterator[WorldArray]:
        return iter(self.levels)

    def __len__(self) -> int:
        return len(self._refs)


# ---------------------------------------------------------------------------
# Functions


def create(
    path: str,
    shape: Sequence[int] | None = None,
    dtype=None,
    voxel_size=None,
    translation=None,
    *,
    grid=None,
    axes: Sequence[str] | None = None,
    units: Units = "nanometer",
    chunks: Sequence[int] | None = None,
    shards: Sequence[int] | None = None,
    fill_value=0,
    codecs="zstd",
    format: str | None = None,
    attrs: dict | None = None,
    group_attrs: dict | None = None,
    exists: str = "error",
    store_fill: bool = False,
    separator: str | None = None,
    axis_types: Sequence[str | None] | None = None,
    context: ts.Context | None = None,
    retries: Retry | None = None,
) -> WorldArray:
    """Create a placed level: the group document (made when absent, else its one dataset
    entry added or replaced) and then the array, as OME-NGFF zarr v2 (OME 0.4) or v3 (OME
    0.5) following the group's format (Behaviours 11 and 12). ``chunks`` is required.
    Nothing is ever deleted or overwritten: with ``exists="error"`` anything at ``path``
    raises ``FileExistsError``; with ``"adopt"`` an identical array (shape, dtype, chunks,
    shards, placement) is opened for writing and a different one raises ``ValueError``
    naming every difference. Returns a writable ``WorldArray``."""
    if grid is not None:
        # a Grid planned the output (Behaviour 18): its facts are the array's, chunks and
        # shards overridable; the placement arguments must not also be given
        if any(v is not None for v in (shape, voxel_size, translation, axes)):
            raise ValueError("pass grid= or shape, voxel_size, translation and axes, not both")
        shape, voxel_size, translation = grid.shape, grid.voxel_size, grid.translation
        axes, units = grid.axes, grid.units
        chunks = grid.chunks if chunks is None else chunks
        shards = grid.shards if shards is None else shards
    if shape is None or dtype is None or voxel_size is None or translation is None:
        raise ValueError(
            "create needs shape, dtype, voxel_size and translation, or grid= and dtype"
        )
    if chunks is None:
        raise ValueError("chunks is required: the package never picks a chunk shape")
    ndim = len(shape)
    axes = tuple(str(a) for a in axes) if axes is not None else default_axes(ndim)
    context = context if context is not None else shared_context(0)
    kinds = _write.axis_kinds(axes, axis_types)
    vs = _write.per_axis(voxel_size, kinds, 1.0, f"{path}: voxel_size")
    tr = _write.per_axis(translation, kinds, 0.0, f"{path}: translation")
    vol_units: Units = units if isinstance(units, (str, dict)) or units is None else None

    def adopt(normalized: str, fmt: str) -> WorldArray:
        vol = WorldArray(
            normalized,
            mode="r+",
            units=None,
            context=context,
            retries=retries,
            store_fill=store_fill,
        )
        diffs = _write.differences(
            vol.meta,
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            shards=shards,
            voxel_size=vs,
            translation=tr,
        )
        if diffs:
            raise ValueError(
                f"{normalized} exists and differs from what was asked for; the package never "
                "overwrites:\n  " + "\n  ".join(diffs)
            )
        group_path = normalized.rsplit("/", 1)[0]
        _write.ensure_group_entry(
            group_path,
            normalized.rsplit("/", 1)[-1],
            vs,
            tr,
            axes,
            units,
            axis_types,
            group_attrs,
            format=fmt,
        )
        return WorldArray(
            normalized,
            mode="r+",
            units=vol_units,
            context=context,
            retries=retries,
            store_fill=store_fill,
        )

    normalized, fmt, adopted = _write.create_array(
        path,
        shape,
        dtype,
        vs,
        tr,
        axes=axes,
        units=units,
        chunks=chunks,
        shards=shards,
        fill_value=fill_value,
        codecs=codecs,
        format=format,
        attrs=attrs,
        group_attrs=group_attrs,
        exists=exists,
        store_fill=store_fill,
        separator=separator,
        axis_types=axis_types,
        context=context,
    )
    if adopted:
        return adopt(normalized, fmt)
    return WorldArray(
        normalized,
        mode="r+",
        units=vol_units,
        context=context,
        retries=retries,
        store_fill=store_fill,
    )


def _reopen(path: str, opts: dict, recipe=None) -> WorldArray:
    return WorldArray(path, context=from_recipe(recipe), **opts)


def open(path: str, **kwargs) -> WorldArray:  # noqa: A001
    return WorldArray(path, **kwargs)


def open_pyramid(path: str, **kwargs) -> Pyramid:
    return Pyramid(path, **kwargs)


def metadata(
    path: str, *, level: str | int | None = None, units: Units = None, open: bool = True
) -> ArrayMeta:
    """The ``ArrayMeta`` of an array (or of ``level`` under a group), in native units unless
    ``units`` says otherwise. ``open=False`` reads it from the documents alone, with no
    tensorstore opened."""
    if open:
        vol = WorldArray(path, level=level, units=units)
        return vol.meta if units is None else vol.grid
    node = detect(path)
    if level == "first":
        level = 0
    if node.format == "precomputed" and (level is not None or node.scale_key is not None):
        wanted = (
            node.scale_key if node.scale_key is not None else level
        )  # a scale path is that scale
        meta = read_meta_light(node, scale_index=precomputed_scale_index(node.docs["info"], wanted))
    elif node.kind == "group":
        refs = discover(node)
        if not refs:
            raise ValueError(
                f"{path}: no levels found (no multiscales datasets, no s0, no precomputed scales)"
            )
        if level is None:
            raise ValueError(f"{path} is a group; pass level= one of {[r.name for r in refs]}")
        ref = (
            refs[level]
            if isinstance(level, int)
            else next((r for r in refs if r.name == level), None)
        )
        if ref is None:
            raise ValueError(f"{path}: no level {level!r}; levels: {[r.name for r in refs]}")
        meta = read_meta_light(detect(ref.path))
    else:
        meta = read_meta_light(node)
    if units is None:
        return meta
    space, time = _unit_targets(units)
    return meta.in_units(space=space, time=time)


def open_tensorstore(
    path: str, *, level: str | int | None = None, context: ts.Context | None = None
) -> ts.TensorStore:
    """The raw tensorstore of an array or level: C order, origin 0, a precomputed single
    channel dropped. The access fallbacks and shared context apply. No world metadata is
    read, so an array whose placement attributes are odd (a coordinate array, a field) still
    opens."""
    context = context if context is not None else shared_context(0)
    node = detect(path)
    if level == "first":
        level = 0
    if node.format == "precomputed":
        wanted = node.scale_key if level is None else level
        if wanted is None:
            keys = [str(s.get("key", i)) for i, s in enumerate(node.docs["info"]["scales"])]
            raise ValueError(f"{path} is a precomputed volume; pass level= one of {keys}")
        index = (
            wanted
            if isinstance(wanted, int)
            else precomputed_scale_index(node.docs["info"], wanted)
        )
        return open_array(node, context=context, scale_index=index)
    if node.kind == "group":
        refs = discover(node)
        if level is None:
            raise ValueError(f"{path} is a group; pass level= one of {[r.name for r in refs]}")
        ref = (
            refs[level]
            if isinstance(level, int)
            else next((r for r in refs if r.name == level), None)
        )
        if ref is None:
            raise ValueError(f"{path}: no level {level!r}; levels: {[r.name for r in refs]}")
        node = detect(ref.path)
    elif level is not None:
        warnings.warn(f"{path} is an array; level={level!r} is ignored", stacklevel=2)
    return open_array(node, context=context)


__all__ = [
    "Pyramid",
    "WorldArray",
    "create",
    "metadata",
    "open",
    "open_pyramid",
    "open_tensorstore",
    "U",
]
