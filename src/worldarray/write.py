"""Writing: OME-NGFF documents and placed creation (spec: Convention 6, Behaviours 10 to 16).

The package writes OME-NGFF zarr only: zarr v2 with OME 0.4 (``.zgroup`` and ``.zattrs``
holding ``multiscales``) or zarr v3 with OME 0.5 (``zarr.json`` with ``attributes.ome``),
never 0.4 inside v3, never N5 or precomputed. A written ``translation`` is voxel 0's centre,
written always, zeros included; ``scale`` is the voxel size; every number is the float it
was given. Nothing here deletes, truncates or overwrites: ``create_array`` raises on an
existing path unless told to adopt, and a group document is read, its one dataset entry
replaced or added, and written back with every other key as it was.

The pure functions (``ome_multiscales``, ``level_placement``, ``level_shape``,
``chunk_key``, ``parse_chunk_key``) need no store. ``create_array`` writes the group
document and then the array through tensorstore; ``worldarray.array.create`` wraps it in a
writable ``WorldArray`` and does the adopt check.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import tensorstore as ts

from .geometry import snap_integral
from .placement import spatial_axes
from .store import (
    _DRIVER,
    Location,
    Node,
    basename,
    detect,
    is_refused,
    is_root,
    kvstore_specs,
    normalize_path,
    parent,
    with_access,
)
from .units import ome_name

logger = logging.getLogger(__name__)

SPACE, TIME, CHANNEL = "space", "time", "channel"
ZSTD_LEVEL = 3  # what zarr-python's default zstd(level=0) means
RACE_ATTEMPTS = 5  # a creation that lost the race to another process is re-read this often
RACE_DELAY = 0.5  # seconds, times the attempt (registration-station's shared_array)
RTOL = 1e-6

Levels = Sequence[tuple[str, Sequence[float], Sequence[float]]]


# ---------------------------------------------------------------------------
# Axes, units, per-axis values


def axis_kinds(axes: Sequence[str], types: Sequence[str | None] | None = None) -> tuple:
    """The OME ``type`` of each axis: as given, else from the name (``z``, ``y``, ``x``
    space, ``t`` time, ``c`` channel; an unnamed array's last three axes space)."""
    axes = tuple(str(a) for a in axes)
    if types is not None:
        types = tuple(types)
        if len(types) != len(axes):
            raise ValueError(f"axis_types has {len(types)} entries for {len(axes)} axes")
        for t in types:
            if t not in (SPACE, TIME, CHANNEL, None):
                raise ValueError(f"axis type {t!r}; use 'space', 'time', 'channel' or None")
        return types
    sp = set(spatial_axes(axes))
    out: list[str | None] = []
    for i, a in enumerate(axes):
        if a == "t":
            out.append(TIME)
        elif a == "c":
            out.append(CHANNEL)
        elif a in ("z", "y", "x") or i in sp:
            out.append(SPACE)
        else:
            out.append(None)
    return tuple(out)


def axis_units(units, axes: Sequence[str], kinds: Sequence[str | None]) -> tuple[str, ...]:
    """Per-axis unit strings: a str is the unit of the space axes, a dict by kind gives
    ``space`` and ``time``, a sequence is one per axis; ``""`` where there is none."""
    n = len(axes)
    if units is None:
        return ("",) * n
    if isinstance(units, str):
        return tuple(units if k == SPACE else "" for k in kinds)
    if isinstance(units, dict):
        return tuple(str(units.get(k) or "") if k in (SPACE, TIME) else "" for k in kinds)
    seq = tuple(str(u) if u else "" for u in units)
    if len(seq) != n:
        raise ValueError(f"units has {len(seq)} entries for {n} axes")
    return seq


def per_axis(values, kinds: Sequence[str | None], fill: float, what: str) -> tuple[float, ...]:
    """``values`` as one float per axis: given for every axis, or for the space axes only
    (the others take ``fill``)."""
    values = tuple(float(v) for v in values)
    n = len(kinds)
    space = [i for i, k in enumerate(kinds) if k == SPACE]
    if len(values) == n:
        return values
    if len(values) == len(space):
        out = [float(fill)] * n
        for i, v in zip(space, values):
            out[i] = v
        return tuple(out)
    raise ValueError(
        f"{what} has {len(values)} entries; the array has {n} axes, {len(space)} spatial"
    )


def _broadcast(value, n: int, what: str) -> tuple[float, ...]:
    if np.isscalar(value):
        return (float(value),) * n
    out = tuple(float(v) for v in value)
    if len(out) != n:
        raise ValueError(f"{what} has {len(out)} entries for {n} axes")
    return out


# ---------------------------------------------------------------------------
# Documents (Behaviour 10)


def ome_multiscales(
    axes: Sequence[str],
    units="nanometer",
    levels: Levels = (),
    version: str = "0.5",
    axis_types: Sequence[str | None] | None = None,
    name: str = "",
    metadata: dict | None = None,
    type: str | None = None,
) -> dict:
    """The complete group attributes of an OME-NGFF multiscale group: ``{"multiscales":
    [...]}`` for 0.4 (zarr v2), ``{"ome": {"version": "0.5", "multiscales": [...]}}`` for
    0.5 (zarr v3). ``levels`` are ``(path, voxel_size, translation)`` with the placement
    explicit per level, over the space axes or every axis; ``translation`` is voxel 0's
    centre and is written always. Units are written in OME's names: any spelling the unit
    table knows (``"nm"``, ``"µm"``, ``"nanometre"``) becomes ``"nanometer"``,
    ``"micrometer"``, ...; a unit it does not know is written as given. ``type`` (the
    downsampling method, OME's optional field), ``name`` and ``metadata`` are the
    multiscale's own."""
    axes = tuple(str(a) for a in axes)
    kinds = axis_kinds(axes, axis_types)
    unit_list = tuple(ome_unit(u) for u in axis_units(units, axes, kinds))
    axis_docs = []
    for a, k, u in zip(axes, kinds, unit_list):
        doc: dict[str, Any] = {"name": a}
        if k is not None:
            doc["type"] = k
        if u and k in (SPACE, TIME, None):
            doc["unit"] = u
        axis_docs.append(doc)
    datasets = [dataset_entry(path, vs, tr, kinds) for path, vs, tr in levels]
    multiscale: dict[str, Any] = {"name": name, "axes": axis_docs, "datasets": datasets}
    if type is not None:
        multiscale["type"] = str(type)
    if metadata:
        multiscale["metadata"] = dict(metadata)
    if version == "0.4":
        return {"multiscales": [{"version": "0.4", **multiscale}]}
    if version == "0.5":
        return {"ome": {"version": "0.5", "multiscales": [multiscale]}}
    raise ValueError(f"OME version {version!r}; the package writes 0.4 (zarr v2) or 0.5 (zarr v3)")


def ome_unit(unit: str) -> str:
    """``unit`` in OME's name when the unit table knows it, else as given (``""`` stays
    ``""``, ``"pixel"`` stays ``"pixel"``: Convention 3 welcomes unknown units)."""
    try:
        return ome_name(unit) if unit else ""
    except ValueError:
        return unit


def dataset_entry(path: str, voxel_size, translation, kinds: Sequence[str | None]) -> dict:
    vs = per_axis(voxel_size, kinds, 1.0, f"{path}: voxel_size")
    tr = per_axis(translation, kinds, 0.0, f"{path}: translation")
    return {
        "path": str(path).strip("/"),
        "coordinateTransformations": [
            {"type": "scale", "scale": list(vs)},
            {"type": "translation", "translation": list(tr)},
        ],
    }


def level_placement(
    voxel_size, translation, factors, origin=0
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The placement of the level whose voxel ``j`` covers source voxels ``origin + j f ..
    origin + (j + 1) f - 1`` on each axis: ``(voxel_size f, translation + voxel_size (origin +
    (f - 1) / 2))``. Convention 1's rule inverted, and the centre a block mean produces."""
    vs = tuple(float(v) for v in voxel_size)
    tr = tuple(float(t) for t in translation)
    if len(tr) != len(vs):
        raise ValueError("voxel_size and translation must have the same length")
    f = _broadcast(factors, len(vs), "factors")
    o = _broadcast(origin, len(vs), "origin")
    return (
        tuple(v * fi for v, fi in zip(vs, f)),
        tuple(t + v * (oi + (fi - 1) / 2) for t, v, fi, oi in zip(tr, vs, f, o)),
    )


def level_shape(shape, factors, origin=0) -> tuple[int, ...]:
    """``ceil((n - origin) / f)`` per axis: the voxels of a level that covers source voxels
    from ``origin`` on. The quotient is snapped to a whole number within ``1e-6`` first
    (Convention 3), so a factor that is a ratio of float voxel sizes, ``0.3 / 0.1 =
    2.9999999999999996``, gives 3 voxels for 9, not 4."""
    shape = tuple(int(n) for n in shape)
    f = _broadcast(factors, len(shape), "factors")
    o = _broadcast(origin, len(shape), "origin")
    return tuple(
        max(0, int(math.ceil(float(snap_integral((n - oi) / fi)))))
        for n, fi, oi in zip(shape, f, o)
    )


def expand_spatial(values, spatial: Sequence[int], ndim: int, fill) -> tuple:
    """Spatial-length ``values`` spread onto the ``spatial`` axes of an ``ndim`` array, the
    other axes taking ``fill``; values already of length ``ndim`` pass through."""
    values = tuple(values)
    if len(values) == ndim:
        return values
    if len(values) != len(spatial):
        raise ValueError(f"expected {ndim} or {len(spatial)} values, got {len(values)}")
    out = [fill] * ndim
    for i, v in zip(spatial, values):
        out[i] = v
    return tuple(out)


# ---------------------------------------------------------------------------
# Chunk keys (Behaviour 15)

_PRECOMPUTED_KEY = re.compile(r"^(\d+)-(\d+)(?:_(\d+)-(\d+))*$")


def chunk_key(
    index: Sequence[int],
    format: str,
    separator: str | None = None,
    encoding: str = "default",
    chunk_shape: Sequence[int] | None = None,
    shape: Sequence[int] | None = None,
    origin: Sequence[int] | None = None,
) -> str:
    """The storage key, relative to the array, of the write unit at ``index`` (C order, the
    storage grid: shards for a sharded zarr v3 array). zarr v2 joins with
    ``dimension_separator`` (default ``.``); zarr v3 ``default`` encoding gives ``c/0/1/2``
    with its separator (default ``/``) and ``v2`` encoding ``0.1.2``; N5 is x first; precomputed
    is ``x0-x1_y0-y1_z0-z1`` in the scale's voxel coordinates, ``origin`` (its
    ``voxel_offset``) included and the last range clipped to ``shape``."""
    index = tuple(int(i) for i in index)
    if format == "zarr2":
        return (separator or ".").join(str(i) for i in index)
    if format == "zarr3":
        if encoding == "default":
            sep = separator or "/"
            return sep.join(["c", *(str(i) for i in index)])
        if encoding == "v2":
            return (separator or ".").join(str(i) for i in index)
        raise ValueError(f"zarr v3 chunk_key_encoding {encoding!r}; use 'default' or 'v2'")
    if format == "n5":
        return "/".join(str(i) for i in reversed(index))
    if format == "precomputed":
        if chunk_shape is None or shape is None:
            raise ValueError("precomputed chunk keys need chunk_shape and shape")
        origin = tuple(int(o) for o in origin) if origin is not None else (0,) * len(index)
        parts = []
        for i, c, n, o in zip(
            reversed(index), reversed(chunk_shape), reversed(shape), reversed(origin)
        ):
            start = o + i * int(c)
            stop = min(start + int(c), o + int(n))
            parts.append(f"{start}-{stop}")
        return "_".join(parts)
    raise ValueError(f"unknown format {format!r}")


def parse_chunk_key(
    key: str,
    format: str,
    separator: str | None = None,
    encoding: str = "default",
    chunk_shape: Sequence[int] | None = None,
    shape: Sequence[int] | None = None,
    origin: Sequence[int] | None = None,
    ndim: int | None = None,
) -> tuple[int, ...] | None:
    """``chunk_key`` inverted: the index of a write unit, or None for a key that is not a
    chunk (``zarr.json``, ``.zarray``, ``attributes.json``, ``info``, a ``c/`` key of the
    wrong arity)."""
    key = str(key)
    if format == "zarr2" or (format == "zarr3" and encoding == "v2"):
        parts = key.split(separator or ".")
        found = _digits(parts)
    elif format == "zarr3":
        parts = key.split(separator or "/")
        found = _digits(parts[1:]) if parts and parts[0] == "c" and len(parts) > 1 else None
    elif format == "n5":
        found = _digits(key.split("/"))
        found = found[::-1] if found is not None else None
    elif format == "precomputed":
        if chunk_shape is None or shape is None or not _PRECOMPUTED_KEY.match(key):
            return None
        ranges = [tuple(int(v) for v in r.split("-")) for r in key.split("_")]
        if len(ranges) != len(chunk_shape):
            return None
        origin = tuple(int(o) for o in origin) if origin is not None else (0,) * len(ranges)
        idx = []
        for (start, _stop), c, o in zip(ranges, reversed(chunk_shape), reversed(origin)):
            if (start - o) % int(c) != 0 or start < o:
                return None
            idx.append((start - o) // int(c))
        found = tuple(idx[::-1])
    else:
        raise ValueError(f"unknown format {format!r}")
    if found is None or (ndim is not None and len(found) != ndim):
        return None
    return found


def _digits(parts: Sequence[str]) -> tuple[int, ...] | None:
    if not parts or not all(p.isdigit() for p in parts):
        return None
    return tuple(int(p) for p in parts)


# ---------------------------------------------------------------------------
# Codecs


def codec_metadata(codecs, format: str) -> dict:
    """The array-metadata members for ``codecs`` in the format's dialect: ``{"codecs":
    [...]}`` for zarr v3, ``{"compressor": ...}`` for zarr v2. ``"zstd"`` is level 3,
    ``"none"`` is uncompressed; a list is a zarr v3 codecs list and a dict a zarr v2
    compressor, taken as given."""
    if codecs in ("zstd", None):
        if format == "zarr3":
            return {
                "codecs": [
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    {"name": "zstd", "configuration": {"level": ZSTD_LEVEL}},
                ]
            }
        return {"compressor": {"id": "zstd", "level": ZSTD_LEVEL}}
    if codecs == "none":
        if format == "zarr3":
            return {"codecs": [{"name": "bytes", "configuration": {"endian": "little"}}]}
        return {"compressor": None}
    if isinstance(codecs, list):
        if format != "zarr3":
            raise ValueError("a codecs list is zarr v3's dialect; zarr v2 takes a compressor dict")
        return {"codecs": list(codecs)}
    if isinstance(codecs, dict):
        if format != "zarr2":
            raise ValueError("a compressor dict is zarr v2's dialect; zarr v3 takes a codecs list")
        return {"compressor": dict(codecs)}
    raise ValueError(
        f"codecs {codecs!r}; use 'zstd', 'none', a zarr v3 codecs list or a zarr v2 compressor dict"
    )


def codecs_of(node: Node | str) -> tuple[Any, tuple[int, ...] | None]:
    """An existing array's codecs in its format's dialect, and its shard shape (None when
    unsharded): what ``add_level`` copies from the level below, and what a mirror of an
    array passes to ``create(codecs=, shards=)`` (cellmap-flow's sync). A path is detected."""
    if isinstance(node, str):
        node = detect(node)
    if node.format == "zarr2":
        return node.docs[".zarray"].get("compressor"), None
    if node.format == "zarr3":
        doc = node.docs["zarr.json"]
        codecs = list(doc.get("codecs", []) or [])
        for codec in codecs:
            if codec.get("name") == "sharding_indexed":
                conf = codec["configuration"]
                return list(conf.get("codecs", [])), tuple(
                    doc["chunk_grid"]["configuration"]["chunk_shape"]
                )
        return codecs, None
    raise ValueError(f"{node.path}: the package writes zarr v2 and v3 only, not {node.format}")


# ---------------------------------------------------------------------------
# Group documents (Behaviours 11 and 12)


def read_group_attrs(loc: Location, format: str) -> tuple[dict, dict | None]:
    """``(attributes, raw zarr.json)`` of a group: the raw document is kept for zarr v3 so a
    rewrite changes nothing but the attributes."""
    if format == "zarr3":
        doc = loc.read_json("zarr.json") or {}
        return dict(doc.get("attributes", {}) or {}), doc
    return dict(loc.read_json(".zattrs") or {}), None


def write_group_attrs(loc: Location, format: str, attrs: dict, raw: dict | None) -> None:
    if format == "zarr3":
        doc = {"zarr_format": 3, "node_type": "group", **(raw or {}), "attributes": attrs}
        loc.write_json("zarr.json", doc)
        return
    if loc.read(".zgroup") is None:
        loc.write_json(".zgroup", {"zarr_format": 2})
    loc.write_json(".zattrs", attrs)


def _multiscales_slot(attrs: dict) -> str | None:
    """Where the multiscales list lives: ``"ome"`` (0.5 nesting) or ``"top"`` (0.4), None
    when the group has none."""
    ome = attrs.get("ome")
    if isinstance(ome, dict) and ome.get("multiscales"):
        return "ome"
    if attrs.get("multiscales"):
        return "top"
    return None


def merged_group_attrs(
    attrs: dict,
    format: str,
    name: str,
    voxel_size,
    translation,
    axes: Sequence[str],
    units,
    axis_types: Sequence[str | None] | None = None,
    group_attrs: dict | None = None,
) -> dict:
    """``attrs`` with the dataset entry for ``name`` added or replaced in the group's first
    multiscale, the datasets kept in order of voxel size, and nothing else touched. A group
    without multiscales gets a fresh document in the format's OME version. ``group_attrs``
    are merged on top."""
    axes = tuple(str(a) for a in axes)
    kinds = axis_kinds(axes, axis_types)
    merged = dict(attrs)
    slot = _multiscales_slot(merged)
    if slot is None:
        version = "0.4" if format == "zarr2" else "0.5"
        fresh = ome_multiscales(axes, units, [(name, voxel_size, translation)], version, axis_types)
        if version == "0.4":
            merged["multiscales"] = fresh["multiscales"]
        else:
            merged["ome"] = {
                **(merged.get("ome") if isinstance(merged.get("ome"), dict) else {}),
                **fresh["ome"],
            }
    else:
        container = merged["ome"] if slot == "ome" else merged
        multiscales = list(container["multiscales"])
        ms = dict(multiscales[0])
        existing_axes = tuple(
            str(a["name"] if isinstance(a, dict) else a) for a in ms.get("axes", [])
        )
        if existing_axes and existing_axes != axes:
            raise ValueError(
                f"the group's multiscales axes are {existing_axes}; the level's are {axes}"
            )
        existing_units = tuple(
            str((a.get("unit") if isinstance(a, dict) else None) or "") for a in ms.get("axes", [])
        )
        wanted_units = axis_units(units, axes, kinds)
        if existing_units and any(
            e != w for e, w, k in zip(existing_units, wanted_units, kinds) if k in (SPACE, TIME)
        ):
            raise ValueError(
                f"the group's axis units are {existing_units}; the level's are {wanted_units}"
            )
        wanted = dataset_entry(name, voxel_size, translation, kinds)
        current = [d for d in ms.get("datasets", []) if str(d.get("path", "")).strip("/") == name]
        if not (len(current) == 1 and same_entry(current[0], wanted)):
            others = [d for d in ms.get("datasets", []) if d not in current]
            ms["datasets"] = sorted([*others, wanted], key=_dataset_fineness)
            multiscales[0] = ms
            if slot == "ome":
                merged["ome"] = {**container, "multiscales": multiscales}
            else:
                merged["multiscales"] = multiscales
    if group_attrs:
        merged.update(group_attrs)
    return merged


def _entry_placement(dataset: dict) -> tuple[list[float], list[float] | None]:
    scale: list[float] = []
    translation: list[float] | None = None
    for t in dataset.get("coordinateTransformations", []) or []:
        if t.get("type") == "scale":
            scale = [float(v) for v in t["scale"]]
        elif t.get("type") == "translation":
            translation = [float(v) for v in t["translation"]]
    return scale, translation


def same_entry(a: dict, b: dict, rtol: float = 1e-6) -> bool:
    """Two dataset entries that say the same: same path, same scale and translation within
    ``rtol`` (a missing translation is zeros). What lets an adopt write nothing."""
    if str(a.get("path", "")).strip("/") != str(b.get("path", "")).strip("/"):
        return False
    sa, ta = _entry_placement(a)
    sb, tb = _entry_placement(b)
    if len(sa) != len(sb) or not sa:
        return False
    ta = ta if ta is not None else [0.0] * len(sa)
    tb = tb if tb is not None else [0.0] * len(sb)
    return bool(
        np.allclose(sa, sb, rtol=rtol, atol=0.0) and np.allclose(ta, tb, rtol=rtol, atol=1e-9)
    )


def _dataset_fineness(dataset: dict) -> float:
    for t in dataset.get("coordinateTransformations", []) or []:
        if t.get("type") == "scale":
            return float(np.prod([float(v) for v in t["scale"]]))
    return float("inf")


# ---------------------------------------------------------------------------
# Creation (Behaviour 11)


def group_format(group_path: str, format: str | None) -> tuple[str, list[str]]:
    """The format to write at ``group_path`` and the ancestors lacking a group document,
    outermost first, ``group_path`` itself included when it has none. The walk goes up from
    ``group_path`` until a node exists: a zarr v2 or v3 group decides the format (a
    contradicting ``format`` raises; an array, an N5 or a precomputed node raise). When no
    ancestor exists the walk ends at the nearest ancestor named ``*.zarr``, the store root,
    and ``format`` or zarr v3 decides; with no such ancestor either, only ``group_path``
    gets a document (Behaviour 11; cellmap-analyze's and cellmap-flow's containers)."""
    group_path = normalize_path(group_path)
    missing: list[str] = []
    p: str | None = group_path
    while p is not None and not is_root(p):
        gnode = detect(p, missing_ok=True)
        if gnode is not None:
            if gnode.kind != "group":
                raise ValueError(f"{p} is an array; a level's parent must be a group")
            if gnode.format not in ("zarr2", "zarr3"):
                raise ValueError(
                    f"{p} is {gnode.format}; the package writes zarr v2 and v3 groups only"
                )
            if format is not None and format != gnode.format:
                raise ValueError(
                    f"{p} is a {gnode.format} group; format={format!r} would mix zarr versions"
                )
            return gnode.format, missing[::-1]
        missing.append(p)
        if basename(p).endswith(".zarr"):
            return format or "zarr3", missing[::-1]
        p = parent(p)
    return format or "zarr3", [group_path]


def write_bare_group(path: str, format: str) -> None:
    """A group document with no attributes: what a missing ancestor gets."""
    loc = Location(path)
    if format == "zarr3":
        loc.write_json("zarr.json", {"zarr_format": 3, "node_type": "group"})
    else:
        loc.write_json(".zgroup", {"zarr_format": 2})


def ensure_group_entry(
    group_path: str,
    name: str,
    voxel_size,
    translation,
    axes: Sequence[str],
    units="nanometer",
    axis_types: Sequence[str | None] | None = None,
    group_attrs: dict | None = None,
    format: str | None = None,
) -> None:
    """List the level ``name`` in the group document at ``group_path``: its dataset entry
    added, or replaced when it says something else, nothing else touched; the document, and
    any missing ancestor's up to the ``.zarr`` root, created when absent (``format`` as in
    ``create``). An entry that already says the same, placement within rtol 1e-6, leaves
    the document unwritten, so adopting an existing array writes nothing. The write is a
    read-modify-write with no lock, re-read afterwards: an entry lost to another process's
    concurrent write is written again, ``RACE_ATTEMPTS`` times, then ``RuntimeError``
    (Behaviour 12). Public for an array the package did not create: mesh-n-bone's symlinked
    ``s0``, a level filled before it is listed."""
    group_path = normalize_path(group_path)
    fmt, missing = group_format(group_path, format)
    for p in missing:
        if p != group_path:
            write_bare_group(p, fmt)
    loc = Location(group_path)
    for _ in range(RACE_ATTEMPTS):
        attrs, raw = read_group_attrs(loc, fmt)
        merged = merged_group_attrs(
            attrs, fmt, name, voxel_size, translation, axes, units, axis_types, group_attrs
        )
        if json.loads(json.dumps(merged)) == attrs:
            return  # the group already says this
        write_group_attrs(loc, fmt, merged, raw)
    raise RuntimeError(
        f"{group_path}: the dataset entry for {name!r} was lost {RACE_ATTEMPTS} times to another "
        "process writing the group document; create and add_level on one group belong to one "
        "process (Behaviour 12)"
    )


def array_spec(
    kvstore: dict,
    format: str,
    shape: Sequence[int],
    dtype,
    chunks: Sequence[int],
    shards: Sequence[int] | None,
    fill_value,
    codecs,
    axes: Sequence[str],
    attrs: dict | None,
    separator: str | None,
    store_fill: bool,
) -> dict:
    dtype = np.dtype(dtype)
    spec: dict[str, Any] = {
        "driver": _DRIVER[format],
        "kvstore": kvstore,
        "store_data_equal_to_fill_value": bool(store_fill),
    }
    members = codec_metadata(codecs, format)
    if format == "zarr3":
        grid = list(shards) if shards is not None else list(chunks)
        codec_list = members["codecs"]
        if shards is not None:
            codec_list = [
                {
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": list(chunks),
                        "codecs": codec_list,
                        "index_codecs": [
                            {"name": "bytes", "configuration": {"endian": "little"}},
                            {"name": "crc32c"},
                        ],
                    },
                }
            ]
        metadata: dict[str, Any] = {
            "shape": list(shape),
            "data_type": dtype.name,
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": grid}},
            "codecs": codec_list,
            "fill_value": _json_fill(fill_value, dtype),
            "dimension_names": [str(a) for a in axes],
        }
        if separator is not None:
            metadata["chunk_key_encoding"] = {
                "name": "default",
                "configuration": {"separator": separator},
            }
        if attrs:
            metadata["attributes"] = dict(attrs)
    else:
        metadata = {
            "shape": list(shape),
            "chunks": list(chunks),
            "dtype": dtype.str
            if dtype.byteorder != "=" or dtype.itemsize == 1
            else dtype.newbyteorder("<").str,
            "compressor": members["compressor"],
            "fill_value": _json_fill(fill_value, dtype),
            "dimension_separator": separator or ".",
        }
    spec["metadata"] = metadata
    return spec


def _json_fill(fill_value, dtype: np.dtype):
    if fill_value is None:
        return 0 if dtype.kind != "b" else False
    if isinstance(fill_value, float) and math.isnan(fill_value):
        return "NaN"
    if dtype.kind == "b":
        return bool(fill_value)
    return float(fill_value) if dtype.kind == "f" else int(fill_value)


def create_array(
    path: str,
    shape: Sequence[int],
    dtype,
    voxel_size,
    translation,
    *,
    axes: Sequence[str],
    units,
    chunks: Sequence[int],
    shards: Sequence[int] | None,
    fill_value,
    codecs,
    format: str | None,
    attrs: dict | None,
    group_attrs: dict | None,
    exists: str,
    store_fill: bool,
    separator: str | None,
    axis_types: Sequence[str | None] | None,
    context: ts.Context,
) -> tuple[str, str, bool]:
    """Write the group document, then the array. Returns ``(path, format, adopted)``:
    ``adopted`` is True when another process created the array between the probe and the
    attempt and ``exists == "adopt"`` (the caller then verifies it). Raises
    ``FileExistsError`` when something is already at ``path`` and ``exists == "error"``."""
    path = normalize_path(path)
    name, group_path = path.rsplit("/", 1)[-1], path.rsplit("/", 1)[0]
    if exists not in ("error", "adopt"):
        raise ValueError("exists is 'error' or 'adopt'")
    shape = tuple(int(n) for n in shape)
    chunks = tuple(int(c) for c in chunks)
    if len(chunks) != len(shape):
        raise ValueError(f"chunks {chunks} must have one entry per axis of {shape}")
    if shards is not None:
        shards = tuple(int(s) for s in shards)
        if len(shards) != len(shape) or any(s % c for s, c in zip(shards, chunks)):
            raise ValueError(
                f"shards {shards} must be a whole number of chunks {chunks} on every axis"
            )
    axes = tuple(str(a) for a in axes)
    if len(axes) != len(shape):
        raise ValueError(f"axes {axes} must name every axis of {shape}")
    kinds = axis_kinds(axes, axis_types)
    vs = per_axis(voxel_size, kinds, 1.0, f"{path}: voxel_size")
    tr = per_axis(translation, kinds, 0.0, f"{path}: translation")

    fmt, _missing = group_format(group_path, format)
    if fmt == "zarr2" and shards is not None:
        raise ValueError("shards are zarr v3 only")
    if detect(path, missing_ok=True) is not None:
        if exists == "error":
            raise FileExistsError(
                f"{path} exists; the package never overwrites (exists='adopt' opens an identical array)"
            )
        return path, fmt, True

    # the group documents first (Behaviour 11), then the array
    ensure_group_entry(group_path, name, vs, tr, axes, units, axis_types, group_attrs, format=fmt)

    def attempt(kvstore: dict) -> ts.TensorStore:
        spec = array_spec(
            kvstore,
            fmt,
            shape,
            dtype,
            chunks,
            shards,
            fill_value,
            codecs,
            axes,
            attrs,
            separator,
            store_fill,
        )
        return ts.open(spec, create=True, open=False, context=context).result()

    try:
        with_access(kvstore_specs(path), attempt)
    except Exception as e:  # noqa: BLE001
        if "ALREADY_EXISTS" not in str(e) or is_refused(e):
            raise
        if exists == "error":
            raise FileExistsError(f"{path} was created by another process meanwhile") from e
        _wait_for(path)
        return path, fmt, True
    if fmt == "zarr2" and attrs:
        Location(path).write_json(".zattrs", dict(attrs))
    return path, fmt, False


def _wait_for(path: str) -> None:
    """Re-read a path another process is creating: its metadata appears late on NRS."""
    for k in range(1, RACE_ATTEMPTS + 1):
        if detect(path, missing_ok=True) is not None:
            return
        time.sleep(RACE_DELAY * k)
    raise FileExistsError(
        f"{path}: another process started creating it and its metadata has not appeared"
    )


def differences(
    meta,
    *,
    shape: Sequence[int],
    dtype,
    chunks: Sequence[int],
    shards: Sequence[int] | None,
    voxel_size: Sequence[float],
    translation: Sequence[float],
    attrs: dict | None = None,
) -> list[str]:
    """How an existing array's ``ArrayMeta`` differs from what a caller wanted to create:
    one line per field, empty when identical (placement within ``RTOL``). Of the array's
    attributes only the keys of ``attrs`` are compared, as JSON stores them; every other key
    is someone else's."""
    out = []
    if tuple(meta.shape) != tuple(int(n) for n in shape):
        out.append(f"shape: found {tuple(meta.shape)}, wanted {tuple(shape)}")
    if np.dtype(meta.dtype) != np.dtype(dtype):
        out.append(f"dtype: found {np.dtype(meta.dtype)}, wanted {np.dtype(dtype)}")
    if tuple(meta.chunk_shape) != tuple(int(c) for c in chunks):
        out.append(f"chunks: found {tuple(meta.chunk_shape)}, wanted {tuple(chunks)}")
    found_shards = tuple(meta.shard_shape) if meta.shard_shape is not None else None
    wanted_shards = tuple(int(s) for s in shards) if shards is not None else None
    if found_shards != wanted_shards:
        out.append(f"shards: found {found_shards}, wanted {wanted_shards}")
    for what, found, wanted in (
        ("voxel_size", meta.voxel_size, voxel_size),
        ("translation", meta.translation, translation),
    ):
        f, w = np.asarray(found, dtype=float), np.asarray(wanted, dtype=float)
        if f.shape != w.shape or not np.allclose(f, w, rtol=RTOL, atol=RTOL):
            out.append(
                f"{what}: found {tuple(float(x) for x in f)}, wanted {tuple(float(x) for x in w)}"
            )
    found_attrs = meta.attrs or {}
    for key, wanted in json.loads(json.dumps(attrs or {})).items():
        if key not in found_attrs:
            out.append(f"attrs[{key!r}]: missing, wanted {wanted!r}")
        elif found_attrs[key] != wanted:
            out.append(f"attrs[{key!r}]: found {found_attrs[key]!r}, wanted {wanted!r}")
    return out
