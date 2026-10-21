"""What an array is and where it sits: ``ArrayMeta``, read per format by one precedence.

The rule everything else rests on: a **translation is the centre of voxel 0** (OME-NGFF,
COSEM ``transform``, CF coordinates, Neuroglancer's own zarr reader). Formats that store
voxel 0's corner are moved half a voxel along the spatial axes on reading: precomputed's
``voxel_offset`` and funlib's ``offset`` (0 when only a ``resolution`` is given). N5 levels
with ``pixelResolution`` and ``downsamplingFactors`` and no offset follow BigDataViewer: a
downsampled voxel's centre is the centre of the block it covers (spec, Conventions 1 and 4).

Per-axis lists shorter than the array (a 3-entry ``voxel_size`` on a ``(c, z, y, x)``
prediction) cover the trailing axes; the leading axes get 1.0, 0.0 or ``""``. OME lists
must match the array exactly.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, replace

import numpy as np
import tensorstore as ts

from . import units as U
from .store import Location, Node, basename, multiscales_of, parent

logger = logging.getLogger(__name__)

DEFAULT_AXES = ("t", "c", "z", "y", "x")
SPATIAL_NAMES = frozenset({"z", "y", "x"})
RTOL = 1e-6
PLACED_BY = ("ome", "cosem", "funlib", "n5", "precomputed", "dimension_units")

_warned_unitless: set[str] = set()


@dataclass(frozen=True)
class ArrayMeta:
    """One entry per axis, in the array's C order. The same field names as chunkmirage's
    ``ArrayInfo`` so its adapter is a one-liner."""

    shape: tuple[int, ...]
    dtype: np.dtype
    chunk_shape: tuple[int, ...]
    axes: tuple[str, ...]
    axes_named: bool
    units: tuple[str, ...]  # "" when the store gives none; any string is accepted
    voxel_size: tuple[float, ...]  # one index step, in `units`; 1.0 on unitless axes
    translation: tuple[float, ...]  # the CENTRE of voxel 0, in `units`
    format: str  # zarr2 | zarr3 | n5 | precomputed
    path: str
    parent: str | None = None
    attrs: dict = field(default_factory=dict)
    group_attrs: dict = field(default_factory=dict)
    placed_by: str | None = None  # which rule gave voxel_size and translation; None: nothing did

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", np.dtype(self.dtype))
        n = len(self.shape)
        for name in ("chunk_shape", "axes", "units", "voxel_size", "translation"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} must have length {n}, got {getattr(self, name)}")
        object.__setattr__(self, "voxel_size", tuple(float(v) for v in self.voxel_size))
        object.__setattr__(self, "translation", tuple(float(v) for v in self.translation))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def spatial(self) -> tuple[int, ...]:
        """Indices of the axes named z, y or x when any are; else the last three."""
        return spatial_axes(self.axes)

    @property
    def corner(self) -> tuple[float, ...]:
        """``translation - voxel_size / 2`` along the spatial axes, unchanged elsewhere."""
        sp = set(self.spatial)
        return tuple(
            t - v / 2 if i in sp else t
            for i, (t, v) in enumerate(zip(self.translation, self.voxel_size))
        )

    def in_units(self, space: str | None = None, time: str | None = None) -> ArrayMeta:
        """The same metadata with spatial axes in ``space`` and time axes in ``time``.
        Axes not asked about keep their unit string, whatever it is. A spatial axis with no
        unit is taken as nanometres, with a warning once per array (a BigDataViewer N5 with
        a unitless micrometre ``resolution`` would otherwise be read a thousandfold too
        small, silently); an unknown unit on an axis being converted raises."""
        sp = set(self.spatial)
        units, vs, tr = list(self.units), list(self.voxel_size), list(self.translation)
        for i, unit in enumerate(self.units):
            k = U.kind(unit)
            target = None
            if i in sp and space is not None and k in ("space", "unknown"):
                target = space
                if unit == "" and self.path not in _warned_unitless:
                    _warned_unitless.add(self.path)
                    logger.warning(
                        "%s: no unit stored on spatial axis %r; taking nanometres (pass units=None to keep the store's numbers)",
                        self.path,
                        self.axes[i],
                    )
                f = U.factor(unit, space, missing_as=U.NANOMETER)
            elif k == "time" and time is not None:
                target = time
                f = U.factor(unit, time)
            if target is not None:
                units[i], vs[i], tr[i] = target, vs[i] * f, tr[i] * f
        return replace(self, units=tuple(units), voxel_size=tuple(vs), translation=tuple(tr))

    def in_nm(self) -> ArrayMeta:
        return self.in_units(space=U.NANOMETER)

    def drop_axis(self, axis: int) -> ArrayMeta:
        keep = [i for i in range(self.ndim) if i != axis]
        pick = lambda t: tuple(t[i] for i in keep)  # noqa: E731
        return replace(
            self,
            shape=pick(self.shape),
            chunk_shape=pick(self.chunk_shape),
            axes=pick(self.axes),
            units=pick(self.units),
            voxel_size=pick(self.voxel_size),
            translation=pick(self.translation),
        )


def spatial_axes(axes: tuple[str, ...]) -> tuple[int, ...]:
    named = tuple(i for i, a in enumerate(axes) if a in SPATIAL_NAMES)
    if named:
        return named
    return tuple(range(max(0, len(axes) - 3), len(axes)))


def default_axes(ndim: int) -> tuple[str, ...]:
    if ndim <= len(DEFAULT_AXES):
        return DEFAULT_AXES[len(DEFAULT_AXES) - ndim :]
    return tuple(f"dim_{i}" for i in range(ndim - len(DEFAULT_AXES))) + DEFAULT_AXES


def corner_to_centre(corner, voxel_size, axes=None) -> tuple[float, ...]:
    """Voxel 0's centre from its corner: half a voxel on along the spatial axes (those named
    z, y or x; the last three when ``axes`` is None or names none of them)."""
    corner, voxel_size = tuple(corner), tuple(voxel_size)
    sp = set(spatial_axes(tuple(axes) if axes is not None else default_axes(len(corner))))
    return tuple(
        float(c) + (float(v) / 2 if i in sp else 0.0)
        for i, (c, v) in enumerate(zip(corner, voxel_size))
    )


# ---------------------------------------------------------------------------
# Reading


@dataclass
class _Resolved:
    """What a parser found: per-axis tuples, each None when the store said nothing."""

    axes: tuple[str, ...] | None = None
    units: tuple[str, ...] | None = None
    voxel_size: tuple[float, ...] | None = None
    translation: tuple[float, ...] | None = None
    placed_by: str | None = None


def read_meta(
    node: Node,
    store: ts.TensorStore,
    *,
    scale_index: int | None = None,
    conflicts: str = "error",
) -> ArrayMeta:
    """The metadata of ``store``, an array ``node`` (or one scale of a precomputed node)
    opened by ``store.open_array``, so already in C order with its origin at 0.
    ``conflicts``: what a per-array ``voxel_size`` that disagrees with the OME scale does,
    ``"error"`` (cellmap-analyze's choice, catches stale attributes) or ``"warn"`` (a server
    that would rather show the image)."""
    if conflicts not in ("error", "warn"):
        raise ValueError("conflicts is 'error' or 'warn'")
    shape = tuple(int(s) for s in store.shape)
    dtype = store.dtype.numpy_dtype
    chunk_shape = tuple(int(c) for c in store.chunk_layout.read_chunk.shape)
    ndim = len(shape)
    up = parent(node.path)

    if node.format == "precomputed":
        attrs, group_attrs = {}, node.docs["info"]
        found = _precomputed(node.docs["info"], scale_index or 0, ndim)
    elif node.format == "n5":
        attrs = node.docs.get("attributes.json", {})
        group_attrs = (Location(up).read_json("attributes.json") or {}) if up else {}
        found = _n5(attrs, group_attrs, node, ndim)
    else:
        attrs, group_attrs = _zarr_attrs(node, up)
        found = _zarr(attrs, group_attrs, basename(node.path), ndim, node, conflicts)

    axes_named = found.axes is not None
    axes = tuple(str(a) for a in found.axes) if found.axes is not None else default_axes(ndim)
    return ArrayMeta(
        shape=shape,
        dtype=dtype,
        chunk_shape=chunk_shape,
        axes=axes,
        axes_named=axes_named,
        units=found.units if found.units is not None else ("",) * ndim,
        voxel_size=found.voxel_size if found.voxel_size is not None else (1.0,) * ndim,
        translation=found.translation if found.translation is not None else (0.0,) * ndim,
        format=node.format,
        path=node.path
        if scale_index is None
        else f"{node.path}/{node.docs['info']['scales'][scale_index]['key']}",
        parent=up,
        attrs=attrs,
        group_attrs=group_attrs,
        placed_by=found.placed_by if found.voxel_size is not None else None,
    )


def _floats(values, ndim: int, what: str, pad: float | None = None) -> tuple[float, ...]:
    """``values`` as floats, one per axis. Shorter lists cover the trailing axes and the
    leading ones get ``pad``; with ``pad`` None (OME) the length must match."""
    values = tuple(float(v) for v in values)
    if len(values) < ndim and pad is not None:
        values = (float(pad),) * (ndim - len(values)) + values
    if len(values) != ndim:
        raise ValueError(
            f"{what} has {len(values)} entries for a {ndim}-dimensional array: {values}"
        )
    return values


def _units(values, ndim: int, pad: str | None = None) -> tuple[str, ...]:
    out = tuple("" if v is None else str(v) for v in values)
    if len(out) < ndim and pad is not None:
        out = (pad,) * (ndim - len(out)) + out
    if len(out) != ndim:
        raise ValueError(f"units has {len(out)} entries for a {ndim}-dimensional array: {out}")
    return out


def _axes(names, ndim: int) -> tuple[str, ...] | None:
    """Axis names, the leading axes defaulted when the list is short; None when it is empty
    or too long."""
    names = tuple(str(a) for a in names or ())
    if not names or len(names) > ndim:
        return None
    return default_axes(ndim)[: ndim - len(names)] + names


# --- zarr ------------------------------------------------------------------


def _zarr_attrs(node: Node, up: str | None) -> tuple[dict, dict]:
    if node.format == "zarr3":
        attrs = node.docs["zarr.json"].get("attributes", {}) or {}
    else:
        attrs = node.location.read_json(".zattrs") or {}
    return attrs, _group_attrs_at(up, node.format)


def _group_attrs_at(path: str | None, fmt: str) -> dict:
    """A group's user attributes. zarr groups are read in both formats, because a zarr v3
    array inside a v2 group whose ``.zattrs`` holds the OME multiscales (cellmap-analyze
    writes these) is a layout in the wild; the array's own format wins where both exist."""
    if path is None:
        return {}
    loc = Location(path)
    if fmt == "n5":
        return loc.read_json("attributes.json") or {}
    docs = loc.read_many(("zarr.json", ".zattrs"))
    v3 = (json.loads(docs["zarr.json"]) if docs["zarr.json"] else {}).get("attributes", {}) or {}
    v2 = json.loads(docs[".zattrs"]) if docs[".zattrs"] else {}
    return {**v2, **v3} if fmt == "zarr3" else {**v3, **v2}


def _ome_hit(node: Node, group_attrs: dict, max_depth: int = 4) -> tuple[dict, dict] | None:
    """The ``(multiscale, dataset)`` listing this array: in the parent, or in a further
    ancestor when the dataset path has more than one component (``"seg/s0"``). A parent that
    lists levels but not this array gets a warning, and the array is placed by its own
    attributes or not at all."""
    name = basename(node.path)
    hit = _dataset_for(multiscales_of(group_attrs), name)
    if hit is not None:
        return hit
    ancestor, rel = parent(node.path), name
    for _ in range(max_depth - 1):
        if ancestor is None or parent(ancestor) is None:
            break
        rel = f"{basename(ancestor)}/{rel}"
        ancestor = parent(ancestor)
        hit = _dataset_for(multiscales_of(_group_attrs_at(ancestor, node.format)), rel)
        if hit is not None:
            return hit
    if multiscales_of(group_attrs):
        logger.warning(
            "%s: the parent group's multiscales list no dataset named %r; the array is placed by its own attributes, or not at all",
            node.path,
            name,
        )
    return None


def _dataset_for(multiscales: list[dict], name: str) -> tuple[dict, dict] | None:
    """``(multiscale, dataset)`` whose dataset ``path`` is ``name``: matched by name, never by
    position, so an array the group does not list borrows no other level's scale."""
    for ms in multiscales:
        for ds in ms.get("datasets", []) or []:
            if str(ds.get("path", "")).strip("/") == name:
                return ms, ds
    return None


def _zarr(
    attrs: dict, group_attrs: dict, name: str, ndim: int, node: Node, conflicts: str
) -> _Resolved:
    found = _Resolved()
    hit = _ome_hit(node, group_attrs)
    if hit is not None:
        found = _ome(hit[0], hit[1], ndim, node.path)
    if found.voxel_size is None:
        found = _array_attrs(attrs, ndim, node.path)
    else:
        _check_against_attrs(attrs, found, ndim, node.path, conflicts)
    if found.axes is None:
        dims = attrs.get("_ARRAY_DIMENSIONS") or node.docs.get("zarr.json", {}).get(
            "dimension_names"
        )
        if dims and len(dims) == ndim and all(d for d in dims):
            found.axes = tuple(str(d) for d in dims)
    if found.voxel_size is None and node.format == "zarr3" and attrs.get("dimension_units"):
        units, vs = _dimension_units(attrs["dimension_units"], ndim)
        found.units, found.voxel_size, found.placed_by = units, vs, "dimension_units"
    return found


def _dimension_units(spec: list, ndim: int) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """tensorstore's ``dimension_units`` attribute: ``"4nm"``, ``null``, ``""``."""
    units, vs = [], []
    for entry in spec:
        if not entry:
            units.append("")
            vs.append(1.0)
            continue
        m = re.match(r"^\s*([-+0-9.eE]*)\s*(.*)$", str(entry))
        vs.append(float(m.group(1)) if m and m.group(1) else 1.0)
        units.append(m.group(2).strip() if m else "")
    return _units(units, ndim), _floats(vs, ndim, "dimension_units")


def _ome(multiscale: dict, dataset: dict, ndim: int, path: str) -> _Resolved:
    """OME-NGFF 0.4 and 0.5: the dataset's own transforms, then the multiscale-level ones.
    0.6 (``coordinateSystems``): the intrinsic system only, chunkmirage's rule."""
    transforms = list(dataset.get("coordinateTransformations", []) or [])
    systems = multiscale.get("coordinateSystems")
    if systems:
        axes_spec = _ome06_system(multiscale, transforms, systems).get("axes", [])
        level_transforms: list[dict] = []
        transforms = _unpack_sequence(transforms)
    else:
        axes_spec = multiscale.get("axes", []) or []
        level_transforms = list(multiscale.get("coordinateTransformations", []) or [])

    scale = np.ones(ndim)
    translation = np.zeros(ndim)
    for t in transforms + level_transforms:
        kind = t.get("type")
        if kind == "scale":
            s = np.asarray(_floats(t["scale"], ndim, f"{path}: OME scale"))
            scale, translation = scale * s, translation * s
        elif kind == "translation":
            translation = translation + np.asarray(
                _floats(t["translation"], ndim, f"{path}: OME translation")
            )
        # identity and anything else: no effect (0.6's other types lead to other systems)

    axes: tuple[str, ...] | None = None
    units: tuple[str, ...] | None = None
    if axes_spec and len(axes_spec) == ndim:
        axes = tuple(str(a["name"] if isinstance(a, dict) else a) for a in axes_spec)
        units = tuple(
            str((a.get("unit") if isinstance(a, dict) else None) or "") for a in axes_spec
        )
    return _Resolved(
        axes, units, tuple(float(v) for v in scale), tuple(float(v) for v in translation), "ome"
    )


def _ome06_system(multiscale: dict, transforms: list[dict], systems: list[dict]) -> dict:
    wanted = next(
        (t.get("output") for t in transforms if isinstance(t, dict) and t.get("output")), None
    )
    for system in systems:
        if wanted is None or system.get("name") == wanted:
            return system
    return systems[0]


def _unpack_sequence(transforms: list[dict]) -> list[dict]:
    out: list[dict] = []
    for t in transforms:
        if t.get("type") == "sequence":
            out.extend(_unpack_sequence(list(t.get("transformations", []) or [])))
        elif t.get("type") in ("scale", "translation"):
            out.append(t)
    return out


def _array_attrs(attrs: dict, ndim: int, path: str) -> _Resolved:
    """funlib (``resolution``/``voxel_size``, ``offset``, ``units``, ``axis_names``) or COSEM
    (``transform``) attributes on the array itself, C order. An ``offset`` with no voxel
    size places the array with voxels of 1."""
    found = _Resolved()
    transform = attrs.get("transform")
    if isinstance(transform, dict) and "scale" in transform:
        return _cosem_transform(transform, ndim, path)
    found.axes = _axes(attrs.get("axis_names"), ndim)
    vs = attrs.get("voxel_size", attrs.get("resolution"))
    offset = attrs.get("offset")
    if vs is None and offset is None:
        return found
    found.voxel_size = _floats(
        vs if vs is not None else [1.0] * ndim, ndim, f"{path}: voxel_size", pad=1.0
    )
    if attrs.get("units") is not None:
        raw = attrs["units"] if isinstance(attrs["units"], list) else [attrs["units"]] * ndim
        found.units = _units(raw, ndim, pad="")
    corner = _floats(
        offset if offset is not None else [0.0] * ndim, ndim, f"{path}: offset", pad=0.0
    )
    found.translation = corner_to_centre(corner, found.voxel_size, found.axes)
    found.placed_by = "funlib"
    return found


def _cosem_transform(transform: dict, ndim: int, path: str) -> _Resolved:
    order = transform.get("ordering", "C")
    rev = (lambda v: tuple(v)[::-1]) if order == "F" else tuple
    scale = _floats(rev(transform["scale"]), ndim, f"{path}: transform.scale", pad=1.0)
    translate = _floats(
        rev(transform.get("translate", [0.0] * ndim)), ndim, f"{path}: transform.translate", pad=0.0
    )
    axes = _axes(rev(transform["axes"]), ndim) if transform.get("axes") else None
    units = _units(rev(transform["units"]), ndim, pad="") if transform.get("units") else None
    return _Resolved(axes, units, scale, translate, "cosem")  # translate is already a centre


def _check_against_attrs(
    attrs: dict, found: _Resolved, ndim: int, path: str, conflicts: str
) -> None:
    """An OME translation beside a funlib ``offset``: equal (cellmap-analyze's writer) or a
    corner half a voxel below (funlib's writer) pass silently; anything else warns. A
    disagreeing voxel size raises (or warns, per ``conflicts``): that is how stale
    attributes are caught."""
    assert found.voxel_size is not None and found.translation is not None
    vs = np.asarray(found.voxel_size)
    attr_vs = attrs.get("voxel_size", attrs.get("resolution"))
    if attr_vs is not None and len(attr_vs) <= ndim:
        padded = np.asarray(_floats(attr_vs, ndim, f"{path}: voxel_size", pad=1.0))
        if not np.allclose(padded, vs, rtol=RTOL, atol=0):
            message = f"{path}: OME scale {found.voxel_size} disagrees with the array's voxel_size {list(attr_vs)}"
            if conflicts == "error":
                raise ValueError(message)
            logger.warning("%s; using the OME scale", message)
    offset = attrs.get("offset")
    if offset is None or len(offset) > ndim:
        return
    offset = np.asarray(_floats(offset, ndim, f"{path}: offset", pad=0.0))
    tr = np.asarray(found.translation)
    corner = np.asarray(corner_to_centre(offset, vs, found.axes))
    tol = RTOL * np.maximum(np.abs(vs), 1.0)
    if np.all(np.abs(offset - tr) <= tol) or np.all(np.abs(corner - tr) <= tol):
        return
    logger.warning(
        "%s: offset %s is neither the OME translation %s nor the corner below it; using the translation",
        path,
        offset.tolist(),
        tr.tolist(),
    )


# --- N5 --------------------------------------------------------------------


def _n5(attrs: dict, group_attrs: dict, node: Node, ndim: int) -> _Resolved:
    """COSEM ``transform`` on the array; else the group's ``multiscales[].datasets[].transform``
    for this path; else ``pixelResolution``/``resolution`` (array, else group) times the
    level's ``downsamplingFactors``, plus ``offset``. Lists other than ``transform`` are
    x-first and reversed to C order here. Without an ``offset``: a funlib ``resolution``
    means a corner at 0 (translation half a voxel); a ``pixelResolution`` follows
    BigDataViewer, ``(f - 1) / 2`` full-resolution voxels on."""
    name = basename(node.path)
    transform = attrs.get("transform")
    if isinstance(transform, dict) and "scale" in transform:
        return _cosem_transform(transform, ndim, name)
    hit = _ome_hit(node, group_attrs)
    if hit is not None and isinstance(hit[1].get("transform"), dict):
        return _cosem_transform(hit[1]["transform"], ndim, name)

    axes = _axes(list(attrs.get("axes", group_attrs.get("axes", [])) or [])[::-1], ndim)
    unit: str | None = None
    base, key = None, None
    for source in (attrs, group_attrs):
        if "pixelResolution" in source:
            pr = source["pixelResolution"]
            base, unit = (
                (pr.get("dimensions"), pr.get("unit")) if isinstance(pr, dict) else (pr, None)
            )
            key = "pixelResolution"
            break
        if "resolution" in source:
            base, key = source["resolution"], "resolution"
            break
    if base is None:
        return _Resolved(axes)
    base = np.asarray(_floats(list(base)[::-1], ndim, f"{name}: {key}", pad=1.0))  # x-first -> C
    factors = np.asarray(
        _floats(
            list(attrs.get("downsamplingFactors", [1]))[::-1],
            ndim,
            f"{name}: downsamplingFactors",
            pad=1.0,
        )
    )
    voxel_size = base * factors

    units_list = attrs.get("units", group_attrs.get("units"))
    if unit is None and units_list is not None:
        units = _units(list(units_list)[::-1], ndim, pad="")
    else:
        units = tuple(
            (str(unit or "") if i in set(spatial_axes(axes or default_axes(ndim))) else "")
            for i in range(ndim)
        )

    if "offset" in attrs:
        corner = _floats(list(attrs["offset"])[::-1], ndim, f"{name}: offset", pad=0.0)
        translation = corner_to_centre(corner, voxel_size, axes)
    elif key == "resolution":
        translation = corner_to_centre((0.0,) * ndim, voxel_size, axes)  # funlib: a corner at 0
    else:
        # BigDataViewer: the downsampled voxel's centre is the centre of the block it covers
        translation = tuple(float(v) for v in (factors - 1) / 2 * base)
    return _Resolved(axes, units, tuple(float(v) for v in voxel_size), tuple(translation), "n5")


# --- precomputed -------------------------------------------------------------


def _precomputed(info: dict, scale_index: int, ndim: int) -> _Resolved:
    scale = info["scales"][scale_index]
    resolution = [float(v) for v in scale["resolution"]][::-1]  # xyz nm -> zyx
    voxel_offset = [float(v) for v in scale.get("voxel_offset", [0, 0, 0])][::-1]
    spatial_tr = [(o + 0.5) * r for o, r in zip(voxel_offset, resolution)]
    if ndim == 4:  # channel kept (more than one)
        return _Resolved(
            ("c", "z", "y", "x"),
            ("", "nm", "nm", "nm"),
            (1.0, *resolution),
            (0.0, *spatial_tr),
            "precomputed",
        )
    return _Resolved(
        ("z", "y", "x"), ("nm", "nm", "nm"), tuple(resolution), tuple(spatial_tr), "precomputed"
    )
