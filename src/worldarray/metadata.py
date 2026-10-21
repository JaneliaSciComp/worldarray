"""What an array is and where it sits: ``ArrayMeta``, read per format by one precedence.

The rule everything else rests on: a **translation is the centre of voxel 0** (OME-NGFF,
COSEM ``transform``, CF coordinates, Neuroglancer's own zarr reader). Formats that store
voxel 0's corner are moved half a voxel along the spatial axes on reading: precomputed's
``voxel_offset`` and funlib's ``offset`` (0 when only a ``resolution`` is given). N5 levels
with ``pixelResolution`` and ``downsamplingFactors`` and no offset follow BigDataViewer: a
downsampled voxel's centre is the centre of the block it covers (spec, Conventions 1 and 4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import numpy as np
import tensorstore as ts

from . import units as U
from .store import Location, Node, basename, parent

logger = logging.getLogger(__name__)

DEFAULT_AXES = ("t", "c", "z", "y", "x")
SPATIAL_NAMES = frozenset({"z", "y", "x"})
RTOL = 1e-6


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
        unit is taken as nanometres; an unknown unit on an axis being converted raises."""
        sp = set(self.spatial)
        units, vs, tr = list(self.units), list(self.voxel_size), list(self.translation)
        for i, unit in enumerate(self.units):
            k = U.kind(unit)
            target = None
            if i in sp and space is not None and k in ("space", "unknown"):
                target = space
                if unit == "":
                    logger.debug(
                        "%s: no unit on axis %r, taking nanometres", self.path, self.axes[i]
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


def corner_to_centre(corner, voxel_size, axes) -> tuple[float, ...]:
    sp = set(spatial_axes(axes))
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


def read_meta(node: Node, store: ts.TensorStore, *, scale_index: int | None = None) -> ArrayMeta:
    """The metadata of ``store``, an array ``node`` (or one scale of a precomputed node)
    opened by ``store.open_array``, so already in C order with its origin at 0."""
    shape = tuple(int(s) for s in store.shape)
    dtype = store.dtype.numpy_dtype
    chunk_shape = tuple(int(c) for c in store.chunk_layout.read_chunk.shape)
    ndim = len(shape)
    up = parent(node.path)

    if node.format == "precomputed":
        attrs, group_attrs, found = (
            {},
            node.docs["info"],
            _precomputed(node.docs["info"], scale_index or 0, ndim),
        )
    elif node.format == "n5":
        attrs = node.docs.get("attributes.json", {})
        group_attrs = (Location(up).read_json("attributes.json") or {}) if up else {}
        found = _n5(attrs, group_attrs, basename(node.path), ndim)
    else:
        attrs, group_attrs = _zarr_attrs(node, up)
        found = _zarr(attrs, group_attrs, basename(node.path), ndim, node)

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
    )


def _floats(values, ndim: int, what: str) -> tuple[float, ...]:
    values = tuple(float(v) for v in values)
    if len(values) != ndim:
        raise ValueError(
            f"{what} has {len(values)} entries for a {ndim}-dimensional array: {values}"
        )
    return values


# --- zarr ------------------------------------------------------------------


def _zarr_attrs(node: Node, up: str | None) -> tuple[dict, dict]:
    if node.format == "zarr3":
        attrs = node.docs["zarr.json"].get("attributes", {}) or {}
        group_doc = Location(up).read_json("zarr.json") if up else None
        group_attrs = (group_doc or {}).get("attributes", {}) or {}
    else:
        attrs = node.location.read_json(".zattrs") or {}
        group_attrs = (Location(up).read_json(".zattrs") or {}) if up else {}
    return attrs, group_attrs


def _multiscales(group_attrs: dict) -> list[dict]:
    ome = group_attrs.get("ome")
    if isinstance(ome, dict) and "multiscales" in ome:
        return list(ome["multiscales"])
    return list(group_attrs.get("multiscales", []) or [])


def _dataset_for(multiscales: list[dict], name: str) -> tuple[dict, dict] | None:
    """``(multiscale, dataset)`` whose dataset ``path`` is ``name``: matched by name, never by
    position, so an array the group does not list borrows no other level's scale."""
    for ms in multiscales:
        for ds in ms.get("datasets", []) or []:
            if str(ds.get("path", "")).strip("/") == name:
                return ms, ds
    return None


def _zarr(attrs: dict, group_attrs: dict, name: str, ndim: int, node: Node) -> _Resolved:
    found = _Resolved()
    hit = _dataset_for(_multiscales(group_attrs), name)
    if hit is not None:
        found = _ome(hit[0], hit[1], ndim, node.path)
    if found.voxel_size is None:
        found = _array_attrs(attrs, ndim, node.path)
    else:
        _check_against_attrs(attrs, found, ndim, node.path)
    if found.axes is None:
        dims = attrs.get("_ARRAY_DIMENSIONS") or node.docs.get("zarr.json", {}).get(
            "dimension_names"
        )
        if dims and len(dims) == ndim and all(d for d in dims):
            found.axes = tuple(str(d) for d in dims)
    if found.voxel_size is None and node.format == "zarr3" and attrs.get("dimension_units"):
        units, vs = _dimension_units(attrs["dimension_units"], ndim)
        found.units, found.voxel_size = units, vs
    return found


def _dimension_units(spec: list, ndim: int) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """tensorstore's ``dimension_units`` attribute: ``"4nm"``, ``null``, ``""``."""
    import re

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


def _units(values, ndim: int) -> tuple[str, ...]:
    out = tuple("" if v is None else str(v) for v in values)
    if len(out) != ndim:
        raise ValueError(f"units has {len(out)} entries for a {ndim}-dimensional array: {out}")
    return out


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
        axes, units, tuple(float(v) for v in scale), tuple(float(v) for v in translation)
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
    (``transform``) attributes on the array itself, C order."""
    found = _Resolved()
    transform = attrs.get("transform")
    if isinstance(transform, dict) and "scale" in transform:
        return _cosem_transform(transform, ndim, path)
    vs = attrs.get("voxel_size", attrs.get("resolution"))
    if vs is None:
        if attrs.get("axis_names"):
            found.axes = (
                tuple(str(a) for a in attrs["axis_names"])
                if len(attrs["axis_names"]) == ndim
                else None
            )
        return found
    found.voxel_size = _floats(vs, ndim, f"{path}: voxel_size")
    if attrs.get("axis_names") and len(attrs["axis_names"]) == ndim:
        found.axes = tuple(str(a) for a in attrs["axis_names"])
    if attrs.get("units") is not None:
        found.units = _units(
            attrs["units"] if isinstance(attrs["units"], list) else [attrs["units"]] * ndim, ndim
        )
    offset = attrs.get("offset", [0.0] * ndim)
    corner = _floats(offset, ndim, f"{path}: offset")
    found.translation = corner_to_centre(corner, found.voxel_size, found.axes or default_axes(ndim))
    return found


def _cosem_transform(transform: dict, ndim: int, path: str) -> _Resolved:
    order = transform.get("ordering", "C")
    rev = (lambda v: tuple(v)[::-1]) if order == "F" else tuple
    scale = _floats(rev(transform["scale"]), ndim, f"{path}: transform.scale")
    translate = _floats(
        rev(transform.get("translate", [0.0] * ndim)), ndim, f"{path}: transform.translate"
    )
    axes = tuple(str(a) for a in rev(transform["axes"])) if transform.get("axes") else None
    units = _units(rev(transform["units"]), ndim) if transform.get("units") else None
    return _Resolved(axes, units, scale, translate)  # translate is already a centre


def _check_against_attrs(attrs: dict, found: _Resolved, ndim: int, path: str) -> None:
    """An OME translation beside a funlib ``offset``: equal (cellmap-analyze's writer) or a
    corner half a voxel below (funlib's writer) pass silently; anything else warns. A
    disagreeing voxel size raises: that is how stale attributes are caught."""
    assert found.voxel_size is not None and found.translation is not None
    vs = np.asarray(found.voxel_size)
    attr_vs = attrs.get("voxel_size", attrs.get("resolution"))
    if (
        attr_vs is not None
        and len(attr_vs) == ndim
        and not np.allclose(attr_vs, vs, rtol=RTOL, atol=0)
    ):
        raise ValueError(
            f"{path}: OME scale {found.voxel_size} disagrees with the array's voxel_size {list(attr_vs)}"
        )
    offset = attrs.get("offset")
    if offset is None or len(offset) != ndim:
        return
    offset = np.asarray(offset, dtype=float)
    tr = np.asarray(found.translation)
    axes = found.axes or default_axes(ndim)
    corner = np.asarray(corner_to_centre(offset, vs, axes))
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


def _n5(attrs: dict, group_attrs: dict, name: str, ndim: int) -> _Resolved:
    """COSEM ``transform`` on the array; else the group's ``multiscales[].datasets[].transform``
    for this path; else ``pixelResolution``/``resolution`` (array, else group) times the
    level's ``downsamplingFactors``, plus ``offset``. Lists other than ``transform`` are
    x-first and reversed to C order here."""
    transform = attrs.get("transform")
    if isinstance(transform, dict) and "scale" in transform:
        return _cosem_transform(transform, ndim, name)
    hit = _dataset_for(group_attrs.get("multiscales", []) or [], name)
    if hit is not None and isinstance(hit[1].get("transform"), dict):
        return _cosem_transform(hit[1]["transform"], ndim, name)

    unit: str | None = None
    base = None
    for source in (attrs, group_attrs):
        if "pixelResolution" in source:
            pr = source["pixelResolution"]
            if isinstance(pr, dict):
                base, unit = pr.get("dimensions"), pr.get("unit")
            else:
                base = pr
            break
        if "resolution" in source:
            base = source["resolution"]
            break
    if base is None:
        return _Resolved()
    base = np.asarray(_floats(base, ndim, f"{name}: pixelResolution"))[::-1]  # x-first -> C
    factors = np.asarray(
        _floats(attrs.get("downsamplingFactors", [1] * ndim), ndim, f"{name}: downsamplingFactors")
    )[::-1]
    voxel_size = base * factors

    units_list = attrs.get("units", group_attrs.get("units"))
    if unit is None and units_list is not None:
        units = _units(list(units_list)[::-1], ndim)
    else:
        units = tuple(str(unit or "") for _ in range(ndim))
    axes = default_axes(ndim)

    if "offset" in attrs:
        corner = np.asarray(_floats(attrs["offset"], ndim, f"{name}: offset"))[::-1]
        translation = corner_to_centre(corner, voxel_size, axes)
    else:
        # BigDataViewer: the downsampled voxel's centre is the centre of the block it covers
        translation = tuple(float(v) for v in (factors - 1) / 2 * base)
    return _Resolved(None, units, tuple(float(v) for v in voxel_size), tuple(translation))


# --- precomputed -------------------------------------------------------------


def _precomputed(info: dict, scale_index: int, ndim: int) -> _Resolved:
    scale = info["scales"][scale_index]
    resolution = [float(v) for v in scale["resolution"]][::-1]  # xyz nm -> zyx
    voxel_offset = [float(v) for v in scale.get("voxel_offset", [0, 0, 0])][::-1]
    spatial_tr = [(o + 0.5) * r for o, r in zip(voxel_offset, resolution)]
    if ndim == 4:  # channel kept (more than one)
        return _Resolved(
            ("c", "z", "y", "x"), ("", "nm", "nm", "nm"), (1.0, *resolution), (0.0, *spatial_tr)
        )
    return _Resolved(("z", "y", "x"), ("nm", "nm", "nm"), tuple(resolution), tuple(spatial_tr))
