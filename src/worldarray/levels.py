"""The levels of a pyramid, and choosing one for a voxel size.

A group's levels are the paths in its ``multiscales[0].datasets`` (OME-NGFF, COSEM N5), in
that order; without that, ``s0, s1, ...`` are probed. A precomputed volume's levels are its
``scales``, in order. Levels are matched by name, never by position (spec, Conventions 4;
Behaviour 3).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .metadata import ArrayMeta
from .store import Location, Node, basename, detect, join, parent

Mode = Literal["floor", "exact", "nearest", "resample"]


@dataclass(frozen=True)
class LevelRef:
    name: str
    path: str  # the array path (for precomputed, the volume root)
    scale_index: int | None = None  # precomputed only


def _group_multiscale_paths(node: Node) -> list[str]:
    if node.format == "zarr3":
        attrs = node.docs["zarr.json"].get("attributes", {}) or {}
    elif node.format == "zarr2":
        attrs = node.location.read_json(".zattrs") or {}
    else:
        attrs = node.docs.get("attributes.json", {}) or {}
    ome = attrs.get("ome")
    multiscales = (
        ome["multiscales"]
        if isinstance(ome, dict) and "multiscales" in ome
        else attrs.get("multiscales")
    )
    if not multiscales:
        return []
    return [str(ds["path"]).strip("/") for ds in multiscales[0].get("datasets", []) if "path" in ds]


def _probe_levels(path: str) -> list[str]:
    names: list[str] = []
    for i in range(64):
        name = f"s{i}"
        docs = Location(join(path, name)).read_many(("zarr.json", ".zarray", "attributes.json"))
        if all(v is None for v in docs.values()):
            break
        names.append(name)
    return names


def discover(node: Node) -> list[LevelRef]:
    """The levels of a group node, or of the group an array node belongs to (just the array
    itself when its parent is not a multiscale group)."""
    if node.format == "precomputed":
        return [
            LevelRef(str(s.get("key", i)), node.path, i)
            for i, s in enumerate(node.docs["info"]["scales"])
        ]
    if node.kind == "array":
        up = parent(node.path)
        if up is not None:
            try:
                group = detect(up)
            except FileNotFoundError:
                group = None
            if (
                group is not None
                and group.kind == "group"
                and basename(node.path) in _group_multiscale_paths(group)
            ):
                return discover(group)
        return [LevelRef(basename(node.path), node.path)]
    names = _group_multiscale_paths(node) or _probe_levels(node.path)
    return [LevelRef(name, join(node.path, name)) for name in names]


def _spatial_voxel_size(meta: ArrayMeta) -> np.ndarray:
    return np.asarray([meta.voxel_size[i] for i in meta.spatial], dtype=float)


def same_voxel_size(a, b) -> bool:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=1e-6, atol=0))


def coarser_anywhere(resolution, target) -> bool:
    r, t = np.asarray(resolution, dtype=float), np.asarray(target, dtype=float)
    return bool(np.any((r > t) & ~np.isclose(r, t, rtol=1e-6, atol=0)))


def select_level(
    levels: Sequence[tuple[str, ArrayMeta]], voxel_size, mode: Mode = "floor"
) -> tuple[str, ArrayMeta]:
    """The ``(name, meta)`` to read at ``voxel_size`` (spatial axes, the pyramid's units).

    - ``floor``: the level at ``voxel_size``, else the coarsest level that is no coarser than
      it on any axis, else the first.
    - ``exact``: the level at ``voxel_size``, else ``ValueError`` listing the levels.
    - ``nearest``: the level closest on a log scale summed over the axes; the finer on a tie.
    - ``resample``: the level to resample from: the level at ``voxel_size`` if there is one;
      else the coarsest level that is no coarser on any axis; else the finest.
    """
    if not levels:
        raise ValueError("no levels to choose from")
    target = np.asarray(voxel_size, dtype=float)
    sizes = [(name, meta, _spatial_voxel_size(meta)) for name, meta in levels]
    if target.shape != sizes[0][2].shape:
        raise ValueError(
            f"voxel_size has {target.size} entries, the levels have {sizes[0][2].size} spatial axes"
        )
    for name, meta, vs in sizes:
        if same_voxel_size(vs, target):
            return name, meta
    if mode == "exact":
        raise ValueError(f"no level at voxel size {target.tolist()}; levels: " + _describe(sizes))
    if mode in ("floor", "resample"):
        candidates = [(name, meta) for name, meta, vs in sizes if not coarser_anywhere(vs, target)]
        if not candidates:
            return levels[0]
        # the coarsest of them: largest voxel volume
        return max(candidates, key=lambda nm: float(np.prod(_spatial_voxel_size(nm[1]))))
    if mode == "nearest":

        def distance(entry):
            vs = entry[2]
            return float(np.sum(np.abs(np.log(vs) - np.log(target)))), float(np.prod(vs))

        name, meta, _ = min(sizes, key=distance)
        return name, meta
    raise ValueError(f"unknown mode {mode!r}")


def _describe(sizes) -> str:
    return ", ".join(f"{name} {vs.tolist()}" for name, _, vs in sizes)
