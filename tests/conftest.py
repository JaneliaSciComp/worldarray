"""Synthetic stores, written through tensorstore, for every format the package reads.

``build_case`` turns one entry of ``tests/fixtures/metadata.json`` into a store under
``tmp_path`` and returns the path to open. The fixture describes stores by their attribute
documents so chunkmirage's TypeScript side can build the same ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tensorstore as ts

FIXTURES = Path(__file__).parent / "fixtures"


def kv(root: Path) -> ts.KvStore:
    return ts.KvStore.open({"driver": "file", "path": str(root) + "/"}).result()


def write_json(root: Path, key: str, obj: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    kv(root).write(key, json.dumps(obj).encode()).result()


def read_json(root: Path, key: str) -> dict:
    return json.loads(kv(root).read(key).result().value)


def write_zarr2(
    path: Path, data: np.ndarray, *, attrs: dict | None = None, chunks=None, compressor=None
) -> None:
    spec = {
        "driver": "zarr",
        "kvstore": {"driver": "file", "path": str(path) + "/"},
        "metadata": {
            "shape": list(data.shape),
            "chunks": list(chunks or data.shape),
            "dtype": data.dtype.str,
            "compressor": compressor,
        },
    }
    ts.open(spec, create=True, delete_existing=True).result().write(data).result()
    if attrs is not None:
        write_json(path, ".zattrs", attrs)


def write_zarr2_group(path: Path, attrs: dict | None = None) -> None:
    write_json(path, ".zgroup", {"zarr_format": 2})
    if attrs is not None:
        write_json(path, ".zattrs", attrs)


def write_zarr3(
    path: Path, data: np.ndarray, *, attrs: dict | None = None, chunks=None, dimension_names=None
) -> None:
    metadata = {
        "shape": list(data.shape),
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": list(chunks or data.shape)},
        },
        "data_type": str(data.dtype),
    }
    if attrs is not None:
        metadata["attributes"] = attrs
    if dimension_names is not None:
        metadata["dimension_names"] = list(dimension_names)
    spec = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(path) + "/"},
        "metadata": metadata,
    }
    ts.open(spec, create=True, delete_existing=True).result().write(data).result()


def write_zarr3_group(path: Path, attrs: dict | None = None) -> None:
    write_json(
        path, "zarr.json", {"zarr_format": 3, "node_type": "group", "attributes": attrs or {}}
    )


def write_n5(path: Path, data: np.ndarray, *, attrs: dict | None = None, chunks=None) -> None:
    """``data`` is in C order; N5 stores x-first, so tensorstore sees it transposed."""
    shape_f = list(data.shape[::-1])
    spec = {
        "driver": "n5",
        "kvstore": {"driver": "file", "path": str(path) + "/"},
        "metadata": {
            "dimensions": shape_f,
            "blockSize": list((chunks or data.shape)[::-1]),
            "dataType": str(data.dtype),
            "compression": {"type": "raw"},
        },
    }
    ts.open(spec, create=True, delete_existing=True).result().write(
        np.ascontiguousarray(data.T)
    ).result()
    if attrs:
        doc = read_json(path, "attributes.json")
        doc.update(attrs)
        write_json(path, "attributes.json", doc)


def write_n5_group(path: Path, attrs: dict | None = None) -> None:
    write_json(path, "attributes.json", {"n5": "2.5.0", **(attrs or {})})


def write_precomputed(path: Path, scales: list[dict]) -> None:
    """``scales``: dicts with ``data`` (C order zyx), ``resolution`` and ``voxel_offset`` (xyz)."""
    for i, scale in enumerate(scales):
        data = scale["data"]
        spec = {
            "driver": "neuroglancer_precomputed",
            "kvstore": {"driver": "file", "path": str(path) + "/"},
            "multiscale_metadata": {
                "data_type": str(data.dtype),
                "num_channels": 1,
                "type": "image",
            },
            "scale_metadata": {
                "size": list(data.shape[::-1]),
                "resolution": scale["resolution"],
                "voxel_offset": scale.get("voxel_offset", [0, 0, 0]),
                "chunk_size": list(data.shape[::-1]),
                "encoding": "raw",
                "key": scale.get("key", f"s{i}"),
            },
        }
        modes = (
            {"create": True, "delete_existing": True} if i == 0 else {"create": True, "open": True}
        )
        store = ts.open(spec, **modes).result()
        store.write(np.ascontiguousarray(data.T)[..., np.newaxis]).result()


def ome_multiscales(axes, datasets, *, version="0.4", top_transforms=None, units=None) -> dict:
    """An OME-NGFF multiscales document. ``datasets``: ``(path, scale, translation)``."""
    axis_docs = []
    for a in axes:
        doc = {"name": a, "type": "space" if a in "zyx" else ("time" if a == "t" else "channel")}
        unit = (units or {}).get(a, "nanometer" if a in "zyx" else None)
        if unit:
            doc["unit"] = unit
        axis_docs.append(doc)
    ds_docs = []
    for path, scale, translation in datasets:
        cts = [{"type": "scale", "scale": list(scale)}]
        if translation is not None:
            cts.append({"type": "translation", "translation": list(translation)})
        ds_docs.append({"path": path, "coordinateTransformations": cts})
    ms = {"version": version, "axes": axis_docs, "datasets": ds_docs}
    if top_transforms:
        ms["coordinateTransformations"] = top_transforms
    return (
        {"multiscales": [ms]}
        if version == "0.4"
        else {"ome": {"version": version, "multiscales": [ms]}}
    )


def pattern(shape, dtype="uint16") -> np.ndarray:
    return (np.arange(int(np.prod(shape))).reshape(shape) % 251).astype(dtype)


def build_case(tmp_path: Path, case: dict) -> str:
    """The path to open for one fixture case."""
    data = pattern(case["shape"], case.get("dtype", "uint16"))
    fmt = case["format"]
    name = case.get("array", "s0")
    group = tmp_path / case["name"]
    array = group / name
    if fmt == "zarr2":
        write_zarr2_group(group, case.get("group_attrs"))
        write_zarr2(array, data, attrs=case.get("attrs"), compressor=case.get("compressor"))
        if case.get("patch_zarray"):
            doc = read_json(array, ".zarray")
            doc["compressor"].update(case["patch_zarray"])
            write_json(array, ".zarray", doc)
    elif fmt == "zarr3":
        write_zarr3_group(group, case.get("group_attrs"))
        write_zarr3(
            array, data, attrs=case.get("attrs"), dimension_names=case.get("dimension_names")
        )
    elif fmt == "n5":
        write_n5_group(group, case.get("group_attrs"))
        write_n5(array, data, attrs=case.get("attrs"))
    elif fmt == "precomputed":
        write_precomputed(
            group,
            [
                {
                    "data": data,
                    "resolution": case["resolution"],
                    "voxel_offset": case.get("voxel_offset", [0, 0, 0]),
                }
            ],
        )
        return str(group / "s0")
    else:
        raise ValueError(fmt)
    return str(array)


@pytest.fixture(scope="session")
def metadata_cases() -> list[dict]:
    return json.loads((FIXTURES / "metadata.json").read_text())["cases"]


@pytest.fixture
def cube(tmp_path: Path) -> tuple[str, np.ndarray]:
    """One OME zarr v2 level: 4 nm voxels, translation (2, 2, 2), so the corner is the origin."""
    data = pattern((6, 8, 10))
    group = tmp_path / "cube.zarr"
    write_zarr2_group(group, ome_multiscales("zyx", [("s0", (4, 4, 4), (2, 2, 2))]))
    write_zarr2(group / "s0", data, chunks=(3, 4, 5))
    return str(group / "s0"), data


@pytest.fixture
def pyramid(tmp_path: Path) -> tuple[str, dict[str, np.ndarray]]:
    """Three OME zarr v2 levels at 4, 8 and 16 nm, corner at the origin, each level the
    mean-free subsample of the one below (values are the level's own pattern)."""
    group = tmp_path / "pyr.zarr"
    levels = {"s0": pattern((16, 16, 16)), "s1": pattern((8, 8, 8)), "s2": pattern((4, 4, 4))}
    datasets = [
        ("s0", (4, 4, 4), (2, 2, 2)),
        ("s1", (8, 8, 8), (4, 4, 4)),
        ("s2", (16, 16, 16), (8, 8, 8)),
    ]
    write_zarr2_group(group, ome_multiscales("zyx", datasets))
    for name, data in levels.items():
        write_zarr2(group / name, data)
    return str(group), levels
