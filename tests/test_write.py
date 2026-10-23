"""Writing: Convention 6 and Behaviours 10 to 16 (spec, section 5 "Writing (0.2)")."""

from __future__ import annotations

import json
import multiprocessing as mp
import pickle

import numpy as np
import pytest
import tensorstore as ts

import worldarray as wa
from conftest import (
    FIXTURES,
    ome_multiscales,
    pattern,
    write_n5,
    write_precomputed,
    write_zarr2,
    write_zarr2_group,
    write_zarr3,
    write_zarr3_group,
)

WRITE_CASES = json.loads((FIXTURES / "write.json").read_text())["cases"]


# --- Behaviour 10: documents round-trip -------------------------------------------------


def _fixture_call(case, units, axis_types):
    return wa.ome_multiscales(
        case["axes"],
        units,
        [tuple(lvl) for lvl in case["levels"]],
        version=case["version"],
        axis_types=axis_types,
        name=case.get("multiscale_name", ""),
        metadata=case.get("metadata"),
        type=case.get("multiscale_type"),
    )


@pytest.mark.parametrize("case", WRITE_CASES, ids=[c["name"] for c in WRITE_CASES])
def test_ome_multiscales_matches_the_fixture(case):
    """The broadcast inputs and their resolved per-axis form give the pinned document."""
    doc = _fixture_call(case, case["units"], case.get("axis_types"))
    assert json.loads(json.dumps(doc)) == case["expect"]
    resolved = case["resolved"]
    per_axis = _fixture_call(case, resolved["axis_units"], resolved["axis_types"])
    assert json.loads(json.dumps(per_axis)) == case["expect"]


def test_units_are_written_in_ome_names_and_unknown_ones_as_given():
    for spelling in ("nm", "nanometre", "Nanometers", "nanometer"):
        assert wa.units.ome_name(spelling) == "nanometer"
    assert wa.units.ome_name("um") == wa.units.ome_name("\u00b5m") == "micrometer"
    assert wa.units.ome_name("s") == "second"
    assert wa.units.ome_name("km") == wa.units.ome_name("Kilometres") == "kilometer"
    with pytest.raises(ValueError, match="unknown unit"):
        wa.units.ome_name("pixel")
    doc = wa.ome_multiscales(("z", "y", "x"), "pixel", [("s0", (1, 1, 1), (0.5,) * 3)])
    assert [a["unit"] for a in doc["ome"]["multiscales"][0]["axes"]] == ["pixel"] * 3
    doc = wa.ome_multiscales(("z", "y", "x"), "nm", [("s0", (1, 1, 1), (0.5,) * 3)], type="mean")
    ms = doc["ome"]["multiscales"][0]
    assert [a["unit"] for a in ms["axes"]] == ["nanometer"] * 3 and ms["type"] == "mean"
    plain = wa.ome_multiscales(("z", "y", "x"), "nm", [("s0", (1, 1, 1), (0.5,) * 3)])
    assert "type" not in plain["ome"]["multiscales"][0]


def test_level_shape_absorbs_float_factor_noise():
    f = 0.3 / 0.1  # 2.9999999999999996, chunkmirage's ratio of float voxel sizes
    assert wa.level_shape((9, 9, 9), f) == (3, 3, 3)
    assert wa.level_shape((10, 9, 8), 3) == (4, 3, 3)
    assert wa.level_shape((9, 9, 9), f, origin=3) == (2, 2, 2)
    assert wa.level_shape((9,), (9 + 1e-9,)) == (1,)


@pytest.mark.parametrize("fmt", ["zarr2", "zarr3"])
@pytest.mark.parametrize(
    "axes,vs,tr,shape,units",
    [
        (("z", "y", "x"), (8, 8, 8), (4, 4, 4), (8, 8, 8), "nanometer"),
        (("c", "z", "y", "x"), (8, 8, 8), (4, 4, 4), (2, 8, 8, 8), "nanometer"),  # spatial-length
        (
            ("t", "c", "z", "y", "x"),
            (1, 1, 4, 4, 4),
            (0, 0, 2, 2, 2),
            (3, 2, 8, 8, 8),
            {"space": "nanometer", "time": "second"},
        ),
    ],
)
def test_create_round_trips_placement(tmp_path, fmt, axes, vs, tr, shape, units):
    path = tmp_path / f"vol.{fmt}" / "s0"
    vol = wa.create(
        str(path), shape, "uint8", vs, tr, axes=axes, chunks=shape, format=fmt, units=units
    )
    assert vol.writable and vol.meta.format == fmt
    meta = wa.metadata(str(path))
    kinds = wa.write.axis_kinds(axes)
    assert meta.voxel_size == wa.write.per_axis(vs, kinds, 1.0, "vs")
    assert meta.translation == wa.write.per_axis(tr, kinds, 0.0, "tr")
    assert meta.axes == axes and meta.placed_by == "ome"
    attrs = wa.detect(str(path.parent)).attrs
    if fmt == "zarr2":
        assert attrs["multiscales"][0]["version"] == "0.4" and "ome" not in attrs
        axis_docs = attrs["multiscales"][0]["axes"]
    else:
        assert attrs["ome"]["version"] == "0.5" and "multiscales" not in attrs
        axis_docs = attrs["ome"]["multiscales"][0]["axes"]
    by_name = {a["name"]: a for a in axis_docs}
    assert "unit" not in by_name.get("c", {})  # units on space and time axes only
    if "t" in by_name:
        assert by_name["t"] == {"name": "t", "type": "time", "unit": "second"}


def test_level_placement_and_shape_invert_the_reader_rule():
    # the reader's N5 case: base 4 nm with the corner at the origin (translation 2), factor 2
    assert wa.level_placement((4, 4, 4), (2, 2, 2), 2) == ((8.0, 8.0, 8.0), (4.0, 4.0, 4.0))
    assert wa.level_shape((16, 16, 15), 2) == (8, 8, 8)
    # an origin moves the level by that many source voxels (mesh-n-bone's bug cannot be written)
    vs, tr = wa.level_placement((4, 4, 4), (2, 2, 2), 2, origin=16)
    assert vs == (8.0, 8.0, 8.0) and tr == (68.0, 68.0, 68.0)
    assert wa.level_shape((48, 48, 48), 2, origin=16) == (16, 16, 16)
    # per-axis factors, non-spatial axes untouched
    assert wa.level_placement((1, 4, 4), (0, 2, 2), (1, 2, 2)) == ((1.0, 8.0, 8.0), (0.0, 4.0, 4.0))


def test_level_placement_agrees_with_the_block_mean(tmp_path):
    """A level added by factors sits exactly where Behaviour 4's block mean of the source
    lands: reading the source over the level's box at the level's voxel size gives the mean
    of each 2x2x2 block, with no half-voxel shift."""
    data = pattern((16, 16, 16), "float32")
    group = tmp_path / "pyr.zarr"
    s0 = wa.create(
        str(group / "s0"), data.shape, data.dtype, (4, 4, 4), (2, 2, 2), chunks=(8, 8, 8)
    )
    s0.write_index((0, 0, 0), data.shape, data)
    s1 = wa.open_pyramid(str(group), mode="r+").add_level("s1", factors=2)
    assert (s1.shape, s1.voxel_size, s1.translation) == (
        (8, 8, 8),
        (8.0, 8.0, 8.0),
        (4.0, 4.0, 4.0),
    )
    mean = wa.open(str(group / "s0")).read(s1.box, voxel_size=s1.voxel_size, method="linear")
    expected = data.reshape(8, 2, 8, 2, 8, 2).mean(axis=(1, 3, 5))
    np.testing.assert_allclose(mean, expected)
    s1.write_index((0, 0, 0), s1.shape, mean.astype(np.float32))
    np.testing.assert_allclose(wa.open(str(group / "s1")).read_index((0, 0, 0), s1.shape), expected)


# --- Behaviour 11: create ---------------------------------------------------------------


def test_create_follows_the_group_format_and_refuses_a_contradiction(tmp_path):
    group = tmp_path / "v2.zarr"
    write_zarr2_group(group, ome_multiscales("zyx", [("s0", (4, 4, 4), (2, 2, 2))]))
    write_zarr2(group / "s0", pattern((8, 8, 8)))
    s1 = wa.create(str(group / "s1"), (4, 4, 4), "uint16", (8, 8, 8), (4, 4, 4), chunks=(4, 4, 4))
    assert s1.meta.format == "zarr2"
    attrs = wa.detect(str(group)).attrs
    assert [d["path"] for d in attrs["multiscales"][0]["datasets"]] == ["s0", "s1"]
    with pytest.raises(ValueError, match="mix zarr versions"):
        wa.create(
            str(group / "s2"),
            (2, 2, 2),
            "uint16",
            (16, 16, 16),
            (8, 8, 8),
            chunks=(2, 2, 2),
            format="zarr3",
        )
    # no group document: the format argument decides, zarr v3 by default
    fresh = wa.create(
        str(tmp_path / "new.zarr" / "s0"),
        (4, 4, 4),
        "uint8",
        (1, 1, 1),
        (0.5, 0.5, 0.5),
        chunks=(4, 4, 4),
    )
    assert fresh.meta.format == "zarr3" and wa.detect(str(tmp_path / "new.zarr")).format == "zarr3"


def test_create_requires_chunks_and_whole_shards(tmp_path):
    with pytest.raises(ValueError, match="chunks is required"):
        wa.create(str(tmp_path / "a.zarr/s0"), (8, 8, 8), "uint8", (1, 1, 1), (0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match="whole number of chunks"):
        wa.create(
            str(tmp_path / "b.zarr/s0"),
            (8, 8, 8),
            "uint8",
            (1, 1, 1),
            (0.5,) * 3,
            chunks=(4, 4, 4),
            shards=(6, 6, 6),
        )
    with pytest.raises(ValueError, match="zarr v3 only"):
        wa.create(
            str(tmp_path / "c.zarr/s0"),
            (8, 8, 8),
            "uint8",
            (1, 1, 1),
            (0.5,) * 3,
            chunks=(4, 4, 4),
            shards=(8, 8, 8),
            format="zarr2",
        )


def test_codecs_default_to_zstd_and_follow_the_dialect(tmp_path):
    v3 = wa.create(
        str(tmp_path / "v3.zarr/s0"), (8, 8, 8), "uint16", (1, 1, 1), (0.5,) * 3, chunks=(8, 8, 8)
    )
    codecs = wa.detect(v3.level_path).docs["zarr.json"]["codecs"]
    assert [c["name"] for c in codecs] == ["bytes", "zstd"] and codecs[1]["configuration"][
        "level"
    ] == 3
    v2 = wa.create(
        str(tmp_path / "v2.zarr/s0"),
        (8, 8, 8),
        "uint16",
        (1, 1, 1),
        (0.5,) * 3,
        chunks=(8, 8, 8),
        format="zarr2",
    )
    assert wa.detect(v2.level_path).docs[".zarray"]["compressor"] == {"id": "zstd", "level": 3}
    raw = wa.create(
        str(tmp_path / "raw.zarr/s0"),
        (8, 8, 8),
        "uint16",
        (1, 1, 1),
        (0.5,) * 3,
        chunks=(8, 8, 8),
        codecs="none",
    )
    assert [c["name"] for c in wa.detect(raw.level_path).docs["zarr.json"]["codecs"]] == ["bytes"]
    with pytest.raises(ValueError, match="dialect"):
        wa.create(
            str(tmp_path / "x.zarr/s0"),
            (8, 8, 8),
            "uint16",
            (1, 1, 1),
            (0.5,) * 3,
            chunks=(8, 8, 8),
            codecs={"id": "zstd", "level": 1},
        )


def test_create_never_overwrites_and_adopts_only_an_identical_array(tmp_path):
    path = str(tmp_path / "vol.zarr/s0")
    args = ((8, 8, 8), "uint16", (4, 4, 4), (2, 2, 2))
    first = wa.create(path, *args, chunks=(4, 4, 4))
    first.write_index((0, 0, 0), (4, 4, 4), np.full((4, 4, 4), 9, np.uint16))
    with pytest.raises(FileExistsError):
        wa.create(path, *args, chunks=(4, 4, 4))
    again = wa.create(path, *args, chunks=(4, 4, 4), exists="adopt")
    assert (
        again.writable and int(again.read_index((0, 0, 0), (4, 4, 4)).sum()) == 9 * 64
    )  # the data survived
    with pytest.raises(ValueError) as e:
        wa.create(path, (8, 8, 16), "uint8", (4, 4, 4), (2, 2, 2), chunks=(4, 4, 4), exists="adopt")
    message = str(e.value)
    assert (
        "shape: found (8, 8, 8), wanted (8, 8, 16)" in message
        and "dtype: found uint16, wanted uint8" in message
    )
    with pytest.raises(ValueError, match="translation: found"):
        wa.create(path, *args[:3], (3, 2, 2), chunks=(4, 4, 4), exists="adopt")


def test_create_keeps_everyone_elses_attributes(tmp_path):
    group = tmp_path / "shared.zarr"
    write_zarr3_group(
        group,
        {
            "fs": [4, 4, 4],
            "registry": {"owner": "someone"},
            **ome_multiscales("zyx", [("s0", (4, 4, 4), (2, 2, 2))], version="0.5"),
        },
    )
    before = wa.detect(str(group)).attrs
    vol = wa.create(
        str(group / "s1"),
        (4, 4, 4),
        "uint16",
        (8, 8, 8),
        (4, 4, 4),
        chunks=(4, 4, 4),
        attrs={"voxel_size": [8, 8, 8], "note": "legacy keys pass through"},
        group_attrs={"built_by": "test"},
    )
    after = wa.detect(str(group)).attrs
    assert after["fs"] == before["fs"] and after["registry"] == before["registry"]
    assert after["built_by"] == "test"
    assert [d["path"] for d in after["ome"]["multiscales"][0]["datasets"]] == ["s0", "s1"]
    assert (
        after["ome"]["multiscales"][0]["datasets"][0]
        == before["ome"]["multiscales"][0]["datasets"][0]
    )
    assert vol.meta.attrs["note"] == "legacy keys pass through" and vol.meta.attrs[
        "voxel_size"
    ] == [8, 8, 8]
    # the array's own attributes agree with the OME placement, so nothing warns or raises
    assert wa.metadata(vol.level_path).voxel_size == (8.0, 8.0, 8.0)


def test_fill_only_units_are_not_stored_unless_asked(tmp_path):
    vol = wa.create(
        str(tmp_path / "a.zarr/s0"), (8, 8, 8), "uint8", (1, 1, 1), (0.5,) * 3, chunks=(4, 4, 4)
    )
    vol.write_index((0, 0, 0), (4, 4, 4), np.zeros((4, 4, 4), np.uint8))
    vol.write_index((4, 4, 4), (8, 8, 8), np.ones((4, 4, 4), np.uint8))
    assert vol.list_chunks() == [(1, 1, 1)]
    kept = wa.create(
        str(tmp_path / "b.zarr/s0"),
        (8, 8, 8),
        "uint8",
        (1, 1, 1),
        (0.5,) * 3,
        chunks=(4, 4, 4),
        store_fill=True,
    )
    kept.write_index((0, 0, 0), (4, 4, 4), np.zeros((4, 4, 4), np.uint8))
    assert kept.list_chunks() == [(0, 0, 0)]


def test_a_lost_creation_race_is_adopted(tmp_path, monkeypatch):
    """The probe says absent, the store says present: another process got there first."""
    path = str(tmp_path / "race.zarr/s0")
    wa.create(path, (8, 8, 8), "uint16", (4, 4, 4), (2, 2, 2), chunks=(4, 4, 4))
    real_detect, calls = wa.write.detect, {"n": 0}

    def forgetful_detect(p, *, missing_ok=False):
        if p.endswith("/s0") and calls["n"] == 0:
            calls["n"] += 1
            return None  # the probe before the attempt misses it
        return real_detect(p, missing_ok=missing_ok)

    monkeypatch.setattr(wa.write, "detect", forgetful_detect)
    monkeypatch.setattr(wa.write, "RACE_DELAY", 0.0)
    vol = wa.create(
        path, (8, 8, 8), "uint16", (4, 4, 4), (2, 2, 2), chunks=(4, 4, 4), exists="adopt"
    )
    assert vol.writable and vol.shape == (8, 8, 8)
    calls["n"] = 0
    with pytest.raises(FileExistsError, match="another process"):
        wa.create(path, (8, 8, 8), "uint16", (4, 4, 4), (2, 2, 2), chunks=(4, 4, 4))


def test_open_for_write_and_pickling(tmp_path):
    path = str(tmp_path / "vol.zarr/s0")
    wa.create(path, (8, 8, 8), "uint16", (4, 4, 4), (2, 2, 2), chunks=(4, 4, 4))
    ro = wa.open(path)
    assert not ro.writable
    with pytest.raises(ValueError, match="read-only"):
        ro.write_index((0, 0, 0), (4, 4, 4), np.zeros((4, 4, 4), np.uint16))
    rw = wa.open(path, mode="r+")
    assert rw.writable and rw.voxel_size == (4.0, 4.0, 4.0)
    again = pickle.loads(pickle.dumps(rw))
    assert again.writable and again.translation == rw.translation and again.path == rw.path
    again.write_index((4, 4, 4), (8, 8, 8), np.full((4, 4, 4), 3, np.uint16))
    assert int(ro.read_index((4, 4, 4), (8, 8, 8)).sum()) == 3 * 64


# --- Behaviour 13: aligned writes -------------------------------------------------------


def test_write_index_alignment_rules(tmp_path):
    vol = wa.create(
        str(tmp_path / "vol.zarr/s0"), (10, 8, 8), "uint16", (1, 1, 1), (0.5,) * 3, chunks=(4, 4, 4)
    )
    vol.write_index((0, 0, 0), (4, 8, 8), np.ones((4, 8, 8), np.uint16))  # whole units
    vol.write_index((8, 0, 0), (10, 4, 4), np.ones((2, 4, 4), np.uint16))  # the clipped last unit
    with pytest.raises(ValueError, match=r"axis 1 \(y\) is not whole write units of 4"):
        vol.write_index((0, 2, 0), (4, 6, 4), np.ones((4, 4, 4), np.uint16))
    with pytest.raises(ValueError, match="nothing is cast"):
        vol.write_index((0, 0, 0), (4, 4, 4), np.ones((4, 4, 4), np.uint8))
    with pytest.raises(ValueError, match="outside the array"):
        vol.write_index((8, 0, 0), (12, 4, 4), np.ones((4, 4, 4), np.uint16))
    with pytest.raises(ValueError, match="data has shape"):
        vol.write_index((0, 0, 0), (4, 4, 4), np.ones((4, 4, 2), np.uint16))
    vol.write_index(
        (1, 1, 1), (3, 3, 3), np.full((2, 2, 2), 5, np.uint16), partial=True
    )  # read-modify-write
    got = vol.read_index((0, 0, 0), (4, 4, 4))
    assert got[1, 1, 1] == 5 and got[0, 0, 0] == 1 and got[3, 3, 3] == 1
    with pytest.raises(ValueError, match="no resampling on write"):
        wa.open(
            vol.level_path, mode="r+", voxel_size=(2, 2, 2), on_mismatch="resample"
        ).write_index((0, 0, 0), (2, 2, 2), np.zeros((2, 2, 2), np.uint16))


def _write_unit(args):
    path, index, value = args
    vol = wa.open(path, mode="r+")
    begin = tuple(i * c for i, c in zip(index, vol.write_shape))
    end = tuple(b + c for b, c in zip(begin, vol.write_shape))
    vol.write_index(begin, end, np.full(vol.write_shape, value, vol.dtype))
    return index


def test_two_processes_write_disjoint_units(tmp_path):
    path = str(tmp_path / "vol.zarr/s0")
    wa.create(path, (8, 8, 8), "uint16", (1, 1, 1), (0.5,) * 3, chunks=(4, 4, 4))
    jobs = [(path, idx, 1 + i) for i, idx in enumerate(np.ndindex(2, 2, 2))]
    with mp.get_context("spawn").Pool(2) as pool:
        done = pool.map(_write_unit, jobs)
    assert sorted(done) == sorted(j[1] for j in jobs)
    got = wa.open(path).read_index((0, 0, 0), (8, 8, 8))
    for i, idx in enumerate(np.ndindex(2, 2, 2)):
        sl = tuple(slice(4 * k, 4 * k + 4) for k in idx)
        assert np.all(got[sl] == 1 + i), idx


def test_write_by_world_box_uses_the_centre_rule(tmp_path):
    vol = wa.create(
        str(tmp_path / "vol.zarr/s0"), (8, 8, 8), "uint16", (4, 4, 4), (2, 2, 2), chunks=(4, 4, 4)
    )
    vol.write(
        ((0.0, 0.0, 0.0), (16.0, 16.0, 16.0)), np.full((4, 4, 4), 2, np.uint16)
    )  # voxels 0..3
    assert int(vol.read_index((0, 0, 0), (4, 4, 4)).sum()) == 2 * 64
    with pytest.raises(ValueError, match="outside the array"):
        vol.write(((16.0, 16.0, 16.0), (40.0, 40.0, 40.0)), np.zeros((6, 6, 6), np.uint16))


# --- Behaviour 14: add_level ------------------------------------------------------------


def test_add_level_places_sizes_orders_and_copies(tmp_path):
    group = tmp_path / "pyr.zarr"
    s0 = wa.create(
        str(group / "s0"),
        (16, 16, 16),
        "uint32",
        (4, 4, 4),
        (2, 2, 2),
        chunks=(4, 4, 4),
        shards=(8, 8, 8),
        fill_value=7,
    )
    pyr = wa.open_pyramid(str(group), mode="r+")
    s2 = pyr.add_level("s2", factors=4)
    s1 = pyr.add_level("s1", factors=2)
    assert pyr.names == ["s0", "s1", "s2"]  # inserted in order of voxel size, not of creation
    assert (s1.shape, s1.voxel_size, s1.translation) == (
        (8, 8, 8),
        (8.0, 8.0, 8.0),
        (4.0, 4.0, 4.0),
    )
    assert (s2.shape, s2.voxel_size, s2.translation) == (
        (4, 4, 4),
        (16.0, 16.0, 16.0),
        (8.0, 8.0, 8.0),
    )
    assert (
        s1.dtype == s0.dtype
        and s1.meta.chunk_shape == (4, 4, 4)
        and s1.meta.shard_shape == (8, 8, 8)
    )
    assert s1.meta.fill_value == 7 and s1.list_chunks() == []  # metadata plus an empty array
    codecs = wa.detect(s1.level_path).docs["zarr.json"]["codecs"]
    assert codecs[0]["name"] == "sharding_indexed"
    # an explicit placement, a level from another level, and replacing a same-named entry
    s3 = pyr.add_level(
        "s3", voxel_size=(32, 32, 32), translation=(16, 16, 16), shape=(2, 2, 2), from_level="s2"
    )
    assert s3.translation == (16.0, 16.0, 16.0)
    again = pyr.add_level("s3", factors=2, from_level="s2", exists="adopt")
    assert again.translation == (16.0, 16.0, 16.0) and pyr.names.count("s3") == 1
    with pytest.raises(ValueError, match="mode='r\\+'"):
        wa.open_pyramid(str(group)).add_level("s4", factors=2)


# --- Behaviours 15 and 16: chunk keys and write units -----------------------------------


@pytest.mark.parametrize(
    "fmt,kwargs,index,key",
    [
        ("zarr2", {}, (1, 2, 3), "1.2.3"),
        ("zarr2", {"separator": "/"}, (1, 2, 3), "1/2/3"),
        ("zarr3", {}, (1, 2, 3), "c/1/2/3"),
        ("zarr3", {"separator": "."}, (1, 2, 3), "c.1.2.3"),
        ("zarr3", {"encoding": "v2"}, (1, 2, 3), "1.2.3"),
        ("n5", {}, (1, 2, 3), "3/2/1"),
        (
            "precomputed",
            {"chunk_shape": (4, 8, 16), "shape": (10, 16, 32), "origin": (5, 0, 7)},
            (2, 1, 1),
            "23-39_8-16_13-15",
        ),
    ],
)
def test_chunk_key_round_trips(fmt, kwargs, index, key):
    assert wa.chunk_key(index, fmt, **kwargs) == key
    assert wa.parse_chunk_key(key, fmt, **kwargs) == index
    assert wa.parse_chunk_key("zarr.json", fmt, **kwargs) is None
    assert wa.parse_chunk_key(".zarray", fmt, **kwargs) is None


def test_array_chunk_keys_and_list_chunks_agree(tmp_path):
    v2 = wa.create(
        str(tmp_path / "v2.zarr/s0"),
        (8, 8, 8),
        "uint8",
        (1, 1, 1),
        (0.5,) * 3,
        chunks=(4, 4, 4),
        format="zarr2",
        separator="/",
    )
    v2.write_index((4, 0, 4), (8, 4, 8), np.ones((4, 4, 4), np.uint8))
    assert v2.chunk_key((1, 0, 1)) == "1/0/1" and v2.list_chunks() == [(1, 0, 1)]
    n5 = tmp_path / "vol.n5"
    write_n5(n5 / "s0", pattern((8, 8, 8)), attrs={"resolution": [1, 1, 1]}, chunks=(4, 4, 4))
    arr = wa.open(str(n5 / "s0"))
    assert arr.chunk_key((1, 0, 1)) == "1/0/1" and arr.parse_chunk_key("attributes.json") is None
    assert len(arr.list_chunks()) == 8
    pre = tmp_path / "pc"
    write_precomputed(
        pre, [{"data": pattern((8, 8, 8)), "resolution": [4, 4, 4], "voxel_offset": [5, 5, 5]}]
    )
    vol = wa.open(str(pre / "s0"))
    assert vol.chunk_key((0, 0, 0)) == "5-13_5-13_5-13" and vol.list_chunks() == [(0, 0, 0)]


def test_write_shape_is_the_shard_when_sharded(tmp_path):
    vol = wa.create(
        str(tmp_path / "s.zarr/s0"),
        (16, 16, 16),
        "uint8",
        (1, 1, 1),
        (0.5,) * 3,
        chunks=(4, 4, 4),
        shards=(8, 8, 8),
    )
    assert (
        vol.meta.chunk_shape == (4, 4, 4)
        and vol.meta.shard_shape == (8, 8, 8)
        and vol.write_shape == (8, 8, 8)
    )
    assert wa.metadata(vol.level_path, open=False).shard_shape == (8, 8, 8)
    assert list(vol.chunks_covering((0, 0, 0), (8, 8, 8), write=True)) == [(0, 0, 0)]
    assert len(list(vol.chunks_covering((0, 0, 0), (8, 8, 8)))) == 8
    with pytest.raises(ValueError, match="not whole write units of 8"):
        vol.write_index((0, 0, 0), (4, 4, 4), np.ones((4, 4, 4), np.uint8))
    vol.write_index((8, 8, 8), (16, 16, 16), np.ones((8, 8, 8), np.uint8))
    assert vol.chunk_key((1, 1, 1)) == "c/1/1/1" and vol.list_chunks() == [(1, 1, 1)]
    plain = wa.create(
        str(tmp_path / "p.zarr/s0"), (8, 8, 8), "uint8", (1, 1, 1), (0.5,) * 3, chunks=(4, 4, 4)
    )
    assert plain.meta.shard_shape is None and plain.write_shape == (4, 4, 4)
    assert ts.open(
        {"driver": "zarr3", "kvstore": {"driver": "file", "path": vol.level_path + "/"}}
    ).result().shape == (16, 16, 16)


# --- Behaviours 11 and 12 as the consumer sessions found them (2026-10-22 review) --------


def _group_doc(path, fmt):
    return json.loads((path / ("zarr.json" if fmt == "zarr3" else ".zgroup")).read_text())


def test_create_writes_missing_ancestor_groups_up_to_the_zarr_root(tmp_path):
    """cellmap-analyze's hierarchy: every group from the container down has a document."""
    root = tmp_path / "a.zarr"
    wa.create(str(root / "sub/mito/s0"), (8, 8, 8), "uint8", (8, 8, 8), (4, 4, 4), chunks=(8, 8, 8))
    for group in (root, root / "sub", root / "sub/mito"):
        doc = _group_doc(group, "zarr3")
        assert doc["zarr_format"] == 3 and doc["node_type"] == "group"
        assert wa.detect(str(group)).kind == "group"
    assert "attributes" not in _group_doc(root, "zarr3")  # bare: nothing invented above the level
    assert "ome" in _group_doc(root / "sub/mito", "zarr3")["attributes"]
    assert not (tmp_path / "zarr.json").exists() and not (tmp_path / ".zgroup").exists()
    assert wa.open_pyramid(str(root / "sub/mito")).names == ["s0"]
    # zarr v2 asked for: .zgroup at every missing ancestor
    v2 = tmp_path / "b.zarr"
    wa.create(
        str(v2 / "mito/s0"),
        (8, 8, 8),
        "uint8",
        (8, 8, 8),
        (4, 4, 4),
        chunks=(8, 8, 8),
        format="zarr2",
    )
    assert _group_doc(v2, "zarr2") == {"zarr_format": 2} and _group_doc(v2 / "mito", "zarr2") == {
        "zarr_format": 2
    }
    assert "multiscales" in json.loads((v2 / "mito/.zattrs").read_text())
    assert not (v2 / ".zattrs").exists()


def test_create_follows_the_nearest_existing_ancestors_format(tmp_path):
    """A v2 container gets v2 levels and OME 0.4 even when the level's own parent is absent."""
    root = tmp_path / "c.zarr"
    write_zarr2_group(root, {"owner": "cellmap-analyze"})
    vol = wa.create(
        str(root / "mito/s0"), (8, 8, 8), "uint8", (8, 8, 8), (4, 4, 4), chunks=(8, 8, 8)
    )
    assert vol.meta.format == "zarr2"
    assert (root / "mito/.zgroup").exists() and (root / "mito/s0/.zarray").exists()
    assert "multiscales" in json.loads((root / "mito/.zattrs").read_text())
    assert json.loads((root / ".zattrs").read_text()) == {"owner": "cellmap-analyze"}  # untouched
    with pytest.raises(ValueError, match="mix zarr versions"):
        wa.create(
            str(root / "er/s0"),
            (8, 8, 8),
            "uint8",
            (8, 8, 8),
            (4, 4, 4),
            chunks=(8, 8, 8),
            format="zarr3",
        )
    assert not (root / "er").exists()
    # an array in the way raises, and no document is written above a container with no .zarr name
    with pytest.raises(ValueError, match="is an array"):
        wa.create(
            str(root / "mito/s0/deeper/s0"),
            (8, 8, 8),
            "uint8",
            (8, 8, 8),
            (4, 4, 4),
            chunks=(8, 8, 8),
        )
    wa.create(
        str(tmp_path / "out/sub/s0"), (8, 8, 8), "uint8", (8, 8, 8), (4, 4, 4), chunks=(8, 8, 8)
    )
    assert (tmp_path / "out/sub/zarr.json").exists() and not (tmp_path / "out/zarr.json").exists()


def test_adopting_an_identical_array_writes_no_group_document(tmp_path, monkeypatch):
    """registration-station's 20 block jobs each call create on one output: only the first
    creation writes; the adopts read, compare and leave the document alone."""
    path = str(tmp_path / "shared.zarr/0")
    wa.create(
        path,
        (8, 8, 8),
        "float32",
        (0.5, 0.5, 0.5),
        (0.25,) * 3,
        chunks=(4, 4, 4),
        group_attrs={"fs": 2},
    )
    real, writes = wa.write.write_group_attrs, []

    def counting(loc, fmt, attrs, raw):
        writes.append(attrs)
        return real(loc, fmt, attrs, raw)

    monkeypatch.setattr(wa.write, "write_group_attrs", counting)
    for _ in range(3):
        vol = wa.create(
            path,
            (8, 8, 8),
            "float32",
            (0.5, 0.5, 0.5),
            (0.25 + 1e-9,) * 3,
            chunks=(4, 4, 4),
            exists="adopt",
            group_attrs={"fs": 2},
        )
        assert vol.writable
    assert writes == []
    # a different translation for the same level is a replacement, written once
    wa.ensure_group_entry(
        str(tmp_path / "shared.zarr"), "0", (0.5, 0.5, 0.5), (1.25,) * 3, ("z", "y", "x")
    )
    assert len(writes) == 1 and wa.open(path).translation == (1.25, 1.25, 1.25)


def test_a_group_entry_lost_to_a_concurrent_writer_is_written_again(tmp_path, monkeypatch):
    group = tmp_path / "parts.zarr"
    wa.create(str(group / "c000"), (4, 4, 4), "float32", (1, 1, 1), (0.5,) * 3, chunks=(4, 4, 4))
    before = json.loads((group / "zarr.json").read_text())
    real, calls = wa.write.write_group_attrs, {"n": 0}

    def racing(loc, fmt, attrs, raw):
        calls["n"] += 1
        real(loc, fmt, attrs, raw)
        if calls["n"] == 1:  # another process's older document lands after ours
            (group / "zarr.json").write_text(json.dumps(before))

    monkeypatch.setattr(wa.write, "write_group_attrs", racing)
    wa.ensure_group_entry(str(group), "c001", (1, 1, 1), (4.5, 0.5, 0.5), ("z", "y", "x"))
    assert calls["n"] == 2 and wa.open_pyramid(str(group)).names == ["c000", "c001"]

    def always_lost(loc, fmt, attrs, raw):
        real(loc, fmt, attrs, raw)
        (group / "zarr.json").write_text(json.dumps(before))

    monkeypatch.setattr(wa.write, "write_group_attrs", always_lost)
    monkeypatch.setattr(wa.write, "RACE_ATTEMPTS", 2)
    with pytest.raises(RuntimeError, match="lost 2 times"):
        wa.ensure_group_entry(str(group), "c002", (1, 1, 1), (8.5, 0.5, 0.5), ("z", "y", "x"))


def test_ensure_group_entry_lists_an_array_the_package_did_not_create(tmp_path, monkeypatch):
    """mesh-n-bone's symlinked s0, cellmap-analyze's level listed after it is filled."""
    group = tmp_path / "pyr.zarr"
    write_zarr3(group / "s0", pattern((8, 8, 8), np.uint8), chunks=(4, 4, 4))
    assert wa.detect(str(group), missing_ok=True) is not None  # a group by its s0, with no document
    wa.ensure_group_entry(str(group), "s0", (8, 8, 8), (4, 4, 4), ("z", "y", "x"))
    assert wa.open_pyramid(str(group)).names == ["s0"]
    assert wa.open(str(group / "s0")).translation == (4.0, 4.0, 4.0)
    assert wa.open(str(group / "s0")).meta.units == ("nanometer",) * 3
    writes = []
    monkeypatch.setattr(wa.write, "write_group_attrs", lambda *a: writes.append(a))
    wa.ensure_group_entry(str(group), "s0", (8, 8, 8), (4, 4, 4), ("z", "y", "x"))
    assert writes == []
    with pytest.raises(ValueError, match="mix zarr versions"):
        wa.ensure_group_entry(
            str(group), "s1", (16, 16, 16), (8, 8, 8), ("z", "y", "x"), format="zarr2"
        )


def test_a_pickled_relative_path_reopens_after_a_chdir(tmp_path, monkeypatch):
    """cellmap-analyze's dask workers change directory before they unpickle."""
    monkeypatch.chdir(tmp_path)
    wa.create("b.zarr/seg/s0", (8, 8, 8), "uint8", (4, 4, 4), (2, 2, 2), chunks=(4, 4, 4))
    vol = wa.open("b.zarr/seg/s0", mode="r+")
    vol.write_index((0, 0, 0), (4, 4, 4), np.full((4, 4, 4), 9, np.uint8))
    blob = pickle.dumps(vol)
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    again = pickle.loads(blob)
    assert again.path == str(tmp_path / "b.zarr/seg/s0")
    assert int(again.read_index((0, 0, 0), (1, 1, 1))[0, 0, 0]) == 9
    assert vol.reopen().level_path == again.level_path


def test_a_scalar_fills_the_box(tmp_path):
    vol = wa.create(
        str(tmp_path / "vol.zarr/s0"), (8, 8, 8), "uint8", (1, 1, 1), (0.5,) * 3, chunks=(4, 4, 4)
    )
    vol.write_index((0, 0, 0), (4, 8, 8), 7)
    vol.write(((4, 0, 0), (8, 4, 4)), np.uint8(3))
    vol.write_chunk((1, 1, 1), 5)
    got = vol.read_index((0, 0, 0), (8, 8, 8))
    assert np.all(got[:4] == 7) and np.all(got[4:, :4, :4] == 3) and np.all(got[4:, 4:, 4:] == 5)
    for bad in (300, -1, 0.5, float("nan"), True + 255):
        with pytest.raises(ValueError, match="does not fit uint8"):
            vol.write_index((0, 0, 0), (4, 4, 4), bad)
    vol.write_index((0, 0, 0), (4, 4, 4), 0)  # the fill over a whole unit leaves it absent
    assert (0, 0, 0) not in vol.list_chunks() and (0, 1, 0) in vol.list_chunks()
    f = wa.create(
        str(tmp_path / "f.zarr/s0"), (4,), "float32", (1,), (0.5,), axes=("x",), chunks=(4,)
    )
    f.write_index((0,), (4,), 0.1)  # rounded to the array's precision
    assert np.all(f.read_index((0,), (4,)) == np.float32(0.1))
    with pytest.raises(ValueError, match="does not fit float32"):
        f.write_index((0,), (4,), 1e300)


def test_a_pickled_handle_keeps_its_context(tmp_path):
    path = str(tmp_path / "vol.zarr/s0")
    wa.create(path, (8, 8, 8), "uint16", (4, 4, 4), (2, 2, 2), chunks=(4, 4, 4))
    capped = wa.open(path, mode="r+", context=wa.capped_context(concurrency=2))
    again = pickle.loads(pickle.dumps(capped))
    assert again.context.spec.to_json() == capped.context.spec.to_json()
    assert again.context is not wa.shared_context(0)
    assert (
        pickle.loads(pickle.dumps(capped)).context is again.context
    )  # one rebuilt context per spec
    shared = pickle.loads(pickle.dumps(wa.open(path)))
    assert shared.context is wa.shared_context(0)


def test_codecs_of_a_path_and_the_fill_write_that_removes_a_key(tmp_path):
    path = str(tmp_path / "vol.zarr/s0")
    vol = wa.create(
        path,
        (8, 8, 8),
        "uint8",
        (4, 4, 4),
        (2, 2, 2),
        chunks=(4, 4, 4),
        format="zarr2",
        codecs={"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1},
    )
    codecs, shards = wa.write.codecs_of(path)
    assert codecs["id"] == "blosc" and codecs["cname"] == "lz4" and shards is None
    vol.write_index((0, 0, 0), (4, 4, 4), np.ones((4, 4, 4), np.uint8))
    assert vol.list_chunks() == [(0, 0, 0)]
    vol.write_index((0, 0, 0), (4, 4, 4), np.zeros((4, 4, 4), np.uint8))
    assert vol.list_chunks() == []  # the one key a write may remove: its own, now all fill
    assert int(vol.read_index((0, 0, 0), (4, 4, 4)).sum()) == 0
