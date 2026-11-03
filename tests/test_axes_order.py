"""Presenting the axes in another order: ``axes_order`` (spec, Conventions 2, Behaviours 14
and 15)."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

import worldarray as wa
from conftest import pattern, write_n5, write_zarr2

XYZ = ("x", "y", "z")


def _xyz(tmp_path, shape=(8, 6, 4), **kw) -> tuple[str, np.ndarray]:
    """An x, y, z array (TensorSwitch's order) at 16, 16, 45 nm, filled with a pattern."""
    path = str(tmp_path / "xyz.zarr/s0")
    vol = wa.create(path, shape, "uint16", (16, 16, 45), (4, 4, 0), axes=XYZ, **kw)
    data = pattern(shape)
    vol.write_index((0,) * len(shape), shape, data)
    return path, data


def test_zyx_presents_an_xyz_array_as_its_transpose(tmp_path):
    path, data = _xyz(tmp_path, chunks=(4, 3, 2))
    stored = wa.open(path)
    assert stored.axes == XYZ and stored.shape == (8, 6, 4)  # the default is the store's order
    vol = wa.open(path, axes_order="zyx")
    assert vol.axes == ("z", "y", "x") and vol.shape == (4, 6, 8) and vol.ts.shape == (4, 6, 8)
    assert vol.voxel_size == (45, 16, 16) and vol.translation == (0, 4, 4)
    assert vol.meta.chunk_shape == (2, 3, 4) and vol.write_shape == (2, 3, 4)
    assert vol.meta.units == ("nanometer",) * 3
    np.testing.assert_array_equal(vol.read_index((0, 0, 0), (4, 6, 8)), data.transpose(2, 1, 0))
    np.testing.assert_array_equal(vol.read(vol.box), data.transpose(2, 1, 0))
    part = vol.read_index((1, 2, 3), (3, 5, 7))
    np.testing.assert_array_equal(part, data[3:7, 2:5, 1:3].transpose(2, 1, 0))
    assert wa.Grid.from_meta(vol.meta).axes == ("z", "y", "x")


def test_a_channel_axis_comes_first_and_select_still_pins_it(tmp_path):
    path = str(tmp_path / "cxyz.zarr/s0")
    data = pattern((2, 8, 6, 4))
    wa.create(
        path, data.shape, "uint16", (16, 16, 45), (4, 4, 0), axes=("c", *XYZ), chunks=(1, 4, 3, 2)
    )
    wa.open(path, mode="r+").write_index((0,) * 4, data.shape, data)
    vol = wa.open(path, axes_order="zyx")
    assert vol.axes == ("c", "z", "y", "x") and vol.meta.chunk_shape == (1, 2, 3, 4)
    np.testing.assert_array_equal(vol.read_index((0,) * 4, vol.shape), data.transpose(0, 3, 2, 1))
    one = wa.open(path, axes_order="zyx", select={"c": 1})
    assert one.axes == ("z", "y", "x")
    np.testing.assert_array_equal(one.read_index((0,) * 3, one.shape), data[1].transpose(2, 1, 0))
    # a channel-last array comes out channel first
    last = str(tmp_path / "xyzc.zarr/s0")
    wa.create(
        last, (8, 6, 4, 2), "uint16", (16, 16, 45), (4, 4, 0), axes=(*XYZ, "c"), chunks=(4, 3, 2, 1)
    )
    assert wa.open(last, axes_order="zyx").axes == ("c", "z", "y", "x")


def test_a_write_through_the_reordered_view_lands_in_the_stores_order(tmp_path):
    path = str(tmp_path / "w.zarr/s0")
    wa.create(
        path,
        (8, 6, 4),
        "uint16",
        (16, 16, 45),
        (4, 4, 0),
        axes=XYZ,
        chunks=(2, 3, 2),
        shards=(4, 6, 2),
    )
    vol = wa.open(path, mode="r+", axes_order="zyx")
    assert vol.meta.shard_shape == (2, 6, 4) and vol.write_shape == (2, 6, 4)
    zyx = pattern((4, 6, 8))
    vol.write_index((0, 0, 0), (2, 6, 4), zyx[:2, :, :4])  # one shard: z 0-2, x 0-4
    vol.write_chunk((1, 0, 1), zyx[2:, :, 4:])
    stored = wa.open(path)
    np.testing.assert_array_equal(
        stored.read_index((0, 0, 0), (4, 6, 2)), zyx[:2, :, :4].transpose(2, 1, 0)
    )
    np.testing.assert_array_equal(
        stored.read_index((4, 0, 2), (8, 6, 4)), zyx[2:, :, 4:].transpose(2, 1, 0)
    )
    with pytest.raises(ValueError, match="whole write units"):
        vol.write_index((0, 0, 0), (1, 6, 4), zyx[:1, :, :4])
    # keys are the store's; the indices are in the presented order
    assert vol.chunk_key((1, 0, 1)) == stored.chunk_key((1, 0, 1)[::-1]) == "c/1/0/1"
    assert vol.chunk_key((0, 0, 1)) == "c/1/0/0"
    assert vol.parse_chunk_key("c/1/0/0") == (0, 0, 1)
    assert vol.list_chunks() == [(0, 0, 0), (1, 0, 1)]
    assert stored.list_chunks() == [(0, 0, 0), (1, 0, 1)]
    for b in vol.blocks():
        assert vol.parse_chunk_key(vol.chunk_key(b.index)) == b.index


def test_a_selected_axis_keeps_its_stored_place_in_a_key_index(tmp_path):
    path = str(tmp_path / "k.zarr/s0")
    wa.create(
        path, (8, 6, 4, 2), "uint16", (16, 16, 45), (4, 4, 0), axes=(*XYZ, "c"), chunks=(4, 3, 2, 1)
    )
    vol = wa.open(path, axes_order="zyx", select={"c": 1})
    assert vol.axes == ("z", "y", "x")
    # stored (x, y, z, c); the key index is (z, y, x, c): c stays last, where it is stored
    assert vol.chunk_key((1, 0, 1, 1)) == "c/1/0/1/1"
    assert vol.chunk_key((0, 1, 1, 0)) == "c/1/1/0/0"
    assert vol.parse_chunk_key("c/1/1/0/0") == (0, 1, 1, 0)


def test_a_sequence_names_the_order_and_a_bad_order_is_refused(tmp_path):
    path, data = _xyz(tmp_path, chunks=(4, 3, 2))
    vol = wa.open(path, axes_order=("y", "z", "x"))
    assert vol.axes == ("y", "z", "x") and vol.voxel_size == (16, 45, 16)
    np.testing.assert_array_equal(vol.read_index((0, 0, 0), vol.shape), data.transpose(1, 2, 0))
    with pytest.raises(ValueError, match="axes_order is"):
        wa.open(path, axes_order="xyz")
    with pytest.raises(ValueError, match="must name each of the axes"):
        wa.open(path, axes_order=("z", "y"))
    with pytest.raises(ValueError, match="names an axis twice"):
        wa.open(path, axes_order=("z", "z", "x"))


def test_n5_and_unnamed_arrays_are_unchanged_under_zyx(tmp_path):
    data = pattern((4, 6, 8))
    write_n5(tmp_path / "a.n5", data, attrs={"resolution": [16, 16, 45], "axes": ["x", "y", "z"]})
    n5 = wa.open(str(tmp_path / "a.n5"), axes_order="zyx", units=None)
    assert n5.axes == ("z", "y", "x") and n5.voxel_size == (45, 16, 16)
    np.testing.assert_array_equal(n5.read_index((0, 0, 0), n5.shape), data)
    write_zarr2(tmp_path / "u.zarr", data, chunks=(2, 3, 4))
    unnamed = wa.open(str(tmp_path / "u.zarr"), axes_order="zyx", units=None)
    assert unnamed.axes == ("z", "y", "x") and not unnamed.meta.axes_named
    np.testing.assert_array_equal(unnamed.read_index((0, 0, 0), unnamed.shape), data)


def test_metadata_with_and_without_opening(tmp_path):
    path, _ = _xyz(tmp_path, chunks=(4, 3, 2))
    for opened in (True, False):
        meta = wa.metadata(path, axes_order="zyx", open=opened)
        assert meta.axes == ("z", "y", "x") and meta.shape == (4, 6, 8)
        assert meta.chunk_shape == (2, 3, 4) and meta.voxel_size == (45, 16, 16)
        assert meta.translation == (0, 4, 4)
    assert wa.metadata(path, open=False).axes == XYZ


def _xyz_pyramid(tmp_path) -> str:
    """s0 at 16, 16, 45 nm and s1 at 32, 32, 45 (x and y halved), stored x, y, z."""
    _xyz(tmp_path, chunks=(4, 3, 2))
    group = str(tmp_path / "xyz.zarr")
    wa.open_pyramid(group, mode="r+").add_level("s1", factors=(2, 2, 1))
    return group


def test_a_pyramid_presents_its_placements_and_level_meta_in_the_order(tmp_path):
    group = _xyz_pyramid(tmp_path)
    pyr = wa.open_pyramid(group, axes_order="zyx")
    assert pyr.placements() == [
        ("s0", (45.0, 16.0, 16.0), (0.0, 4.0, 4.0)),
        ("s1", (45.0, 32.0, 32.0), (0.0, 12.0, 12.0)),
    ]
    assert pyr.level_meta("s1").shape == (4, 3, 4) and pyr.level_meta("s1").axes == ("z", "y", "x")
    assert pyr.by_name("s1").shape == (4, 3, 4) and pyr.by_name("s1").axes == ("z", "y", "x")
    # a level chosen by an anisotropic voxel size, given in the presented order
    vol = wa.open(group, voxel_size=(45, 32, 32), axes_order="zyx")
    assert vol.level_name == "s1" and vol.voxel_size == (45, 32, 32)
    with pytest.raises(ValueError, match="no level at voxel size"):
        wa.open(group, voxel_size=(45, 32, 32))  # read as x, y, z: no such level


def test_base_voxel_size_lands_on_the_presented_axes(tmp_path):
    group = _xyz_pyramid(tmp_path)
    vol = wa.open(group, level="s1", axes_order="zyx", base_voxel_size=(450, 160, 160))
    assert vol.voxel_size == (450, 320, 320)


def test_add_level_takes_its_arguments_in_the_presented_order(tmp_path):
    group = _xyz_pyramid(tmp_path)
    pyr = wa.open_pyramid(group, mode="r+", axes_order="zyx")
    s2 = pyr.add_level("s2", factors=(2, 2, 2), from_level="s1")
    s3 = pyr.add_level("s3", factors=(1, 2, 2), from_level="s1", chunks=(2, 3, 4))
    assert s2.axes == s3.axes == ("z", "y", "x")
    assert s2.voxel_size == (90, 64, 64) and s3.voxel_size == (45, 64, 64)
    stored = wa.open(f"{group}/s3")
    assert stored.axes == XYZ and stored.voxel_size == (64, 64, 45)
    assert stored.shape == (2, 2, 4) and stored.meta.chunk_shape == (4, 3, 2)


def test_the_order_survives_reopen_and_a_pickle(tmp_path):
    path, data = _xyz(tmp_path, chunks=(4, 3, 2))
    vol = wa.open(path, axes_order="zyx")
    for again in (vol.reopen(), pickle.loads(pickle.dumps(vol))):
        assert again.axes == ("z", "y", "x")
        np.testing.assert_array_equal(
            again.read_index((0, 0, 0), again.shape), data.transpose(2, 1, 0)
        )
    assert vol.reopen(axes_order="stored").axes == XYZ
