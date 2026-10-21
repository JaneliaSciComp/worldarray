"""Reads: index boxes, world boxes, levels, selection, placement (spec, Behaviour 1 to 3, 9)."""

from __future__ import annotations

import numpy as np
import pytest

import worldarray as wa
from conftest import (
    ome_multiscales,
    pattern,
    write_n5,
    write_n5_group,
    write_precomputed,
    write_zarr2,
    write_zarr2_group,
    write_zarr3,
    write_zarr3_group,
)
from worldarray.geometry import index_range


def test_index_read_clips_reads_and_pads(cube):
    path, data = cube
    vol = wa.open(path)
    np.testing.assert_array_equal(vol.read_index((1, 2, 3), (4, 6, 9)), data[1:4, 2:6, 3:9])
    out, inside = vol.read_index((-2, 6, 8), (3, 10, 12), padding=7, mask=True)
    assert out.shape == (5, 4, 4) and inside.sum() == 3 * 2 * 2
    np.testing.assert_array_equal(out[2:, :2, :2], data[0:3, 6:8, 8:10])
    assert (out[~inside] == 7).all()
    edge = vol.read_index((-2, 6, 8), (3, 10, 12), padding="edge")
    np.testing.assert_array_equal(edge[0], edge[2])  # the first voxel repeated backwards
    np.testing.assert_array_equal(edge[:, 3, :], edge[:, 1, :])
    assert vol.read_index((0, 0, 0), (2, 2, 2)).flags.c_contiguous
    assert vol.read_index((0, 0, 0), (2, 2, 2)).dtype.isnative


def test_box_entirely_outside_returns_fill_or_edge(cube):
    path, data = cube
    vol = wa.open(path)
    out = vol.read_index((20, 0, 0), (23, 2, 2), padding=5)
    assert out.shape == (3, 2, 2) and (out == 5).all()
    edge = vol.read_index((-4, 0, 0), (-1, 2, 2), padding="edge")
    np.testing.assert_array_equal(edge, np.broadcast_to(data[0, :2, :2], (3, 2, 2)))


def test_world_box_selects_voxels_by_their_centres(cube):
    path, data = cube
    vol = wa.open(path)  # 4 nm, translation 2: voxel i covers [4i, 4i+4)
    np.testing.assert_array_equal(
        vol.read(((0, 0, 0), (16, 16, 16))), data[:4, :4, :4]
    )  # corner-aligned
    np.testing.assert_array_equal(
        vol.read(((2, 2, 2), (18, 18, 18))), data[:4, :4, :4]
    )  # centre-aligned
    off = vol.read(((1.3, 1.3, 1.3), (17.3, 17.3, 17.3)))  # neither: still exactly m voxels
    assert off.shape == (4, 4, 4)
    whole = vol.read(((0, 0, 0), (24, 32, 40)))  # the array's own extent: no padding
    np.testing.assert_array_equal(whole, data)
    with pytest.raises(ValueError, match="aligned"):
        vol.read(((1.3, 1.3, 1.3), (17.3, 17.3, 17.3)), strict=True)
    vol.read(((2, 2, 2), (18, 18, 18)), strict=True)


def test_roi_like_objects_and_funlib_alias(cube):
    path, data = cube

    class Roi:
        offset, shape = (4, 4, 4), (8, 8, 8)

    vol = wa.open(path)
    np.testing.assert_array_equal(vol.read(Roi()), data[1:3, 1:3, 1:3])
    np.testing.assert_array_equal(vol.to_ndarray_ts(Roi()), data[1:3, 1:3, 1:3])
    with pytest.warns(DeprecationWarning):
        legacy = wa.ImageDataInterface(path)
    assert legacy.voxel_size == (4.0, 4.0, 4.0)
    with pytest.warns(DeprecationWarning):
        raw = wa.open_ds_tensorstore(path)
    assert tuple(raw.shape) == data.shape


def test_centre_rule_is_exact_far_from_the_origin():
    s, t = (10.48,), (3.7,)
    for k in (0, 7, 100_000, 123_457):
        centre = (t[0] + k * s[0],)
        assert index_range(centre, (centre[0] + 5 * s[0],), s, t) == ((k,), (k + 5,))
        corner = (t[0] - s[0] / 2 + k * s[0],)
        assert index_range(corner, (corner[0] + 5 * s[0],), s, t) == ((k,), (k + 5,))


def test_time_and_channel_axes(tmp_path):
    data = pattern((4, 2, 3, 4, 5))
    group = tmp_path / "tc.zarr"
    axes = [{"name": "t", "type": "time", "unit": "second"}, {"name": "c", "type": "channel"}] + [
        {"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"
    ]
    ms = {
        "version": "0.5",
        "axes": axes,
        "datasets": [
            {
                "path": "s0",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [0.5, 1, 4, 4, 4]},
                    {"type": "translation", "translation": [0, 0, 2, 2, 2]},
                ],
            }
        ],
    }
    write_zarr3_group(group, {"ome": {"version": "0.5", "multiscales": [ms]}})
    write_zarr3(group / "s0", data, dimension_names=list("tczyx"))
    vol = wa.open(str(group / "s0"), units={"space": "nanometer", "time": "millisecond"})
    assert vol.axes == ("t", "c", "z", "y", "x") and vol.grid.units[0] == "millisecond"
    assert vol.grid.voxel_size[0] == 500.0
    out = vol.read(((500, 0, 0, 0, 0), (1500, 2, 12, 16, 20)))  # a time window and a spatial box
    np.testing.assert_array_equal(out, data[1:3, :, :3, :4, :5])
    spatial_only = vol.read(((0, 0, 0), (12, 16, 20)))  # other axes whole
    np.testing.assert_array_equal(spatial_only, data[:, :, :3, :4, :5])
    one = wa.open(str(group / "s0"), select={"t": 0, "c": 1})
    assert one.axes == ("z", "y", "x") and one.shape == (3, 4, 5)
    np.testing.assert_array_equal(one.read(((0, 0, 0), (12, 16, 20))), data[0, 1])


def test_levels_and_selection(pyramid):
    group, levels = pyramid
    with pytest.raises(ValueError, match="s0.*s1.*s2"):
        wa.open(group)
    assert wa.open(group, level="s1").level_name == "s1"
    assert wa.open(group, voxel_size=(8, 8, 8)).level_name == "s1"
    assert (
        wa.open(group, voxel_size=(12, 12, 12), on_mismatch="relabel").level_name == "s1"
    )  # floor: coarsest no coarser
    pyr = wa.open_pyramid(group)
    assert pyr.names == ["s0", "s1", "s2"]
    assert pyr.select((12, 12, 12), mode="nearest").level_name == "s2"
    with pytest.raises(ValueError, match="no level at"):
        pyr.select((12, 12, 12), mode="exact")
    with pytest.raises(ValueError, match="no level at"):
        wa.open(group, voxel_size=(12, 12, 12))
    with pytest.warns(UserWarning, match="ignored"):
        wa.open(group + "/s0", level="s2")
    assert wa.open(group + "/s1").levels == [
        ("s0", (4.0, 4.0, 4.0)),
        ("s1", (8.0, 8.0, 8.0)),
        ("s2", (16.0, 16.0, 16.0)),
    ]


def test_relabel_reads_voxel_for_voxel_and_renames_the_grid(pyramid):
    group, levels = pyramid
    vol = wa.open(group, voxel_size=(12, 12, 12), on_mismatch="relabel")
    assert vol.voxel_size == (12.0, 12.0, 12.0) and vol.translation == (4.0, 4.0, 4.0)
    out = vol.read(((-2, -2, -2), (22, 22, 22)))  # two relabelled voxels from the corner
    np.testing.assert_array_equal(out, levels["s1"][:2, :2, :2])


def test_group_with_probe_levels_and_pyramid_from_an_array(tmp_path):
    group = tmp_path / "plain.zarr"
    write_zarr2_group(group)
    for name, shape in (("s0", (4, 4, 4)), ("s1", (2, 2, 2))):
        write_zarr2(
            group / name,
            pattern(shape),
            attrs={"resolution": [4, 4, 4] if name == "s0" else [8, 8, 8]},
        )
    assert wa.open_pyramid(str(group)).names == ["s0", "s1"]
    assert wa.open(str(group), level="s1").voxel_size == (8.0, 8.0, 8.0)


def test_placement_agrees_across_formats(tmp_path):
    """One cube, voxels (2, 4, 8) nm in zyx, voxel 0 centred at (11, 22, 44): stored as OME,
    as funlib N5, as COSEM N5 and as precomputed, it reads back in the same place."""
    data = pattern((6, 6, 6), "uint8")
    centre, vs = (11.0, 22.0, 44.0), (2.0, 4.0, 8.0)
    ome = tmp_path / "ome.zarr"
    write_zarr2_group(ome, ome_multiscales("zyx", [("s0", vs, centre)]))
    write_zarr2(ome / "s0", data)
    funlib = tmp_path / "funlib.n5"
    write_n5_group(funlib)
    write_n5(
        funlib / "s0",
        data,
        attrs={"resolution": [8, 4, 2], "offset": [40, 20, 10], "units": ["nm", "nm", "nm"]},
    )
    cosem = tmp_path / "cosem.n5"
    write_n5_group(cosem)
    write_n5(
        cosem / "s0",
        data,
        attrs={
            "transform": {
                "axes": ["z", "y", "x"],
                "scale": [2, 4, 8],
                "translate": [11, 22, 44],
                "units": ["nm", "nm", "nm"],
            }
        },
    )
    pre = tmp_path / "pre"
    write_precomputed(pre, [{"data": data, "resolution": [8, 4, 2], "voxel_offset": [5, 5, 5]}])
    box = ((14.0, 28.0, 56.0), (20.0, 40.0, 80.0))  # corner-aligned: voxels 2..4 on each axis
    for path in (ome / "s0", funlib / "s0", cosem / "s0", pre / "s0"):
        vol = wa.open(str(path))
        assert vol.translation == pytest.approx(centre, abs=0.1 * min(vs)), path
        assert vol.voxel_size == pytest.approx(vs), path
        np.testing.assert_array_equal(vol.read(box), data[2:5, 2:5, 2:5], err_msg=str(path))


def test_bigdataviewer_level_matches_ome_level(tmp_path):
    """An N5 pyramid with pixelResolution and downsamplingFactors and no offset sits where the
    equivalent OME pyramid (voxel 0 of s0 at the origin) puts its levels."""
    base, s1 = pattern((8, 8, 8)), pattern((4, 4, 4))
    ome = tmp_path / "ome.zarr"
    write_zarr2_group(
        ome, ome_multiscales("zyx", [("s0", (4, 4, 4), (0, 0, 0)), ("s1", (8, 8, 8), (2, 2, 2))])
    )
    write_zarr2(ome / "s0", base)
    write_zarr2(ome / "s1", s1)
    n5 = tmp_path / "bdv.n5"
    write_n5_group(n5, {"pixelResolution": {"dimensions": [4, 4, 4], "unit": "nm"}})
    write_n5(n5 / "s0", base, attrs={"downsamplingFactors": [1, 1, 1]})
    write_n5(n5 / "s1", s1, attrs={"downsamplingFactors": [2, 2, 2]})
    a, b = wa.open(str(ome / "s1")), wa.open(str(n5 / "s1"))
    assert a.translation == pytest.approx(b.translation, abs=0.4)
    box = ((6.0, 6.0, 6.0), (22.0, 22.0, 22.0))
    np.testing.assert_array_equal(a.read(box), b.read(box))


def test_chunks_covering(cube):
    path, _ = cube  # chunks (3, 4, 5) over shape (6, 8, 10)
    vol = wa.open(path)
    assert set(vol.chunks_covering((1, 2, 3), (4, 6, 9))) == set(np.ndindex(2, 2, 2))
    assert list(vol.chunks_covering((3, 4, 5), (6, 8, 10))) == [(1, 1, 1)]
    assert list(vol.chunks_covering((20, 0, 0), (21, 1, 1))) == []
