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


def test_roi_like_objects_are_boxes(cube):
    path, data = cube

    class Roi:
        offset, shape = (4, 4, 4), (8, 8, 8)

    class Span:
        begin, end = (4, 4, 4), (12, 12, 12)

    vol = wa.open(path)
    np.testing.assert_array_equal(vol.read(Roi()), data[1:3, 1:3, 1:3])
    np.testing.assert_array_equal(vol.read(Span()), data[1:3, 1:3, 1:3])
    assert tuple(wa.open_tensorstore(path).shape) == data.shape


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
    kept = wa.open(group, voxel_size=(12, 12, 12), on_mismatch="keep")
    assert kept.level_name == "s1" and kept.voxel_size == (8.0, 8.0, 8.0)
    with pytest.raises(ValueError, match="coarser"):
        pyr.select((2, 2, 2), mode="finer")
    assert pyr.select((2, 2, 2), mode="floor").level_name == "s0"
    assert wa.open(group, level="first").level_name == "s0"
    assert wa.open(group + "/s1").index_box(((0, 0, 0), (16, 16, 16))) == ((0, 0, 0), (2, 2, 2))
    with pytest.warns(UserWarning, match="ignored"):
        wa.open(group + "/s0", level="s2")
    assert (
        wa.open(group + "/s0", level="first").level_name == "s0"
    )  # silent: an array is its own first level
    assert wa.open(group + "/s1").levels == [
        ("s0", (4.0, 4.0, 4.0)),
        ("s1", (8.0, 8.0, 8.0)),
        ("s2", (16.0, 16.0, 16.0)),
    ]


def test_relabel_reads_voxel_for_voxel_and_renames_the_grid(pyramid):
    group, levels = pyramid
    vol = wa.open(group, voxel_size=(12, 12, 12), on_mismatch="relabel")
    assert vol.voxel_size == (12.0, 12.0, 12.0) and vol.translation == (
        6.0,
        6.0,
        6.0,
    )  # the corner stays
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


def test_n5_downsampled_level_matches_ome_level(tmp_path):
    """An N5 pyramid with pixelResolution and downsamplingFactors and no offset sits where the
    equivalent OME pyramid with s0's corner at the origin puts its levels, as Neuroglancer
    draws both: s0 centred at half a voxel, s1 at a whole base voxel (f / 2 * base)."""
    base, s1 = pattern((8, 8, 8)), pattern((4, 4, 4))
    ome = tmp_path / "ome.zarr"
    write_zarr2_group(
        ome, ome_multiscales("zyx", [("s0", (4, 4, 4), (2, 2, 2)), ("s1", (8, 8, 8), (4, 4, 4))])
    )
    write_zarr2(ome / "s0", base)
    write_zarr2(ome / "s1", s1)
    n5 = tmp_path / "bdv.n5"
    write_n5_group(n5, {"pixelResolution": {"dimensions": [4, 4, 4], "unit": "nm"}})
    write_n5(n5 / "s0", base, attrs={"downsamplingFactors": [1, 1, 1]})
    write_n5(n5 / "s1", s1, attrs={"downsamplingFactors": [2, 2, 2]})
    for level in ("s0", "s1"):
        a, b = wa.open(str(ome / level)), wa.open(str(n5 / level))
        assert a.translation == pytest.approx(b.translation), level
        assert b.corner == pytest.approx((0.0, 0.0, 0.0)), level
        box = ((6.0, 6.0, 6.0), (22.0, 22.0, 22.0))
        np.testing.assert_array_equal(a.read(box), b.read(box))


def test_chunks_covering(cube):
    path, _ = cube  # chunks (3, 4, 5) over shape (6, 8, 10)
    vol = wa.open(path)
    assert set(vol.chunks_covering((1, 2, 3), (4, 6, 9))) == set(np.ndindex(2, 2, 2))
    assert list(vol.chunks_covering((3, 4, 5), (6, 8, 10))) == [(1, 1, 1)]
    assert list(vol.chunks_covering((20, 0, 0), (21, 1, 1))) == []


def test_document_less_group_is_probed_and_levels_take_integers(tmp_path):
    group = tmp_path / "bare"  # no .zgroup, no attributes.json: just s0 and s1
    for name, shape in (("s0", (4, 4, 4)), ("s1", (2, 2, 2))):
        write_zarr2(
            group / name,
            pattern(shape),
            attrs={"resolution": [4, 4, 4] if name == "s0" else [8, 8, 8]},
        )
    node = wa.detect(str(group))
    assert node.kind == "group" and node.format == "zarr2" and node.attrs == {}
    assert wa.detect(str(tmp_path / "nowhere"), missing_ok=True) is None
    assert wa.open_pyramid(str(group)).names == ["s0", "s1"]
    assert wa.open(str(group), level=1).level_name == "s1"
    assert wa.open_pyramid(str(group)).by_name(-1).voxel_size == (8.0, 8.0, 8.0)
    assert tuple(wa.open_tensorstore(str(group), level=0).shape) == (4, 4, 4)


def test_read_index_can_pin_axes_per_read(tmp_path):
    data = pattern((3, 2, 4, 5, 6))
    group = tmp_path / "tc5.zarr"
    write_zarr3_group(group)
    write_zarr3(group / "s0", data, dimension_names=list("tczyx"))
    vol = wa.open(str(group / "s0"), units=None)
    out = vol.read_index((0, 0, 0), (4, 5, 6), select={"t": 2, "c": 1})
    np.testing.assert_array_equal(out, data[2, 1])
    out, inside = vol.read_index((0, 0, 0), (4, 5, 8), select={"t": 0, "c": 0}, mask=True)
    assert (
        out.shape == inside.shape == (4, 5, 8)
        and inside[..., :6].all()
        and not inside[..., 6:].any()
    )


def test_base_voxel_size_rescales_every_level(tmp_path):
    """BigStitcher writes OME scale and translation in level-0 voxels but labels them
    micrometre; declaring level 0's true voxel size fixes every level."""
    group = tmp_path / "bs.zarr"
    datasets = [("s0", (1, 1, 1), (0, 0, 0)), ("s1", (2, 2, 2), (0.5, 0.5, 0.5))]
    write_zarr2_group(
        group, ome_multiscales("zyx", datasets, units={a: "micrometer" for a in "zyx"})
    )
    write_zarr2(group / "s0", pattern((8, 8, 8)))
    write_zarr2(group / "s1", pattern((4, 4, 4)))
    s1 = wa.open(
        str(group / "s1"),
        base_voxel_size=(1.0, 0.157, 0.157),
        base_units="micrometer",
        units="micrometer",
    )
    assert s1.voxel_size == pytest.approx((2.0, 0.314, 0.314))
    assert s1.translation == pytest.approx((0.5, 0.0785, 0.0785))
    assert s1.grid.units == ("micrometer",) * 3
    assert wa.open(
        str(group / "s0"), base_voxel_size=(1.0, 0.157, 0.157), units=None
    ).voxel_size == pytest.approx((1.0, 0.157, 0.157))


def test_nested_dataset_paths_find_the_multiscales_two_levels_up(tmp_path, caplog):
    """A fileglancer crop: the group's multiscales list "seg/s0", and the array sits two
    directories below the group (mesh-n-bone's case)."""
    import logging

    group = tmp_path / "crop.zarr"
    write_zarr2_group(
        group,
        ome_multiscales(
            "zyx", [("seg/s0", (4, 4, 4), (2, 2, 2)), ("seg/s1", (8, 8, 8), (4, 4, 4))]
        ),
    )
    write_zarr2_group(group / "seg")
    write_zarr2(group / "seg" / "s0", pattern((4, 4, 4)))
    write_zarr2(group / "seg" / "s1", pattern((2, 2, 2)))
    assert wa.open(str(group / "seg" / "s1")).voxel_size == (8.0, 8.0, 8.0)
    assert wa.open(str(group), level="seg/s1").translation == (4.0, 4.0, 4.0)
    assert wa.open_pyramid(str(group / "seg" / "s0")).names == ["seg/s0", "seg/s1"]
    write_zarr2(group / "seg" / "s9", pattern((2, 2, 2)))  # listed nowhere
    with caplog.at_level(logging.WARNING, logger="worldarray"):
        assert wa.open(str(group / "seg" / "s9"), units=None).meta.placed_by is None
    assert (
        not caplog.records
    )  # the immediate parent lists nothing, so there is nothing to warn about
    write_zarr2(group / "s7", pattern((2, 2, 2)))
    with caplog.at_level(logging.WARNING, logger="worldarray"):
        wa.open(str(group / "s7"), units=None)
    assert any("list no dataset" in r.message for r in caplog.records)


def test_sharded_zarr3_and_sharded_precomputed_read_like_any_other(tmp_path):
    import tensorstore as ts

    data = pattern((64, 64, 64), "uint8")
    z = tmp_path / "sharded.zarr"
    spec = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(z) + "/"},
        "metadata": {
            "shape": [64, 64, 64],
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [32, 32, 32]}},
            "data_type": "uint8",
            "codecs": [
                {
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": [16, 16, 16],
                        "codecs": [{"name": "bytes"}],
                        "index_codecs": [{"name": "bytes"}, {"name": "crc32c"}],
                    },
                }
            ],
        },
    }
    ts.open(spec, create=True, delete_existing=True).result().write(data).result()
    vol = wa.open(str(z), units=None)
    assert vol.meta.chunk_shape == (16, 16, 16)  # the inner read chunk, not the shard
    np.testing.assert_array_equal(vol.read_index((5, 6, 7), (40, 50, 60)), data[5:40, 6:50, 7:60])
    assert set(vol.chunks_covering((0, 0, 0), (17, 1, 1))) == {(0, 0, 0), (1, 0, 0)}

    pre = tmp_path / "sharded_pre"
    spec = {
        "driver": "neuroglancer_precomputed",
        "kvstore": {"driver": "file", "path": str(pre) + "/"},
        "multiscale_metadata": {"data_type": "uint8", "num_channels": 1, "type": "image"},
        "scale_metadata": {
            "size": [64, 64, 64],
            "resolution": [4, 4, 4],
            "voxel_offset": [0, 0, 0],
            "chunk_size": [16, 16, 16],
            "encoding": "raw",
            "key": "s0",
            "sharding": {
                "@type": "neuroglancer_uint64_sharded_v1",
                "preshift_bits": 0,
                "hash": "identity",
                "minishard_bits": 3,
                "shard_bits": 3,
                "minishard_index_encoding": "raw",
                "data_encoding": "raw",
            },
        },
    }
    ts.open(spec, create=True, delete_existing=True).result().write(
        np.ascontiguousarray(data.T)[..., None]
    ).result()
    vol = wa.open(str(pre / "s0"))
    assert vol.voxel_size == (4.0, 4.0, 4.0) and vol.translation == (2.0, 2.0, 2.0)
    np.testing.assert_array_equal(
        vol.read(((20.0, 24.0, 28.0), (160.0, 200.0, 240.0))), data[5:40, 6:50, 7:60]
    )


def test_base_voxel_size_on_a_unitless_n5_pyramid_warns_about_nothing(tmp_path, caplog):
    import logging

    group = tmp_path / "bs.n5"
    write_n5_group(group, {"resolution": [0.157, 0.157, 1.0]})
    write_n5(group / "s0", pattern((4, 8, 8)), attrs={"downsamplingFactors": [1, 1, 1]})
    write_n5(group / "s1", pattern((4, 4, 4)), attrs={"downsamplingFactors": [2, 2, 1]})
    with caplog.at_level(logging.WARNING, logger="worldarray"):
        s1 = wa.open(
            str(group),
            level=1,
            base_voxel_size=(1.0, 0.157, 0.157),
            base_units="micrometer",
            units="micrometer",
        )
    assert not [r for r in caplog.records if "no unit" in r.message]
    assert s1.voxel_size == pytest.approx((1.0, 0.314, 0.314)) and s1.translation == pytest.approx(
        (0.5, 0.157, 0.157)  # the corner at the origin at every level
    )


def test_level_path_is_the_opened_array(pyramid):
    group, _ = pyramid
    vol = wa.open(group, level="s1")
    assert vol.path == group and vol.level_path.endswith("/s1") and vol.level_path == vol.meta.path
    assert wa.open(vol.level_path).level_name == "s1"  # reopens exactly this level, by path alone


def test_level_path_reopens_a_precomputed_scale(tmp_path):
    data = pattern((4, 4, 4), "uint8")
    pre = tmp_path / "pre"
    write_precomputed(
        pre,
        [
            {"data": data, "resolution": [4, 4, 4]},
            {"data": data[::2, ::2, ::2], "resolution": [8, 8, 8]},
        ],
    )
    coarse = wa.open(str(pre), level=1)
    assert coarse.level_path == str(pre) + "/s1"
    again = wa.open(coarse.level_path)
    assert again.level_name == "s1" and again.voxel_size == (8.0, 8.0, 8.0)
    np.testing.assert_array_equal(again.read_index((0, 0, 0), (2, 2, 2)), data[::2, ::2, ::2])


def test_array_path_is_the_array_whatever_voxel_size_asks(pyramid):
    group, levels = pyramid
    vol = wa.open(group + "/s0", voxel_size=(8, 8, 8), on_mismatch="resample")
    assert vol.level_name == "s0" and vol.resampled and vol.source_voxel_size == (4.0, 4.0, 4.0)
    assert vol.shape == (8, 8, 8) and vol.voxel_size == (
        8.0,
        8.0,
        8.0,
    )  # the target grid covering the data
    assert wa.open(group, voxel_size=(8, 8, 8)).level_name == "s1"  # a group still picks a level


def test_relabel_keeps_the_corner(pyramid):
    group, _ = pyramid
    vol = wa.open(group + "/s1", voxel_size=(12, 12, 12), on_mismatch="relabel")
    assert (
        vol.corner == (0.0, 0.0, 0.0) and vol.translation == (6.0, 6.0, 6.0) and not vol.resampled
    )


def test_transform_is_applied_before_padding(cube):
    path, data = cube
    vol = wa.open(path, transform=lambda a: a.astype("float32") + 1000)
    out = vol.read_index((-1, 0, 0), (2, 2, 2), padding=0)
    assert out.dtype == np.float32 and (out[0] == 0).all()
    np.testing.assert_array_equal(out[1:], data[0:2, 0:2, 0:2].astype("float32") + 1000)
    edge = vol.read_index((-1, 0, 0), (2, 2, 2), padding="edge")
    np.testing.assert_array_equal(edge[0], edge[1])
    at8 = vol.read(((0, 0, 0), (16, 16, 16)), voxel_size=(8, 8, 8))
    assert at8.dtype == np.float32 and at8.min() >= 1000


def test_list_chunks(cube, tmp_path):
    path, data = cube  # chunks (3, 4, 5) over (6, 8, 10): every chunk written
    assert wa.open(path).list_chunks() == sorted(np.ndindex(2, 2, 2))
    n5 = tmp_path / "lc.n5"
    write_n5_group(n5)
    write_n5(n5 / "s0", pattern((4, 8, 8)), attrs={"resolution": [4, 4, 4]}, chunks=(2, 8, 8))
    assert wa.open(str(n5 / "s0")).list_chunks() == [(0, 0, 0), (1, 0, 0)]
    z3 = tmp_path / "lc3.zarr"
    write_zarr3_group(z3)
    write_zarr3(z3 / "s0", pattern((4, 4, 4)), chunks=(2, 4, 4))
    assert wa.open(str(z3 / "s0")).list_chunks() == [(0, 0, 0), (1, 0, 0)]


def test_pyramid_selection_reads_the_group_document_only(pyramid, monkeypatch):
    group, _ = pyramid
    pyr = wa.open_pyramid(group)
    assert pyr.voxel_sizes() == [
        ("s0", (4.0, 4.0, 4.0)),
        ("s1", (8.0, 8.0, 8.0)),
        ("s2", (16.0, 16.0, 16.0)),
    ]
    assert pyr._levels == {}  # nothing opened to answer that
    assert pyr.select((8, 8, 8)).level_name == "s1"
    assert set(pyr._levels) == {"s1"}


def test_precomputed_sN_path_means_scale_index(tmp_path):
    data = pattern((4, 4, 4), "uint8")
    pre = tmp_path / "keyed"
    write_precomputed(
        pre,
        [
            {"data": data, "resolution": [4, 4, 4], "key": "4_4_4"},
            {"data": data[::2, ::2, ::2], "resolution": [8, 8, 8], "key": "8_8_8"},
        ],
    )
    assert wa.open(str(pre / "s1")).voxel_size == (8.0, 8.0, 8.0)
    assert wa.open(str(pre / "8_8_8")).level_name == "8_8_8"
    assert tuple(wa.open_tensorstore(str(pre), level="s0").shape) == (4, 4, 4)


def test_an_unlisted_array_under_a_pyramid_does_not_walk_further(tmp_path, caplog):
    import logging

    outer = tmp_path / "outer.zarr"
    write_zarr2_group(outer, ome_multiscales("zyx", [("inner/s0", (4, 4, 4), (2, 2, 2))]))
    inner = outer / "inner"
    write_zarr2_group(
        inner, ome_multiscales("zyx", [("s0", (8, 8, 8), (4, 4, 4))])
    )  # its own pyramid
    write_zarr2(inner / "s0", pattern((2, 2, 2)))
    write_zarr2(inner / "s9", pattern((2, 2, 2)))  # listed by neither
    assert wa.open(str(inner / "s0")).voxel_size == (
        8.0,
        8.0,
        8.0,
    )  # the parent wins over the grandparent
    with caplog.at_level(logging.WARNING, logger="worldarray"):
        assert wa.open(str(inner / "s9"), units=None).meta.placed_by is None
    assert any("list no dataset" in r.message for r in caplog.records)
    node = wa.detect(str(outer))
    assert node.ome_attrs["multiscales"] == node.multiscales


def test_relabel_rescales_everything_by_the_ratio(tmp_path):
    group = tmp_path / "rl.zarr"
    write_zarr2_group(group, ome_multiscales("zyx", [("s0", (4, 4, 4), (10, 10, 10))]))  # corner 8
    write_zarr2(group / "s0", pattern((4, 4, 4)))
    vol = wa.open(str(group / "s0"), voxel_size=(8, 8, 8), on_mismatch="relabel")
    assert (
        vol.voxel_size == (8.0, 8.0, 8.0)
        and vol.translation == (20.0, 20.0, 20.0)
        and vol.corner == (16.0, 16.0, 16.0)
    )
    assert vol.source_voxel_size == (4.0, 4.0, 4.0)  # what the level is (cellmap-flow's report)
    resampled = wa.open(str(group / "s0"), voxel_size=(8, 8, 8), on_mismatch="resample")
    assert resampled.source_voxel_size == (4.0, 4.0, 4.0) and resampled.voxel_size == (
        8.0,
        8.0,
        8.0,
    )


def test_post_transform_runs_after_resampling_and_before_padding(tmp_path):
    data = np.broadcast_to(np.arange(8, dtype="uint8")[:, None, None], (8, 4, 4)).copy()
    group = tmp_path / "pt.zarr"
    write_zarr2_group(group, ome_multiscales("zyx", [("s0", (4, 4, 4), (2, 2, 2))]))
    write_zarr2(group / "s0", data)
    vol = wa.open(str(group / "s0"), method="linear")
    out = vol.read(
        ((0, 0, 0), (48, 16, 16)),
        voxel_size=(8, 8, 8),
        padding=0,
        post_transform=lambda a: a.astype("float32") / 255,
    )
    assert out.dtype == np.float32
    assert list(np.round(out[:4, 0, 0] * 255)) == [0, 2, 4, 6] or list(
        np.round(out[:4, 0, 0] * 255)
    ) == [1, 3, 5, 7]
    assert (out[4:] == 0).all()  # padding untouched by the transform
    plain = vol.read(((0, 0, 0), (16, 16, 16)), post_transform=lambda a: a + 100, padding=0)
    assert plain.min() >= 100


def test_pyramid_placements_come_from_the_document(pyramid, tmp_path):
    group, _ = pyramid
    pyr = wa.open_pyramid(group)
    assert pyr.translations() == [
        ("s0", (2.0, 2.0, 2.0)),
        ("s1", (4.0, 4.0, 4.0)),
        ("s2", (8.0, 8.0, 8.0)),
    ]
    assert pyr._levels == {}
    pre = tmp_path / "pre"
    write_precomputed(
        pre,
        [{"data": pattern((4, 4, 4), "uint8"), "resolution": [4, 4, 4], "voxel_offset": [1, 2, 3]}],
    )
    assert wa.open_pyramid(str(pre)).placements() == [("s0", (4.0, 4.0, 4.0), (14.0, 10.0, 6.0))]


def test_level_metas_from_documents_match_the_opened_ones(pyramid, tmp_path):
    group, _ = pyramid
    pyr = wa.open_pyramid(group)
    for light, opened in zip(pyr.metas(), pyr.levels):
        full = opened.meta
        assert (
            light.shape,
            light.dtype,
            light.chunk_shape,
            light.voxel_size,
            light.translation,
            light.placed_by,
        ) == (
            full.shape,
            full.dtype,
            full.chunk_shape,
            full.voxel_size,
            full.translation,
            full.placed_by,
        )
    assert wa.metadata(group, level="s1", open=False).voxel_size == (8.0, 8.0, 8.0)
    assert wa.metadata(
        group, level="s1", open=False, units="micrometer"
    ).voxel_size == pytest.approx((0.008,) * 3)
    import tensorstore as ts

    z = tmp_path / "sharded.zarr"
    spec = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(z) + "/"},
        "metadata": {
            "shape": [32, 32, 32],
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [16, 16, 16]}},
            "data_type": "uint8",
            "codecs": [
                {
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": [8, 8, 8],
                        "codecs": [{"name": "bytes"}],
                        "index_codecs": [{"name": "bytes"}, {"name": "crc32c"}],
                    },
                }
            ],
        },
    }
    ts.open(spec, create=True, delete_existing=True).result().write(
        pattern((32, 32, 32), "uint8")
    ).result()
    assert (
        wa.metadata(str(z), open=False).chunk_shape == (8, 8, 8) == wa.open(str(z)).meta.chunk_shape
    )
    n5 = tmp_path / "m.n5"
    write_n5_group(n5)
    write_n5(n5 / "s0", pattern((4, 6, 8)), attrs={"resolution": [8, 4, 2]}, chunks=(2, 3, 4))
    light, full = wa.metadata(str(n5 / "s0"), open=False), wa.metadata(str(n5 / "s0"))
    assert (light.shape, light.chunk_shape, light.dtype, light.translation) == (
        full.shape,
        full.chunk_shape,
        full.dtype,
        full.translation,
    )
    pre = tmp_path / "pre"
    write_precomputed(
        pre,
        [{"data": pattern((4, 6, 8), "uint8"), "resolution": [4, 4, 4], "voxel_offset": [1, 2, 3]}],
    )
    light, full = wa.open_pyramid(str(pre)).metas()[0], wa.open(str(pre / "s0")).meta
    assert (light.shape, light.chunk_shape, light.dtype, light.translation) == (
        full.shape,
        full.chunk_shape,
        full.dtype,
        full.translation,
    )


def test_read_can_pin_axes_per_read(tmp_path):
    data = pattern((3, 2, 4, 5, 6))
    group = tmp_path / "tc.zarr"
    write_zarr3_group(group)
    write_zarr3(group / "s0", data, dimension_names=list("tczyx"))
    vol = wa.open(str(group / "s0"), units=None)
    out = vol.read(((0, 0, 0), (4, 5, 6)), select={"t": 2, "c": 1})
    np.testing.assert_array_equal(out, data[2, 1])
    out, inside = vol.read(((0, 0, 0), (4, 5, 8)), select={"t": 0, "c": 0}, mask=True, padding=0)
    assert out.shape == inside.shape == (4, 5, 8) and not inside[..., 6:].any()


def test_callables_see_the_array_without_pinned_axes_and_empty_groups_say_so(tmp_path):
    data = pattern((2, 3, 4, 5))
    group = tmp_path / "c.zarr"
    write_zarr3_group(group)
    write_zarr3(group / "s0", data, dimension_names=list("czyx"))
    vol = wa.open(str(group / "s0"), units=None)
    seen = []
    out = vol.read(
        ((0, 0, 0), (3, 4, 5)), select={"c": 1}, transform=lambda a: seen.append(a.shape) or a + 1
    )
    assert seen == [(3, 4, 5)] and out.shape == (3, 4, 5)
    np.testing.assert_array_equal(out, data[1] + 1)
    out = vol.read(
        ((0, 0, 0), (3, 4, 10)),
        select={"c": 0},
        voxel_size=(2, 2, 2),
        padding=0,
        post_transform=lambda a: seen.append(a.shape) or a,
    )
    assert seen[-1] == out.shape and out.ndim == 3
    empty = tmp_path / "empty.zarr"
    write_zarr2_group(empty)
    with pytest.raises(ValueError, match="no levels found"):
        wa.metadata(str(empty), level="first", open=False)


def test_precomputed_scale_path_is_that_scale_under_level_first(tmp_path):
    """A path naming a scale is that scale, as an array path is that array; ``level=0`` and
    ``"first"`` are silently it and any other level warns (mesh-n-bone's review)."""
    from conftest import pattern

    pre = tmp_path / "pc"
    data = pattern((8, 8, 8))
    write_precomputed(
        pre,
        [
            {"data": data, "resolution": [4, 4, 4], "key": "4nm"},
            {"data": data[::2, ::2, ::2], "resolution": [8, 8, 8], "key": "8nm"},
        ],
    )
    for level in (None, 0, "first"):
        vol = wa.open(str(pre / "8nm"), level=level)
        assert (vol.level_name, vol.voxel_size) == ("8nm", (8.0, 8.0, 8.0)), level
    light = wa.metadata(str(pre / "8nm"), level="first", open=False)
    assert light.voxel_size == (8.0, 8.0, 8.0)
    with pytest.warns(UserWarning, match="names a precomputed scale"):
        assert wa.open(str(pre / "8nm"), level="4nm").level_name == "8nm"


def test_pyramid_opened_through_a_level_path_places_from_the_group_document(pyramid):
    """``placements`` reads the group document whichever path opened the pyramid, so a
    level whose array is gone is still placed and nothing is opened (mesh-n-bone's review)."""
    import shutil

    group, _ = pyramid
    expected = wa.open_pyramid(group).placements()
    shutil.rmtree(f"{group}/s2")
    pyr = wa.open_pyramid(f"{group}/s0")
    assert pyr.names == ["s0", "s1", "s2"]
    assert pyr.placements() == expected
    assert pyr.voxel_sizes()[-1] == ("s2", (16.0, 16.0, 16.0))
    assert pyr.translations()[-1] == ("s2", (8.0, 8.0, 8.0))
