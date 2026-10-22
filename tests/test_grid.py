"""Grids and blocks: tiling, context, store-less grids, the factor schedule, the helpers
(spec, Behaviours 17 to 19)."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

import worldarray as wa
from conftest import ome_multiscales, pattern, write_zarr2, write_zarr2_group


def _cover(blocks, shape, origin=None) -> np.ndarray:
    """How many write boxes hold each voxel of ``shape`` voxels starting at ``origin``."""
    origin = origin or (0,) * len(shape)
    count = np.zeros(shape, dtype=int)
    for b in blocks:
        count[tuple(slice(s - o, e - o) for s, e, o in zip(b.begin, b.end, origin))] += 1
    return count


@pytest.fixture
def placed(tmp_path):
    """One OME zarr v2 level with voxels (3, 4, 5), translation (1.5, 10, -2.5), chunks (4, 5, 6)."""
    data = pattern((10, 12, 14))
    group = tmp_path / "placed.zarr"
    write_zarr2_group(group, ome_multiscales("zyx", [("s0", (3, 4, 5), (1.5, 10, -2.5))]))
    write_zarr2(group / "s0", data, chunks=(4, 5, 6))
    return str(group / "s0"), data


# --- blocks -------------------------------------------------------------------


def test_blocks_partition_the_array_in_c_order():
    bl = wa.blocks((10, 7, 5), (4, 3, 5))
    assert len(bl) == bl.count == 9
    assert np.all(_cover(bl, (10, 7, 5)) == 1)  # no overlap, no gap
    assert [b.index for b in bl] == list(np.ndindex(3, 3, 1))
    assert [b.id for b in bl] == list(range(9))
    assert (bl[8].begin, bl[8].end) == ((8, 6, 0), (10, 7, 5))  # the last block clipped
    assert all((b.read_begin, b.read_end) == (b.begin, b.end) for b in bl)


def test_blocks_partition_a_box_anchored_at_its_begin():
    bl = wa.blocks((10, 7, 5), (4, 3, 5), box=((2, 3, 1), (9, 7, 5)))
    expected = np.zeros((10, 7, 5), dtype=int)
    expected[2:9, 3:7, 1:5] = 1
    assert np.array_equal(_cover(bl, (10, 7, 5)), expected)
    assert [(b.begin, b.end) for b in bl] == [
        ((2, 3, 1), (6, 6, 5)),
        ((2, 6, 1), (6, 7, 5)),
        ((6, 3, 1), (9, 6, 5)),
        ((6, 6, 1), (9, 7, 5)),
    ]


def test_a_box_past_the_array_tiles_exactly_and_only_reads_are_clipped():
    box = ((0, 0, 0), (40, 40, 40))
    bl = wa.blocks((32, 32, 32), (16, 16, 16), context=2, box=box)
    assert len(bl) == 27
    assert np.all(
        _cover(bl, (40, 40, 40)) == 1
    )  # the write boxes are the box, outside the array too
    for b in bl:
        assert all(0 <= lo <= hi <= 32 for lo, hi in zip(b.read_begin, b.read_end))
    last = bl[26]
    assert (last.begin, last.end) == ((32, 32, 32), (40, 40, 40))
    assert (last.read_begin, last.read_end) == ((30, 30, 30), (32, 32, 32))
    unclipped = wa.blocks((32, 32, 32), (16, 16, 16), context=2, box=box, clip=False)[26]
    assert (unclipped.read_begin, unclipped.read_end) == ((30, 30, 30), (42, 42, 42))

    below = wa.blocks((32, 32, 32), (16, 16, 16), context=2, box=((-8, 0, 0), (24, 32, 32)))
    assert np.all(_cover(below, (32, 32, 32), origin=(-8, 0, 0)) == 1)
    assert (below[0].begin, below[0].end) == ((-8, 0, 0), (8, 16, 16))
    assert (below[0].read_begin, below[0].read_end) == ((0, 0, 0), (10, 18, 18))
    assert below.index_of((-8, 0, 0)) == (0, 0, 0) and below.index_of((8, 16, 16)) == (1, 1, 1)
    with pytest.raises(ValueError):
        below.index_of((0, 0, 0))  # not a start on the box-anchored grid

    far = wa.blocks((32,), (16,), context=2, box=((0,), (80,)))[4]  # wholly outside the array
    assert (far.begin, far.end, far.read_begin, far.read_end) == ((64,), (80,), (32,), (32,))


def test_context_is_an_int_one_per_axis_or_before_after_pairs():
    def grown(context):
        b = wa.blocks((12, 12, 12), (4, 4, 4), context=context, clip=False)[13]  # index (1, 1, 1)
        assert (b.begin, b.end) == ((4, 4, 4), (8, 8, 8))
        return b.read_begin, b.read_end

    assert grown(1) == ((3, 3, 3), (9, 9, 9))
    assert grown((0, 1, 2)) == ((4, 3, 2), (8, 9, 10))
    assert grown(((1, 0), (0, 2), (3, 3))) == ((3, 4, 1), (8, 10, 11))
    assert grown((1, (0, 2), 0)) == ((3, 4, 4), (9, 10, 8))
    for bad in (-1, (1, 2), ((1, 2, 3), 0, 0)):
        with pytest.raises(ValueError):
            wa.blocks((12, 12, 12), (4, 4, 4), context=bad)


def test_clip_clips_read_boxes_to_the_array_and_never_write_boxes():
    on = wa.blocks((10,), (4,), context=3)
    off = wa.blocks((10,), (4,), context=3, clip=False)
    assert [(b.read_begin, b.read_end) for b in on] == [((0,), (7,)), ((1,), (10,)), ((5,), (10,))]
    assert [(b.read_begin, b.read_end) for b in off] == [
        ((-3,), (7,)),
        ((1,), (11,)),
        ((5,), (13,)),
    ]
    assert [(b.begin, b.end) for b in on] == [(b.begin, b.end) for b in off]
    assert [(b.begin, b.end) for b in on] == [((0,), (4,)), ((4,), (8,)), ((8,), (10,))]


def test_edges_and_inner_on_a_middle_and_an_edge_block():
    clipped = wa.blocks((12, 12), (4, 4), context=2)
    padded = wa.blocks((12, 12), (4, 4), context=2, clip=False)
    middle = clipped[4]
    assert middle.index == (1, 1)
    assert middle.at_start == (False, False) and middle.at_end == (False, False)
    assert (middle.read_begin, middle.read_end) == ((2, 2), (10, 10))
    assert middle.inner == (slice(2, 6), slice(2, 6))
    assert padded[4] == middle

    edge = clipped[2]
    assert edge.index == (0, 2)  # first along the first axis, last along the second
    assert edge.at_start == (True, False) and edge.at_end == (False, True)
    assert (edge.read_begin, edge.read_end) == ((0, 6), (6, 12))
    assert edge.inner == (slice(0, 4), slice(2, 6))
    edge = padded[2]
    assert (edge.read_begin, edge.read_end) == ((-2, 6), (6, 14))
    assert edge.inner == (slice(2, 6), slice(2, 6))


def test_inner_selects_the_write_box_from_a_read_clipped_or_padded(cube):
    path, data = cube
    vol = wa.open(path)
    for clip in (True, False):
        for b in wa.blocks(vol.shape, (4, 4, 4), context=((1, 2), 2, (0, 3)), clip=clip):
            read = vol.read_index(b.read_begin, b.read_end, padding=0)
            own = data[tuple(slice(s, e) for s, e in zip(b.begin, b.end))]
            assert np.array_equal(read[b.inner], own)


def test_len_ids_and_index_of_agree():
    bl = wa.blocks((9, 10, 11), (2, 3, 4), context=1, box=((1, 0, 2), (9, 10, 11)))
    seen = list(bl)
    assert len(seen) == len(bl) == bl.count == 4 * 4 * 3
    for b in seen:
        assert bl[b.id] == b
        assert bl.index_of(b.begin) == b.index
    for bad in (len(bl), -1):
        with pytest.raises(IndexError):
            bl[bad]
    for begin in ((2, 0, 2), (1, 0, 14)):  # off the grid; past the last block
        with pytest.raises(ValueError):
            bl.index_of(begin)


def test_blocks_and_a_block_pickle_as_plain_records():
    bl = wa.blocks(
        (100, 100, 100),
        (16, 16, 16),
        context=((1, 2), 3, 0),
        box=((-4, 0, 0), (100, 100, 100)),
        clip=False,
    )
    shipped = pickle.loads(pickle.dumps(bl))
    assert shipped == bl and len(shipped) == len(bl) == 343
    assert shipped[200] == bl[200]
    assert pickle.loads(pickle.dumps(bl[200])) == bl[200]
    assert len(pickle.dumps(bl)) < 1000  # the arguments, not the blocks


# --- grids --------------------------------------------------------------------


def test_a_grid_answers_as_a_worldarray_of_the_same_facts(placed):
    path, _ = placed
    vol = wa.open(path)
    g = wa.Grid((10, 12, 14), (3, 4, 5), (1.5, 10, -2.5), chunks=(4, 5, 6))
    assert g.box == vol.box
    assert g.corner == vol.corner
    for box in (
        vol.box,
        ((0, 8, -5), (9, 30, 40)),
        ((1.5, 10, -2.5), (7.5, 18, 7.5)),  # centre-aligned
        ((-10, -10, -10), (100, 100, 100)),
        ((3, 12, 0), (3.1, 13, 1)),
    ):
        assert g.index_box(box) == vol.index_box(box)
    for begin, end in (
        ((0, 0, 0), (10, 12, 14)),
        ((3, 4, 5), (9, 11, 13)),
        ((-5, -5, -5), (2, 2, 2)),
        ((9, 11, 13), (40, 40, 40)),
        ((5, 5, 5), (5, 6, 7)),
    ):
        assert list(g.chunks_covering(begin, end)) == list(vol.chunks_covering(begin, end))
    meta = g.meta(vol.dtype)
    for name in ("shape", "dtype", "chunk_shape", "axes", "axes_named", "units", "voxel_size"):
        assert getattr(meta, name) == getattr(vol.grid, name)
    assert meta.translation == vol.grid.translation and meta.placed_by == "ome"
    assert wa.Grid.from_meta(vol.grid) == g


def test_a_grid_with_a_channel_axis(tmp_path):
    group = tmp_path / "czyx.zarr"
    write_zarr2_group(group, ome_multiscales("czyx", [("s0", (1, 4, 4, 4), (0, 2, 2, 2))]))
    write_zarr2(group / "s0", pattern((2, 6, 8, 10)), chunks=(1, 3, 4, 5))
    vol = wa.open(str(group / "s0"))
    g = wa.Grid((2, 6, 8, 10), (4, 4, 4), (2, 2, 2), axes="czyx", chunks=(1, 3, 4, 5))
    assert g.spatial == vol.spatial == (1, 2, 3)
    assert g.box == vol.box and g.corner == vol.corner
    for box in (((4, 4, 4), (20, 20, 20)), ((1, 4, 4, 4), (2, 20, 20, 20))):
        assert g.index_box(box) == vol.index_box(box)
    assert g.index_box(((4, 4, 4), (20, 20, 20)))[0][0] == 0  # the channel axis whole
    meta = g.meta(vol.dtype)
    for name in ("shape", "chunk_shape", "axes", "units", "voxel_size", "translation"):
        assert getattr(meta, name) == getattr(vol.grid, name)
    assert wa.Grid.from_meta(vol.grid) == g


def test_write_shape_and_default_blocks():
    plain = wa.Grid((64, 64, 64), (8, 8, 8), (4, 4, 4))
    assert plain.write_shape is None
    with pytest.raises(ValueError):
        plain.blocks()
    with pytest.raises(ValueError):
        plain.chunks_covering((0, 0, 0), (1, 1, 1))
    assert plain.blocks((32, 32, 32)) == wa.blocks((64, 64, 64), (32, 32, 32))
    chunked = wa.Grid((64, 64, 64), (8, 8, 8), (4, 4, 4), chunks=(16, 16, 16))
    assert chunked.write_shape == (16, 16, 16)
    assert chunked.blocks() == wa.blocks((64, 64, 64), (16, 16, 16))
    sharded = wa.Grid((64, 64, 64), (8, 8, 8), (4, 4, 4), chunks=(16, 16, 16), shards=(32, 32, 32))
    assert sharded.write_shape == (32, 32, 32)
    assert sharded.blocks(context=2) == wa.blocks((64, 64, 64), (32, 32, 32), context=2)
    assert list(sharded.chunks_covering((0, 0, 0), (40, 1, 1), write=True)) == [
        (0, 0, 0),
        (1, 0, 0),
    ]
    assert len(list(sharded.chunks_covering((0, 0, 0), (40, 1, 1)))) == 3
    with pytest.raises(ValueError):
        wa.Grid((64, 64, 64), (8, 8, 8), (4, 4, 4), chunks=(16, 16, 16), shards=(24, 24, 24))
    with pytest.raises(ValueError):
        wa.Grid((64, 64, 64), (8, 8, 8), (4, 4, 4), shards=(32, 32, 32))


@pytest.mark.parametrize("target", [(8, 8, 8), (6, 4, 2.5), (10.48, 7, 5)])
def test_from_meta_is_behaviour_4s_target_grid(placed, target):
    path, _ = placed
    vol = wa.open(path)
    resampled = wa.open(path, voxel_size=target, on_mismatch="resample")
    g = wa.Grid.from_meta(vol.grid, voxel_size=target)
    assert g.shape == resampled.shape
    assert g.voxel_size == tuple(float(t) for t in target)
    assert g.corner == pytest.approx(resampled.corner)
    assert [g.translation[i] for i in g.spatial] == pytest.approx(resampled.translation)
    assert g.chunks is None
    assert vol.read(g.box, voxel_size=target).shape == g.shape  # the read fills it exactly


# --- schedules and helpers ----------------------------------------------------


def test_level_factors_follow_mesh_n_bones_anisotropic_table():
    factors = wa.level_factors((30, 8, 8), ratio=1.5, max_levels=5)
    assert factors == [(1, 2, 2), (1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
    sizes, vs = [], np.array([30.0, 8.0, 8.0])
    for f in factors:
        vs = vs * f
        sizes.append(tuple(vs))
    assert sizes[:3] == [(30, 16, 16), (30, 32, 32), (60, 64, 64)]  # z waits for y and x at 32
    assert wa.level_factors((8, 8, 30), max_levels=3) == [(2, 2, 1), (2, 2, 1), (2, 2, 2)]
    assert wa.level_factors((8, 8, 20), max_levels=3) == [(2, 2, 1), (2, 2, 2), (2, 2, 2)]
    for iso in ((4, 4, 4), (10.48, 10.48, 10.48)):
        assert wa.level_factors(iso, max_levels=4) == [(2, 2, 2)] * 4


def test_level_factors_stop_when_a_level_fits():
    stop = (64, 64, 64)
    assert wa.level_factors((8, 8, 8), stop_shape=stop, shape=(100, 1000, 1000)) == [(2, 2, 2)] * 4
    assert wa.level_factors((30, 8, 8), stop_shape=stop, shape=(64, 256, 256)) == [(1, 2, 2)] * 2
    assert (
        wa.level_factors((8, 8, 8), max_levels=2, stop_shape=stop, shape=(100, 1000, 1000))
        == [(2, 2, 2)] * 2
    )
    assert wa.level_factors((8, 8, 8), stop_shape=stop, shape=(64, 64, 64)) == []
    for kwargs in ({}, {"stop_shape": stop}, {"max_levels": 2, "ratio": 0.5}):
        with pytest.raises(ValueError):
            wa.level_factors((8, 8, 8), **kwargs)


SHORT_FORMS = {
    "pm": ("picometer", "picometre", "pm"),
    "Å": ("angstrom", "ångström", "å", "Å"),
    "nm": ("nanometer", "nanometre", "nm"),
    "µm": ("micrometer", "micrometre", "micron", "um", "µm", "μm"),
    "mm": ("millimeter", "millimetre", "mm"),
    "cm": ("centimeter", "centimetre", "cm"),
    "m": ("meter", "metre", "m"),
    "ns": ("nanosecond", "ns"),
    "µs": ("microsecond", "us", "µs", "μs"),
    "ms": ("millisecond", "ms"),
    "s": ("second", "sec", "s"),
    "min": ("minute", "min"),
    "h": ("hour", "h"),
    "d": ("day", "d"),
}


def test_abbreviation_for_every_spelling_in_the_table():
    spellings = {s for group in SHORT_FORMS.values() for s in group}
    assert spellings == set(wa.units.TABLE["length"]) | set(wa.units.TABLE["time"])
    for short, group in SHORT_FORMS.items():
        for spelling in group:
            assert wa.units.abbreviation(spelling) == short
    assert wa.units.abbreviation("micrometer") == "µm"  # the micro sign, not Greek mu
    assert wa.units.abbreviation("Nanometers") == "nm"
    for unknown in ("", "pixel", "degrees_north"):
        with pytest.raises(ValueError):
            wa.units.abbreviation(unknown)


def test_path_helpers_are_url_aware(tmp_path):
    s = wa.store
    assert s.parent("s3://b/a/b") == s.parent("s3://b/a/b/") == "s3://b/a"
    assert s.basename("s3://b/a/b") == s.basename("s3://b/a/b/") == "b"
    assert s.join("s3://b/a", "b") == s.join("s3://b/a/", "/b/") == "s3://b/a/b"
    assert s.join("gs://bucket/vol.zarr", "seg", "s0") == "gs://bucket/vol.zarr/seg/s0"
    assert s.parent("https://host/data/vol.zarr/s0") == "https://host/data/vol.zarr"
    assert s.parent("precomputed://gs://b/vol/s0") == "gs://b/vol"
    assert s.parent("s3://b") is None  # a bucket root has no parent
    local = s.join(str(tmp_path), "vol.zarr")
    assert local == f"{tmp_path}/vol.zarr"
    assert s.parent(local) == str(tmp_path) and s.basename(local) == "vol.zarr"
