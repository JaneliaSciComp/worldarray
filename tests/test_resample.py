"""Resampling: the tie rule, chunk independence, no invented labels (spec, Behaviour 4, 5)."""

from __future__ import annotations

import numpy as np
import pytest

import worldarray as wa
from conftest import ome_multiscales, write_zarr2, write_zarr2_group


def _ramp(tmp_path, dtype="uint16"):
    """8 voxels of 4 nm along z (value = z index), corner at the origin."""
    data = np.broadcast_to(np.arange(8, dtype=dtype)[:, None, None], (8, 4, 4)).copy()
    group = tmp_path / "ramp.zarr"
    write_zarr2_group(group, ome_multiscales("zyx", [("s0", (4, 4, 4), (2, 2, 2))]))
    write_zarr2(group / "s0", data, chunks=(2, 4, 4))
    return str(group / "s0"), data


def test_nearest_takes_the_source_voxel_holding_the_target_centre(tmp_path):
    path, _ = _ramp(tmp_path)
    vol = wa.open(path)
    at8 = vol.read(((0, 0, 0), (32, 16, 16)), voxel_size=(8, 8, 8))
    assert list(at8[:, 0, 0]) == [
        1,
        3,
        5,
        7,
    ]  # factor 2: each centre is on a boundary; the upper voxel holds it
    at16 = vol.read(((0, 0, 0), (32, 16, 16)), voxel_size=(16, 16, 16))
    assert list(at16[:, 0, 0]) == [2, 6]
    at5 = vol.read(((0, 0, 0), (40, 16, 16)), voxel_size=(5, 5, 5))  # centres 2.5, 7.5, ... nm
    assert list(at5[:, 0, 0]) == [
        0,
        1,
        3,
        4,
        5,
        6,
        0,
        0,
    ]  # centres 32.5 and 37.5 nm are past the data: fill


def test_anchor_reads_onto_another_grid(tmp_path):
    """cellmap-analyze's case: A has 8 nm voxels with OME translation 0 (corner -4); B has
    4 nm voxels with corner -2. Read onto B's grid, each B centre takes the A voxel holding
    it, [0, 1, 1, 2, 2, 3, 3, 4]; anchored at A's own corner it is [0, 0, 1, 1, 2, 2, 3, 3]."""
    data = np.broadcast_to(np.arange(8, dtype="uint16")[:, None, None], (8, 4, 4)).copy()
    group = tmp_path / "a.zarr"
    write_zarr2_group(group, ome_multiscales("zyx", [("s0", (8, 8, 8), (0, 0, 0))]))
    write_zarr2(group / "s0", data, chunks=(2, 4, 4))
    a = wa.open(str(group / "s0"))
    box = ((-2, -4, -4), (30, 28, 28))
    onto_b = a.read(box, voxel_size=(4, 4, 4), anchor=(-2, -4, -4))
    assert list(onto_b[:, 0, 0]) == [0, 1, 1, 2, 2, 3, 3, 4]
    assert list(a.read(box, voxel_size=(4, 4, 4))[:, 0, 0]) == [0, 0, 1, 1, 2, 2, 3, 3]
    halves = np.concatenate(
        [
            a.read(((-2, -4, -4), (14, 28, 28)), voxel_size=(4, 4, 4), anchor=(-2, -4, -4)),
            a.read(((14, -4, -4), (30, 28, 28)), voxel_size=(4, 4, 4), anchor=(-2, -4, -4)),
        ]
    )
    np.testing.assert_array_equal(halves, onto_b)
    at_open = wa.open(
        str(group / "s0"), voxel_size=(4, 4, 4), on_mismatch="resample", anchor=(-2, -4, -4)
    )
    assert at_open.corner == (-2.0, -4.0, -4.0) and at_open.translation == (0.0, -2.0, -2.0)
    np.testing.assert_array_equal(at_open.read(box), onto_b)


def test_linear_is_the_block_mean_for_whole_factors(tmp_path):
    path, _ = _ramp(tmp_path, "float32")
    vol = wa.open(path, method="linear")
    out = vol.read(((0, 0, 0), (32, 16, 16)), voxel_size=(8, 8, 8))
    assert list(out[:, 0, 0]) == pytest.approx([0.5, 2.5, 4.5, 6.5])
    assert out.dtype == np.float32


def test_two_adjoining_reads_agree_with_one_read_across_both(tmp_path):
    path, _ = _ramp(tmp_path, "float32")
    vol = wa.open(path)
    for method, target in (
        ("nearest", (5, 5, 5)),
        ("linear", (5, 5, 5)),
        ("nearest", (8, 8, 8)),
        ("linear", (3, 3, 3)),
    ):
        whole = vol.read(((0, 0, 0), (30, 15, 15)), voxel_size=target, method=method)
        first = vol.read(((0, 0, 0), (15, 15, 15)), voxel_size=target, method=method)
        second = vol.read(((15, 0, 0), (30, 15, 15)), voxel_size=target, method=method)
        np.testing.assert_array_equal(
            np.concatenate([first, second]), whole, err_msg=f"{method} {target}"
        )


def test_nearest_invents_no_labels_and_pads_outside(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.choice(np.array([3, 17, 4000000000], dtype=np.uint32), size=(8, 8, 8))
    group = tmp_path / "labels.zarr"
    write_zarr2_group(group, ome_multiscales("zyx", [("s0", (4, 4, 4), (2, 2, 2))]))
    write_zarr2(group / "s0", data)
    vol = wa.open(str(group / "s0"))
    out, inside = vol.read(((0, 0, 0), (48, 48, 48)), voxel_size=(8, 8, 8), padding=0, mask=True)
    assert out.shape == (6, 6, 6) and inside.sum() == 4**3
    assert set(np.unique(out[inside])) <= set(np.unique(data))
    assert (out[~inside] == 0).all()
    edge = vol.read(((0, 0, 0), (48, 48, 48)), voxel_size=(8, 8, 8), padding="edge")
    np.testing.assert_array_equal(edge[4:, :4, :4], np.broadcast_to(edge[3, :4, :4], (2, 4, 4)))


def test_resample_target_set_at_open(tmp_path):
    path, data = _ramp(tmp_path)
    vol = wa.open(path, voxel_size=(8, 8, 8), on_mismatch="resample")
    assert (
        vol.voxel_size == (8.0, 8.0, 8.0)
        and vol.translation == (4.0, 4.0, 4.0)
        and vol.corner == (0.0, 0.0, 0.0)
    )
    np.testing.assert_array_equal(vol.read(((0, 0, 0), (32, 16, 16)))[:, 0, 0], [1, 3, 5, 7])
