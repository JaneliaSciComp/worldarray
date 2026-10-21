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


def test_nearest_ties_go_to_the_lower_index(tmp_path):
    path, _ = _ramp(tmp_path)
    vol = wa.open(path)
    at8 = vol.read(((0, 0, 0), (32, 16, 16)), voxel_size=(8, 8, 8))
    assert list(at8[:, 0, 0]) == [0, 2, 4, 6]  # factor 2: the stride read, by the tie rule
    at16 = vol.read(((0, 0, 0), (32, 16, 16)), voxel_size=(16, 16, 16))
    assert list(at16[:, 0, 0]) == [1, 5]  # factor 4: nearest the block centre, not its first voxel


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
    np.testing.assert_array_equal(vol.read(((0, 0, 0), (32, 16, 16)))[:, 0, 0], [0, 2, 4, 6])
