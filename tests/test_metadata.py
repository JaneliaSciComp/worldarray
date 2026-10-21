"""Conformance: the fixture cases, and the two consistency rules (spec, Conventions 1, 3, 4)."""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest

import worldarray as wa
from conftest import FIXTURES, build_case, ome_multiscales, pattern, write_zarr2, write_zarr2_group

CASES = json.loads((FIXTURES / "metadata.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_metadata_matches_fixture(tmp_path, case):
    path = build_case(tmp_path, case)
    vol = wa.open(path, units=None)
    meta, expect = vol.meta, case["expect"]
    assert meta.voxel_size == pytest.approx(expect["voxel_size"])
    assert meta.translation == pytest.approx(expect["translation"])
    assert list(meta.axes) == expect["axes"]
    assert list(meta.units) == expect["units"]
    assert meta.axes_named is expect["axes_named"]
    assert meta.shape == tuple(case["shape"])
    if "dtype" in expect:
        assert meta.dtype == np.dtype(expect["dtype"])
    for key in expect.get("attrs_include", []):
        assert key in meta.attrs
    if expect.get("reads_back"):
        data = pattern(case["shape"], case.get("dtype", "uint16"))
        np.testing.assert_array_equal(vol.read_index((0,) * data.ndim, data.shape), data)


def _ome_with_offset(tmp_path, offset):
    group = tmp_path / "o.zarr"
    write_zarr2_group(group, ome_multiscales("zyx", [("s0", (4, 4, 4), (10, 10, 10))]))
    attrs = {"voxel_size": [4, 4, 4], "offset": offset} if offset is not None else None
    write_zarr2(group / "s0", pattern((2, 2, 2)), attrs=attrs)
    return str(group / "s0")


def test_offset_equal_to_translation_or_half_a_voxel_below_is_silent(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="worldarray"):
        assert wa.open(_ome_with_offset(tmp_path, [10, 10, 10])).translation == (10.0, 10.0, 10.0)
        assert wa.open(_ome_with_offset(tmp_path, [8, 8, 8])).translation == (10.0, 10.0, 10.0)
    assert not caplog.records


def test_other_offset_warns_and_the_translation_wins(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="worldarray"):
        assert wa.open(_ome_with_offset(tmp_path, [3, 3, 3])).translation == (10.0, 10.0, 10.0)
    assert any("neither" in r.message for r in caplog.records)


def test_disagreeing_voxel_size_raises(tmp_path):
    group = tmp_path / "v.zarr"
    write_zarr2_group(group, ome_multiscales("zyx", [("s0", (4, 4, 4), None)]))
    write_zarr2(group / "s0", pattern((2, 2, 2)), attrs={"voxel_size": [8, 8, 8]})
    with pytest.raises(ValueError, match="disagrees"):
        wa.open(str(group / "s0"))


@pytest.mark.parametrize(
    "unit,scale,nm",
    [
        ("µm", 0.004, 4.0),
        ("μm", 0.004, 4.0),
        ("um", 0.004, 4.0),
        ("micrometre", 0.004, 4.0),
        ("nanometer", 4.0, 4.0),
        (None, 4.0, 4.0),
    ],
)
def test_units_convert_to_nm(tmp_path, unit, scale, nm):
    group = tmp_path / "u.zarr"
    units = {a: unit for a in "zyx"} if unit else {a: "" for a in "zyx"}
    write_zarr2_group(group, ome_multiscales("zyx", [("s0", (scale,) * 3, None)], units=units))
    write_zarr2(group / "s0", pattern((2, 2, 2)))
    vol = wa.open(str(group / "s0"))
    assert vol.voxel_size == pytest.approx((nm, nm, nm))
    assert wa.open(str(group / "s0"), units=None).voxel_size == pytest.approx((scale,) * 3)
    assert wa.open(str(group / "s0"), units="micrometer").voxel_size == pytest.approx(
        (nm / 1000,) * 3
    )


def test_unknown_unit_raises_only_when_converted(tmp_path):
    group = tmp_path / "w.zarr"
    write_zarr2_group(
        group,
        ome_multiscales("zyx", [("s0", (1, 1, 1), None)], units={a: "furlong" for a in "zyx"}),
    )
    write_zarr2(group / "s0", pattern((2, 2, 2)))
    assert wa.open(str(group / "s0"), units=None).meta.units == ("furlong",) * 3
    with pytest.raises(ValueError, match="furlong"):
        wa.open(str(group / "s0"))
