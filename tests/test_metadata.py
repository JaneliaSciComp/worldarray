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
    assert meta.placed_by == expect["placed_by"]
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


def test_voxel_size_conflict_can_warn_instead(tmp_path, caplog):
    group = tmp_path / "v.zarr"
    write_zarr2_group(group, ome_multiscales("zyx", [("s0", (4, 4, 4), None)]))
    write_zarr2(group / "s0", pattern((2, 2, 2)), attrs={"voxel_size": [8, 8, 8]})
    with caplog.at_level(logging.WARNING, logger="worldarray"):
        vol = wa.open(str(group / "s0"), metadata_conflicts="warn")
    assert vol.voxel_size == (4.0, 4.0, 4.0) and any(
        "disagrees" in r.message for r in caplog.records
    )


def test_converting_a_unitless_axis_warns_once(tmp_path, caplog):
    group = tmp_path / "nounit.zarr"
    write_zarr2_group(group)
    write_zarr2(
        group / "s0", pattern((2, 2, 2)), attrs={"resolution": [4, 4, 4], "offset": [0, 0, 0]}
    )
    with caplog.at_level(logging.WARNING, logger="worldarray"):
        assert wa.open(str(group / "s0")).voxel_size == (4.0, 4.0, 4.0)
        wa.open(str(group / "s0"))
    assert sum("no unit stored" in r.message for r in caplog.records) == 1
    assert wa.open(str(group / "s0"), units=None).meta.units == ("", "", "")


def test_hybrid_layout_zarr3_array_in_a_zarr2_ome_group(tmp_path):
    from conftest import write_zarr3

    group = tmp_path / "hybrid.zarr"
    write_zarr2_group(group, ome_multiscales("zyx", [("s0", (4, 4, 4), (2, 2, 2))]))
    write_zarr3(group / "s0", pattern((2, 2, 2)))
    vol = wa.open(str(group / "s0"))
    assert vol.meta.format == "zarr3" and vol.meta.placed_by == "ome"
    assert vol.voxel_size == (4.0, 4.0, 4.0) and vol.translation == (2.0, 2.0, 2.0)


def test_lenient_units_take_unknown_units_as_the_target(tmp_path, caplog):
    group = tmp_path / "px.zarr"
    write_zarr2_group(
        group, ome_multiscales("zyx", [("s0", (4, 4, 4), None)], units={a: "pixel" for a in "zyx"})
    )
    write_zarr2(group / "s0", pattern((2, 2, 2)))
    with pytest.raises(ValueError):
        wa.open(str(group / "s0"))
    with caplog.at_level(logging.WARNING, logger="worldarray"):
        vol = wa.open(str(group / "s0"), strict_units=False)
    assert vol.voxel_size == (4.0, 4.0, 4.0) and any(
        "unknown unit" in r.message for r in caplog.records
    )
    assert "nanometer" in wa.units.TABLE["length"] and "second" in wa.units.TABLE["time"]
    assert wa.open(str(group / "s0"), units=None).meta.fill_value in (None, 0)


def test_parent_group_funlib_attrs_place_an_attribute_less_array(tmp_path):
    """funlib's keys on the parent group, the array carrying none, zarr and N5 alike
    (mesh-n-bone's inputs): read as on the array, offset a corner."""
    from conftest import write_n5, write_n5_group

    for fmt in ("zarr2", "n5"):
        root = tmp_path / f"vol.{fmt}"
        if fmt == "zarr2":
            write_zarr2_group(root, {"voxel_size": [1, 2, 3], "offset": [4, 4, 4]})
            write_zarr2(root / "s0", pattern((4, 4, 4)))
        else:
            write_n5_group(root, {"voxel_size": [1, 2, 3], "offset": [4, 4, 4]})
            write_n5(root / "s0", pattern((4, 4, 4)))
        meta = wa.metadata(str(root / "s0"))
        assert meta.voxel_size == (1.0, 2.0, 3.0), fmt
        assert meta.translation == pytest.approx((4.5, 5.0, 5.5)), fmt
        assert meta.placed_by == "funlib", fmt


def test_n5_array_funlib_keys_are_c_order(tmp_path):
    """``voxel_size``/``offset`` under funlib's names on an N5 array are not reversed, unlike
    ``resolution`` and ``pixelResolution``, because zarr-python's N5 store leaves user
    attributes alone."""
    from conftest import write_n5

    root = tmp_path / "vol.n5"
    write_n5(root / "s0", pattern((4, 4, 4)), attrs={"voxel_size": [1, 2, 3], "offset": [4, 4, 4]})
    meta = wa.metadata(str(root / "s0"))
    assert meta.voxel_size == (1.0, 2.0, 3.0)
    assert meta.translation == pytest.approx((4.5, 5.0, 5.5))
    assert meta.placed_by == "funlib"
