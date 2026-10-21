"""Box algebra: funlib's Roi vocabulary on floats, half-open (spec, section 2)."""

from __future__ import annotations

import numpy as np

import worldarray as wa
from worldarray import Box


def test_box_algebra():
    a, b = Box((0, 0, 0), (10, 10, 10)), Box((5, -5, 2), (20, 5, 8))
    assert a.intersect(b) == Box((5, 0, 2), (10, 5, 8))
    assert a.union(b) == Box((0, -5, 0), (20, 10, 10))
    assert a.intersect(Box((20, 20, 20), (30, 30, 30))).empty
    assert (
        a.contains((0, 9.9, 5))
        and not a.contains((0, 10, 5))
        and a.contains(Box((1, 1, 1), (2, 2, 2)))
    )
    assert a.grow(2) == Box((-2, -2, -2), (12, 12, 12)) and a.grow(0, (1, 2, 3)).end == (
        11.0,
        12.0,
        13.0,
    )
    assert a.shift((1, 2, 3)).begin == (1.0, 2.0, 3.0) and a.centre == (5.0, 5.0, 5.0)

    class Roi:  # funlib-shaped
        offset, shape = (2, 2, 2), (4, 4, 4)

    assert a.intersect(Roi()) == Box((2, 2, 2), (6, 6, 6))


def test_box_snapping_and_index_round_trip():
    vs, tr = (4.0, 4.0, 4.0), (2.0, 2.0, 2.0)  # voxel corners at multiples of 4
    box = Box((1.3, 5.0, -3.0), (9.1, 8.0, 0.0))
    assert box.snap(vs, tr) == Box((0, 4, -4), (12, 8, 0))
    assert box.snap(vs, tr, "shrink") == Box((4, 8, 0), (8, 8, 0))
    assert box.snap(vs, tr, "closest") == Box((0, 4, -4), (8, 8, 0))
    assert box.snap(vs, tr).to_index(vs, tr) == ((0, 1, -1), (3, 2, 0))
    assert Box.from_index((0, 1, -1), (3, 2, 0), vs, tr) == box.snap(vs, tr)


def test_array_box_is_its_corner_based_extent(cube):
    path, data = cube  # 4 nm voxels, translation 2: corner 0
    vol = wa.open(path)
    assert vol.box == Box((0, 0, 0), (24, 32, 40))
    roi = Box((-8, 4, 36), (12, 12, 48))
    inside = vol.box.intersect(roi)
    assert inside == Box((0, 4, 36), (12, 12, 40))
    np.testing.assert_array_equal(vol.read(inside), data[0:3, 1:3, 9:10])
