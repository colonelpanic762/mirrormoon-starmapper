"""Tests for the need-to-find search helpers."""

import numpy as np
import pytest

from starmapper.finder import (
    Sighting, distance_range, stars_in_circle, estimate_position,
    format_clock, map_offset, map_line,
)


def unit(v):
    v = np.array(v, dtype=float)
    return v / np.linalg.norm(v)


ORIGIN = np.array([500.0, 500.0, 500.0])


def sighting(direction, half_angle_deg=2.0, origin=ORIGIN, observer="OBS"):
    return Sighting(observer, np.array(origin, dtype=float), unit(direction),
                    np.radians(half_angle_deg))


def test_distance_range_runs_to_the_cluster_edge():
    low, high = distance_range(sighting([1, 0, 0], half_angle_deg=0.01))
    assert low == pytest.approx(0, abs=2)
    assert high == pytest.approx(499, abs=3)


def test_distance_range_is_narrowed_by_name_cube():
    # x in 800-899, on the line of sight along +x from x=500
    bounds = ((800, 899), (500, 599), (500, 599))
    low, high = distance_range(sighting([1, 0, 0], half_angle_deg=0.01), bounds)
    assert low == pytest.approx(300, abs=3)
    assert high == pytest.approx(399, abs=3)


def test_wider_circle_widens_the_distance_range():
    # Line of sight passes 20 PC outside the cube; only a wide circle reaches it
    bounds = ((800, 899), (520, 599), (500, 599))
    assert distance_range(sighting([1, 0, 0], half_angle_deg=0.5), bounds) is None
    low, high = distance_range(sighting([1, 0, 0], half_angle_deg=5.0), bounds)
    assert 280 < low < 320 and 380 < high < 440


def test_stars_in_circle_lists_only_those_inside():
    known = {
        "INSIDE": ORIGIN + 200 * unit([1, 0.02, 0]),
        "OUTSIDE": ORIGIN + 200 * unit([1, 0.2, 0]),
    }
    result = stars_in_circle(sighting([1, 0, 0], half_angle_deg=2.0), known)
    assert [r.name for r in result] == ["INSIDE"]


def test_estimate_position_needs_two_observatories():
    assert estimate_position([sighting([1, 0, 0])]) is None
    same_place = [sighting([1, 0, 0]), sighting([1, 0.01, 0])]
    assert estimate_position(same_place) is None


def test_estimate_position_from_two_observatories():
    target = np.array([300.0, 650.0, 420.0])
    a = np.array([100.0, 100.0, 100.0])
    b = np.array([900.0, 200.0, 700.0])
    position = estimate_position([
        sighting(target - a, origin=a, observer="A"),
        sighting(target - b, origin=b, observer="B"),
    ])
    assert np.allclose(position, target, atol=1e-6)


RIGHT = np.array([1.0, 0.0, 0.0])
UP = np.array([0.0, 1.0, 0.0])


def test_format_clock():
    assert format_clock(3.0) == "3:00"
    assert format_clock(7.5) == "7:30"
    assert format_clock(0.02) == "12:00"
    assert format_clock(11.98) == "12:00"
    assert format_clock(12.4) == "12:20"


def test_map_offset_ignores_depth():
    shallow = map_offset(np.array([100.0, 100.0, 5.0]), RIGHT, UP)
    deep = map_offset(np.array([100.0, 100.0, -700.0]), RIGHT, UP)
    assert shallow.clock == deep.clock == "1:30"
    assert shallow.across == pytest.approx(deep.across) == pytest.approx(np.hypot(100, 100))


def test_map_line_direction_and_foreshortening():
    # Line of sight pointing left and 60 deg into the screen
    line = map_line(sighting([-1, 0, np.sqrt(3)], half_angle_deg=2.0), RIGHT, UP)
    assert line.clock == "9:00"
    assert line.across_per_pc == pytest.approx(0.5)
    # The same error circle looks wider on a foreshortened line
    flat = map_line(sighting([-1, 0, 0], half_angle_deg=2.0), RIGHT, UP)
    assert flat.spread_deg == pytest.approx(2.0)
    assert line.spread_deg > flat.spread_deg
