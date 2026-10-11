"""Tests for ship's-map orientation calibration."""

import json
from pathlib import Path

import numpy as np
import pytest

from starmapper.navigation import NavCalibration

DATA = Path(__file__).parent.parent / "data"
SHIP = np.array([400.0, 500.0, 600.0])


def frame(seed=0):
    """A random orthonormal (right, up, depth) frame."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q[0], q[1], q[2]


def star(right, up, depth, across, height, into):
    """Coordinates of a star at a map offset from the ship, at some depth."""
    position = SHIP + across * right + height * up + into * depth
    return tuple(float(c) for c in position)


def calibrated(right, up, depth):
    cal = NavCalibration(ship_position=tuple(SHIP))
    # Stars level with the diamond, at wildly different depths
    cal.add_sample("left", star(right, up, depth, -150, 0, 20))
    cal.add_sample("left", star(right, up, depth, -220, 0, -480))
    cal.add_sample("right", star(right, up, depth, 90, 0, 300))
    # Stars directly above/below the diamond
    cal.add_sample("down", star(right, up, depth, 0, -130, 190))
    cal.add_sample("up", star(right, up, depth, 0, 260, -350))
    return cal


def test_depth_does_not_affect_the_fitted_axes():
    right, up, depth = frame()
    axes = calibrated(right, up, depth).get_screen_axes()
    assert np.allclose(axes.right, right, atol=1e-9)
    assert np.allclose(axes.up, up, atol=1e-9)


def test_two_samples_on_one_side_per_line_are_enough():
    right, up, depth = frame(1)
    cal = NavCalibration(ship_position=tuple(SHIP))
    cal.add_sample("left", star(right, up, depth, -150, 0, 20))
    cal.add_sample("left", star(right, up, depth, -220, 0, -480))
    assert not cal.is_calibrated()
    cal.add_sample("down", star(right, up, depth, 0, -130, 190))
    cal.add_sample("down", star(right, up, depth, 0, -300, -80))
    axes = cal.get_screen_axes()
    assert np.allclose(axes.right, right, atol=1e-9)
    assert np.allclose(axes.up, up, atol=1e-9)


def test_needs_ship_position():
    cal = NavCalibration()
    for direction, coords in [("left", (1, 2, 3)), ("left", (4, 9, 6)),
                              ("up", (7, 1, 9)), ("up", (2, 8, 4))]:
        cal.add_sample(direction, coords)
    assert not cal.is_calibrated()


def test_sample_errors_flag_a_misread_coordinate():
    right, up, depth = frame(2)
    cal = calibrated(right, up, depth)
    assert max(error for _, _, error in cal.sample_errors()) < 1e-6

    # A third left star read 40 PC too high
    cal.add_sample("left", star(right, up, depth, -300, 40, 100))
    worst = max(cal.sample_errors(), key=lambda item: item[2])
    assert worst[0] == "left" and worst[2] > 10


def test_december_samples_give_axis_aligned_map():
    saved = json.loads((DATA / "stars.json").read_text())["nav_calibration"]
    december = {
        "ship_position": [437, 391, 781],
        "left_samples": [[282, 390, 788], [212, 397, 287], [57, 394, 244]],
        "right_samples": [], "up_samples": [],
        "down_samples": [[442, 264, 976], [435, 212, 189], [434, 119, 971]],
    }
    cal = NavCalibration.from_dict(december)
    axes = cal.get_screen_axes()
    assert np.degrees(np.arccos(axes.right @ np.array([1.0, 0, 0]))) < 2
    assert np.degrees(np.arccos(axes.up @ np.array([0, 1.0, 0]))) < 2
    assert max(error for _, _, error in cal.sample_errors()) < 10
    assert "off" in cal.status()
    # The saved calibration still loads
    NavCalibration.from_dict(saved).status()
