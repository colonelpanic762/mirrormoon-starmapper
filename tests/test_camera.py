"""Tests for the first-person moon camera."""

import numpy as np
import pytest

from starmapper.calibration import build_camera_frame, project_star_raw
from starmapper.camera import MoonCamera, horizontal_fov


def make_camera(pitch_deg=30.0):
    return MoonCamera.looking_at(np.array([-0.6, 0.3, 0.7]), np.radians(pitch_deg))


def axes(cam):
    return np.array([cam.right, cam.camera_up, cam.forward])


def assert_orthonormal_left_handed(cam):
    m = axes(cam)
    assert np.allclose(m @ m.T, np.eye(3), atol=1e-9)
    # Game frame is left-handed: right = up x forward
    assert np.allclose(np.cross(cam.camera_up, cam.forward), cam.right, atol=1e-9)


def test_looking_at_points_forward_at_target():
    cam = make_camera()
    target = np.array([-0.6, 0.3, 0.7])
    assert np.allclose(cam.forward, target / np.linalg.norm(target))
    assert_orthonormal_left_handed(cam)


def test_horizontal_fov_from_vertical():
    # ~84 deg measured on 1280x800; the same vertical FOV gives 90 deg at 16:9
    assert np.degrees(horizontal_fov(16 / 10)) == pytest.approx(84.0, abs=0.2)
    assert np.degrees(horizontal_fov(16 / 9)) == pytest.approx(90.0, abs=0.2)


def test_walking_full_circle_returns_to_start():
    cam = make_camera()
    start = axes(cam)
    for _ in range(36):
        cam.walk(np.radians(10))
    assert np.allclose(axes(cam), start, atol=1e-9)
    assert_orthonormal_left_handed(cam)


def test_turning_full_circle_returns_to_start():
    cam = make_camera()
    start = axes(cam)
    for _ in range(24):
        cam.turn(np.radians(15))
    assert np.allclose(axes(cam), start, atol=1e-9)


def test_turning_keeps_local_up_fixed():
    cam = make_camera()
    up = cam.local_up.copy()
    cam.turn(np.radians(37))
    assert np.allclose(cam.local_up, up)
    assert not np.allclose(cam.forward, make_camera().forward)


def test_camera_is_tilted_up_from_horizon_by_pitch():
    cam = make_camera(pitch_deg=30)
    elevation = np.degrees(np.arcsin(np.dot(cam.forward, cam.local_up)))
    assert elevation == pytest.approx(30.0)


def test_walking_keeps_right_axis_and_moves_local_up_forward():
    cam = make_camera()
    right = cam.right.copy()
    up_before = cam.local_up.copy()
    forward_before = cam.forward.copy()
    cam.walk(np.radians(10))
    assert np.allclose(cam.right, right)
    # Standing point moved 10 deg around the moon, toward where we were facing
    assert np.degrees(np.arccos(np.dot(cam.local_up, up_before))) == pytest.approx(10.0)
    assert np.dot(cam.local_up - up_before, forward_before) > 0


def test_turning_right_moves_stars_left_on_screen():
    cam = make_camera()
    star = cam.forward.copy()
    cam.turn(np.radians(5))
    x, _, depth = cam.project(star, aspect=1.6)
    assert depth > 0 and x < 0


def test_walking_forward_moves_stars_ahead_up_on_screen():
    cam = make_camera()
    star = cam.forward.copy()
    cam.walk(np.radians(5))
    _, y, depth = cam.project(star, aspect=1.6)
    assert depth > 0 and y > 0


def test_twist_keeps_forward_and_rolls_the_view():
    cam = make_camera()
    forward = cam.forward.copy()
    up = cam.local_up.copy()
    cam.twist(np.radians(40))
    assert np.allclose(cam.forward, forward)
    assert not np.allclose(cam.local_up, up)
    assert_orthonormal_left_handed(cam)


def test_changing_pitch_keeps_the_view():
    cam = make_camera(pitch_deg=30)
    before = axes(cam)
    cam.pitch = np.radians(20)
    assert np.allclose(axes(cam), before)
    elevation = np.degrees(np.arcsin(np.dot(cam.forward, cam.local_up)))
    assert elevation == pytest.approx(20.0)


def test_project_and_unproject_round_trip():
    cam = make_camera()
    cam.turn(0.3)
    cam.walk(0.2)
    cam.twist(0.5)
    for x, y in [(0.0, 0.0), (0.7, -0.3), (-0.9, 0.9)]:
        direction = cam.screen_to_direction(x, y, aspect=1.6)
        px, py, depth = cam.project(direction, aspect=1.6)
        assert depth > 0
        assert (px, py) == pytest.approx((x, y))


def test_star_behind_camera_has_negative_depth():
    cam = make_camera()
    _, _, depth = cam.project(-cam.forward, aspect=1.6)
    assert depth < 0


def test_matches_screenshot_calibration_projection():
    """Same star lands on the same pixel as the validated screenshot model."""
    up_theta, up_phi, yaw, pitch = 0.9, -2.1, 1.3, np.radians(30)
    forward, right, camera_up = build_camera_frame(up_theta, up_phi, yaw, pitch)
    cam = MoonCamera(right=right, camera_up=camera_up, forward=forward, pitch=pitch)

    width, height = 1280, 800
    fov_h = horizontal_fov(width / height)
    observer = np.array([590.0, 949.0, 307.0])
    star = observer + 300 * (forward + 0.4 * right - 0.25 * camera_up)

    px, py, _ = project_star_raw(
        star, observer, up_theta, up_phi, yaw, pitch, fov_h, width, height,
    )
    x, y, depth = cam.project(star - observer, aspect=width / height)
    assert depth > 0
    assert width / 2 * (1 + x) == pytest.approx(px)
    assert height / 2 * (1 - y) == pytest.approx(py)
