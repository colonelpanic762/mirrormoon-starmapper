"""Tests for the first-person observatory view and its data handling."""

import json
from pathlib import Path

import numpy as np
import pytest

import main
from starmapper import Observation, ObservatorySettings, ObservatoryView
from starmapper.camera import MoonCamera
from starmapper.stitching import load_session, solve_stitching

DATA = Path(__file__).parent.parent / "data"


@pytest.fixture
def data_path(tmp_path):
    """A copy of the real data without anything the by-eye view has saved."""
    data = json.loads((DATA / "stars.json").read_text())
    data.pop("observatory_camera", None)
    data["observations"] = [
        o for o in data["observations"]
        if (o.get("view_state") or {}).get("source") != "observatory_view"
    ]
    path = tmp_path / "stars.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def stars():
    return main.load_stars(DATA / "stars.json")


def make_view(stars, **kwargs):
    observer = main.find_star_by_name(stars, "OBS1")
    kwargs.setdefault("visible_names", ["GIOVAN", "AUY/14", "DAV21"])
    return ObservatoryView(observer=observer, stars=stars, **kwargs)


def test_opening_view_has_all_visible_stars_in_frame(stars):
    view = make_view(stars)
    aspect = view.settings.aspect
    for name in ["GIOVAN", "AUY/14", "DAV21"]:
        x, y, depth = view.camera.project(view.directions[view.names.index(name)], aspect)
        assert depth > 0 and abs(x) < 1 and abs(y) < 1, name


def test_check_star_is_hidden_from_the_view(stars):
    robert = main.find_star_by_name(stars, "ROBERT")
    assert "ROBERT" in make_view(stars).names
    assert "ROBERT" not in make_view(stars, check_star=robert).names


def test_held_keys_move_at_game_rates(stars):
    view = make_view(stars, settings=ObservatorySettings(turn_rate_deg_s=40.0))
    up = view.camera.local_up.copy()
    view.key_press('w')
    view.advance(1.5)
    # Walking 10 deg/s for 1.5 s
    assert np.degrees(np.arccos(np.dot(view.camera.local_up, up))) == pytest.approx(15.0)

    view.key_release('w')
    view._held.clear()
    forward = view.camera.forward_horizontal.copy()
    view.key_press('d')
    view.advance(0.5)
    turned = np.degrees(np.arccos(np.dot(view.camera.forward_horizontal, forward)))
    assert turned == pytest.approx(20.0)


def test_shift_moves_slowly(stars):
    view = make_view(stars)
    up = view.camera.local_up.copy()
    view.key_press('W')
    view.advance(1.0)
    assert np.degrees(np.arccos(np.dot(view.camera.local_up, up))) == pytest.approx(2.0)


def test_changing_tilt_keeps_the_view_and_reports_settings(stars):
    seen = []
    view = make_view(stars, on_settings_changed=lambda s: seen.append(s.tilt_deg))
    forward = view.camera.forward.copy()
    view.key_press(']')
    assert seen == [14.0]
    assert np.allclose(view.camera.forward, forward)


def test_two_clicks_make_an_observation(stars):
    saved = []
    view = make_view(stars, on_observation=saved.append)
    assert view.click(0.2, 0.1) is None

    observation = view.click(0.3, 0.1)
    assert saved == [observation]
    assert observation.observer_coordinates == view.observer.coordinates

    aspect = view.settings.aspect
    center = view.camera.screen_to_direction(0.2 / aspect, 0.1, aspect)
    edge = view.camera.screen_to_direction(0.3 / aspect, 0.1, aspect)
    assert np.allclose(observation.direction, center)
    assert observation.cone_half_angle == pytest.approx(np.arccos(np.dot(center, edge)))
    assert observation.view_state["source"] == "observatory_view"


def test_escape_cancels_a_half_made_mark(stars):
    view = make_view(stars)
    view.click(0.2, 0.1)
    view.key_press('escape')
    assert view.mark_direction is None
    assert view.click(0.5, 0.5) is None  # Counts as a first click again


def test_mark_at_true_orientation_finds_the_anomaly(stars):
    """With the view set to the orientation fitted from the OBS1 screenshot,
    clicking where the anomaly was marked lands close to ROBERT."""
    observer = main.find_star_by_name(stars, "OBS1")
    robert = main.find_star_by_name(stars, "ROBERT")
    session = load_session(DATA / "stitch-OBS1.json", observer, stars)
    result = solve_stitching(session, progress_callback=lambda message: None)
    calibration = result.best_solution().calibrations[0]

    view = make_view(stars, check_star=robert)
    forward = calibration.forward_vector()
    view.camera = MoonCamera.looking_at(forward, view.camera.pitch)
    # Twist until GIOVAN sits where it was marked in the screenshot
    image = session.images[0]
    aspect = image.image_width / image.image_height
    giovan = next(m for m in image.known_markers if m.name == "GIOVAN")
    target = np.array([
        (giovan.pixel_x / image.image_width * 2 - 1) * aspect,
        1 - giovan.pixel_y / image.image_height * 2,
    ])
    direction = view.directions[view.names.index("GIOVAN")]
    best = min(
        np.radians(np.arange(0, 360, 0.25)),
        key=lambda a: _twisted_distance(view, direction, a, target),
    )
    view.camera.twist(best)

    anomaly = image.get_correspondence("ANOMALY")
    plot_x = (anomaly.pixel_x / image.image_width * 2 - 1) * aspect
    plot_y = 1 - anomaly.pixel_y / image.image_height * 2
    view.click(plot_x, plot_y)
    observation = view.click(plot_x + 0.05, plot_y)

    assert np.degrees(view.check_error(observation)) < 1.5


def _twisted_distance(view, direction, angle, target):
    camera = MoonCamera(view.camera.right.copy(), view.camera.camera_up.copy(),
                        view.camera.forward.copy(), view.camera.pitch)
    camera.twist(angle)
    aspect = view.settings.aspect
    x, y, _ = camera.project(direction, aspect)
    return np.hypot(x * aspect - target[0], y - target[1])


def test_visible_stars_are_sorted_and_unlocated_ones_recorded(data_path, stars):
    observer = main.find_star_by_name(stars, "OBS1")
    located, unlocated = main.record_visible_stars(
        data_path, observer, ["giovan", "AUT56", "NEWONE", "ZZZ/12"],
    )
    assert located == ["GIOVAN"]
    assert unlocated == ["AUT56", "NEWONE", "ZZZ/12"]

    entries = json.loads(data_path.read_text())["stars"]
    by_name = {e.get("player_name") or e.get("original_name"): e for e in entries}
    assert by_name["AUT56"]["seen_from"] == ["OBS1"]
    assert by_name["NEWONE"]["coordinates"] is None
    assert by_name["ZZZ/12"]["original_name"] == "ZZZ/12"

    # Recording again adds nothing
    main.record_visible_stars(data_path, observer, ["AUT56", "NEWONE"])
    assert len(json.loads(data_path.read_text())["stars"]) == len(entries)


def test_by_eye_observation_replaces_only_its_own_kind(data_path):
    before = main.load_observations(data_path)

    def by_eye(direction):
        return Observation(
            observer_coordinates="590949307",
            direction_vector=direction,
            cone_half_angle=0.05,
            timestamp="2026-10-10T00:00:00Z",
            id=str(direction),
            view_state={"source": "observatory_view"},
        )

    main.replace_observation(data_path, by_eye((1.0, 0.0, 0.0)), {"source": "observatory_view"})
    main.replace_observation(data_path, by_eye((0.0, 1.0, 0.0)), {"source": "observatory_view"})

    after = main.load_observations(data_path)
    # The stitched OBS1 observation and the other observatories' are kept
    assert len(after) == len(before) + 1
    assert after[-1].direction_vector == (0.0, 1.0, 0.0)


def test_settings_round_trip(data_path):
    assert main.load_observatory_settings(data_path) == ObservatorySettings()
    settings = ObservatorySettings(tilt_deg=17.0, turn_rate_deg_s=60.0, aspect=16 / 9)
    main.save_observatory_settings(data_path, settings)
    assert main.load_observatory_settings(data_path) == settings
