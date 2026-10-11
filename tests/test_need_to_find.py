"""Tests for recording sightings, the finder report, and locating stars."""

import json
from pathlib import Path

import numpy as np
import pytest

import main
from starmapper import ObservatoryView

DATA = Path(__file__).parent.parent / "data"


@pytest.fixture
def data_path(tmp_path):
    """Real star data with a clean slate of sightings."""
    data = json.loads((DATA / "stars.json").read_text())
    for star in data["stars"]:
        star.pop("sightings", None)
        star.pop("seen_from", None)
        # The tests use AUT56 as their not-yet-located star
        if star.get("player_name") == "AUT56":
            star["coordinates"] = None
    # Ship's map square to the cluster: right = +X, up = +Y
    data["nav_calibration"] = {
        "ship_position": [437, 391, 781],
        "left_samples": [[282, 391, 788], [212, 391, 287]],
        "right_samples": [], "up_samples": [],
        "down_samples": [[437, 264, 976], [437, 212, 189]],
    }
    path = tmp_path / "stars.json"
    path.write_text(json.dumps(data))
    return path


def entry(data_path, name):
    stars = json.loads(data_path.read_text())["stars"]
    return next(s for s in stars if name in (s.get("player_name"), s.get("original_name")))


def direction_between(data_path, observer_name, star_name):
    stars = main.load_stars(data_path)
    a = np.array(main.find_star_by_name(stars, observer_name).position, dtype=float)
    b = np.array(main.find_star_by_name(stars, star_name).position, dtype=float)
    return (b - a) / np.linalg.norm(b - a)


def test_marking_a_need_to_find_star_in_the_view_saves_a_sighting(data_path):
    stars = main.load_stars(data_path)
    observer = main.find_star_by_name(stars, "OBS1")
    main.record_visible_stars(data_path, observer, ["AUT56"])
    saved = []
    view = ObservatoryView(
        observer=observer, stars=stars, visible_names=["GIOVAN", "AUY/14"],
        unlocated_names=main.unlocated_stars_seen_from(data_path, observer),
        on_sighting=lambda *args: saved.append(args),
        on_observation=lambda o: pytest.fail("should not save an anomaly observation"),
    )
    view.key_press('1')
    assert view.target == "AUT56"
    view.click(0.1, 0.2)
    view.click(0.15, 0.2)

    assert len(saved) == 1 and saved[0][0] == "AUT56"
    assert "AUT56" in view.sightings
    view.key_press('0')
    assert view.target == "ANOMALY"


def test_sighting_is_stored_once_per_observatory(data_path):
    stars = main.load_stars(data_path)
    obs1 = main.find_star_by_name(stars, "OBS1")
    obs007 = main.find_star_by_name(stars, "OBS007")
    main.save_sighting(data_path, "AUT56", obs1, np.array([1.0, 0, 0]), 0.05)
    main.save_sighting(data_path, "AUT56", obs1, np.array([0, 1.0, 0]), 0.03)
    main.save_sighting(data_path, "AUT56", obs007, np.array([0, 0, 1.0]), 0.02)

    sightings = entry(data_path, "AUT56")["sightings"]
    assert [(s["observer"], s["direction"]) for s in sightings] == [
        ("OBS1", [0.0, 1.0, 0.0]), ("OBS007", [0.0, 0.0, 1.0]),
    ]
    assert main.load_sightings_from(data_path, obs1).keys() == {"AUT56"}


def test_need_to_find_list_persists_between_runs(data_path):
    stars = main.load_stars(data_path)
    observer = main.find_star_by_name(stars, "OBS1")
    main.record_visible_stars(data_path, observer, ["AUT56", "UAT/02"])
    # A later run that names nothing still knows what is outstanding here
    assert main.unlocated_stars_seen_from(data_path, observer) == ["AUT56", "UAT/02"]
    assert main.unlocated_stars_seen_from(
        data_path, main.find_star_by_name(stars, "OBS007")) == []


def test_find_report_for_a_marked_star(data_path):
    # Pretend GIOVAN's neighbour is unknown: sight "UAT/02" 5 deg from GIOVAN
    stars = main.load_stars(data_path)
    observer = main.find_star_by_name(stars, "OBS1")
    main.record_visible_stars(data_path, observer, ["UAT/02"])
    direction = direction_between(data_path, "OBS1", "GIOVAN")
    main.save_sighting(data_path, "UAT/02", observer, direction, np.radians(2.0))

    report = main.format_find_report(data_path, "uat/02")
    assert "UAT/02" in report and "OBS1" in report
    assert "rule out" in report.lower() and "GIOVAN" in report
    # GIOVAN is at (297, 927, 558) and OBS1 at (590, 949, 307): on a map with
    # right = +X and up = +Y that is almost due left
    assert "along 8:50 on a clock face" in report
    assert "For scale: GIOVAN is at 8:50, 294 PC across" in report
    # Name cube: x 300-399, y 000-099, z 200-299
    assert "300-399" in report


def test_find_report_without_map_calibration_says_so(data_path):
    data = json.loads(data_path.read_text())
    del data["nav_calibration"]
    data_path.write_text(json.dumps(data))

    stars = main.load_stars(data_path)
    observer = main.find_star_by_name(stars, "OBS1")
    main.record_visible_stars(data_path, observer, ["AUT56"])
    main.save_sighting(data_path, "AUT56", observer, np.array([1.0, 0, 0]), 0.05)
    report = main.format_find_report(data_path, "AUT56")
    assert "not calibrated" in report
    assert "clock face" not in report


def test_find_report_for_an_unmarked_star_says_so(data_path):
    stars = main.load_stars(data_path)
    main.record_visible_stars(data_path, main.find_star_by_name(stars, "OBS1"), ["AUT56"])
    report = main.format_find_report(data_path, "AUT56")
    assert "not marked" in report.lower()


def test_find_report_estimates_position_from_two_observatories(data_path):
    stars = main.load_stars(data_path)
    for name in ("OBS1", "OBS007"):
        observer = main.find_star_by_name(stars, name)
        main.record_visible_stars(data_path, observer, ["MYSTERY"])
        main.save_sighting(data_path, "MYSTERY", observer,
                           direction_between(data_path, name, "ROBERT"), np.radians(1.0))
    report = main.format_find_report(data_path, "MYSTERY")
    assert "(282, 669, 591)" in report


def test_find_overview_lists_all_outstanding_stars(data_path):
    stars = main.load_stars(data_path)
    observer = main.find_star_by_name(stars, "OBS1")
    main.record_visible_stars(data_path, observer, ["AUT56", "UAT/02"])
    main.save_sighting(data_path, "AUT56", observer, np.array([1.0, 0, 0]), 0.05)
    overview = main.format_find_overview(data_path)
    assert "AUT56" in overview and "UAT/02" in overview
    assert "marked" in overview


def test_locating_a_star_sets_coordinates_and_scores_sightings(data_path):
    stars = main.load_stars(data_path)
    observer = main.find_star_by_name(stars, "OBS1")
    main.record_visible_stars(data_path, observer, ["AUT56"])
    true_position = np.array([300.0, 700.0, 500.0])
    origin = np.array(observer.position, dtype=float)
    direction = (true_position - origin) / np.linalg.norm(true_position - origin)
    main.save_sighting(data_path, "AUT56", observer, direction, np.radians(2.0))

    message = main.mark_star_located(data_path, "AUT56", "300700500")
    assert entry(data_path, "AUT56")["coordinates"] == "300700500"
    assert "0.0°" in message and "OBS1" in message
    assert main.unlocated_stars_seen_from(data_path, observer) == []


def test_locating_rejects_bad_coordinates(data_path):
    with pytest.raises(ValueError):
        main.mark_star_located(data_path, "AUT56", "12345")
