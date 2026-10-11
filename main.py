#!/usr/bin/env python3
"""Star Mapper - MirrorMoon EP anomaly triangulation tool."""

import argparse
import json
from pathlib import Path

import numpy as np

from starmapper import (
    Star,
    Observation,
    parse_coordinates,
    distance,
    show_map,
    plot_star_map,
    ObservatorySettings,
    plot_star_map_with_observations,
    triangulate,
    format_result,
    navigate_to_target,
    navigate_with_calibration,
    NavCalibration,
    format_navigation_result,
)


def find_star_by_name(stars: list[Star], name: str) -> Star | None:
    """Find a star by player_name or original_name (case-insensitive)."""
    name_lower = name.lower()
    for star in stars:
        if star.player_name and star.player_name.lower() == name_lower:
            return star
        if star.original_name and star.original_name.lower() == name_lower:
            return star
    return None


def load_stars(data_path: Path) -> list[Star]:
    """Load stars from a JSON file."""
    with open(data_path) as f:
        data = json.load(f)

    stars = []
    for star_data in data.get("stars", []):
        star = Star(
            coordinates=star_data.get("coordinates"),
            original_name=star_data.get("original_name"),
            player_name=star_data.get("player_name"),
            discovered_by_user=star_data.get("discovered_by_user", False),
            has_observatory=star_data.get("has_observatory"),
            observatory_active=star_data.get("observatory_active"),
            anomaly_visible=star_data.get("anomaly_visible"),
            notes=star_data.get("notes"),
        )
        # Handle detector field (not in dataclass, stored as attribute)
        if star_data.get("has_detector"):
            star.has_detector = True
        stars.append(star)

    return stars


def load_observations(data_path: Path) -> list[Observation]:
    """Load observations from the JSON data file."""
    with open(data_path) as f:
        data = json.load(f)

    observations = []
    for obs_data in data.get("observations", []):
        observations.append(Observation.from_dict(obs_data))

    return observations


def save_observation(data_path: Path, observation: Observation) -> None:
    """Append a new observation to the JSON data file."""
    with open(data_path) as f:
        data = json.load(f)

    if "observations" not in data:
        data["observations"] = []

    data["observations"].append(observation.to_dict())

    # Write atomically (write to temp, then rename)
    temp_path = data_path.with_suffix('.json.tmp')
    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=2)
    temp_path.rename(data_path)


def save_stitched_observation(
    data_path: Path,
    observer: Star,
    result,
    label: str,
    session_path: Path,
) -> Observation | None:
    """Save a stitching result's direction to a label as an anomaly observation.

    Replaces any observation previously saved from the same session file and
    label, so re-running a solve does not pile up duplicates.

    Returns the saved Observation, or None if the label is not marked.
    """
    from datetime import datetime, timezone
    import uuid

    found = result.label_direction(label)
    if found is None:
        return None
    direction, spread = found
    best = result.best_solution()

    # Uncertainty: the worse of the reference-star fit and the disagreement
    # between images about where the labelled star is
    cone_half_angle = max(np.radians(best.known_star_error), spread)

    source = {
        "source": "stitching",
        "session": session_path.name,
        "label": label,
    }
    observation = Observation(
        observer_coordinates=observer.coordinates,
        direction_vector=tuple(float(c) for c in direction),
        cone_half_angle=float(cone_half_angle),
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        id=str(uuid.uuid4()),
        view_state={**source, "fov_horizontal": best.calibrations[0].fov_horizontal},
    )

    replace_observation(data_path, observation, source)
    return observation


def _write_data(data_path: Path, data: dict) -> None:
    """Write the JSON data file atomically (write to temp, then rename)."""
    temp_path = data_path.with_suffix('.json.tmp')
    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=2)
    temp_path.rename(data_path)


def replace_observation(data_path: Path, observation: Observation, match: dict) -> None:
    """Save an observation, replacing earlier ones from the same observer
    whose view_state has all the key/value pairs in match."""
    with open(data_path) as f:
        data = json.load(f)

    def superseded(obs_data: dict) -> bool:
        view_state = obs_data.get("view_state") or {}
        return (
            obs_data.get("observer_coordinates") == observation.observer_coordinates
            and all(view_state.get(k) == v for k, v in match.items())
        )

    data["observations"] = [
        o for o in data.get("observations", []) if not superseded(o)
    ]
    data["observations"].append(observation.to_dict())
    _write_data(data_path, data)


def load_observatory_settings(data_path: Path) -> ObservatorySettings:
    """Load saved first-person view settings, or defaults."""
    with open(data_path) as f:
        saved = json.load(f).get("observatory_camera", {})
    settings = ObservatorySettings()
    for key, value in saved.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    return settings


def save_observatory_settings(data_path: Path, settings: ObservatorySettings) -> None:
    """Save first-person view settings to the JSON data file."""
    with open(data_path) as f:
        data = json.load(f)
    new = {
        "tilt_deg": settings.tilt_deg,
        "turn_rate_deg_s": settings.turn_rate_deg_s,
        "aspect": settings.aspect,
        "horizon_below_center_deg": settings.horizon_below_center_deg,
    }
    if data.get("observatory_camera") != new:
        data["observatory_camera"] = new
        _write_data(data_path, data)


def record_visible_stars(
    data_path: Path,
    observer: Star,
    names: list[str],
) -> tuple[list[str], list[str]]:
    """Sort stars seen from an observatory into located and unlocated.

    Unlocated stars (no coordinates, or not in the database yet) are recorded
    in the data file with the observatory added to their "seen_from" list.

    Returns:
        (located, unlocated) lists of display names
    """
    with open(data_path) as f:
        data = json.load(f)

    located = []
    unlocated = []
    changed = False

    for name in names:
        entry = next(
            (s for s in data["stars"]
             if name.lower() in ((s.get("player_name") or "").lower(),
                                 (s.get("original_name") or "").lower())),
            None,
        )
        if entry is not None and entry.get("coordinates"):
            located.append(entry.get("player_name") or entry.get("original_name"))
            continue

        if entry is None:
            # Original names look like "XXX/##"; anything else is a player name
            is_original = len(name) == 6 and name[3] == "/"
            entry = {
                "coordinates": None,
                "original_name": name.upper() if is_original else None,
                "player_name": None if is_original else name.upper(),
                "discovered_by_user": False,
                "has_observatory": None,
                "observatory_active": None,
                "anomaly_visible": None,
            }
            data["stars"].append(entry)
            changed = True

        seen_from = entry.setdefault("seen_from", [])
        if observer.display_name not in seen_from:
            seen_from.append(observer.display_name)
            changed = True
        unlocated.append(entry.get("player_name") or entry.get("original_name"))

    if changed:
        _write_data(data_path, data)

    return located, unlocated


def _find_entry(data: dict, name: str) -> dict | None:
    """Find a star's raw data entry by player or original name (case-insensitive)."""
    wanted = name.lower()
    for entry in data["stars"]:
        if wanted in ((entry.get("player_name") or "").lower(),
                      (entry.get("original_name") or "").lower()):
            return entry
    return None


def _entry_name(entry: dict) -> str:
    return entry.get("player_name") or entry.get("original_name")


def unlocated_stars_seen_from(data_path: Path, observer: Star) -> list[str]:
    """Names of stars without coordinates that were seen from an observatory."""
    with open(data_path) as f:
        data = json.load(f)
    return [
        _entry_name(entry) for entry in data["stars"]
        if not entry.get("coordinates")
        and observer.display_name in entry.get("seen_from", [])
    ]


def save_sighting(
    data_path: Path,
    star_name: str,
    observer: Star,
    direction: np.ndarray,
    cone_half_angle: float,
) -> None:
    """Save a line of sight to an unlocated star, replacing the earlier
    sighting of it from the same observatory."""
    from datetime import datetime, timezone

    with open(data_path) as f:
        data = json.load(f)
    entry = _find_entry(data, star_name)
    if entry is None:
        raise ValueError(f"Star '{star_name}' not found")

    sightings = [
        s for s in entry.get("sightings", [])
        if s["observer"] != observer.display_name
    ]
    sightings.append({
        "observer": observer.display_name,
        "direction": [float(c) for c in direction],
        "cone_half_angle": float(cone_half_angle),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    })
    entry["sightings"] = sightings
    _write_data(data_path, data)


def load_sightings_from(data_path: Path, observer: Star) -> dict[str, tuple[np.ndarray, float]]:
    """Sightings of still-unlocated stars from an observatory:
    name -> (direction, cone_half_angle)."""
    with open(data_path) as f:
        data = json.load(f)
    result = {}
    for entry in data["stars"]:
        if entry.get("coordinates"):
            continue
        for s in entry.get("sightings", []):
            if s["observer"] == observer.display_name:
                result[_entry_name(entry)] = (np.array(s["direction"]), s["cone_half_angle"])
    return result


def _entry_sightings(entry: dict, stars: list[Star]):
    """Finder Sighting objects for a raw star entry (skips unknown observatories)."""
    from starmapper.finder import Sighting

    result = []
    for s in entry.get("sightings", []):
        observer = find_star_by_name(stars, s["observer"])
        if observer is None or not observer.has_position:
            continue
        result.append(Sighting(
            observer_name=observer.display_name,
            origin=np.array(observer.position, dtype=float),
            direction=np.array(s["direction"], dtype=float),
            cone_half_angle=s["cone_half_angle"],
        ))
    return result


def format_find_overview(data_path: Path) -> str:
    """List every unlocated star and where it has been seen and marked."""
    with open(data_path) as f:
        data = json.load(f)

    lines = ["== Need to find =="]
    for entry in data["stars"]:
        if entry.get("coordinates"):
            continue
        seen_from = entry.get("seen_from", [])
        marked_from = [s["observer"] for s in entry.get("sightings", [])]
        if seen_from:
            places = ", ".join(
                f"{name} ({'marked' if name in marked_from else 'not marked'})"
                for name in seen_from
            )
            lines.append(f"  {_entry_name(entry)}: seen from {places}")
        else:
            lines.append(f"  {_entry_name(entry)}: no sightings recorded")
    if len(lines) == 1:
        lines.append("  Nothing outstanding")
    return "\n".join(lines)


def format_find_report(data_path: Path, name: str) -> str:
    """Describe where to look for an unlocated star from each observatory
    it was marked from."""
    from starmapper.finder import (
        distance_range, stars_in_circle, estimate_position, map_line, map_offset,
        CLUSTER_BOUNDS,
    )
    from starmapper import decode_original_name

    with open(data_path) as f:
        data = json.load(f)
    entry = _find_entry(data, name)
    if entry is None:
        return f"Star '{name}' not found"

    display = _entry_name(entry)
    lines = [f"== Finding {display} =="]
    if entry.get("coordinates"):
        x, y, z = parse_coordinates(entry["coordinates"])
        lines.append(f"  Already located at ({x}, {y}, {z})")
        return "\n".join(lines)

    bounds = CLUSTER_BOUNDS
    if entry.get("original_name"):
        decoded = decode_original_name(entry["original_name"])
        if decoded:
            bounds = decoded
            lines.append(
                "  Name puts it at X {0}-{1}, Y {2}-{3}, Z {4}-{5}".format(
                    *(v for axis in decoded for v in axis))
            )

    stars = load_stars(data_path)
    known = {
        s.display_name: np.array(s.position, dtype=float)
        for s in stars if s.has_position
    }
    sightings = _entry_sightings(entry, stars)
    marked_from = {s.observer_name for s in sightings}
    calibration = load_calibration(data_path)
    axes = calibration.get_screen_axes() if calibration else None

    for observatory in entry.get("seen_from", []):
        if observatory not in marked_from:
            lines.append(f"\n  Seen from {observatory} but not marked yet.")
            lines.append(f"    Open the view (--observatory {observatory}) and mark it.")
    if not sightings and not entry.get("seen_from"):
        lines.append("  No sightings recorded.")

    for sighting in sightings:
        others = {n: p for n, p in known.items() if n != sighting.observer_name}
        lines.append(
            f"\n  From {sighting.observer_name} "
            f"(marked ±{np.degrees(sighting.cone_half_angle):.1f}°)"
        )
        reach = distance_range(sighting, bounds)
        if reach is None:
            lines.append("    Distance:    the mark does not pass through the name's region; "
                         "re-check the mark or the name")
        else:
            lines.append(f"    Distance:    {reach[0]:.0f}-{reach[1]:.0f} PC "
                         f"from {sighting.observer_name}")

        inside = stars_in_circle(sighting, others)
        if inside:
            lines.append("    Rule out:    " + ", ".join(
                f"{star.name} ({star.distance:.0f} PC)" for star in inside))
        else:
            lines.append("    Rule out:    no known stars inside your circle")

        if axes is None:
            lines.append("    Ship's map:  not calibrated (see --calibrate-nav), "
                         "so no search direction")
            continue

        line = map_line(sighting, axes.right, axes.up)
        dot = (f"{sighting.observer_name}'s dot (your diamond, if you are parked there)")
        lines.append(f"    On the ship's map, if it has not been rotated since calibrating:")
        if line.across_per_pc < 0.15:
            lines.append(f"      The line of sight points almost straight into the map, so")
            lines.append(f"      candidates bunch up around {dot}.")
        else:
            lines.append(f"      Search outward from {dot}")
            lines.append(f"      along {line.clock} on a clock face, "
                         f"within about {line.spread_deg:.0f}° either side.")
            if reach is not None:
                lines.append(
                    f"      That stretch runs from {reach[0] * line.across_per_pc:.0f} to "
                    f"{reach[1] * line.across_per_pc:.0f} PC across the map."
                )
        # Known stars near the line give a sense of scale on the unlabelled map
        scale = []
        for name, position in others.items():
            offset = map_offset(position - sighting.origin, axes.right, axes.up)
            turn = abs((offset.clock_hours - line.clock_hours + 6) % 12 - 6)
            if offset.across > 30:
                scale.append((turn, name, offset))
        for _, name, offset in sorted(scale, key=lambda item: item[0])[:3]:
            lines.append(f"      For scale: {name} is at {offset.clock}, "
                         f"{offset.across:.0f} PC across.")

    position = estimate_position(sightings)
    if position is not None:
        lines.append(
            f"\n  Estimated position from {len(marked_from)} observatories: "
            f"({position[0]:.0f}, {position[1]:.0f}, {position[2]:.0f})"
        )

    return "\n".join(lines)


def mark_star_located(data_path: Path, name: str, coordinates: str) -> str:
    """Record a found star's coordinates and score any sightings of it.

    Creates the star if it is not in the data file yet.

    Raises:
        ValueError: If coordinates is not a 9-digit string
    """
    x, y, z = parse_coordinates(coordinates)

    with open(data_path) as f:
        data = json.load(f)
    entry = _find_entry(data, name)
    if entry is None:
        is_original = len(name) == 6 and name[3] == "/"
        entry = {
            "coordinates": None,
            "original_name": name.upper() if is_original else None,
            "player_name": None if is_original else name.upper(),
            "discovered_by_user": False,
            "has_observatory": None,
            "observatory_active": None,
            "anomaly_visible": None,
        }
        data["stars"].append(entry)
    entry["coordinates"] = coordinates
    _write_data(data_path, data)

    lines = [f"✓ {_entry_name(entry)} located at ({x}, {y}, {z})"]
    position = np.array([x, y, z], dtype=float)
    for sighting in _entry_sightings(entry, load_stars(data_path)):
        offset = position - sighting.origin
        cos_angle = np.clip(
            np.dot(offset / np.linalg.norm(offset), sighting.direction), -1, 1)
        error = np.degrees(np.arccos(cos_angle))
        circle = np.degrees(sighting.cone_half_angle)
        verdict = "inside" if error <= circle else "outside"
        lines.append(
            f"  Your mark from {sighting.observer_name} was {error:.1f}° off "
            f"({verdict} your ±{circle:.1f}° circle)"
        )
    return "\n".join(lines)


def clear_all_observations(data_path: Path) -> int:
    """Remove all observations. Returns count deleted."""
    with open(data_path) as f:
        data = json.load(f)

    count = len(data.get("observations", []))
    data["observations"] = []

    temp_path = data_path.with_suffix('.json.tmp')
    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=2)
    temp_path.rename(data_path)

    return count


def load_calibration(data_path: Path) -> NavCalibration | None:
    """Load navigation calibration from the JSON data file."""
    with open(data_path) as f:
        data = json.load(f)

    cal_data = data.get("nav_calibration")
    if cal_data is None:
        return None

    return NavCalibration.from_dict(cal_data)


def save_calibration(data_path: Path, calibration: NavCalibration) -> None:
    """Save navigation calibration to the JSON data file."""
    with open(data_path) as f:
        data = json.load(f)

    data["nav_calibration"] = calibration.to_dict()

    # Write atomically
    temp_path = data_path.with_suffix('.json.tmp')
    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=2)
    temp_path.rename(data_path)


def clear_calibration(data_path: Path) -> bool:
    """Remove navigation calibration. Returns True if there was one to remove."""
    with open(data_path) as f:
        data = json.load(f)

    had_calibration = "nav_calibration" in data
    if had_calibration:
        del data["nav_calibration"]

        temp_path = data_path.with_suffix('.json.tmp')
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
        temp_path.rename(data_path)

    return had_calibration


def main():
    parser = argparse.ArgumentParser(description="MirrorMoon EP Star Mapper")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/stars.json"),
        help="Path to star data JSON file",
    )
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Show 3D visualization of star map",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Hide star labels in visualization (except observatories)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Save visualization to file instead of displaying (e.g., --save map.png)",
    )
    # Observatory view arguments
    parser.add_argument(
        "--observatory", "-o",
        type=str,
        help="Open the first-person view from this observatory (by name)",
    )
    parser.add_argument(
        "--visible",
        nargs='*',
        metavar="STAR",
        help="Stars you can see from the observatory. Known ones are highlighted "
             "and centered; the rest are recorded as needing to be found",
    )
    parser.add_argument(
        "--tilt",
        type=float,
        metavar="DEGREES",
        help="Upward camera tilt from the horizon (saved; default 13)",
    )
    parser.add_argument(
        "--turn-time",
        type=float,
        metavar="SECONDS",
        help="Seconds for one full turn with A or D held in-game (saved)",
    )
    parser.add_argument(
        "--aspect",
        type=str,
        metavar="W:H",
        help="Game window aspect ratio, e.g. 16:10 or 16:9 (saved; default 16:10)",
    )
    parser.add_argument(
        "--check-against",
        type=str,
        metavar="STAR",
        help="Hide this star (the known anomaly) and report how far each mark was from it",
    )
    parser.add_argument(
        "--find",
        nargs='?',
        const='',
        metavar="STAR",
        help="Where to look for a star that has been seen but not located. "
             "Without a name, lists all such stars",
    )
    parser.add_argument(
        "--located",
        nargs=2,
        metavar=("STAR", "COORDS"),
        help="Record the 9-digit coordinates of a star you found",
    )
    # Triangulation arguments
    parser.add_argument(
        "--triangulate", "-t",
        action="store_true",
        help="Run triangulation on stored observations",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["closest_approach", "ray_intersection"],
        default="closest_approach",
        help="Triangulation algorithm (default: closest_approach)",
    )
    parser.add_argument(
        "--show-cones",
        action="store_true",
        help="Show observation cones in 3D visualization",
    )
    parser.add_argument(
        "--list-observations",
        action="store_true",
        help="List all stored observations",
    )
    parser.add_argument(
        "--clear-observations",
        action="store_true",
        help="Delete all stored observations",
    )
    # Navigation arguments
    parser.add_argument(
        "--navigate", "-n",
        type=str,
        metavar="TARGET",
        help="Calculate direction to target (9-digit coords or original name like TYU/29)",
    )
    parser.add_argument(
        "--navigate-interactive", "-ni",
        type=str,
        metavar="TARGET",
        help="Interactive navigation with position updates (like a compass)",
    )
    parser.add_argument(
        "--ref-ll",
        type=str,
        metavar="COORDS",
        help="Lower-left reference star coordinates (9 digits) [legacy mode]",
    )
    parser.add_argument(
        "--ref-lr",
        type=str,
        metavar="COORDS",
        help="Lower-right reference star coordinates (9 digits) [legacy mode]",
    )
    parser.add_argument(
        "--ref-ur",
        type=str,
        metavar="COORDS",
        help="Upper-right reference star coordinates (9 digits) [legacy mode]",
    )
    parser.add_argument(
        "--ref-ul",
        type=str,
        metavar="COORDS",
        help="Upper-left reference star coordinates (9 digits) [legacy mode]",
    )
    # Calibration-based navigation
    parser.add_argument(
        "--calibrate-nav",
        action="store_true",
        help="Start/continue navigation calibration (interactive)",
    )
    parser.add_argument(
        "--add-sample",
        nargs=2,
        metavar=("DIRECTION", "COORDS"),
        help="Add calibration sample: --add-sample left 123456789",
    )
    parser.add_argument(
        "--ship-position",
        type=str,
        metavar="COORDS",
        help="Set ship position for calibration (9 digits)",
    )
    parser.add_argument(
        "--show-calibration",
        action="store_true",
        help="Show current navigation calibration status",
    )
    parser.add_argument(
        "--clear-calibration",
        action="store_true",
        help="Delete navigation calibration",
    )
    # Screenshot calibration tool
    parser.add_argument(
        "--calibrate-screenshot",
        type=Path,
        metavar="IMAGE",
        help="Open interactive tool to calibrate view from screenshot",
    )
    parser.add_argument(
        "--screenshot-observatory",
        type=str,
        metavar="NAME",
        help="Observatory name for screenshot calibration (required with --calibrate-screenshot)",
    )
    # Multi-image stitching
    parser.add_argument(
        "--stitch",
        nargs='+',
        type=Path,
        metavar="IMAGE",
        help="Stitch multiple screenshots to estimate unknown star positions",
    )
    parser.add_argument(
        "--stitch-session",
        type=Path,
        metavar="FILE",
        help="Session file for stitch markings: loaded if it exists, saved after every mark",
    )
    parser.add_argument(
        "--stitch-solve",
        action="store_true",
        help="Solve the --stitch-session file and print results without opening the GUI",
    )
    parser.add_argument(
        "--stitch-save-observation",
        type=str,
        metavar="LABEL",
        help="With --stitch-solve: save the direction to this correspondence label "
             "(e.g. ANOMALY) as an observation for --triangulate",
    )
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Data file not found: {args.data}")
        return 1

    stars = load_stars(args.data)

    stars_with_coords = [s for s in stars if s.has_position]
    stars_without_coords = [s for s in stars if not s.has_position]

    print(f"Loaded {len(stars)} stars ({len(stars_with_coords)} with coordinates, {len(stars_without_coords)} pending)")

    print("\n== Stars with known coordinates ==")
    for star in stars_with_coords:
        x, y, z = parse_coordinates(star.coordinates)
        status = ""
        if star.has_observatory:
            status = " [OBSERVATORY]"
        if star.anomaly_visible:
            status += " [ANOMALY VISIBLE]"
        print(f"  {star.display_name}: ({x}, {y}, {z}){status}")

    if stars_without_coords:
        print("\n== Stars pending coordinates ==")
        for star in stars_without_coords:
            print(f"  {star.display_name}")

    if len(stars_with_coords) >= 2:
        s1, s2 = stars_with_coords[0], stars_with_coords[1]
        d = distance(s1, s2)
        print(f"\nDistance between {s1.display_name} and {s2.display_name}: {d:.2f}")

    # Handle observation management flags
    if args.clear_observations:
        count = clear_all_observations(args.data)
        print(f"\nCleared {count} observations")
        return 0

    if args.list_observations:
        observations = load_observations(args.data)
        if not observations:
            print("\nNo observations stored")
        else:
            print(f"\n== Stored Observations ({len(observations)}) ==")
            for obs in observations:
                deg = np.degrees(obs.cone_half_angle)
                print(f"  [{obs.id[:8]}...] From {obs.observer_coordinates} | "
                      f"Half-angle: {deg:.1f}° | {obs.timestamp}")
        return 0

    # Handle screenshot calibration tool
    if args.calibrate_screenshot:
        if not args.screenshot_observatory:
            print("\nError: --screenshot-observatory required with --calibrate-screenshot")
            return 1

        observer = find_star_by_name(stars, args.screenshot_observatory)
        if observer is None:
            print(f"\nError: Observatory '{args.screenshot_observatory}' not found")
            return 1
        if not observer.has_position:
            print(f"\nError: Observatory '{observer.display_name}' has no known coordinates")
            return 1

        if not args.calibrate_screenshot.exists():
            print(f"\nError: Screenshot not found: {args.calibrate_screenshot}")
            return 1

        from starmapper.screenshot_tool import create_interactive_session
        import matplotlib.pyplot as plt

        print(f"\n== Screenshot Calibration ==")
        print(f"Observatory: {observer.display_name} at {observer.position}")
        print(f"Screenshot: {args.calibrate_screenshot}")
        print()
        print("Instructions:")
        print("  1. Type a star name in the text box and press Enter")
        print("  2. Click on that star in the image")
        print("  3. Repeat for 2+ stars to calibrate")
        print("  4. Green circles = your marks, Red X = projected positions")
        print()

        def on_complete(session):
            print(f"\n✓ Calibration complete!")
            print(session.calibration)

        fig, session = create_interactive_session(
            args.calibrate_screenshot,
            observer,
            stars,
            on_calibration_complete=on_complete
        )
        plt.show()
        return 0

    # Handle multi-image stitching
    if args.stitch_solve and not args.stitch_session:
        print("\nError: --stitch-session required with --stitch-solve")
        return 1
    if args.stitch_save_observation and not args.stitch_solve:
        print("\nError: --stitch-solve required with --stitch-save-observation")
        return 1

    if args.stitch or args.stitch_session:
        have_session_file = bool(args.stitch_session and args.stitch_session.exists())
        if not have_session_file and (args.stitch_solve or not args.stitch):
            print(f"\nError: Session file not found: {args.stitch_session}")
            return 1

        observatory_name = args.screenshot_observatory
        if not observatory_name and have_session_file:
            observatory_name = json.loads(args.stitch_session.read_text()).get("observatory")
        if not observatory_name:
            print("\nError: --screenshot-observatory required with --stitch")
            return 1

        observer = find_star_by_name(stars, observatory_name)
        if observer is None:
            print(f"\nError: Observatory '{observatory_name}' not found")
            return 1
        if not observer.has_position:
            print(f"\nError: Observatory '{observer.display_name}' has no known coordinates")
            return 1

        session = None
        if have_session_file:
            from starmapper.stitching import load_session
            session = load_session(args.stitch_session, observer, stars)
            args.stitch = [img.image_path for img in session.images]
            print(f"\nLoaded stitching session from {args.stitch_session}")

        # Check all images exist
        for img_path in args.stitch:
            if not img_path.exists():
                print(f"\nError: Image not found: {img_path}")
                return 1

        if args.stitch_solve:
            from starmapper.stitching import solve_stitching
            from starmapper.stitching_tool import _print_navigation_results

            # Same settings as the GUI's Solve button
            result = solve_stitching(session, fixed_pitch=30.0)
            if not result.solutions:
                print("No valid solutions found")
                return 1
            _print_navigation_results(session, result)

            if args.stitch_save_observation:
                label = args.stitch_save_observation.upper()
                observation = save_stitched_observation(
                    args.data, observer, result, label, args.stitch_session,
                )
                if observation is None:
                    print(f"Error: '{label}' is not marked in this session")
                    return 1
                d = observation.direction_vector
                print(f"✓ Saved observation of {label} from {observer.display_name}")
                print(f"  Direction: ({d[0]:.4f}, {d[1]:.4f}, {d[2]:.4f})")
                print(f"  Cone half-angle: {np.degrees(observation.cone_half_angle):.2f}°")
            return 0

        from starmapper.stitching_tool import run_stitching_tool
        import matplotlib.pyplot as plt

        print(f"\n== Multi-Image Stitching ==")
        print(f"Observatory: {observer.display_name} at {observer.position}")
        print(f"Images: {len(args.stitch)}")
        for i, p in enumerate(args.stitch):
            print(f"  {i + 1}. {p.name}")
        print()
        print("Instructions:")
        print("  1. Select mode: 'Known Stars' or 'Correspondences'")
        print("  2. Known Stars: Type name → click on star (must be in database)")
        print("  3. Correspondences: Type label → click on star (any label)")
        print("  4. Use ← → arrows or Prev/Next to switch images")
        print("  5. Mark same unnamed stars with same label across images")
        print("  6. Click 'Solve' when ready")
        print()

        def on_result(result):
            print(f"\n✓ Stitching complete! {len(result.solutions)} solutions")
            best = result.best_solution()
            if best:
                print(f"RMS error: {best.total_error:.2f}°")

        session, state = run_stitching_tool(
            args.stitch, observer, stars,
            session=session, session_path=args.stitch_session,
        )
        return 0

    # Handle calibration management
    if args.clear_calibration:
        if clear_calibration(args.data):
            print("\nNavigation calibration cleared")
        else:
            print("\nNo calibration to clear")
        return 0

    if args.show_calibration:
        calibration = load_calibration(args.data)
        if calibration is None:
            print("\nNo navigation calibration stored")
            print("Use --calibrate-nav to start calibration")
        else:
            print("\n== Navigation Calibration ==")
            print(calibration.status())
        return 0

    if args.ship_position:
        calibration = load_calibration(args.data) or NavCalibration()
        calibration.ship_position = parse_coordinates(args.ship_position)
        save_calibration(args.data, calibration)
        print(f"\nShip position set to {calibration.ship_position}")
        return 0

    if args.add_sample:
        direction, coords = args.add_sample
        direction = direction.lower()
        if direction not in ("left", "right", "up", "down"):
            print(f"\nError: Invalid direction '{direction}'. Use: left, right, up, down")
            return 1

        calibration = load_calibration(args.data) or NavCalibration()
        sample_coords = parse_coordinates(coords)
        calibration.add_sample(direction, sample_coords)
        save_calibration(args.data, calibration)

        print(f"\n== Sample Added ==")
        print(f"Direction: {direction}")
        print(f"Coordinates: {sample_coords}")
        print(f"\n{calibration.status()}")
        return 0

    if args.calibrate_nav:
        # Interactive calibration mode
        calibration = load_calibration(args.data) or NavCalibration()

        print("\n== Navigation Calibration ==")
        print("For each direction, enter coordinates of stars visible in that direction.")
        print("Each sample = a star that's to the LEFT/RIGHT/UP/DOWN of your ship.\n")
        print("Commands: 'status', 'done', 'help', 'reset' (clear all samples)\n")

        if calibration.ship_position is None:
            print("First, set your ship's position.")
            ship_input = input("Ship position (9 digits): ").strip()
            if ship_input.lower() == 'done':
                save_calibration(args.data, calibration)
                print("\nCalibration saved.")
                return 0
            calibration.ship_position = parse_coordinates(ship_input)
            save_calibration(args.data, calibration)

        print(f"Ship: {calibration.ship_position}")
        print("(Use --ship-position to change if you've moved)\n")

        while True:
            try:
                line = input("Enter [direction] [coords] (or status/done/help): ").strip()
            except EOFError:
                break

            if not line:
                continue
            if line.lower() == 'done':
                break
            if line.lower() == 'status':
                print(f"\n{calibration.status()}\n")
                continue
            if line.lower() == 'help':
                print("\nCommands:")
                print("  left 123456789   - Add star that's to the LEFT of your ship")
                print("  right 123456789  - Add star that's to the RIGHT of your ship")
                print("  up 123456789     - Add star that's ABOVE your ship on screen")
                print("  down 123456789   - Add star that's BELOW your ship on screen")
                print("  status           - Show calibration progress")
                print("  reset            - Clear all samples and start over")
                print("  done             - Save and exit")
                print("\nTip: Stars further from ship center give better calibration.\n")
                continue
            if line.lower() == 'reset':
                calibration = NavCalibration(ship_position=calibration.ship_position)
                print("Samples cleared. Ship position kept.")
                continue

            parts = line.split()
            if len(parts) != 2:
                print("Invalid input. Use: [direction] [coords]")
                continue

            direction, coords = parts
            direction = direction.lower()
            if direction not in ("left", "right", "up", "down"):
                print(f"Invalid direction '{direction}'. Use: left, right, up, down")
                continue

            try:
                sample_coords = parse_coordinates(coords)
                calibration.add_sample(direction, sample_coords)
                vec = getattr(calibration, f"{direction}_vector")
                if vec is not None:
                    print(f"  ✓ {direction}: {sample_coords} → vector ({vec[0]:.2f}, {vec[1]:.2f}, {vec[2]:.2f})")
                else:
                    print(f"  + {direction}: {sample_coords} (need 1 more sample)")
            except ValueError as e:
                print(f"Error: {e}")

        save_calibration(args.data, calibration)
        print(f"\n{calibration.status()}")
        print("\nCalibration saved.")
        return 0

    # Handle interactive navigation (compass mode)
    if args.navigate_interactive:
        calibration = load_calibration(args.data)
        if calibration is None or not calibration.is_calibrated():
            print("\nError: No calibration found. Run --calibrate-nav first.")
            return 1

        target = args.navigate_interactive
        print(f"\n== Interactive Navigation to {target} ==")

        # Show initial direction
        try:
            result = navigate_with_calibration(calibration, target)
            coords = result.target_coords
            print(f"Target: {target} → ({coords[0]}, {coords[1]}, {coords[2]})")
            print("Update your position as you travel to get updated directions.")
            print("Enter coordinates of a nearby star, or 'done' to exit.\n")
            print(f"From ship position {calibration.ship_position}:")
            print(f"  → {result.clock_description} ({result.distance_from_view_center:.0f} PC)\n")
        except ValueError as e:
            print(f"Error: {e}")
            return 1

        # Interactive loop
        while True:
            try:
                line = input("Current position (or 'done'): ").strip()
            except EOFError:
                break

            if not line or line.lower() == 'done':
                break

            try:
                current_pos = parse_coordinates(line)
                # Calculate direction from new position (keeps same orientation/axes)
                result = navigate_with_calibration(calibration, target, from_position=current_pos)
                print(f"  → {result.clock_description} ({result.distance_from_view_center:.0f} PC)")
            except ValueError as e:
                print(f"  Error: {e}")

        print("\nNavigation complete.")
        return 0

    # Handle navigation
    if args.navigate:
        # Check if using legacy 4-corner mode
        refs = {
            'll': args.ref_ll,
            'lr': args.ref_lr,
            'ur': args.ref_ur,
            'ul': args.ref_ul,
        }
        has_any_ref = any(v is not None for v in refs.values())

        if has_any_ref:
            # Legacy mode: require all 4 corners
            missing = [k for k, v in refs.items() if v is None]
            if missing:
                print(f"\nError: Legacy navigation requires all 4 reference stars.")
                print(f"Missing: {', '.join('--ref-' + m for m in missing)}")
                return 1

            try:
                result = navigate_to_target(refs, args.navigate)
                print(f"\n== Navigation (legacy mode) ==")
                print(format_navigation_result(result, verbose=True))
            except ValueError as e:
                print(f"\nError: {e}")
                return 1
        else:
            # Calibration mode: use stored calibration
            calibration = load_calibration(args.data)
            if calibration is None or not calibration.is_calibrated():
                print("\nError: No calibration found. Either:")
                print("  1. Run --calibrate-nav to calibrate, OR")
                print("  2. Use legacy mode with --ref-ll/lr/ur/ul")
                return 1

            try:
                result = navigate_with_calibration(calibration, args.navigate)
                print(f"\n== Navigation ==")
                print(format_navigation_result(result, verbose=True))
            except ValueError as e:
                print(f"\nError: {e}")
                return 1
        return 0

    # Handle triangulation
    if args.triangulate:
        observations = load_observations(args.data)
        if len(observations) < 2:
            print(f"\nNeed at least 2 observations for triangulation (have {len(observations)})")
            return 1

        print(f"\n== Triangulating from {len(observations)} observations ==")
        result = triangulate(observations, method=args.method)
        print(format_result(result))

        # Show visualization if requested
        if args.visualize or args.show_cones:
            print("\nLaunching 3D visualization with cones...")
            fig = plot_star_map_with_observations(
                stars,
                observations=observations,
                result=result,
                show_labels=not args.no_labels,
            )
            if args.save:
                fig.savefig(args.save, dpi=150, bbox_inches='tight')
                print(f"Visualization saved to: {args.save}")
            else:
                import matplotlib.pyplot as plt
                plt.show()
        return 0

    # Show or save 3D visualization if requested (with optional cones)
    if args.visualize or args.show_cones or (args.save and not args.observatory):
        observations = load_observations(args.data) if args.show_cones else None
        fig = plot_star_map_with_observations(
            stars,
            observations=observations,
            show_labels=not args.no_labels,
        )
        if args.save:
            fig.savefig(args.save, dpi=150, bbox_inches='tight')
            print(f"\nVisualization saved to: {args.save}")
        else:
            print("\nLaunching 3D visualization...")
            import matplotlib.pyplot as plt
            plt.show()

    # Show observatory view if requested
    if args.located:
        try:
            print("\n" + mark_star_located(args.data, args.located[0], args.located[1]))
        except ValueError as error:
            print(f"\nError: {error}")
            return 1
        return 0

    if args.find is not None:
        if args.find:
            print("\n" + format_find_report(args.data, args.find))
        else:
            print("\n" + format_find_overview(args.data))
        return 0

    # Camera settings are saved whether or not the view is opened
    if args.tilt is not None or args.turn_time is not None or args.aspect is not None:
        settings = load_observatory_settings(args.data)
        if args.tilt is not None:
            settings.tilt_deg = args.tilt
        if args.turn_time is not None:
            if args.turn_time <= 0:
                print("\nError: --turn-time must be positive")
                return 1
            settings.turn_rate_deg_s = 360.0 / args.turn_time
        if args.aspect is not None:
            try:
                width, height = (float(part) for part in args.aspect.split(":"))
                settings.aspect = width / height
            except (ValueError, ZeroDivisionError):
                print(f"\nError: --aspect must look like 16:10, got '{args.aspect}'")
                return 1
        save_observatory_settings(args.data, settings)
        print(f"\nSaved camera settings: tilt {settings.tilt_deg:.0f}°, full turn "
              f"{360.0 / settings.turn_rate_deg_s:.1f} s, aspect {settings.aspect:.3f}")

    if args.observatory:
        observer = find_star_by_name(stars, args.observatory)
        if observer is None:
            print(f"\nError: Star '{args.observatory}' not found")
            return 1
        if not observer.has_position:
            print(f"\nError: Star '{observer.display_name}' has no known coordinates")
            return 1

        check_star = None
        if args.check_against:
            check_star = find_star_by_name(stars, args.check_against)
            if check_star is None or not check_star.has_position:
                print(f"\nError: --check-against star '{args.check_against}' has no known coordinates")
                return 1

        settings = load_observatory_settings(args.data)

        located, _ = record_visible_stars(args.data, observer, args.visible or [])
        unlocated = unlocated_stars_seen_from(args.data, observer)
        stars = load_stars(args.data)

        print(f"\n== Observatory View from {observer.display_name} ==")
        print(f"Position: ({observer.position[0]:.0f}, {observer.position[1]:.0f}, {observer.position[2]:.0f})")
        print(f"Camera tilt: {settings.tilt_deg:.0f}°, full turn: "
              f"{360.0 / settings.turn_rate_deg_s:.1f} s, aspect: {settings.aspect:.3f}")
        if located:
            print(f"Visible, known position: {', '.join(located)}")
        if unlocated:
            print(f"Visible, need to find:   {', '.join(unlocated)}")
        if check_star:
            print(f"Checking marks against {check_star.display_name} (hidden from the view)")

        def on_observation(observation):
            replace_observation(args.data, observation, {"source": "observatory_view"})
            print(f"\n✓ Observation saved")
            print(f"  Direction: ({observation.direction_vector[0]:.4f}, "
                  f"{observation.direction_vector[1]:.4f}, {observation.direction_vector[2]:.4f})")
            print(f"  Cone half-angle: {np.degrees(observation.cone_half_angle):.1f}°")
            error = view.check_error(observation)
            if error is not None:
                print(f"  {check_star.display_name} was {np.degrees(error):.1f}° from your mark")

        def on_sighting(name, direction, radius):
            save_sighting(args.data, name, observer, direction, radius)
            print(f"\n✓ Sighting of {name} saved (±{np.degrees(radius):.1f}°). "
                  f"See where to look with: --find {name}")

        from starmapper import ObservatoryView
        view = ObservatoryView(
            observer=observer,
            stars=stars,
            visible_names=located,
            unlocated_names=unlocated,
            settings=settings,
            check_star=check_star,
            on_observation=on_observation,
            on_settings_changed=lambda s: save_observatory_settings(args.data, s),
            sightings=load_sightings_from(args.data, observer),
            on_sighting=on_sighting,
        )

        if args.save:
            view.fig.savefig(args.save, dpi=150, facecolor=view.fig.get_facecolor())
            print(f"Observatory view saved to: {args.save}")
        else:
            print("Launching observatory view...")
            import matplotlib.pyplot as plt
            plt.show()

    return 0


if __name__ == "__main__":
    exit(main())
