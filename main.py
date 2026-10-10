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
    plot_observatory_view,
    create_interactive_observatory_view,
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

    with open(data_path) as f:
        data = json.load(f)

    def from_same_source(obs_data: dict) -> bool:
        view_state = obs_data.get("view_state") or {}
        return all(view_state.get(k) == v for k, v in source.items())

    data["observations"] = [
        o for o in data.get("observations", []) if not from_same_source(o)
    ]
    data["observations"].append(observation.to_dict())

    temp_path = data_path.with_suffix('.json.tmp')
    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=2)
    temp_path.rename(data_path)

    return observation


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
        help="Show 2D star chart from this observatory (by name)",
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        default=0.0,
        help="Horizontal view direction in degrees (default: 0)",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=45.0,
        help="Vertical view direction in degrees (default: 45)",
    )
    parser.add_argument(
        "--rotation",
        type=float,
        default=0.0,
        help="View roll in degrees (default: 0)",
    )
    parser.add_argument(
        "--flip-horizontal",
        action="store_true",
        help="Mirror the view horizontally",
    )
    parser.add_argument(
        "--flip-vertical",
        action="store_true",
        help="Mirror the view vertically",
    )
    # Interactive and triangulation arguments
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Enable click-to-observe mode (requires --observatory)",
    )
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
    if args.observatory:
        observer = find_star_by_name(stars, args.observatory)
        if observer is None:
            print(f"\nError: Star '{args.observatory}' not found")
            return 1
        if not observer.has_position:
            print(f"\nError: Star '{observer.display_name}' has no known coordinates")
            return 1

        print(f"\n== Observatory View from {observer.display_name} ==")
        print(f"Position: ({observer.position[0]:.0f}, {observer.position[1]:.0f}, {observer.position[2]:.0f})")
        print(f"View: azimuth={args.azimuth}°, elevation={args.elevation}°, rotation={args.rotation}°")

        # Convert degrees to radians
        az_rad = np.radians(args.azimuth)
        el_rad = np.radians(args.elevation)
        rot_rad = np.radians(args.rotation)

        if args.interactive:
            # Interactive mode: click to record observations
            print("\n[INTERACTIVE MODE]")
            print("  1. Click to mark anomaly center")
            print("  2. Click again to set uncertainty radius")
            print("  Observation will auto-save to data file\n")

            def on_observation_complete(observation):
                save_observation(args.data, observation)
                deg = np.degrees(observation.cone_half_angle)
                print(f"\n✓ Observation saved!")
                print(f"  ID: {observation.id}")
                print(f"  Direction: ({observation.direction_vector[0]:.3f}, "
                      f"{observation.direction_vector[1]:.3f}, {observation.direction_vector[2]:.3f})")
                print(f"  Cone half-angle: {deg:.1f}°")

            fig, state = create_interactive_observatory_view(
                observer=observer,
                stars=stars,
                center_azimuth=az_rad,
                center_elevation=el_rad,
                rotation=rot_rad,
                flip_horizontal=args.flip_horizontal,
                flip_vertical=args.flip_vertical,
                on_observation_complete=on_observation_complete,
            )
        else:
            # Non-interactive view
            fig = plot_observatory_view(
                observer=observer,
                stars=stars,
                center_azimuth=az_rad,
                center_elevation=el_rad,
                rotation=rot_rad,
                show_labels=not args.no_labels,
                flip_horizontal=args.flip_horizontal,
                flip_vertical=args.flip_vertical,
            )

        if args.save:
            fig.savefig(args.save, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
            print(f"Observatory view saved to: {args.save}")
        else:
            print("Launching observatory view...")
            import matplotlib.pyplot as plt
            plt.show()

    return 0


if __name__ == "__main__":
    exit(main())
