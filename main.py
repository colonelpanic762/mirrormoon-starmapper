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
        "--ref-ll",
        type=str,
        metavar="COORDS",
        help="Lower-left reference star coordinates (9 digits)",
    )
    parser.add_argument(
        "--ref-lr",
        type=str,
        metavar="COORDS",
        help="Lower-right reference star coordinates (9 digits)",
    )
    parser.add_argument(
        "--ref-ur",
        type=str,
        metavar="COORDS",
        help="Upper-right reference star coordinates (9 digits)",
    )
    parser.add_argument(
        "--ref-ul",
        type=str,
        metavar="COORDS",
        help="Upper-left reference star coordinates (9 digits)",
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

    # Handle navigation
    if args.navigate:
        # Validate that all reference stars are provided
        refs = {
            'll': args.ref_ll,
            'lr': args.ref_lr,
            'ur': args.ref_ur,
            'ul': args.ref_ul,
        }
        missing = [k for k, v in refs.items() if v is None]
        if missing:
            print(f"\nError: Navigation requires all 4 reference stars.")
            print(f"Missing: {', '.join('--ref-' + m for m in missing)}")
            print("\nUsage: python main.py --navigate TYU/29 \\")
            print("         --ref-ll 649496127 --ref-lr 795566583 \\")
            print("         --ref-ur 826552773 --ref-ul 627461143")
            return 1

        try:
            result = navigate_to_target(refs, args.navigate)
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
