#!/usr/bin/env python3
"""Star Mapper - MirrorMoon EP anomaly triangulation tool."""

import argparse
import json
from pathlib import Path

import numpy as np

from starmapper import (
    Star,
    parse_coordinates,
    distance,
    show_map,
    plot_star_map,
    plot_observatory_view,
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

    # Show or save 3D visualization if requested
    if args.visualize or (args.save and not args.observatory):
        fig = plot_star_map(stars, show_labels=not args.no_labels)
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
