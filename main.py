#!/usr/bin/env python3
"""Star Mapper - MirrorMoon EP anomaly triangulation tool."""

import argparse
import json
from pathlib import Path

from starmapper import Star, parse_coordinates, distance


def load_stars(data_path: Path) -> list[Star]:
    """Load stars from a JSON file."""
    with open(data_path) as f:
        data = json.load(f)

    stars = []
    for star_data in data.get("stars", []):
        stars.append(Star(
            coordinates=star_data["coordinates"],
            original_name=star_data["original_name"],
            player_name=star_data.get("player_name"),
            discovered_by_user=star_data.get("discovered_by_user", False),
            has_observatory=star_data.get("has_observatory", False),
            observatory_active=star_data.get("observatory_active", False),
            anomaly_visible=star_data.get("anomaly_visible", False),
        ))

    return stars


def main():
    parser = argparse.ArgumentParser(description="MirrorMoon EP Star Mapper")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/stars.json"),
        help="Path to star data JSON file",
    )
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Data file not found: {args.data}")
        return 1

    stars = load_stars(args.data)
    print(f"Loaded {len(stars)} stars")

    for star in stars:
        x, y, z = parse_coordinates(star.coordinates)
        print(f"  {star.display_name}: ({x}, {y}, {z})")

    if len(stars) >= 2:
        d = distance(stars[0], stars[1])
        print(f"\nDistance between {stars[0].display_name} and {stars[1].display_name}: {d:.2f}")

    return 0


if __name__ == "__main__":
    exit(main())
