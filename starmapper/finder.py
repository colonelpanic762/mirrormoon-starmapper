"""Search helpers for stars that have been seen but not located.

A sighting is a line of sight from an observatory with an uncertainty
circle, marked by eye in the observatory view. The ship's map is a flat
projection, so on it a line of sight is a straight line running out from
the observatory's dot; with the map's orientation known (NavCalibration)
its direction can be given as a clock direction.
"""

from dataclasses import dataclass

import numpy as np

from .coords import Observation, to_coordinate_string
from .triangulation import triangulate


Bounds = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]

CLUSTER_BOUNDS: Bounds = ((0, 999), (0, 999), (0, 999))


@dataclass
class Sighting:
    """A marked line of sight to a star from an observatory."""
    observer_name: str
    origin: np.ndarray          # Observatory position
    direction: np.ndarray       # Unit vector toward the star
    cone_half_angle: float      # Radius of the user's error circle (radians)


@dataclass
class NearbyStar:
    """A known star in relation to a line of sight."""
    name: str
    angle_deg: float   # Angle from the line of sight, seen from the observatory
    distance: float    # Distance from the observatory (PC)


def distance_range(
    sighting: Sighting,
    bounds: Bounds = CLUSTER_BOUNDS,
) -> tuple[float, float] | None:
    """Range of distances from the observatory at which the star could be.

    A distance counts if the error circle at that distance reaches into the
    bounds (the whole cluster, or the cube decoded from an original name).

    Returns:
        (nearest, farthest) in PC, or None if the circle never reaches the bounds.
    """
    low = np.array([b[0] for b in bounds], dtype=float)
    high = np.array([b[1] for b in bounds], dtype=float)

    t = np.arange(0.0, 1800.0, 1.0)  # Longer than the cluster's diagonal
    points = sighting.origin + t[:, None] * sighting.direction
    outside = np.linalg.norm(points - np.clip(points, low, high), axis=1)
    reachable = t[outside <= t * np.tan(sighting.cone_half_angle) + 1e-9]

    if len(reachable) == 0:
        return None
    return float(reachable[0]), float(reachable[-1])


def _nearby(sighting: Sighting, known: dict[str, np.ndarray]) -> list[NearbyStar]:
    """All known stars with their angle from the line of sight, nearest first."""
    result = []
    for name, position in known.items():
        offset = np.asarray(position, dtype=float) - sighting.origin
        distance = float(np.linalg.norm(offset))
        if distance < 1e-6:
            continue
        cos_angle = np.clip(np.dot(offset / distance, sighting.direction), -1, 1)
        result.append(NearbyStar(name, float(np.degrees(np.arccos(cos_angle))), distance))
    return sorted(result, key=lambda star: star.angle_deg)


def stars_in_circle(sighting: Sighting, known: dict[str, np.ndarray]) -> list[NearbyStar]:
    """Known stars inside the error circle: candidates the user can rule out."""
    limit = np.degrees(sighting.cone_half_angle)
    return [star for star in _nearby(sighting, known) if star.angle_deg <= limit]


def estimate_position(sightings: list[Sighting]) -> np.ndarray | None:
    """Position where sightings from different observatories meet.

    Returns None unless at least two sightings come from different places.
    """
    origins = {tuple(np.round(s.origin, 6)) for s in sightings}
    if len(origins) < 2:
        return None

    observations = [
        Observation(
            observer_coordinates=to_coordinate_string(*(int(round(c)) for c in s.origin)),
            direction_vector=tuple(s.direction),
            cone_half_angle=s.cone_half_angle,
            timestamp="",
        )
        for s in sightings
    ]
    return triangulate(observations, method="ray_intersection").estimated_position


@dataclass
class MapOffset:
    """Where something appears on the ship's map relative to a reference dot."""
    clock_hours: float   # 12 = up, 3 = right
    across: float        # Distance across the map, in PC

    @property
    def clock(self) -> str:
        return format_clock(self.clock_hours)


@dataclass
class MapLine:
    """A line of sight as drawn on the ship's map."""
    clock_hours: float       # Direction the line leaves the observatory's dot
    across_per_pc: float     # Map distance covered per PC along the line of sight (0-1)
    spread_deg: float        # Half-width of the wedge on the map, from the error circle

    @property
    def clock(self) -> str:
        return format_clock(self.clock_hours)


def format_clock(hours: float) -> str:
    """Clock direction to the nearest 10 minutes, e.g. "7:30"."""
    minutes = round(hours * 60 / 10) * 10 % 720
    return f"{minutes // 60 or 12}:{minutes % 60:02d}"


def map_offset(offset: np.ndarray, right: np.ndarray, up: np.ndarray) -> MapOffset:
    """Project a 3D offset onto the map's right/up axes."""
    x = float(np.dot(offset, right))
    y = float(np.dot(offset, up))
    hours = (np.arctan2(x, y) * 6 / np.pi) % 12
    return MapOffset(clock_hours=float(hours), across=float(np.hypot(x, y)))


def map_line(sighting: Sighting, right: np.ndarray, up: np.ndarray) -> MapLine:
    """How a sighting's line of sight appears on the ship's map.

    The wedge's half-width assumes the worst case, where the error circle's
    full radius lies across the map.
    """
    projected = map_offset(sighting.direction, right, up)
    spread = np.degrees(np.arctan2(np.tan(sighting.cone_half_angle), projected.across))
    return MapLine(
        clock_hours=projected.clock_hours,
        across_per_pc=projected.across,
        spread_deg=float(spread),
    )
