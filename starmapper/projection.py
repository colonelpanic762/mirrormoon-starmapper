"""Projection math for observatory sky views.

This module handles the conversion from 3D galaxy coordinates to 2D star chart
positions as seen from an observer. Uses stereographic projection which preserves
angles (conformal), making constellation patterns appear correct.

Coordinate conventions:
- Galaxy: X, Y, Z each 0-999 (from 9-digit coordinate string)
- Azimuth: 0 = +X direction, increases counterclockwise when viewed from +Y
- Elevation: 0 = horizon (XZ plane), +90° = zenith (+Y direction)
- Chart: Origin at center of view, X increases right, Y increases up
"""

from dataclasses import dataclass
import numpy as np

from .coords import Star


@dataclass
class ChartPosition:
    """Position of a star on the 2D star chart after projection."""
    x: float              # Horizontal position on chart
    y: float              # Vertical position on chart
    visible: bool         # Whether the star is in the visible hemisphere
    distance: float       # Distance from observer (for sizing)


@dataclass
class ViewState:
    """Current viewing orientation for the observatory."""
    azimuth: float = 0.0        # Horizontal look direction [0, 2*pi)
    elevation: float = np.pi/4  # Vertical look direction [-pi/2, pi/2]
    rotation: float = 0.0       # Roll angle [0, 2*pi)

    def rotate_horizontal(self, delta: float) -> None:
        """Rotate view left/right (A/D keys)."""
        self.azimuth = (self.azimuth + delta) % (2 * np.pi)

    def rotate_vertical(self, delta: float) -> None:
        """Rotate view up/down (W/S keys)."""
        self.elevation = np.clip(
            self.elevation + delta,
            -np.pi/2 + 0.1,
            np.pi/2 - 0.1
        )

    def roll(self, delta: float) -> None:
        """Roll the view."""
        self.rotation = (self.rotation + delta) % (2 * np.pi)


def compute_direction_vectors(
    observer_pos: np.ndarray,
    target_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute unit direction vectors from observer to each target.

    Args:
        observer_pos: (3,) array - observer's XYZ coordinates
        target_positions: (N, 3) array - target stars' XYZ coordinates

    Returns:
        directions: (N, 3) array of unit vectors pointing from observer to each target
        distances: (N,) array of distances to each target
    """
    # Subtract observer position to get relative vectors
    relative = target_positions - observer_pos

    # Compute distances
    distances = np.linalg.norm(relative, axis=1)

    # Normalize to unit vectors (handle zero distance)
    safe_distances = np.where(distances == 0, 1, distances)
    directions = relative / safe_distances[:, np.newaxis]

    return directions, distances


def cartesian_to_spherical(
    direction_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert direction vectors to spherical coordinates (azimuth, elevation).

    Convention:
    - Azimuth: angle in XZ plane from +X axis, counterclockwise when viewed from +Y
    - Elevation: angle from XZ plane, positive toward +Y

    Args:
        direction_vectors: (N, 3) unit vectors

    Returns:
        azimuths: (N,) array in radians [0, 2*pi)
        elevations: (N,) array in radians [-pi/2, pi/2]
    """
    x = direction_vectors[:, 0]
    y = direction_vectors[:, 1]
    z = direction_vectors[:, 2]

    # Elevation: angle from XZ plane (arcsin of Y component for unit vectors)
    elevations = np.arcsin(np.clip(y, -1, 1))

    # Azimuth: angle in XZ plane from +X axis
    azimuths = np.arctan2(z, x)
    azimuths = np.mod(azimuths, 2 * np.pi)  # Normalize to [0, 2*pi)

    return azimuths, elevations


def stereographic_projection(
    azimuths: np.ndarray,
    elevations: np.ndarray,
    center_azimuth: float = 0.0,
    center_elevation: float = np.pi / 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project spherical coordinates onto 2D chart using stereographic projection.

    The projection is centered on (center_azimuth, center_elevation) and projects
    the hemisphere facing that direction. Stereographic projection preserves angles
    (conformal), so constellation shapes appear correct.

    Args:
        azimuths: (N,) azimuth angles in radians
        elevations: (N,) elevation angles in radians
        center_azimuth: Azimuth of chart center (radians)
        center_elevation: Elevation of chart center (radians)

    Returns:
        x: (N,) horizontal chart coordinates
        y: (N,) vertical chart coordinates
        visible: (N,) boolean mask for stars in the visible hemisphere
    """
    # Convert center point to Cartesian on unit sphere
    cx = np.cos(center_elevation) * np.cos(center_azimuth)
    cy = np.sin(center_elevation)
    cz = np.cos(center_elevation) * np.sin(center_azimuth)
    center = np.array([cx, cy, cz])

    # Convert all points to Cartesian on unit sphere
    px = np.cos(elevations) * np.cos(azimuths)
    py = np.sin(elevations)
    pz = np.cos(elevations) * np.sin(azimuths)
    points = np.stack([px, py, pz], axis=1)

    # Build local coordinate frame at center point
    # "up" is toward zenith projected onto tangent plane
    up_raw = np.array([0, 1, 0])  # Global up (+Y)

    # Right vector: perpendicular to both center and up
    right = np.cross(center, up_raw)
    if np.linalg.norm(right) < 1e-6:
        # Looking straight up or down, use +X as right
        right = np.array([1, 0, 0])
    right = right / np.linalg.norm(right)

    # Recompute up to be perpendicular to both center and right
    up = np.cross(right, center)
    up = up / np.linalg.norm(up)

    # Compute angle between center direction and each point
    cos_angle = np.dot(points, center)

    # Points behind the observer (cos_angle < 0) are not visible
    # Use small negative threshold to include horizon
    visible = cos_angle > -0.1

    # Stereographic projection scale factor
    # r = 2 * tan(theta/2) where theta is angular distance from center
    # Using identity: tan(theta/2) = sin(theta) / (1 + cos(theta))
    # Scale factor is 2 / (1 + cos_angle)
    scale = 2.0 / (1.0 + cos_angle + 1e-10)

    # Project onto local right/up axes
    x = np.dot(points, right) * scale
    y = np.dot(points, up) * scale

    return x, y, visible


def apply_view_rotation(
    x: np.ndarray,
    y: np.ndarray,
    rotation_angle: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the chart view by a given angle.

    This simulates rotating the camera around its view axis (roll).

    Args:
        x: (N,) horizontal chart coordinates
        y: (N,) vertical chart coordinates
        rotation_angle: Rotation in radians, counterclockwise

    Returns:
        x_rot: (N,) rotated horizontal coordinates
        y_rot: (N,) rotated vertical coordinates
    """
    cos_r = np.cos(rotation_angle)
    sin_r = np.sin(rotation_angle)

    x_rot = x * cos_r - y * sin_r
    y_rot = x * sin_r + y * cos_r

    return x_rot, y_rot


def project_stars_from_observer(
    observer: Star,
    stars: list[Star],
    center_azimuth: float = 0.0,
    center_elevation: float = np.pi / 4,
    rotation: float = 0.0,
) -> list[tuple[Star, ChartPosition]]:
    """Project all stars onto a 2D chart as seen from the observer.

    Args:
        observer: The Star object representing the observatory location
        stars: List of stars to project
        center_azimuth: Direction the observer is facing (radians)
        center_elevation: Elevation angle of view center (radians)
        rotation: View rotation around the center direction (radians)

    Returns:
        List of (star, chart_position) tuples for stars with known coordinates
    """
    if observer.position is None:
        raise ValueError("Observer must have known coordinates")

    # Filter to stars with positions (excluding observer itself)
    valid_stars = [
        s for s in stars
        if s.has_position and s.coordinates != observer.coordinates
    ]

    if not valid_stars:
        return []

    # Get positions as array
    positions = np.array([s.position for s in valid_stars])

    # Compute directions and distances
    directions, distances = compute_direction_vectors(
        observer.position, positions
    )

    # Convert to spherical coordinates
    azimuths, elevations = cartesian_to_spherical(directions)

    # Project to 2D chart
    x, y, visible = stereographic_projection(
        azimuths, elevations, center_azimuth, center_elevation
    )

    # Apply view rotation
    x, y = apply_view_rotation(x, y, rotation)

    # Build result list
    results = []
    for i, star in enumerate(valid_stars):
        pos = ChartPosition(
            x=x[i],
            y=y[i],
            visible=visible[i],
            distance=distances[i]
        )
        results.append((star, pos))

    return results


def chart_to_direction(
    x: float,
    y: float,
    center_azimuth: float,
    center_elevation: float,
    rotation: float = 0.0,
) -> np.ndarray:
    """Convert a chart position back to a 3D direction vector.

    This is the inverse of the projection, used when the user clicks
    on the chart to mark an anomaly position.

    Args:
        x: Horizontal chart coordinate
        y: Vertical chart coordinate
        center_azimuth: Azimuth of chart center (radians)
        center_elevation: Elevation of chart center (radians)
        rotation: View rotation that was applied (radians)

    Returns:
        Unit vector (3,) in galaxy coordinate system pointing toward
        the clicked position in the sky
    """
    # Undo view rotation
    cos_r = np.cos(-rotation)
    sin_r = np.sin(-rotation)
    x_unrot = x * cos_r - y * sin_r
    y_unrot = x * sin_r + y * cos_r

    # Inverse stereographic projection
    # Given x, y on chart, find point on unit sphere
    r_sq = x_unrot**2 + y_unrot**2
    denom = 4.0 + r_sq

    # Local coordinates on unit sphere (in tangent frame)
    lx = 4.0 * x_unrot / denom
    ly = 4.0 * y_unrot / denom
    lz = (4.0 - r_sq) / denom  # Component toward center direction

    # Build the same local frame as in stereographic_projection
    cx = np.cos(center_elevation) * np.cos(center_azimuth)
    cy = np.sin(center_elevation)
    cz = np.cos(center_elevation) * np.sin(center_azimuth)
    center = np.array([cx, cy, cz])

    up_raw = np.array([0, 1, 0])
    right = np.cross(center, up_raw)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1, 0, 0])
    right = right / np.linalg.norm(right)
    up = np.cross(right, center)
    up = up / np.linalg.norm(up)

    # Transform from local frame to global coordinates
    # Point in local frame: lx * right + ly * up + lz * center
    direction = lx * right + ly * up + lz * center

    # Normalize (should already be unit length, but ensure)
    direction = direction / np.linalg.norm(direction)

    return direction
