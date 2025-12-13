"""Triangulation algorithms for locating anomalies from multiple observations.

Uses ray/cone intersection to estimate the 3D position where observation
directions converge, along with confidence estimates.
"""

from dataclasses import dataclass
from typing import Literal
import numpy as np
from scipy.optimize import minimize

from .coords import Observation, parse_coordinates, to_coordinate_string


@dataclass
class TriangulationResult:
    """Result of triangulating multiple observations."""
    estimated_position: np.ndarray  # (x, y, z) coordinates
    coordinate_string: str  # 9-digit format for display
    confidence_radius: float  # Uncertainty sphere radius
    observations_used: int
    method: str
    residual_error: float  # Average angular error from observations


def triangulate(
    observations: list[Observation],
    method: Literal["ray_intersection", "closest_approach"] = "closest_approach",
) -> TriangulationResult | None:
    """Triangulate anomaly position from multiple observations.

    Args:
        observations: List of observations (need at least 2)
        method: Algorithm to use:
            - "ray_intersection": Least-squares intersection of direction rays
            - "closest_approach": Minimize distance to cone surfaces (more robust)

    Returns:
        TriangulationResult if successful, None if insufficient observations
    """
    if len(observations) < 2:
        return None

    if method == "ray_intersection":
        return _triangulate_ray_intersection(observations)
    else:
        return _triangulate_closest_approach(observations)


def _triangulate_ray_intersection(observations: list[Observation]) -> TriangulationResult:
    """Find point minimizing squared distance to all observation rays.

    Uses least-squares to solve the overdetermined system where we want
    to find P such that P is as close as possible to all lines:
        L_i = O_i + t * d_i

    The solution minimizes: sum_i || P - closest_point_on_line_i ||^2
    """
    # Build the normal equation matrices
    # For each ray, the projection matrix onto the perpendicular subspace is:
    # P_i = I - d_i @ d_i.T
    # We want: sum_i P_i @ X = sum_i P_i @ O_i

    n = len(observations)
    A = np.zeros((3, 3))
    b = np.zeros(3)

    for obs in observations:
        origin = obs.origin
        direction = obs.direction

        # Projection matrix onto perpendicular subspace
        P = np.eye(3) - np.outer(direction, direction)

        A += P
        b += P @ origin

    # Solve the system
    try:
        position = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if singular
        position = np.linalg.pinv(A) @ b

    # Clamp to galaxy bounds
    position = np.clip(position, 0, 999)

    # Calculate residual error (average angular deviation)
    residual = _calculate_residual_error(position, observations)

    # Confidence radius based on cone angles and residual
    avg_cone = np.mean([obs.cone_half_angle for obs in observations])
    confidence = _estimate_confidence_radius(position, observations, residual)

    return TriangulationResult(
        estimated_position=position,
        coordinate_string=_position_to_coordinate_string(position),
        confidence_radius=confidence,
        observations_used=len(observations),
        method="ray_intersection",
        residual_error=residual,
    )


def _triangulate_closest_approach(observations: list[Observation]) -> TriangulationResult:
    """Find point minimizing weighted distance to observation cone surfaces.

    More robust than ray intersection as it accounts for the uncertainty
    cones rather than treating observations as perfect rays.
    """
    # Initial guess: centroid of observer positions
    initial = np.mean([obs.origin for obs in observations], axis=0)

    def objective(pos):
        """Sum of squared distances to cone surfaces."""
        total = 0.0
        for obs in observations:
            dist = point_to_cone_distance(
                pos, obs.origin, obs.direction, obs.cone_half_angle
            )
            # Weight by inverse cone angle (tighter cones = more weight)
            weight = 1.0 / (obs.cone_half_angle + 0.01)
            total += weight * dist ** 2
        return total

    # Optimize
    result = minimize(
        objective,
        initial,
        method='L-BFGS-B',
        bounds=[(0, 999), (0, 999), (0, 999)],
    )

    position = result.x

    # Calculate residual error
    residual = _calculate_residual_error(position, observations)

    # Confidence radius
    confidence = _estimate_confidence_radius(position, observations, residual)

    return TriangulationResult(
        estimated_position=position,
        coordinate_string=_position_to_coordinate_string(position),
        confidence_radius=confidence,
        observations_used=len(observations),
        method="closest_approach",
        residual_error=residual,
    )


def point_to_cone_distance(
    point: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    half_angle: float,
) -> float:
    """Calculate signed distance from point to cone surface.

    Returns:
        Positive if outside cone, negative if inside, zero on surface.
    """
    # Vector from cone apex (origin) to point
    to_point = point - origin
    dist_to_apex = np.linalg.norm(to_point)

    if dist_to_apex < 1e-10:
        return 0.0

    # Angle between cone axis and direction to point
    to_point_normalized = to_point / dist_to_apex
    cos_angle = np.clip(np.dot(to_point_normalized, direction), -1.0, 1.0)
    actual_angle = np.arccos(cos_angle)

    # Angular distance from cone surface
    angle_error = actual_angle - half_angle

    # Convert angular error to approximate linear distance
    # This is the perpendicular distance to the cone surface
    return angle_error * dist_to_apex


def point_to_ray_distance(
    point: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
) -> float:
    """Calculate perpendicular distance from point to ray."""
    to_point = point - origin

    # Project onto ray direction
    projection_length = np.dot(to_point, direction)

    # Only consider forward direction (ray, not line)
    if projection_length < 0:
        return np.linalg.norm(to_point)

    # Closest point on ray
    closest = origin + projection_length * direction

    return np.linalg.norm(point - closest)


def _calculate_residual_error(position: np.ndarray, observations: list[Observation]) -> float:
    """Calculate average angular deviation from estimated position."""
    errors = []
    for obs in observations:
        # Direction from observer to estimated position
        to_position = position - obs.origin
        dist = np.linalg.norm(to_position)

        if dist < 1e-10:
            errors.append(0.0)
            continue

        actual_direction = to_position / dist
        cos_angle = np.clip(np.dot(actual_direction, obs.direction), -1.0, 1.0)
        angle = np.arccos(cos_angle)

        errors.append(angle)

    return float(np.mean(errors))


def _estimate_confidence_radius(
    position: np.ndarray,
    observations: list[Observation],
    residual: float,
) -> float:
    """Estimate confidence radius based on geometry and residual error.

    Uses the average distance to observers and cone angles to estimate
    the spatial uncertainty.
    """
    # Average distance to observers
    distances = [np.linalg.norm(position - obs.origin) for obs in observations]
    avg_distance = np.mean(distances)

    # Average cone half-angle
    avg_cone = np.mean([obs.cone_half_angle for obs in observations])

    # Base uncertainty from cone angles at the estimated distance
    base_uncertainty = avg_distance * np.tan(avg_cone)

    # Scale by residual error and number of observations
    # More observations = lower uncertainty
    n_factor = 1.0 / np.sqrt(len(observations))

    # Residual factor: if residual >> cone angle, increase uncertainty
    residual_factor = max(1.0, residual / (avg_cone + 0.01))

    return base_uncertainty * n_factor * residual_factor


def _position_to_coordinate_string(position: np.ndarray) -> str:
    """Convert position array to 9-digit coordinate string."""
    x = int(np.clip(np.round(position[0]), 0, 999))
    y = int(np.clip(np.round(position[1]), 0, 999))
    z = int(np.clip(np.round(position[2]), 0, 999))
    return to_coordinate_string(x, y, z)


def format_result(result: TriangulationResult) -> str:
    """Format triangulation result for display."""
    pos = result.estimated_position
    deg_error = np.degrees(result.residual_error)

    lines = [
        f"=== Triangulation Result ===",
        f"Estimated position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})",
        f"Coordinate string: {result.coordinate_string}",
        f"Confidence radius: {result.confidence_radius:.1f} units",
        f"Observations used: {result.observations_used}",
        f"Method: {result.method}",
        f"Residual error: {deg_error:.2f}°",
    ]
    return "\n".join(lines)
