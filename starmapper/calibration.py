"""FOV and view parameter calibration from screenshots.

This module provides tools to determine camera FOV and view orientation
by matching known star positions in screenshots to their projected positions.

Uses the "spherical moon" camera model where:
- The player walks on a tiny spherical moon
- "Up" = local gravity (radial from moon center)
- Camera has a fixed upward pitch (like a wedge under it) - shared across images
- Camera is always level (no roll)
- Only yaw (horizontal rotation) is controllable per-image
"""

import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize

from .coords import Star, parse_coordinates


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SphericalCalibration:
    """Calibration result using spherical moon camera model.

    This represents the camera orientation on a spherical moon surface:
    - up_theta, up_phi define the local "up" direction (gravity)
    - yaw is the horizontal rotation around that up vector
    - pitch is the fixed upward tilt from horizon (like a wedge under camera)
    - fov is the horizontal field of view
    """
    fov_horizontal: float  # Horizontal field of view in degrees
    up_theta: float        # Polar angle of up vector (radians, 0=+Y, π=-Y)
    up_phi: float          # Azimuthal angle of up vector (radians)
    yaw: float             # Horizontal rotation around up (radians)
    pitch: float = 0.0     # Upward tilt from horizon (radians, positive = looking up)
    residual_error: float = 0.0  # Mean pixel error after optimization

    def up_vector(self) -> np.ndarray:
        """Get the local up direction as a unit vector."""
        return spherical_to_cartesian(self.up_theta, self.up_phi)

    def forward_vector(self) -> np.ndarray:
        """Get the forward (look) direction as a unit vector."""
        forward, _, _ = build_camera_frame(self.up_theta, self.up_phi, self.yaw, self.pitch)
        return forward

    def __str__(self) -> str:
        up = self.up_vector()
        return (
            f"FOV: {self.fov_horizontal:.1f}°\n"
            f"Up: ({up[0]:.2f}, {up[1]:.2f}, {up[2]:.2f})\n"
            f"Yaw: {np.degrees(self.yaw):.1f}°\n"
            f"Pitch: {np.degrees(self.pitch):.1f}°\n"
            f"Residual error: {self.residual_error:.1f} pixels"
        )


@dataclass
class CalibrationResult:
    """Legacy calibration result (deprecated, use SphericalCalibration).

    Kept for backwards compatibility with old code.
    """
    fov_horizontal: float  # Horizontal field of view in degrees
    azimuth: float         # View azimuth in radians
    elevation: float       # View elevation in radians
    rotation: float        # View rotation in radians
    flip_horizontal: bool  # Whether view is flipped horizontally
    flip_vertical: bool    # Whether view is flipped vertically
    residual_error: float  # Mean pixel error after optimization

    def __str__(self) -> str:
        return (
            f"FOV: {self.fov_horizontal:.1f}°\n"
            f"Azimuth: {np.degrees(self.azimuth):.1f}°\n"
            f"Elevation: {np.degrees(self.elevation):.1f}°\n"
            f"Rotation: {np.degrees(self.rotation):.1f}°\n"
            f"Flip H/V: {self.flip_horizontal}/{self.flip_vertical}\n"
            f"Residual error: {self.residual_error:.1f} pixels"
        )


@dataclass
class StarMarker:
    """A star marked in a screenshot."""
    name: str
    pixel_x: float  # X position in image (0 = left)
    pixel_y: float  # Y position in image (0 = top)
    coords_3d: tuple[int, int, int]  # Known 3D coordinates


# =============================================================================
# Spherical Geometry Functions
# =============================================================================

def spherical_to_cartesian(theta: float, phi: float) -> np.ndarray:
    """Convert spherical coordinates to unit vector.

    Args:
        theta: Polar angle from +Y axis (0 = up, π = down)
        phi: Azimuthal angle in XZ plane from +X axis

    Returns:
        Unit vector [x, y, z]
    """
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.cos(theta),
        np.sin(theta) * np.sin(phi),
    ])


def build_camera_frame(
    up_theta: float,
    up_phi: float,
    yaw: float,
    pitch: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build camera coordinate frame from spherical parameters.

    The camera model:
    - "Up" is the local gravity direction (radial from moon center)
    - "Forward" starts in the horizontal plane, then pitches up
    - "Right" completes the orthonormal frame
    - Yaw rotates around the up axis, pitch tilts up from horizon

    Args:
        up_theta: Polar angle of up vector (0 = +Y, π = -Y)
        up_phi: Azimuthal angle of up vector
        yaw: Horizontal rotation angle
        pitch: Upward tilt from horizon (radians, positive = looking up)

    Returns:
        (forward, right, up) tuple of unit vectors
    """
    # Local "up" direction (gravity)
    up = spherical_to_cartesian(up_theta, up_phi)

    # Need a reference direction to define "forward" at yaw=0
    # Use world +X projected onto the horizontal plane, unless up is parallel to X
    world_ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(up, world_ref)) > 0.99:
        # Up is nearly parallel to X, use Z instead
        world_ref = np.array([0.0, 0.0, 1.0])

    # Project reference onto horizontal plane (perpendicular to up)
    forward_ref = world_ref - np.dot(world_ref, up) * up
    forward_ref = forward_ref / np.linalg.norm(forward_ref)

    # Right at yaw=0 - use cross(up, forward) for left-handed coordinate system
    # (game appears to use left-handed coords based on horizontal flip behavior)
    right_ref = np.cross(up, forward_ref)
    right_ref = right_ref / np.linalg.norm(right_ref)

    # Apply yaw rotation around up axis
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    forward_horiz = cos_y * forward_ref + sin_y * right_ref
    right = -sin_y * forward_ref + cos_y * right_ref

    # Apply pitch rotation around right axis (tilt up from horizon)
    # This rotates forward toward up and adjusts the camera's up accordingly
    cos_p = np.cos(pitch)
    sin_p = np.sin(pitch)
    forward = cos_p * forward_horiz + sin_p * up
    camera_up = -sin_p * forward_horiz + cos_p * up

    return forward, right, camera_up


def project_star(
    star_pos: np.ndarray,
    observer_pos: np.ndarray,
    calibration: SphericalCalibration,
    image_width: int,
    image_height: int,
) -> tuple[float, float, bool]:
    """Project a star to image coordinates using spherical camera model.

    Args:
        star_pos: 3D position of the star
        observer_pos: 3D position of the observer
        calibration: SphericalCalibration with camera parameters
        image_width, image_height: Image dimensions

    Returns:
        (pixel_x, pixel_y, visible) tuple
    """
    return project_star_raw(
        star_pos, observer_pos,
        calibration.up_theta, calibration.up_phi, calibration.yaw,
        calibration.pitch,
        np.radians(calibration.fov_horizontal),
        image_width, image_height,
    )


def project_star_raw(
    star_pos: np.ndarray,
    observer_pos: np.ndarray,
    up_theta: float,
    up_phi: float,
    yaw: float,
    pitch: float,
    fov_h: float,
    image_width: int,
    image_height: int,
    equirectangular: bool = False,
) -> tuple[float, float, bool]:
    """Project a star to image coordinates (raw parameters version).

    Args:
        star_pos: 3D position of the star
        observer_pos: 3D position of the observer
        up_theta, up_phi: Spherical coords of local "up" vector
        yaw: Horizontal rotation angle
        pitch: Upward tilt from horizon (radians, positive = looking up)
        fov_h: Horizontal field of view in radians
        image_width, image_height: Image dimensions
        equirectangular: If True, use equirectangular projection instead of rectilinear

    Returns:
        (pixel_x, pixel_y, visible) tuple
    """
    # Direction from observer to star
    direction = star_pos - observer_pos
    distance = np.linalg.norm(direction)
    if distance < 1e-6:
        return 0, 0, False
    direction = direction / distance

    # Build camera frame
    forward, right, up = build_camera_frame(up_theta, up_phi, yaw, pitch)

    # Components in camera frame
    forward_comp = np.dot(direction, forward)
    right_comp = np.dot(direction, right)
    up_comp = np.dot(direction, up)

    if equirectangular:
        # Equirectangular projection: angles map linearly to pixels
        # Good for wide FOV and skybox rendering
        if forward_comp <= -0.99:  # Behind camera
            return 0, 0, False

        # Horizontal angle from forward direction
        horiz_angle = np.arctan2(right_comp, forward_comp)

        # Vertical angle (elevation from horizontal plane through forward)
        horiz_dist = np.sqrt(forward_comp**2 + right_comp**2)
        vert_angle = np.arctan2(up_comp, horiz_dist)

        # Map to pixels
        fov_v = fov_h * image_height / image_width
        pixel_x = image_width / 2 + (horiz_angle / (fov_h / 2)) * (image_width / 2)
        pixel_y = image_height / 2 - (vert_angle / (fov_v / 2)) * (image_height / 2)

    else:
        # Rectilinear (pinhole) projection
        if forward_comp <= 0:
            return 0, 0, False

        screen_x = right_comp / forward_comp
        screen_y = up_comp / forward_comp

        # Convert to pixel coordinates
        scale = image_width / (2 * np.tan(fov_h / 2))
        pixel_x = image_width / 2 + screen_x * scale
        pixel_y = image_height / 2 - screen_y * scale  # Y flipped (0 at top)

    # Check if in frame
    visible = (0 <= pixel_x <= image_width and 0 <= pixel_y <= image_height)

    return pixel_x, pixel_y, visible


def compute_direction(
    pixel_x: float,
    pixel_y: float,
    calibration: SphericalCalibration,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Compute 3D direction from pixel position.

    Inverse of project_star.

    Args:
        pixel_x, pixel_y: Pixel position in image
        calibration: SphericalCalibration with camera parameters
        image_width, image_height: Image dimensions

    Returns:
        Unit direction vector in world coordinates
    """
    return compute_direction_raw(
        pixel_x, pixel_y,
        calibration.up_theta, calibration.up_phi, calibration.yaw,
        calibration.pitch,
        np.radians(calibration.fov_horizontal),
        image_width, image_height,
    )


def compute_direction_raw(
    pixel_x: float,
    pixel_y: float,
    up_theta: float,
    up_phi: float,
    yaw: float,
    pitch: float,
    fov_h: float,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Compute 3D direction from pixel position (raw parameters version).

    Args:
        pixel_x, pixel_y: Pixel position in image
        up_theta, up_phi: Spherical coords of local "up" vector
        yaw: Horizontal rotation angle
        pitch: Upward tilt from horizon (radians, positive = looking up)
        fov_h: Horizontal field of view in radians
        image_width, image_height: Image dimensions

    Returns:
        Unit direction vector in world coordinates
    """
    # Build camera frame
    forward, right, up = build_camera_frame(up_theta, up_phi, yaw, pitch)

    # Convert pixel to screen coordinates (rectilinear)
    scale = image_width / (2 * np.tan(fov_h / 2))
    screen_x = (pixel_x - image_width / 2) / scale
    screen_y = -(pixel_y - image_height / 2) / scale

    # Direction in world space
    direction = forward + screen_x * right + screen_y * up
    direction = direction / np.linalg.norm(direction)

    return direction


def project_star_to_image(
    star_pos: np.ndarray,
    observer_pos: np.ndarray,
    azimuth: float,
    elevation: float,
    rotation: float,
    fov_h: float,
    image_width: int,
    image_height: int,
    flip_h: bool = False,
    flip_v: bool = False,
) -> tuple[float, float, bool]:
    """Project a 3D star position to image pixel coordinates.

    Uses perspective projection matching typical game cameras.

    Args:
        star_pos: 3D position of the star
        observer_pos: 3D position of the observer
        azimuth: View direction azimuth (radians)
        elevation: View direction elevation (radians)
        rotation: View roll (radians)
        fov_h: Horizontal field of view (radians)
        image_width: Image width in pixels
        image_height: Image height in pixels
        flip_h: Flip horizontally
        flip_v: Flip vertically

    Returns:
        (pixel_x, pixel_y, visible) tuple
    """
    # Direction from observer to star
    direction = star_pos - observer_pos
    distance = np.linalg.norm(direction)
    if distance < 1e-6:
        return 0, 0, False
    direction = direction / distance

    # Build view coordinate frame
    # Forward vector (where camera is looking)
    forward = np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.sin(elevation),
        np.cos(elevation) * np.sin(azimuth)
    ])

    # Right vector (before roll)
    world_up = np.array([0, 1, 0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1, 0, 0])
    right = right / np.linalg.norm(right)

    # Up vector (before roll)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # Apply roll rotation
    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)
    right_rolled = cos_r * right + sin_r * up
    up_rolled = -sin_r * right + cos_r * up

    # Project star direction onto view plane
    # Component along forward (depth)
    depth = np.dot(direction, forward)

    # Star is behind camera
    if depth <= 0:
        return 0, 0, False

    # Components along right and up (screen space)
    screen_x = np.dot(direction, right_rolled) / depth
    screen_y = np.dot(direction, up_rolled) / depth

    # Apply flips
    if flip_h:
        screen_x = -screen_x
    if flip_v:
        screen_y = -screen_y

    # Convert to pixel coordinates
    # FOV determines the scale
    scale = image_width / (2 * np.tan(fov_h / 2))

    pixel_x = image_width / 2 + screen_x * scale
    pixel_y = image_height / 2 - screen_y * scale  # Y is flipped (0 at top)

    # Check if in frame
    visible = (0 <= pixel_x <= image_width and 0 <= pixel_y <= image_height)

    return pixel_x, pixel_y, visible


def calibration_error(
    params: np.ndarray,
    markers: list[StarMarker],
    observer_pos: np.ndarray,
    image_width: int,
    image_height: int,
    flip_h: bool,
    flip_v: bool,
) -> float:
    """Compute total squared error between projected and marked positions."""
    fov_h, azimuth, elevation, rotation = params

    total_error = 0.0
    for marker in markers:
        star_pos = np.array(marker.coords_3d, dtype=float)
        px, py, visible = project_star_to_image(
            star_pos, observer_pos,
            azimuth, elevation, rotation, fov_h,
            image_width, image_height,
            flip_h, flip_v
        )

        if not visible:
            # Heavy penalty for stars that should be visible but aren't
            total_error += 10000
        else:
            dx = px - marker.pixel_x
            dy = py - marker.pixel_y
            total_error += dx*dx + dy*dy

    return total_error


def calibrate_from_markers(
    markers: list[StarMarker],
    observer_pos: tuple[int, int, int],
    image_width: int,
    image_height: int,
    initial_fov: float = np.radians(90),
) -> CalibrationResult:
    """Calibrate FOV and view parameters from marked star positions.

    Args:
        markers: List of StarMarker objects (at least 2, preferably 3+)
        observer_pos: Observer's 3D coordinates
        image_width: Screenshot width in pixels
        image_height: Screenshot height in pixels
        initial_fov: Initial FOV guess in radians

    Returns:
        CalibrationResult with optimized parameters
    """
    if len(markers) < 2:
        raise ValueError("Need at least 2 marked stars for calibration")

    obs_pos = np.array(observer_pos, dtype=float)

    # Compute directions to all stars for initial estimate
    star_directions = []
    for m in markers:
        vec = np.array(m.coords_3d) - obs_pos
        vec = vec / np.linalg.norm(vec)
        star_directions.append(vec)

    # Estimate initial view direction: average of star directions
    avg_dir = np.mean(star_directions, axis=0)
    avg_dir = avg_dir / np.linalg.norm(avg_dir)
    init_azimuth = np.arctan2(avg_dir[2], avg_dir[0])
    init_elevation = np.arcsin(np.clip(avg_dir[1], -1, 1))

    # Try all flip combinations and multiple rotation starts
    best_result = None
    best_error = float('inf')

    for flip_h in [False, True]:
        for flip_v in [False, True]:
            for init_rotation in [0, np.pi/2, np.pi, -np.pi/2]:
                # Initial parameters: [fov, azimuth, elevation, rotation]
                x0 = np.array([initial_fov, init_azimuth, init_elevation, init_rotation])

                # Bounds - allow full rotation range
                bounds = [
                    (np.radians(60), np.radians(120)),  # FOV: 60-120 degrees
                    (init_azimuth - np.pi/2, init_azimuth + np.pi/2),  # Azimuth near initial
                    (-np.pi/2 + 0.1, np.pi/2 - 0.1),     # Elevation
                    (-np.pi, np.pi),                      # Rotation
                ]

                # Optimize
                try:
                    result = minimize(
                        calibration_error,
                        x0,
                        args=(markers, obs_pos, image_width, image_height, flip_h, flip_v),
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 500}
                    )

                    if result.fun < best_error:
                        best_error = result.fun
                        best_result = CalibrationResult(
                            fov_horizontal=np.degrees(result.x[0]),
                            azimuth=result.x[1],
                            elevation=result.x[2],
                            rotation=result.x[3],
                            flip_horizontal=flip_h,
                            flip_vertical=flip_v,
                            residual_error=np.sqrt(best_error / len(markers)),
                        )
                except Exception:
                    continue

    return best_result


# =============================================================================
# Spherical Calibration
# =============================================================================

def _spherical_calibration_error(
    params: np.ndarray,
    markers: list[StarMarker],
    observer_pos: np.ndarray,
    image_width: int,
    image_height: int,
) -> float:
    """Compute total squared error for spherical calibration.

    Note: Single-image calibration assumes pitch=0. For proper pitch
    estimation, use the multi-image stitching solver.
    """
    fov_h, up_theta, up_phi, yaw = params
    pitch = 0.0  # Single-image can't estimate pitch; assume level

    total_error = 0.0
    for marker in markers:
        star_pos = np.array(marker.coords_3d, dtype=float)
        px, py, visible = project_star_raw(
            star_pos, observer_pos,
            up_theta, up_phi, yaw, pitch, fov_h,
            image_width, image_height,
        )

        if not visible:
            # Compute angular penalty to give gradient signal
            direction = star_pos - observer_pos
            direction = direction / np.linalg.norm(direction)
            forward, _, _ = build_camera_frame(up_theta, up_phi, yaw, pitch)
            dot = np.dot(direction, forward)
            total_error += 10000 * (1 - dot)
        else:
            dx = px - marker.pixel_x
            dy = py - marker.pixel_y
            total_error += dx * dx + dy * dy

    return total_error


def calibrate_spherical(
    markers: list[StarMarker],
    observer_pos: tuple[int, int, int],
    image_width: int,
    image_height: int,
    initial_fov: float = np.radians(90),
) -> SphericalCalibration:
    """Calibrate using spherical moon camera model.

    This is the primary calibration function. It finds the camera orientation
    (up direction + yaw) that best matches marked star positions.

    Args:
        markers: List of StarMarker objects (at least 2, preferably 3+)
        observer_pos: Observer's 3D coordinates
        image_width: Screenshot width in pixels
        image_height: Screenshot height in pixels
        initial_fov: Initial FOV guess in radians

    Returns:
        SphericalCalibration with optimized parameters
    """
    if len(markers) < 2:
        raise ValueError("Need at least 2 marked stars for calibration")

    obs_pos = np.array(observer_pos, dtype=float)

    # Compute average direction to stars for initial yaw guess
    star_directions = []
    for m in markers:
        vec = np.array(m.coords_3d) - obs_pos
        vec = vec / np.linalg.norm(vec)
        star_directions.append(vec)

    avg_dir = np.mean(star_directions, axis=0)
    avg_dir = avg_dir / np.linalg.norm(avg_dir)
    init_yaw = np.arctan2(avg_dir[2], avg_dir[0])

    # Try multiple initial up vector orientations
    init_up_configs = [
        (np.pi / 2, 0.0),       # Up toward +X
        (np.pi / 2, np.pi/2),   # Up toward +Z
        (np.pi / 2, np.pi),     # Up toward -X
        (np.pi / 2, -np.pi/2),  # Up toward -Z
        (0.1, 0.0),             # Up toward +Y
        (np.pi - 0.1, 0.0),     # Up toward -Y
    ]

    best_result = None
    best_error = float('inf')

    for init_up_theta, init_up_phi in init_up_configs:
        # Initial parameters: [fov, up_theta, up_phi, yaw]
        x0 = np.array([initial_fov, init_up_theta, init_up_phi, init_yaw])

        bounds = [
            (np.radians(70), np.radians(110)),  # FOV
            (0.01, np.pi - 0.01),                # up_theta (avoid poles)
            (-np.pi, np.pi),                     # up_phi
            (-np.pi, np.pi),                     # yaw
        ]

        try:
            result = minimize(
                _spherical_calibration_error,
                x0,
                args=(markers, obs_pos, image_width, image_height),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )

            if result.fun < best_error:
                best_error = result.fun
                best_result = result

        except Exception:
            continue

    if best_result is None:
        raise ValueError("Calibration failed to converge")

    return SphericalCalibration(
        fov_horizontal=np.degrees(best_result.x[0]),
        up_theta=best_result.x[1],
        up_phi=best_result.x[2],
        yaw=best_result.x[3],
        residual_error=np.sqrt(best_error / len(markers)),
    )


# =============================================================================
# Legacy Functions (for backwards compatibility)
# =============================================================================

def compute_direction_from_pixel(
    pixel_x: float,
    pixel_y: float,
    calibration: CalibrationResult,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Convert a pixel position to a 3D direction vector (legacy version).

    This is the inverse of projection - given where a star appears
    in the image, compute the direction it lies in.

    Note: For new code, use compute_direction() with SphericalCalibration.

    Args:
        pixel_x: X position in image
        pixel_y: Y position in image
        calibration: Calibrated view parameters
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        Unit direction vector in world coordinates
    """
    fov_h = np.radians(calibration.fov_horizontal)

    # Convert pixel to normalized screen coordinates
    scale = image_width / (2 * np.tan(fov_h / 2))
    screen_x = (pixel_x - image_width / 2) / scale
    screen_y = -(pixel_y - image_height / 2) / scale  # Flip Y back

    # Undo flips
    if calibration.flip_horizontal:
        screen_x = -screen_x
    if calibration.flip_vertical:
        screen_y = -screen_y

    # Build view coordinate frame
    azimuth = calibration.azimuth
    elevation = calibration.elevation
    rotation = calibration.rotation

    forward = np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.sin(elevation),
        np.cos(elevation) * np.sin(azimuth)
    ])

    world_up = np.array([0, 1, 0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1, 0, 0])
    right = right / np.linalg.norm(right)

    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # Apply roll
    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)
    right_rolled = cos_r * right + sin_r * up
    up_rolled = -sin_r * right + cos_r * up

    # Direction in world space
    direction = forward + screen_x * right_rolled + screen_y * up_rolled
    direction = direction / np.linalg.norm(direction)

    return direction


def estimate_star_position(
    direction: np.ndarray,
    observer_pos: tuple[int, int, int],
    name_bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None = None,
) -> tuple[tuple[int, int, int], float]:
    """Estimate a star's 3D position from its direction ray.

    Intersects the direction ray with the cluster bounds (0-999)
    and optionally with name-derived bounds.

    Args:
        direction: Unit direction vector from observer
        observer_pos: Observer's 3D position
        name_bounds: Optional ((x_min, x_max), (y_min, y_max), (z_min, z_max))
                    from decoding an original name

    Returns:
        (estimated_position, uncertainty_radius) tuple
    """
    obs = np.array(observer_pos, dtype=float)

    # Default bounds: full cluster
    bounds = name_bounds if name_bounds else ((0, 999), (0, 999), (0, 999))

    # Find where ray intersects the bounding box
    # Ray: P = obs + t * direction, t > 0

    t_min = 0.0
    t_max = float('inf')

    for i in range(3):
        if abs(direction[i]) < 1e-10:
            # Ray parallel to this axis
            if obs[i] < bounds[i][0] or obs[i] > bounds[i][1]:
                # Observer outside bounds on this axis, no intersection
                t_max = -1
                break
        else:
            t1 = (bounds[i][0] - obs[i]) / direction[i]
            t2 = (bounds[i][1] - obs[i]) / direction[i]

            if t1 > t2:
                t1, t2 = t2, t1

            t_min = max(t_min, t1)
            t_max = min(t_max, t2)

    if t_max < t_min or t_max < 0:
        # No valid intersection
        # Return a point along the ray at some reasonable distance
        t = 500.0
        point = obs + t * direction
        return (int(point[0]), int(point[1]), int(point[2])), 500.0

    # Use midpoint of valid range as estimate
    t_mid = (max(0, t_min) + t_max) / 2
    point = obs + t_mid * direction

    # Uncertainty is half the valid range length
    uncertainty = (t_max - max(0, t_min)) / 2

    # Clamp to valid coordinates
    estimated = (
        int(np.clip(point[0], 0, 999)),
        int(np.clip(point[1], 0, 999)),
        int(np.clip(point[2], 0, 999)),
    )

    return estimated, uncertainty
