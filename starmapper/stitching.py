"""Multi-image stitching for coordinate prediction.

This module links multiple observatory screenshots using constellation
correspondences to estimate positions of unknown stars when known-position
reference stars are split across different images.

Uses the "spherical moon" camera model where:
- The player walks on a tiny spherical moon at ~10°/sec
- "Up" = local gravity = radial direction from moon center
- Camera always looks at the horizon (no pitch)
- Camera is always level (no roll)
- Only yaw (horizontal rotation) is controllable
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable
from scipy.optimize import minimize
from pathlib import Path

from .calibration import (
    SphericalCalibration, StarMarker,
    spherical_to_cartesian, build_camera_frame,
    project_star_raw, compute_direction_raw,
)


# =============================================================================
# Note: Core spherical geometry functions (spherical_to_cartesian, build_camera_frame,
# project_star_raw, compute_direction_raw) are imported from calibration.py
# =============================================================================


@dataclass
class CorrespondenceMarker:
    """A star marked for cross-image correspondence matching."""
    label: str           # User-assigned label (e.g., "VOID6", "A", "B")
    pixel_x: float
    pixel_y: float


@dataclass
class ImageMarking:
    """All markings for a single screenshot."""
    image_path: Path
    image_width: int
    image_height: int
    known_markers: list[StarMarker] = field(default_factory=list)
    correspondence_markers: list[CorrespondenceMarker] = field(default_factory=list)

    def get_correspondence(self, label: str) -> CorrespondenceMarker | None:
        """Get correspondence marker by label."""
        for m in self.correspondence_markers:
            if m.label == label:
                return m
        return None


@dataclass
class StitchingSession:
    """State for a multi-image stitching session."""
    observer_pos: tuple[int, int, int]
    known_stars: dict[str, tuple[int, int, int]]  # name -> 3D coords
    images: list[ImageMarking] = field(default_factory=list)

    def add_image(self, path: Path, width: int, height: int) -> int:
        """Add an image and return its index."""
        self.images.append(ImageMarking(
            image_path=path,
            image_width=width,
            image_height=height,
        ))
        return len(self.images) - 1

    def add_known_marker(self, image_idx: int, name: str, px: float, py: float) -> bool:
        """Add a known-position star marker to an image."""
        if name not in self.known_stars:
            return False
        if image_idx >= len(self.images):
            return False

        # Remove existing marker for this star
        img = self.images[image_idx]
        img.known_markers = [m for m in img.known_markers if m.name != name]

        img.known_markers.append(StarMarker(
            name=name,
            pixel_x=px,
            pixel_y=py,
            coords_3d=self.known_stars[name],
        ))
        return True

    def add_correspondence(self, image_idx: int, label: str, px: float, py: float):
        """Add a correspondence marker to an image."""
        if image_idx >= len(self.images):
            return

        img = self.images[image_idx]
        # Remove existing marker with this label
        img.correspondence_markers = [
            m for m in img.correspondence_markers if m.label != label
        ]

        img.correspondence_markers.append(CorrespondenceMarker(
            label=label,
            pixel_x=px,
            pixel_y=py,
        ))

    def get_all_correspondence_labels(self) -> set[str]:
        """Get all unique correspondence labels across all images."""
        labels = set()
        for img in self.images:
            for m in img.correspondence_markers:
                labels.add(m.label)
        return labels

    def get_cross_image_correspondences(self) -> dict[str, list[tuple[int, CorrespondenceMarker]]]:
        """Get correspondences that appear in multiple images.

        Returns dict mapping label -> [(image_idx, marker), ...]
        Only includes labels that appear in 2+ images.
        """
        by_label: dict[str, list[tuple[int, CorrespondenceMarker]]] = {}
        for idx, img in enumerate(self.images):
            for m in img.correspondence_markers:
                if m.label not in by_label:
                    by_label[m.label] = []
                by_label[m.label].append((idx, m))

        # Filter to only cross-image correspondences
        return {k: v for k, v in by_label.items() if len(v) >= 2}


@dataclass
class StitchingSolution:
    """A solution from the spherical moon camera model."""
    calibrations: list[SphericalCalibration]  # One per image
    total_error: float
    known_star_error: float      # Error from known-position stars
    correspondence_error: float  # Error from cross-image correspondences


@dataclass
class StitchingResult:
    """Results of multi-image stitching."""
    session: StitchingSession
    solutions: list[StitchingSolution]  # Sorted by total_error

    def best_solution(self) -> StitchingSolution | None:
        """Get the lowest-error solution."""
        return self.solutions[0] if self.solutions else None

    def get_direction_rays(
        self,
        label: str,
        max_solutions: int = 4,
    ) -> list[tuple[np.ndarray, StitchingSolution]]:
        """Get direction rays for a correspondence label.

        Returns list of (direction_vector, solution) tuples,
        one per distinct solution (up to max_solutions).
        """
        rays = []
        for sol in self.solutions[:max_solutions]:
            # Find which image has this label
            for idx, img in enumerate(self.session.images):
                marker = img.get_correspondence(label)
                if marker:
                    # Compute direction using spherical model
                    cal = sol.calibrations[idx]
                    fov_h = np.radians(cal.fov_horizontal)
                    direction = compute_direction_raw(
                        marker.pixel_x, marker.pixel_y,
                        cal.up_theta, cal.up_phi, cal.yaw, cal.pitch, fov_h,
                        img.image_width, img.image_height,
                    )
                    rays.append((direction, sol))
                    break  # Only need one instance per solution
        return rays


def _project_star_with_depth(
    star_pos: np.ndarray,
    observer_pos: np.ndarray,
    up_theta: float,
    up_phi: float,
    yaw: float,
    pitch: float,
    fov_h: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float]:
    """Project star and return (pixel_x, pixel_y, depth).

    Unlike project_star_raw, this always returns coordinates
    even if outside image bounds, and returns depth for penalty calculation.
    Uses rectilinear (pinhole) projection.
    """
    direction = star_pos - observer_pos
    distance = np.linalg.norm(direction)
    if distance < 1e-6:
        return 0, 0, -1  # Invalid
    direction = direction / distance

    forward, right, up = build_camera_frame(up_theta, up_phi, yaw, pitch)

    depth = np.dot(direction, forward)
    if depth <= 0.01:  # Behind or at camera
        return 0, 0, depth

    # Rectilinear projection
    screen_x = np.dot(direction, right) / depth
    screen_y = np.dot(direction, up) / depth

    scale = image_width / (2 * np.tan(fov_h / 2))
    pixel_x = image_width / 2 + screen_x * scale
    pixel_y = image_height / 2 - screen_y * scale

    return pixel_x, pixel_y, depth


def _stitching_error_spherical(
    params: np.ndarray,
    session: StitchingSession,
    cross_correspondences: dict[str, list[tuple[int, CorrespondenceMarker]]],
    correspondence_weight: float = 1.0,
) -> float:
    """Compute total error for multi-image stitching using spherical model.

    params layout: [fov, pitch, up_theta_0, up_phi_0, yaw_0, up_theta_1, ...]
    - fov: shared FOV in radians
    - pitch: shared upward tilt from horizon (radians)
    - up_theta_i, up_phi_i: spherical coords of local "up" vector for image i
    - yaw_i: horizontal rotation for image i
    """
    n_images = len(session.images)
    obs_pos = np.array(session.observer_pos, dtype=float)

    total_known_error = 0.0
    total_corr_error = 0.0
    total_orientation_error = 0.0

    fov_h = params[0]
    pitch = params[1]

    # Per-image known star errors
    for i, img in enumerate(session.images):
        base = 2 + i * 3  # Skip FOV and pitch, then 3 params per image
        up_theta = params[base]
        up_phi = params[base + 1]
        yaw = params[base + 2]

        # Build camera frame once for this image
        forward, right, camera_up = build_camera_frame(up_theta, up_phi, yaw, pitch)

        for marker in img.known_markers:
            star_pos = np.array(marker.coords_3d, dtype=float)
            direction = star_pos - obs_pos
            direction = direction / np.linalg.norm(direction)

            px, py, depth = _project_star_with_depth(
                star_pos, obs_pos,
                up_theta, up_phi, yaw, pitch, fov_h,
                img.image_width, img.image_height,
            )

            if depth <= 0.01:
                # Star is behind camera - very large penalty plus angular error
                # to give gradient signal toward correct orientation
                dot = np.dot(direction, forward)
                # Strong penalty that increases as star goes more behind
                total_known_error += 100000 * (1 - dot)
            else:
                # Star is in front - compute pixel error (even if outside bounds)
                dx = px - marker.pixel_x
                dy = py - marker.pixel_y
                total_known_error += dx * dx + dy * dy

            # ALWAYS check orientation (whether star is in front or behind)
            # Stars in upper half of image should have positive up_comp
            up_comp = np.dot(direction, camera_up)
            in_upper_half = marker.pixel_y < img.image_height / 2

            # If orientation is wrong, add VERY LARGE penalty
            # Must be larger than any possible pixel error to force correct orientation
            if in_upper_half and up_comp < 0:
                # Star should be above horizon but isn't - HUGE penalty
                total_orientation_error += 1000000 * (1.0 + abs(up_comp))
            elif not in_upper_half and up_comp > 0:
                # Star should be below horizon but isn't - HUGE penalty
                total_orientation_error += 1000000 * (1.0 + abs(up_comp))

    # Cross-image correspondence errors
    for label, occurrences in cross_correspondences.items():
        if len(occurrences) < 2:
            continue

        # Compute direction for each occurrence
        directions = []
        for img_idx, marker in occurrences:
            img = session.images[img_idx]
            base = 2 + img_idx * 3
            up_theta = params[base]
            up_phi = params[base + 1]
            yaw = params[base + 2]

            direction = compute_direction_raw(
                marker.pixel_x, marker.pixel_y,
                up_theta, up_phi, yaw, pitch, fov_h,
                img.image_width, img.image_height,
            )
            directions.append(direction)

        # Penalize variance in directions
        mean_dir = np.mean(directions, axis=0)
        mean_dir = mean_dir / np.linalg.norm(mean_dir)

        for d in directions:
            dot = np.clip(np.dot(d, mean_dir), -1, 1)
            angle = np.arccos(dot)
            # Convert to "pixel-equivalent" error (~10 pixels per degree)
            pixel_equiv = np.degrees(angle) * 10
            total_corr_error += pixel_equiv * pixel_equiv

    return total_known_error + correspondence_weight * total_corr_error + total_orientation_error


def solve_stitching(
    session: StitchingSession,
    initial_fov: float = 90.0,
    initial_pitch: float = 25.0,
    fixed_fov: float | None = None,  # If set, FOV won't be optimized
    fixed_pitch: float | None = None,  # If set, pitch won't be optimized
    correspondence_weight: float = 0.1,  # Reduced: prioritize known stars
    progress_callback: Callable[[str], None] | None = None,
) -> StitchingResult:
    """Solve multi-image stitching problem using spherical moon model.

    The spherical model assumes:
    - Player walks on a tiny moon surface
    - "Up" = local gravity (radial from moon center)
    - Camera has fixed upward pitch (like a wedge under it)
    - Camera always level (no roll freedom)
    - Only yaw (horizontal rotation) is controllable

    Each image has 3 parameters: up_theta, up_phi (local up direction), yaw.
    All images share FOV and pitch parameters.

    Args:
        session: StitchingSession with marked images
        initial_fov: Initial FOV guess in degrees (default 90°)
        initial_pitch: Initial upward pitch guess in degrees (default 25°)
        correspondence_weight: Weight for correspondence errors vs known star errors
        progress_callback: Optional callback for progress updates

    Returns:
        StitchingResult with solutions sorted by error
    """
    if not session.images:
        return StitchingResult(session=session, solutions=[])

    n_images = len(session.images)
    obs_pos = np.array(session.observer_pos, dtype=float)
    cross_correspondences = session.get_cross_image_correspondences()

    def log(msg: str):
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    log(f"Solving stitching for {n_images} images using spherical moon model...")

    # Debug: show what directions known stars are in
    for i, img in enumerate(session.images):
        if img.known_markers:
            log(f"  Image {i} known stars:")
            for m in img.known_markers:
                vec = np.array(m.coords_3d) - obs_pos
                vec = vec / np.linalg.norm(vec)
                log(f"    {m.name}: dir=({vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f}), "
                    f"pixel=({m.pixel_x:.0f}, {m.pixel_y:.0f})")

    # Compute initial camera direction from known stars
    # We need to find up vectors that put the stars IN FRONT of the camera
    per_image_avg_dir = []
    for img in session.images:
        if img.known_markers:
            # Compute average direction to known stars
            directions = []
            for m in img.known_markers:
                vec = np.array(m.coords_3d) - obs_pos
                vec = vec / np.linalg.norm(vec)
                directions.append(vec)
            avg_dir = np.mean(directions, axis=0)
            avg_dir = avg_dir / np.linalg.norm(avg_dir)
            per_image_avg_dir.append(avg_dir)
        else:
            per_image_avg_dir.append(np.array([0.0, 0.0, 1.0]))

    # Global average direction (where stars generally are)
    global_avg_dir = np.mean(per_image_avg_dir, axis=0)
    global_avg_dir = global_avg_dir / np.linalg.norm(global_avg_dir)
    log(f"Average star direction: ({global_avg_dir[0]:.3f}, {global_avg_dir[1]:.3f}, {global_avg_dir[2]:.3f})")

    # Generate initial up vectors that are PERPENDICULAR to the viewing direction
    # This ensures the stars will be in front of the camera
    def generate_perpendicular_ups(view_dir):
        """Generate up vectors perpendicular to a viewing direction."""
        ups = []
        # Try world Y as up (if not parallel to view_dir)
        world_y = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(view_dir, world_y)) < 0.9:
            up = world_y - np.dot(world_y, view_dir) * view_dir
            up = up / np.linalg.norm(up)
            ups.append(up)
            ups.append(-up)
        # Try world X as reference
        world_x = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(view_dir, world_x)) < 0.9:
            up = world_x - np.dot(world_x, view_dir) * view_dir
            up = up / np.linalg.norm(up)
            ups.append(up)
            ups.append(-up)
        # Try world Z as reference
        world_z = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(view_dir, world_z)) < 0.9:
            up = world_z - np.dot(world_z, view_dir) * view_dir
            up = up / np.linalg.norm(up)
            ups.append(up)
            ups.append(-up)
        return ups

    candidate_ups = generate_perpendicular_ups(global_avg_dir)
    log(f"Generated {len(candidate_ups)} candidate up vectors perpendicular to view")

    # Convert up vectors to spherical coordinates
    def up_to_spherical(up):
        """Convert up unit vector to (theta, phi) spherical coords."""
        # up = (sin(theta)*cos(phi), cos(theta), sin(theta)*sin(phi))
        theta = np.arccos(np.clip(up[1], -1, 1))
        phi = np.arctan2(up[2], up[0])
        return theta, phi

    init_up_configs = []
    for up in candidate_ups:
        theta, phi = up_to_spherical(up)
        init_up_configs.append((theta, phi))

    # Also add the original configurations as fallback
    init_up_configs.extend([
        (np.pi / 2, 0.0),      # Up pointing toward +X
        (np.pi / 2, np.pi/2),  # Up pointing toward +Z
        (np.pi / 2, np.pi),    # Up pointing toward -X
        (np.pi / 2, -np.pi/2), # Up pointing toward -Z
        (0.1, 0.0),            # Up pointing toward +Y (nearly)
        (np.pi - 0.1, 0.0),    # Up pointing toward -Y (nearly)
    ])

    log(f"Trying {len(init_up_configs)} initial configurations")

    best_result = None
    best_error = float('inf')

    for init_up_theta, init_up_phi in init_up_configs:
        # Build initial parameters: [fov, pitch, up_theta_0, up_phi_0, yaw_0, ...]
        # Use fixed values if provided, otherwise use initial guesses with bounds
        fov_val = np.radians(fixed_fov if fixed_fov is not None else initial_fov)
        pitch_val = np.radians(fixed_pitch if fixed_pitch is not None else initial_pitch)
        x0 = [fov_val, pitch_val]

        # If fixed, use very tight bounds (effectively constant)
        if fixed_fov is not None:
            fov_bounds = (fov_val - 0.001, fov_val + 0.001)
        else:
            fov_bounds = (np.radians(70), np.radians(120))

        if fixed_pitch is not None:
            pitch_bounds = (pitch_val - 0.001, pitch_val + 0.001)
        else:
            pitch_bounds = (np.radians(10), np.radians(45))

        bounds = [fov_bounds, pitch_bounds]

        # For each image, compute a per-image up vector that's perpendicular
        # to that image's star directions (so stars will be in front)
        for i in range(n_images):
            avg_dir = per_image_avg_dir[i]

            # Use the template up vector, but if this image has stars in
            # a very different direction, adjust the up to be perpendicular
            init_up = spherical_to_cartesian(init_up_theta, init_up_phi)

            # Check if the template up is roughly perpendicular to star direction
            dot_with_stars = abs(np.dot(init_up, avg_dir))
            if dot_with_stars > 0.7:
                # Template up is too aligned with stars - generate a perpendicular one
                # using the image's star direction
                world_y = np.array([0.0, 1.0, 0.0])
                if abs(np.dot(avg_dir, world_y)) < 0.9:
                    init_up = world_y - np.dot(world_y, avg_dir) * avg_dir
                else:
                    world_x = np.array([1.0, 0.0, 0.0])
                    init_up = world_x - np.dot(world_x, avg_dir) * avg_dir
                init_up = init_up / np.linalg.norm(init_up)
                img_up_theta, img_up_phi = up_to_spherical(init_up)
            else:
                img_up_theta = init_up_theta
                img_up_phi = init_up_phi

            # Project average star direction onto the horizontal plane (perpendicular to up)
            horiz_component = avg_dir - np.dot(avg_dir, init_up) * init_up
            horiz_norm = np.linalg.norm(horiz_component)

            if horiz_norm > 0.1:
                horiz_component = horiz_component / horiz_norm

                # Compute reference forward at yaw=0
                world_ref = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(init_up, world_ref)) > 0.99:
                    world_ref = np.array([0.0, 0.0, 1.0])
                forward_ref = world_ref - np.dot(world_ref, init_up) * init_up
                forward_ref = forward_ref / np.linalg.norm(forward_ref)
                right_ref = np.cross(forward_ref, init_up)
                right_ref = right_ref / np.linalg.norm(right_ref)

                # Yaw angle to point toward the horizontal component of stars
                yaw = np.arctan2(np.dot(horiz_component, right_ref),
                                np.dot(horiz_component, forward_ref))
            else:
                yaw = 0.0

            # Check if up vector orientation is correct for this image
            # by verifying that stars in upper half of image have positive up_comp
            img = session.images[i]
            if img.known_markers:
                # Build camera frame with current params to check orientation
                forward_test, right_test, camera_up_test = build_camera_frame(
                    img_up_theta, img_up_phi, yaw, pitch_val
                )

                # Count orientation agreements
                correct_orientation = 0
                wrong_orientation = 0
                for marker in img.known_markers:
                    star_dir = np.array(marker.coords_3d) - obs_pos
                    star_dir = star_dir / np.linalg.norm(star_dir)
                    up_comp = np.dot(star_dir, camera_up_test)

                    # Expected: upper half (pixel_y < center) -> up_comp > 0
                    in_upper_half = marker.pixel_y < img.image_height / 2
                    up_comp_positive = up_comp > 0

                    if in_upper_half == up_comp_positive:
                        correct_orientation += 1
                    else:
                        wrong_orientation += 1

                # If mostly wrong, flip the up vector
                if wrong_orientation > correct_orientation:
                    init_up = -init_up
                    img_up_theta, img_up_phi = up_to_spherical(init_up)
                    # Yaw stays the same direction but we need to recalculate
                    # since up flipped, forward direction should flip too
                    yaw = yaw + np.pi  # Rotate 180° to maintain forward direction
                    if yaw > np.pi:
                        yaw -= 2 * np.pi

            x0.extend([img_up_theta, img_up_phi, yaw])
            bounds.extend([
                (0.01, np.pi - 0.01),  # up_theta: avoid poles
                (-np.pi, np.pi),       # up_phi: full circle
                (-np.pi, np.pi),       # yaw: full circle
            ])

        x0 = np.array(x0)

        try:
            result = minimize(
                _stitching_error_spherical,
                x0,
                args=(session, cross_correspondences, correspondence_weight),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000},
            )

            if result.fun < best_error:
                best_error = result.fun
                best_result = result

        except Exception as e:
            log(f"Optimization failed for init ({init_up_theta:.2f}, {init_up_phi:.2f}): {e}")
            continue

    if best_result is None:
        log("No valid solution found")
        return StitchingResult(session=session, solutions=[])

    # Build solution from best result
    calibrations = []
    fov_deg = np.degrees(best_result.x[0])
    pitch_rad = best_result.x[1]
    pitch_deg = np.degrees(pitch_rad)

    for i in range(n_images):
        base = 2 + i * 3  # Skip FOV and pitch
        up_theta = best_result.x[base]
        up_phi = best_result.x[base + 1]
        yaw = best_result.x[base + 2]

        calibrations.append(SphericalCalibration(
            fov_horizontal=fov_deg,
            up_theta=up_theta,
            up_phi=up_phi,
            yaw=yaw,
            pitch=pitch_rad,
            residual_error=0,
        ))

    # Compute separate error components
    known_error = _stitching_error_spherical(
        best_result.x, session, {}, 0
    )
    corr_error = best_result.fun - known_error

    solution = StitchingSolution(
        calibrations=calibrations,
        total_error=best_result.fun,
        known_star_error=known_error,
        correspondence_error=corr_error,
    )

    log(f"\nBest solution: error={best_result.fun:.1f} "
        f"(known={known_error:.1f}, corr={corr_error:.1f})")
    log(f"  Shared FOV: {fov_deg:.1f}°, Pitch: {pitch_deg:.1f}°")
    for i, cal in enumerate(calibrations):
        up = spherical_to_cartesian(cal.up_theta, cal.up_phi)
        forward, right, _ = build_camera_frame(cal.up_theta, cal.up_phi, cal.yaw, cal.pitch)
        log(f"  Image {i}: up=({up[0]:.2f}, {up[1]:.2f}, {up[2]:.2f}), "
            f"yaw={np.degrees(cal.yaw):.1f}°")
        log(f"    forward=({forward[0]:.2f}, {forward[1]:.2f}, {forward[2]:.2f})")

    # Debug: show where known stars project to with this solution
    log("\nProjection check for known stars:")
    fov_h = np.radians(fov_deg)
    for i, img in enumerate(session.images):
        cal = calibrations[i]
        for m in img.known_markers:
            star_pos = np.array(m.coords_3d, dtype=float)
            px, py, visible = project_star_raw(
                star_pos, obs_pos,
                cal.up_theta, cal.up_phi, cal.yaw, cal.pitch, fov_h,
                img.image_width, img.image_height,
            )
            dx = px - m.pixel_x if visible else float('inf')
            dy = py - m.pixel_y if visible else float('inf')
            err = np.sqrt(dx*dx + dy*dy) if visible else float('inf')
            log(f"  Image {i} {m.name}: marked=({m.pixel_x:.0f},{m.pixel_y:.0f}), "
                f"projected=({px:.0f},{py:.0f}), visible={visible}, err={err:.1f}")

    # Debug: show correspondence direction agreement
    log("\nCorrespondence direction check:")
    for label, occurrences in cross_correspondences.items():
        if len(occurrences) < 2:
            continue
        directions = []
        for img_idx, marker in occurrences:
            img = session.images[img_idx]
            cal = calibrations[img_idx]
            direction = compute_direction_raw(
                marker.pixel_x, marker.pixel_y,
                cal.up_theta, cal.up_phi, cal.yaw, cal.pitch, fov_h,
                img.image_width, img.image_height,
            )
            directions.append((img_idx, direction))

        # Show angular error between all pairs
        for j in range(len(directions)):
            for k in range(j + 1, len(directions)):
                idx_j, dir_j = directions[j]
                idx_k, dir_k = directions[k]
                dot = np.clip(np.dot(dir_j, dir_k), -1, 1)
                angle_deg = np.degrees(np.arccos(dot))
                log(f"  {label}: img{idx_j}↔img{idx_k} = {angle_deg:.1f}°")

    return StitchingResult(session=session, solutions=[solution])


def estimate_star_from_stitching(
    result: StitchingResult,
    label: str,
    name_bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None = None,
) -> list[tuple[tuple[int, int, int], float, np.ndarray, StitchingSolution]]:
    """Estimate star position from stitching result.

    Args:
        result: StitchingResult from solve_stitching
        label: Correspondence label to estimate
        name_bounds: Optional coordinate bounds from original name

    Returns:
        List of (estimated_pos, uncertainty, direction, solution) tuples,
        one per distinct solution.
    """
    from .calibration import estimate_star_position

    estimates = []
    rays = result.get_direction_rays(label)

    for direction, solution in rays:
        pos, uncertainty = estimate_star_position(
            direction,
            result.session.observer_pos,
            name_bounds,
        )
        estimates.append((pos, uncertainty, direction, solution))

    return estimates
