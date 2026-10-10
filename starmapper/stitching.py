"""Multi-image stitching for coordinate prediction.

This module links multiple observatory screenshots using constellation
correspondences to estimate positions of unknown stars when known-position
reference stars are split across different images.

Uses the "spherical moon" camera model where:
- The player walks on a tiny spherical moon at ~10°/sec
- "Up" = local gravity = radial direction from moon center
- Camera has a fixed upward pitch - shared across images
- Camera is always level (no roll)
- Only yaw (horizontal rotation) is controllable per-image

Local "up" plus yaw spans every 3D rotation, so each image's orientation is
fitted as a free rotation and then expressed in these terms.
"""

import json
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Callable
from scipy.optimize import minimize_scalar
from pathlib import Path

from .calibration import (
    SphericalCalibration, StarMarker,
    camera_frame_to_params,
    project_star_raw, compute_direction_raw,
)


# =============================================================================
# Note: Core spherical geometry functions (camera_frame_to_params,
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
    observatory_name: str | None = None  # Saved in session files

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


def build_known_stars(stars, observer) -> dict[str, tuple[int, int, int]]:
    """Build name -> 3D coords dict of all positioned stars except the observer."""
    known_stars = {}
    for star in stars:
        if star.has_position and star.coordinates != observer.coordinates:
            name = star.player_name or star.original_name
            if name:
                known_stars[name] = star.position
    return known_stars


def save_session(session: StitchingSession, path: Path) -> None:
    """Save a session's images and markings to a JSON file.

    Image paths are stored relative to the session file. Known-star
    coordinates are not stored; they are looked up again on load.
    """
    path = Path(path)
    data = {
        "observatory": session.observatory_name,
        "images": [
            {
                "path": os.path.relpath(img.image_path, path.parent),
                "width": img.image_width,
                "height": img.image_height,
                "known": {
                    m.name: [round(m.pixel_x, 1), round(m.pixel_y, 1)]
                    for m in img.known_markers
                },
                "correspondences": {
                    m.label: [round(m.pixel_x, 1), round(m.pixel_y, 1)]
                    for m in img.correspondence_markers
                },
            }
            for img in session.images
        ],
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


def load_session(path: Path, observer, stars) -> StitchingSession:
    """Load a session saved by save_session.

    Args:
        path: Session JSON file
        observer: The Star object for the observatory
        stars: List of all stars (for looking up known-star coordinates)

    Raises:
        ValueError: If a known marker names a star with no coordinates
    """
    path = Path(path)
    data = json.loads(path.read_text())

    session = StitchingSession(
        observer_pos=observer.position,
        known_stars=build_known_stars(stars, observer),
        observatory_name=data.get("observatory"),
    )

    for img in data["images"]:
        idx = session.add_image(path.parent / img["path"], img["width"], img["height"])
        for name, (px, py) in img.get("known", {}).items():
            if not session.add_known_marker(idx, name, px, py):
                raise ValueError(
                    f"{path}: star '{name}' in image {idx} has no known coordinates"
                )
        for label, (px, py) in img.get("correspondences", {}).items():
            session.add_correspondence(idx, label, px, py)

    return session


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

    def label_direction(self, label: str) -> tuple[np.ndarray, float] | None:
        """Mean direction to a correspondence label under the best solution.

        Averages over every image the label is marked in.

        Returns:
            (unit_direction, spread) where spread is the largest angle in
            radians between the mean and any single image's direction
            (0 if marked in only one image), or None if the label is unmarked.
        """
        best = self.best_solution()
        if best is None:
            return None

        directions = []
        for idx, img in enumerate(self.session.images):
            marker = img.get_correspondence(label)
            if marker:
                cal = best.calibrations[idx]
                directions.append(compute_direction_raw(
                    marker.pixel_x, marker.pixel_y,
                    cal.up_theta, cal.up_phi, cal.yaw, cal.pitch,
                    np.radians(cal.fov_horizontal),
                    img.image_width, img.image_height,
                ))
        if not directions:
            return None

        mean = np.mean(directions, axis=0)
        mean = mean / np.linalg.norm(mean)
        spread = max(np.arccos(np.clip(np.dot(d, mean), -1, 1)) for d in directions)
        return mean, float(spread)

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


def _pixel_to_camera_ray(
    pixel_x: float,
    pixel_y: float,
    fov_h: float,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Unit ray for a pixel in camera coordinates (right, up, forward)."""
    scale = image_width / (2 * np.tan(fov_h / 2))
    ray = np.array([
        (pixel_x - image_width / 2) / scale,
        -(pixel_y - image_height / 2) / scale,
        1.0,
    ])
    return ray / np.linalg.norm(ray)


def _fit_rotation(world_dirs: np.ndarray, camera_rays: np.ndarray) -> np.ndarray:
    """Best rotation R with R @ world_dir ~= camera_ray (Wahba's problem, via SVD).

    The rows of R are the camera's right, up and forward axes in world
    coordinates. R is a proper rotation, which matches the left-handed
    frame of build_camera_frame (right = up x forward).
    """
    u, _, vt = np.linalg.svd(camera_rays.T @ world_dirs)
    d = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    return u @ d @ vt


def _angles_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle in degrees between matching rows of two arrays of unit vectors."""
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1), -1, 1)))


def _fit_images(
    session: StitchingSession,
    fov_h: float,
    cross_correspondences: dict[str, list[tuple[int, CorrespondenceMarker]]],
) -> tuple[list[np.ndarray | None], np.ndarray, np.ndarray]:
    """Fit a rotation per image at a given FOV.

    Images with 2+ known stars are fitted directly. Images with fewer are
    then fitted using correspondences to already-fitted images as extra
    reference directions, repeating until no more images can be chained in.

    Returns:
        (rotations, known_residuals, correspondence_residuals), where
        rotations has None for images that could not be fitted and the
        residuals are angles in degrees.
    """
    obs_pos = np.array(session.observer_pos, dtype=float)
    n_images = len(session.images)
    rotations: list[np.ndarray | None] = [None] * n_images

    def ray(img: ImageMarking, marker) -> np.ndarray:
        return _pixel_to_camera_ray(
            marker.pixel_x, marker.pixel_y, fov_h, img.image_width, img.image_height
        )

    known_world = []
    known_rays = []
    for img in session.images:
        world = [np.array(m.coords_3d, dtype=float) - obs_pos for m in img.known_markers]
        known_world.append([w / np.linalg.norm(w) for w in world])
        known_rays.append([ray(img, m) for m in img.known_markers])

    def correspondence_directions() -> dict[str, np.ndarray]:
        """Mean world direction per label, from images fitted so far."""
        result = {}
        for label, occurrences in cross_correspondences.items():
            dirs = [
                rotations[idx].T @ ray(session.images[idx], marker)
                for idx, marker in occurrences
                if rotations[idx] is not None
            ]
            if dirs:
                mean = np.mean(dirs, axis=0)
                result[label] = mean / np.linalg.norm(mean)
        return result

    progress = True
    while progress:
        progress = False
        linked = correspondence_directions()
        for i, img in enumerate(session.images):
            if rotations[i] is not None:
                continue
            world = list(known_world[i])
            rays = list(known_rays[i])
            # Only fall back on correspondences when known stars aren't enough
            if len(world) < 2:
                for m in img.correspondence_markers:
                    if m.label in linked:
                        world.append(linked[m.label])
                        rays.append(ray(img, m))
            if len(world) >= 2:
                rotations[i] = _fit_rotation(np.array(world), np.array(rays))
                progress = True

    known_residuals = []
    for i in range(n_images):
        if rotations[i] is not None and known_world[i]:
            projected = np.array(known_world[i]) @ rotations[i].T
            known_residuals.extend(_angles_deg(projected, np.array(known_rays[i])))

    corr_residuals = []
    linked = correspondence_directions()
    for label, occurrences in cross_correspondences.items():
        for idx, marker in occurrences:
            if rotations[idx] is not None:
                direction = rotations[idx].T @ ray(session.images[idx], marker)
                corr_residuals.extend(_angles_deg(direction[None, :], linked[label][None, :]))

    return rotations, np.array(known_residuals), np.array(corr_residuals)


def solve_stitching(
    session: StitchingSession,
    fixed_fov: float | None = None,  # If set, FOV won't be optimized
    fixed_pitch: float = 0.0,
    correspondence_weight: float = 0.1,  # Reduced: prioritize known stars
    progress_callback: Callable[[str], None] | None = None,
) -> StitchingResult:
    """Solve multi-image stitching problem.

    Each image's camera orientation is a free 3D rotation, fitted in closed
    form from its known stars (see _fit_rotation). All images share one FOV,
    found by a 1D search unless fixed_fov is given.

    A shared camera pitch cannot be measured from the star fit: local "up"
    plus yaw already spans every rotation. fixed_pitch only decides how each
    fitted orientation is split into a local "up" vector and a yaw.

    Args:
        session: StitchingSession with marked images
        fixed_fov: Horizontal FOV in degrees, or None to fit it
        fixed_pitch: Assumed upward camera pitch in degrees
        correspondence_weight: Weight for correspondence errors vs known star
            errors when fitting the FOV
        progress_callback: Optional callback for progress updates

    Returns:
        StitchingResult with one solution, or none if some image has too few
        markers to be fitted. Solution errors are RMS angles in degrees.
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

    log(f"Solving stitching for {n_images} images...")

    def rms(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean(values ** 2))) if len(values) else 0.0

    def cost(fov_h: float) -> float:
        rotations, known_res, corr_res = _fit_images(session, fov_h, cross_correspondences)
        if any(r is None for r in rotations):
            return float('inf')
        return float(np.sum(known_res ** 2) + correspondence_weight * np.sum(corr_res ** 2))

    if fixed_fov is not None:
        fov_h = np.radians(fixed_fov)
    else:
        # Coarse scan first: the cost is not guaranteed to have a single minimum
        candidates = np.radians(np.arange(40.0, 140.1, 2.0))
        costs = [cost(f) for f in candidates]
        if not np.isfinite(min(costs)):
            fov_h = np.radians(90.0)
        else:
            coarse = candidates[int(np.argmin(costs))]
            step = np.radians(2.0)
            fov_h = minimize_scalar(
                cost, bounds=(coarse - step, coarse + step), method='bounded',
            ).x

    rotations, known_res, corr_res = _fit_images(session, fov_h, cross_correspondences)

    unsolved = [i for i, r in enumerate(rotations) if r is None]
    if unsolved:
        for i in unsolved:
            log(f"  Image {i}: not enough markers (need 2+ known stars, or "
                f"correspondences linking it to a solved image)")
        log("No valid solution found")
        return StitchingResult(session=session, solutions=[])

    fov_deg = np.degrees(fov_h)
    pitch_rad = np.radians(fixed_pitch)

    calibrations = []
    for rotation in rotations:
        right, camera_up, forward = rotation
        up_theta, up_phi, yaw = camera_frame_to_params(forward, right, camera_up, pitch_rad)
        calibrations.append(SphericalCalibration(
            fov_horizontal=fov_deg,
            up_theta=up_theta,
            up_phi=up_phi,
            yaw=yaw,
            pitch=pitch_rad,
            residual_error=0,
        ))

    solution = StitchingSolution(
        calibrations=calibrations,
        total_error=rms(np.concatenate([known_res, corr_res])),
        known_star_error=rms(known_res),
        correspondence_error=rms(corr_res),
    )

    log(f"\nSolution: RMS error={solution.total_error:.2f}° "
        f"(known={solution.known_star_error:.2f}°, corr={solution.correspondence_error:.2f}°)")
    log(f"  Shared FOV: {fov_deg:.1f}°{' (fixed)' if fixed_fov is not None else ''}, "
        f"assumed pitch: {fixed_pitch:.1f}°")
    for i, cal in enumerate(calibrations):
        up = cal.up_vector()
        forward = cal.forward_vector()
        log(f"  Image {i}: up=({up[0]:.2f}, {up[1]:.2f}, {up[2]:.2f}), "
            f"yaw={np.degrees(cal.yaw):.1f}°")
        log(f"    forward=({forward[0]:.2f}, {forward[1]:.2f}, {forward[2]:.2f})")

    # Show where known stars project to with this solution
    log("\nProjection check for known stars:")
    for i, img in enumerate(session.images):
        cal = calibrations[i]
        for m in img.known_markers:
            star_pos = np.array(m.coords_3d, dtype=float)
            px, py, visible = project_star_raw(
                star_pos, obs_pos,
                cal.up_theta, cal.up_phi, cal.yaw, cal.pitch, fov_h,
                img.image_width, img.image_height,
            )
            err = np.hypot(px - m.pixel_x, py - m.pixel_y) if visible else float('inf')
            log(f"  Image {i} {m.name}: marked=({m.pixel_x:.0f},{m.pixel_y:.0f}), "
                f"projected=({px:.0f},{py:.0f}), visible={visible}, err={err:.1f}px")

    # Show correspondence direction agreement
    log("\nCorrespondence direction check:")
    for label, occurrences in cross_correspondences.items():
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
