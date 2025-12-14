"""Ship star map navigation helper.

Calculates directions to targets based on reference stars visible on the
ship's star map display. Uses calibration samples to establish a local
coordinate frame, then projects target directions onto it.
"""

from dataclasses import dataclass, field
from typing import Literal
import numpy as np

from .coords import parse_coordinates, LETTER_TO_RANGE


# Type alias for screen directions
ScreenDirection = Literal["left", "right", "up", "down"]


@dataclass
class NavCalibration:
    """Stores calibration data for ship star map navigation.

    Calibration is built up by sampling star coordinates while moving
    in known screen directions. The samples are used to compute 3D
    vectors corresponding to screen directions.
    """
    ship_position: tuple[int, int, int] | None = None

    # Samples: list of (x, y, z) coordinates for each direction
    left_samples: list[tuple[int, int, int]] = field(default_factory=list)
    right_samples: list[tuple[int, int, int]] = field(default_factory=list)
    up_samples: list[tuple[int, int, int]] = field(default_factory=list)
    down_samples: list[tuple[int, int, int]] = field(default_factory=list)

    # Computed vectors (None until enough samples)
    left_vector: np.ndarray | None = None
    right_vector: np.ndarray | None = None
    up_vector: np.ndarray | None = None
    down_vector: np.ndarray | None = None

    def add_sample(self, direction: ScreenDirection, coords: tuple[int, int, int]) -> None:
        """Add a calibration sample for a screen direction."""
        samples = getattr(self, f"{direction}_samples")
        samples.append(coords)
        self._recompute_vectors()

    def _recompute_vectors(self) -> None:
        """Recompute direction vectors from current samples."""
        # Need at least 2 samples to compute a vector
        if len(self.left_samples) >= 2:
            self.left_vector = self._compute_vector(self.left_samples)
        if len(self.right_samples) >= 2:
            self.right_vector = self._compute_vector(self.right_samples)
        if len(self.up_samples) >= 2:
            self.up_vector = self._compute_vector(self.up_samples)
        if len(self.down_samples) >= 2:
            self.down_vector = self._compute_vector(self.down_samples)

        # Orthogonalize if we have both horizontal and vertical
        self._orthogonalize()

    def _compute_vector(self, samples: list[tuple[int, int, int]]) -> np.ndarray:
        """Compute direction vector by averaging directions from ship to each sample.

        Each sample represents a star visible in that screen direction. We compute
        the unit vector from ship to each sample and average them.
        """
        if len(samples) < 2:
            return None

        if self.ship_position is None:
            # Without ship position, fall back to line-fitting between samples
            points = np.array(samples, dtype=np.float64)
            vec = points[-1] - points[0]
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else None

        ship = np.array(self.ship_position, dtype=np.float64)

        # Compute normalized direction from ship to each sample
        directions = []
        for sample in samples:
            vec = np.array(sample, dtype=np.float64) - ship
            norm = np.linalg.norm(vec)
            if norm > 0:
                directions.append(vec / norm)

        if len(directions) < 2:
            return None

        # Average the normalized direction vectors
        avg_direction = np.mean(directions, axis=0)
        norm = np.linalg.norm(avg_direction)
        if norm > 0:
            return avg_direction / norm
        return None

    def _orthogonalize(self) -> None:
        """Orthogonalize up/down vectors against left/right using Gram-Schmidt."""
        # Get horizontal vector (prefer left, fall back to right)
        h_vec = None
        if self.left_vector is not None:
            h_vec = self.left_vector
        elif self.right_vector is not None:
            h_vec = -self.right_vector  # Negate since right is opposite of left

        if h_vec is None:
            return

        # Orthogonalize vertical vectors
        if self.up_vector is not None:
            self.up_vector = self._orthogonalize_against(self.up_vector, h_vec)
        if self.down_vector is not None:
            self.down_vector = self._orthogonalize_against(self.down_vector, h_vec)

    def _orthogonalize_against(self, vec: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Remove component of vec in direction of ref (Gram-Schmidt)."""
        projection = np.dot(vec, ref) * ref
        orthogonal = vec - projection
        norm = np.linalg.norm(orthogonal)
        if norm > 0:
            return orthogonal / norm
        return vec  # Return original if orthogonalization fails

    def get_screen_axes(self) -> 'ScreenAxes | None':
        """Get ScreenAxes from calibration, or None if insufficient data."""
        # Need at least one horizontal and one vertical vector
        h_vec = self.left_vector if self.left_vector is not None else (
            -self.right_vector if self.right_vector is not None else None
        )
        v_vec = self.up_vector if self.up_vector is not None else (
            -self.down_vector if self.down_vector is not None else None
        )

        if h_vec is None or v_vec is None:
            return None

        # Use ship position as center, or average of all samples
        if self.ship_position is not None:
            center = np.array(self.ship_position, dtype=np.float64)
        else:
            all_samples = (self.left_samples + self.right_samples +
                          self.up_samples + self.down_samples)
            if all_samples:
                center = np.mean([np.array(s) for s in all_samples], axis=0)
            else:
                center = np.array([500, 500, 500], dtype=np.float64)

        return ScreenAxes(
            right=-h_vec,  # Right is opposite of left
            up=v_vec,
            center=center,
        )

    def is_calibrated(self) -> bool:
        """Check if calibration has enough data for navigation."""
        return self.get_screen_axes() is not None

    def status(self) -> str:
        """Return a status string describing calibration state."""
        lines = []
        if self.ship_position:
            lines.append(f"Ship: ({self.ship_position[0]}, {self.ship_position[1]}, {self.ship_position[2]})")

        for direction in ["left", "right", "up", "down"]:
            samples = getattr(self, f"{direction}_samples")
            vec = getattr(self, f"{direction}_vector")
            status = f"{len(samples)} samples"
            if vec is not None:
                status += f" → ({vec[0]:.2f}, {vec[1]:.2f}, {vec[2]:.2f})"
            lines.append(f"  {direction.capitalize()}: {status}")

        if self.is_calibrated():
            lines.append("✓ Calibration complete - ready for navigation")
        else:
            needed = []
            if self.left_vector is None and self.right_vector is None:
                needed.append("horizontal (left or right)")
            if self.up_vector is None and self.down_vector is None:
                needed.append("vertical (up or down)")
            lines.append(f"✗ Need more samples: {', '.join(needed)}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize for JSON storage."""
        return {
            "ship_position": self.ship_position,
            "left_samples": self.left_samples,
            "right_samples": self.right_samples,
            "up_samples": self.up_samples,
            "down_samples": self.down_samples,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'NavCalibration':
        """Deserialize from JSON storage."""
        cal = cls(
            ship_position=tuple(data["ship_position"]) if data.get("ship_position") else None,
            left_samples=[tuple(s) for s in data.get("left_samples", [])],
            right_samples=[tuple(s) for s in data.get("right_samples", [])],
            up_samples=[tuple(s) for s in data.get("up_samples", [])],
            down_samples=[tuple(s) for s in data.get("down_samples", [])],
        )
        cal._recompute_vectors()
        return cal


@dataclass
class ScreenAxes:
    """Represents the orientation of the ship's star map view."""
    right: np.ndarray  # 3D direction corresponding to screen-right
    up: np.ndarray     # 3D direction corresponding to screen-up
    center: np.ndarray # 3D position at center of view


@dataclass
class NavigationResult:
    """Result of calculating direction to a target."""
    target_coords: tuple[int, int, int]
    target_name: str | None
    clock_direction: float  # Hours (12 = up, 3 = right, 6 = down, 9 = left)
    clock_description: str  # e.g., "9 o'clock"
    left_component: float   # Negative = left, positive = right
    down_component: float   # Negative = down, positive = up
    distance_from_view_center: float


def decode_original_name(name: str) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    """Decode an original name (XXX/##) to coordinate ranges.

    Args:
        name: Original name like "TYU/29" or "IAY/53"

    Returns:
        Tuple of ((x_min, x_max), (y_min, y_max), (z_min, z_max)) or None if invalid
    """
    # Handle both "TYU/29" and "TYU29" formats
    clean_name = name.upper().replace("/", "")

    if len(clean_name) < 3:
        return None

    letters = clean_name[:3]

    try:
        x_base = LETTER_TO_RANGE[letters[0]] * 100
        y_base = LETTER_TO_RANGE[letters[1]] * 100
        z_base = LETTER_TO_RANGE[letters[2]] * 100
    except KeyError:
        return None

    return (
        (x_base, x_base + 99),
        (y_base, y_base + 99),
        (z_base, z_base + 99),
    )


def original_name_center(name: str) -> tuple[int, int, int] | None:
    """Get the center point of an original name's coordinate range.

    Args:
        name: Original name like "TYU/29"

    Returns:
        (x, y, z) center coordinates, or None if invalid name
    """
    ranges = decode_original_name(name)
    if ranges is None:
        return None

    return (
        (ranges[0][0] + ranges[0][1]) // 2,
        (ranges[1][0] + ranges[1][1]) // 2,
        (ranges[2][0] + ranges[2][1]) // 2,
    )


def calculate_screen_axes(
    lower_left: tuple[int, int, int],
    lower_right: tuple[int, int, int],
    upper_right: tuple[int, int, int],
    upper_left: tuple[int, int, int],
) -> ScreenAxes:
    """Calculate screen coordinate axes from four reference stars.

    Args:
        lower_left: Coordinates of star in lower-left of screen
        lower_right: Coordinates of star in lower-right of screen
        upper_right: Coordinates of star in upper-right of screen
        upper_left: Coordinates of star in upper-left of screen

    Returns:
        ScreenAxes with right, up, and center vectors
    """
    ll = np.array(lower_left, dtype=np.float64)
    lr = np.array(lower_right, dtype=np.float64)
    ur = np.array(upper_right, dtype=np.float64)
    ul = np.array(upper_left, dtype=np.float64)

    # Screen-right: average of left→right vectors on both edges
    right_lower = lr - ll
    right_upper = ur - ul
    screen_right = (right_lower + right_upper) / 2

    # Screen-up: average of bottom→top vectors on both edges
    up_left = ul - ll
    up_right = ur - lr
    screen_up = (up_left + up_right) / 2

    # View center: average of all four reference stars
    center = (ll + lr + ur + ul) / 4

    return ScreenAxes(
        right=screen_right,
        up=screen_up,
        center=center,
    )


def calculate_direction(
    screen_axes: ScreenAxes,
    target: tuple[int, int, int],
    target_name: str | None = None,
) -> NavigationResult:
    """Calculate the direction to pan to reach a target.

    Args:
        screen_axes: The current view's screen axes
        target: Target (x, y, z) coordinates
        target_name: Optional name for display

    Returns:
        NavigationResult with clock direction and components
    """
    target_pos = np.array(target, dtype=np.float64)

    # Direction from view center to target
    direction = target_pos - screen_axes.center
    distance = float(np.linalg.norm(direction))

    # Project onto screen axes (dot product)
    # Positive = right/up, Negative = left/down
    right_component = float(np.dot(direction, screen_axes.right))
    up_component = float(np.dot(direction, screen_axes.up))

    # Convert to clock direction
    # atan2(right, up) gives angle where:
    #   up only (0, +) = 0 rad = 12 o'clock
    #   right only (+, 0) = pi/2 rad = 3 o'clock
    #   down only (0, -) = pi rad = 6 o'clock
    #   left only (-, 0) = -pi/2 rad = 9 o'clock
    angle_rad = np.arctan2(right_component, up_component)
    angle_hours = (angle_rad * 6 / np.pi) % 12
    if angle_hours == 0:
        angle_hours = 12

    # Generate clock description
    clock_desc = _format_clock_direction(angle_hours)

    return NavigationResult(
        target_coords=target,
        target_name=target_name,
        clock_direction=angle_hours,
        clock_description=clock_desc,
        left_component=-right_component,  # Flip sign so positive = left
        down_component=-up_component,     # Flip sign so positive = down
        distance_from_view_center=distance,
    )


def _format_clock_direction(hours: float) -> str:
    """Format a clock direction as a human-readable string."""
    # Round to nearest hour or half-hour
    rounded = round(hours * 2) / 2
    if rounded == 0 or rounded == 12:
        rounded = 12

    if rounded == int(rounded):
        return f"{int(rounded)} o'clock"
    else:
        whole = int(rounded)
        return f"{whole}:{30} o'clock"


def navigate_to_target(
    reference_stars: dict[str, str],
    target: str,
) -> NavigationResult:
    """High-level function to calculate navigation from reference stars to target.

    Args:
        reference_stars: Dict with keys 'll', 'lr', 'ur', 'ul' mapping to
                        9-digit coordinate strings
        target: Either a 9-digit coordinate string or an original name like "TYU/29"

    Returns:
        NavigationResult with direction to target
    """
    # Parse reference stars
    ll = parse_coordinates(reference_stars['ll'])
    lr = parse_coordinates(reference_stars['lr'])
    ur = parse_coordinates(reference_stars['ur'])
    ul = parse_coordinates(reference_stars['ul'])

    # Calculate screen axes
    axes = calculate_screen_axes(ll, lr, ur, ul)

    # Parse target
    target_name = None
    if len(target) == 9 and target.isdigit():
        target_coords = parse_coordinates(target)
    else:
        # Try as original name
        target_name = target
        center = original_name_center(target)
        if center is None:
            raise ValueError(f"Invalid target: {target}. Expected 9-digit coordinates or original name (e.g., TYU/29)")
        target_coords = center

    return calculate_direction(axes, target_coords, target_name)


def format_navigation_result(result: NavigationResult, verbose: bool = False) -> str:
    """Format navigation result for display."""
    lines = []

    if result.target_name:
        lines.append(f"Target: {result.target_name} → ({result.target_coords[0]}, {result.target_coords[1]}, {result.target_coords[2]})")
    else:
        lines.append(f"Target: ({result.target_coords[0]}, {result.target_coords[1]}, {result.target_coords[2]})")

    lines.append(f"Direction: {result.clock_description}")

    if verbose:
        lines.append(f"Distance from view center: {result.distance_from_view_center:.1f} PC")

        # Describe relative direction in words
        h_dir = "LEFT" if result.left_component > 0 else "RIGHT"
        v_dir = "DOWN" if result.down_component > 0 else "UP"
        h_mag = abs(result.left_component)
        v_mag = abs(result.down_component)

        if h_mag > v_mag * 3:
            lines.append(f"Pan mostly {h_dir}")
        elif v_mag > h_mag * 3:
            lines.append(f"Pan mostly {v_dir}")
        else:
            lines.append(f"Pan {v_dir}-{h_dir}")

    return "\n".join(lines)


def navigate_with_calibration(
    calibration: NavCalibration,
    target: str,
    from_position: tuple[int, int, int] | None = None,
) -> NavigationResult:
    """Navigate to a target using calibrated screen axes.

    Args:
        calibration: NavCalibration with sufficient samples
        target: Either a 9-digit coordinate string or an original name like "TYU/29"
        from_position: Override the center point (for navigation updates during travel)

    Returns:
        NavigationResult with direction to target

    Raises:
        ValueError: If calibration is insufficient or target is invalid
    """
    axes = calibration.get_screen_axes()
    if axes is None:
        raise ValueError("Calibration incomplete. Need at least 2 samples each for "
                        "one horizontal (left/right) and one vertical (up/down) direction.")

    # Override center if a different position is specified
    if from_position is not None:
        axes = ScreenAxes(
            right=axes.right,
            up=axes.up,
            center=np.array(from_position, dtype=np.float64),
        )

    # Parse target
    target_name = None
    if len(target) == 9 and target.isdigit():
        target_coords = parse_coordinates(target)
    else:
        # Try as original name
        target_name = target
        center = original_name_center(target)
        if center is None:
            raise ValueError(f"Invalid target: {target}. Expected 9-digit coordinates or original name (e.g., TYU/29)")
        target_coords = center

    return calculate_direction(axes, target_coords, target_name)
