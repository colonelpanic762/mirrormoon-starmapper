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

    Calibration is built up by reading the coordinates of stars that sit
    level with the ship's diamond on the map (left/right) and directly
    above or below it (up/down). The map's 3D axes are fitted from those.
    """
    ship_position: tuple[int, int, int] | None = None

    # Samples: list of (x, y, z) coordinates for each direction
    left_samples: list[tuple[int, int, int]] = field(default_factory=list)
    right_samples: list[tuple[int, int, int]] = field(default_factory=list)
    up_samples: list[tuple[int, int, int]] = field(default_factory=list)
    down_samples: list[tuple[int, int, int]] = field(default_factory=list)

    # Fitted map axes as 3D unit vectors (None until enough samples)
    right_axis: np.ndarray | None = None
    up_axis: np.ndarray | None = None

    def add_sample(self, direction: ScreenDirection, coords: tuple[int, int, int]) -> None:
        """Add a calibration sample for a screen direction."""
        samples = getattr(self, f"{direction}_samples")
        samples.append(coords)
        self._recompute_axes()

    def _offsets(self, direction: ScreenDirection) -> np.ndarray:
        """Offsets from the ship to the samples in one screen direction."""
        samples = getattr(self, f"{direction}_samples")
        ship = np.array(self.ship_position, dtype=np.float64)
        return np.array(samples, dtype=np.float64).reshape(-1, 3) - ship

    @staticmethod
    def _plane_normal(offsets: np.ndarray) -> np.ndarray | None:
        """Unit normal of the best-fit plane through the ship and the samples."""
        if len(offsets) < 2:
            return None
        _, singular, vt = np.linalg.svd(offsets)
        if singular[1] < 1e-6:
            return None  # Samples are in line with the ship; no plane defined
        return vt[-1]

    def _recompute_axes(self) -> None:
        """Fit the map's axes from the current samples.

        The map is a flat projection, so every star level with the ship's
        diamond lies in one plane through the ship whatever its depth, and
        the map's up axis is that plane's normal. Likewise stars directly
        above or below the diamond define the right axis. Depth drops out
        entirely; the side each sample was on only fixes the axes' signs.
        """
        self.right_axis = None
        self.up_axis = None
        if self.ship_position is None:
            return

        left, right = self._offsets("left"), self._offsets("right")
        up, down = self._offsets("up"), self._offsets("down")
        up_axis = self._plane_normal(np.vstack([left, right]))
        right_axis = self._plane_normal(np.vstack([up, down]))
        if up_axis is None or right_axis is None:
            return

        if np.sum(right @ right_axis) - np.sum(left @ right_axis) < 0:
            right_axis = -right_axis
        if np.sum(up @ up_axis) - np.sum(down @ up_axis) < 0:
            up_axis = -up_axis

        # The two fits are independent; square them up
        up_axis = up_axis - np.dot(up_axis, right_axis) * right_axis
        norm = np.linalg.norm(up_axis)
        if norm < 1e-6:
            return
        self.right_axis = right_axis
        self.up_axis = up_axis / norm

    def sample_errors(self) -> list[tuple[str, tuple, float]]:
        """How far each sample is from where the fitted axes say it should be.

        Returns:
            (direction, coordinates, error in PC) per sample. A left/right
            sample's error is its height above or below the diamond's level;
            an up/down sample's is its sideways offset. A large value usually
            means a misread coordinate or a star that was not on the line.
        """
        if self.right_axis is None:
            return []
        errors = []
        for direction in ("left", "right", "up", "down"):
            axis = self.up_axis if direction in ("left", "right") else self.right_axis
            for coords, offset in zip(getattr(self, f"{direction}_samples"),
                                      self._offsets(direction)):
                errors.append((direction, coords, float(abs(np.dot(offset, axis)))))
        return errors

    def get_screen_axes(self) -> 'ScreenAxes | None':
        """Get ScreenAxes from calibration, or None if insufficient data."""
        if self.right_axis is None:
            return None
        return ScreenAxes(
            right=self.right_axis,
            up=self.up_axis,
            center=np.array(self.ship_position, dtype=np.float64),
        )

    def is_calibrated(self) -> bool:
        """Check if calibration has enough data for navigation."""
        return self.get_screen_axes() is not None

    def status(self) -> str:
        """Return a status string describing calibration state."""
        lines = []
        if self.ship_position:
            lines.append(f"Ship: ({self.ship_position[0]}, {self.ship_position[1]}, {self.ship_position[2]})")
        else:
            lines.append("Ship position not set")

        errors = {(d, tuple(c)): e for d, c, e in self.sample_errors()}
        for direction in ["left", "right", "up", "down"]:
            samples = getattr(self, f"{direction}_samples")
            lines.append(f"  {direction.capitalize()}: {len(samples)} samples")
            for sample in samples:
                error = errors.get((direction, tuple(sample)))
                suffix = f"  ({error:.0f} PC off the line)" if error is not None else ""
                lines.append(f"    ({sample[0]}, {sample[1]}, {sample[2]}){suffix}")

        if self.is_calibrated():
            r, u = self.right_axis, self.up_axis
            lines.append(f"  Map right: ({r[0]:+.2f}, {r[1]:+.2f}, {r[2]:+.2f})")
            lines.append(f"  Map up:    ({u[0]:+.2f}, {u[1]:+.2f}, {u[2]:+.2f})")
            lines.append("✓ Calibration complete - ready for navigation")
        else:
            needed = []
            if self.ship_position is None:
                needed.append("ship position")
            if len(self.left_samples) + len(self.right_samples) < 2:
                needed.append("2 stars level with the ship (left or right)")
            if len(self.up_samples) + len(self.down_samples) < 2:
                needed.append("2 stars directly above or below the ship (up or down)")
            lines.append(f"✗ Need: {', '.join(needed) or 'samples that are not in line with the ship'}")

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
        cal._recompute_axes()
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
        raise ValueError("Calibration incomplete. Need the ship position and at least "
                        "2 samples each for horizontal (left/right) and vertical (up/down).")

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
