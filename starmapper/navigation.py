"""Ship star map navigation helper.

Calculates directions to targets based on reference stars visible on the
ship's star map display. Uses screen position of known stars to establish
a local coordinate frame, then projects target directions onto it.
"""

from dataclasses import dataclass
import numpy as np

from .coords import parse_coordinates, LETTER_TO_RANGE


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
