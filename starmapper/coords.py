"""Coordinate parsing and transformation utilities."""

from dataclasses import dataclass, field
from typing import Tuple
import numpy as np

# Letter to coordinate range mapping (first digit of each axis)
LETTER_TO_RANGE = {
    "A": 0, "E": 1, "I": 2, "U": 3, "Y": 4,
    "H": 5, "J": 6, "L": 7, "T": 8, "X": 9,
}

RANGE_TO_LETTER = {v: k for k, v in LETTER_TO_RANGE.items()}


@dataclass
class Star:
    """Represents a star in the MirrorMoon EP galaxy."""

    coordinates: str | None  # 9-digit location "XXXYYYZZZ", or None if unknown
    original_name: str | None  # XXX/## format
    player_name: str | None = None  # max 6 chars, uppercase
    discovered_by_user: bool = False
    has_observatory: bool | None = None
    observatory_active: bool | None = None
    anomaly_visible: bool | None = None
    notes: str | None = None  # Optional notes about this star

    # Cached position for performance
    _position: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.coordinates is not None:
            if len(self.coordinates) != 9 or not self.coordinates.isdigit():
                raise ValueError(f"Coordinates must be 9 digits, got: {self.coordinates}")

    @property
    def position(self) -> np.ndarray | None:
        """Return (x, y, z) position as numpy array, or None if coordinates unknown."""
        if self.coordinates is None:
            return None
        if self._position is None:
            self._position = np.array(parse_coordinates(self.coordinates), dtype=np.float64)
        return self._position

    @property
    def has_position(self) -> bool:
        """Return True if this star has known coordinates."""
        return self.coordinates is not None

    @property
    def display_name(self) -> str:
        """Return player name if set, otherwise original name, or '(unnamed)'."""
        return self.player_name or self.original_name or "(unnamed)"


def parse_coordinates(coord_string: str) -> Tuple[int, int, int]:
    """Parse a 9-digit coordinate string into (x, y, z) tuple.

    Example: "123456789" -> (123, 456, 789)
    """
    if len(coord_string) != 9 or not coord_string.isdigit():
        raise ValueError(f"Coordinate string must be exactly 9 digits: {coord_string}")

    x = int(coord_string[0:3])
    y = int(coord_string[3:6])
    z = int(coord_string[6:9])
    return (x, y, z)


def to_coordinate_string(x: int, y: int, z: int) -> str:
    """Convert (x, y, z) coordinates to 9-digit string.

    Example: (123, 456, 789) -> "123456789"
    """
    if not all(0 <= v <= 999 for v in (x, y, z)):
        raise ValueError(f"Coordinates must be in range 0-999: ({x}, {y}, {z})")

    return f"{x:03d}{y:03d}{z:03d}"


def distance(star1: Star, star2: Star) -> float:
    """Calculate Euclidean distance between two stars."""
    return float(np.linalg.norm(star1.position - star2.position))


def generate_original_name(coord_string: str, index: int = 0) -> str:
    """Generate the original name (XXX/##) from coordinates.

    The three letters correspond to the first digit of each axis.
    """
    x, y, z = parse_coordinates(coord_string)

    letter_x = RANGE_TO_LETTER[x // 100]
    letter_y = RANGE_TO_LETTER[y // 100]
    letter_z = RANGE_TO_LETTER[z // 100]

    return f"{letter_x}{letter_y}{letter_z}/{index:02d}"
