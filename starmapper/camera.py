"""First-person camera for a player standing on a tiny spherical moon.

Models the in-game observatory controls:
- Turning (A/D) rotates around local "up" (the radial direction from the
  moon's center, i.e. where the player is standing)
- Walking (W/S) moves the standing point around the moon
- The camera is tilted up from the horizon by a fixed pitch

The camera's three axes are the state. Stars never move; only the camera does.
"""

from dataclasses import dataclass

import numpy as np


# Vertical FOV in degrees. Fitted as ~84° horizontal on 1280x800 screenshots;
# assuming the game fixes the vertical FOV, that is 90° horizontal at 16:9.
# Unconfirmed at other aspect ratios.
VERTICAL_FOV_DEG = 58.7

# Walking speed around the moon (360° / 36 s circumnavigation, measured)
WALK_RATE_DEG_PER_SEC = 10.0


def vertical_fov() -> float:
    """Vertical field of view in radians."""
    return np.radians(VERTICAL_FOV_DEG)


def horizontal_fov(aspect: float) -> float:
    """Horizontal field of view in radians for a width/height aspect ratio."""
    return 2 * np.arctan(np.tan(vertical_fov() / 2) * aspect)


@dataclass
class MoonCamera:
    """Camera orientation on the moon surface.

    right, camera_up and forward are orthonormal unit vectors in world
    coordinates, in the game's left-handed frame (right = camera_up x forward).
    """
    right: np.ndarray
    camera_up: np.ndarray
    forward: np.ndarray
    pitch: float = 0.0  # Upward tilt from horizon (radians)

    @classmethod
    def looking_at(cls, direction: np.ndarray, pitch: float = 0.0) -> "MoonCamera":
        """Create a camera with the given direction at the center of view.

        The twist around that direction is arbitrary (world +Y is kept as
        close to screen-up as possible).
        """
        forward = np.asarray(direction, dtype=float)
        forward = forward / np.linalg.norm(forward)

        hint = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(forward, hint)) > 0.99:
            hint = np.array([1.0, 0.0, 0.0])
        camera_up = hint - np.dot(hint, forward) * forward
        camera_up = camera_up / np.linalg.norm(camera_up)

        return cls(
            right=np.cross(camera_up, forward),
            camera_up=camera_up,
            forward=forward,
            pitch=pitch,
        )

    @property
    def local_up(self) -> np.ndarray:
        """Local gravity "up": the player's position on the moon."""
        return np.sin(self.pitch) * self.forward + np.cos(self.pitch) * self.camera_up

    @property
    def forward_horizontal(self) -> np.ndarray:
        """Walking direction: forward projected onto the local horizon."""
        return np.cos(self.pitch) * self.forward - np.sin(self.pitch) * self.camera_up

    def _set_from_local(self, forward_horizontal: np.ndarray, up: np.ndarray) -> None:
        """Rebuild the camera axes from horizon-frame vectors and the pitch."""
        cos_p = np.cos(self.pitch)
        sin_p = np.sin(self.pitch)
        self.forward = cos_p * forward_horizontal + sin_p * up
        self.camera_up = -sin_p * forward_horizontal + cos_p * up

    def turn(self, angle: float) -> None:
        """Turn right by angle (radians) around local up. Negative turns left."""
        forward_horizontal = self.forward_horizontal
        up = self.local_up
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        new_forward = cos_a * forward_horizontal + sin_a * self.right
        self.right = -sin_a * forward_horizontal + cos_a * self.right
        self._set_from_local(new_forward, up)

    def walk(self, angle: float) -> None:
        """Walk forward by angle (radians) around the moon. Negative walks back."""
        forward_horizontal = self.forward_horizontal
        up = self.local_up
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        self._set_from_local(
            cos_a * forward_horizontal - sin_a * up,
            sin_a * forward_horizontal + cos_a * up,
        )

    def twist(self, angle: float) -> None:
        """Roll the view around its center by angle (radians).

        Not a game control: it is equivalent to standing somewhere else on
        the moon while looking at the same point in the sky.
        """
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        right = cos_a * self.right + sin_a * self.camera_up
        self.camera_up = -sin_a * self.right + cos_a * self.camera_up
        self.right = right

    def project(self, direction: np.ndarray, aspect: float) -> tuple[float, float, float]:
        """Project a world direction to normalized screen coordinates.

        Returns:
            (x, y, depth) with x and y in [-1, 1] across the frame (x right,
            y up). depth <= 0 means the direction is behind the camera, in
            which case x and y are 0.
        """
        direction = np.asarray(direction, dtype=float)
        depth = float(np.dot(direction, self.forward))
        if depth <= 0:
            return 0.0, 0.0, depth
        half_height = np.tan(vertical_fov() / 2)
        x = np.dot(direction, self.right) / depth / (half_height * aspect)
        y = np.dot(direction, self.camera_up) / depth / half_height
        return float(x), float(y), depth

    def screen_to_direction(self, x: float, y: float, aspect: float) -> np.ndarray:
        """Unit world direction for normalized screen coordinates."""
        half_height = np.tan(vertical_fov() / 2)
        direction = (
            self.forward
            + x * half_height * aspect * self.right
            + y * half_height * self.camera_up
        )
        return direction / np.linalg.norm(direction)
