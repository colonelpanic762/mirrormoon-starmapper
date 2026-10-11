"""First-person observatory view.

Draws the sky from an observatory the way the game does (perspective
projection, no flips) and lets the user line it up by eye using the game's
own controls, then mark the anomaly with a two-click uncertainty cone.
"""

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from .camera import MoonCamera, WALK_RATE_DEG_PER_SEC, vertical_fov
from .coords import Star, Observation


BACKGROUND = '#0d0d1a'
TWIST_RATE_DEG_PER_SEC = 30.0
SLOW_FACTOR = 0.2          # Speed multiplier while Shift is held
TICK_SECONDS = 0.03
# OS key auto-repeat arrives as release+press pairs; a release only counts
# once it has gone this long without a new press
RELEASE_DEBOUNCE_SECONDS = 0.08

# key -> (movement, sign)
MOVE_KEYS = {
    'd': ('turn', 1), 'a': ('turn', -1),
    'w': ('walk', 1), 's': ('walk', -1),
    'e': ('twist', 1), 'q': ('twist', -1),
}

HELP_TEXT = (
    "A/D   turn\n"
    "W/S   walk\n"
    "Q/E   twist sky\n"
    "Shift slow\n"
    "[ ]   camera tilt\n"
    "0-9   what to mark\n"
    "Click it, then click\n"
    "its error radius\n"
    "Esc   clear mark"
)

ANOMALY = "ANOMALY"


@dataclass
class ObservatorySettings:
    """Camera settings that are tuned once and saved."""
    # Upward camera tilt from the true horizon. Provisional: fitted from the
    # curve of the moon's edge in two screenshots (12-14°); tune by comparing
    # a turn in-game with a turn here.
    tilt_deg: float = 13.0
    turn_rate_deg_s: float = 45.0     # Placeholder until timed in-game
    aspect: float = 16 / 10           # Frame width / height
    # How far below the center of view the moon's edge appears (measured
    # from screenshots). On a tiny moon the visible edge sits below the true
    # horizon, by this angle minus the tilt.
    horizon_below_center_deg: float = 20.0


class ObservatoryView:
    """Interactive first-person view from an observatory."""

    def __init__(
        self,
        observer: Star,
        stars: list[Star],
        visible_names: list[str] | None = None,
        unlocated_names: list[str] | None = None,
        settings: ObservatorySettings | None = None,
        check_star: Star | None = None,
        on_observation: Callable[[Observation], None] | None = None,
        on_settings_changed: Callable[[ObservatorySettings], None] | None = None,
        sightings: dict[str, tuple[np.ndarray, float]] | None = None,
        on_sighting: Callable[[str, np.ndarray, float], None] | None = None,
    ):
        """
        Args:
            observer: The observatory star
            stars: All stars; those with coordinates are drawn
            visible_names: Display names of stars the user can see in-game
                (highlighted, and used to aim the opening view)
            unlocated_names: Visible stars without coordinates ("need to find")
            settings: Camera settings
            check_star: A star at the anomaly's true position. It is hidden
                from the view, and each marked observation reports how far
                off it was.
            on_observation: Called with each completed anomaly Observation
            on_settings_changed: Called when the tilt is changed in the view
            sightings: Earlier marks of unlocated stars from this observatory,
                name -> (direction, error radius in radians); drawn as circles
            on_sighting: Called with (name, direction, radius) when an
                unlocated star is marked
        """
        self.observer = observer
        self.settings = settings or ObservatorySettings()
        self.unlocated_names = list(unlocated_names or [])
        self.check_star = check_star
        self.on_observation = on_observation
        self.on_settings_changed = on_settings_changed
        self.sightings = dict(sightings or {})
        self.on_sighting = on_sighting
        self.target = ANOMALY   # What the next mark is of

        visible = set(visible_names or [])
        origin = np.array(observer.position, dtype=float)

        self.names: list[str] = []
        self.highlighted: list[bool] = []
        directions = []
        distances = []
        for star in stars:
            if not star.has_position or star.coordinates == observer.coordinates:
                continue
            if check_star is not None and star.coordinates == check_star.coordinates:
                continue
            offset = np.array(star.position, dtype=float) - origin
            distance = np.linalg.norm(offset)
            self.names.append(star.display_name)
            self.highlighted.append(star.display_name in visible)
            directions.append(offset / distance)
            distances.append(distance)
        self.directions = np.array(directions).reshape(-1, 3)
        self.distances = np.array(distances)

        self.check_direction = None
        if check_star is not None:
            offset = np.array(check_star.position, dtype=float) - origin
            self.check_direction = offset / np.linalg.norm(offset)

        self.camera = self._opening_camera(origin)

        # Marking state
        self.mark_direction: np.ndarray | None = None
        self.mark_radius: float | None = None   # Set once the mark is complete
        self.status = self._prompt()

        # Held movement keys: key -> slow flag
        self._held: dict[str, bool] = {}
        self._released_at: dict[str, float] = {}
        self._last_tick = time.monotonic()

        self._build_figure()
        self._draw_ground()
        self.redraw()

    def _opening_camera(self, origin: np.ndarray) -> MoonCamera:
        """Aim so the highlighted stars are in frame.

        Centers on the midpoint of the two highlighted stars that are
        farthest apart and twists the view to lay that pair along the
        frame's long axis, with the other highlighted stars above it (the
        ground covers the bottom of the frame). With no highlighted stars,
        looks at the cluster center.
        """
        pitch = np.radians(self.settings.tilt_deg)
        chosen = self.directions[np.array(self.highlighted, dtype=bool)]

        if len(chosen) == 0:
            to_center = np.array([500.0, 500.0, 500.0]) - origin
            if np.linalg.norm(to_center) < 1e-6:
                to_center = np.array([0.0, 0.0, 1.0])
            return MoonCamera.looking_at(to_center, pitch)
        if len(chosen) == 1:
            return MoonCamera.looking_at(chosen[0], pitch)

        i, j = np.unravel_index(np.argmin(chosen @ chosen.T), (len(chosen),) * 2)
        a, b = chosen[i], chosen[j]
        camera = MoonCamera.looking_at(a + b, pitch)

        # a and b are symmetric about the view center, so making a's
        # camera_up component zero lays the pair horizontally
        angle = np.arctan2(np.dot(a, camera.camera_up), np.dot(a, camera.right))
        camera.twist(angle)
        if abs(np.dot(a, camera.camera_up)) > 1e-6:
            camera.twist(-2 * angle)

        if np.dot(chosen.mean(axis=0), camera.camera_up) < 0:
            camera.twist(np.pi)
        return camera

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    def _build_figure(self) -> None:
        aspect = self.settings.aspect
        self.fig = plt.figure(figsize=(13, 7), facecolor=BACKGROUND)
        self.ax = self.fig.add_axes([0.02, 0.04, 0.76, 0.92])
        self.ax.set_facecolor('black')
        self.ax.set_xlim(-aspect, aspect)
        self.ax.set_ylim(-1, 1)
        self.ax.set_aspect('equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_color('#444466')

        # Closer stars are drawn larger, as on the ship's map
        sizes = 12 + 70 * np.clip(1 - self.distances / 1200, 0, 1)
        colors = ['#ffd75e' if h else '#c8c8d8' for h in self.highlighted]
        self._scatter = self.ax.scatter(
            np.zeros(len(self.names)), np.zeros(len(self.names)),
            s=sizes, c=colors, zorder=3,
        )
        self._labels = [
            self.ax.text(
                0, 0, name, clip_on=True, zorder=4,
                color='#ffd75e' if h else '#77778a',
                fontsize=9 if h else 7,
                fontweight='bold' if h else 'normal',
            )
            for name, h in zip(self.names, self.highlighted)
        ]

        self.ax.plot(0, 0, '+', color='#444466', markersize=12, zorder=1)
        self._ground = None
        self._sighting_artists: dict[str, tuple] = {}
        self._mark_point, = self.ax.plot([], [], 'x', color='#ff5566',
                                         markersize=11, markeredgewidth=2, zorder=5)
        self._mark_circle, = self.ax.plot([], [], '-', color='#ff5566',
                                          linewidth=1.2, zorder=5)
        self._status_text = self.ax.text(
            0.01, 0.99, "", transform=self.ax.transAxes, va='top',
            color='white', fontsize=9, fontfamily='monospace', zorder=6,
        )

        self._panel_text = self.fig.text(
            0.80, 0.95, "", va='top', color='#aaaabb',
            fontsize=9, fontfamily='monospace',
        )

        canvas = self.fig.canvas
        # Drop matplotlib's own shortcuts (s = save, q = quit, ...)
        handler_id = getattr(canvas.manager, 'key_press_handler_id', None)
        if handler_id is not None:
            canvas.mpl_disconnect(handler_id)
        canvas.mpl_connect('key_press_event', lambda e: self.key_press(e.key))
        canvas.mpl_connect('key_release_event', lambda e: self.key_release(e.key))
        canvas.mpl_connect('button_press_event', self._on_click)
        canvas.mpl_connect('motion_notify_event', self._on_motion)

        self._timer = canvas.new_timer(interval=int(TICK_SECONDS * 1000))
        self._timer.add_callback(self._tick)
        self._timer.start()

    def _to_plot(self, direction: np.ndarray) -> tuple[float, float] | None:
        """Plot coordinates of a world direction, or None if behind the camera."""
        aspect = self.settings.aspect
        x, y, depth = self.camera.project(direction, aspect)
        if depth <= 0:
            return None
        return x * aspect, y

    def _to_direction(self, plot_x: float, plot_y: float) -> np.ndarray:
        aspect = self.settings.aspect
        return self.camera.screen_to_direction(plot_x / aspect, plot_y, aspect)

    def _cone_outline(self, center: np.ndarray, radius: float) -> tuple[list, list]:
        """Plot-space outline of a cone of directions around center."""
        # Two unit vectors perpendicular to center
        helper = self.camera.right if abs(np.dot(center, self.camera.right)) < 0.9 \
            else self.camera.camera_up
        u = np.cross(center, helper)
        u = u / np.linalg.norm(u)
        v = np.cross(center, u)

        xs, ys = [], []
        for t in np.linspace(0, 2 * np.pi, 73):
            direction = np.cos(radius) * center + np.sin(radius) * (np.cos(t) * u + np.sin(t) * v)
            point = self._to_plot(direction)
            if point is None:
                xs.append(np.nan)
                ys.append(np.nan)
            else:
                xs.append(point[0])
                ys.append(point[1])
        return xs, ys

    def _ground_edge(self, plot_x: np.ndarray) -> np.ndarray:
        """Plot y of the moon's edge at each plot x.

        The edge is every direction at a fixed angle below the true horizon.
        It does not depend on where the camera is pointed, only on the tilt.
        """
        tilt = np.radians(self.settings.tilt_deg)
        dip = np.radians(max(self.settings.horizon_below_center_deg - self.settings.tilt_deg, 0))
        half_height = np.tan(vertical_fov() / 2)
        sx = plot_x * half_height

        # Solve (cos(tilt)*sy + sin(tilt)) / |(sx, sy, 1)| = -sin(dip) for sy
        c, s, d = np.cos(tilt), np.sin(tilt), np.sin(dip)
        qa = c * c - d * d
        qb = 2 * c * s
        qc = s * s - d * d * (sx * sx + 1)
        sy = (-qb - np.sqrt(np.maximum(qb * qb - 4 * qa * qc, 0))) / (2 * qa)
        return sy / half_height

    def _draw_ground(self) -> None:
        if self._ground is not None:
            self._ground.remove()
        xs = np.linspace(-self.settings.aspect, self.settings.aspect, 81)
        self._ground = self.ax.fill_between(
            xs, self._ground_edge(xs), -1.0,
            color='#5c5c58', alpha=0.9, linewidth=0, zorder=4.5,
        )

    def redraw(self, preview_radius: float | None = None) -> None:
        """Update the drawing for the current camera and marking state."""
        offsets = np.full((len(self.names), 2), np.nan)
        for i, direction in enumerate(self.directions):
            point = self._to_plot(direction)
            if point is not None:
                offsets[i] = point
                self._labels[i].set_position((point[0] + 0.025, point[1] + 0.02))
            in_frame = (
                point is not None
                and abs(point[0]) <= self.settings.aspect and abs(point[1]) <= 1
            )
            self._labels[i].set_visible(in_frame)
        self._scatter.set_offsets(offsets)

        radius = self.mark_radius if self.mark_radius is not None else preview_radius
        point = self._to_plot(self.mark_direction) if self.mark_direction is not None else None
        if point is None:
            self._mark_point.set_data([], [])
        else:
            self._mark_point.set_data([point[0]], [point[1]])
        if self.mark_direction is not None and radius:
            self._mark_circle.set_data(*self._cone_outline(self.mark_direction, radius))
        else:
            self._mark_circle.set_data([], [])

        self._draw_sightings()
        self._status_text.set_text(self.status)

        up = self.camera.local_up
        lines = [
            f"From {self.observer.display_name}",
            "",
            HELP_TEXT,
            "",
            f"Tilt: {self.settings.tilt_deg:.0f}°",
            f"Standing at (up):",
            f" ({up[0]:+.2f},{up[1]:+.2f},{up[2]:+.2f})",
        ]
        lines += ["", "Marking:", f"{'>' if self.target == ANOMALY else ' '}0 anomaly"]
        if self.unlocated_names:
            lines += ["", "Need to find:"]
            for number, name in enumerate(self.unlocated_names, start=1):
                pointer = '>' if self.target == name else ' '
                key = str(number) if number <= 9 else ' '
                marked = " ✓" if name in self.sightings else ""
                lines.append(f"{pointer}{key} {name}{marked}")
        self._panel_text.set_text("\n".join(lines))

        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def key_press(self, key: str | None) -> None:
        if key is None:
            return
        slow = key.startswith('shift+') or (len(key) == 1 and key.isupper())
        base = key.removeprefix('shift+').lower()

        if base in MOVE_KEYS:
            self._held[base] = slow
            self._released_at.pop(base, None)
        elif base in ('[', ']'):
            self.set_tilt(self.settings.tilt_deg + (1 if base == ']' else -1))
        elif base.isdigit():
            self.select_target(int(base))
        elif base == 'escape':
            self.clear_mark()

    def key_release(self, key: str | None) -> None:
        if key is None:
            return
        base = key.removeprefix('shift+').lower()
        if base in self._held:
            self._released_at[base] = time.monotonic()

    def _tick(self) -> None:
        now = time.monotonic()
        dt = min(now - self._last_tick, 0.1)
        self._last_tick = now

        for key, released in list(self._released_at.items()):
            if now - released > RELEASE_DEBOUNCE_SECONDS:
                self._held.pop(key, None)
                del self._released_at[key]

        if self._held:
            self.advance(dt)

    def advance(self, dt: float) -> None:
        """Apply the held movement keys for dt seconds."""
        rates = {
            'turn': self.settings.turn_rate_deg_s,
            'walk': WALK_RATE_DEG_PER_SEC,
            'twist': TWIST_RATE_DEG_PER_SEC,
        }
        for key, slow in self._held.items():
            movement, sign = MOVE_KEYS[key]
            angle = np.radians(rates[movement]) * dt * sign * (SLOW_FACTOR if slow else 1)
            getattr(self.camera, movement)(angle)
        self.redraw()

    def set_tilt(self, tilt_deg: float) -> None:
        """Change the camera tilt. The view stays; only turning behaves differently."""
        self.settings.tilt_deg = float(np.clip(
            tilt_deg, 0, self.settings.horizon_below_center_deg,
        ))
        self.camera.pitch = np.radians(self.settings.tilt_deg)
        self._draw_ground()
        if self.on_settings_changed:
            self.on_settings_changed(self.settings)
        self.redraw()

    # ------------------------------------------------------------------
    # Marking the anomaly
    # ------------------------------------------------------------------

    def _prompt(self) -> str:
        what = "the anomaly" if self.target == ANOMALY else self.target
        return f"Line up the view, then click {what}"

    def select_target(self, number: int) -> None:
        """Choose what the next mark is of: 0 = anomaly, 1-9 = need-to-find star."""
        if number == 0:
            self.target = ANOMALY
        elif number <= len(self.unlocated_names):
            self.target = self.unlocated_names[number - 1]
        else:
            return
        self.clear_mark()

    def clear_mark(self) -> None:
        self.mark_direction = None
        self.mark_radius = None
        self.status = self._prompt()
        self.redraw()

    def _draw_sightings(self) -> None:
        """Draw saved sightings of unlocated stars as labelled circles."""
        for name, (direction, radius) in self.sightings.items():
            if name not in self._sighting_artists:
                line, = self.ax.plot([], [], '-', color='#55ccdd', linewidth=1.0, zorder=3.5)
                text = self.ax.text(0, 0, name, color='#55ccdd', fontsize=8,
                                    clip_on=True, zorder=3.5)
                self._sighting_artists[name] = (line, text)
            line, text = self._sighting_artists[name]
            line.set_data(*self._cone_outline(direction, radius))
            point = self._to_plot(direction)
            in_frame = (
                point is not None
                and abs(point[0]) <= self.settings.aspect and abs(point[1]) <= 1
            )
            text.set_visible(in_frame)
            if in_frame:
                # Below the center, clear of star labels (above-right of their stars)
                text.set_position((point[0] + 0.025, point[1] - 0.06))

    def click(self, plot_x: float, plot_y: float) -> Observation | None:
        """Handle a click at plot coordinates.

        The first click marks the current target, the second sets the
        uncertainty radius. For the anomaly this completes an observation,
        which is returned; for a need-to-find star it saves a sighting.
        """
        direction = self._to_direction(plot_x, plot_y)

        if self.mark_direction is None or self.mark_radius is not None:
            self.mark_direction = direction
            self.mark_radius = None
            self.status = "Click again to set how far off it could be (Esc to cancel)"
            self.redraw()
            return None

        radius = float(np.arccos(np.clip(np.dot(direction, self.mark_direction), -1, 1)))
        if radius < 1e-4:
            return None  # Double-click on the same spot; wait for a real radius
        self.mark_radius = radius

        if self.target != ANOMALY:
            self.sightings[self.target] = (self.mark_direction.copy(), radius)
            self.status = (
                f"{self.target} sighting saved (±{np.degrees(radius):.1f}°). "
                f"Click to re-mark."
            )
            if self.on_sighting:
                self.on_sighting(self.target, self.mark_direction.copy(), radius)
            # The saved circle replaces the working mark
            self.mark_direction = None
            self.mark_radius = None
            self.redraw()
            return None

        observation = Observation(
            observer_coordinates=self.observer.coordinates,
            direction_vector=tuple(float(c) for c in self.mark_direction),
            cone_half_angle=radius,
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            id=str(uuid.uuid4()),
            view_state={
                "source": "observatory_view",
                "tilt_deg": self.settings.tilt_deg,
                "camera": {
                    "right": self.camera.right.tolist(),
                    "up": self.camera.camera_up.tolist(),
                    "forward": self.camera.forward.tolist(),
                },
            },
        )

        self.status = f"Observation saved (±{np.degrees(radius):.1f}°). Click to re-mark."
        error = self.check_error(observation)
        if error is not None:
            self.status += (
                f"\n{self.check_star.display_name} was {np.degrees(error):.1f}° "
                f"from your mark"
            )
        if self.on_observation:
            self.on_observation(observation)
        self.redraw()
        return observation

    def check_error(self, observation: Observation) -> float | None:
        """Angle in radians between an observation and the check star, if set."""
        if self.check_direction is None:
            return None
        return float(np.arccos(np.clip(
            np.dot(observation.direction, self.check_direction), -1, 1,
        )))

    def _on_click(self, event) -> None:
        if event.inaxes == self.ax and event.xdata is not None:
            self.click(event.xdata, event.ydata)

    def _on_motion(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None:
            return
        if self.mark_direction is None or self.mark_radius is not None:
            return
        direction = self._to_direction(event.xdata, event.ydata)
        radius = float(np.arccos(np.clip(np.dot(direction, self.mark_direction), -1, 1)))
        self.redraw(preview_radius=radius)
