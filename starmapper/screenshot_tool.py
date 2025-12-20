"""Interactive tool for marking stars in screenshots.

This module provides a matplotlib-based interface for clicking on stars
in observatory screenshots to calibrate view parameters.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button, TextBox
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

from .calibration import (
    StarMarker, SphericalCalibration, calibrate_spherical,
    compute_direction, estimate_star_position, project_star,
)
from .coords import Star


@dataclass
class DirectionRay:
    """A direction ray from clicking on an unknown star."""
    pixel_x: float
    pixel_y: float
    direction: np.ndarray  # Unit vector in world coordinates
    estimated_pos: tuple[int, int, int]
    uncertainty: float
    label: str | None = None  # Optional user-provided label


@dataclass
class ScreenshotSession:
    """State for an interactive screenshot marking session."""
    image_path: Path
    observer_pos: tuple[int, int, int]
    known_stars: dict[str, tuple[int, int, int]]  # name -> 3D coords
    markers: list[StarMarker] = field(default_factory=list)
    calibration: SphericalCalibration | None = None
    pending_star_name: str | None = None
    # Estimation mode
    estimation_mode: bool = False
    direction_rays: list[DirectionRay] = field(default_factory=list)
    pending_ray_label: str | None = None

    def add_marker(self, name: str, pixel_x: float, pixel_y: float) -> bool:
        """Add a marker for a known star."""
        if name not in self.known_stars:
            return False

        # Remove existing marker for this star if any
        self.markers = [m for m in self.markers if m.name != name]

        self.markers.append(StarMarker(
            name=name,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            coords_3d=self.known_stars[name]
        ))
        return True

    def calibrate(self, image_width: int, image_height: int) -> bool:
        """Run calibration with current markers."""
        if len(self.markers) < 2:
            return False

        try:
            self.calibration = calibrate_spherical(
                self.markers, self.observer_pos,
                image_width, image_height
            )
            return self.calibration is not None
        except Exception as e:
            print(f"Calibration error: {e}")
            return False


def create_interactive_session(
    image_path: str | Path,
    observer: Star,
    stars: list[Star],
    on_calibration_complete: callable = None,
    on_ray_recorded: callable = None,
) -> tuple[plt.Figure, ScreenshotSession]:
    """Create an interactive session for marking stars in a screenshot.

    Args:
        image_path: Path to the screenshot image
        observer: The Star object for the observatory
        stars: List of all stars (for looking up coordinates)
        on_calibration_complete: Callback when calibration succeeds
        on_ray_recorded: Callback when a direction ray is recorded (estimation mode)

    Returns:
        (figure, session) tuple
    """
    image_path = Path(image_path)

    # Build dict of known star coordinates
    known_stars = {}
    for star in stars:
        if star.has_position and star.coordinates != observer.coordinates:
            name = star.player_name or star.original_name
            if name:
                known_stars[name] = star.position

    session = ScreenshotSession(
        image_path=image_path,
        observer_pos=observer.position,
        known_stars=known_stars,
    )

    # Load and display image
    img = mpimg.imread(str(image_path))
    img_height, img_width = img.shape[:2]

    fig, ax = plt.subplots(figsize=(14, 9))
    plt.subplots_adjust(bottom=0.2)

    ax.imshow(img)
    ax.set_title(f"Click on stars to mark them - {image_path.name}")

    # Text showing instructions and status
    status_text = ax.text(
        0.02, 0.98, "Type star name, then click on it",
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
        color='white'
    )

    # Store references for callbacks
    marker_plots = []
    projected_plots = []
    ray_plots = []

    def update_display():
        """Update the display with current markers and projections."""
        nonlocal marker_plots, projected_plots, ray_plots

        # Clear old plots
        for plot in marker_plots + projected_plots + ray_plots:
            plot.remove()
        marker_plots = []
        projected_plots = []
        ray_plots = []

        # Draw markers (known stars)
        for marker in session.markers:
            plot, = ax.plot(marker.pixel_x, marker.pixel_y, 'go', markersize=12,
                           markerfacecolor='none', markeredgewidth=2)
            marker_plots.append(plot)
            text = ax.text(marker.pixel_x + 10, marker.pixel_y, marker.name,
                          color='lime', fontsize=9, fontweight='bold')
            marker_plots.append(text)

        # If calibrated, show projected positions for known stars
        if session.calibration:
            for marker in session.markers:
                px, py, vis = project_star(
                    np.array(marker.coords_3d, dtype=float),
                    np.array(session.observer_pos, dtype=float),
                    session.calibration,
                    img_width, img_height,
                )
                if vis:
                    plot, = ax.plot(px, py, 'rx', markersize=10, markeredgewidth=2)
                    projected_plots.append(plot)

        # Draw direction rays (unknown stars)
        for ray in session.direction_rays:
            plot, = ax.plot(ray.pixel_x, ray.pixel_y, 'mo', markersize=14,
                           markerfacecolor='none', markeredgewidth=2)
            ray_plots.append(plot)
            label = ray.label or f"({ray.estimated_pos[0]}, {ray.estimated_pos[1]}, {ray.estimated_pos[2]})"
            text = ax.text(ray.pixel_x + 10, ray.pixel_y - 15, label,
                          color='magenta', fontsize=9, fontweight='bold')
            ray_plots.append(text)

        # Update status
        if session.estimation_mode:
            status_lines = ["[ESTIMATION MODE] Click on unknown stars"]
            status_lines.append(f"Direction rays: {len(session.direction_rays)}")
            if session.pending_ray_label:
                status_lines.append(f"Label for next click: {session.pending_ray_label}")
            else:
                status_lines.append("(Type label in textbox, or click directly)")
        else:
            status_lines = ["[CALIBRATION MODE] Mark known stars"]
            status_lines.append(f"Markers: {len(session.markers)} | Known stars: {len(session.known_stars)}")
            if session.pending_star_name:
                status_lines.append(f"Ready to mark: {session.pending_star_name}")

        if session.calibration:
            status_lines.append(f"✓ Calibrated: FOV={session.calibration.fov_horizontal:.1f}°, "
                              f"error={session.calibration.residual_error:.1f}px")

        status_text.set_text('\n'.join(status_lines))
        fig.canvas.draw_idle()

    def on_click(event):
        """Handle mouse click on the image."""
        if event.inaxes != ax:
            return

        if session.estimation_mode:
            # Estimation mode: record direction ray
            if session.calibration is None:
                print("Must calibrate before estimating positions")
                return

            direction = compute_direction(
                event.xdata, event.ydata,
                session.calibration,
                img_width, img_height
            )

            estimated_pos, uncertainty = estimate_star_position(
                direction, session.observer_pos
            )

            ray = DirectionRay(
                pixel_x=event.xdata,
                pixel_y=event.ydata,
                direction=direction,
                estimated_pos=estimated_pos,
                uncertainty=uncertainty,
                label=session.pending_ray_label,
            )
            session.direction_rays.append(ray)
            session.pending_ray_label = None

            print(f"\n★ Direction ray recorded at ({event.xdata:.0f}, {event.ydata:.0f})")
            print(f"  Direction: ({direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f})")
            print(f"  Estimated position: {estimated_pos}")
            print(f"  Uncertainty: ±{uncertainty:.0f} PC")
            if ray.label:
                print(f"  Label: {ray.label}")

            if on_ray_recorded:
                on_ray_recorded(ray, session)

            update_display()

        else:
            # Calibration mode: mark known star
            if session.pending_star_name is None:
                return

            if session.add_marker(session.pending_star_name, event.xdata, event.ydata):
                print(f"Marked {session.pending_star_name} at ({event.xdata:.0f}, {event.ydata:.0f})")
                session.pending_star_name = None

                # Auto-calibrate if we have enough markers
                if len(session.markers) >= 2:
                    if session.calibrate(img_width, img_height):
                        print(f"Calibration: FOV={session.calibration.fov_horizontal:.1f}°, "
                              f"error={session.calibration.residual_error:.1f}px")
                        if on_calibration_complete:
                            on_calibration_complete(session)

                update_display()
            else:
                print(f"Unknown star: {session.pending_star_name}")
                session.pending_star_name = None

    def on_key(event):
        """Handle key press."""
        if event.key == 'escape':
            session.pending_star_name = None
            session.pending_ray_label = None
            update_display()

    # Text input for star name / ray label
    ax_textbox = plt.axes([0.15, 0.05, 0.25, 0.05])
    textbox = TextBox(ax_textbox, 'Name: ', initial='')

    def on_submit(text):
        text = text.strip()
        if not text:
            textbox.set_val('')
            return

        if session.estimation_mode:
            # In estimation mode, text is a label for the next ray
            session.pending_ray_label = text.upper()
            print(f"Next click will be labeled: {session.pending_ray_label}")
        else:
            # In calibration mode, text is a star name to mark
            text_upper = text.upper()
            matches = [name for name in session.known_stars if text_upper in name.upper()]
            if len(matches) == 1:
                session.pending_star_name = matches[0]
                print(f"Click on {matches[0]} to mark it")
            elif len(matches) > 1:
                print(f"Multiple matches: {matches[:5]}...")
                session.pending_star_name = None
            elif text_upper in session.known_stars:
                session.pending_star_name = text_upper
                print(f"Click on {text_upper} to mark it")
            else:
                print(f"Star '{text}' not found in known stars")
                session.pending_star_name = None

        textbox.set_val('')
        update_display()

    textbox.on_submit(on_submit)

    # Mode toggle button
    ax_mode = plt.axes([0.42, 0.05, 0.15, 0.05])
    btn_mode = Button(ax_mode, 'Estimate Mode')

    def on_mode_toggle(event):
        if not session.calibration:
            print("Must calibrate before switching to estimation mode")
            return
        session.estimation_mode = not session.estimation_mode
        session.pending_star_name = None
        session.pending_ray_label = None
        btn_mode.label.set_text('Calibrate Mode' if session.estimation_mode else 'Estimate Mode')
        mode_name = "ESTIMATION" if session.estimation_mode else "CALIBRATION"
        print(f"\nSwitched to {mode_name} mode")
        update_display()

    btn_mode.on_clicked(on_mode_toggle)

    # Calibrate button
    ax_calibrate = plt.axes([0.59, 0.05, 0.12, 0.05])
    btn_calibrate = Button(ax_calibrate, 'Calibrate')

    def on_calibrate(event):
        if session.calibrate(img_width, img_height):
            print(f"Calibration successful!")
            print(session.calibration)
            if on_calibration_complete:
                on_calibration_complete(session)
            update_display()
        else:
            print("Need at least 2 markers to calibrate")

    btn_calibrate.on_clicked(on_calibrate)

    # Clear button
    ax_clear = plt.axes([0.73, 0.05, 0.08, 0.05])
    btn_clear = Button(ax_clear, 'Clear')

    def on_clear(event):
        session.markers = []
        session.calibration = None
        session.pending_star_name = None
        session.estimation_mode = False
        session.direction_rays = []
        session.pending_ray_label = None
        btn_mode.label.set_text('Estimate Mode')
        update_display()

    btn_clear.on_clicked(on_clear)

    # Undo button (remove last ray)
    ax_undo = plt.axes([0.83, 0.05, 0.08, 0.05])
    btn_undo = Button(ax_undo, 'Undo')

    def on_undo(event):
        if session.estimation_mode and session.direction_rays:
            removed = session.direction_rays.pop()
            print(f"Removed ray: {removed.label or 'unlabeled'}")
            update_display()
        elif not session.estimation_mode and session.markers:
            removed = session.markers.pop()
            print(f"Removed marker: {removed.name}")
            # Re-calibrate
            if len(session.markers) >= 2:
                session.calibrate(img_width, img_height)
            else:
                session.calibration = None
            update_display()

    btn_undo.on_clicked(on_undo)

    # Connect events
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    update_display()

    return fig, session


def estimate_from_click(
    session: ScreenshotSession,
    pixel_x: float,
    pixel_y: float,
    image_width: int,
    image_height: int,
    name_bounds: tuple = None,
) -> tuple[tuple[int, int, int], float, np.ndarray]:
    """Estimate a star's position from where it was clicked.

    Args:
        session: Calibrated session
        pixel_x, pixel_y: Click position
        image_width, image_height: Image dimensions
        name_bounds: Optional coordinate bounds from original name

    Returns:
        (estimated_position, uncertainty, direction_vector)
    """
    if session.calibration is None:
        raise ValueError("Session must be calibrated first")

    direction = compute_direction(
        pixel_x, pixel_y, session.calibration,
        image_width, image_height
    )

    position, uncertainty = estimate_star_position(
        direction, session.observer_pos, name_bounds
    )

    return position, uncertainty, direction
