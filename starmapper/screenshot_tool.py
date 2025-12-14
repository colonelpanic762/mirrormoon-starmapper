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
    StarMarker, CalibrationResult, calibrate_from_markers,
    compute_direction_from_pixel, estimate_star_position, project_star_to_image
)
from .coords import Star


@dataclass
class ScreenshotSession:
    """State for an interactive screenshot marking session."""
    image_path: Path
    observer_pos: tuple[int, int, int]
    known_stars: dict[str, tuple[int, int, int]]  # name -> 3D coords
    markers: list[StarMarker] = field(default_factory=list)
    calibration: CalibrationResult | None = None
    pending_star_name: str | None = None

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
            self.calibration = calibrate_from_markers(
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
) -> tuple[plt.Figure, ScreenshotSession]:
    """Create an interactive session for marking stars in a screenshot.

    Args:
        image_path: Path to the screenshot image
        observer: The Star object for the observatory
        stars: List of all stars (for looking up coordinates)
        on_calibration_complete: Callback when calibration succeeds

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

    def update_display():
        """Update the display with current markers and projections."""
        nonlocal marker_plots, projected_plots

        # Clear old marker plots
        for plot in marker_plots:
            plot.remove()
        marker_plots = []

        for plot in projected_plots:
            plot.remove()
        projected_plots = []

        # Draw markers
        for marker in session.markers:
            plot, = ax.plot(marker.pixel_x, marker.pixel_y, 'go', markersize=12,
                           markerfacecolor='none', markeredgewidth=2)
            marker_plots.append(plot)
            text = ax.text(marker.pixel_x + 10, marker.pixel_y, marker.name,
                          color='lime', fontsize=9, fontweight='bold')
            marker_plots.append(text)

        # If calibrated, show projected positions
        if session.calibration:
            for marker in session.markers:
                px, py, vis = project_star_to_image(
                    np.array(marker.coords_3d, dtype=float),
                    np.array(session.observer_pos, dtype=float),
                    session.calibration.azimuth,
                    session.calibration.elevation,
                    session.calibration.rotation,
                    np.radians(session.calibration.fov_horizontal),
                    img_width, img_height,
                    session.calibration.flip_horizontal,
                    session.calibration.flip_vertical
                )
                if vis:
                    plot, = ax.plot(px, py, 'rx', markersize=10, markeredgewidth=2)
                    projected_plots.append(plot)

        # Update status
        status_lines = [f"Markers: {len(session.markers)} | Known stars: {len(session.known_stars)}"]
        if session.pending_star_name:
            status_lines.append(f"Ready to mark: {session.pending_star_name}")
        if session.calibration:
            status_lines.append(f"Calibrated: FOV={session.calibration.fov_horizontal:.1f}°, "
                              f"error={session.calibration.residual_error:.1f}px")
        status_text.set_text('\n'.join(status_lines))

        fig.canvas.draw_idle()

    def on_click(event):
        """Handle mouse click on the image."""
        if event.inaxes != ax:
            return
        if session.pending_star_name is None:
            return

        # Add marker
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
            update_display()

    # Text input for star name
    ax_textbox = plt.axes([0.15, 0.05, 0.3, 0.05])
    textbox = TextBox(ax_textbox, 'Star name: ', initial='')

    def on_submit(text):
        text = text.strip().upper()
        if text:
            # Check if it matches a known star
            matches = [name for name in session.known_stars if text in name.upper()]
            if len(matches) == 1:
                session.pending_star_name = matches[0]
                print(f"Click on {matches[0]} to mark it")
            elif len(matches) > 1:
                print(f"Multiple matches: {matches[:5]}...")
                session.pending_star_name = None
            elif text in session.known_stars:
                session.pending_star_name = text
                print(f"Click on {text} to mark it")
            else:
                print(f"Star '{text}' not found in known stars")
                session.pending_star_name = None
        textbox.set_val('')
        update_display()

    textbox.on_submit(on_submit)

    # Calibrate button
    ax_calibrate = plt.axes([0.5, 0.05, 0.15, 0.05])
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
    ax_clear = plt.axes([0.7, 0.05, 0.1, 0.05])
    btn_clear = Button(ax_clear, 'Clear')

    def on_clear(event):
        session.markers = []
        session.calibration = None
        session.pending_star_name = None
        update_display()

    btn_clear.on_clicked(on_clear)

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

    direction = compute_direction_from_pixel(
        pixel_x, pixel_y, session.calibration,
        image_width, image_height
    )

    position, uncertainty = estimate_star_position(
        direction, session.observer_pos, name_bounds
    )

    return position, uncertainty, direction
