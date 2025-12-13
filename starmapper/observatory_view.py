"""2D star chart visualization from observatory viewpoint.

Renders a stereographic projection of the sky as seen from an observatory,
matching the in-game MirrorMoon EP aesthetic with a dark background and
white/colored stars.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Any
import uuid

from .coords import Star, Observation
from .projection import (
    project_stars_from_observer,
    chart_to_direction,
    ChartPosition,
    ViewState,
)


class ClickState(Enum):
    """State machine for interactive observation mode."""
    WAITING_FIRST_CLICK = auto()   # Ready for anomaly direction click
    WAITING_SECOND_CLICK = auto()  # Ready for uncertainty radius click
    OBSERVATION_COMPLETE = auto()  # Done, observation recorded


@dataclass
class InteractiveState:
    """Tracks state during interactive observation mode."""
    click_state: ClickState = ClickState.WAITING_FIRST_CLICK
    first_click_pos: tuple[float, float] | None = None
    first_click_direction: np.ndarray | None = None
    preview_radius: float = 0.0
    observation: Observation | None = None

    # Visual elements for cleanup
    markers: list[Any] = field(default_factory=list)
    preview_circle: Any = None
    status_text: Any = None


def plot_observatory_view(
    observer: Star,
    stars: list[Star],
    center_azimuth: float = 0.0,
    center_elevation: float = np.pi / 4,
    rotation: float = 0.0,
    show_labels: bool = True,
    anomaly_position: tuple[float, float] | None = None,
    fov_radius: float = 2.0,
    figsize: tuple[int, int] = (12, 10),
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
) -> plt.Figure:
    """Create a 2D star chart showing the sky from an observatory.

    Args:
        observer: Observatory star location
        stars: All stars to potentially display
        center_azimuth: Horizontal viewing direction (radians)
        center_elevation: Vertical viewing direction (radians)
        rotation: Roll rotation of the view (radians)
        show_labels: Whether to display star labels
        anomaly_position: Optional (x, y) chart position of marked anomaly
        fov_radius: Field of view radius in projection units
        figsize: Figure size in inches

    Returns:
        matplotlib Figure object
    """
    # Project all stars
    projected = project_stars_from_observer(
        observer, stars, center_azimuth, center_elevation, rotation
    )

    # Apply flips if requested (models viewing same sky from different moon positions)
    if flip_horizontal or flip_vertical:
        projected = [
            (star, ChartPosition(
                x=-pos.x if flip_horizontal else pos.x,
                y=-pos.y if flip_vertical else pos.y,
                visible=pos.visible,
                distance=pos.distance
            ))
            for star, pos in projected
        ]

    # Create figure with dark background
    fig, ax = plt.subplots(figsize=figsize, facecolor='#0d0d1a')
    ax.set_facecolor('#0d0d1a')

    # Plot stars
    _plot_stars(ax, projected, show_labels, fov_radius)

    # Plot anomaly marker if specified
    if anomaly_position is not None:
        _plot_anomaly(ax, anomaly_position)

    # Plot a subtle circular border for the field of view
    _plot_fov_border(ax, fov_radius)

    # Style the chart
    _style_chart(ax, observer, fov_radius, center_azimuth, center_elevation, rotation)

    plt.tight_layout()
    return fig


def _plot_stars(
    ax: plt.Axes,
    projected_stars: list[tuple[Star, ChartPosition]],
    show_labels: bool,
    fov_radius: float,
) -> None:
    """Plot stars on the chart with size based on distance."""
    for star, pos in projected_stars:
        # Skip stars outside the field of view or behind observer
        if not pos.visible:
            continue
        if abs(pos.x) > fov_radius * 1.2 or abs(pos.y) > fov_radius * 1.2:
            continue

        # Size inversely proportional to distance (with limits)
        # Closer stars appear larger
        base_size = np.clip(800 / (pos.distance + 100), 10, 80)

        # Determine color and marker based on star type
        if star.has_observatory:
            color = '#ff4444'  # Red for observatories
            marker = '*'
            size = base_size * 2
            label_color = '#ff6666'
            fontweight = 'bold'
        elif getattr(star, 'has_detector', False):
            color = '#ffdd44'  # Yellow for detectors
            marker = '^'
            size = base_size * 1.5
            label_color = '#ffee88'
            fontweight = 'bold'
        elif star.discovered_by_user:
            color = '#44ff44'  # Green for discoveries
            marker = 'o'
            size = base_size * 1.2
            label_color = '#88ff88'
            fontweight = 'normal'
        else:
            color = '#ffffff'  # White for regular stars
            marker = 'o'
            size = base_size
            label_color = '#cccccc'
            fontweight = 'normal'

        # Plot the star
        ax.scatter(
            pos.x, pos.y,
            s=size,
            c=color,
            marker=marker,
            edgecolors='white',
            linewidths=0.3,
            alpha=0.9,
            zorder=2
        )

        # Add label if enabled
        if show_labels:
            ax.annotate(
                star.display_name,
                (pos.x, pos.y),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                color=label_color,
                fontweight=fontweight,
                alpha=0.9,
            )


def _plot_anomaly(ax: plt.Axes, position: tuple[float, float]) -> None:
    """Plot the anomaly as a circle outline (matching in-game appearance)."""
    x, y = position
    circle = patches.Circle(
        (x, y),
        radius=0.08,
        fill=False,
        color='white',
        linewidth=2,
        zorder=3
    )
    ax.add_patch(circle)

    # Add a subtle glow effect
    glow = patches.Circle(
        (x, y),
        radius=0.1,
        fill=False,
        color='white',
        linewidth=1,
        alpha=0.3,
        zorder=2
    )
    ax.add_patch(glow)


def _plot_fov_border(ax: plt.Axes, fov_radius: float) -> None:
    """Draw a subtle circular border for the field of view."""
    circle = patches.Circle(
        (0, 0),
        radius=fov_radius,
        fill=False,
        color='#333355',
        linewidth=1,
        linestyle='--',
        alpha=0.5,
        zorder=1
    )
    ax.add_patch(circle)


def _style_chart(
    ax: plt.Axes,
    observer: Star,
    fov_radius: float,
    azimuth: float,
    elevation: float,
    rotation: float,
) -> None:
    """Apply visual styling to the chart."""
    # Set axis limits with small margin
    margin = fov_radius * 0.1
    ax.set_xlim(-fov_radius - margin, fov_radius + margin)
    ax.set_ylim(-fov_radius - margin, fov_radius + margin)
    ax.set_aspect('equal')

    # Hide axes for cleaner look
    ax.axis('off')

    # Add title with view info
    az_deg = np.degrees(azimuth)
    el_deg = np.degrees(elevation)
    rot_deg = np.degrees(rotation)

    title = f"Observatory View: {observer.display_name}"
    subtitle = f"Az: {az_deg:.0f}°  El: {el_deg:.0f}°  Rot: {rot_deg:.0f}°"

    ax.set_title(
        f"{title}\n{subtitle}",
        color='white',
        fontsize=12,
        pad=10
    )

    # Add coordinate info at bottom
    if observer.coordinates:
        coord_text = f"Observer position: ({observer.position[0]:.0f}, {observer.position[1]:.0f}, {observer.position[2]:.0f})"
        ax.text(
            0.5, -0.02,
            coord_text,
            transform=ax.transAxes,
            ha='center',
            va='top',
            fontsize=9,
            color='#666688'
        )


def show_observatory_view(
    observer: Star,
    stars: list[Star],
    **kwargs
) -> None:
    """Plot and display the observatory view interactively."""
    plot_observatory_view(observer, stars, **kwargs)
    plt.show()


def _calculate_angular_distance(
    pos1: tuple[float, float],
    pos2: tuple[float, float],
    azimuth: float,
    elevation: float,
    rotation: float,
    flip_horizontal: bool,
    flip_vertical: bool,
) -> float:
    """Calculate angular distance between two chart positions in radians.

    Converts both chart positions to 3D direction vectors, then computes
    the angle between them using the dot product.
    """
    # Account for flips when converting back to directions
    x1, y1 = pos1
    x2, y2 = pos2
    if flip_horizontal:
        x1, x2 = -x1, -x2
    if flip_vertical:
        y1, y2 = -y1, -y2

    dir1 = chart_to_direction(x1, y1, azimuth, elevation, rotation)
    dir2 = chart_to_direction(x2, y2, azimuth, elevation, rotation)

    # Angle between unit vectors
    dot = np.clip(np.dot(dir1, dir2), -1.0, 1.0)
    return float(np.arccos(dot))


def _update_status_text(ax: plt.Axes, state: InteractiveState, text: str) -> None:
    """Update the status text displayed on the chart."""
    if state.status_text is not None:
        state.status_text.remove()

    state.status_text = ax.text(
        0.5, 0.02,
        text,
        transform=ax.transAxes,
        ha='center',
        va='bottom',
        fontsize=10,
        color='#88ff88',
        bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#88ff88', alpha=0.8)
    )


def _add_crosshair_marker(ax: plt.Axes, state: InteractiveState, x: float, y: float) -> None:
    """Add a crosshair marker at the clicked position."""
    size = 0.1
    # Horizontal line
    h_line = ax.plot([x - size, x + size], [y, y], color='#88ff88', linewidth=2, zorder=10)[0]
    # Vertical line
    v_line = ax.plot([x, x], [y - size, y + size], color='#88ff88', linewidth=2, zorder=10)[0]
    # Center dot
    dot = ax.scatter([x], [y], c='#88ff88', s=30, zorder=11)

    state.markers.extend([h_line, v_line, dot])


def _update_preview_circle(ax: plt.Axes, state: InteractiveState, radius: float) -> None:
    """Update the preview circle showing uncertainty radius."""
    if state.preview_circle is not None:
        state.preview_circle.remove()

    if state.first_click_pos is not None and radius > 0:
        x, y = state.first_click_pos
        state.preview_circle = patches.Circle(
            (x, y),
            radius=radius,
            fill=False,
            color='#ffff88',
            linewidth=1.5,
            linestyle='--',
            alpha=0.7,
            zorder=9
        )
        ax.add_patch(state.preview_circle)


def create_interactive_observatory_view(
    observer: Star,
    stars: list[Star],
    center_azimuth: float = 0.0,
    center_elevation: float = np.pi / 4,
    rotation: float = 0.0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    fov_radius: float = 2.0,
    on_observation_complete: Callable[[Observation], None] | None = None,
) -> tuple[plt.Figure, InteractiveState]:
    """Create an interactive observatory view for recording observations.

    Two-click workflow:
    1. First click marks the anomaly direction (center of uncertainty cone)
    2. Second click sets the uncertainty radius (cone half-angle)

    Args:
        observer: Observatory star location
        stars: All stars to potentially display
        center_azimuth: Horizontal viewing direction (radians)
        center_elevation: Vertical viewing direction (radians)
        rotation: Roll rotation of the view (radians)
        flip_horizontal: Mirror the view horizontally
        flip_vertical: Mirror the view vertically
        fov_radius: Field of view radius in projection units
        on_observation_complete: Callback when observation is recorded

    Returns:
        Tuple of (figure, interactive_state)
    """
    # Create the base view
    fig = plot_observatory_view(
        observer=observer,
        stars=stars,
        center_azimuth=center_azimuth,
        center_elevation=center_elevation,
        rotation=rotation,
        show_labels=True,
        fov_radius=fov_radius,
        flip_horizontal=flip_horizontal,
        flip_vertical=flip_vertical,
    )

    ax = fig.axes[0]
    state = InteractiveState()

    # Store view parameters for direction calculation
    view_params = {
        'azimuth': center_azimuth,
        'elevation': center_elevation,
        'rotation': rotation,
        'flip_horizontal': flip_horizontal,
        'flip_vertical': flip_vertical,
    }

    # Initial status text
    _update_status_text(ax, state, "Click to mark anomaly center")

    def on_click(event):
        """Handle mouse click events."""
        # Ignore clicks when toolbar tools (zoom/pan) are active
        toolbar = fig.canvas.manager.toolbar
        if toolbar is not None and toolbar.mode != '':
            return

        # Only respond to left-click
        if event.button != 1:
            return

        if event.inaxes != ax:
            return

        x, y = event.xdata, event.ydata

        if state.click_state == ClickState.WAITING_FIRST_CLICK:
            # First click: mark anomaly direction
            state.first_click_pos = (x, y)

            # Convert chart position to direction vector (accounting for flips)
            chart_x = -x if flip_horizontal else x
            chart_y = -y if flip_vertical else y
            state.first_click_direction = chart_to_direction(
                chart_x, chart_y, center_azimuth, center_elevation, rotation
            )

            # Visual feedback
            _add_crosshair_marker(ax, state, x, y)
            _update_status_text(ax, state, "Click to set uncertainty radius")

            state.click_state = ClickState.WAITING_SECOND_CLICK
            fig.canvas.draw_idle()

        elif state.click_state == ClickState.WAITING_SECOND_CLICK:
            # Second click: set uncertainty radius
            angular_distance = _calculate_angular_distance(
                state.first_click_pos,
                (x, y),
                **view_params
            )

            # Create the observation
            observation = Observation(
                id=str(uuid.uuid4()),
                observer_coordinates=observer.coordinates,
                direction_vector=tuple(state.first_click_direction),
                cone_half_angle=angular_distance,
                timestamp=datetime.utcnow().isoformat() + "Z",
                view_state={
                    'azimuth': center_azimuth,
                    'elevation': center_elevation,
                    'rotation': rotation,
                    'flip_horizontal': flip_horizontal,
                    'flip_vertical': flip_vertical,
                }
            )

            state.observation = observation
            state.click_state = ClickState.OBSERVATION_COMPLETE

            # Final visual: solid circle for recorded observation
            if state.preview_circle is not None:
                state.preview_circle.remove()

            final_circle = patches.Circle(
                state.first_click_pos,
                radius=_angular_to_chart_radius(angular_distance, fov_radius),
                fill=False,
                color='#88ff88',
                linewidth=2,
                zorder=9
            )
            ax.add_patch(final_circle)
            state.markers.append(final_circle)

            # Update status
            degrees = np.degrees(angular_distance)
            _update_status_text(
                ax, state,
                f"✓ Observation recorded (cone half-angle: {degrees:.1f}°)"
            )

            fig.canvas.draw_idle()

            # Callback
            if on_observation_complete is not None:
                on_observation_complete(observation)

    def on_motion(event):
        """Handle mouse motion for preview circle."""
        if state.click_state != ClickState.WAITING_SECOND_CLICK:
            return
        if event.inaxes != ax:
            return

        x, y = event.xdata, event.ydata

        # Calculate angular distance from first click
        angular_distance = _calculate_angular_distance(
            state.first_click_pos,
            (x, y),
            **view_params
        )

        # Convert angular distance to chart radius for preview
        chart_radius = _angular_to_chart_radius(angular_distance, fov_radius)
        _update_preview_circle(ax, state, chart_radius)

        # Update status with current angle
        degrees = np.degrees(angular_distance)
        _update_status_text(ax, state, f"Uncertainty radius: {degrees:.1f}° (click to confirm)")

        fig.canvas.draw_idle()

    # Connect event handlers
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    return fig, state


def _angular_to_chart_radius(angular_distance: float, fov_radius: float) -> float:
    """Convert angular distance (radians) to chart radius.

    This is approximate - assumes roughly linear mapping near center.
    For stereographic projection, small angles map approximately linearly.
    """
    # The FOV radius of 2.0 corresponds to roughly 90 degrees (pi/2)
    # So 1 radian ≈ 2.0 / (pi/2) ≈ 1.27 chart units
    return angular_distance * (fov_radius / (np.pi / 2))
