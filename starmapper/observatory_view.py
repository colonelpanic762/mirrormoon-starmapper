"""2D star chart visualization from observatory viewpoint.

Renders a stereographic projection of the sky as seen from an observatory,
matching the in-game MirrorMoon EP aesthetic with a dark background and
white/colored stars.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from .coords import Star
from .projection import (
    project_stars_from_observer,
    ChartPosition,
    ViewState,
)


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
