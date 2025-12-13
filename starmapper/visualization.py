"""3D visualization of the star map using matplotlib."""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from .coords import Star, Observation
from .triangulation import TriangulationResult


def plot_star_map(stars: list[Star],
                  show_labels: bool = True,
                  label_observatories: bool = True,
                  figsize: tuple[int, int] = (12, 10)) -> plt.Figure:
    """Create a 3D scatter plot of stars.

    Args:
        stars: List of Star objects to plot
        show_labels: Whether to show labels on all stars
        label_observatories: Always label observatory stars (even if show_labels=False)
        figsize: Figure size in inches

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Separate stars by category for different colors
    regular_stars = []
    discovered_stars = []
    observatory_stars = []
    detector_stars = []

    for star in stars:
        if not star.has_position:
            continue  # Skip stars without coordinates

        pos = star.position

        if star.has_observatory:
            observatory_stars.append((pos, star))
        elif getattr(star, 'has_detector', False):
            detector_stars.append((pos, star))
        elif star.discovered_by_user:
            discovered_stars.append((pos, star))
        else:
            regular_stars.append((pos, star))

    # Plot each category with different colors/sizes
    if regular_stars:
        positions = np.array([s[0] for s in regular_stars])
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                   c='lightblue', s=20, alpha=0.6, label='Known stars')
        if show_labels:
            for pos, star in regular_stars:
                ax.text(pos[0], pos[1], pos[2], f'  {star.display_name}',
                        fontsize=7, alpha=0.7)

    if discovered_stars:
        positions = np.array([s[0] for s in discovered_stars])
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                   c='lime', s=40, alpha=0.8, label='Your discoveries')
        if show_labels:
            for pos, star in discovered_stars:
                ax.text(pos[0], pos[1], pos[2], f'  {star.display_name}',
                        fontsize=8, color='darkgreen')

    if detector_stars:
        positions = np.array([s[0] for s in detector_stars])
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                   c='yellow', s=60, marker='^', alpha=0.9, label='Detectors')
        if show_labels or label_observatories:
            for pos, star in detector_stars:
                ax.text(pos[0], pos[1], pos[2], f'  {star.display_name}',
                        fontsize=9, color='orange', fontweight='bold')

    if observatory_stars:
        positions = np.array([s[0] for s in observatory_stars])
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                   c='red', s=100, marker='*', alpha=1.0, label='Observatories')
        if show_labels or label_observatories:
            for pos, star in observatory_stars:
                ax.text(pos[0], pos[1], pos[2], f'  {star.display_name}',
                        fontsize=10, color='darkred', fontweight='bold')

    # Draw bounding box for the galaxy (0-999 on each axis)
    _draw_bounding_box(ax)

    # Labels and formatting
    ax.set_xlabel('X (PC)')
    ax.set_ylabel('Y (PC)')
    ax.set_zlabel('Z (PC)')
    ax.set_title('MirrorMoon EP - Season 123 Star Map')
    ax.legend(loc='upper left')

    # Set axis limits to full galaxy range
    ax.set_xlim(0, 999)
    ax.set_ylim(0, 999)
    ax.set_zlim(0, 999)

    return fig


def _draw_bounding_box(ax, color='gray', alpha=0.3):
    """Draw wireframe bounding box for the galaxy cube."""
    # Define the 8 corners of the cube
    corners = np.array([
        [0, 0, 0], [999, 0, 0], [999, 999, 0], [0, 999, 0],
        [0, 0, 999], [999, 0, 999], [999, 999, 999], [0, 999, 999]
    ])

    # Define edges as pairs of corner indices
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]

    for i, j in edges:
        ax.plot3D(*zip(corners[i], corners[j]), color=color, alpha=alpha, linewidth=0.5)


def show_map(stars: list[Star], **kwargs):
    """Plot and display the star map interactively."""
    plot_star_map(stars, **kwargs)
    plt.show()


def plot_observation_cones(
    ax: Axes3D,
    observations: list[Observation],
    cone_length: float = 500.0,
    segments: int = 24,
) -> None:
    """Draw wireframe cones representing observation directions and uncertainties.

    Args:
        ax: 3D matplotlib axes
        observations: List of observations to visualize
        cone_length: Length of cone from apex (observer) in coordinate units
        segments: Number of segments for the circular base
    """
    for i, obs in enumerate(observations):
        origin = obs.origin
        direction = obs.direction
        half_angle = obs.cone_half_angle

        # Generate cone wireframe
        _draw_cone(ax, origin, direction, half_angle, cone_length, segments,
                   color=_get_observation_color(i), alpha=0.6)


def _draw_cone(
    ax: Axes3D,
    apex: np.ndarray,
    direction: np.ndarray,
    half_angle: float,
    length: float,
    segments: int,
    color: str = 'cyan',
    alpha: float = 0.5,
) -> None:
    """Draw a wireframe cone in 3D.

    The cone has its apex at 'apex', extends in 'direction' for 'length',
    with opening angle 'half_angle'.
    """
    # Create orthonormal basis with direction as z-axis
    # Find a vector not parallel to direction
    if abs(direction[0]) < 0.9:
        perp1 = np.cross(direction, np.array([1, 0, 0]))
    else:
        perp1 = np.cross(direction, np.array([0, 1, 0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)

    # Radius at the base of the cone
    radius = length * np.tan(half_angle)

    # Generate points on the circular base
    angles = np.linspace(0, 2 * np.pi, segments + 1)
    base_center = apex + direction * length

    base_points = []
    for angle in angles:
        point = base_center + radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
        base_points.append(point)

    base_points = np.array(base_points)

    # Draw the circular base
    ax.plot3D(base_points[:, 0], base_points[:, 1], base_points[:, 2],
              color=color, alpha=alpha, linewidth=1)

    # Draw lines from apex to base (generators)
    for i in range(0, segments, max(1, segments // 8)):
        ax.plot3D([apex[0], base_points[i, 0]],
                  [apex[1], base_points[i, 1]],
                  [apex[2], base_points[i, 2]],
                  color=color, alpha=alpha * 0.7, linewidth=0.5)

    # Draw the central ray
    ax.plot3D([apex[0], base_center[0]],
              [apex[1], base_center[1]],
              [apex[2], base_center[2]],
              color=color, alpha=alpha, linewidth=1.5, linestyle='--')


def plot_triangulation_result(
    ax: Axes3D,
    result: TriangulationResult,
) -> None:
    """Mark the estimated anomaly position with a marker and confidence sphere.

    Args:
        ax: 3D matplotlib axes
        result: Triangulation result to visualize
    """
    pos = result.estimated_position

    # Plot the estimated position as a prominent X marker
    ax.scatter([pos[0]], [pos[1]], [pos[2]],
               c='magenta', s=200, marker='X', alpha=1.0,
               label=f'Estimated: {result.coordinate_string}', zorder=10)

    # Draw confidence sphere as wireframe
    _draw_sphere_wireframe(ax, pos, result.confidence_radius,
                           color='magenta', alpha=0.3)

    # Add text label
    ax.text(pos[0], pos[1], pos[2] + result.confidence_radius + 20,
            f'  {result.coordinate_string}\n  ±{result.confidence_radius:.0f}',
            fontsize=9, color='magenta', fontweight='bold')


def _draw_sphere_wireframe(
    ax: Axes3D,
    center: np.ndarray,
    radius: float,
    color: str = 'magenta',
    alpha: float = 0.3,
    resolution: int = 16,
) -> None:
    """Draw a wireframe sphere."""
    # Generate sphere wireframe
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution // 2)

    # Draw latitude lines
    for vi in v:
        x = center[0] + radius * np.sin(vi) * np.cos(u)
        y = center[1] + radius * np.sin(vi) * np.sin(u)
        z = center[2] + radius * np.cos(vi) * np.ones_like(u)
        ax.plot3D(x, y, z, color=color, alpha=alpha, linewidth=0.5)

    # Draw longitude lines
    for ui in u[::4]:  # Every 4th longitude line
        x = center[0] + radius * np.sin(v) * np.cos(ui)
        y = center[1] + radius * np.sin(v) * np.sin(ui)
        z = center[2] + radius * np.cos(v)
        ax.plot3D(x, y, z, color=color, alpha=alpha, linewidth=0.5)


def _get_observation_color(index: int) -> str:
    """Get a distinct color for each observation."""
    colors = ['cyan', 'yellow', 'lime', 'orange', 'pink', 'lightblue']
    return colors[index % len(colors)]


def plot_star_map_with_observations(
    stars: list[Star],
    observations: list[Observation] | None = None,
    result: TriangulationResult | None = None,
    show_labels: bool = True,
    figsize: tuple[int, int] = (14, 12),
) -> plt.Figure:
    """Create a combined 3D view with stars, observation cones, and triangulation result.

    Args:
        stars: List of Star objects
        observations: Optional list of observations to show as cones
        result: Optional triangulation result to mark
        show_labels: Whether to show star labels
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig = plot_star_map(stars, show_labels=show_labels, figsize=figsize)
    ax = fig.axes[0]

    if observations:
        plot_observation_cones(ax, observations)

    if result:
        plot_triangulation_result(ax, result)
        ax.legend(loc='upper left')

    # Update title
    title = 'MirrorMoon EP - Star Map'
    if observations:
        title += f' ({len(observations)} observations)'
    if result:
        title += ' [TRIANGULATED]'
    ax.set_title(title)

    return fig
