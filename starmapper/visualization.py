"""3D visualization of the star map using matplotlib."""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from .coords import Star


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
