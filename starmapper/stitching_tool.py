"""Interactive tool for multi-image stitching.

This module provides a matplotlib-based interface for marking stars
across multiple screenshots to enable coordinate prediction via stitching.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button, TextBox
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .stitching import (
    StitchingSession, ImageMarking, CorrespondenceMarker,
    StitchingResult, solve_stitching, estimate_star_from_stitching,
)
from .calibration import (
    StarMarker, spherical_to_cartesian, project_star_raw,
)
from .coords import Star


@dataclass
class StitchingToolState:
    """UI state for the stitching tool."""
    current_image_idx: int = 0
    mode: str = "known"  # "known" or "correspondence"
    pending_name: str | None = None
    show_projections: bool = True
    result: StitchingResult | None = None


def create_stitching_tool(
    image_paths: list[str | Path],
    observer: Star,
    stars: list[Star],
    on_result: Callable[[StitchingResult], None] | None = None,
) -> tuple[plt.Figure, StitchingSession, StitchingToolState]:
    """Create an interactive multi-image stitching tool.

    Args:
        image_paths: List of screenshot paths to stitch
        observer: The Star object for the observatory
        stars: List of all stars (for looking up coordinates)
        on_result: Callback when stitching is solved

    Returns:
        (figure, session, state) tuple
    """
    image_paths = [Path(p) for p in image_paths]

    if not image_paths:
        raise ValueError("Need at least one image")

    # Build dict of known star coordinates
    known_stars = {}
    for star in stars:
        if star.has_position and star.coordinates != observer.coordinates:
            name = star.player_name or star.original_name
            if name:
                known_stars[name] = star.position

    session = StitchingSession(
        observer_pos=observer.position,
        known_stars=known_stars,
    )

    state = StitchingToolState()

    # Load all images and add to session
    images_data = []
    for path in image_paths:
        img = mpimg.imread(str(path))
        height, width = img.shape[:2]
        session.add_image(path, width, height)
        images_data.append(img)

    # Create figure with space for controls
    fig, ax = plt.subplots(figsize=(16, 10))
    plt.subplots_adjust(bottom=0.25, right=0.75)

    # Display first image
    img_display = ax.imshow(images_data[0])
    ax.set_title(f"Image 1/{len(image_paths)}: {image_paths[0].name}")

    # Status text
    status_text = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
        color='white'
    )

    # Results panel (right side) - use fixed limits to prevent auto-scaling errors
    ax_results = plt.axes([0.77, 0.25, 0.21, 0.70])
    ax_results.set_facecolor('#1a1a1a')
    ax_results.set_xlim(0, 1)
    ax_results.set_ylim(0, 1)
    ax_results.set_autoscale_on(False)
    ax_results.set_xticks([])
    ax_results.set_yticks([])
    for spine in ax_results.spines.values():
        spine.set_visible(False)
    ax_results.set_title("Stitching Results", color='white', fontsize=10)
    results_text = ax_results.text(
        0.05, 0.95, "No solution yet.\n\nMark stars, then click Solve.",
        transform=ax_results.transAxes, fontsize=8,
        verticalalignment='top', fontfamily='monospace',
        color='#aaaaaa', wrap=True,
    )

    # Store plot objects for cleanup
    marker_plots = []

    def update_display():
        """Update the display for current image."""
        nonlocal marker_plots

        # Clear old plots
        for plot in marker_plots:
            plot.remove()
        marker_plots = []

        idx = state.current_image_idx
        img_marking = session.images[idx]

        # Update image
        img_display.set_data(images_data[idx])
        ax.set_title(f"Image {idx + 1}/{len(image_paths)}: {image_paths[idx].name}")

        # Draw known star markers (green circles)
        for m in img_marking.known_markers:
            plot, = ax.plot(m.pixel_x, m.pixel_y, 'go', markersize=14,
                           markerfacecolor='none', markeredgewidth=2)
            marker_plots.append(plot)
            text = ax.text(m.pixel_x + 12, m.pixel_y, m.name,
                          color='lime', fontsize=9, fontweight='bold')
            marker_plots.append(text)

        # Draw correspondence markers (cyan diamonds)
        for m in img_marking.correspondence_markers:
            plot, = ax.plot(m.pixel_x, m.pixel_y, 'cD', markersize=12,
                           markerfacecolor='none', markeredgewidth=2)
            marker_plots.append(plot)
            text = ax.text(m.pixel_x + 12, m.pixel_y - 12, m.label,
                          color='cyan', fontsize=9, fontweight='bold')
            marker_plots.append(text)

        # If we have a result, show projected positions using spherical model
        if state.result and state.show_projections:
            best = state.result.best_solution()
            if best and idx < len(best.calibrations):
                cal = best.calibrations[idx]
                obs_pos = np.array(session.observer_pos, dtype=float)

                # Project known stars using spherical model
                for m in img_marking.known_markers:
                    star_pos = np.array(m.coords_3d, dtype=float)
                    px, py, vis = project_star_raw(
                        star_pos, obs_pos,
                        cal.up_theta, cal.up_phi, cal.yaw,
                        np.radians(cal.fov_horizontal),
                        img_marking.image_width, img_marking.image_height,
                    )
                    if vis:
                        plot, = ax.plot(px, py, 'rx', markersize=10, markeredgewidth=2)
                        marker_plots.append(plot)

        # Update status
        status_lines = []
        mode_str = "KNOWN STARS" if state.mode == "known" else "CORRESPONDENCES"
        status_lines.append(f"[{mode_str}] Image {idx + 1}/{len(image_paths)}")
        status_lines.append(f"Known markers: {len(img_marking.known_markers)}")
        status_lines.append(f"Correspondences: {len(img_marking.correspondence_markers)}")

        if state.pending_name:
            status_lines.append(f"Ready to mark: {state.pending_name}")

        # Show cross-image correspondence summary
        cross = session.get_cross_image_correspondences()
        if cross:
            status_lines.append(f"Cross-image links: {len(cross)}")

        status_text.set_text('\n'.join(status_lines))
        fig.canvas.draw_idle()

    def update_results_panel():
        """Update the results panel."""
        if not state.result or not state.result.solutions:
            results_text.set_text("No solution yet.\n\nMark stars, then click Solve.")
            return

        lines = []
        best = state.result.best_solution()

        lines.append(f"Solutions found: {len(state.result.solutions)}")
        lines.append(f"Best error: {best.total_error:.1f}")
        lines.append(f"  Known: {best.known_star_error:.1f}")
        lines.append(f"  Corr: {best.correspondence_error:.1f}")
        lines.append("")

        # Show shared FOV (same for all images)
        lines.append(f"Shared FOV: {best.calibrations[0].fov_horizontal:.1f}°")
        lines.append("")

        # Show per-image calibrations using spherical model
        lines.append("Per-image orientations:")
        for i, cal in enumerate(best.calibrations):
            up = spherical_to_cartesian(cal.up_theta, cal.up_phi)
            lines.append(f"Image {i + 1}:")
            lines.append(f"  up=({up[0]:.2f},{up[1]:.2f},{up[2]:.2f})")
            lines.append(f"  yaw={np.degrees(cal.yaw):.0f}°")

        # Show correspondence directions
        lines.append("")
        lines.append("Unknown star estimates:")
        for label in session.get_all_correspondence_labels():
            estimates = estimate_star_from_stitching(state.result, label)
            if estimates:
                pos, unc, direction, _ = estimates[0]
                lines.append(f"  {label}: ({pos[0]}, {pos[1]}, {pos[2]})")
                lines.append(f"    ±{unc:.0f} PC")

        results_text.set_text('\n'.join(lines))

    def on_click(event):
        """Handle mouse click on image."""
        if event.inaxes != ax:
            return
        if state.pending_name is None:
            return

        idx = state.current_image_idx
        px, py = event.xdata, event.ydata

        if state.mode == "known":
            if session.add_known_marker(idx, state.pending_name, px, py):
                print(f"Marked known star {state.pending_name} at ({px:.0f}, {py:.0f})")
            else:
                print(f"Unknown star: {state.pending_name}")
        else:
            session.add_correspondence(idx, state.pending_name, px, py)
            print(f"Marked correspondence {state.pending_name} at ({px:.0f}, {py:.0f})")

        state.pending_name = None
        update_display()

    def on_key(event):
        """Handle key press."""
        if event.key == 'escape':
            state.pending_name = None
            update_display()
        elif event.key == 'left':
            state.current_image_idx = (state.current_image_idx - 1) % len(image_paths)
            update_display()
        elif event.key == 'right':
            state.current_image_idx = (state.current_image_idx + 1) % len(image_paths)
            update_display()

    # Text input
    ax_textbox = plt.axes([0.12, 0.12, 0.20, 0.04])
    textbox = TextBox(ax_textbox, 'Name: ', initial='')

    def on_submit(text):
        text = text.strip()
        if not text:
            textbox.set_val('')
            return

        text_upper = text.upper()

        if state.mode == "known":
            # Look up in known stars
            matches = [name for name in session.known_stars if text_upper in name.upper()]
            if len(matches) == 1:
                state.pending_name = matches[0]
                print(f"Click to mark {matches[0]}")
            elif len(matches) > 1:
                print(f"Multiple matches: {matches[:5]}...")
            elif text_upper in session.known_stars:
                state.pending_name = text_upper
                print(f"Click to mark {text_upper}")
            else:
                print(f"Star '{text}' not in known stars")
        else:
            # Correspondence mode - any label is valid
            state.pending_name = text_upper
            print(f"Click to mark correspondence {text_upper}")

        textbox.set_val('')
        update_display()

    textbox.on_submit(on_submit)

    # Mode toggle button
    ax_mode = plt.axes([0.12, 0.04, 0.18, 0.05])
    btn_mode = Button(ax_mode, 'Mode: Known Stars')

    def on_mode_toggle(event):
        if state.mode == "known":
            state.mode = "correspondence"
            btn_mode.label.set_text('Mode: Correspondences')
        else:
            state.mode = "known"
            btn_mode.label.set_text('Mode: Known Stars')
        state.pending_name = None
        print(f"Switched to {state.mode.upper()} mode")
        update_display()

    btn_mode.on_clicked(on_mode_toggle)

    # Image navigation buttons
    ax_prev = plt.axes([0.35, 0.12, 0.08, 0.04])
    btn_prev = Button(ax_prev, '← Prev')

    def on_prev(event):
        state.current_image_idx = (state.current_image_idx - 1) % len(image_paths)
        update_display()

    btn_prev.on_clicked(on_prev)

    ax_next = plt.axes([0.44, 0.12, 0.08, 0.04])
    btn_next = Button(ax_next, 'Next →')

    def on_next(event):
        state.current_image_idx = (state.current_image_idx + 1) % len(image_paths)
        update_display()

    btn_next.on_clicked(on_next)

    # Solve button
    ax_solve = plt.axes([0.54, 0.12, 0.10, 0.04])
    btn_solve = Button(ax_solve, 'Solve')

    def on_solve(event):
        print("\nSolving stitching problem...")

        # Check we have enough constraints
        total_known = sum(len(img.known_markers) for img in session.images)
        cross = session.get_cross_image_correspondences()

        if total_known < 1:
            print("Need at least 1 known-position star marked")
            return

        state.result = solve_stitching(session)

        if state.result.solutions:
            print(f"\nFound {len(state.result.solutions)} solutions")
            if on_result:
                on_result(state.result)
        else:
            print("No valid solutions found")

        update_results_panel()
        update_display()

    btn_solve.on_clicked(on_solve)

    # Clear button
    ax_clear = plt.axes([0.66, 0.12, 0.08, 0.04])
    btn_clear = Button(ax_clear, 'Clear All')

    def on_clear(event):
        for img in session.images:
            img.known_markers = []
            img.correspondence_markers = []
        state.result = None
        state.pending_name = None
        update_results_panel()
        update_display()

    btn_clear.on_clicked(on_clear)

    # Connect events
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Store widget references on figure to prevent garbage collection
    fig._stitch_widgets = [textbox, btn_mode, btn_prev, btn_next, btn_solve, btn_clear]

    update_display()

    return fig, session, state


def run_stitching_tool(
    image_paths: list[str | Path],
    observer: Star,
    stars: list[Star],
):
    """Run the stitching tool interactively.

    Args:
        image_paths: List of screenshot paths
        observer: Observatory star
        stars: All known stars
    """
    fig, session, state = create_stitching_tool(image_paths, observer, stars)
    plt.show()
    return session, state
