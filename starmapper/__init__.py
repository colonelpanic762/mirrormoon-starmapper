"""Star Mapper - MirrorMoon EP anomaly triangulation tool."""

from .coords import Star, Observation, parse_coordinates, to_coordinate_string, distance
from .visualization import plot_star_map, show_map, plot_star_map_with_observations
from .projection import (
    ChartPosition,
    ViewState,
    SphericalViewState,
    project_stars_from_observer,
    project_stars_spherical,
    chart_to_direction,
    chart_to_direction_spherical,
)
from .observatory_view import (
    plot_observatory_view,
    plot_observatory_view_spherical,
    show_observatory_view,
    show_observatory_view_spherical,
    create_interactive_observatory_view,
    create_interactive_observatory_view_spherical,
    ClickState,
    InteractiveState,
)
from .triangulation import (
    triangulate,
    TriangulationResult,
    point_to_cone_distance,
    format_result,
)
from .navigation import (
    navigate_to_target,
    navigate_with_calibration,
    NavCalibration,
    NavigationResult,
    decode_original_name,
    original_name_center,
    format_navigation_result,
)
from .calibration import (
    # Spherical moon model (primary)
    SphericalCalibration,
    calibrate_spherical,
    spherical_to_cartesian,
    build_camera_frame,
    project_star,
    compute_direction,
    # Legacy model (for backwards compatibility)
    CalibrationResult,
    calibrate_from_markers,
    compute_direction_from_pixel,
    # Shared
    StarMarker,
    estimate_star_position,
)
from .screenshot_tool import (
    ScreenshotSession,
    DirectionRay,
    create_interactive_session,
    estimate_from_click,
)
from .stitching import (
    StitchingSession,
    ImageMarking,
    CorrespondenceMarker,
    StitchingResult,
    StitchingSolution,
    solve_stitching,
    estimate_star_from_stitching,
)
from .stitching_tool import (
    create_stitching_tool,
    run_stitching_tool,
)

__all__ = [
    # Core data types
    "Star",
    "Observation",
    "ChartPosition",
    "ViewState",
    "ClickState",
    "InteractiveState",
    "NavCalibration",
    "NavigationResult",
    # Coordinate functions
    "parse_coordinates",
    "to_coordinate_string",
    "distance",
    # 3D visualization
    "plot_star_map",
    "show_map",
    "plot_star_map_with_observations",
    # Observatory projection (spherical model)
    "SphericalViewState",
    "project_stars_spherical",
    "chart_to_direction_spherical",
    "plot_observatory_view_spherical",
    "show_observatory_view_spherical",
    "create_interactive_observatory_view_spherical",
    # Observatory projection (legacy)
    "project_stars_from_observer",
    "chart_to_direction",
    "plot_observatory_view",
    "show_observatory_view",
    "create_interactive_observatory_view",
    # Triangulation
    "triangulate",
    "TriangulationResult",
    "point_to_cone_distance",
    "format_result",
    # Navigation
    "navigate_to_target",
    "navigate_with_calibration",
    "decode_original_name",
    "original_name_center",
    "format_navigation_result",
    # Calibration (spherical moon model)
    "SphericalCalibration",
    "calibrate_spherical",
    "spherical_to_cartesian",
    "build_camera_frame",
    "project_star",
    "compute_direction",
    # Calibration (legacy, for backwards compatibility)
    "CalibrationResult",
    "calibrate_from_markers",
    "compute_direction_from_pixel",
    # Calibration (shared)
    "StarMarker",
    "estimate_star_position",
    # Screenshot tool
    "ScreenshotSession",
    "DirectionRay",
    "create_interactive_session",
    "estimate_from_click",
    # Multi-image stitching
    "StitchingSession",
    "ImageMarking",
    "CorrespondenceMarker",
    "StitchingResult",
    "StitchingSolution",
    "solve_stitching",
    "estimate_star_from_stitching",
    "create_stitching_tool",
    "run_stitching_tool",
]
