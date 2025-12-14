"""Star Mapper - MirrorMoon EP anomaly triangulation tool."""

from .coords import Star, Observation, parse_coordinates, to_coordinate_string, distance
from .visualization import plot_star_map, show_map, plot_star_map_with_observations
from .projection import (
    ChartPosition,
    ViewState,
    project_stars_from_observer,
    chart_to_direction,
)
from .observatory_view import (
    plot_observatory_view,
    show_observatory_view,
    create_interactive_observatory_view,
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
    CalibrationResult,
    StarMarker,
    calibrate_from_markers,
    compute_direction_from_pixel,
    estimate_star_position,
)
from .screenshot_tool import (
    ScreenshotSession,
    DirectionRay,
    create_interactive_session,
    estimate_from_click,
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
    # Observatory projection
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
    # Calibration
    "CalibrationResult",
    "StarMarker",
    "calibrate_from_markers",
    "compute_direction_from_pixel",
    "estimate_star_position",
    # Screenshot tool
    "ScreenshotSession",
    "DirectionRay",
    "create_interactive_session",
    "estimate_from_click",
]
