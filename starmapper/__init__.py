"""Star Mapper - MirrorMoon EP anomaly triangulation tool."""

from .coords import Star, parse_coordinates, to_coordinate_string, distance
from .visualization import plot_star_map, show_map
from .projection import (
    ChartPosition,
    ViewState,
    project_stars_from_observer,
    chart_to_direction,
)
from .observatory_view import plot_observatory_view, show_observatory_view

__all__ = [
    # Core data types
    "Star",
    "ChartPosition",
    "ViewState",
    # Coordinate functions
    "parse_coordinates",
    "to_coordinate_string",
    "distance",
    # 3D visualization
    "plot_star_map",
    "show_map",
    # Observatory projection
    "project_stars_from_observer",
    "chart_to_direction",
    "plot_observatory_view",
    "show_observatory_view",
]
