"""Visualization renderers for displaying data using Rich."""

from .statistics_renderer import (
    render_single_dimension_chart,
    render_opposition_support_chart,
    render_opposition_support_by_specialization,
)

__all__ = [
    "render_single_dimension_chart",
    "render_opposition_support_chart",
    "render_opposition_support_by_specialization",
]
