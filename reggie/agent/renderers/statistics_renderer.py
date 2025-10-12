"""Statistics visualization renderers using Rich library."""

from typing import Dict, Any
from rich.console import Console
from rich.text import Text

# Initialize console for rendering
console = Console()


def render_single_dimension_chart(data: Dict[str, Any]) -> None:
    """Render a horizontal bar chart for single-dimension statistics.

    This function displays statistics grouped by a single dimension
    (sentiment, category, or topic) with horizontal bars.

    Args:
        data: Dictionary containing:
            - type: 'statistics'
            - group_by: 'sentiment', 'category', or 'topic'
            - total_comments: int
            - breakdown: List[Dict] with 'value', 'count', 'percentage'
            - filters: Dict of applied filters (optional)
    """
    group_by = data.get("group_by", "unknown")
    total_comments = data.get("total_comments", 0)
    breakdown = data.get("breakdown", [])
    filters = data.get("filters")

    # Build title
    title = Text()
    title.append(f"\nBreakdown by {group_by.capitalize()}\n", style="bold cyan")
    if filters:
        filter_parts = []
        for key, value in filters.items():
            if isinstance(value, list):
                filter_parts.append(f"{key}={','.join(value)}")
            else:
                filter_parts.append(f"{key}={value}")
        title.append(f"Filters: {', '.join(filter_parts)}\n", style="dim")
    title.append(f"Total comments: {total_comments}\n", style="dim")

    console.print(title)

    if not breakdown:
        console.print("[dim]No data to display[/dim]\n")
        return

    # Find max percentage for scaling
    max_percentage = max(item.get("percentage", 0) for item in breakdown)
    if max_percentage == 0:
        max_percentage = 1  # Avoid division by zero

    # Define maximum bar width in characters
    max_bar_width = 30

    # Find longest label for alignment
    max_label_length = max(len(str(item.get("value", ""))) for item in breakdown)

    # Render each item
    for item in breakdown:
        value = str(item.get("value", "unknown"))
        count = item.get("count", 0)
        percentage = item.get("percentage", 0)

        # Calculate bar width based on percentage
        bar_width = int((percentage / max_percentage) * max_bar_width)
        bar = "━" * bar_width

        # Determine color based on value (for sentiment)
        color = _get_color_for_value(value, group_by)

        # Create the line
        line = Text()
        line.append(f"  {value:<{max_label_length}}  ", style="white")
        line.append(bar, style=color)
        line.append(f"  {count} ({percentage}%)", style="white")

        console.print(line)

    console.print()  # Add blank line after chart


def _get_color_for_value(value: str, group_by: str) -> str:
    """Determine color for a value based on its type.

    Args:
        value: The value to color
        group_by: The dimension being grouped by

    Returns:
        Rich color string
    """
    if group_by == "sentiment":
        sentiment_colors = {
            "for": "green",
            "against": "red",
            "mixed": "yellow",
            "unclear": "dim"
        }
        return sentiment_colors.get(value.lower(), "white")

    # Default color for categories and topics
    return "cyan"


def render_opposition_support_chart(data: Dict[str, Any]) -> None:
    """Render a centered horizontal bar chart showing opposition vs support.

    This function displays sentiment breakdown by category using
    centered bars with opposition on the left and support on the right.

    Args:
        data: Dictionary containing:
            - type: 'opposition_support'
            - document_id: str
            - document_title: str (optional)
            - total_comments: int
            - breakdown: Dict[category, Dict[sentiment, count]]
    """
    # TODO: Implement in Task 3.2
    pass
