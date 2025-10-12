"""Statistics visualization renderers using Rich library."""

from typing import Dict, Any
from rich.console import Console
from rich.text import Text

# Initialize console for rendering
console = Console()

# Color scheme configuration
COLORS = {
    # Sentiment colors
    "sentiment_for": "green",
    "sentiment_against": "red",
    "sentiment_mixed": "yellow",
    "sentiment_unclear": "dim",

    # UI colors
    "title": "bold cyan",
    "subtitle": "dim",
    "label": "white",
    "category_name": "bold",  # Bold default color adapts to light/dark themes
    "separator": "dim",
    "default": "cyan",

    # Bar colors (for non-sentiment visualizations)
    "bar_default": "cyan",
}


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
    title.append(f"\nBreakdown by {group_by.capitalize()}\n", style=COLORS["title"])
    if filters:
        filter_parts = []
        for key, value in filters.items():
            if isinstance(value, list):
                filter_parts.append(f"{key}={','.join(value)}")
            else:
                filter_parts.append(f"{key}={value}")
        title.append(f"Filters: {', '.join(filter_parts)}\n", style=COLORS["subtitle"])
    title.append(f"Total comments: {total_comments}\n", style=COLORS["subtitle"])

    console.print(title)

    if not breakdown:
        console.print(f"[{COLORS['subtitle']}]No data to display[/{COLORS['subtitle']}]\n")
        return

    # Find max percentage for scaling
    max_percentage = max(item.get("percentage", 0) for item in breakdown)
    if max_percentage == 0:
        max_percentage = 1  # Avoid division by zero

    # Define maximum bar width in characters
    max_bar_width = 30

    # Find longest label for alignment (with reasonable max)
    max_label_length = min(max(len(str(item.get("value", ""))) for item in breakdown), 50)

    # Render each item
    for item in breakdown:
        value = str(item.get("value", "unknown"))
        count = item.get("count", 0)
        percentage = item.get("percentage", 0)

        # Truncate long values
        if len(value) > max_label_length:
            value = value[:max_label_length-3] + "..."

        # Calculate bar width based on percentage (minimum 1 char if > 0%)
        bar_width = max(int((percentage / max_percentage) * max_bar_width), 1 if percentage > 0 else 0)
        bar = "━" * bar_width

        # Determine color based on value (for sentiment)
        color = _get_color_for_value(value, group_by)

        # Create the line
        line = Text()
        line.append(f"  {value:<{max_label_length}}  ", style=COLORS["category_name"])
        line.append(bar, style=color)
        line.append(f"  {count} ({percentage}%)", style=COLORS["label"])

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
            "for": COLORS["sentiment_for"],
            "against": COLORS["sentiment_against"],
            "mixed": COLORS["sentiment_mixed"],
            "unclear": COLORS["sentiment_unclear"]
        }
        return sentiment_colors.get(value.lower(), COLORS["label"])

    # Default color for categories and topics
    return COLORS["bar_default"]


def render_opposition_support_chart(data: Dict[str, Any]) -> None:
    """Render a centered horizontal bar chart showing opposition vs support.

    This function displays sentiment breakdown by category using
    centered bars with opposition on the left and support on the right.
    The center separator "|" is always at the screen center.

    Args:
        data: Dictionary containing:
            - type: 'opposition_support'
            - document_id: str
            - document_title: str (optional)
            - total_comments: int
            - breakdown: Dict[category, Dict[sentiment, count]]
    """
    document_title = data.get("document_title", "Unknown Document")
    total_comments = data.get("total_comments", 0)
    breakdown = data.get("breakdown", {})

    # Build title
    title = Text()
    title.append("\nOpposition/Support Breakdown by Category\n", style=COLORS["title"])
    if document_title:
        title.append(f"Document: {document_title}\n", style=COLORS["subtitle"])
    title.append(f"Total Comments for Document: {total_comments}\n\n", style=COLORS["subtitle"])

    console.print(title)

    if not breakdown:
        console.print(f"[{COLORS['subtitle']}]No data to display[/{COLORS['subtitle']}]\n")
        return

    # Sort categories by total comment count (descending)
    sorted_categories = sorted(
        breakdown.items(),
        key=lambda item: sum(item[1].values()),
        reverse=True
    )

    # Calculate center position and available space
    terminal_width = console.width
    center_pos = terminal_width // 2

    # Layout: [Category Name] [Total Count] [Opposition bars...] | [Support bars...]
    # Column widths
    max_category_name_length = max(len(str(cat)) for cat in breakdown.keys())
    category_col_width = min(max_category_name_length + 2, 45)

    max_total_count = max(sum(sentiments.values()) for sentiments in breakdown.values())
    count_col_width = len(str(max_total_count)) + 4  # Add padding

    # Space from end of count column to center (for opposition bars)
    left_space = center_pos - category_col_width - count_col_width - 2
    # Space from center to edge (for support bars)
    right_space = terminal_width - center_pos - 2

    # Maximum bar width on each side (reserve chars for counts and spacing)
    max_left_bar_width = max(left_space - 20, 10)
    max_right_bar_width = max(right_space - 20, 10)

    # Print column headers (centered in each column)
    headers = Text()

    # Header 1: "Commenter Role" - centered in category_col_width
    header1 = "Commenter Role"
    header1_padding = (category_col_width - len(header1)) // 2
    headers.append(" " * header1_padding, style=COLORS["label"])
    headers.append(header1, style=COLORS["subtitle"])
    headers.append(" " * (category_col_width - len(header1) - header1_padding), style=COLORS["label"])

    # Header 2: "Total Comments" - centered in count_col_width
    header2 = "Total Comments"
    header2_padding = (count_col_width - len(header2)) // 2
    headers.append(" " * header2_padding, style=COLORS["label"])
    headers.append(header2, style=COLORS["subtitle"])
    headers.append(" " * (count_col_width - len(header2) - header2_padding), style=COLORS["label"])

    # Header 3: "Comments Opposing" - should end near center
    header3 = "Comments Opposing"

    # Calculate where we are currently and where we need to be
    current_pos = category_col_width + count_col_width

    # We want header3 to end near the center, so:
    # current_pos + padding_before + len(header3) + padding_after = center_pos - 1
    # Let's center it in the available space
    available_space = center_pos - current_pos - 1
    header3_padding_left = (available_space - len(header3)) // 2
    header3_padding_right = available_space - len(header3) - header3_padding_left

    headers.append(" " * header3_padding_left, style=COLORS["label"])
    headers.append(header3, style=COLORS["subtitle"])
    headers.append(" " * header3_padding_right, style=COLORS["label"])

    # Add space before "Comments Supporting" header (no pipe in header row)
    headers.append(" ", style=COLORS["label"])

    # Header 4: "Comments Supporting"
    header4 = " Comments Supporting"
    headers.append(header4, style=COLORS["subtitle"])

    console.print(headers)
    console.print()  # Blank line after headers

    # Render each category (in sorted order)
    for category, sentiments in sorted_categories:
        # Get counts for opposition (against) and support (for)
        against_count = sentiments.get("against", 0)
        for_count = sentiments.get("for", 0)

        # Calculate total for this category (ALL sentiments)
        category_total = sum(sentiments.values())

        if category_total == 0:
            # Skip categories with no comments
            continue

        # Calculate percentages based on TOTAL comments in category
        against_pct = (against_count / category_total * 100) if category_total > 0 else 0
        for_pct = (for_count / category_total * 100) if category_total > 0 else 0

        # Calculate bar widths based on percentages
        against_bar_width = int((against_pct / 100) * max_left_bar_width)
        for_bar_width = int((for_pct / 100) * max_right_bar_width)

        # Build opposition content (right-aligned, ending at center)
        opposition_text = Text()
        if against_count > 0:
            # Build opposition with arrow on LEFT: arrow, bars, count (ends at "|")
            opposition_text.append("◄", style=COLORS["sentiment_against"])
            opposition_text.append("━" * against_bar_width, style=COLORS["sentiment_against"])
            opposition_text.append(f" {against_count} ({against_pct:.0f}%)", style=COLORS["sentiment_against"])

        # Build the row in columns: [Category Name] [Count] [Opposition] | [Support]

        # Column 1: Category name (right-aligned, bold for visibility)
        category_name = category[:category_col_width-2] if len(category) > category_col_width-2 else category
        category_padding = " " * (category_col_width - len(category_name))

        # Column 2: Total count (right-aligned in gray)
        count_str = str(category_total)
        count_padding = " " * (count_col_width - len(count_str))

        # Build left side content before opposition bars
        left_content = Text()
        left_content.append(category_padding, style=COLORS["label"])
        left_content.append(category_name, style=COLORS["category_name"])
        left_content.append(count_padding, style=COLORS["label"])
        left_content.append(count_str, style=COLORS["subtitle"])

        # Calculate how much space we need to fill before the opposition bars end at center
        current_length = len(left_content.plain)
        opposition_length = len(opposition_text.plain)

        # We want: current_length + padding + opposition_length + 1 = center_pos
        padding_before_opposition = center_pos - current_length - opposition_length - 1
        if padding_before_opposition > 0:
            left_content.append(" " * padding_before_opposition, style=COLORS["label"])

        left_content.append(opposition_text)

        # Center separator
        left_content.append("|", style=COLORS["separator"])

        # Build right side content (support)
        right_content = Text()
        right_content.append(" ", style=COLORS["label"])

        if for_count > 0:
            right_content.append(f"{for_count} ({for_pct:.0f}%) ", style=COLORS["sentiment_for"])
            right_content.append("━" * for_bar_width, style=COLORS["sentiment_for"])
            right_content.append("►", style=COLORS["sentiment_for"])

        # Combine and print
        full_line = Text()
        full_line.append(left_content)
        full_line.append(right_content)
        console.print(full_line)

    console.print()  # Add blank line after chart
