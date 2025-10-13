"""Cost report rendering utilities for CLI display."""

from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ..models.cost import CostReport, UsageCost


def render_cost_report(
    report: CostReport,
    console: Optional[Console] = None,
    title: str = "OpenAI API Cost Report"
) -> None:
    """Render a cost report as a formatted table.

    Args:
        report: The cost report to render
        console: Optional Rich console (creates new one if not provided)
        title: Title for the cost report table
    """
    if console is None:
        console = Console()

    # Don't display anything if there are no costs
    if report.total_cost_usd == 0:
        return

    # Create the main table
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("Operation", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Tokens", justify="right", style="white")
    table.add_column("Cost (USD)", justify="right", style="green")

    # Add categorization row if there are costs
    if report.categorization_cost.total_tokens > 0:
        table.add_row(
            "Categorization",
            _format_model_name(report.categorization_cost.model_name),
            _format_tokens(report.categorization_cost),
            f"${report.categorization_cost.cost_usd:.4f}"
        )

    # Add embedding row if there are costs
    if report.embedding_cost.total_tokens > 0:
        table.add_row(
            "Embeddings",
            _format_model_name(report.embedding_cost.model_name),
            _format_tokens(report.embedding_cost),
            f"${report.embedding_cost.cost_usd:.4f}"
        )

    # Add agent row if there are costs
    if report.agent_cost.total_tokens > 0:
        table.add_row(
            "Agent",
            _format_model_name(report.agent_cost.model_name),
            _format_tokens(report.agent_cost),
            f"${report.agent_cost.cost_usd:.4f}"
        )

    # Add separator and total
    table.add_section()
    table.add_row(
        "[bold]Total[/bold]",
        "",
        "",
        f"[bold]${report.total_cost_usd:.4f}[/bold]"
    )

    # Display the table
    console.print()
    console.print(table)


def render_cost_summary(
    report: CostReport,
    console: Optional[Console] = None
) -> None:
    """Render a compact cost summary (one-line format).

    Args:
        report: The cost report to render
        console: Optional Rich console (creates new one if not provided)
    """
    if console is None:
        console = Console()

    # Don't display anything if there are no costs
    if report.total_cost_usd == 0:
        return

    console.print(f"\n[dim]Total API cost: [green]${report.total_cost_usd:.4f}[/green][/dim]")


def render_session_cost_report(
    report: CostReport,
    console: Optional[Console] = None
) -> None:
    """Render a cost report for a discussion session.

    This is a simplified version for interactive sessions where we want
    to show the total cost accumulated during the session.

    Args:
        report: The cost report to render
        console: Optional Rich console (creates new one if not provided)
    """
    if console is None:
        console = Console()

    # Don't display anything if there are no costs
    if report.total_cost_usd == 0:
        return

    # Create a simple panel with session cost
    console.print()
    console.print(Panel.fit(
        f"[bold]Session API Cost:[/bold] [green]${report.total_cost_usd:.4f}[/green]\n"
        f"[dim]Agent calls: {report.agent_cost.total_tokens:,} tokens[/dim]",
        border_style="dim"
    ))


def _format_model_name(model_name: Optional[str]) -> str:
    """Format model name for display, with truncation if needed.

    Args:
        model_name: The model name to format

    Returns:
        Formatted model name string
    """
    if not model_name:
        return "N/A"

    # Truncate long model names (e.g., text-embedding-3-small -> text-emb...)
    if len(model_name) > 15:
        return model_name[:12] + "..."

    return model_name


def _format_tokens(usage: UsageCost) -> str:
    """Format token counts for display.

    Args:
        usage: UsageCost object with token counts

    Returns:
        Formatted string with token breakdown
    """
    # For operations with both prompt and completion tokens (chat models)
    if usage.prompt_tokens > 0 and usage.completion_tokens > 0:
        return f"{usage.total_tokens:,} ({usage.prompt_tokens:,} + {usage.completion_tokens:,})"

    # For operations with only total tokens (embeddings)
    return f"{usage.total_tokens:,}"
