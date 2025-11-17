"""Main CLI interface for Reggie"""

import asyncio
import logging
import os
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..db import init_db
from ..pipeline import DocumentLoader, CommentProcessor
from ..pipeline.streamer import DocumentStreamer
from ..config import get_config
from ..logging_config import setup_logging
from .cost_renderer import render_cost_report, render_session_cost_report
from .progress import (
    LoadingProgressDisplay,
    ProcessingProgressDisplay,
    create_loading_progress_callback,
    create_processing_progress_callback,
)
from .streaming_progress import (
    StreamingProgressDisplay,
    create_streaming_progress_callback,
)

# Load environment variables
load_dotenv()

# Configure logging
setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))

console = Console()


@click.group()
def cli():
    """Reggie: AI Agent for exploring Regulations.gov

    Load documents, analyze comments, and explore regulatory insights.
    """
    pass


@cli.command()
@click.option(
    "--force",
    is_flag=True,
    help="Force re-initialization of database (drops existing data)",
)
def init(force: bool):
    """Initialize the database schema."""
    if force:
        console.print("[yellow]Warning: This will drop all existing data![/yellow]")
        if not click.confirm("Are you sure you want to continue?"):
            console.print("[red]Aborted.[/red]")
            return

    with console.status("[bold green]Initializing database..."):
        init_db()

    console.print("[bold green]✓[/bold green] Database initialized successfully!")


@cli.command()
@click.argument("document_id")
def load(document_id: str):
    """Load a document and its comments from Regulations.gov.

    DOCUMENT_ID: The document ID (e.g., CMS-2025-0304-0009)

    Comments are fetched sequentially with a 4-second delay between requests
    to stay under the API rate limit of 1000 requests per hour.

    Example:
        reggie load CMS-2025-0304-0009
    """
    try:
        # Create progress display
        display = LoadingProgressDisplay(document_id, console=console)

        async def _load():
            loader = DocumentLoader(error_collector=display.error_collector)
            # Create progress callback
            callback = create_loading_progress_callback(display)
            stats = await loader.load_document(document_id, progress_callback=callback)
            return stats

        # Use context manager for log suppression and cleanup
        with display:
            display.start()
            stats = asyncio.run(_load())
            display.stop()

        console.print("\n[bold green]✓[/bold green] Document loaded successfully!\n")
        console.print("[bold]Statistics:[/bold]")
        console.print(f"  • Comments loaded: {stats['comments_processed']}")
        console.print(f"  • Errors: {stats['errors']}")
        console.print(f"  • Duration: {stats['duration']:.1f}s")
        console.print("\n[dim]Note: Comments are stored but not yet categorized or embedded.[/dim]")
        console.print("[dim]Use 'reggie process <document_id>' to categorize and embed comments.[/dim]")

    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        logging.exception("Error loading document")
        return


@cli.command()
@click.argument("document_id")
@click.option(
    "--batch-size",
    default=10,
    help="Number of comments to process in parallel (default: 10)",
)
@click.option(
    "--skip-processed",
    is_flag=True,
    help="Skip comments that have already been processed (categorized/embedded)",
)
@click.option(
    "--trace",
    is_flag=True,
    help="Enable LangSmith tracing for debugging and evaluation",
)
def process(document_id: str, batch_size: int, skip_processed: bool, trace: bool):
    """Process comments: categorize and embed.

    DOCUMENT_ID: The document ID (e.g., CMS-2025-0304-0009)

    This processes comments that have already been loaded with 'reggie load'.

    Example:
        reggie process CMS-2025-0304-0009
        reggie process CMS-2025-0304-0009 --skip-processed
    """
    # Enable LangSmith tracing if requested
    if trace:
        config = get_config()
        config.apply_langsmith(enable_tracing=True)
        console.print("[dim]LangSmith tracing enabled[/dim]")

    # Verify required environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        console.print(
            f"[red]Error: Missing required environment variables: {', '.join(missing)}[/red]"
        )
        console.print("\nPlease set the following environment variables:")
        for var in missing:
            console.print(f"  - {var}")
        return

    try:
        # Get comment count first to initialize progress display
        def _get_count():
            from ..db.unit_of_work import UnitOfWork

            with UnitOfWork() as uow:
                return uow.comment_statistics.count_comments_for_document(
                    document_id, skip_processed=skip_processed
                )

        total_comments = _get_count()

        if total_comments == 0:
            if skip_processed:
                console.print(f"\n[yellow]No unprocessed comments found for document '{document_id}'.[/yellow]\n")
            else:
                console.print(f"\n[yellow]No comments found for document '{document_id}'.[/yellow]")
                console.print("\nUse 'reggie load <document_id>' to load comments first.\n")
            return

        # Create progress display
        display = ProcessingProgressDisplay(document_id, console=console)

        async def _process():
            processor = CommentProcessor(error_collector=display.error_collector)
            # Create progress callback
            callback = create_processing_progress_callback(display)
            stats = await processor.process_comments(
                document_id,
                batch_size=batch_size,
                skip_processed=skip_processed,
                progress_callback=callback
            )
            return stats

        # Use context manager for log suppression and cleanup
        with display:
            display.start(total=total_comments, skip_processed=skip_processed)
            stats = asyncio.run(_process())
            display.stop()

        console.print("\n[bold green]✓[/bold green] Comments processed successfully!\n")
        console.print("[bold]Statistics:[/bold]")
        console.print(f"  • Comments processed: {stats['comments_processed']}")
        console.print(f"  • Chunks created: {stats['chunks_created']}")
        console.print(f"  • Errors: {stats['errors']}")
        console.print(f"  • Duration: {stats['duration']:.1f}s")

        # Display cost report if available
        if "cost_report" in stats:
            render_cost_report(stats["cost_report"], console=console)

    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        logging.exception("Error processing comments")
        return


@cli.command()
@click.argument("document_id")
@click.option(
    "--trace",
    is_flag=True,
    help="Enable LangSmith tracing for debugging and evaluation",
)
def stream(document_id: str, trace: bool):
    """Stream a document: download, process, and store in one pass.

    DOCUMENT_ID: The document ID (e.g., CMS-2025-0304-0009)

    This command combines loading and processing into a single streaming
    operation. For each comment:
    1. Download from Regulations.gov API (with rate limiting)
    2. Immediately categorize and embed
    3. Immediately save to database

    The command displays running cost totals as it progresses, making
    efficient use of rate limit waiting time for processing.

    Example:
        reggie stream CMS-2025-0304-0009
    """
    # Enable LangSmith tracing if requested
    if trace:
        config = get_config()
        config.apply_langsmith(enable_tracing=True)
        console.print("[dim]LangSmith tracing enabled[/dim]")

    # Verify required environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        console.print(
            f"[red]Error: Missing required environment variables: {', '.join(missing)}[/red]"
        )
        console.print("\nPlease set the following environment variables:")
        for var in missing:
            console.print(f"  - {var}")
        return

    try:
        # Create progress display
        display = StreamingProgressDisplay(document_id, console=console)

        async def _stream():
            streamer = DocumentStreamer.create(error_collector=display.error_collector)
            # Create progress callback
            callback = create_streaming_progress_callback(display)
            stats = await streamer.stream_document(document_id, progress_callback=callback)
            return stats

        # Use context manager for log suppression and cleanup
        with display:
            display.start()
            stats = asyncio.run(_stream())
            display.stop()

        console.print("\n[bold green]✓[/bold green] Document streamed successfully!\n")
        console.print("[bold]Statistics:[/bold]")
        console.print(f"  • Comments processed: {stats['comments_processed']}")
        console.print(f"  • Chunks created: {stats['chunks_created']}")
        if stats.get('skipped', 0) > 0:
            console.print(f"  • Comments skipped: {stats['skipped']}")
        console.print(f"  • Errors: {stats['errors']}")
        console.print(f"  • Duration: {stats['duration']:.1f}s")

        # Display cost report
        if "cost_report" in stats:
            render_cost_report(stats["cost_report"], console=console)

    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        logging.exception("Error streaming document")
        return


@cli.command()
def list():
    """List all documents currently loaded in the database."""

    async def _list():
        loader = DocumentLoader()
        documents = await loader.list_documents()
        return documents

    try:
        documents = asyncio.run(_list())

        if not documents:
            console.print("\n[yellow]No documents loaded yet.[/yellow]")
            console.print("\nUse 'reggie load <document_id>' to load a document.\n")
            return

        table = Table(title="\nLoaded Documents", show_header=True, header_style="bold")
        table.add_column("Document ID", style="cyan")
        table.add_column("Title", style="white", max_width=50)
        table.add_column("Docket ID", style="magenta")
        table.add_column("Comments", justify="right", style="green")
        table.add_column("Categories", justify="right", style="blue")
        table.add_column("Loaded At", style="dim")

        for doc in documents:
            table.add_row(
                doc["id"],
                doc["title"][:50] + "..." if doc["title"] and len(doc["title"]) > 50 else doc["title"] or "N/A",
                doc["docket_id"] or "N/A",
                str(doc["comment_count"]),
                str(doc["unique_categories"]),
                doc["loaded_at"].strftime("%Y-%m-%d %H:%M") if doc["loaded_at"] else "N/A",
            )

        console.print(table)
        console.print()

    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        logging.exception("Error listing documents")


@cli.command()
@click.argument("document_id")
@click.option(
    "--trace",
    is_flag=True,
    help="Enable LangSmith tracing for debugging and evaluation",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging to see detailed debug information",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Override the default discussion model (e.g., gpt-5-mini, gpt-4o-mini)",
)
def discuss(document_id: str, trace: bool, verbose: bool, model: str):
    """Start an interactive discussion session about a document.

    DOCUMENT_ID: The document ID to discuss (e.g., CMS-2025-0304-0009)

    This starts an interactive chat where you can:
    • Ask statistical questions (e.g., "How many physicians support this?")
    • Search comment content (e.g., "What did people say about costs?")
    • Explore sentiment, categories, and topics

    Example:
        reggie discuss CMS-2025-0304-0009
    """
    from ..agent import DiscussionAgent
    from ..agent.status import set_status_callback, clear_status_callback
    from ..agent.visualizations import set_visualization_callback, clear_visualization_callback
    from ..agent.renderers import render_single_dimension_chart
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.live import Live
    from langchain_core.messages import AIMessage

    # Enable LangSmith tracing if requested
    if trace:
        config = get_config()
        config.apply_langsmith(enable_tracing=True)
        console.print("[dim]LangSmith tracing enabled[/dim]")

    # Enable verbose logging if requested
    if verbose:
        logging.getLogger("reggie").setLevel(logging.DEBUG)
        console.print("[dim]Verbose logging enabled[/dim]")

    # Verify required environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        console.print(
            f"[red]Error: Missing required environment variables: {', '.join(missing)}[/red]"
        )
        console.print("\nPlease set the following environment variables:")
        for var in missing:
            console.print(f"  - {var}")
        return

    def _verify_document():
        """Verify the document exists and has processed comments."""
        from ..db import get_connection

        with get_connection() as conn:
            # Check document exists
            cur = conn.execute(
                "SELECT title FROM documents WHERE id = ?",
                (document_id,)
            )
            doc = cur.fetchone()

            if not doc:
                return None, "Document not found"

            # Check for processed comments (with embeddings)
            cur = conn.execute(
                """
                SELECT COUNT(DISTINCT c.id)
                FROM comments c
                JOIN comment_chunks cc ON c.id = cc.comment_id
                WHERE c.document_id = ?
                """,
                (document_id,)
            )
            count = cur.fetchone()[0]

            if count == 0:
                return doc[0], "no_comments"

            return doc[0], count

    async def _run_discussion():
        """Run the interactive discussion."""
        # Verify document
        doc_title, status = _verify_document()

        if doc_title is None:
            console.print(f"\n[red]Error: Document '{document_id}' not found.[/red]")
            console.print("\nUse 'reggie list' to see available documents.")
            console.print("Use 'reggie load <document_id>' to load a new document.\n")
            return

        if status == "no_comments":
            console.print(f"\n[yellow]Warning: Document '{document_id}' has no processed comments.[/yellow]")
            console.print("\nUse 'reggie process <document_id>' to process comments first.\n")
            return

        # Initialize agent
        agent = DiscussionAgent.create(document_id=document_id, model=model)

        # Set up status callback to display status messages in gray
        set_status_callback(lambda msg: console.print(f"[dim]{msg}[/dim]"))

        # Set up visualization callback to render charts
        set_visualization_callback(lambda data: render_single_dimension_chart(data))

        # Display welcome message
        console.print()
        console.print(Panel.fit(
            f"[bold cyan]Discussion Mode[/bold cyan]\n\n"
            f"Document: {doc_title[:60]}{'...' if len(doc_title) > 60 else ''}\n"
            f"Processed comments: {status}\n\n"
            f"Ask me anything about this document's comments!\n"
            f"Type 'exit' or 'quit' to end the session.",
            border_style="cyan"
        ))
        console.print()

        # Main conversation loop
        while True:
            try:
                # Get user input
                user_input = console.input("[bold green]You:[/bold green] ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit", "q"]:
                    # Display cost report before exiting
                    cost_report = agent.get_cost_report()
                    render_session_cost_report(cost_report, console=console)
                    console.print("\n[dim]Goodbye![/dim]\n")
                    break

                # Stream the response with fallback to non-streaming if needed
                console.print()

                response_buffer = ""
                response_started = False
                live_display = None
                status = console.status("[bold cyan]Thinking...", spinner="dots")
                status.start()

                try:
                    # Attempt streaming first
                    try:
                        async for token, metadata in agent.stream(user_input):
                            # Only display content from AIMessage objects (not ToolMessage)
                            if isinstance(token, AIMessage):
                                # Display content tokens from AI messages only
                                if hasattr(token, 'content') and isinstance(token.content, str) and token.content:
                                    if not response_started:
                                        # Stop the thinking indicator and start showing the response
                                        status.stop()
                                        console.print("[bold blue]Assistant:[/bold blue]")
                                        # Initialize Live display for streaming markdown
                                        live_display = Live(Markdown(""), console=console, refresh_per_second=10)
                                        live_display.start()
                                        response_started = True

                                    # Accumulate tokens and update live display
                                    response_buffer += token.content
                                    if live_display:
                                        live_display.update(Markdown(response_buffer))

                        # Stop live display (leaves final content visible)
                        if live_display:
                            live_display.stop()

                        if not response_started:
                            console.print()

                    except Exception as streaming_error:
                        # Streaming failed - fall back to non-streaming
                        if status._live.is_started:
                            status.stop()
                        if live_display:
                            try:
                                live_display.stop()
                            except:
                                pass

                        # Check if it's a verification/streaming error
                        error_str = str(streaming_error).lower()
                        if "stream" in error_str or "organization" in error_str or "verification" in error_str:
                            console.print("[yellow]Note: Streaming not available (organization may need verification). Falling back to non-streaming mode...[/yellow]\n")
                        else:
                            # Log the error for debugging
                            logging.debug(f"Streaming error: {streaming_error}")
                            console.print("[yellow]Streaming unavailable, using non-streaming mode...[/yellow]\n")

                        # Retry without streaming
                        try:
                            with console.status("[bold cyan]Thinking...", spinner="dots"):
                                response = await agent.invoke(user_input)

                            console.print("[bold blue]Assistant:[/bold blue]")
                            console.print(Markdown(response))
                            console.print()

                        except Exception as invoke_error:
                            # Both streaming and non-streaming failed
                            console.print("[red]Error communicating with OpenAI:[/red]")
                            console.print(f"[dim]The request failed with the following error:[/dim]")
                            console.print(f"[red]{invoke_error}[/red]\n")
                            console.print("[dim]Possible solutions:[/dim]")
                            console.print("[dim]• Check that your OpenAI API key is valid[/dim]")
                            console.print("[dim]• Verify the model name is correct (e.g., gpt-5-mini, gpt-4o-mini)[/dim]")
                            console.print("[dim]• Ensure you have sufficient API credits[/dim]\n")

                finally:
                    # Ensure status and live display are stopped even if there's an error
                    if status._live.is_started:
                        status.stop()
                    if live_display:
                        try:
                            live_display.stop()
                        except:
                            pass  # Already stopped

            except KeyboardInterrupt:
                # Display cost report before exiting
                cost_report = agent.get_cost_report()
                render_session_cost_report(cost_report, console=console)
                console.print("\n\n[dim]Goodbye![/dim]\n")
                break
            except Exception as e:
                console.print(f"\n[red]Error:[/red] {e}\n")
                logging.exception("Error in discussion")

        # Clear callbacks when session ends
        clear_status_callback()
        clear_visualization_callback()

    try:
        asyncio.run(_run_discussion())
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        logging.exception("Error starting discussion")


@cli.command()
@click.argument("document_id")
def visualize(document_id: str):
    """Display opposition/support visualization for a document.

    DOCUMENT_ID: The document ID to visualize (e.g., CMS-2025-0304-0009)

    Shows a breakdown of opposition (against) vs support (for) across
    all commenter categories using centered horizontal bars.

    Example:
        reggie visualize CMS-2025-0304-0009
    """
    from ..db.unit_of_work import UnitOfWork
    from ..agent.renderers import render_opposition_support_chart, render_opposition_support_by_specialization

    def _visualize():
        """Generate and display the visualization."""
        with UnitOfWork() as uow:
            # Verify document exists
            cur = uow._conn.execute(
                "SELECT title FROM documents WHERE id = ?",
                (document_id,)
            )
            doc = cur.fetchone()

            if not doc:
                console.print(f"\n[red]Error: Document '{document_id}' not found.[/red]")
                console.print("\nUse 'reggie list' to see available documents.")
                console.print("Use 'reggie load <document_id>' to load a new document.\n")
                return

            doc_title = doc[0]

            # Check for processed comments
            cur = uow._conn.execute(
                """
                SELECT COUNT(DISTINCT c.id)
                FROM comments c
                WHERE c.document_id = ?
                """,
                (document_id,)
            )
            count = cur.fetchone()[0]

            if count == 0:
                console.print(f"\n[yellow]Warning: Document '{document_id}' has no comments.[/yellow]")
                console.print("\nUse 'reggie load <document_id>' to load comments first.\n")
                return

            # Get sentiment by category data
            breakdown = uow.comment_analytics.get_sentiment_by_category(
                document_id=document_id
            )

            if not breakdown:
                console.print(f"\n[yellow]No categorized comments found for document '{document_id}'.[/yellow]")
                console.print("\nUse 'reggie process <document_id>' to process comments first.\n")
                return

            # Calculate total comments (for display)
            total_comments = sum(
                sum(sentiments.values()) for sentiments in breakdown.values()
            )

            # SECTION 1: Render the category breakdown visualization
            render_opposition_support_chart({
                "type": "opposition_support",
                "document_id": document_id,
                "document_title": doc_title,
                "total_comments": total_comments,
                "breakdown": breakdown
            })

            # SECTION 2: Render physician specializations breakdown
            physician_category = "Physicians & Surgeons"
            if physician_category in breakdown:
                physician_breakdown = uow.comment_analytics.get_sentiment_by_specialization(
                    document_id=document_id,
                    field_name="doctor_specialization",
                    category_filter=physician_category
                )

                if physician_breakdown:
                    # Calculate total for physician category
                    physician_total = sum(breakdown[physician_category].values())
                    render_opposition_support_by_specialization(
                        section_title="Physician Specialization",
                        category_name=physician_category,
                        category_total=physician_total,
                        breakdown=physician_breakdown
                    )

            # SECTION 3: Render licensed professional types breakdown
            licensed_professional_category = "Other Licensed Clinicians"
            if licensed_professional_category in breakdown:
                licensed_professional_breakdown = uow.comment_analytics.get_sentiment_by_specialization(
                    document_id=document_id,
                    field_name="licensed_professional_type",
                    category_filter=licensed_professional_category
                )

                if licensed_professional_breakdown:
                    # Calculate total for licensed professional category
                    licensed_professional_total = sum(breakdown[licensed_professional_category].values())
                    render_opposition_support_by_specialization(
                        section_title="Licensed Professional Type",
                        category_name=licensed_professional_category,
                        category_total=licensed_professional_total,
                        breakdown=licensed_professional_breakdown
                    )

    try:
        _visualize()
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        logging.exception("Error generating visualization")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
