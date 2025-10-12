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
from ..config import get_config
from ..logging_config import setup_logging

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

    async def _init():
        await init_db()

    with console.status("[bold green]Initializing database..."):
        asyncio.run(_init())

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
    async def _load():
        loader = DocumentLoader()
        stats = await loader.load_document(document_id)
        return stats

    console.print(f"\n[bold]Loading document:[/bold] {document_id}\n")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing...", total=None)
            stats = asyncio.run(_load())

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

    async def _process():
        processor = CommentProcessor()
        stats = await processor.process_comments(
            document_id,
            batch_size=batch_size,
            skip_processed=skip_processed
        )
        return stats

    console.print(f"\n[bold]Processing comments for:[/bold] {document_id}")
    if skip_processed:
        console.print("[dim]Skipping already processed comments[/dim]\n")
    else:
        console.print()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing...", total=None)
            stats = asyncio.run(_process())

        console.print("\n[bold green]✓[/bold green] Comments processed successfully!\n")
        console.print("[bold]Statistics:[/bold]")
        console.print(f"  • Comments processed: {stats['comments_processed']}")
        console.print(f"  • Chunks created: {stats['chunks_created']}")
        console.print(f"  • Errors: {stats['errors']}")
        console.print(f"  • Duration: {stats['duration']:.1f}s")

    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        logging.exception("Error processing comments")
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
def discuss(document_id: str, trace: bool, verbose: bool):
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

    async def _verify_document():
        """Verify the document exists and has processed comments."""
        from ..db.connection import get_connection

        async with get_connection() as conn:
            async with conn.cursor() as cur:
                # Check document exists
                await cur.execute(
                    "SELECT title FROM documents WHERE id = %s",
                    (document_id,)
                )
                doc = await cur.fetchone()

                if not doc:
                    return None, "Document not found"

                # Check for processed comments (with embeddings)
                await cur.execute(
                    """
                    SELECT COUNT(DISTINCT c.id)
                    FROM comments c
                    JOIN comment_chunks cc ON c.id = cc.comment_id
                    WHERE c.document_id = %s
                    """,
                    (document_id,)
                )
                count = (await cur.fetchone())[0]

                if count == 0:
                    return doc[0], "no_comments"

                return doc[0], count

    async def _run_discussion():
        """Run the interactive discussion."""
        # Verify document
        doc_title, status = await _verify_document()

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
        agent = DiscussionAgent.create(document_id=document_id)

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
                    console.print("\n[dim]Goodbye![/dim]\n")
                    break

                # TEMPORARY: Using non-streaming invoke for gpt-5-mini testing
                console.print()

                with console.status("[bold cyan]Thinking...", spinner="dots"):
                    response = await agent.invoke(user_input)

                console.print("[bold blue]Assistant:[/bold blue]")
                console.print(Markdown(response))
                console.print()

                # # Stream the response with thinking indicator
                # console.print()

                # response_buffer = ""
                # response_started = False
                # live_display = None
                # status = console.status("[bold cyan]Thinking...", spinner="dots")
                # status.start()

                # try:
                #     async for token, metadata in agent.stream(user_input):
                #         # Only display content from AIMessage objects (not ToolMessage)
                #         if isinstance(token, AIMessage):
                #             # Display content tokens from AI messages only
                #             if hasattr(token, 'content') and isinstance(token.content, str) and token.content:
                #                 if not response_started:
                #                     # Stop the thinking indicator and start showing the response
                #                     status.stop()
                #                     console.print("[bold blue]Assistant:[/bold blue]")
                #                     # Initialize Live display for streaming markdown
                #                     live_display = Live(Markdown(""), console=console, refresh_per_second=10)
                #                     live_display.start()
                #                     response_started = True

                #                 # Accumulate tokens and update live display
                #                 response_buffer += token.content
                #                 if live_display:
                #                     live_display.update(Markdown(response_buffer))

                #     # Stop live display (leaves final content visible)
                #     if live_display:
                #         live_display.stop()

                # finally:
                #     # Ensure status and live display are stopped even if there's an error
                #     if status._live.is_started:
                #         status.stop()
                #     if live_display:
                #         try:
                #             live_display.stop()
                #         except:
                #             pass  # Already stopped

                # if not response_started:
                #     console.print()

            except KeyboardInterrupt:
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


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
