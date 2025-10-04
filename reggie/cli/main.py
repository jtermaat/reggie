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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

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
@click.option(
    "--batch-size",
    default=10,
    help="Number of comments to process in parallel (default: 10)",
)
def load(document_id: str, batch_size: int):
    """Load a document and its comments from Regulations.gov.

    DOCUMENT_ID: The document ID (e.g., CMS-2025-0304-0009)

    Example:
        reggie load CMS-2025-0304-0009
    """
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

    async def _load():
        loader = DocumentLoader()
        stats = await loader.load_document(document_id, batch_size=batch_size)
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
def process(document_id: str, batch_size: int):
    """Process comments: categorize and embed.

    DOCUMENT_ID: The document ID (e.g., CMS-2025-0304-0009)

    This processes comments that have already been loaded with 'reggie load'.

    Example:
        reggie process CMS-2025-0304-0009
    """
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
        stats = await processor.process_comments(document_id, batch_size=batch_size)
        return stats

    console.print(f"\n[bold]Processing comments for:[/bold] {document_id}\n")

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
@click.argument("document_id", required=False)
def discuss(document_id: str = None):
    """Start an interactive discussion session (coming soon).

    DOCUMENT_ID: Optional document ID to discuss (e.g., CMS-2025-0304-0009)

    Example:
        reggie discuss
        reggie discuss CMS-2025-0304-0009
    """
    console.print("\n[yellow]The 'discuss' command is coming soon![/yellow]")
    console.print("\nThis feature will enable you to:")
    console.print("  • Ask questions about loaded documents")
    console.print("  • Query comment sentiment and categories")
    console.print("  • Search for specific topics in comments")
    console.print()

    if document_id:
        console.print(f"Target document: {document_id}\n")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
