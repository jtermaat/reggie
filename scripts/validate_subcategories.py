"""Validation script for checking sub-category classification quality.

This script analyzes the quality of doctor_specialization and licensed_professional_type
classifications to ensure they are being applied correctly and with good coverage.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path to import reggie modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from reggie.db.connection import get_connection
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


async def validate_subcategories(document_id: str):
    """Validate sub-category classifications for a document.

    Args:
        document_id: The document ID to validate
    """
    console.print(f"\n[bold]Validating sub-categories for document:[/bold] {document_id}\n")

    async with get_connection() as conn:
        async with conn.cursor() as cur:
            # 1. Check for orphaned doctor_specializations
            await cur.execute("""
                SELECT COUNT(*)
                FROM comments
                WHERE document_id = %s
                AND doctor_specialization IS NOT NULL
                AND category != 'Physicians & Surgeons'
            """, (document_id,))
            orphaned_specializations = (await cur.fetchone())[0]

            # 2. Check for orphaned licensed_professional_types
            await cur.execute("""
                SELECT COUNT(*)
                FROM comments
                WHERE document_id = %s
                AND licensed_professional_type IS NOT NULL
                AND category != 'Other Licensed Clinicians'
            """, (document_id,))
            orphaned_prof_types = (await cur.fetchone())[0]

            # 3. Get physician coverage statistics
            await cur.execute("""
                SELECT
                    COUNT(*) as total_physicians,
                    COUNT(doctor_specialization) as with_specialization,
                    COUNT(CASE WHEN doctor_specialization IS NULL THEN 1 END) as without_specialization,
                    ROUND(100.0 * COUNT(doctor_specialization) / NULLIF(COUNT(*), 0), 1) as coverage_pct
                FROM comments
                WHERE document_id = %s
                AND category = 'Physicians & Surgeons'
            """, (document_id,))
            physician_stats = await cur.fetchone()

            # 4. Get licensed professional coverage statistics
            await cur.execute("""
                SELECT
                    COUNT(*) as total_professionals,
                    COUNT(licensed_professional_type) as with_type,
                    COUNT(CASE WHEN licensed_professional_type IS NULL THEN 1 END) as without_type,
                    ROUND(100.0 * COUNT(licensed_professional_type) / NULLIF(COUNT(*), 0), 1) as coverage_pct
                FROM comments
                WHERE document_id = %s
                AND category = 'Other Licensed Clinicians'
            """, (document_id,))
            professional_stats = await cur.fetchone()

            # 5. Get top doctor specializations
            await cur.execute("""
                SELECT
                    doctor_specialization,
                    COUNT(*) as count,
                    ROUND(100.0 * COUNT(*) / (
                        SELECT COUNT(*)
                        FROM comments
                        WHERE document_id = %s
                        AND category = 'Physicians & Surgeons'
                        AND doctor_specialization IS NOT NULL
                    ), 1) as percentage
                FROM comments
                WHERE document_id = %s
                AND category = 'Physicians & Surgeons'
                AND doctor_specialization IS NOT NULL
                GROUP BY doctor_specialization
                ORDER BY count DESC
                LIMIT 15
            """, (document_id, document_id))
            top_specializations = await cur.fetchall()

            # 6. Get top licensed professional types
            await cur.execute("""
                SELECT
                    licensed_professional_type,
                    COUNT(*) as count,
                    ROUND(100.0 * COUNT(*) / (
                        SELECT COUNT(*)
                        FROM comments
                        WHERE document_id = %s
                        AND category = 'Other Licensed Clinicians'
                        AND licensed_professional_type IS NOT NULL
                    ), 1) as percentage
                FROM comments
                WHERE document_id = %s
                AND category = 'Other Licensed Clinicians'
                AND licensed_professional_type IS NOT NULL
                GROUP BY licensed_professional_type
                ORDER BY count DESC
                LIMIT 15
            """, (document_id, document_id))
            top_prof_types = await cur.fetchall()

    # Display validation results

    # 1. Data Integrity Check
    console.print(Panel.fit(
        "[bold cyan]Data Integrity Check[/bold cyan]\n\n"
        f"Orphaned doctor_specializations: {orphaned_specializations}\n"
        f"Orphaned licensed_professional_types: {orphaned_prof_types}\n\n"
        f"{'[green]✓ No orphaned sub-categories found[/green]' if orphaned_specializations == 0 and orphaned_prof_types == 0 else '[red]✗ Orphaned sub-categories detected - this indicates a validation error![/red]'}",
        border_style="cyan"
    ))
    console.print()

    # 2. Physician Specialization Coverage
    if physician_stats and physician_stats[0] > 0:
        console.print(Panel.fit(
            "[bold cyan]Physician Specialization Coverage[/bold cyan]\n\n"
            f"Total physicians: {physician_stats[0]}\n"
            f"With specialization: {physician_stats[1]} ({physician_stats[3]}%)\n"
            f"Without specialization: {physician_stats[2]}\n\n"
            f"{'[green]✓ Good coverage (>80%)[/green]' if physician_stats[3] and physician_stats[3] > 80 else '[yellow]⚠ Consider reviewing unspecified physicians[/yellow]' if physician_stats[3] and physician_stats[3] > 50 else '[red]✗ Low coverage (<50%) - may need prompt improvement[/red]'}",
            border_style="cyan"
        ))
        console.print()

        # Show top specializations
        if top_specializations:
            table = Table(title="Top Doctor Specializations", show_header=True, header_style="bold")
            table.add_column("Specialization", style="cyan")
            table.add_column("Count", justify="right", style="green")
            table.add_column("Percentage", justify="right", style="blue")

            for spec, count, pct in top_specializations:
                table.add_row(spec, str(count), f"{pct}%")

            console.print(table)
            console.print()
    else:
        console.print("[dim]No physicians found in this document.[/dim]\n")

    # 3. Licensed Professional Type Coverage
    if professional_stats and professional_stats[0] > 0:
        console.print(Panel.fit(
            "[bold cyan]Licensed Professional Type Coverage[/bold cyan]\n\n"
            f"Total licensed professionals: {professional_stats[0]}\n"
            f"With type specified: {professional_stats[1]} ({professional_stats[3]}%)\n"
            f"Without type: {professional_stats[2]}\n\n"
            f"{'[green]✓ Good coverage (>80%)[/green]' if professional_stats[3] and professional_stats[3] > 80 else '[yellow]⚠ Consider reviewing unspecified professionals[/yellow]' if professional_stats[3] and professional_stats[3] > 50 else '[red]✗ Low coverage (<50%) - may need prompt improvement[/red]'}",
            border_style="cyan"
        ))
        console.print()

        # Show top professional types
        if top_prof_types:
            table = Table(title="Top Licensed Professional Types", show_header=True, header_style="bold")
            table.add_column("Professional Type", style="cyan")
            table.add_column("Count", justify="right", style="green")
            table.add_column("Percentage", justify="right", style="blue")

            for prof_type, count, pct in top_prof_types:
                table.add_row(prof_type, str(count), f"{pct}%")

            console.print(table)
            console.print()
    else:
        console.print("[dim]No licensed professionals found in this document.[/dim]\n")

    # 4. Sample unspecified physicians/professionals for manual review
    console.print("[bold]Sample Comments for Manual Review:[/bold]\n")

    async with get_connection() as conn:
        async with conn.cursor() as cur:
            # Sample unspecified physicians
            await cur.execute("""
                SELECT id, first_name, last_name, organization,
                       LEFT(comment_text, 200) as preview
                FROM comments
                WHERE document_id = %s
                AND category = 'Physicians & Surgeons'
                AND doctor_specialization IS NULL
                ORDER BY RANDOM()
                LIMIT 5
            """, (document_id,))
            unspecified_physicians = await cur.fetchall()

            if unspecified_physicians:
                console.print("[cyan]Physicians without specialization:[/cyan]")
                for comment_id, first, last, org, preview in unspecified_physicians:
                    name = f"{first or ''} {last or ''}".strip() or "N/A"
                    console.print(f"  • ID: {comment_id}")
                    console.print(f"    Name: {name} | Org: {org or 'N/A'}")
                    console.print(f"    Preview: {preview}...")
                    console.print()

            # Sample unspecified professionals
            await cur.execute("""
                SELECT id, first_name, last_name, organization,
                       LEFT(comment_text, 200) as preview
                FROM comments
                WHERE document_id = %s
                AND category = 'Other Licensed Clinicians'
                AND licensed_professional_type IS NULL
                ORDER BY RANDOM()
                LIMIT 5
            """, (document_id,))
            unspecified_professionals = await cur.fetchall()

            if unspecified_professionals:
                console.print("[cyan]Licensed professionals without type:[/cyan]")
                for comment_id, first, last, org, preview in unspecified_professionals:
                    name = f"{first or ''} {last or ''}".strip() or "N/A"
                    console.print(f"  • ID: {comment_id}")
                    console.print(f"    Name: {name} | Org: {org or 'N/A'}")
                    console.print(f"    Preview: {preview}...")
                    console.print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("[red]Error: Document ID required[/red]")
        console.print("\nUsage: python validate_subcategories.py <document_id>")
        console.print("Example: python validate_subcategories.py CMS-2025-0304-0009")
        sys.exit(1)

    document_id = sys.argv[1]
    asyncio.run(validate_subcategories(document_id))
