"""CSV bulk importer for regulations.gov exports."""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Generator

from ..db.unit_of_work import UnitOfWork
from ..utils import ErrorCollector

logger = logging.getLogger(__name__)


class CSVImporter:
    """Imports comments from a regulations.gov CSV bulk download.

    This class handles parsing CSV files exported from regulations.gov's
    bulk download feature and storing the comments in the database.
    Comments are stored as raw data (not processed/categorized) - use
    'reggie process' afterward to categorize and embed them.
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        error_collector: Optional[ErrorCollector] = None,
    ):
        """Initialize the CSV importer.

        Args:
            database_url: PostgreSQL database connection URL
            error_collector: Optional error collector for aggregating errors
        """
        self.database_url = database_url
        self.error_collector = error_collector

    def _count_comments(self, csv_path: Path) -> int:
        """Count the number of public submissions in the CSV.

        Args:
            csv_path: Path to the CSV file

        Returns:
            Number of public submission rows
        """
        count = 0
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Document Type") == "Public Submission":
                    count += 1
        return count

    def _parse_csv(self, csv_path: Path) -> Generator[dict, None, None]:
        """Parse CSV file and yield normalized comment records.

        Args:
            csv_path: Path to the CSV file

        Yields:
            Normalized comment data dictionaries
        """
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip non-comment rows (e.g., the proposed rule itself)
                if row.get("Document Type") != "Public Submission":
                    continue

                # Get parent document ID
                parent_doc_id = row.get("Comment on Document ID", "").strip()
                if not parent_doc_id:
                    # Skip comments without a parent document
                    continue

                yield self._normalize_row(row, parent_doc_id)

    def _normalize_row(self, row: dict, document_id: str) -> dict:
        """Convert CSV row to comment data structure matching store_comment.

        Args:
            row: CSV row dictionary
            document_id: Parent document ID

        Returns:
            Normalized comment data dictionary
        """
        # Parse posted date from ISO format (e.g., "2025-08-01T04:00Z")
        posted_date = None
        posted_date_str = row.get("Posted Date", "")
        if posted_date_str:
            try:
                posted_date = datetime.fromisoformat(
                    posted_date_str.replace("Z", "+00:00")
                )
            except ValueError:
                pass

        # Build metadata with location and attachment info
        metadata = {
            "city": row.get("City", ""),
            "state": row.get("State/Province", ""),
            "zip": row.get("Zip/Postal Code", ""),
            "country": row.get("Country", ""),
            "tracking_number": row.get("Tracking Number", ""),
            "content_files": row.get("Content Files", ""),
            "attachment_files": row.get("Attachment Files", ""),
            "source": "csv_import",
        }

        # Match the structure expected by store_comment
        return {
            "id": row.get("Document ID", "").strip(),
            "document_id": document_id,
            "attributes": {
                "comment": row.get("Comment", ""),
                "firstName": row.get("First Name", ""),
                "lastName": row.get("Last Name", ""),
                "organization": row.get("Organization Name", ""),
                "postedDate": posted_date.isoformat() if posted_date else None,
                **metadata,
            },
        }

    async def _ensure_document(self, uow: UnitOfWork, document_id: str) -> None:
        """Ensure document record exists, creating a minimal one if needed.

        Args:
            uow: Unit of Work instance
            document_id: Document ID to ensure exists
        """
        if not await uow.documents.document_exists(document_id):
            # Extract docket ID from document ID (e.g., "CMS-2025-0304" from "CMS-2025-0304-0009")
            docket_id = None
            parts = document_id.rsplit("-", 1)
            if len(parts) == 2:
                docket_id = parts[0]

            # Create minimal document record
            await uow.documents.store_document(
                {
                    "id": document_id,
                    "attributes": {
                        "objectId": document_id,
                        "title": f"Imported from CSV: {document_id}",
                        "docketId": docket_id,
                        "documentType": "Proposed Rule",
                    },
                }
            )
            await uow.commit()
            logger.info(f"Created document record for {document_id}")

    async def import_csv(
        self,
        csv_path: Path,
        commit_every: int = 100,
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        """Import comments from CSV file into database.

        Args:
            csv_path: Path to the CSV file
            commit_every: Commit to database after this many comments
            progress_callback: Optional callback for progress updates

        Returns:
            Statistics about the import process
        """
        logger.info(f"Starting CSV import from {csv_path}")

        stats = {
            "document_id": None,
            "comments_imported": 0,
            "comments_skipped": 0,
            "errors": 0,
            "start_time": datetime.now(),
        }

        # Count total comments first for progress tracking
        logger.info("Counting comments in CSV...")
        total_comments = self._count_comments(csv_path)
        logger.info(f"Found {total_comments} comments to import")

        if progress_callback:
            progress_callback("init", total=total_comments)

        async with UnitOfWork(self.database_url) as uow:
            batch_count = 0

            for comment_data in self._parse_csv(csv_path):
                try:
                    comment_id = comment_data["id"]
                    document_id = comment_data["document_id"]

                    # Track document ID (should be same for all comments)
                    if stats["document_id"] is None:
                        stats["document_id"] = document_id
                        # Ensure document record exists
                        await self._ensure_document(uow, document_id)

                    # Skip if comment already exists
                    if await uow.comments.comment_exists(comment_id):
                        stats["comments_skipped"] += 1
                        if progress_callback:
                            progress_callback(
                                "update",
                                imported=stats["comments_imported"],
                                skipped=stats["comments_skipped"],
                            )
                        continue

                    # Store comment (without classification)
                    await uow.comments.store_comment(
                        comment_data,
                        document_id,
                        category=None,
                        sentiment=None,
                        topics=None,
                    )
                    stats["comments_imported"] += 1
                    batch_count += 1

                    # Update progress
                    if progress_callback:
                        progress_callback(
                            "update",
                            imported=stats["comments_imported"],
                            skipped=stats["comments_skipped"],
                        )

                    # Commit every N comments
                    if batch_count >= commit_every:
                        await uow.commit()
                        logger.info(
                            f"Imported {stats['comments_imported']} comments "
                            f"(committed to database)"
                        )
                        batch_count = 0

                except Exception as e:
                    logger.debug(f"Error importing comment: {e}")
                    if self.error_collector:
                        self.error_collector.collect(
                            error_type="CSV Import Error",
                            message=str(e),
                            context={"comment_id": comment_data.get("id", "unknown")},
                        )
                    stats["errors"] += 1

            # Final commit for any remaining comments
            if batch_count > 0:
                await uow.commit()
                logger.info(
                    f"Imported {stats['comments_imported']} comments total "
                    f"(all committed to database)"
                )

        stats["end_time"] = datetime.now()
        stats["duration"] = (stats["end_time"] - stats["start_time"]).total_seconds()

        if progress_callback:
            progress_callback("complete", **stats)

        logger.info(
            f"CSV import completed: {stats['comments_imported']} imported, "
            f"{stats['comments_skipped']} skipped, {stats['errors']} errors, "
            f"{stats['duration']:.1f}s"
        )

        return stats
