"""Integration tests for database operations using real PostgreSQL"""

import pytest
from datetime import datetime
from psycopg.types.json import Json

from reggie.db.repository import DocumentRepository, CommentRepository, CommentChunkRepository


@pytest.mark.integration
class TestDocumentRepository:
    """Test document repository with real database."""

    async def test_store_document(self, test_db, sample_document_data):
        """Store document in database."""
        await DocumentRepository.store_document(sample_document_data, test_db)

        # Verify it was stored
        async with test_db.cursor() as cur:
            await cur.execute(
                "SELECT id, title, object_id, docket_id FROM documents WHERE id = %s",
                (sample_document_data["id"],)
            )
            row = await cur.fetchone()

        assert row is not None, "Document should be stored in database"
        assert row[0] == sample_document_data["id"]
        assert row[1] == sample_document_data["attributes"]["title"]
        assert row[2] == sample_document_data["attributes"]["objectId"]
        assert row[3] == sample_document_data["attributes"]["docketId"]

    async def test_store_document_upsert_updates_existing(self, test_db, sample_document_data):
        """Upserting same document updates existing record."""
        # Store initial document
        await DocumentRepository.store_document(sample_document_data, test_db)

        # Update document data
        updated_data = sample_document_data.copy()
        updated_data["attributes"]["title"] = "Updated Title"

        # Upsert with updated data
        await DocumentRepository.store_document(updated_data, test_db)

        # Verify only one record exists with updated title
        async with test_db.cursor() as cur:
            await cur.execute(
                "SELECT COUNT(*), title FROM documents WHERE id = %s GROUP BY title",
                (sample_document_data["id"],)
            )
            row = await cur.fetchone()

        assert row[0] == 1, "Should have exactly one document"
        assert row[1] == "Updated Title", "Title should be updated"

    async def test_list_documents_empty(self, test_db):
        """List documents returns empty list when no documents exist."""
        documents = await DocumentRepository.list_documents(test_db)
        assert documents == []

    async def test_list_documents_with_comment_counts(self, test_db, sample_document_data):
        """List documents includes comment counts and stats."""
        # Store document
        await DocumentRepository.store_document(sample_document_data, test_db)
        doc_id = sample_document_data["id"]

        # Store some comments
        async with test_db.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO comments (id, document_id, comment_text, category)
                VALUES
                    ('C1', %s, 'Comment 1', 'Physicians & Surgeons'),
                    ('C2', %s, 'Comment 2', 'Physicians & Surgeons'),
                    ('C3', %s, 'Comment 3', 'Patients & Caregivers')
                """,
                (doc_id, doc_id, doc_id)
            )
        await test_db.commit()

        # List documents
        documents = await DocumentRepository.list_documents(test_db)

        assert len(documents) == 1
        doc = documents[0]
        assert doc["id"] == doc_id
        assert doc["comment_count"] == 3
        assert doc["unique_categories"] == 2  # Two different categories

    async def test_list_documents_ordered_by_created_at(self, test_db):
        """List documents returns newest first."""
        # Store multiple documents
        for i in range(3):
            doc_data = {
                "id": f"DOC-{i}",
                "attributes": {
                    "objectId": f"OBJ-{i}",
                    "title": f"Document {i}",
                    "docketId": "DOCKET-001",
                    "documentType": "Rule",
                    "postedDate": "2024-01-01T00:00:00Z"
                }
            }
            await DocumentRepository.store_document(doc_data, test_db)

        documents = await DocumentRepository.list_documents(test_db)

        assert len(documents) == 3
        # Should be ordered newest first (DESC)
        assert documents[0]["id"] == "DOC-2"
        assert documents[1]["id"] == "DOC-1"
        assert documents[2]["id"] == "DOC-0"


@pytest.mark.integration
class TestCommentRepository:
    """Test comment repository with real database."""

    async def test_comment_exists_false_when_not_exists(self, test_db):
        """comment_exists returns False when comment doesn't exist."""
        exists = await CommentRepository.comment_exists("NONEXISTENT", test_db)
        assert exists is False

    async def test_comment_exists_true_when_exists(self, test_db, sample_document_data, sample_comment_data):
        """comment_exists returns True when comment exists."""
        # Store document first
        await DocumentRepository.store_document(sample_document_data, test_db)

        # Store comment
        await CommentRepository.store_comment(
            sample_comment_data,
            sample_document_data["id"],
            conn=test_db
        )

        exists = await CommentRepository.comment_exists(sample_comment_data["id"], test_db)
        assert exists is True

    async def test_store_comment(self, test_db, sample_document_data, sample_comment_data):
        """Store comment in database."""
        # Store document first
        await DocumentRepository.store_document(sample_document_data, test_db)

        # Store comment
        await CommentRepository.store_comment(
            sample_comment_data,
            sample_document_data["id"],
            category="Physicians & Surgeons",
            sentiment="for",
            conn=test_db
        )

        # Verify it was stored
        async with test_db.cursor() as cur:
            await cur.execute(
                """
                SELECT id, document_id, comment_text, category, sentiment,
                       first_name, last_name, organization
                FROM comments WHERE id = %s
                """,
                (sample_comment_data["id"],)
            )
            row = await cur.fetchone()

        assert row is not None
        assert row[0] == sample_comment_data["id"]
        assert row[1] == sample_document_data["id"]
        assert row[2] == sample_comment_data["attributes"]["comment"]
        assert row[3] == "Physicians & Surgeons"
        assert row[4] == "for"
        assert row[5] == sample_comment_data["attributes"]["firstName"]
        assert row[6] == sample_comment_data["attributes"]["lastName"]
        assert row[7] == sample_comment_data["attributes"]["organization"]

    async def test_store_comment_upsert_preserves_classification(self, test_db, sample_document_data, sample_comment_data):
        """Upserting comment preserves existing classification."""
        # Store document first
        await DocumentRepository.store_document(sample_document_data, test_db)

        # Store comment with classification
        await CommentRepository.store_comment(
            sample_comment_data,
            sample_document_data["id"],
            category="Physicians & Surgeons",
            sentiment="for",
            conn=test_db
        )

        # Update comment without classification
        updated_data = sample_comment_data.copy()
        updated_data["attributes"]["comment"] = "Updated comment text"

        await CommentRepository.store_comment(
            updated_data,
            sample_document_data["id"],
            conn=test_db
        )

        # Verify classification was preserved
        async with test_db.cursor() as cur:
            await cur.execute(
                "SELECT comment_text, category, sentiment FROM comments WHERE id = %s",
                (sample_comment_data["id"],)
            )
            row = await cur.fetchone()

        assert row[0] == "Updated comment text"
        assert row[1] == "Physicians & Surgeons"  # Preserved
        assert row[2] == "for"  # Preserved

    async def test_update_comment_classification(self, test_db, sample_document_data, sample_comment_data):
        """Update comment classification."""
        # Store document and comment
        await DocumentRepository.store_document(sample_document_data, test_db)
        await CommentRepository.store_comment(
            sample_comment_data,
            sample_document_data["id"],
            conn=test_db
        )

        # Update classification
        await CommentRepository.update_comment_classification(
            sample_comment_data["id"],
            "Patients & Caregivers",
            "against",
            test_db
        )

        # Verify update
        async with test_db.cursor() as cur:
            await cur.execute(
                "SELECT category, sentiment FROM comments WHERE id = %s",
                (sample_comment_data["id"],)
            )
            row = await cur.fetchone()

        assert row[0] == "Patients & Caregivers"
        assert row[1] == "against"

    async def test_get_comments_for_document(self, test_db, sample_document_data):
        """Get all comments for a document."""
        # Store document
        await DocumentRepository.store_document(sample_document_data, test_db)
        doc_id = sample_document_data["id"]

        # Store multiple comments
        async with test_db.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO comments (id, document_id, comment_text, first_name, last_name, organization)
                VALUES
                    ('C1', %s, 'Comment 1', 'John', 'Doe', 'Org A'),
                    ('C2', %s, 'Comment 2', 'Jane', 'Smith', 'Org B'),
                    ('C3', %s, 'Comment 3', 'Bob', 'Jones', NULL)
                """,
                (doc_id, doc_id, doc_id)
            )
        await test_db.commit()

        # Get comments
        comments = await CommentRepository.get_comments_for_document(doc_id, test_db)

        assert len(comments) == 3
        # Verify structure: (id, comment_text, first_name, last_name, organization)
        assert comments[0][0] == 'C1'
        assert comments[0][1] == 'Comment 1'
        assert comments[0][2] == 'John'
        assert comments[1][0] == 'C2'
        assert comments[2][4] is None  # NULL organization


@pytest.mark.integration
class TestCommentChunkRepository:
    """Test comment chunk repository with real database."""

    async def test_delete_chunks_for_comment(self, test_db, sample_document_data, sample_comment_data):
        """Delete chunks for a comment."""
        # Setup: store document and comment
        await DocumentRepository.store_document(sample_document_data, test_db)
        await CommentRepository.store_comment(
            sample_comment_data,
            sample_document_data["id"],
            conn=test_db
        )

        # Store some chunks
        async with test_db.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO comment_chunks (comment_id, chunk_text, chunk_index, embedding)
                VALUES
                    (%s, 'Chunk 1', 0, %s),
                    (%s, 'Chunk 2', 1, %s)
                """,
                (
                    sample_comment_data["id"], [0.1] * 1536,
                    sample_comment_data["id"], [0.2] * 1536
                )
            )
        await test_db.commit()

        # Delete chunks
        await CommentChunkRepository.delete_chunks_for_comment(
            sample_comment_data["id"],
            test_db
        )

        # Verify deletion
        async with test_db.cursor() as cur:
            await cur.execute(
                "SELECT COUNT(*) FROM comment_chunks WHERE comment_id = %s",
                (sample_comment_data["id"],)
            )
            count = (await cur.fetchone())[0]

        assert count == 0

    async def test_store_comment_chunks(self, test_db, sample_document_data, sample_comment_data):
        """Store comment chunks with embeddings."""
        # Setup: store document and comment
        await DocumentRepository.store_document(sample_document_data, test_db)
        await CommentRepository.store_comment(
            sample_comment_data,
            sample_document_data["id"],
            conn=test_db
        )

        # Store chunks
        chunks_with_embeddings = [
            ("First chunk of text", [0.1] * 1536),
            ("Second chunk of text", [0.2] * 1536),
            ("Third chunk of text", [0.3] * 1536)
        ]

        await CommentChunkRepository.store_comment_chunks(
            sample_comment_data["id"],
            chunks_with_embeddings,
            test_db
        )

        # Verify storage
        async with test_db.cursor() as cur:
            await cur.execute(
                """
                SELECT chunk_text, chunk_index, embedding
                FROM comment_chunks
                WHERE comment_id = %s
                ORDER BY chunk_index
                """,
                (sample_comment_data["id"],)
            )
            rows = await cur.fetchall()

        assert len(rows) == 3
        assert rows[0][0] == "First chunk of text"
        assert rows[0][1] == 0
        assert len(rows[0][2]) == 1536
        assert rows[1][1] == 1
        assert rows[2][1] == 2

    async def test_store_comment_chunks_replaces_existing(self, test_db, sample_document_data, sample_comment_data):
        """Storing chunks replaces existing chunks."""
        # Setup: store document and comment
        await DocumentRepository.store_document(sample_document_data, test_db)
        await CommentRepository.store_comment(
            sample_comment_data,
            sample_document_data["id"],
            conn=test_db
        )

        # Store initial chunks
        chunks1 = [("Old chunk 1", [0.1] * 1536), ("Old chunk 2", [0.2] * 1536)]
        await CommentChunkRepository.store_comment_chunks(
            sample_comment_data["id"],
            chunks1,
            test_db
        )

        # Store new chunks (should replace)
        chunks2 = [("New chunk", [0.9] * 1536)]
        await CommentChunkRepository.store_comment_chunks(
            sample_comment_data["id"],
            chunks2,
            test_db
        )

        # Verify only new chunks exist
        async with test_db.cursor() as cur:
            await cur.execute(
                "SELECT chunk_text FROM comment_chunks WHERE comment_id = %s",
                (sample_comment_data["id"],)
            )
            rows = await cur.fetchall()

        assert len(rows) == 1
        assert rows[0][0] == "New chunk"

    async def test_store_empty_chunks_list(self, test_db, sample_document_data, sample_comment_data):
        """Storing empty chunks list does nothing."""
        # Setup: store document and comment
        await DocumentRepository.store_document(sample_document_data, test_db)
        await CommentRepository.store_comment(
            sample_comment_data,
            sample_document_data["id"],
            conn=test_db
        )

        # Store empty list
        await CommentChunkRepository.store_comment_chunks(
            sample_comment_data["id"],
            [],
            test_db
        )

        # Verify no chunks stored
        async with test_db.cursor() as cur:
            await cur.execute(
                "SELECT COUNT(*) FROM comment_chunks WHERE comment_id = %s",
                (sample_comment_data["id"],)
            )
            count = (await cur.fetchone())[0]

        assert count == 0


@pytest.mark.integration
class TestCascadingDeletes:
    """Test cascading deletes work correctly."""

    async def test_delete_document_cascades_to_comments_and_chunks(self, test_db, sample_document_data, sample_comment_data):
        """Deleting document cascades to comments and chunks."""
        # Store document, comment, and chunks
        await DocumentRepository.store_document(sample_document_data, test_db)
        await CommentRepository.store_comment(
            sample_comment_data,
            sample_document_data["id"],
            conn=test_db
        )
        await CommentChunkRepository.store_comment_chunks(
            sample_comment_data["id"],
            [("Chunk", [0.1] * 1536)],
            test_db
        )

        # Delete document
        async with test_db.cursor() as cur:
            await cur.execute(
                "DELETE FROM documents WHERE id = %s",
                (sample_document_data["id"],)
            )
        await test_db.commit()

        # Verify comments and chunks were also deleted
        async with test_db.cursor() as cur:
            await cur.execute("SELECT COUNT(*) FROM comments WHERE id = %s", (sample_comment_data["id"],))
            comment_count = (await cur.fetchone())[0]

            await cur.execute("SELECT COUNT(*) FROM comment_chunks WHERE comment_id = %s", (sample_comment_data["id"],))
            chunk_count = (await cur.fetchone())[0]

        assert comment_count == 0, "Comments should be deleted when document is deleted"
        assert chunk_count == 0, "Chunks should be deleted when document is deleted"
