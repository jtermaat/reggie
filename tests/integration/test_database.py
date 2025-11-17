"""Integration tests for database operations using SQLite"""

import pytest

from reggie.db.repositories.document_repository import DocumentRepository
from reggie.db.repositories.comment_repository import CommentRepository
from reggie.db.repositories.chunk_repository import ChunkRepository


@pytest.mark.integration
class TestDocumentRepository:
    """Test document repository with real database."""

    def test_store_document(self, test_db, sample_document_data):
        """Store document in database."""
        repo = DocumentRepository(test_db)
        repo.store_document(sample_document_data)
        test_db.commit()

        # Verify it was stored
        cur = test_db.execute(
            "SELECT id, title, object_id, docket_id FROM documents WHERE id = ?",
            (sample_document_data["id"],)
        )
        row = cur.fetchone()

        assert row is not None, "Document should be stored in database"
        assert row[0] == sample_document_data["id"]
        assert row[1] == sample_document_data["attributes"]["title"]
        assert row[2] == sample_document_data["attributes"]["objectId"]
        assert row[3] == sample_document_data["attributes"]["docketId"]

    def test_store_document_upsert_updates_existing(self, test_db, sample_document_data):
        """Upserting same document updates existing record."""
        repo = DocumentRepository(test_db)

        # Store initial document
        repo.store_document(sample_document_data)
        test_db.commit()

        # Update document data
        updated_data = sample_document_data.copy()
        updated_data["attributes"] = sample_document_data["attributes"].copy()
        updated_data["attributes"]["title"] = "Updated Title"

        # Upsert with updated data
        repo.store_document(updated_data)
        test_db.commit()

        # Verify only one record exists with updated title
        cur = test_db.execute(
            "SELECT COUNT(*), title FROM documents WHERE id = ? GROUP BY title",
            (sample_document_data["id"],)
        )
        row = cur.fetchone()

        assert row[0] == 1, "Should have exactly one document"
        assert row[1] == "Updated Title", "Title should be updated"

    def test_list_documents_empty(self, test_db):
        """List documents returns empty list when no documents exist."""
        repo = DocumentRepository(test_db)
        documents = repo.list_documents()
        assert documents == []

    def test_list_documents_with_comment_counts(self, test_db, sample_document_data):
        """List documents includes comment counts and stats."""
        doc_repo = DocumentRepository(test_db)

        # Store document
        doc_repo.store_document(sample_document_data)
        test_db.commit()
        doc_id = sample_document_data["id"]

        # Store some comments
        test_db.execute(
            """
            INSERT INTO comments (id, document_id, comment_text, category)
            VALUES
                ('C1', ?, 'Comment 1', 'Physicians & Surgeons'),
                ('C2', ?, 'Comment 2', 'Physicians & Surgeons'),
                ('C3', ?, 'Comment 3', 'Patients & Caregivers')
            """,
            (doc_id, doc_id, doc_id)
        )
        test_db.commit()

        # List documents
        documents = doc_repo.list_documents()

        assert len(documents) == 1
        doc = documents[0]
        assert doc["id"] == doc_id
        assert doc["comment_count"] == 3
        assert doc["unique_categories"] == 2  # Two different categories

    def test_list_documents_ordered_by_created_at(self, test_db):
        """List documents returns newest first."""
        import time
        repo = DocumentRepository(test_db)

        # Store multiple documents with small delays to ensure distinct timestamps
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
            repo.store_document(doc_data)
            test_db.commit()
            if i < 2:  # Only delay between documents, not after the last one
                time.sleep(1.1)  # SQLite timestamps have second precision

        documents = repo.list_documents()

        assert len(documents) == 3
        # Should be ordered newest first (DESC)
        assert documents[0]["id"] == "DOC-2"
        assert documents[1]["id"] == "DOC-1"
        assert documents[2]["id"] == "DOC-0"


@pytest.mark.integration
class TestCommentRepository:
    """Test comment repository with real database."""

    def test_comment_exists_false_when_not_exists(self, test_db):
        """comment_exists returns False when comment doesn't exist."""
        repo = CommentRepository(test_db)
        exists = repo.comment_exists("NONEXISTENT")
        assert exists is False

    def test_comment_exists_true_when_exists(self, test_db, sample_document_data, sample_comment_data):
        """comment_exists returns True when comment exists."""
        doc_repo = DocumentRepository(test_db)
        comment_repo = CommentRepository(test_db)

        # Store document first
        doc_repo.store_document(sample_document_data)
        test_db.commit()

        # Store comment
        comment_repo.store_comment(
            sample_comment_data,
            sample_document_data["id"]
        )
        test_db.commit()

        exists = comment_repo.comment_exists(sample_comment_data["id"])
        assert exists is True

    def test_store_comment(self, test_db, sample_document_data, sample_comment_data):
        """Store comment in database."""
        doc_repo = DocumentRepository(test_db)
        comment_repo = CommentRepository(test_db)

        # Store document first
        doc_repo.store_document(sample_document_data)
        test_db.commit()

        # Store comment
        comment_repo.store_comment(
            sample_comment_data,
            sample_document_data["id"],
            category="Physicians & Surgeons",
            sentiment="for"
        )
        test_db.commit()

        # Verify it was stored
        cur = test_db.execute(
            """
            SELECT id, document_id, comment_text, category, sentiment,
                   first_name, last_name, organization
            FROM comments WHERE id = ?
            """,
            (sample_comment_data["id"],)
        )
        row = cur.fetchone()

        assert row is not None
        assert row[0] == sample_comment_data["id"]
        assert row[1] == sample_document_data["id"]
        assert row[2] == sample_comment_data["attributes"]["comment"]
        assert row[3] == "Physicians & Surgeons"
        assert row[4] == "for"
        assert row[5] == sample_comment_data["attributes"]["firstName"]
        assert row[6] == sample_comment_data["attributes"]["lastName"]
        assert row[7] == sample_comment_data["attributes"]["organization"]

    def test_store_comment_upsert_preserves_classification(self, test_db, sample_document_data, sample_comment_data):
        """Upserting comment preserves existing classification."""
        doc_repo = DocumentRepository(test_db)
        comment_repo = CommentRepository(test_db)

        # Store document first
        doc_repo.store_document(sample_document_data)
        test_db.commit()

        # Store comment with classification
        comment_repo.store_comment(
            sample_comment_data,
            sample_document_data["id"],
            category="Physicians & Surgeons",
            sentiment="for"
        )
        test_db.commit()

        # Update comment without classification
        updated_data = sample_comment_data.copy()
        updated_data["attributes"] = sample_comment_data["attributes"].copy()
        updated_data["attributes"]["comment"] = "Updated comment text"

        comment_repo.store_comment(
            updated_data,
            sample_document_data["id"]
        )
        test_db.commit()

        # Verify classification was preserved
        cur = test_db.execute(
            "SELECT comment_text, category, sentiment FROM comments WHERE id = ?",
            (sample_comment_data["id"],)
        )
        row = cur.fetchone()

        assert row[0] == "Updated comment text"
        assert row[1] == "Physicians & Surgeons"  # Preserved
        assert row[2] == "for"  # Preserved

    def test_update_comment_classification(self, test_db, sample_document_data, sample_comment_data):
        """Update comment classification."""
        doc_repo = DocumentRepository(test_db)
        comment_repo = CommentRepository(test_db)

        # Store document and comment
        doc_repo.store_document(sample_document_data)
        comment_repo.store_comment(
            sample_comment_data,
            sample_document_data["id"]
        )
        test_db.commit()

        # Update classification
        comment_repo.update_comment_classification(
            sample_comment_data["id"],
            "Patients & Caregivers",
            "against"
        )
        test_db.commit()

        # Verify update
        cur = test_db.execute(
            "SELECT category, sentiment FROM comments WHERE id = ?",
            (sample_comment_data["id"],)
        )
        row = cur.fetchone()

        assert row[0] == "Patients & Caregivers"
        assert row[1] == "against"

    def test_get_comments_for_document(self, test_db, sample_document_data):
        """Get all comments for a document."""
        doc_repo = DocumentRepository(test_db)
        comment_repo = CommentRepository(test_db)

        # Store document
        doc_repo.store_document(sample_document_data)
        test_db.commit()
        doc_id = sample_document_data["id"]

        # Store multiple comments
        test_db.execute(
            """
            INSERT INTO comments (id, document_id, comment_text, first_name, last_name, organization)
            VALUES
                ('C1', ?, 'Comment 1', 'John', 'Doe', 'Org A'),
                ('C2', ?, 'Comment 2', 'Jane', 'Smith', 'Org B'),
                ('C3', ?, 'Comment 3', 'Bob', 'Jones', NULL)
            """,
            (doc_id, doc_id, doc_id)
        )
        test_db.commit()

        # Get comments
        comments = comment_repo.get_comments_for_document(doc_id)

        assert len(comments) == 3
        # Verify structure: CommentData objects
        assert comments[0].id == 'C1'
        assert comments[0].comment_text == 'Comment 1'
        assert comments[0].first_name == 'John'
        assert comments[1].id == 'C2'
        assert comments[2].organization is None  # NULL organization


@pytest.mark.integration
class TestCommentChunkRepository:
    """Test comment chunk repository with real database."""

    def test_delete_chunks_for_comment(self, test_db, sample_document_data, sample_comment_data):
        """Delete chunks for a comment."""
        doc_repo = DocumentRepository(test_db)
        comment_repo = CommentRepository(test_db)
        chunk_repo = ChunkRepository(test_db)

        # Setup: store document and comment
        doc_repo.store_document(sample_document_data)
        comment_repo.store_comment(
            sample_comment_data,
            sample_document_data["id"]
        )
        test_db.commit()

        # Store some chunks using vector serialization
        from reggie.db.utils.vector_utils import serialize_vector
        test_db.execute(
            """
            INSERT INTO comment_chunks (comment_id, chunk_text, chunk_index, embedding)
            VALUES
                (?, 'Chunk 1', 0, ?),
                (?, 'Chunk 2', 1, ?)
            """,
            (
                sample_comment_data["id"], serialize_vector([0.1] * 1536),
                sample_comment_data["id"], serialize_vector([0.2] * 1536)
            )
        )
        test_db.commit()

        # Delete chunks
        chunk_repo.delete_chunks_for_comment(sample_comment_data["id"])
        test_db.commit()

        # Verify deletion
        cur = test_db.execute(
            "SELECT COUNT(*) FROM comment_chunks WHERE comment_id = ?",
            (sample_comment_data["id"],)
        )
        count = cur.fetchone()[0]

        assert count == 0

    def test_store_comment_chunks(self, test_db, sample_document_data, sample_comment_data):
        """Store comment chunks with embeddings."""
        doc_repo = DocumentRepository(test_db)
        comment_repo = CommentRepository(test_db)
        chunk_repo = ChunkRepository(test_db)

        # Setup: store document and comment
        doc_repo.store_document(sample_document_data)
        comment_repo.store_comment(
            sample_comment_data,
            sample_document_data["id"]
        )
        test_db.commit()

        # Store chunks
        chunks_with_embeddings = [
            ("First chunk of text", [0.1] * 1536),
            ("Second chunk of text", [0.2] * 1536),
            ("Third chunk of text", [0.3] * 1536)
        ]

        chunk_repo.store_comment_chunks(
            sample_comment_data["id"],
            chunks_with_embeddings
        )
        test_db.commit()

        # Verify storage
        cur = test_db.execute(
            """
            SELECT chunk_text, chunk_index
            FROM comment_chunks
            WHERE comment_id = ?
            ORDER BY chunk_index
            """,
            (sample_comment_data["id"],)
        )
        rows = cur.fetchall()

        assert len(rows) == 3
        assert rows[0][0] == "First chunk of text"
        assert rows[0][1] == 0
        assert rows[1][1] == 1
        assert rows[2][1] == 2

    def test_store_comment_chunks_replaces_existing(self, test_db, sample_document_data, sample_comment_data):
        """Storing chunks replaces existing chunks."""
        doc_repo = DocumentRepository(test_db)
        comment_repo = CommentRepository(test_db)
        chunk_repo = ChunkRepository(test_db)

        # Setup: store document and comment
        doc_repo.store_document(sample_document_data)
        comment_repo.store_comment(
            sample_comment_data,
            sample_document_data["id"]
        )
        test_db.commit()

        # Store initial chunks
        chunks1 = [("Old chunk 1", [0.1] * 1536), ("Old chunk 2", [0.2] * 1536)]
        chunk_repo.store_comment_chunks(
            sample_comment_data["id"],
            chunks1
        )
        test_db.commit()

        # Store new chunks (should replace)
        chunks2 = [("New chunk", [0.9] * 1536)]
        chunk_repo.store_comment_chunks(
            sample_comment_data["id"],
            chunks2
        )
        test_db.commit()

        # Verify only new chunks exist
        cur = test_db.execute(
            "SELECT chunk_text FROM comment_chunks WHERE comment_id = ?",
            (sample_comment_data["id"],)
        )
        rows = cur.fetchall()

        assert len(rows) == 1
        assert rows[0][0] == "New chunk"

    def test_store_empty_chunks_list(self, test_db, sample_document_data, sample_comment_data):
        """Storing empty chunks list does nothing."""
        doc_repo = DocumentRepository(test_db)
        comment_repo = CommentRepository(test_db)
        chunk_repo = ChunkRepository(test_db)

        # Setup: store document and comment
        doc_repo.store_document(sample_document_data)
        comment_repo.store_comment(
            sample_comment_data,
            sample_document_data["id"]
        )
        test_db.commit()

        # Store empty list
        chunk_repo.store_comment_chunks(
            sample_comment_data["id"],
            []
        )
        test_db.commit()

        # Verify no chunks stored
        cur = test_db.execute(
            "SELECT COUNT(*) FROM comment_chunks WHERE comment_id = ?",
            (sample_comment_data["id"],)
        )
        count = cur.fetchone()[0]

        assert count == 0


@pytest.mark.integration
class TestCascadingDeletes:
    """Test cascading deletes work correctly."""

    def test_delete_document_cascades_to_comments_and_chunks(self, test_db, sample_document_data, sample_comment_data):
        """Deleting document cascades to comments and chunks."""
        doc_repo = DocumentRepository(test_db)
        comment_repo = CommentRepository(test_db)
        chunk_repo = ChunkRepository(test_db)

        # Store document, comment, and chunks
        doc_repo.store_document(sample_document_data)
        comment_repo.store_comment(
            sample_comment_data,
            sample_document_data["id"]
        )
        chunk_repo.store_comment_chunks(
            sample_comment_data["id"],
            [("Chunk", [0.1] * 1536)]
        )
        test_db.commit()

        # Delete document
        test_db.execute(
            "DELETE FROM documents WHERE id = ?",
            (sample_document_data["id"],)
        )
        test_db.commit()

        # Verify comments and chunks were also deleted
        cur = test_db.execute("SELECT COUNT(*) FROM comments WHERE id = ?", (sample_comment_data["id"],))
        comment_count = cur.fetchone()[0]

        cur = test_db.execute("SELECT COUNT(*) FROM comment_chunks WHERE comment_id = ?", (sample_comment_data["id"],))
        chunk_count = cur.fetchone()[0]

        assert comment_count == 0, "Comments should be deleted when document is deleted"
        assert chunk_count == 0, "Chunks should be deleted when document is deleted"
