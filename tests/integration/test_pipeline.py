"""End-to-end pipeline integration tests with mocked external APIs and real database"""

import pytest

from reggie.pipeline.loader import DocumentLoader
from reggie.pipeline.processor import CommentProcessor
from reggie.db.repository import DocumentRepository, CommentRepository, CommentChunkRepository


@pytest.mark.integration
class TestDocumentLoadingPipeline:
    """Test document loading end-to-end flow."""

    async def test_load_document_stores_metadata_and_comments(
        self,
        test_db,
        mock_regulations_api,
        mocker
    ):
        """Load document flow: mocked API → real DB storage."""
        # Mock sleep to speed up test
        mocker.patch("asyncio.sleep", new_callable=mocker.AsyncMock)

        # Setup mock API responses
        doc_id = "CMS-2024-0001-0001"
        object_id = "test-object-id"

        # Mock document response
        mock_regulations_api.httpx_mock.add_response(
            url=f"{mock_regulations_api.base_url}/documents/{doc_id}",
            json={
                "data": {
                    "id": doc_id,
                    "attributes": {
                        "objectId": object_id,
                        "title": "Test Document",
                        "docketId": "TEST-DOCKET",
                        "documentType": "Rule",
                        "postedDate": "2024-01-01T00:00:00Z"
                    }
                }
            }
        )

        # Mock comments list (single page)
        mock_regulations_api.httpx_mock.add_response(
            url=f"{mock_regulations_api.base_url}/comments",
            json={
                "data": [
                    {"id": "C1"},
                    {"id": "C2"}
                ],
                "meta": {"hasNextPage": False}
            },
            match_params={
                "filter[commentOnId]": object_id,
                "page[size]": "250",
                "page[number]": "1"
            }
        )

        # Mock comment details
        mock_regulations_api.httpx_mock.add_response(
            url=f"{mock_regulations_api.base_url}/comments/C1",
            json={
                "data": {
                    "id": "C1",
                    "attributes": {
                        "comment": "First comment",
                        "firstName": "John",
                        "lastName": "Doe",
                        "organization": "Test Org",
                        "postedDate": "2024-01-15T10:00:00Z"
                    }
                }
            }
        )

        mock_regulations_api.httpx_mock.add_response(
            url=f"{mock_regulations_api.base_url}/comments/C2",
            json={
                "data": {
                    "id": "C2",
                    "attributes": {
                        "comment": "Second comment",
                        "firstName": "Jane",
                        "lastName": "Smith",
                        "postedDate": "2024-01-15T11:00:00Z"
                    }
                }
            }
        )

        # Get connection string from test_db
        from reggie.config import DatabaseConfig
        db_config = DatabaseConfig()
        connection_string = db_config.connection_string

        # Load document
        loader = DocumentLoader(connection_string=connection_string)
        stats = await loader.load_document(doc_id)

        # Verify stats
        assert stats["comments_processed"] == 2
        assert stats["errors"] == 0

        # Verify document was stored in real DB
        async with test_db.cursor() as cur:
            await cur.execute(
                "SELECT id, title, object_id FROM documents WHERE id = %s",
                (doc_id,)
            )
            doc_row = await cur.fetchone()

        assert doc_row is not None
        assert doc_row[0] == doc_id
        assert doc_row[1] == "Test Document"
        assert doc_row[2] == object_id

        # Verify comments were stored
        async with test_db.cursor() as cur:
            await cur.execute(
                "SELECT id, comment_text FROM comments WHERE document_id = %s ORDER BY id",
                (doc_id,)
            )
            comment_rows = await cur.fetchall()

        assert len(comment_rows) == 2
        assert comment_rows[0][0] == "C1"
        assert comment_rows[0][1] == "First comment"
        assert comment_rows[1][0] == "C2"
        assert comment_rows[1][1] == "Second comment"

    async def test_load_document_skips_existing_comments(
        self,
        test_db,
        mock_regulations_api,
        mocker
    ):
        """Load document skips comments that already exist."""
        mocker.patch("asyncio.sleep", new_callable=mocker.AsyncMock)

        doc_id = "CMS-2024-0001-0001"
        object_id = "test-object-id"

        # Setup mock responses
        mock_regulations_api.httpx_mock.add_response(
            url=f"{mock_regulations_api.base_url}/documents/{doc_id}",
            json={
                "data": {
                    "id": doc_id,
                    "attributes": {
                        "objectId": object_id,
                        "title": "Test Document",
                        "docketId": "TEST",
                        "documentType": "Rule",
                        "postedDate": "2024-01-01T00:00:00Z"
                    }
                }
            }
        )

        mock_regulations_api.httpx_mock.add_response(
            url=f"{mock_regulations_api.base_url}/comments",
            json={
                "data": [{"id": "C1"}],
                "meta": {"hasNextPage": False}
            },
            match_params={
                "filter[commentOnId]": object_id,
                "page[size]": "250",
                "page[number]": "1"
            }
        )

        mock_regulations_api.httpx_mock.add_response(
            url=f"{mock_regulations_api.base_url}/comments/C1",
            json={
                "data": {
                    "id": "C1",
                    "attributes": {
                        "comment": "Test comment",
                        "postedDate": "2024-01-15T10:00:00Z"
                    }
                }
            }
        )

        # Get connection string
        from reggie.config import DatabaseConfig
        db_config = DatabaseConfig()
        connection_string = db_config.connection_string

        # First load
        loader = DocumentLoader(connection_string=connection_string)
        stats1 = await loader.load_document(doc_id)
        assert stats1["comments_processed"] == 1

        # Second load (should skip existing comment)
        loader2 = DocumentLoader(connection_string=connection_string)
        stats2 = await loader2.load_document(doc_id)
        assert stats2["comments_processed"] == 0  # Skipped

        # Verify still only 1 comment in DB
        async with test_db.cursor() as cur:
            await cur.execute(
                "SELECT COUNT(*) FROM comments WHERE document_id = %s",
                (doc_id,)
            )
            count = (await cur.fetchone())[0]

        assert count == 1


@pytest.mark.integration
class TestCommentProcessingPipeline:
    """Test comment processing end-to-end flow."""

    async def test_process_comments_flow(
        self,
        test_db,
        mock_openai,
        sample_document_data
    ):
        """Process comments flow: real DB → mocked categorization → mocked embedding → DB storage."""
        # Store document and comments in DB first
        await DocumentRepository.store_document(sample_document_data, test_db)
        doc_id = sample_document_data["id"]

        async with test_db.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO comments (id, document_id, comment_text, first_name, organization)
                VALUES
                    ('C1', %s, 'I support this regulation', 'John', 'Medical Association'),
                    ('C2', %s, 'This needs more study', 'Jane', 'Research Institute')
                """,
                (doc_id, doc_id)
            )
        await test_db.commit()

        # Get connection string
        from reggie.config import DatabaseConfig
        db_config = DatabaseConfig()
        connection_string = db_config.connection_string

        # Process comments
        processor = CommentProcessor(connection_string=connection_string)
        stats = await processor.process_comments(doc_id, batch_size=2)

        # Verify stats
        assert stats["comments_processed"] == 2
        assert stats["chunks_created"] > 0
        assert stats["errors"] == 0

        # Verify classifications were stored
        async with test_db.cursor() as cur:
            await cur.execute(
                "SELECT category, sentiment FROM comments WHERE document_id = %s ORDER BY id",
                (doc_id,)
            )
            rows = await cur.fetchall()

        assert len(rows) == 2
        # Mock returns "Physicians & Surgeons" and "for"
        assert rows[0][0] == "Physicians & Surgeons"
        assert rows[0][1] == "for"
        assert rows[1][0] == "Physicians & Surgeons"
        assert rows[1][1] == "for"

        # Verify chunks were stored
        async with test_db.cursor() as cur:
            await cur.execute(
                """
                SELECT comment_id, chunk_text, embedding
                FROM comment_chunks
                WHERE comment_id IN ('C1', 'C2')
                ORDER BY comment_id, chunk_index
                """
            )
            chunk_rows = await cur.fetchall()

        assert len(chunk_rows) > 0
        # Verify embeddings are stored
        for row in chunk_rows:
            assert row[2] is not None  # embedding
            assert len(row[2]) == 1536

    async def test_process_comments_handles_errors_gracefully(
        self,
        test_db,
        mocker,
        sample_document_data
    ):
        """Process comments handles errors gracefully."""
        from reggie.models import CommentClassification, Category, Sentiment

        # Store document and comments
        await DocumentRepository.store_document(sample_document_data, test_db)
        doc_id = sample_document_data["id"]

        async with test_db.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO comments (id, document_id, comment_text)
                VALUES ('C1', %s, 'Test comment')
                """,
                (doc_id,)
            )
        await test_db.commit()

        # Mock categorizer to fail
        mock_categorizer = mocker.patch(
            "reggie.pipeline.processor.CommentCategorizer"
        )
        mock_categorizer_instance = mock_categorizer.return_value
        mock_categorizer_instance.categorize_batch = mocker.AsyncMock(
            side_effect=Exception("API Error")
        )

        # Get connection string
        from reggie.config import DatabaseConfig
        db_config = DatabaseConfig()
        connection_string = db_config.connection_string

        # Process should fail but be caught
        processor = CommentProcessor(connection_string=connection_string)

        with pytest.raises(Exception):
            await processor.process_comments(doc_id)

    async def test_process_comments_empty_document(self, test_db, sample_document_data):
        """Process comments handles document with no comments."""
        # Store document without comments
        await DocumentRepository.store_document(sample_document_data, test_db)

        # Get connection string
        from reggie.config import DatabaseConfig
        db_config = DatabaseConfig()
        connection_string = db_config.connection_string

        processor = CommentProcessor(connection_string=connection_string)
        stats = await processor.process_comments(sample_document_data["id"])

        assert stats["comments_processed"] == 0
        assert stats["chunks_created"] == 0


@pytest.mark.integration
class TestFullEndToEndPipeline:
    """Test complete pipeline from load to process."""

    async def test_full_pipeline_load_then_process(
        self,
        test_db,
        mock_regulations_api,
        mock_openai,
        mocker
    ):
        """Complete pipeline: load document → process comments."""
        mocker.patch("asyncio.sleep", new_callable=mocker.AsyncMock)

        doc_id = "CMS-2024-0001-0001"
        object_id = "test-object-id"

        # Mock API responses for loading
        mock_regulations_api.httpx_mock.add_response(
            url=f"{mock_regulations_api.base_url}/documents/{doc_id}",
            json={
                "data": {
                    "id": doc_id,
                    "attributes": {
                        "objectId": object_id,
                        "title": "Medicare Test Rule",
                        "docketId": "CMS-2024-0001",
                        "documentType": "Rule",
                        "postedDate": "2024-01-01T00:00:00Z"
                    }
                }
            }
        )

        mock_regulations_api.httpx_mock.add_response(
            url=f"{mock_regulations_api.base_url}/comments",
            json={
                "data": [{"id": "C1"}],
                "meta": {"hasNextPage": False}
            },
            match_params={
                "filter[commentOnId]": object_id,
                "page[size]": "250",
                "page[number]": "1"
            }
        )

        mock_regulations_api.httpx_mock.add_response(
            url=f"{mock_regulations_api.base_url}/comments/C1",
            json={
                "data": {
                    "id": "C1",
                    "attributes": {
                        "comment": "As a physician, I support this regulation.",
                        "firstName": "Dr. John",
                        "lastName": "Smith",
                        "organization": "American Medical Association",
                        "postedDate": "2024-01-15T10:00:00Z"
                    }
                }
            }
        )

        # Get connection string
        from reggie.config import DatabaseConfig
        db_config = DatabaseConfig()
        connection_string = db_config.connection_string

        # Step 1: Load document
        loader = DocumentLoader(connection_string=connection_string)
        load_stats = await loader.load_document(doc_id)

        assert load_stats["comments_processed"] == 1
        assert load_stats["errors"] == 0

        # Step 2: Process comments
        processor = CommentProcessor(connection_string=connection_string)
        process_stats = await processor.process_comments(doc_id)

        assert process_stats["comments_processed"] == 1
        assert process_stats["chunks_created"] > 0
        assert process_stats["errors"] == 0

        # Verify final state in database
        async with test_db.cursor() as cur:
            # Check document
            await cur.execute(
                "SELECT title FROM documents WHERE id = %s",
                (doc_id,)
            )
            doc_title = (await cur.fetchone())[0]
            assert doc_title == "Medicare Test Rule"

            # Check comment with classification
            await cur.execute(
                "SELECT comment_text, category, sentiment FROM comments WHERE id = 'C1'"
            )
            comment_row = await cur.fetchone()
            assert "physician" in comment_row[0].lower()
            assert comment_row[1] == "Physicians & Surgeons"
            assert comment_row[2] == "for"

            # Check chunks exist
            await cur.execute(
                "SELECT COUNT(*) FROM comment_chunks WHERE comment_id = 'C1'"
            )
            chunk_count = (await cur.fetchone())[0]
            assert chunk_count > 0
