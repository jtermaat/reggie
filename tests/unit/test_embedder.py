"""Unit tests for CommentEmbedder"""

import pytest
from unittest.mock import AsyncMock

from reggie.pipeline.embedder import CommentEmbedder
from reggie.exceptions import ConfigurationException


@pytest.mark.unit
class TestEmbedderInitialization:
    """Test embedder initialization."""

    def test_embedder_requires_api_key(self, mocker):
        """Embedder raises error without OpenAI API key."""
        mocker.patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False)

        with pytest.raises(ConfigurationException) as exc_info:
            CommentEmbedder()

        assert "OPENAI_API_KEY" in str(exc_info.value)

    def test_embedder_initialization_with_api_key(self, mocker):
        """Embedder initializes successfully with API key."""
        mocker.patch("reggie.pipeline.embedder.OpenAIEmbeddings")
        mocker.patch("reggie.pipeline.embedder.tiktoken")

        embedder = CommentEmbedder(openai_api_key="sk-test-key")
        assert embedder is not None

    def test_embedder_uses_custom_chunk_params(self, mocker):
        """Embedder accepts custom chunk size and overlap."""
        mocker.patch("reggie.pipeline.embedder.OpenAIEmbeddings")
        mocker.patch("reggie.pipeline.embedder.tiktoken")
        mock_splitter = mocker.patch(
            "reggie.pipeline.embedder.RecursiveCharacterTextSplitter.from_tiktoken_encoder"
        )

        embedder = CommentEmbedder(
            openai_api_key="sk-test-key",
            chunk_size=500,
            chunk_overlap=100
        )

        # Verify splitter was called with custom params
        mock_splitter.assert_called_once()
        call_kwargs = mock_splitter.call_args[1]
        assert call_kwargs["chunk_size"] == 500
        assert call_kwargs["chunk_overlap"] == 100


@pytest.mark.unit
class TestTextChunking:
    """Test text chunking logic."""

    def test_chunk_text_empty_string_returns_empty(self, mocker):
        """Chunking empty text returns empty list."""
        mocker.patch("reggie.pipeline.embedder.OpenAIEmbeddings")
        mocker.patch("reggie.pipeline.embedder.tiktoken")

        embedder = CommentEmbedder(openai_api_key="sk-test-key")
        chunks = embedder.chunk_text("")

        assert chunks == []

    def test_chunk_text_whitespace_only_returns_empty(self, mocker):
        """Chunking whitespace-only text returns empty list."""
        mocker.patch("reggie.pipeline.embedder.OpenAIEmbeddings")
        mocker.patch("reggie.pipeline.embedder.tiktoken")

        embedder = CommentEmbedder(openai_api_key="sk-test-key")
        chunks = embedder.chunk_text("   \n\t  ")

        assert chunks == []

    def test_chunk_text_small_text_single_chunk(self, mocker):
        """Small text produces single chunk."""
        mocker.patch("reggie.pipeline.embedder.OpenAIEmbeddings")
        mocker.patch("reggie.pipeline.embedder.tiktoken")

        # Mock the text splitter
        mock_splitter = mocker.MagicMock()
        mock_splitter.split_text.return_value = ["This is a short comment"]
        mocker.patch(
            "reggie.pipeline.embedder.RecursiveCharacterTextSplitter.from_tiktoken_encoder",
            return_value=mock_splitter
        )

        embedder = CommentEmbedder(openai_api_key="sk-test-key")
        chunks = embedder.chunk_text("This is a short comment")

        assert len(chunks) == 1
        assert chunks[0] == "This is a short comment"

    def test_chunk_text_long_text_multiple_chunks(self, mocker):
        """Long text produces multiple chunks."""
        mocker.patch("reggie.pipeline.embedder.OpenAIEmbeddings")
        mocker.patch("reggie.pipeline.embedder.tiktoken")

        # Mock the text splitter to return multiple chunks
        mock_splitter = mocker.MagicMock()
        mock_splitter.split_text.return_value = ["Chunk 1", "Chunk 2", "Chunk 3"]
        mocker.patch(
            "reggie.pipeline.embedder.RecursiveCharacterTextSplitter.from_tiktoken_encoder",
            return_value=mock_splitter
        )

        embedder = CommentEmbedder(openai_api_key="sk-test-key")
        chunks = embedder.chunk_text("A very long comment that gets split into multiple chunks")

        assert len(chunks) == 3
        assert chunks == ["Chunk 1", "Chunk 2", "Chunk 3"]


@pytest.mark.unit
class TestEmbedding:
    """Test embedding generation."""

    async def test_embed_chunks_empty_list_returns_empty(self, mocker):
        """Embedding empty list returns empty tuple with zero token count."""
        mocker.patch("reggie.pipeline.embedder.OpenAIEmbeddings")
        mocker.patch("reggie.pipeline.embedder.tiktoken")

        embedder = CommentEmbedder(openai_api_key="sk-test-key")
        embeddings = await embedder.embed_chunks([])

        assert embeddings == ([], 0)

    async def test_embed_chunks_success(self, mocker):
        """Embedding chunks returns vectors and token count."""
        mock_embeddings = mocker.patch("reggie.pipeline.embedder.OpenAIEmbeddings")
        mock_instance = mock_embeddings.return_value
        mock_instance.aembed_documents = AsyncMock(
            return_value=[[0.1] * 1536, [0.2] * 1536]
        )
        mocker.patch("reggie.pipeline.embedder.tiktoken")

        embedder = CommentEmbedder(openai_api_key="sk-test-key")
        embeddings, token_count = await embedder.embed_chunks(["Chunk 1", "Chunk 2"])

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1536
        assert len(embeddings[1]) == 1536
        assert embeddings[0][0] == 0.1
        assert embeddings[1][0] == 0.2
        assert isinstance(token_count, int)

    async def test_embed_chunks_error_returns_zero_vectors(self, mocker):
        """Embedding error returns zero vectors and token count."""
        mock_embeddings = mocker.patch("reggie.pipeline.embedder.OpenAIEmbeddings")
        mock_instance = mock_embeddings.return_value
        mock_instance.aembed_documents = AsyncMock(
            side_effect=Exception("API Error")
        )
        mocker.patch("reggie.pipeline.embedder.tiktoken")

        embedder = CommentEmbedder(openai_api_key="sk-test-key")
        embeddings, token_count = await embedder.embed_chunks(["Chunk 1"])

        assert len(embeddings) == 1
        assert embeddings[0] == [0.0] * 1536  # Zero vector
        assert isinstance(token_count, int)

    async def test_embed_chunks_batch_processing(self, mocker):
        """Embedding processes in batches and returns token count."""
        mock_embeddings = mocker.patch("reggie.pipeline.embedder.OpenAIEmbeddings")
        mock_instance = mock_embeddings.return_value

        # Track batch sizes
        batch_sizes = []

        async def mock_embed(chunks):
            batch_sizes.append(len(chunks))
            return [[0.1] * 1536] * len(chunks)

        mock_instance.aembed_documents = mock_embed
        mocker.patch("reggie.pipeline.embedder.tiktoken")
        mocker.patch("asyncio.sleep", new_callable=AsyncMock)

        embedder = CommentEmbedder(openai_api_key="sk-test-key")

        # Create 250 chunks, batch_size=100
        chunks = [f"Chunk {i}" for i in range(250)]
        embeddings, token_count = await embedder.embed_chunks(chunks, batch_size=100)

        assert len(embeddings) == 250
        # Should have 3 batches: 100, 100, 50
        assert batch_sizes == [100, 100, 50]
        assert isinstance(token_count, int)


@pytest.mark.unit
class TestChunkAndEmbed:
    """Test combined chunking and embedding."""

    async def test_chunk_and_embed_empty_text(self, mocker):
        """Chunk and embed empty text returns empty tuple with zero token count."""
        mocker.patch("reggie.pipeline.embedder.OpenAIEmbeddings")
        mocker.patch("reggie.pipeline.embedder.tiktoken")

        embedder = CommentEmbedder(openai_api_key="sk-test-key")
        result = await embedder.chunk_and_embed("")

        assert result == ([], 0)

    async def test_chunk_and_embed_returns_tuples(self, mocker):
        """Chunk and embed returns (chunk, embedding) tuples and token count."""
        mock_embeddings = mocker.patch("reggie.pipeline.embedder.OpenAIEmbeddings")
        mock_instance = mock_embeddings.return_value
        mock_instance.aembed_documents = AsyncMock(
            return_value=[[0.1] * 1536, [0.2] * 1536]
        )
        mocker.patch("reggie.pipeline.embedder.tiktoken")

        # Mock text splitter
        mock_splitter = mocker.MagicMock()
        mock_splitter.split_text.return_value = ["Chunk 1", "Chunk 2"]
        mocker.patch(
            "reggie.pipeline.embedder.RecursiveCharacterTextSplitter.from_tiktoken_encoder",
            return_value=mock_splitter
        )

        embedder = CommentEmbedder(openai_api_key="sk-test-key")
        result, token_count = await embedder.chunk_and_embed("Some text to chunk and embed")

        assert len(result) == 2
        assert result[0] == ("Chunk 1", [0.1] * 1536)
        assert result[1] == ("Chunk 2", [0.2] * 1536)
        assert isinstance(token_count, int)


@pytest.mark.unit
class TestBatchProcessing:
    """Test batch comment processing."""

    async def test_process_comments_batch_all_success(self, mocker):
        """Batch processing handles all successful comments and returns token counts."""
        mock_embeddings = mocker.patch("reggie.pipeline.embedder.OpenAIEmbeddings")
        mock_instance = mock_embeddings.return_value
        mock_instance.aembed_documents = AsyncMock(
            return_value=[[0.1] * 1536]
        )
        mocker.patch("reggie.pipeline.embedder.tiktoken")

        # Mock text splitter
        mock_splitter = mocker.MagicMock()
        mock_splitter.split_text.return_value = ["Chunk"]
        mocker.patch(
            "reggie.pipeline.embedder.RecursiveCharacterTextSplitter.from_tiktoken_encoder",
            return_value=mock_splitter
        )

        embedder = CommentEmbedder(openai_api_key="sk-test-key")

        comments = [
            {"comment_text": "Comment 1"},
            {"comment_text": "Comment 2"},
            {"comment_text": "Comment 3"}
        ]

        results = await embedder.process_comments_batch(comments, batch_size=2)

        assert len(results) == 3
        for result in results:
            chunks_embeddings, token_count = result
            assert len(chunks_embeddings) == 1  # One chunk per comment
            assert chunks_embeddings[0][0] == "Chunk"
            assert isinstance(token_count, int)

    async def test_process_comments_batch_handles_errors(self, mocker):
        """Batch processing handles errors gracefully and returns token counts."""
        mock_embeddings = mocker.patch("reggie.pipeline.embedder.OpenAIEmbeddings")
        mock_instance = mock_embeddings.return_value

        # First succeeds, second fails, third succeeds
        mock_instance.aembed_documents = AsyncMock(
            side_effect=[
                [[0.1] * 1536],
                Exception("Embedding error"),
                [[0.2] * 1536]
            ]
        )
        mocker.patch("reggie.pipeline.embedder.tiktoken")

        # Mock text splitter
        mock_splitter = mocker.MagicMock()
        mock_splitter.split_text.return_value = ["Chunk"]
        mocker.patch(
            "reggie.pipeline.embedder.RecursiveCharacterTextSplitter.from_tiktoken_encoder",
            return_value=mock_splitter
        )

        embedder = CommentEmbedder(openai_api_key="sk-test-key")

        comments = [
            {"comment_text": "Comment 1"},
            {"comment_text": "Comment 2"},
            {"comment_text": "Comment 3"}
        ]

        results = await embedder.process_comments_batch(comments, batch_size=10)

        assert len(results) == 3
        # First should succeed
        chunks_embeddings_0, token_count_0 = results[0]
        assert len(chunks_embeddings_0) == 1
        assert chunks_embeddings_0[0][1] == [0.1] * 1536
        assert isinstance(token_count_0, int)
        # Second gets zero vector due to embedding error (but chunk still exists)
        chunks_embeddings_1, token_count_1 = results[1]
        assert len(chunks_embeddings_1) == 1
        assert chunks_embeddings_1[0][1] == [0.0] * 1536  # Zero vector fallback
        assert isinstance(token_count_1, int)
        # Third should succeed
        chunks_embeddings_2, token_count_2 = results[2]
        assert len(chunks_embeddings_2) == 1
        assert chunks_embeddings_2[0][1] == [0.2] * 1536
        assert isinstance(token_count_2, int)

    async def test_process_comments_batch_empty_list(self, mocker):
        """Batch processing handles empty list."""
        mocker.patch("reggie.pipeline.embedder.OpenAIEmbeddings")
        mocker.patch("reggie.pipeline.embedder.tiktoken")

        embedder = CommentEmbedder(openai_api_key="sk-test-key")
        results = await embedder.process_comments_batch([])

        assert results == []
