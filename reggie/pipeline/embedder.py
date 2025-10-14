"""LangChain pipeline for chunking and embedding comments"""

import asyncio
import logging
from typing import List, Dict, Tuple, Optional

import tiktoken
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..config import get_config
from ..exceptions import ConfigurationException

logger = logging.getLogger(__name__)


class CommentEmbedder:
    """Chunks and embeds comments using LangChain and OpenAI embeddings."""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        """Initialize the embedder.

        Args:
            openai_api_key: OpenAI API key. If None, reads from config/env.
            chunk_size: Target size for text chunks in tokens. If None, uses config default.
            chunk_overlap: Number of tokens to overlap between chunks. If None, uses config default.
        """
        config = get_config()

        api_key = openai_api_key or config.openai_api_key
        if not api_key:
            raise ConfigurationException(
                "OPENAI_API_KEY must be set in environment or .env file"
            )

        # Use config values if not provided
        chunk_size = chunk_size or config.chunk_size
        chunk_overlap = chunk_overlap or config.chunk_overlap

        # Store config values
        self.embedding_model = config.embedding_model
        self.embedding_dimension = config.embedding_dimension

        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            api_key=api_key,
        )

        # Initialize tokenizer for accurate token counting
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")

        # Initialize text splitter
        # Using token-based splitting for accuracy
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        chunks = self.text_splitter.split_text(text)
        return chunks

    async def embed_chunks(
        self,
        chunks: List[str],
        batch_size: int = None,
    ) -> Tuple[List[List[float]], int]:
        """Embed text chunks asynchronously.

        Args:
            chunks: List of text chunks to embed
            batch_size: Number of chunks to embed in each batch. If None, uses config default.

        Returns:
            Tuple of (embedding vectors, total_tokens)
        """
        if not chunks:
            return [], 0

        config = get_config()
        batch_size = batch_size or config.embedding_batch_size

        all_embeddings = []
        total_tokens = 0

        # Process in batches to respect rate limits
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            try:
                # Count tokens for this batch
                batch_tokens = sum(self._count_tokens(chunk) for chunk in batch)
                total_tokens += batch_tokens

                # Use LangChain's async embed method
                embeddings = await self.embeddings.aembed_documents(batch)
                all_embeddings.extend(embeddings)

                # Rate limiting between batches
                if i + batch_size < len(chunks):
                    await asyncio.sleep(config.embedding_rate_limit_sleep)

            except Exception as e:
                logger.error(f"Error embedding batch {i//batch_size}: {e}")
                # Add zero vectors for failed embeddings
                all_embeddings.extend([[0.0] * self.embedding_dimension] * len(batch))

        return all_embeddings, total_tokens

    async def chunk_and_embed(
        self,
        text: str,
    ) -> Tuple[List[Tuple[str, List[float]]], int]:
        """Chunk text and generate embeddings.

        Args:
            text: Text to process

        Returns:
            Tuple of (list of (chunk_text, embedding) tuples, total_tokens)
        """
        # Chunk the text
        chunks = self.chunk_text(text)

        if not chunks:
            return [], 0

        # Embed the chunks
        embeddings, total_tokens = await self.embed_chunks(chunks)

        # Pair chunks with embeddings
        chunks_with_embeddings = list(zip(chunks, embeddings))
        return chunks_with_embeddings, total_tokens

    async def process_comments_batch(
        self,
        comments: List[Dict],
        batch_size: int = None,
    ) -> List[Tuple[List[Tuple[str, List[float]]], int]]:
        """Process multiple comments in parallel.

        Args:
            comments: List of comment dicts with 'comment_text' field
            batch_size: Number of comments to process in parallel. If None, uses config default.

        Returns:
            List of results, where each result is a tuple of (list of (chunk, embedding) tuples, total_tokens)
        """
        config = get_config()
        batch_size = batch_size or config.default_batch_size

        results = []

        for i in range(0, len(comments), batch_size):
            batch = comments[i:i + batch_size]

            # Process batch in parallel
            tasks = [
                self.chunk_and_embed(c.get("comment_text", ""))
                for c in batch
            ]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Error processing comment: {result}")
                    results.append(([], 0))  # Empty result for failed comment
                else:
                    results.append(result)

            logger.info(
                f"Processed comments {i+1}-{min(i+batch_size, len(comments))} of {len(comments)}"
            )

        return results
