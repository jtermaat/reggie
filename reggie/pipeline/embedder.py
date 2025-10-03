"""LangChain pipeline for chunking and embedding comments"""

import os
import asyncio
import logging
from typing import List, Dict, Tuple

import tiktoken
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langsmith import traceable

from ..config import setup_langsmith

logger = logging.getLogger(__name__)


class CommentEmbedder:
    """Chunks and embeds comments using LangChain and OpenAI embeddings."""

    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIMENSION = 1536

    def __init__(
        self,
        openai_api_key: str = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """Initialize the embedder.

        Args:
            openai_api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            chunk_size: Target size for text chunks in tokens
            chunk_overlap: Number of tokens to overlap between chunks
        """
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set")

        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.EMBEDDING_MODEL,
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

    @traceable(name="chunk_text")
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

    @traceable(name="embed_chunks")
    async def embed_chunks(
        self,
        chunks: List[str],
        batch_size: int = 100,
    ) -> List[List[float]]:
        """Embed text chunks asynchronously.

        Args:
            chunks: List of text chunks to embed
            batch_size: Number of chunks to embed in each batch

        Returns:
            List of embedding vectors
        """
        if not chunks:
            return []

        all_embeddings = []

        # Process in batches to respect rate limits
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            try:
                # Use LangChain's async embed method
                embeddings = await self.embeddings.aembed_documents(batch)
                all_embeddings.extend(embeddings)

                # Rate limiting between batches
                if i + batch_size < len(chunks):
                    await asyncio.sleep(1.0)

            except Exception as e:
                logger.error(f"Error embedding batch {i//batch_size}: {e}")
                # Add zero vectors for failed embeddings
                all_embeddings.extend([[0.0] * self.EMBEDDING_DIMENSION] * len(batch))

        return all_embeddings

    @traceable(name="chunk_and_embed")
    async def chunk_and_embed(
        self,
        text: str,
    ) -> List[Tuple[str, List[float]]]:
        """Chunk text and generate embeddings.

        Args:
            text: Text to process

        Returns:
            List of (chunk_text, embedding) tuples
        """
        # Chunk the text
        chunks = self.chunk_text(text)

        if not chunks:
            return []

        # Embed the chunks
        embeddings = await self.embed_chunks(chunks)

        # Pair chunks with embeddings
        return list(zip(chunks, embeddings))

    @traceable(name="process_comments_batch")
    async def process_comments_batch(
        self,
        comments: List[Dict],
        batch_size: int = 10,
    ) -> List[List[Tuple[str, List[float]]]]:
        """Process multiple comments in parallel.

        Args:
            comments: List of comment dicts with 'comment_text' field
            batch_size: Number of comments to process in parallel

        Returns:
            List of results, where each result is a list of (chunk, embedding) tuples
        """
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
                    results.append([])  # Empty result for failed comment
                else:
                    results.append(result)

            logger.info(
                f"Processed comments {i+1}-{min(i+batch_size, len(comments))} of {len(comments)}"
            )

        return results
