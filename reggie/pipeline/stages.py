"""Pipeline stages for comment processing"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

from .categorizer import CommentCategorizer
from .embedder import CommentEmbedder
from ..models import CommentData, CommentClassification

logger = logging.getLogger(__name__)


class PipelineStage(ABC):
    """Base class for pipeline stages."""

    @abstractmethod
    async def process(
        self, comment_data: CommentData
    ) -> Tuple[CommentData, Dict]:
        """Process a comment through this stage.

        Args:
            comment_data: Comment data to process

        Returns:
            Tuple of (updated comment_data, stage_result_dict)
        """
        pass


class CategorizationStage(PipelineStage):
    """Stage for categorizing comments."""

    def __init__(self, categorizer: CommentCategorizer):
        """Initialize the categorization stage.

        Args:
            categorizer: CommentCategorizer instance
        """
        self.categorizer = categorizer

    async def process(
        self, comment_data: CommentData
    ) -> Tuple[CommentData, Dict]:
        """Categorize a comment.

        Args:
            comment_data: Comment data to categorize

        Returns:
            Tuple of (comment_data, classification_dict)
        """
        classification = await self.categorizer.categorize(
            comment_text=comment_data.comment_text,
            first_name=comment_data.first_name,
            last_name=comment_data.last_name,
            organization=comment_data.organization,
        )

        result = {
            "classification": classification,
            "category": classification.category.value,
            "sentiment": classification.sentiment.value,
            "topics": [topic.value for topic in classification.topics],
            "doctor_specialization": classification.doctor_specialization.value if classification.doctor_specialization else None,
            "licensed_professional_type": classification.licensed_professional_type.value if classification.licensed_professional_type else None,
        }

        return comment_data, result


class EmbeddingStage(PipelineStage):
    """Stage for chunking and embedding comments."""

    def __init__(self, embedder: CommentEmbedder):
        """Initialize the embedding stage.

        Args:
            embedder: CommentEmbedder instance
        """
        self.embedder = embedder

    async def process(
        self, comment_data: CommentData
    ) -> Tuple[CommentData, Dict]:
        """Chunk and embed a comment.

        Args:
            comment_data: Comment data to embed

        Returns:
            Tuple of (comment_data, embedding_result_dict)
        """
        chunks_with_embeddings, tokens = await self.embedder.chunk_and_embed(
            comment_data.comment_text
        )

        result = {
            "chunks": chunks_with_embeddings,
            "num_chunks": len(chunks_with_embeddings),
            "tokens": tokens,
        }

        return comment_data, result


class BatchCategorizationStage(PipelineStage):
    """Stage for batch categorization of comments."""

    def __init__(self, categorizer: CommentCategorizer):
        """Initialize the batch categorization stage.

        Args:
            categorizer: CommentCategorizer instance
        """
        self.categorizer = categorizer

    def set_error_collector(self, error_collector):
        """Set the error collector for this stage.

        Args:
            error_collector: ErrorCollector instance
        """
        self.categorizer.error_collector = error_collector

    async def process_batch(
        self, comment_data_list: List[CommentData]
    ) -> List[Dict]:
        """Categorize a batch of comments.

        Args:
            comment_data_list: List of comment data to categorize

        Returns:
            List of classification result dicts
        """
        # Convert CommentData to dict format expected by categorizer
        comment_dicts = [
            {
                "id": cd.id,
                "comment_text": cd.comment_text,
                "first_name": cd.first_name,
                "last_name": cd.last_name,
                "organization": cd.organization,
            }
            for cd in comment_data_list
        ]

        classifications = await self.categorizer.categorize_batch(
            comment_dicts, batch_size=len(comment_dicts)
        )

        results = []
        for classification in classifications:
            result = {
                "classification": classification,
                "category": classification.category.value,
                "sentiment": classification.sentiment.value,
                "topics": [topic.value for topic in classification.topics],
                "doctor_specialization": classification.doctor_specialization.value if classification.doctor_specialization else None,
                "licensed_professional_type": classification.licensed_professional_type.value if classification.licensed_professional_type else None,
            }
            results.append(result)

        return results

    async def process(
        self, comment_data: CommentData
    ) -> Tuple[CommentData, Dict]:
        """Process single comment (delegates to categorizer).

        Args:
            comment_data: Comment data to categorize

        Returns:
            Tuple of (comment_data, classification_dict)
        """
        results = await self.process_batch([comment_data])
        return comment_data, results[0]


class BatchEmbeddingStage(PipelineStage):
    """Stage for batch embedding of comments."""

    def __init__(self, embedder: CommentEmbedder):
        """Initialize the batch embedding stage.

        Args:
            embedder: CommentEmbedder instance
        """
        self.embedder = embedder

    async def process_batch(
        self, comment_data_list: List[CommentData]
    ) -> List[Dict]:
        """Chunk and embed a batch of comments.

        Args:
            comment_data_list: List of comment data to embed

        Returns:
            List of embedding result dicts (each with 'chunks', 'num_chunks', 'tokens')
        """
        # Convert CommentData to dict format expected by embedder
        comment_dicts = [
            {
                "id": cd.id,
                "comment_text": cd.comment_text,
            }
            for cd in comment_data_list
        ]

        all_chunks_and_tokens = await self.embedder.process_comments_batch(
            comment_dicts, batch_size=len(comment_dicts)
        )

        results = []
        for chunks_with_embeddings, tokens in all_chunks_and_tokens:
            result = {
                "chunks": chunks_with_embeddings,
                "num_chunks": len(chunks_with_embeddings),
                "tokens": tokens,
            }
            results.append(result)

        return results

    async def process(
        self, comment_data: CommentData
    ) -> Tuple[CommentData, Dict]:
        """Process single comment (delegates to embedder).

        Args:
            comment_data: Comment data to embed

        Returns:
            Tuple of (comment_data, embedding_dict)
        """
        results = await self.process_batch([comment_data])
        return comment_data, results[0]
