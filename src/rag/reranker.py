"""
Cross-encoder reranking for CausalRAG.

Uses cross-encoder models to rerank initial retrieval results
for improved relevance.
"""

import logging
from typing import List, Optional, Tuple

from sentence_transformers import CrossEncoder

from src.rag.models.retrieval_models import RetrievalResult

logger = logging.getLogger(__name__)

# Module-level model cache for singleton pattern
_MODEL_CACHE: dict = {}


class CrossEncoderReranker:
    """
    Rerank initial results using cross-encoder.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (default)
    Supports batch scoring for efficiency.
    Uses module-level caching to avoid reloading model.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        max_length: int = 512,
    ):
        """
        Initialize reranker with cross-encoder model.

        Args:
            model_name: HuggingFace model name for cross-encoder
            batch_size: Batch size for scoring (default 32)
            max_length: Maximum sequence length (default 512)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

    @property
    def model(self) -> CrossEncoder:
        """Lazy-load and cache the cross-encoder model."""
        if self.model_name not in _MODEL_CACHE:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            _MODEL_CACHE[self.model_name] = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
            )
            logger.info(f"Cross-encoder model loaded successfully")
        return _MODEL_CACHE[self.model_name]

    def rerank(
        self,
        results: List[RetrievalResult],
        query,  # ParsedQuery or str
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Rerank results using cross-encoder scoring.

        Uses batch processing for efficiency. Scores are normalized
        to [0, 1] range using sigmoid.

        Args:
            results: Initial retrieval results
            query: Original query for relevance scoring
            top_k: Number of top results to return

        Returns:
            Reranked list of RetrievalResult with updated scores
        """
        if not results:
            return []

        query_text = query.text if hasattr(query, 'text') else str(query)

        # Build query-document pairs for batch scoring
        pairs = []
        for result in results:
            content = result.content if hasattr(result, 'content') else str(result)
            pairs.append((query_text, content))

        # Batch score all pairs
        scores = self._batch_score(pairs)

        # Combine scores with results
        scored_results: List[Tuple[float, RetrievalResult]] = list(zip(scores, results))

        # Sort by score descending
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # Create new RetrievalResult objects with updated scores
        reranked = []
        for score, result in scored_results[:top_k]:
            reranked_result = RetrievalResult(
                source_id=result.source_id,
                content=result.content,
                source=result.source,
                score=score,
                retrieval_method=result.retrieval_method,
                metadata={
                    **result.metadata,
                    "reranker_score": score,
                    "original_score": result.score,
                },
            )
            reranked.append(reranked_result)

        logger.debug(
            f"Reranked {len(results)} results to top {len(reranked)}, "
            f"score range: [{reranked[-1].score:.3f}, {reranked[0].score:.3f}]"
            if reranked else ""
        )

        return reranked

    def _batch_score(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Score query-document pairs in batches.

        Args:
            pairs: List of (query, document) tuples

        Returns:
            List of relevance scores normalized to [0, 1]
        """
        if not pairs:
            return []

        try:
            # CrossEncoder.predict returns raw logits, apply sigmoid for [0, 1] range
            raw_scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )

            # Normalize scores to [0, 1] using sigmoid
            import numpy as np
            normalized_scores = 1 / (1 + np.exp(-raw_scores))

            return normalized_scores.tolist()

        except Exception as e:
            logger.error(f"Batch scoring failed: {e}")
            # Return fallback scores on error
            return [0.5] * len(pairs)

    def _score_pair(self, query: str, document: str) -> float:
        """
        Score a single query-document pair using cross-encoder.

        Prefer using _batch_score for efficiency with multiple pairs.

        Args:
            query: Query text
            document: Document text

        Returns:
            Relevance score normalized to [0, 1]
        """
        scores = self._batch_score([(query, document)])
        return scores[0] if scores else 0.5
