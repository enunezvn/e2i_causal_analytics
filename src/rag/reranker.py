"""
Cross-encoder reranking for CausalRAG.

Uses cross-encoder models to rerank initial retrieval results
for improved relevance.
"""

import logging
import threading
from dataclasses import is_dataclass, replace
from typing import Any, Dict, List, Protocol, Sequence, Tuple, TypeVar, cast

from sentence_transformers import CrossEncoder
from torch import nn

logger = logging.getLogger(__name__)

# Module-level model cache for singleton pattern
_MODEL_CACHE: Dict[str, CrossEncoder] = {}
_MODEL_CACHE_LOCK = threading.Lock()


class RerankableResult(Protocol):
    """Structural contract shared by both repository retrieval result models."""

    content: str
    score: float
    metadata: Dict[str, Any]


ResultT = TypeVar("ResultT", bound=RerankableResult)


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
        raise_on_error: bool = False,
    ):
        """
        Initialize reranker with cross-encoder model.

        Args:
            model_name: HuggingFace model name for cross-encoder
            batch_size: Batch size for scoring (default 32)
            max_length: Maximum sequence length (default 512)
            raise_on_error: Propagate inference failures so an orchestrator can
                preserve the upstream ranking. The legacy default keeps the
                historical neutral-score fallback.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.raise_on_error = raise_on_error

    @property
    def model(self) -> CrossEncoder:
        """Lazy-load and cache the cross-encoder model."""
        if self.model_name not in _MODEL_CACHE:
            # Cold requests can arrive concurrently via the API's worker
            # thread. Single-flight construction avoids duplicate downloads
            # and duplicate model-sized memory spikes.
            with _MODEL_CACHE_LOCK:
                if self.model_name not in _MODEL_CACHE:
                    logger.info(f"Loading cross-encoder model: {self.model_name}")
                    _MODEL_CACHE[self.model_name] = CrossEncoder(
                        self.model_name,
                        max_length=self.max_length,
                    )
                    logger.info("Cross-encoder model loaded successfully")
        return _MODEL_CACHE[self.model_name]

    def rerank(
        self,
        results: Sequence[ResultT],
        query,  # ParsedQuery or str
        top_k: int = 5,
    ) -> List[ResultT]:
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

        query_text = query.text if hasattr(query, "text") else str(query)

        # Build query-document pairs for batch scoring
        pairs = []
        for result in results:
            pairs.append((query_text, result.content))

        # Batch score all pairs
        scores = self._batch_score(pairs)

        # Combine scores with results
        scored_results: List[Tuple[float, ResultT]] = list(zip(scores, results, strict=False))

        # Sort by score descending
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # Create new RetrievalResult objects with updated scores
        reranked: List[ResultT] = []
        for score, result in scored_results[:top_k]:
            reranked.append(self._copy_with_score(result, score))

        logger.debug(
            f"Reranked {len(results)} results to top {len(reranked)}, "
            f"score range: [{reranked[-1].score:.3f}, {reranked[0].score:.3f}]"
            if reranked
            else ""
        )

        return reranked

    @staticmethod
    def _copy_with_score(result: ResultT, score: float) -> ResultT:
        """Return a scored copy without collapsing the caller's result model.

        The repository has two intentional retrieval boundaries:

        * ``src.rag.models.retrieval_models.RetrievalResult`` (Pydantic), used
          by the legacy ``CausalRAG`` orchestrator; and
        * ``src.rag.types.RetrievalResult`` (dataclass), used by the live
          ``/api/v1/rag/search`` hybrid retriever.

        Reconstructing every row as the former loses live-only fields (``id``,
        graph context, latency, and raw score) and raises before reranking can
        serve the API.  Copy through the model's native update mechanism so
        both contracts remain intact.
        """
        metadata = {
            **(result.metadata or {}),
            "reranker_score": score,
            "original_score": float(result.score),
        }

        if is_dataclass(result) and not isinstance(result, type):
            return cast(ResultT, replace(result, score=score, metadata=metadata))

        model_copy = getattr(result, "model_copy", None)
        if callable(model_copy):
            return cast(ResultT, model_copy(update={"score": score, "metadata": metadata}))

        raise TypeError(
            "CrossEncoderReranker requires a dataclass or Pydantic retrieval result; "
            f"got {type(result).__name__}"
        )

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
            # Make the activation explicit. CrossEncoder defaults vary across
            # sentence-transformers/model configurations; applying Sigmoid in
            # predict yields probabilities exactly once and avoids either raw
            # logits or a second post-processing sigmoid.
            predictions = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                activation_fn=nn.Sigmoid(),
            )

            import numpy as np

            scores = np.asarray(predictions, dtype=float).reshape(-1)
            if len(scores) != len(pairs):
                raise ValueError(
                    f"Cross-encoder returned {len(scores)} scores for {len(pairs)} pairs"
                )
            if not np.all(np.isfinite(scores)):
                raise ValueError("Cross-encoder returned non-finite scores")
            if np.any((scores < 0.0) | (scores > 1.0)):
                raise ValueError("Cross-encoder activation returned scores outside [0, 1]")
            return cast(List[float], scores.tolist())

        except Exception as e:
            logger.error(f"Batch scoring failed: {e}")
            if self.raise_on_error:
                raise
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
