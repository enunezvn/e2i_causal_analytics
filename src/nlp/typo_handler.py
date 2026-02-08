"""
FastText-based typo correction for pharmaceutical domain.

Provides subword-based typo detection and correction for:
- Brand names (Kisqali, Fabhalta, Remibrutinib)
- KPIs (TRx, NRx, conversion_rate)
- Regions (Northeast, Midwest)
- Agent names
- Journey stages

Uses character n-grams for robust typo handling:
- "Kiqsali" → "Kisqali"
- "conversin rate" → "conversion_rate"

Performance target: <50ms latency per correction.
"""

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np

logger = logging.getLogger(__name__)

# Check for fasttext installation
try:
    import fasttext

    FASTTEXT_AVAILABLE = True
except ImportError:
    fasttext = None
    FASTTEXT_AVAILABLE = False
    logger.warning("fasttext not installed. Typo correction will use fallback mode.")


# Default model path (relative to project root)
DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "e2i_fasttext.bin"

# Correction cache
_CORRECTION_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_MAX_SIZE = 500
_CACHE_TTL_SECONDS = 3600  # 1 hour


@dataclass
class CorrectionResult:
    """Result of a typo correction attempt."""

    original: str
    corrected: str
    confidence: float
    category: Optional[str] = None
    was_corrected: bool = False
    latency_ms: float = 0.0


# Canonical vocabulary for E2I domain
CANONICAL_VOCABULARY = {
    "brands": ["Remibrutinib", "Fabhalta", "Kisqali", "ribociclib", "iptacopan"],
    "regions": [
        "northeast",
        "southeast",
        "south",
        "midwest",
        "west",
        "east",
        "southwest",
        "northwest",
        "central",
    ],
    "agents": [
        "orchestrator",
        "causal_impact",
        "gap_analyzer",
        "heterogeneous_optimizer",
        "drift_monitor",
        "experiment_designer",
        "health_score",
        "prediction_synthesizer",
        "resource_optimizer",
        "explainer",
        "feedback_learner",
        "tool_composer",
    ],
    "kpis": [
        "TRx",
        "NRx",
        "NBRx",
        "conversion_rate",
        "engagement_score",
        "treatment_effect",
        "causal_impact",
        "HCP_coverage",
        "time_to_therapy",
        "ROI",
        "AUC",
        "acceptance_rate",
        "market_share",
        "adoption_rate",
        "growth_rate",
        "churn_rate",
    ],
    "journey_stages": [
        "diagnosis",
        "initial_treatment",
        "treatment_optimization",
        "maintenance",
        "treatment_switch",
        "discontinuation",
    ],
    "workstreams": ["WS1", "WS2", "WS3"],
    "entities": [
        "HCP",
        "physician",
        "territory",
        "brand",
        "region",
        "patient",
        "prescriber",
        "specialist",
        "provider",
    ],
}

# Common abbreviations mapping
ABBREVIATION_EXPANSIONS = {
    "trx": "TRx",
    "nrx": "NRx",
    "nbrx": "NBRx",
    "hcp": "HCP",
    "roi": "ROI",
    "auc": "AUC",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
    "mw": "midwest",
    "ws1": "WS1",
    "ws2": "WS2",
    "ws3": "WS3",
}


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row: List[int] = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row: List[int] = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _normalized_edit_similarity(s1: str, s2: str) -> float:
    """Calculate normalized edit similarity (1 - normalized_distance)."""
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    distance = _levenshtein_distance(s1.lower(), s2.lower())
    return 1.0 - (distance / max_len)


class TypoHandler:
    """
    FastText-based typo correction for pharmaceutical domain.

    Uses subword embeddings to detect and correct typos in:
    - Brand names
    - KPIs
    - Regions
    - Agent names

    Falls back to edit-distance based correction when fastText is unavailable.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        similarity_threshold: float = 0.65,
        edit_distance_threshold: float = 0.70,
        cache_enabled: bool = True,
        vocabulary: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize TypoHandler.

        Args:
            model_path: Path to fastText model file (.bin)
            similarity_threshold: Minimum similarity for fastText match
            edit_distance_threshold: Minimum similarity for edit-distance fallback
            cache_enabled: Enable caching of corrections
            vocabulary: Optional custom vocabulary dict
        """
        self.similarity_threshold = similarity_threshold
        self.edit_distance_threshold = edit_distance_threshold
        self.cache_enabled = cache_enabled
        self.vocabulary = vocabulary or CANONICAL_VOCABULARY

        self._model: Optional[Any] = None
        self._model_path = model_path or str(DEFAULT_MODEL_PATH)
        self._all_candidates: List[str] = []
        self._candidate_to_category: Dict[str, str] = {}

        # Build flat candidate list
        self._build_candidate_index()

        # Try to load model
        self._load_model()

    def _build_candidate_index(self) -> None:
        """Build flat list of all canonical terms with category mapping."""
        self._all_candidates = []
        self._candidate_to_category = {}

        for category, terms in self.vocabulary.items():
            for term in terms:
                self._all_candidates.append(term)
                self._candidate_to_category[term.lower()] = category

    def _load_model(self) -> None:
        """Load fastText model if available."""
        if not FASTTEXT_AVAILABLE:
            logger.info("FastText not available, using edit-distance fallback")
            return

        if os.path.exists(self._model_path):
            try:
                self._model = fasttext.load_model(self._model_path)
                logger.info(f"Loaded fastText model from {self._model_path}")
            except Exception as e:
                logger.warning(f"Failed to load fastText model: {e}")
                self._model = None
        else:
            logger.info(f"No fastText model at {self._model_path}, using edit-distance fallback")

    @property
    def is_fasttext_available(self) -> bool:
        """Check if fastText model is loaded."""
        return self._model is not None

    def _get_cache_key(self, term: str) -> str:
        """Generate cache key for a term."""
        return hashlib.md5(term.lower().encode()).hexdigest()

    def _get_cached(self, term: str) -> Optional[CorrectionResult]:
        """Get cached correction if valid."""
        if not self.cache_enabled:
            return None

        cache_key = self._get_cache_key(term)
        cached = _CORRECTION_CACHE.get(cache_key)

        if cached:
            if time.time() - cached["timestamp"] < _CACHE_TTL_SECONDS:
                return cast(CorrectionResult, cached["result"])
            else:
                del _CORRECTION_CACHE[cache_key]

        return None

    def _cache_result(self, term: str, result: CorrectionResult) -> None:
        """Cache correction result."""
        if not self.cache_enabled:
            return

        cache_key = self._get_cache_key(term)

        # Evict oldest if full
        if len(_CORRECTION_CACHE) >= _CACHE_MAX_SIZE:
            oldest_key = min(
                _CORRECTION_CACHE.keys(), key=lambda k: _CORRECTION_CACHE[k]["timestamp"]
            )
            del _CORRECTION_CACHE[oldest_key]

        _CORRECTION_CACHE[cache_key] = {
            "result": result,
            "timestamp": time.time(),
        }

    def _find_best_match_fasttext(
        self,
        term: str,
        candidates: Optional[List[str]] = None,
    ) -> Tuple[Optional[str], float]:
        """
        Find best matching canonical term using fastText embeddings.

        Args:
            term: Input term to match
            candidates: Optional list of candidates (defaults to all)

        Returns:
            (best_match, similarity_score) or (None, 0.0) if no match
        """
        if self._model is None:
            return None, 0.0

        candidates = candidates or self._all_candidates
        term_vec = self._model.get_word_vector(term.lower())

        best_match = None
        best_score = 0.0

        for candidate in candidates:
            candidate_vec = self._model.get_word_vector(candidate.lower())
            score = _cosine_similarity(term_vec, candidate_vec)

            if score > best_score:
                best_score = score
                best_match = candidate

        if best_score < self.similarity_threshold:
            return None, best_score

        return best_match, best_score

    def _find_best_match_edit_distance(
        self,
        term: str,
        candidates: Optional[List[str]] = None,
    ) -> Tuple[Optional[str], float]:
        """
        Find best matching canonical term using edit distance.

        Args:
            term: Input term to match
            candidates: Optional list of candidates (defaults to all)

        Returns:
            (best_match, similarity_score) or (None, 0.0) if no match
        """
        candidates = candidates or self._all_candidates

        best_match = None
        best_score = 0.0

        for candidate in candidates:
            score = _normalized_edit_similarity(term, candidate)

            if score > best_score:
                best_score = score
                best_match = candidate

        if best_score < self.edit_distance_threshold:
            return None, best_score

        return best_match, best_score

    def correct_term(
        self,
        term: str,
        category: Optional[str] = None,
    ) -> CorrectionResult:
        """
        Correct a potentially misspelled term.

        Args:
            term: Input term to correct
            category: Optional category to restrict candidates

        Returns:
            CorrectionResult with corrected term and metadata
        """
        start_time = time.time()
        term_lower = term.lower().strip()

        # Check cache first
        cached = self._get_cached(term)
        if cached:
            return cached

        # Check if it's an abbreviation
        if term_lower in ABBREVIATION_EXPANSIONS:
            expanded = ABBREVIATION_EXPANSIONS[term_lower]
            result = CorrectionResult(
                original=term,
                corrected=expanded,
                confidence=1.0,
                category=self._candidate_to_category.get(expanded.lower()),
                was_corrected=True,
                latency_ms=(time.time() - start_time) * 1000,
            )
            self._cache_result(term, result)
            return result

        # Check if already canonical
        if term_lower in [c.lower() for c in self._all_candidates]:
            # Find the properly cased version
            for candidate in self._all_candidates:
                if candidate.lower() == term_lower:
                    result = CorrectionResult(
                        original=term,
                        corrected=candidate,
                        confidence=1.0,
                        category=self._candidate_to_category.get(term_lower),
                        was_corrected=False,
                        latency_ms=(time.time() - start_time) * 1000,
                    )
                    self._cache_result(term, result)
                    return result

        # Get candidates for matching
        candidates = self._all_candidates
        if category and category in self.vocabulary:
            candidates = self.vocabulary[category]

        # Try fastText first, fall back to edit distance
        if self.is_fasttext_available:
            match, score = self._find_best_match_fasttext(term, candidates)
            method = "fasttext"
        else:
            match, score = self._find_best_match_edit_distance(term, candidates)
            method = "edit_distance"

        latency_ms = (time.time() - start_time) * 1000

        if match:
            result = CorrectionResult(
                original=term,
                corrected=match,
                confidence=score,
                category=self._candidate_to_category.get(match.lower()),
                was_corrected=True,
                latency_ms=latency_ms,
            )
            logger.debug(f"Corrected '{term}' → '{match}' ({method}, {score:.3f})")
        else:
            result = CorrectionResult(
                original=term,
                corrected=term,
                confidence=0.0,
                category=None,
                was_corrected=False,
                latency_ms=latency_ms,
            )
            logger.debug(f"No correction for '{term}' (best score: {score:.3f})")

        self._cache_result(term, result)
        return result

    def correct_query(
        self,
        query: str,
        correct_all_words: bool = False,
    ) -> Tuple[str, List[CorrectionResult]]:
        """
        Correct typos in a full query string.

        Args:
            query: Full query string
            correct_all_words: If True, attempt to correct every word.
                             If False, only correct words that look like typos.

        Returns:
            (corrected_query, list of CorrectionResults)
        """
        start_time = time.time()

        # Tokenize query
        words = query.split()
        corrections: List[CorrectionResult] = []
        corrected_words: List[str] = []

        for word in words:
            # Strip punctuation for matching
            word_clean = word.strip(".,!?;:()[]{}\"'")

            if not word_clean:
                corrected_words.append(word)
                continue

            # Skip very short words unless they're abbreviations
            if len(word_clean) < 2 and word_clean.lower() not in ABBREVIATION_EXPANSIONS:
                corrected_words.append(word)
                continue

            # Check if word needs correction
            should_correct = correct_all_words

            if not should_correct:
                # Heuristic: correct if word is >3 chars and not in common English words
                # and looks like it could be a domain term
                word_lower = word_clean.lower()
                is_potential_term = (
                    len(word_clean) > 3
                    or word_lower in ABBREVIATION_EXPANSIONS
                    or any(word_lower.startswith(cat[:3]) for cat in self.vocabulary.keys())
                )
                should_correct = is_potential_term

            if should_correct:
                result = self.correct_term(word_clean)

                if result.was_corrected:
                    # Preserve original punctuation
                    prefix = (
                        word[: len(word) - len(word.lstrip(".,!?;:()[]{}\"'"))]
                        if word != word.lstrip(".,!?;:()[]{}\"'")
                        else ""
                    )
                    suffix = (
                        word[len(word.rstrip(".,!?;:()[]{}\"'")) :]
                        if word != word.rstrip(".,!?;:()[]{}\"'")
                        else ""
                    )
                    corrected_words.append(f"{prefix}{result.corrected}{suffix}")
                    corrections.append(result)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)

        corrected_query = " ".join(corrected_words)
        total_latency = (time.time() - start_time) * 1000

        logger.info(
            f"Query correction: '{query[:50]}...' → '{corrected_query[:50]}...' "
            f"({len(corrections)} corrections, {total_latency:.1f}ms)"
        )

        return corrected_query, corrections

    def get_suggestions(
        self,
        term: str,
        top_k: int = 5,
        category: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """
        Get top-k correction suggestions for a term.

        Args:
            term: Input term
            top_k: Number of suggestions to return
            category: Optional category to restrict candidates

        Returns:
            List of (suggestion, score) tuples, sorted by score descending
        """
        candidates = self._all_candidates
        if category and category in self.vocabulary:
            candidates = self.vocabulary[category]

        suggestions = []

        for candidate in candidates:
            if self.is_fasttext_available and self._model is not None:
                term_vec = self._model.get_word_vector(term.lower())
                cand_vec = self._model.get_word_vector(candidate.lower())
                score = _cosine_similarity(term_vec, cand_vec)
            else:
                score = _normalized_edit_similarity(term, candidate)

            suggestions.append((candidate, score))

        # Sort by score descending
        suggestions.sort(key=lambda x: x[1], reverse=True)

        return suggestions[:top_k]

    def clear_cache(self) -> int:
        """Clear the correction cache."""
        count = len(_CORRECTION_CACHE)
        _CORRECTION_CACHE.clear()
        logger.info(f"Cleared {count} cached corrections")
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics."""
        return {
            "fasttext_available": self.is_fasttext_available,
            "model_path": self._model_path,
            "vocabulary_size": len(self._all_candidates),
            "categories": list(self.vocabulary.keys()),
            "cache_size": len(_CORRECTION_CACHE),
            "similarity_threshold": self.similarity_threshold,
            "edit_distance_threshold": self.edit_distance_threshold,
        }


# Singleton instance for module-level access
_default_handler: Optional[TypoHandler] = None


def get_typo_handler() -> TypoHandler:
    """Get or create the default TypoHandler instance."""
    global _default_handler
    if _default_handler is None:
        _default_handler = TypoHandler()
    return _default_handler


def correct_term(term: str, category: Optional[str] = None) -> CorrectionResult:
    """Convenience function to correct a single term."""
    return get_typo_handler().correct_term(term, category)


def correct_query(
    query: str, correct_all_words: bool = False
) -> Tuple[str, List[CorrectionResult]]:
    """Convenience function to correct a full query."""
    return get_typo_handler().correct_query(query, correct_all_words)
