"""
Query optimization with domain context and LLM enhancement.

Expands queries with domain knowledge:
- Typo correction: FastText-based subword correction
- Rule-based: Add synonyms from domain_vocabulary.yaml
- LLM-based: Claude-powered semantic expansion
- HyDE: Hypothetical Document Embeddings for improved retrieval

CRITICAL: This is for OPERATIONAL queries only.
NOT for: Medical/clinical query expansion.
"""

import asyncio
import hashlib
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union, cast

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.nlp.typo_handler import CorrectionResult, TypoHandler

logger = logging.getLogger(__name__)

# Expansion cache for repeated queries
_EXPANSION_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_MAX_SIZE = 200
_CACHE_TTL_SECONDS = 3600  # 1 hour


class QueryOptimizer:
    """
    Expand queries with domain knowledge and LLM enhancement for better retrieval.

    Features:
    1. Rule-based expansion with domain synonyms (fast, no API call)
    2. LLM-based semantic expansion via Claude (richer understanding)
    3. HyDE: Generate hypothetical documents for embedding-based retrieval

    Example:
        Rule-based: "TRx for Kisqali" → "Total prescriptions TRx Kisqali breast cancer"
        LLM-based:  "TRx for Kisqali" → "Kisqali total prescription volume HR+ breast
                    cancer market performance ribociclib commercial adoption trends"
        HyDE:       "TRx for Kisqali" → [hypothetical document about Kisqali prescriptions]
    """

    def __init__(
        self,
        vocabulary_path: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 256,
        temperature: float = 0.3,
        cache_enabled: bool = True,
        typo_correction_enabled: bool = True,
        fasttext_model_path: Optional[str] = None,
    ):
        """
        Initialize with domain vocabulary and LLM settings.

        Args:
            vocabulary_path: Path to domain_vocabulary.yaml
            model: Claude model for LLM expansion
            max_tokens: Max tokens for LLM response
            temperature: LLM temperature (lower = more focused)
            cache_enabled: Enable query expansion caching
            typo_correction_enabled: Enable fastText typo correction
            fasttext_model_path: Optional path to fastText model
        """
        self.vocabulary_path = vocabulary_path
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache_enabled = cache_enabled
        self.typo_correction_enabled = typo_correction_enabled

        self._synonyms: Dict[str, List[str]] = {}
        self._kpi_relations: Dict[str, List[str]] = {}
        self._client: Optional[anthropic.Anthropic] = None
        self._typo_handler: Optional[TypoHandler] = None

        self._load_vocabulary()

        # Initialize typo handler if enabled
        if typo_correction_enabled:
            try:
                self._typo_handler = TypoHandler(
                    model_path=fasttext_model_path,
                    cache_enabled=cache_enabled,
                )
                logger.info(
                    f"Typo correction enabled (fasttext={self._typo_handler.is_fasttext_available})"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize typo handler: {e}")
                self._typo_handler = None

    def _get_client(self) -> anthropic.Anthropic:
        """Lazy-load Anthropic client."""
        if self._client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable required")
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def _load_vocabulary(self) -> None:
        """Load domain vocabulary from YAML file."""
        # Default pharmaceutical commercial domain synonyms
        self._synonyms = {
            "trx": ["total prescriptions", "total rx", "prescription volume"],
            "nrx": ["new prescriptions", "new rx", "new starts"],
            "kisqali": ["ribociclib", "hr+ breast cancer", "her2-"],
            "fabhalta": ["iptacopan", "pnh", "paroxysmal nocturnal hemoglobinuria"],
            "remibrutinib": ["csu", "chronic spontaneous urticaria"],
            "hcp": ["healthcare provider", "physician", "prescriber"],
            "conversion": ["conversion rate", "rx conversion", "switch rate"],
            "market share": ["share of voice", "sov", "competitive share"],
            "adoption": ["adoption rate", "uptake", "market penetration"],
            "territory": ["sales territory", "geographic region", "coverage area"],
        }

        self._kpi_relations = {
            "trx": ["nrx", "conversion_rate", "market_share"],
            "nrx": ["trx", "new_patient_starts", "time_to_first_rx"],
            "conversion_rate": ["trx", "nrx", "abandonment_rate"],
            "adoption_rate": ["market_share", "penetration", "growth_rate"],
        }

    def expand(self, query: Union[str, Any]) -> str:
        """
        Expand query with domain knowledge (rule-based, synchronous).

        Args:
            query: ParsedQuery or string

        Returns:
            Expanded query string
        """
        query_text = query.text if hasattr(query, "text") else str(query)
        query_lower = query_text.lower()

        expansions = []

        # Add synonyms for recognized terms
        for term, synonyms in self._synonyms.items():
            if term in query_lower:
                expansions.extend(synonyms[:2])  # Limit expansions

        # Add related KPIs
        for kpi, related in self._kpi_relations.items():
            if kpi in query_lower:
                expansions.extend(related[:2])

        # Combine original query with expansions
        if expansions:
            return f"{query_text} {' '.join(expansions)}"

        return query_text

    def add_temporal_context(self, query: str, time_range: Optional[str] = None) -> str:
        """
        Add temporal context to query.

        Args:
            query: Query text
            time_range: Optional time range (e.g., "last 30 days")

        Returns:
            Query with temporal context
        """
        if time_range:
            return f"{query} {time_range}"
        return query

    # =========================================================================
    # Typo Correction
    # =========================================================================

    def correct_typos(
        self,
        query: Union[str, Any],
        correct_all_words: bool = False,
    ) -> Dict[str, Any]:
        """
        Correct typos in a query using fastText subword embeddings.

        Falls back to edit-distance based correction when fastText is unavailable.

        Args:
            query: Query text or ParsedQuery
            correct_all_words: If True, attempt correction on all words

        Returns:
            Dict with:
                - original: Original query
                - corrected: Typo-corrected query
                - corrections: List of corrections made
                - latency_ms: Processing time
        """
        query_text = query.text if hasattr(query, "text") else str(query)

        if not self._typo_handler:
            return {
                "original": query_text,
                "corrected": query_text,
                "corrections": [],
                "latency_ms": 0.0,
            }

        start_time = time.time()
        corrected, corrections = self._typo_handler.correct_query(
            query_text, correct_all_words=correct_all_words
        )
        latency_ms = (time.time() - start_time) * 1000

        return {
            "original": query_text,
            "corrected": corrected,
            "corrections": [
                {
                    "original": c.original,
                    "corrected": c.corrected,
                    "confidence": c.confidence,
                    "category": c.category,
                }
                for c in corrections
            ],
            "latency_ms": latency_ms,
        }

    def correct_term(
        self,
        term: str,
        category: Optional[str] = None,
    ) -> CorrectionResult:
        """
        Correct a single term.

        Args:
            term: Term to correct
            category: Optional category hint (brands, kpis, regions, etc.)

        Returns:
            CorrectionResult with corrected term and metadata
        """
        if not self._typo_handler:
            return CorrectionResult(
                original=term,
                corrected=term,
                confidence=0.0,
                was_corrected=False,
            )

        return self._typo_handler.correct_term(term, category)

    def get_typo_suggestions(
        self,
        term: str,
        top_k: int = 5,
        category: Optional[str] = None,
    ) -> List[tuple]:
        """
        Get typo correction suggestions for a term.

        Args:
            term: Term to get suggestions for
            top_k: Number of suggestions
            category: Optional category hint

        Returns:
            List of (suggestion, score) tuples
        """
        if not self._typo_handler:
            return []

        return self._typo_handler.get_suggestions(term, top_k, category)

    # =========================================================================
    # LLM-Enhanced Query Expansion
    # =========================================================================

    def _build_cache_key(self, query: str, expansion_type: str) -> str:
        """Build cache key for query expansion."""
        content = f"{expansion_type}:{query}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cached(self, cache_key: str) -> Optional[str]:
        """Get cached expansion if valid."""
        if not self.cache_enabled:
            return None

        cached = _EXPANSION_CACHE.get(cache_key)
        if cached:
            # Check TTL
            if time.time() - cached["timestamp"] < _CACHE_TTL_SECONDS:
                logger.debug(f"Cache hit for {cache_key[:8]}...")
                return cast(str, cached["result"])
            else:
                # Expired, remove
                del _EXPANSION_CACHE[cache_key]

        return None

    def _cache_result(self, cache_key: str, result: str) -> None:
        """Cache expansion result with TTL."""
        if not self.cache_enabled:
            return

        # Evict oldest entries if cache is full
        if len(_EXPANSION_CACHE) >= _CACHE_MAX_SIZE:
            oldest_key = min(
                _EXPANSION_CACHE.keys(), key=lambda k: _EXPANSION_CACHE[k]["timestamp"]
            )
            del _EXPANSION_CACHE[oldest_key]

        _EXPANSION_CACHE[cache_key] = {
            "result": result,
            "timestamp": time.time(),
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIConnectionError)),
    )
    def _call_llm(self, prompt: str) -> str:
        """Call Claude API with retry logic."""
        client = self._get_client()

        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract text from the first content block
        first_block = response.content[0]
        if hasattr(first_block, "text"):
            return first_block.text  # type: ignore[union-attr]
        return ""

    async def _call_llm_async(self, prompt: str) -> str:
        """Async wrapper for LLM call."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._call_llm, prompt)

    def expand_with_llm(
        self,
        query: Union[str, Any],
        context: Optional[str] = None,
    ) -> str:
        """
        Expand query using Claude for semantic understanding.

        Falls back to rule-based expansion on API failure.

        Args:
            query: Original query text or ParsedQuery
            context: Optional conversation context

        Returns:
            LLM-expanded query string
        """
        query_text = query.text if hasattr(query, "text") else str(query)

        # Check cache first
        cache_key = self._build_cache_key(query_text, "llm_expand")
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            prompt = self._build_expansion_prompt(query_text, context)
            expanded = self._call_llm(prompt)

            # Clean up response (remove quotes, extra whitespace)
            expanded = expanded.strip().strip("\"'")

            # Cache successful result
            self._cache_result(cache_key, expanded)

            logger.info(f"LLM expanded: '{query_text[:50]}...' → '{expanded[:50]}...'")
            return expanded

        except Exception as e:
            logger.warning(f"LLM expansion failed, falling back to rule-based: {e}")
            return self.expand(query_text)

    async def expand_with_llm_async(
        self,
        query: Union[str, Any],
        context: Optional[str] = None,
    ) -> str:
        """
        Async version of expand_with_llm.

        Args:
            query: Original query text or ParsedQuery
            context: Optional conversation context

        Returns:
            LLM-expanded query string
        """
        query_text = query.text if hasattr(query, "text") else str(query)

        # Check cache first
        cache_key = self._build_cache_key(query_text, "llm_expand")
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            prompt = self._build_expansion_prompt(query_text, context)
            expanded = await self._call_llm_async(prompt)

            # Clean up response
            expanded = expanded.strip().strip("\"'")

            # Cache successful result
            self._cache_result(cache_key, expanded)

            logger.info(f"LLM expanded: '{query_text[:50]}...' → '{expanded[:50]}...'")
            return expanded

        except Exception as e:
            logger.warning(f"LLM expansion failed, falling back to rule-based: {e}")
            return self.expand(query_text)

    def _build_expansion_prompt(self, query: str, context: Optional[str] = None) -> str:
        """Build prompt for LLM query expansion."""
        context_section = ""
        if context:
            context_section = f"\nConversation context: {context}\n"

        return f"""You are a pharmaceutical commercial analytics expert. Expand this search query with relevant terms to improve retrieval.

Domain: Pharmaceutical sales, HCP targeting, prescription analytics
Brands: Kisqali (breast cancer), Fabhalta (PNH), Remibrutinib (CSU)
KPIs: TRx, NRx, market share, conversion rate, adoption rate

Original query: "{query}"
{context_section}
Expand the query by adding:
1. Synonyms and related terms
2. Relevant KPIs if applicable
3. Related business concepts

Return ONLY the expanded query as a single line. Do not include explanations.

Expanded query:"""

    # =========================================================================
    # HyDE: Hypothetical Document Embeddings
    # =========================================================================

    def generate_hyde_document(
        self,
        query: Union[str, Any],
        document_type: str = "insight",
    ) -> str:
        """
        Generate a hypothetical document for HyDE retrieval.

        HyDE creates a hypothetical document that would answer the query,
        then uses that document's embedding for retrieval. This often
        improves retrieval quality for complex queries.

        Args:
            query: Original query text or ParsedQuery
            document_type: Type of document to generate (insight, report, analysis)

        Returns:
            Hypothetical document text for embedding
        """
        query_text = query.text if hasattr(query, "text") else str(query)

        # Check cache first
        cache_key = self._build_cache_key(f"{query_text}:{document_type}", "hyde")
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            prompt = self._build_hyde_prompt(query_text, document_type)
            hyde_doc = self._call_llm(prompt)

            # Cache successful result
            self._cache_result(cache_key, hyde_doc)

            logger.info(f"Generated HyDE document for: '{query_text[:50]}...'")
            return hyde_doc

        except Exception as e:
            logger.warning(f"HyDE generation failed, using expanded query: {e}")
            # Fallback: use expanded query as pseudo-document
            return self.expand(query_text)

    async def generate_hyde_document_async(
        self,
        query: Union[str, Any],
        document_type: str = "insight",
    ) -> str:
        """
        Async version of generate_hyde_document.

        Args:
            query: Original query text or ParsedQuery
            document_type: Type of document to generate

        Returns:
            Hypothetical document text for embedding
        """
        query_text = query.text if hasattr(query, "text") else str(query)

        # Check cache first
        cache_key = self._build_cache_key(f"{query_text}:{document_type}", "hyde")
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            prompt = self._build_hyde_prompt(query_text, document_type)
            hyde_doc = await self._call_llm_async(prompt)

            # Cache successful result
            self._cache_result(cache_key, hyde_doc)

            logger.info(f"Generated HyDE document for: '{query_text[:50]}...'")
            return hyde_doc

        except Exception as e:
            logger.warning(f"HyDE generation failed, using expanded query: {e}")
            return self.expand(query_text)

    def _build_hyde_prompt(self, query: str, document_type: str) -> str:
        """Build prompt for HyDE document generation."""
        type_instructions = {
            "insight": "a concise business insight or finding",
            "report": "an executive summary paragraph",
            "analysis": "a data analysis conclusion",
        }

        instruction = type_instructions.get(document_type, type_instructions["insight"])

        return f"""You are a pharmaceutical commercial analytics system. Generate {instruction} that would answer this question.

Domain: Pharmaceutical sales analytics for Kisqali, Fabhalta, Remibrutinib
Data types: Prescriptions (TRx, NRx), market share, HCP activities, territory performance

Question: "{query}"

Write a realistic {document_type} that directly answers this question using specific metrics, trends, or findings. Be concrete and use plausible numbers.

{document_type.capitalize()}:"""

    # =========================================================================
    # Combined Optimization
    # =========================================================================

    async def optimize_query(
        self,
        query: Union[str, Any],
        use_typo_correction: bool = True,
        use_llm: bool = True,
        use_hyde: bool = False,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full query optimization pipeline.

        Pipeline order:
        1. Typo correction (fastText or edit-distance)
        2. Rule-based expansion (synonyms)
        3. LLM expansion (Claude)
        4. HyDE document generation (optional)

        Args:
            query: Original query
            use_typo_correction: Whether to correct typos first
            use_llm: Whether to use LLM expansion
            use_hyde: Whether to generate HyDE document
            context: Optional conversation context

        Returns:
            Dict with:
                - original: Original query text
                - typo_corrected: Typo-corrected query (if enabled)
                - typo_corrections: List of corrections made
                - rule_expanded: Rule-based expansion
                - llm_expanded: LLM expansion (if use_llm=True)
                - hyde_document: HyDE document (if use_hyde=True)
                - recommended: Best expansion to use for retrieval
        """
        query_text = query.text if hasattr(query, "text") else str(query)

        result: Dict[str, Any] = {
            "original": query_text,
            "typo_corrected": None,
            "typo_corrections": [],
            "rule_expanded": None,
            "llm_expanded": None,
            "hyde_document": None,
            "recommended": None,
        }

        # Step 1: Typo correction
        working_query = query_text
        if use_typo_correction and self._typo_handler:
            typo_result = self.correct_typos(query_text)
            result["typo_corrected"] = typo_result["corrected"]
            result["typo_corrections"] = typo_result["corrections"]
            working_query = typo_result["corrected"]

        # Step 2: Rule-based expansion
        result["rule_expanded"] = self.expand(working_query)

        # Step 3: LLM expansion
        if use_llm:
            result["llm_expanded"] = await self.expand_with_llm_async(working_query, context)

        # Step 4: HyDE document
        if use_hyde:
            result["hyde_document"] = await self.generate_hyde_document_async(working_query)

        # Determine recommended expansion
        if use_hyde and result["hyde_document"]:
            result["recommended"] = result["hyde_document"]
        elif use_llm and result["llm_expanded"]:
            result["recommended"] = result["llm_expanded"]
        else:
            result["recommended"] = result["rule_expanded"]

        return result

    def clear_cache(self) -> int:
        """
        Clear the expansion cache.

        Returns:
            Number of entries cleared
        """
        count = len(_EXPANSION_CACHE)
        _EXPANSION_CACHE.clear()
        logger.info(f"Cleared {count} cached query expansions")
        return count
