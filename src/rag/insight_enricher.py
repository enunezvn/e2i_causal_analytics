"""
LLM-based insight enrichment for CausalRAG.

Uses Claude to synthesize retrieved insights into actionable summaries.

CRITICAL: This is for OPERATIONAL insights only.
NOT for: Medical advice, clinical recommendations, drug information synthesis.
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.rag.models.insight_models import EnrichedInsight

logger = logging.getLogger(__name__)

# Response cache for repeated queries (TTL handled by LRU eviction)
_RESPONSE_CACHE: Dict[str, EnrichedInsight] = {}
_CACHE_MAX_SIZE = 100


class InsightEnricher:
    """
    Use Claude to synthesize retrieved insights.

    NOT for:
    - Medical advice
    - Clinical recommendations
    - Drug information synthesis

    Only for operational/commercial insights:
    - Sales performance analysis
    - HCP targeting insights
    - Market share trends
    - Conversion rate analysis
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        temperature: float = 0.3,
        cache_enabled: bool = True,
    ):
        """
        Initialize with Claude model.

        Args:
            model: Claude model identifier
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (lower = more deterministic)
            cache_enabled: Whether to cache responses for repeated queries
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache_enabled = cache_enabled
        self._client: Optional[anthropic.AsyncAnthropic] = None

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        """Lazy-load Anthropic async client."""
        if self._client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is required")
            self._client = anthropic.AsyncAnthropic(api_key=api_key)
        return self._client

    async def enrich(
        self,
        retrieved: List,  # List[RetrievalResult]
        query,  # ParsedQuery
        max_findings: int = 5,
    ) -> EnrichedInsight:
        """
        Synthesize operational insights from retrieved content.

        Args:
            retrieved: Retrieved results from RAG
            query: Original parsed query
            max_findings: Maximum key findings to extract

        Returns:
            EnrichedInsight with summary and findings
        """
        if not retrieved:
            return EnrichedInsight(
                summary="No relevant insights found for this query.",
                key_findings=[],
                supporting_evidence=[],
                confidence=0.0,
                data_freshness=None,
            )

        query_text = query.text if hasattr(query, "text") else str(query)

        # Check cache first
        cache_key = self._build_cache_key(query_text, retrieved)
        if self.cache_enabled and cache_key in _RESPONSE_CACHE:
            logger.debug(f"Cache hit for query: {query_text[:50]}...")
            return _RESPONSE_CACHE[cache_key]

        # Build context from retrieved results
        context_parts = []
        for i, result in enumerate(retrieved[:10], 1):
            content = result.content if hasattr(result, "content") else str(result)
            source = result.source if hasattr(result, "source") else "unknown"
            context_parts.append(f"[{i}] Source: {source}\n{content}")

        context = "\n\n".join(context_parts)

        # Generate enriched insight
        prompt = self._build_prompt(query_text, context, max_findings)

        try:
            response = await self._generate(prompt)
            enriched = self._parse_response(response, retrieved)

            # Cache the result
            if self.cache_enabled:
                self._cache_response(cache_key, enriched)

            return enriched

        except Exception as e:
            logger.error(f"Insight enrichment failed: {e}")
            # Return graceful fallback
            return EnrichedInsight(
                summary=f"Unable to synthesize insights: {str(e)[:100]}",
                key_findings=[],
                supporting_evidence=retrieved[:5],
                confidence=0.0,
                data_freshness=datetime.now(),
            )

    def _build_cache_key(self, query: str, retrieved: List) -> str:
        """Build cache key from query and retrieved content."""
        content_hash = hash(
            tuple(r.source_id if hasattr(r, "source_id") else str(r) for r in retrieved[:10])
        )
        return f"{hash(query)}:{content_hash}"

    def _cache_response(self, key: str, response: EnrichedInsight) -> None:
        """Cache response with LRU eviction."""
        if len(_RESPONSE_CACHE) >= _CACHE_MAX_SIZE:
            # Remove oldest entry (simple FIFO, could use LRU)
            oldest_key = next(iter(_RESPONSE_CACHE))
            del _RESPONSE_CACHE[oldest_key]
        _RESPONSE_CACHE[key] = response

    def _build_prompt(self, query: str, context: str, max_findings: int) -> str:
        """Build the enrichment prompt with structured output format."""
        return f"""Analyze the following operational business data to answer the query.

IMPORTANT: This is for pharmaceutical COMMERCIAL operations analysis only.
DO NOT provide medical advice, clinical recommendations, or drug information.

Query: {query}

Retrieved Context:
{context}

Respond in the following JSON format:
{{
    "summary": "A concise 2-3 sentence summary answering the query",
    "key_findings": [
        "Finding 1 as a clear bullet point",
        "Finding 2 as a clear bullet point"
    ],
    "confidence": 0.85
}}

Rules:
1. Summary should directly answer the query in 2-3 sentences
2. Include up to {max_findings} key findings as actionable bullet points
3. Confidence score (0-1) based on data quality, relevance, and completeness:
   - 0.9-1.0: Comprehensive, recent data with high relevance
   - 0.7-0.9: Good data coverage with some gaps
   - 0.5-0.7: Partial data, some relevance
   - Below 0.5: Limited or tangential data
4. Focus on: sales trends, HCP targeting, market share, conversion rates, regional performance

Respond ONLY with valid JSON, no additional text."""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((anthropic.APIError, anthropic.APITimeoutError)),
    )
    async def _generate(self, prompt: str) -> str:
        """
        Generate response using Claude API with retry logic.

        Args:
            prompt: The prompt to send to Claude

        Returns:
            Raw response text from Claude

        Raises:
            anthropic.APIError: On API failures after retries
        """
        logger.debug(f"Generating insight with {self.model}")

        message = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract text content from response
        response_text = ""
        for block in message.content:
            if hasattr(block, "text"):
                response_text += block.text

        logger.debug(f"Generated response with {message.usage.output_tokens} tokens")
        return response_text

    def _parse_response(
        self,
        response: str,
        retrieved: List,
    ) -> EnrichedInsight:
        """
        Parse LLM response into EnrichedInsight.

        Handles JSON parsing with fallback for malformed responses.

        Args:
            response: Raw response from Claude
            retrieved: Original retrieved results for evidence

        Returns:
            EnrichedInsight with parsed fields
        """
        # Get data freshness from retrieved results
        data_freshness = self._extract_freshness(retrieved)

        if not response or not response.strip():
            return EnrichedInsight(
                summary="No response generated from language model.",
                key_findings=[],
                supporting_evidence=retrieved[:5],
                confidence=0.0,
                data_freshness=data_freshness,
            )

        try:
            # Try to extract JSON from response
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                parsed = json.loads(response)

            # Extract fields with defaults
            summary = parsed.get("summary", "")
            key_findings = parsed.get("key_findings", [])
            confidence = float(parsed.get("confidence", 0.5))

            # Validate confidence range
            confidence = max(0.0, min(1.0, confidence))

            # Validate key_findings is a list
            if not isinstance(key_findings, list):
                key_findings = [str(key_findings)] if key_findings else []

            return EnrichedInsight(
                summary=summary,
                key_findings=key_findings,
                supporting_evidence=retrieved[:5],
                confidence=confidence,
                data_freshness=data_freshness,
            )

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # Fallback: use raw response as summary
            return EnrichedInsight(
                summary=response[:500] if response else "Unable to parse response.",
                key_findings=[],
                supporting_evidence=retrieved[:5],
                confidence=0.3,  # Lower confidence for unparsed response
                data_freshness=data_freshness,
            )

    def _extract_freshness(self, retrieved: List) -> Optional[datetime]:
        """Extract the most recent timestamp from retrieved results."""
        latest = None
        for result in retrieved:
            metadata = getattr(result, "metadata", {})
            if isinstance(metadata, dict):
                timestamp = metadata.get("timestamp") or metadata.get("created_at")
                if timestamp:
                    try:
                        if isinstance(timestamp, datetime):
                            ts = timestamp
                        elif isinstance(timestamp, str):
                            ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        else:
                            continue
                        if latest is None or ts > latest:
                            latest = ts
                    except (ValueError, TypeError):
                        continue
        return latest or datetime.now()
