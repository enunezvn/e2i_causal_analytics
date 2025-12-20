# src/e2i/agents/orchestrator/classifier/pipeline.py
"""
Main classification pipeline orchestrating all stages.

This module provides the main entry point for query classification,
coordinating the 4-stage pipeline:
1. Feature Extraction (rule-based, ~10ms)
2. Domain Mapping (rule-based)
3. Dependency Detection (rule-based + optional LLM)
4. Pattern Selection (deterministic)

The pipeline automatically determines when to escalate to the LLM layer
for complex queries.
"""

import time
from typing import Optional

from anthropic import Anthropic

from .feature_extractor import FeatureExtractor
from .domain_mapper import DomainMapper
from .dependency_detector import DependencyDetector
from .pattern_selector import PatternSelector
from .schemas import ClassificationResult, ExtractedFeatures, DomainMapping


class ClassificationPipeline:
    """
    Orchestrates the 4-stage classification pipeline.
    """

    # Thresholds for LLM escalation
    MULTI_DOMAIN_THRESHOLD = 2
    WORD_COUNT_THRESHOLD = 25
    AMBIGUITY_THRESHOLD = 0.6

    def __init__(
        self,
        llm_client: Optional[Anthropic] = None,
        enable_llm_layer: bool = True,
    ):
        """
        Initialize pipeline components.
        
        Args:
            llm_client: Anthropic client for LLM-based classification
            enable_llm_layer: Whether to use LLM for complex queries
        """
        self.feature_extractor = FeatureExtractor()
        self.domain_mapper = DomainMapper()
        self.dependency_detector = DependencyDetector(llm_client=llm_client)
        self.pattern_selector = PatternSelector()
        
        self.llm_client = llm_client
        self.enable_llm_layer = enable_llm_layer

    async def classify(
        self,
        query: str,
        context: Optional[dict] = None,
        is_followup: bool = False,
        context_source: Optional[str] = None,
    ) -> ClassificationResult:
        """
        Run full classification pipeline.
        
        Args:
            query: User query text
            context: Optional conversation context
            is_followup: Whether this is a follow-up query
            context_source: Source of context if follow-up
            
        Returns:
            ClassificationResult with routing decision
        """
        start_time = time.time()
        used_llm = False

        # =====================================================================
        # Stage 1: Feature Extraction (always rule-based)
        # =====================================================================
        features = self.feature_extractor.extract(query, context)

        # =====================================================================
        # Stage 2: Domain Mapping (rule-based)
        # =====================================================================
        domain_mapping = self.domain_mapper.map_domains(features)

        # =====================================================================
        # Determine if LLM layer is needed
        # =====================================================================
        needs_llm = self._should_use_llm(features, domain_mapping)

        # =====================================================================
        # Stage 3: Dependency Detection
        # =====================================================================
        dependency_analysis = await self.dependency_detector.detect(
            query=query,
            features=features,
            domain_mapping=domain_mapping,
            use_llm=needs_llm and self.enable_llm_layer,
        )
        
        if needs_llm and self.enable_llm_layer:
            used_llm = True

        # =====================================================================
        # Stage 4: Pattern Selection
        # =====================================================================
        result = self.pattern_selector.select(
            features=features,
            domain_mapping=domain_mapping,
            dependency_analysis=dependency_analysis,
            is_followup=is_followup,
            context_source=context_source,
            classification_start_time=start_time,
            used_llm=used_llm,
        )

        return result

    def classify_sync(
        self,
        query: str,
        context: Optional[dict] = None,
        is_followup: bool = False,
        context_source: Optional[str] = None,
    ) -> ClassificationResult:
        """
        Synchronous version of classify for non-async contexts.
        Note: This version cannot use LLM layer.
        
        Args:
            query: User query text
            context: Optional conversation context
            is_followup: Whether this is a follow-up query
            context_source: Source of context if follow-up
            
        Returns:
            ClassificationResult with routing decision
        """
        import asyncio
        
        # Try to get existing event loop, create new one if needed
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, use run_coroutine_threadsafe
                import concurrent.futures
                future = asyncio.run_coroutine_threadsafe(
                    self.classify(query, context, is_followup, context_source),
                    loop
                )
                return future.result(timeout=30)
            else:
                return loop.run_until_complete(
                    self.classify(query, context, is_followup, context_source)
                )
        except RuntimeError:
            # No event loop, create a new one
            return asyncio.run(
                self.classify(query, context, is_followup, context_source)
            )

    def _should_use_llm(
        self,
        features: ExtractedFeatures,
        domain_mapping: DomainMapping,
    ) -> bool:
        """Determine if LLM layer should be used."""
        # Use LLM if:
        # 1. Multi-domain query
        if domain_mapping.domain_count >= self.MULTI_DOMAIN_THRESHOLD:
            return True

        # 2. Long/complex query
        if features.structural.word_count > self.WORD_COUNT_THRESHOLD:
            return True

        # 3. Ambiguous domain mapping (close confidences)
        if domain_mapping.domains_detected and len(domain_mapping.domains_detected) > 1:
            top_conf = domain_mapping.domains_detected[0].confidence
            second_conf = domain_mapping.domains_detected[1].confidence
            if top_conf - second_conf < 0.2:  # Close race
                return True

        # 4. Has conditional/sequence structure
        if features.structural.has_conditional or features.structural.has_sequence:
            return True

        return False

    def get_features(self, query: str) -> ExtractedFeatures:
        """
        Extract features only (for debugging/analysis).
        
        Args:
            query: User query text
            
        Returns:
            ExtractedFeatures
        """
        return self.feature_extractor.extract(query)

    def get_domain_mapping(self, query: str) -> DomainMapping:
        """
        Get domain mapping only (for debugging/analysis).
        
        Args:
            query: User query text
            
        Returns:
            DomainMapping
        """
        features = self.feature_extractor.extract(query)
        return self.domain_mapper.map_domains(features)
