"""
E2I Query-Time Entity Extractor

PURPOSE: Extract entities from user queries for agent routing and filter construction
SCOPE: NLP Layer (query-time) - NOT episode extraction (Graphiti handles that)

Key Distinction:
- Query-Time (this extractor): Fast, synchronous, for routing decisions
- Episode-Time (Graphiti): Async, LLM-based, for knowledge graph building

Usage:
    from src.ontology import E2IQueryExtractor

    extractor = E2IQueryExtractor('config/domain_vocabulary.yaml')

    # Extract from user query
    user_query = "What's the efficacy of remi in the northeast?"
    result = extractor.extract_for_routing(user_query)

    # Use extracted entities for agent routing
    agent = route_to_agent(result.entities)
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import yaml  # type: ignore[import-untyped]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExtractionContext(Enum):
    """Where extraction is used"""

    QUERY_ROUTING = "query_routing"  # Route query to correct agent
    FILTER_CONSTRUCTION = "filter_construction"  # Build SQL WHERE clauses
    VALIDATION = "validation"  # Validate user input


@dataclass
class Entity:
    """Extracted entity for query routing"""

    text: str
    entity_type: str
    confidence: float
    vocabulary_match: bool  # Was this from vocabulary or inferred?
    routing_hint: Optional[str] = None  # Which agent/tier to route to


@dataclass
class QueryExtractionResult:
    """Result of query-time extraction"""

    entities: List[Entity]
    brand_filter: Optional[str] = None
    region_filter: Optional[str] = None
    kpi_filter: Optional[str] = None
    suggested_agent: Optional[str] = None
    suggested_tier: Optional[int] = None

    def to_filter_dict(self) -> Dict:
        """Convert to SQL filter parameters"""
        filters = {}

        if self.brand_filter:
            filters["brand"] = self.brand_filter
        if self.region_filter:
            filters["region"] = self.region_filter
        if self.kpi_filter:
            filters["kpi"] = self.kpi_filter

        return filters


class E2IQueryExtractor:
    """
    Fast entity extraction for query routing in E2I NLP Layer

    Designed for:
    - User query parsing
    - Agent routing decisions
    - SQL filter construction
    - Real-time response (< 50ms)

    NOT designed for:
    - Episode extraction (use Graphiti)
    - Knowledge graph building (use Graphiti)
    - Relationship extraction (use Graphiti)
    """

    def __init__(self, vocab_path: str):
        """
        Initialize query extractor

        Args:
            vocab_path: Path to domain_vocabulary.yaml
        """
        with open(vocab_path, "r") as f:
            self.vocab = yaml.safe_load(f)

        logger.info(f"Loaded vocabulary with {len(self.vocab)} categories")

    def extract_for_routing(self, query: str) -> QueryExtractionResult:
        """
        Extract entities from user query for routing

        This is FAST (< 50ms) because it only uses vocabulary matching.
        No LLM calls, no GliNER - those are for Graphiti episodes.

        Args:
            query: User's natural language query

        Returns:
            QueryExtractionResult with entities and routing hints
        """
        import time

        start = time.time()

        entities = []

        # Extract brands
        brand_entity = self._extract_brand(query)
        if brand_entity:
            entities.append(brand_entity)

        # Extract regions
        region_entity = self._extract_region(query)
        if region_entity:
            entities.append(region_entity)

        # Extract KPIs
        kpi_entities = self._extract_kpis(query)
        entities.extend(kpi_entities)

        # Extract diagnosis codes
        diagnosis_entities = self._extract_diagnosis_codes(query)
        entities.extend(diagnosis_entities)

        # Determine routing
        suggested_agent, suggested_tier = self._determine_routing(entities, query)

        # Build result
        result = QueryExtractionResult(
            entities=entities,
            brand_filter=brand_entity.text if brand_entity else None,
            region_filter=region_entity.text if region_entity else None,
            kpi_filter=kpi_entities[0].text if kpi_entities else None,
            suggested_agent=suggested_agent,
            suggested_tier=suggested_tier,
        )

        elapsed_ms = (time.time() - start) * 1000
        logger.info(f"Query extraction: {len(entities)} entities in {elapsed_ms:.1f}ms")

        return result

    def _extract_brand(self, query: str) -> Optional[Entity]:
        """Extract pharmaceutical brand"""
        query_lower = query.lower()

        # Check main brand names
        for brand in self.vocab.get("brands", {}).get("values", []):
            if brand.lower() in query_lower:
                return Entity(
                    text=brand,
                    entity_type="brand",
                    confidence=1.0,
                    vocabulary_match=True,
                    routing_hint="brand_specific_analysis",
                )

        # Check aliases
        for canonical, aliases in self.vocab.get("aliases", {}).items():
            if isinstance(aliases, list):
                for alias in aliases:
                    if alias.lower() in query_lower:
                        # Determine if this is a brand
                        if canonical in self.vocab.get("brands", {}).get("values", []):
                            return Entity(
                                text=canonical,  # Use canonical form
                                entity_type="brand",
                                confidence=0.95,
                                vocabulary_match=True,
                                routing_hint="brand_specific_analysis",
                            )

        return None

    def _extract_region(self, query: str) -> Optional[Entity]:
        """Extract US region"""
        query_lower = query.lower()

        for region in self.vocab.get("regions", {}).get("values", []):
            if region.lower() in query_lower:
                return Entity(
                    text=region,
                    entity_type="region",
                    confidence=1.0,
                    vocabulary_match=True,
                    routing_hint="regional_analysis",
                )

        return None

    def _extract_kpis(self, query: str) -> List[Entity]:
        """Extract KPI mentions"""
        entities = []
        query_lower = query.lower()

        # Check KPI vocabulary
        for kpi_name, kpi_data in self.vocab.get("kpis", {}).items():
            if isinstance(kpi_data, dict):
                # Check display name
                display_name = kpi_data.get("display_name", kpi_name)
                if display_name.lower() in query_lower:
                    entities.append(
                        Entity(
                            text=display_name,
                            entity_type="kpi",
                            confidence=1.0,
                            vocabulary_match=True,
                            routing_hint="kpi_analysis",
                        )
                    )
                    continue

                # Check aliases
                aliases = kpi_data.get("aliases", [])
                for alias in aliases:
                    if alias.lower() in query_lower:
                        entities.append(
                            Entity(
                                text=display_name,  # Use display name, not alias
                                entity_type="kpi",
                                confidence=0.95,
                                vocabulary_match=True,
                                routing_hint="kpi_analysis",
                            )
                        )
                        break

        # Detect common KPI patterns
        kpi_patterns = {
            r"adoption\s+rate": "adoption_rate",
            r"market\s+share": "market_share",
            r"patient\s+count": "patient_count",
            r"prescription\s+volume": "prescription_volume",
        }

        for pattern, kpi_type in kpi_patterns.items():
            if re.search(pattern, query_lower):
                entities.append(
                    Entity(
                        text=kpi_type.replace("_", " ").title(),
                        entity_type="kpi",
                        confidence=0.8,
                        vocabulary_match=False,  # Inferred from pattern
                        routing_hint="kpi_analysis",
                    )
                )

        return entities

    def _extract_diagnosis_codes(self, query: str) -> List[Entity]:
        """Extract ICD-10 or other diagnosis codes"""
        entities = []

        # ICD-10 pattern: Letter followed by 2 digits, optional decimal and more digits
        icd10_pattern = self.vocab.get("diagnosis_codes", {}).get("pattern", "")

        if icd10_pattern:
            matches = re.findall(icd10_pattern, query)
            for match in matches:
                entities.append(
                    Entity(
                        text=match,
                        entity_type="diagnosis_code",
                        confidence=1.0,
                        vocabulary_match=True,
                        routing_hint="clinical_analysis",
                    )
                )

        return entities

    def _determine_routing(
        self, entities: List[Entity], query: str
    ) -> Tuple[Optional[str], Optional[int]]:
        """
        Determine which agent and tier to route query to

        Returns:
            (agent_name, tier_number)
        """
        query_lower = query.lower()

        # Tier 0: ML Foundation
        if any(term in query_lower for term in ["drift", "model", "training", "feature"]):
            return ("drift_monitor", 0)

        # Tier 1: Causal Inference
        if any(term in query_lower for term in ["causal", "cause", "effect", "impact"]):
            return ("causal_impact", 1)

        # Tier 2: Prediction
        if any(term in query_lower for term in ["predict", "forecast", "future", "trend"]):
            return ("prediction_synthesizer", 2)

        # Tier 3: Monitoring
        if any(term in query_lower for term in ["monitor", "track", "alert"]):
            return ("drift_monitor", 3)

        # Tier 4: Explanation
        if any(term in query_lower for term in ["explain", "why", "how", "interpret"]):
            return ("explainer", 5)

        # Default: Orchestrator
        return ("orchestrator", None)


# ============================================
# VOCABULARY-GRAPHITI BRIDGE
# ============================================


class VocabularyEnrichedGraphiti:
    """
    Bridge between E2I domain vocabulary and Graphiti episodes

    Purpose: Help Graphiti understand E2I domain entities by adding
    vocabulary hints to episode content before extraction.
    """

    def __init__(self, graphiti_client: Any, query_extractor: E2IQueryExtractor):
        """
        Initialize bridge

        Args:
            graphiti_client: Graphiti instance
            query_extractor: E2IQueryExtractor for vocabulary scanning
        """
        self.graphiti = graphiti_client
        self.extractor = query_extractor

    async def add_episode_with_vocab_hints(
        self, content: str, name: str, source_description: str, group_id: Optional[str] = None
    ):
        """
        Add episode to Graphiti with vocabulary entity hints

        This helps Graphiti's LLM better understand domain entities.

        Args:
            content: Episode content (agent output, user query, etc.)
            name: Episode identifier
            source_description: Source of the episode
            group_id: Optional tenant isolation ID
        """
        # Step 1: Quick vocabulary scan
        extraction = self.extractor.extract_for_routing(content)

        # Step 2: Annotate content with entity type hints
        enriched_content = self._annotate_content(content, extraction.entities)

        # Step 3: Build metadata with vocabulary entities
        {
            "vocabulary_entities": [
                {"text": e.text, "type": e.entity_type, "vocabulary_match": e.vocabulary_match}
                for e in extraction.entities
            ],
            "source_agent": self._infer_source_agent(source_description),
        }

        # Step 4: Send to Graphiti
        await self.graphiti.add_episode(
            name=name,
            content=enriched_content,
            source_description=source_description,
            group_id=group_id,
            # metadata=metadata  # Uncomment if Graphiti supports metadata
        )

        logger.info(f"Added episode '{name}' with {len(extraction.entities)} vocabulary hints")

    def _annotate_content(self, content: str, entities: List[Entity]) -> str:
        """
        Add inline entity type hints

        Example:
        Before: "Remibrutinib adoption increased in northeast"
        After:  "Remibrutinib [E2I:brand] adoption increased in northeast [E2I:region]"
        """
        annotated = content

        # Sort entities by length (longest first) to avoid partial replacements
        sorted_entities = sorted(entities, key=lambda e: len(e.text), reverse=True)

        for entity in sorted_entities:
            # Add type annotation
            annotation = f"{entity.text} [E2I:{entity.entity_type}]"

            # Replace first occurrence only
            annotated = annotated.replace(entity.text, annotation, 1)

        return annotated

    def _infer_source_agent(self, source_description: str) -> str:
        """Infer which E2I agent produced this content"""
        source_lower = source_description.lower()

        if "causal" in source_lower:
            return "causal_impact"
        elif "predict" in source_lower:
            return "prediction_synthesizer"
        elif "explain" in source_lower:
            return "explainer"
        elif "drift" in source_lower:
            return "drift_monitor"
        else:
            return "unknown"


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    """
    Example usage demonstrating query-time extraction
    """

    # Initialize query extractor
    extractor = E2IQueryExtractor("config/domain_vocabulary.yaml")

    # Test query 1: Brand + Region
    query1 = "What's the efficacy of remi in the northeast?"
    result1 = extractor.extract_for_routing(query1)

    print("=" * 60)
    print(f"Query: {query1}")
    print(f"Entities: {len(result1.entities)}")
    for entity in result1.entities:
        print(f"  - {entity.text} ({entity.entity_type}) [confidence: {entity.confidence}]")
    print(f"Suggested agent: {result1.suggested_agent}")
    print(f"SQL filters: {result1.to_filter_dict()}")

    # Test query 2: KPI-focused
    query2 = "Show me the adoption rate trend for Remibrutinib"
    result2 = extractor.extract_for_routing(query2)

    print("\n" + "=" * 60)
    print(f"Query: {query2}")
    print(f"Entities: {len(result2.entities)}")
    for entity in result2.entities:
        print(f"  - {entity.text} ({entity.entity_type})")
    print(f"Suggested agent: {result2.suggested_agent}")

    # Test query 3: Causal analysis
    query3 = "What caused the increase in CSU diagnoses?"
    result3 = extractor.extract_for_routing(query3)

    print("\n" + "=" * 60)
    print(f"Query: {query3}")
    print(f"Entities: {len(result3.entities)}")
    for entity in result3.entities:
        print(f"  - {entity.text} ({entity.entity_type})")
    print(f"Suggested agent: {result3.suggested_agent}")
    print(f"Suggested tier: {result3.suggested_tier}")

    # Example: Vocabulary-Graphiti Bridge
    print("\n" + "=" * 60)
    print("VOCABULARY-GRAPHITI BRIDGE EXAMPLE")
    print("=" * 60)

    # Simulated Graphiti client (would be real in production)
    class MockGraphiti:
        async def add_episode(self, **kwargs):
            print("\nEpisode added to Graphiti:")
            print(f"  Name: {kwargs['name']}")
            print(f"  Content: {kwargs['content'][:100]}...")
            print(f"  Source: {kwargs['source_description']}")

    graphiti = MockGraphiti()
    bridge = VocabularyEnrichedGraphiti(graphiti, extractor)

    # Simulate agent output
    import asyncio

    async def demo_bridge():
        await bridge.add_episode_with_vocab_hints(
            content="Analysis shows Remibrutinib adoption increased 40% in northeast among oncologists",
            name="causal_impact_001",
            source_description="Causal Impact Agent - Treatment Adoption Analysis",
            group_id="remibrutinib_csu",
        )

    asyncio.run(demo_bridge())

    print("\nDemo complete!")
