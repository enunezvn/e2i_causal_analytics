"""
E2I Hybrid RAG - Entity Extractor

Extracts domain-specific entities from natural language queries for graph search.
Uses fixed vocabularies from domain configuration - NO medical NER required.

Extracts:
- Brands (Remibrutinib, Fabhalta, Kisqali)
- Regions (Northeast, South, Midwest, West)
- KPIs (TRx, NRx, conversion_rate, etc.)
- Agents (orchestrator, causal_impact, etc.)
- Patient journey stages
- Time references (Q1, Q2, YTD, etc.)
- HCP segments

Part of Phase 1, Checkpoint 1.6.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.rag.exceptions import EntityExtractionError
from src.rag.types import ExtractedEntities

logger = logging.getLogger(__name__)


@dataclass
class EntityVocabulary:
    """
    Domain vocabulary for entity extraction.

    Loaded from YAML config or provided directly.
    Contains normalized forms and aliases for each entity type.
    """

    brands: Dict[str, List[str]] = field(default_factory=dict)
    regions: Dict[str, List[str]] = field(default_factory=dict)
    kpis: Dict[str, List[str]] = field(default_factory=dict)
    agents: Dict[str, List[str]] = field(default_factory=dict)
    journey_stages: Dict[str, List[str]] = field(default_factory=dict)
    time_references: Dict[str, List[str]] = field(default_factory=dict)
    hcp_segments: Dict[str, List[str]] = field(default_factory=dict)

    @classmethod
    def from_default(cls) -> "EntityVocabulary":
        """
        Create vocabulary from central VocabularyRegistry with entity extraction aliases.

        Canonical values sourced from config/domain_vocabulary.yaml via VocabularyRegistry.
        Extended aliases for NLP matching defined here to support entity extraction.
        """
        try:
            from src.ontology import VocabularyRegistry

            vocab = VocabularyRegistry.load()
            canonical_brands = vocab.get_brands()
            canonical_regions = vocab.get_regions()
            vocab.get_agent_names()
            vocab.get_journey_stages()
            vocab.get_hcp_segments()
        except Exception:
            # Fallback if VocabularyRegistry unavailable
            canonical_brands = ["Remibrutinib", "Fabhalta", "Kisqali"]
            canonical_regions = ["northeast", "south", "midwest", "west"]

        # Build brands with extraction-friendly aliases
        # Canonical names from VocabularyRegistry, aliases for NLP matching
        brand_aliases = {
            "Remibrutinib": ["remibrutinib", "remi", "btk inhibitor"],
            "Fabhalta": ["fabhalta", "factor b", "factor b inhibitor"],
            "Kisqali": ["kisqali", "ribociclib", "cdk4/6", "cdk4", "cdk6"],
        }
        # Ensure all canonical brands are included
        brands = {}
        for brand in canonical_brands:
            brands[brand] = brand_aliases.get(brand, [brand.lower()])

        # Region aliases for NLP matching
        region_aliases = {
            "northeast": ["northeast", "ne", "north east", "new england"],
            "south": ["south", "southeast", "se", "southwest", "sw", "southern"],
            "midwest": ["midwest", "mw", "mid west", "central"],
            "west": ["west", "pacific", "northwest", "nw", "western"],
        }
        regions = {}
        for region in canonical_regions:
            regions[region] = region_aliases.get(region, [region.lower()])

        return cls(
            brands=brands,
            regions=regions,
            kpis={
                "trx": ["trx", "total rx", "total prescriptions", "prescriptions"],
                "nrx": ["nrx", "new rx", "new prescriptions"],
                "nbrx": ["nbrx", "new-to-brand", "new to brand"],
                "trx_share": ["trx share", "market share", "share"],
                "conversion_rate": ["conversion", "conversion rate", "convert"],
                "roi": ["roi", "return on investment", "return"],
                "treatment_effect_ate": ["ate", "average treatment effect"],
                "treatment_effect_cate": ["cate", "conditional average treatment effect"],
                "causal_impact": ["causal impact", "impact"],
                "hcp_coverage": ["hcp coverage", "coverage", "reach"],
                "patient_touch_rate": ["patient touch", "touch rate"],
                "adoption": ["adoption", "uptake"],
            },
            agents={
                "orchestrator": ["orchestrator", "coordinator"],
                "causal_impact": ["causal impact", "causal", "impact analyzer"],
                "gap_analyzer": ["gap analyzer", "gap", "gap analysis"],
                "heterogeneous_optimizer": [
                    "heterogeneous optimizer",
                    "het optimizer",
                    "cate optimizer",
                ],
                "drift_monitor": ["drift monitor", "drift", "monitoring"],
                "experiment_designer": ["experiment designer", "experiment", "a/b test"],
                "prediction_synthesizer": ["prediction", "synthesizer", "forecast"],
                "explainer": ["explainer", "explanation"],
                "feedback_learner": ["feedback learner", "feedback", "learning"],
            },
            journey_stages={
                "diagnosis": ["diagnosis", "diagnosed", "dx"],
                "treatment_naive": ["treatment naive", "naive", "newly diagnosed"],
                "first_line": ["first line", "1l", "first-line"],
                "second_line": ["second line", "2l", "second-line"],
                "maintenance": ["maintenance", "maint"],
                "discontinuation": ["discontinuation", "discontinue", "stopped"],
                "switch": ["switch", "switching", "switched"],
            },
            time_references={
                "Q1": ["q1", "q1 2024", "q1 2025", "first quarter"],
                "Q2": ["q2", "q2 2024", "q2 2025", "second quarter"],
                "Q3": ["q3", "q3 2024", "q3 2025", "third quarter"],
                "Q4": ["q4", "q4 2024", "q4 2025", "fourth quarter"],
                "YTD": ["ytd", "year to date", "year-to-date"],
                "MTD": ["mtd", "month to date", "month-to-date"],
                "last_week": ["last week", "past week", "previous week"],
                "last_month": ["last month", "past month", "previous month"],
                "last_quarter": ["last quarter", "past quarter", "previous quarter"],
                "last_year": ["last year", "past year", "previous year"],
            },
            hcp_segments={
                "high_volume": ["high volume", "high-volume", "top prescribers"],
                "medium_volume": ["medium volume", "medium-volume", "mid volume"],
                "low_volume": ["low volume", "low-volume", "low prescribers"],
                "academic": ["academic", "academic center", "teaching hospital"],
                "community": ["community", "community practice", "private practice"],
                "key_opinion_leader": ["kol", "kols", "key opinion leader", "thought leader"],
            },
        )


class EntityExtractor:
    """
    Extracts domain-specific entities from natural language queries.

    Uses fixed vocabularies - NO medical NER or clinical entity extraction.
    Designed for E2I pharmaceutical commercial operations analytics.

    Example:
        ```python
        extractor = EntityExtractor()

        entities = extractor.extract(
            "Why did Kisqali TRx drop in the West during Q3?"
        )

        print(entities.brands)       # ["Kisqali"]
        print(entities.kpis)         # ["trx"]
        print(entities.regions)      # ["west"]
        print(entities.time_references)  # ["Q3"]
        ```
    """

    def __init__(
        self, vocabulary: Optional[EntityVocabulary] = None, config_path: Optional[str] = None
    ):
        """
        Initialize the entity extractor.

        Args:
            vocabulary: Pre-built vocabulary (if None, uses defaults)
            config_path: Path to YAML vocabulary config file
        """
        if vocabulary:
            self.vocabulary = vocabulary
        elif config_path:
            self.vocabulary = self._load_vocabulary(config_path)
        else:
            self.vocabulary = EntityVocabulary.from_default()

        # Build reverse lookup indices for faster extraction
        self._build_indices()

    def _load_vocabulary(self, config_path: str) -> EntityVocabulary:
        """Load vocabulary from YAML config file."""
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("PyYAML not installed, using default vocabulary")
            return EntityVocabulary.from_default()

        try:
            path = Path(config_path)
            if not path.exists():
                logger.warning(f"Config not found: {config_path}, using defaults")
                return EntityVocabulary.from_default()

            with open(path, "r") as f:
                config = yaml.safe_load(f)

            return self._parse_config(config)

        except Exception as e:
            logger.error(f"Failed to load vocabulary from {config_path}: {e}")
            return EntityVocabulary.from_default()

    def _parse_config(self, config: Dict[str, Any]) -> EntityVocabulary:
        """Parse YAML config into EntityVocabulary."""
        vocab = EntityVocabulary()

        # Parse brands
        if "brands" in config:
            brands_data = config["brands"]
            if isinstance(brands_data, dict) and "values" in brands_data:
                for brand in brands_data["values"]:
                    vocab.brands[brand] = [brand.lower()]

        # Parse regions
        if "regions" in config:
            regions_data = config["regions"]
            if isinstance(regions_data, dict) and "values" in regions_data:
                for region in regions_data["values"]:
                    vocab.regions[region] = [region.lower()]

        # Parse time periods
        if "time_periods" in config:
            time_data = config["time_periods"]
            if isinstance(time_data, dict) and "values" in time_data:
                for period in time_data["values"]:
                    vocab.time_references[period] = [period.lower()]

        # Parse HCP segments
        if "hcp_segments" in config:
            segments_data = config["hcp_segments"]
            if isinstance(segments_data, dict) and "values" in segments_data:
                for segment in segments_data["values"]:
                    vocab.hcp_segments[segment] = [segment.lower().replace("_", " ")]

        # Parse patient journey stages
        if "patient_journey_stages" in config:
            journey_data = config["patient_journey_stages"]
            if isinstance(journey_data, dict) and "values" in journey_data:
                for stage in journey_data["values"]:
                    vocab.journey_stages[stage] = [stage.lower().replace("_", " ")]

        # Parse agents
        if "agents" in config:
            agents_data = config["agents"]
            if isinstance(agents_data, dict):
                for _tier, agent_list in agents_data.items():
                    if isinstance(agent_list, list):
                        for agent in agent_list:
                            vocab.agents[agent] = [agent.lower().replace("_", " ")]

        # Merge with defaults to ensure core entities are always present
        default = EntityVocabulary.from_default()
        vocab.brands = {**default.brands, **vocab.brands}
        vocab.kpis = default.kpis  # KPIs from default since YAML structure differs

        return vocab

    def _build_indices(self) -> None:
        """Build reverse lookup indices for fast matching."""
        self._brand_index: Dict[str, str] = {}
        self._region_index: Dict[str, str] = {}
        self._kpi_index: Dict[str, str] = {}
        self._agent_index: Dict[str, str] = {}
        self._journey_index: Dict[str, str] = {}
        self._time_index: Dict[str, str] = {}
        self._segment_index: Dict[str, str] = {}

        # Build each index
        for canonical, aliases in self.vocabulary.brands.items():
            for alias in aliases:
                self._brand_index[alias.lower()] = canonical

        for canonical, aliases in self.vocabulary.regions.items():
            for alias in aliases:
                self._region_index[alias.lower()] = canonical

        for canonical, aliases in self.vocabulary.kpis.items():
            for alias in aliases:
                self._kpi_index[alias.lower()] = canonical

        for canonical, aliases in self.vocabulary.agents.items():
            for alias in aliases:
                self._agent_index[alias.lower()] = canonical

        for canonical, aliases in self.vocabulary.journey_stages.items():
            for alias in aliases:
                self._journey_index[alias.lower()] = canonical

        for canonical, aliases in self.vocabulary.time_references.items():
            for alias in aliases:
                self._time_index[alias.lower()] = canonical

        for canonical, aliases in self.vocabulary.hcp_segments.items():
            for alias in aliases:
                self._segment_index[alias.lower()] = canonical

    def extract(self, query: str) -> ExtractedEntities:
        """
        Extract all entity types from a query.

        Args:
            query: Natural language query text

        Returns:
            ExtractedEntities with all found entities

        Raises:
            EntityExtractionError: If extraction fails
        """
        try:
            query_lower = query.lower()

            return ExtractedEntities(
                brands=self._extract_brands(query_lower),
                regions=self._extract_regions(query_lower),
                kpis=self._extract_kpis(query_lower),
                agents=self._extract_agents(query_lower),
                journey_stages=self._extract_journey_stages(query_lower),
                time_references=self._extract_time_references(query_lower),
                hcp_segments=self._extract_hcp_segments(query_lower),
            )

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            raise EntityExtractionError(
                message=f"Failed to extract entities from query: {e}",
                query=query[:100],
                original_error=e,
            ) from e

    def _extract_brands(self, query: str) -> List[str]:
        """Extract brand names from query."""
        found = set()
        for alias, canonical in self._brand_index.items():
            if self._is_word_match(alias, query):
                found.add(canonical)
        return sorted(found)

    def _extract_regions(self, query: str) -> List[str]:
        """Extract region names from query."""
        found = set()
        for alias, canonical in self._region_index.items():
            if self._is_word_match(alias, query):
                found.add(canonical)
        return sorted(found)

    def _extract_kpis(self, query: str) -> List[str]:
        """Extract KPI names from query."""
        found = set()
        for alias, canonical in self._kpi_index.items():
            if self._is_word_match(alias, query):
                found.add(canonical)
        return sorted(found)

    def _extract_agents(self, query: str) -> List[str]:
        """Extract agent names from query."""
        found = set()
        for alias, canonical in self._agent_index.items():
            if self._is_word_match(alias, query):
                found.add(canonical)
        return sorted(found)

    def _extract_journey_stages(self, query: str) -> List[str]:
        """Extract patient journey stages from query."""
        found = set()
        for alias, canonical in self._journey_index.items():
            if self._is_word_match(alias, query):
                found.add(canonical)
        return sorted(found)

    def _extract_time_references(self, query: str) -> List[str]:
        """Extract time references from query."""
        found = set()
        for alias, canonical in self._time_index.items():
            if self._is_word_match(alias, query):
                found.add(canonical)

        # Also extract year references
        year_pattern = r"\b20[0-9]{2}\b"
        years = re.findall(year_pattern, query)
        found.update(years)

        return sorted(found)

    def _extract_hcp_segments(self, query: str) -> List[str]:
        """Extract HCP segment names from query."""
        found = set()
        for alias, canonical in self._segment_index.items():
            if self._is_word_match(alias, query):
                found.add(canonical)
        return sorted(found)

    def _is_word_match(self, pattern: str, text: str) -> bool:
        """
        Check if pattern exists in text as a word boundary match.

        This prevents partial matches (e.g., "trx" matching "matrix").
        """
        # Escape special regex characters but allow spaces
        escaped = re.escape(pattern)
        # Use word boundaries
        regex = rf"\b{escaped}\b"
        return bool(re.search(regex, text, re.IGNORECASE))

    def extract_with_confidence(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract entities with confidence scores.

        Returns entities with metadata about match quality.

        Args:
            query: Natural language query text

        Returns:
            Dict with entity type -> list of {entity, confidence, source}
        """
        entities = self.extract(query)
        result = {}

        # All matches via vocabulary get high confidence
        for entity_type, values in [
            ("brands", entities.brands),
            ("regions", entities.regions),
            ("kpis", entities.kpis),
            ("agents", entities.agents),
            ("journey_stages", entities.journey_stages),
            ("time_references", entities.time_references),
            ("hcp_segments", entities.hcp_segments),
        ]:
            if values:
                result[entity_type] = [
                    {
                        "entity": v,
                        "confidence": 0.95,  # High confidence for vocabulary matches
                        "source": "vocabulary",
                    }
                    for v in values
                ]

        return result

    def __repr__(self) -> str:
        return (
            f"EntityExtractor("
            f"brands={len(self.vocabulary.brands)}, "
            f"regions={len(self.vocabulary.regions)}, "
            f"kpis={len(self.vocabulary.kpis)})"
        )
