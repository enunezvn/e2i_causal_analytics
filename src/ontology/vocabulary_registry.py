"""
E2I Causal Analytics - Vocabulary Registry

Central vocabulary loader providing a single source of truth for all
domain entities. Eliminates vocabulary duplication across the codebase
by providing cached access to domain_vocabulary.yaml.

Usage:
    from src.ontology import VocabularyRegistry

    vocab = VocabularyRegistry.load()

    # Simple access
    brands = vocab.get_brands()  # ['Remibrutinib', 'Fabhalta', 'Kisqali']
    regions = vocab.get_regions()  # ['northeast', 'south', 'midwest', 'west']

    # Tier-filtered access
    tier2_agents = vocab.get_agents(tier=2)

    # Alias-aware access for entity extraction
    brand_aliases = vocab.get_entity_with_aliases('brands')
    # {'Remibrutinib': ['remi', 'remibrutinib'], 'Fabhalta': ['fab'], ...}
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
import yaml
import logging

logger = logging.getLogger(__name__)


class VocabularyRegistry:
    """
    Central vocabulary registry - single source of truth.

    Loads and caches domain_vocabulary.yaml, providing typed access
    to all domain entities (brands, regions, agents, KPIs, etc.)

    Design Principles:
    - Immutable after load (treat as read-only)
    - Cached via @lru_cache for singleton pattern
    - Fast access (< 1ms for all operations)
    - Fail-fast on missing vocabulary file
    """

    # Default vocabulary path relative to project root
    DEFAULT_VOCAB_PATH = "config/domain_vocabulary.yaml"

    def __init__(self, vocab_data: dict[str, Any], version: str = "unknown"):
        """
        Initialize registry with vocabulary data.

        Use VocabularyRegistry.load() for the cached singleton pattern.

        Args:
            vocab_data: Parsed vocabulary dictionary
            version: Vocabulary version string
        """
        self._data = vocab_data
        self._version = version
        logger.debug(f"VocabularyRegistry initialized (version: {version})")

    @classmethod
    @lru_cache(maxsize=1)
    def load(cls, vocab_path: Optional[str] = None) -> "VocabularyRegistry":
        """
        Load vocabulary from YAML file (cached singleton).

        Args:
            vocab_path: Optional path to vocabulary file.
                       Defaults to config/domain_vocabulary.yaml

        Returns:
            Cached VocabularyRegistry instance

        Raises:
            FileNotFoundError: If vocabulary file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        if vocab_path is None:
            # Find project root by looking for config/ directory
            current = Path(__file__).resolve()
            for parent in current.parents:
                vocab_file = parent / cls.DEFAULT_VOCAB_PATH
                if vocab_file.exists():
                    vocab_path = str(vocab_file)
                    break

            if vocab_path is None:
                raise FileNotFoundError(
                    f"Could not find {cls.DEFAULT_VOCAB_PATH} in project hierarchy"
                )

        path = Path(vocab_path)
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

        logger.info(f"Loading vocabulary from {vocab_path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Extract version from metadata
        version = data.get('metadata', {}).get('version', 'unknown')

        logger.info(f"Loaded vocabulary v{version} with {len(data)} sections")

        return cls(data, version)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the cached singleton (useful for testing)."""
        cls.load.cache_clear()

    @property
    def version(self) -> str:
        """Get vocabulary version."""
        return self._version

    # ================================================================
    # CORE ENTITY ACCESSORS
    # ================================================================

    def get_brands(self) -> list[str]:
        """
        Get list of pharmaceutical brands.

        Returns:
            List of brand names: ['Remibrutinib', 'Fabhalta', 'Kisqali']
        """
        return self._data.get('brands', {}).get('values', [])

    def get_regions(self) -> list[str]:
        """
        Get list of US regions.

        Returns:
            List of region names: ['northeast', 'south', 'midwest', 'west']
        """
        return self._data.get('regions', {}).get('values', [])

    def get_agents(self, tier: Optional[int] = None) -> dict[str, Any]:
        """
        Get agent definitions.

        Args:
            tier: Optional tier filter (0-5). If None, returns all tiers.

        Returns:
            Dictionary of agent configurations by tier or filtered by tier
        """
        agents_data = self._data.get('agents', {})

        if tier is None:
            return agents_data

        # Filter by tier
        tier_key_prefix = f"tier_{tier}_"
        filtered = {}
        for key, value in agents_data.items():
            if key.startswith(tier_key_prefix):
                filtered[key] = value

        return filtered

    def get_agent_names(self, tier: Optional[int] = None) -> list[str]:
        """
        Get flat list of agent names.

        Args:
            tier: Optional tier filter (0-5)

        Returns:
            List of agent names
        """
        agents = self.get_agents(tier)
        names = []
        for key, value in agents.items():
            if isinstance(value, list):
                names.extend(value)
            elif isinstance(value, dict):
                # Handle nested structures
                for v in value.values():
                    if isinstance(v, list):
                        names.extend(v)
        return names

    def get_kpis(self) -> dict[str, Any]:
        """
        Get KPI definitions.

        Returns:
            Dictionary of KPI configurations
        """
        return self._data.get('kpis', {})

    def get_kpi_names(self) -> list[str]:
        """
        Get list of KPI names.

        Returns:
            List of KPI identifiers
        """
        kpis = self.get_kpis()
        return list(kpis.keys())

    def get_journey_stages(self) -> list[str]:
        """
        Get patient journey stages.

        Returns:
            List of journey stage names
        """
        return self._data.get('journey_stages', {}).get('values', [])

    def get_hcp_segments(self) -> list[str]:
        """
        Get HCP segment/specialty values.

        Returns:
            List of HCP segments
        """
        return self._data.get('hcp_segments', {}).get('values', [])

    def get_time_references(self) -> dict[str, Any]:
        """
        Get time reference patterns.

        Returns:
            Dictionary of time reference patterns
        """
        return self._data.get('time_references', {})

    # ================================================================
    # ALIAS-AWARE ACCESSORS (for entity extraction)
    # ================================================================

    def get_entity_with_aliases(self, entity_type: str) -> dict[str, list[str]]:
        """
        Get entities with their aliases for entity extraction.

        Args:
            entity_type: Entity type ('brands', 'regions', 'kpis',
                        'agents', 'journey_stages', 'time_references',
                        'hcp_segments')

        Returns:
            Dictionary mapping canonical names to list of aliases.
            Example: {'Remibrutinib': ['remi', 'remibrutinib', 'Remibrutinib'], ...}
        """
        entity_data = self._data.get(entity_type, {})
        global_aliases = self._data.get('aliases', {})

        if not entity_data:
            return {}

        result = {}

        # Handle different entity structures
        if 'values' in entity_data:
            # Simple list-based entities (brands, regions)
            for value in entity_data.get('values', []):
                # Include the value itself as an alias
                result[value] = [value, value.lower()]
                # Check global aliases for this entity
                if value in global_aliases:
                    entity_aliases = global_aliases[value]
                    if isinstance(entity_aliases, list):
                        result[value].extend(entity_aliases)
                    elif isinstance(entity_aliases, str):
                        result[value].append(entity_aliases)

        if 'aliases' in entity_data:
            # Entities with explicit aliases within the section
            for canonical, aliases in entity_data['aliases'].items():
                if canonical not in result:
                    result[canonical] = [canonical, canonical.lower()]
                if isinstance(aliases, list):
                    result[canonical].extend(aliases)
                elif isinstance(aliases, str):
                    result[canonical].append(aliases)

        # For KPIs, extract from nested structure
        if entity_type == 'kpis':
            for kpi_name, kpi_data in entity_data.items():
                if isinstance(kpi_data, dict):
                    display_name = kpi_data.get('display_name', kpi_name)
                    aliases = kpi_data.get('aliases', [])
                    result[display_name] = [display_name, kpi_name] + list(aliases)

        # For agents, flatten the tier structure
        if entity_type == 'agents':
            for tier_key, agents in entity_data.items():
                if isinstance(agents, list):
                    for agent in agents:
                        result[agent] = [agent, agent.replace('_', ' ')]

        # Deduplicate aliases while preserving order
        for key in result:
            seen = set()
            unique = []
            for alias in result[key]:
                alias_lower = alias.lower()
                if alias_lower not in seen:
                    seen.add(alias_lower)
                    unique.append(alias)
            result[key] = unique

        return result

    def get_aliases(self) -> dict[str, list[str]]:
        """
        Get the global aliases section.

        Returns:
            Dictionary of canonical names to their aliases
        """
        return self._data.get('aliases', {})

    def resolve_alias(self, text: str, entity_type: Optional[str] = None) -> Optional[str]:
        """
        Resolve an alias to its canonical form.

        Args:
            text: Text that might be an alias
            entity_type: Optional entity type to limit search

        Returns:
            Canonical name if alias found, None otherwise
        """
        text_lower = text.lower()

        # Check global aliases first
        aliases = self.get_aliases()
        for canonical, alias_list in aliases.items():
            if isinstance(alias_list, list):
                if text_lower in [a.lower() for a in alias_list]:
                    return canonical

        # Check entity-specific aliases if type provided
        if entity_type:
            entities = self.get_entity_with_aliases(entity_type)
            for canonical, alias_list in entities.items():
                if text_lower in [a.lower() for a in alias_list]:
                    return canonical

        return None

    # ================================================================
    # VALIDATION & ENUMERATION
    # ================================================================

    def get_enum_values(self, enum_name: str) -> list[str]:
        """
        Get values for database ENUM synchronization.

        Args:
            enum_name: Name of the enum ('brand_type', 'region_type', etc.)

        Returns:
            List of valid enum values
        """
        enum_map = {
            'brand_type': self.get_brands,
            'region_type': self.get_regions,
            'journey_stage_type': self.get_journey_stages,
            'hcp_segment_type': self.get_hcp_segments,
        }

        getter = enum_map.get(enum_name)
        if getter:
            return getter()

        return []

    def validate_value(self, value: str, entity_type: str) -> bool:
        """
        Validate a value against the vocabulary.

        Args:
            value: Value to validate
            entity_type: Entity type to validate against

        Returns:
            True if value is valid (exact match or alias)
        """
        # Try exact match first
        entities = self.get_entity_with_aliases(entity_type)

        # Check canonical names
        if value in entities:
            return True

        # Check aliases
        value_lower = value.lower()
        for canonical, aliases in entities.items():
            if value_lower in [a.lower() for a in aliases]:
                return True

        return False

    # ================================================================
    # RAW ACCESS
    # ================================================================

    def get_section(self, section_name: str) -> dict[str, Any]:
        """
        Get a raw section from the vocabulary.

        Args:
            section_name: Name of the section

        Returns:
            Section data or empty dict
        """
        return self._data.get(section_name, {})

    def get_all_sections(self) -> list[str]:
        """
        Get list of all section names in vocabulary.

        Returns:
            List of section names
        """
        return list(self._data.keys())

    def __repr__(self) -> str:
        return f"VocabularyRegistry(version={self._version}, sections={len(self._data)})"


# Convenience function for quick access
def get_vocabulary() -> VocabularyRegistry:
    """
    Get the cached vocabulary registry.

    Convenience wrapper around VocabularyRegistry.load().

    Returns:
        Cached VocabularyRegistry instance
    """
    return VocabularyRegistry.load()
