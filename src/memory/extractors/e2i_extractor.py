"""
E2I Entity and Relationship Extractor
Custom extraction logic for E2I-specific entities and relationships.

This module provides extraction capabilities for:
- Healthcare entities (HCP, Patient, Brand)
- Business entities (KPI, Region)
- Causal relationships (CAUSES, IMPACTS)
- Agent-generated insights
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from ..graphiti_config import E2IEntityType, E2IRelationshipType

logger = logging.getLogger(__name__)


# E2I-specific patterns for entity extraction
ENTITY_PATTERNS = {
    E2IEntityType.HCP: [
        r"(?:Dr\.?|Doctor)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"(?:physician|specialist|HCP)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"(?:NPI[:\s]+)(\d{10})",
    ],
    E2IEntityType.PATIENT: [
        r"patient\s+([A-Z0-9-]+)",
        r"patient\s+ID[:\s]+([A-Z0-9-]+)",
        r"(?:P|PAT)-?(\d+)",
    ],
    E2IEntityType.BRAND: [
        r"(?:Remibrutinib|Fabhalta|Kisqali)",
        r"(?:brand|product|drug)\s+([A-Z][a-z]+)",
    ],
    E2IEntityType.REGION: [
        r"(?:region|territory)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"(?:Northeast|Southeast|Midwest|Southwest|West)\s*(?:region)?",
    ],
    E2IEntityType.KPI: [
        r"(?:NRx|TRx|NBRx|market share|conversion rate|reach)",
        r"KPI[:\s]+([A-Za-z_]+)",
    ],
}

# Relationship trigger phrases
RELATIONSHIP_TRIGGERS = {
    E2IRelationshipType.CAUSES: [
        "causes", "leads to", "results in", "triggers", "drives",
        "is the cause of", "contributes to",
    ],
    E2IRelationshipType.IMPACTS: [
        "impacts", "affects", "influences", "has effect on",
        "increases", "decreases", "improves", "reduces",
    ],
    E2IRelationshipType.PRESCRIBES: [
        "prescribes", "prescribed", "recommends", "treating with",
    ],
    E2IRelationshipType.TREATED_BY: [
        "treated by", "under care of", "patient of",
    ],
    E2IRelationshipType.DISCOVERS: [
        "discovered", "found", "identified", "detected",
    ],
}


@dataclass
class ExtractedMention:
    """A mention of an entity in text."""
    entity_type: E2IEntityType
    text: str
    start: int
    end: int
    confidence: float = 0.8
    normalized_name: Optional[str] = None


@dataclass
class ExtractedRelationshipMention:
    """A mention of a relationship in text."""
    relationship_type: E2IRelationshipType
    source_mention: ExtractedMention
    target_mention: ExtractedMention
    trigger_text: str
    confidence: float = 0.7


class E2IEntityExtractor:
    """
    Custom entity and relationship extractor for E2I domain.

    This extractor uses pattern matching and heuristics to identify
    E2I-specific entities and relationships in text.
    """

    def __init__(self, min_confidence: float = 0.5):
        """
        Initialize the extractor.

        Args:
            min_confidence: Minimum confidence threshold for extraction
        """
        self.min_confidence = min_confidence
        self._compiled_patterns: Dict[E2IEntityType, List[re.Pattern]] = {}

        # Compile patterns
        for entity_type, patterns in ENTITY_PATTERNS.items():
            self._compiled_patterns[entity_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def extract_entities(self, text: str) -> List[ExtractedMention]:
        """
        Extract entity mentions from text.

        Args:
            text: Input text to extract entities from

        Returns:
            List of ExtractedMention objects
        """
        mentions = []

        for entity_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    # Calculate confidence based on pattern specificity
                    confidence = self._calculate_entity_confidence(
                        entity_type, match.group(0), text
                    )

                    if confidence >= self.min_confidence:
                        mentions.append(ExtractedMention(
                            entity_type=entity_type,
                            text=match.group(0),
                            start=match.start(),
                            end=match.end(),
                            confidence=confidence,
                            normalized_name=self._normalize_entity_name(
                                entity_type, match.group(0)
                            ),
                        ))

        # Remove duplicates and overlapping mentions
        mentions = self._deduplicate_mentions(mentions)

        return mentions

    def extract_relationships(
        self,
        text: str,
        entities: Optional[List[ExtractedMention]] = None,
    ) -> List[ExtractedRelationshipMention]:
        """
        Extract relationship mentions from text.

        Args:
            text: Input text
            entities: Optional pre-extracted entities

        Returns:
            List of ExtractedRelationshipMention objects
        """
        if entities is None:
            entities = self.extract_entities(text)

        if len(entities) < 2:
            return []

        relationships = []
        text_lower = text.lower()

        for rel_type, triggers in RELATIONSHIP_TRIGGERS.items():
            for trigger in triggers:
                # Find trigger in text
                idx = text_lower.find(trigger)
                if idx == -1:
                    continue

                # Find entities before and after trigger
                source_entity = None
                target_entity = None

                for entity in entities:
                    if entity.end <= idx:
                        # Entity before trigger - potential source
                        if source_entity is None or entity.end > source_entity.end:
                            source_entity = entity
                    elif entity.start >= idx + len(trigger):
                        # Entity after trigger - potential target
                        if target_entity is None or entity.start < target_entity.start:
                            target_entity = entity

                if source_entity and target_entity:
                    confidence = self._calculate_relationship_confidence(
                        rel_type, source_entity, target_entity, trigger, text
                    )

                    if confidence >= self.min_confidence:
                        relationships.append(ExtractedRelationshipMention(
                            relationship_type=rel_type,
                            source_mention=source_entity,
                            target_mention=target_entity,
                            trigger_text=trigger,
                            confidence=confidence,
                        ))

        return relationships

    def _calculate_entity_confidence(
        self,
        entity_type: E2IEntityType,
        matched_text: str,
        full_text: str,
    ) -> float:
        """Calculate confidence score for an entity mention."""
        base_confidence = 0.7

        # Boost for exact brand names
        if entity_type == E2IEntityType.BRAND:
            if matched_text in ["Remibrutinib", "Fabhalta", "Kisqali"]:
                base_confidence = 0.95

        # Boost for NPI patterns
        if entity_type == E2IEntityType.HCP:
            if re.match(r"\d{10}", matched_text):
                base_confidence = 0.9

        # Boost for structured IDs
        if entity_type == E2IEntityType.PATIENT:
            if re.match(r"[A-Z]-?\d+", matched_text):
                base_confidence = 0.85

        # Context boost - if entity type word appears nearby
        context_window = 50
        start = max(0, full_text.find(matched_text) - context_window)
        end = min(len(full_text), full_text.find(matched_text) + len(matched_text) + context_window)
        context = full_text[start:end].lower()

        if entity_type.value.lower() in context:
            base_confidence = min(1.0, base_confidence + 0.1)

        return base_confidence

    def _calculate_relationship_confidence(
        self,
        rel_type: E2IRelationshipType,
        source: ExtractedMention,
        target: ExtractedMention,
        trigger: str,
        full_text: str,
    ) -> float:
        """Calculate confidence score for a relationship mention."""
        base_confidence = 0.6

        # Boost for strong trigger words
        strong_triggers = ["causes", "impacts", "prescribes"]
        if trigger in strong_triggers:
            base_confidence = 0.75

        # Boost for compatible entity types
        compatible_pairs = {
            E2IRelationshipType.PRESCRIBES: [
                (E2IEntityType.HCP, E2IEntityType.BRAND),
            ],
            E2IRelationshipType.TREATED_BY: [
                (E2IEntityType.PATIENT, E2IEntityType.HCP),
            ],
            E2IRelationshipType.IMPACTS: [
                (E2IEntityType.BRAND, E2IEntityType.KPI),
                (E2IEntityType.HCP, E2IEntityType.KPI),
            ],
        }

        if rel_type in compatible_pairs:
            for source_type, target_type in compatible_pairs[rel_type]:
                if source.entity_type == source_type and target.entity_type == target_type:
                    base_confidence = min(1.0, base_confidence + 0.15)
                    break

        # Distance penalty - closer entities are more likely related
        distance = target.start - source.end
        if distance < 20:
            base_confidence = min(1.0, base_confidence + 0.1)
        elif distance > 100:
            base_confidence = max(0.3, base_confidence - 0.15)

        # Combine with entity confidences
        entity_factor = (source.confidence + target.confidence) / 2
        final_confidence = base_confidence * entity_factor

        return final_confidence

    def _normalize_entity_name(
        self,
        entity_type: E2IEntityType,
        text: str,
    ) -> str:
        """Normalize entity name for deduplication."""
        # Remove titles and prefixes
        text = re.sub(r"^(Dr\.?|Doctor|physician|specialist)\s+", "", text, flags=re.IGNORECASE)

        # Clean up IDs
        if entity_type == E2IEntityType.PATIENT:
            text = re.sub(r"patient\s+(?:ID[:\s]+)?", "", text, flags=re.IGNORECASE)

        # Title case names
        if entity_type in [E2IEntityType.HCP, E2IEntityType.REGION]:
            text = text.title()

        # Preserve brand names exactly
        if entity_type == E2IEntityType.BRAND:
            brand_map = {
                "remibrutinib": "Remibrutinib",
                "fabhalta": "Fabhalta",
                "kisqali": "Kisqali",
            }
            text = brand_map.get(text.lower(), text)

        return text.strip()

    def _deduplicate_mentions(
        self,
        mentions: List[ExtractedMention],
    ) -> List[ExtractedMention]:
        """Remove duplicate and overlapping mentions."""
        if not mentions:
            return []

        # Sort by start position
        mentions.sort(key=lambda m: (m.start, -m.end))

        deduplicated = []
        for mention in mentions:
            # Check for overlap with existing mentions
            overlaps = False
            for existing in deduplicated:
                if (mention.start < existing.end and mention.end > existing.start):
                    # Overlapping - keep the one with higher confidence
                    if mention.confidence > existing.confidence:
                        deduplicated.remove(existing)
                    else:
                        overlaps = True
                    break

            if not overlaps:
                deduplicated.append(mention)

        return deduplicated


def extract_e2i_entities(text: str, min_confidence: float = 0.5) -> List[ExtractedMention]:
    """
    Convenience function to extract E2I entities from text.

    Args:
        text: Input text
        min_confidence: Minimum confidence threshold

    Returns:
        List of ExtractedMention objects
    """
    extractor = E2IEntityExtractor(min_confidence=min_confidence)
    return extractor.extract_entities(text)


def extract_e2i_relationships(
    text: str,
    entities: Optional[List[ExtractedMention]] = None,
    min_confidence: float = 0.5,
) -> List[ExtractedRelationshipMention]:
    """
    Convenience function to extract E2I relationships from text.

    Args:
        text: Input text
        entities: Optional pre-extracted entities
        min_confidence: Minimum confidence threshold

    Returns:
        List of ExtractedRelationshipMention objects
    """
    extractor = E2IEntityExtractor(min_confidence=min_confidence)
    return extractor.extract_relationships(text, entities)
