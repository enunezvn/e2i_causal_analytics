"""Entity definitions for E2I Causal Analytics Feature Store.

Entities are the subjects for which features are computed. In the pharma
commercial domain, our primary entities are:

1. HCP (Healthcare Provider) - Doctors and prescribers
2. Patient - Anonymized patient records for journey analysis
3. Territory - Geographic sales territories
4. Brand - Pharmaceutical products (Remibrutinib, Fabhalta, Kisqali)
5. Trigger - Marketing triggers/events

Note: Composite entities use single composite keys (e.g., "hcp_001_brand_001")
because Feast 0.58.0 only supports single join keys per entity.
"""

from feast import Entity, ValueType

# =============================================================================
# Primary Entities
# =============================================================================

hcp = Entity(
    name="hcp",
    join_keys=["hcp_id"],
    value_type=ValueType.STRING,
    description="Healthcare Provider (physician, prescriber). "
                "Core entity for HCP targeting and conversion analysis.",
    tags={
        "domain": "commercial",
        "owner": "analytics-team",
        "pii": "false",
    },
)

patient = Entity(
    name="patient",
    join_keys=["patient_id"],
    value_type=ValueType.STRING,
    description="Anonymized patient for journey analysis. "
                "Used for adherence, churn, and therapy optimization.",
    tags={
        "domain": "commercial",
        "owner": "analytics-team",
        "pii": "pseudonymized",
        "retention_days": "365",
    },
)

territory = Entity(
    name="territory",
    join_keys=["territory_id"],
    value_type=ValueType.STRING,
    description="Geographic sales territory. "
                "Used for resource allocation and market analysis.",
    tags={
        "domain": "commercial",
        "owner": "sales-ops",
        "pii": "false",
    },
)

brand = Entity(
    name="brand",
    join_keys=["brand_id"],
    value_type=ValueType.STRING,
    description="Pharmaceutical brand/product. "
                "Current brands: Remibrutinib (CSU), Fabhalta (PNH), Kisqali (HR+/HER2-).",
    tags={
        "domain": "commercial",
        "owner": "brand-team",
        "pii": "false",
    },
)

trigger = Entity(
    name="trigger",
    join_keys=["trigger_id"],
    value_type=ValueType.STRING,
    description="Marketing trigger/event. "
                "Used for campaign effectiveness and response analysis.",
    tags={
        "domain": "commercial",
        "owner": "marketing",
        "pii": "false",
    },
)


# =============================================================================
# Composite Entities (using single composite keys)
# Feast 0.58.0 only supports single join keys, so we use composite key strings
# Format: "{entity1_id}_{entity2_id}" e.g., "hcp_001_brand_001"
# =============================================================================

hcp_brand = Entity(
    name="hcp_brand",
    join_keys=["hcp_brand_id"],  # Composite key: "{hcp_id}_{brand_id}"
    value_type=ValueType.STRING,
    description="HCP-Brand composite entity for brand-specific HCP features. "
                "Key format: '{hcp_id}_{brand_id}'",
    tags={
        "domain": "commercial",
        "owner": "analytics-team",
        "composite": "true",
        "key_format": "{hcp_id}_{brand_id}",
    },
)

patient_brand = Entity(
    name="patient_brand",
    join_keys=["patient_brand_id"],  # Composite key: "{patient_id}_{brand_id}"
    value_type=ValueType.STRING,
    description="Patient-Brand composite entity for therapy-specific journey features. "
                "Key format: '{patient_id}_{brand_id}'",
    tags={
        "domain": "commercial",
        "owner": "analytics-team",
        "composite": "true",
        "pii": "pseudonymized",
        "key_format": "{patient_id}_{brand_id}",
    },
)

hcp_territory = Entity(
    name="hcp_territory",
    join_keys=["hcp_territory_id"],  # Composite key: "{hcp_id}_{territory_id}"
    value_type=ValueType.STRING,
    description="HCP-Territory composite for territory-specific targeting. "
                "Key format: '{hcp_id}_{territory_id}'",
    tags={
        "domain": "commercial",
        "owner": "sales-ops",
        "composite": "true",
        "key_format": "{hcp_id}_{territory_id}",
    },
)


# =============================================================================
# Entity Registry
# =============================================================================

# All entities for registration
ALL_ENTITIES = [
    # Primary entities
    hcp,
    patient,
    territory,
    brand,
    trigger,
    # Composite entities
    hcp_brand,
    patient_brand,
    hcp_territory,
]

# Entity lookup by name
ENTITY_MAP = {e.name: e for e in ALL_ENTITIES}


def get_entity(name: str) -> Entity:
    """Get an entity by name.

    Args:
        name: Entity name (e.g., "hcp", "patient")

    Returns:
        Entity object

    Raises:
        KeyError: If entity not found
    """
    if name not in ENTITY_MAP:
        available = ", ".join(ENTITY_MAP.keys())
        raise KeyError(f"Entity '{name}' not found. Available: {available}")
    return ENTITY_MAP[name]
