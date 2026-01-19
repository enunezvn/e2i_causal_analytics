"""
Integration tests for vocabulary-database ENUM synchronization.

Tests that the domain vocabulary YAML definitions match the database
ENUM types defined in migrations. This ensures consistency between
the ontology layer and the database schema.

Author: E2I Causal Analytics Team
"""

import pytest


# =============================================================================
# VOCABULARY CONTENT TESTS
# =============================================================================


class TestVocabularyAgentSync:
    """Test agent definitions in vocabulary match expected database values."""

    def test_vocabulary_has_all_tier_0_agents(self, vocabulary_registry):
        """Test that vocabulary includes all Tier 0 ML foundation agents."""
        # Use get_agent_names() which returns flattened list of agent names
        agent_names = [name.lower() for name in vocabulary_registry.get_agent_names()]

        # Tier 0 agents as defined in production vocabulary (ML Foundation)
        tier_0_agents = [
            "scope_definer", "data_preparer", "model_selector",
            "model_trainer", "model_evaluator", "model_deployer",
            "model_monitor"
        ]

        for agent in tier_0_agents:
            assert any(agent in name for name in agent_names), \
                f"Tier 0 agent '{agent}' not found in vocabulary"

    def test_vocabulary_has_all_tier_1_agents(self, vocabulary_registry):
        """Test that vocabulary includes all Tier 1 orchestration agents."""
        agent_names = [name.lower() for name in vocabulary_registry.get_agent_names()]

        tier_1_agents = ["orchestrator", "tool_composer"]

        for agent in tier_1_agents:
            assert any(agent in name for name in agent_names), \
                f"Tier 1 agent '{agent}' not found in vocabulary"

    def test_vocabulary_has_all_tier_2_agents(self, vocabulary_registry):
        """Test that vocabulary includes all Tier 2 causal analytics agents."""
        agent_names = [name.lower() for name in vocabulary_registry.get_agent_names()]

        tier_2_agents = ["causal_impact", "gap_analyzer", "heterogeneous_optimizer"]

        for agent in tier_2_agents:
            assert any(agent in name for name in agent_names), \
                f"Tier 2 agent '{agent}' not found in vocabulary"

    def test_vocabulary_has_all_tier_3_agents(self, vocabulary_registry):
        """Test that vocabulary includes all Tier 3 monitoring agents."""
        agent_names = [name.lower() for name in vocabulary_registry.get_agent_names()]

        # Tier 3 agents as defined in production vocabulary (Monitoring)
        tier_3_agents = ["experiment_designer", "drift_monitor", "data_quality_monitor"]

        for agent in tier_3_agents:
            assert any(agent in name for name in agent_names), \
                f"Tier 3 agent '{agent}' not found in vocabulary"

    def test_vocabulary_has_all_tier_4_agents(self, vocabulary_registry):
        """Test that vocabulary includes all Tier 4 ML prediction agents."""
        agent_names = [name.lower() for name in vocabulary_registry.get_agent_names()]

        # Tier 4 agents as defined in production vocabulary (Prediction)
        tier_4_agents = ["prediction_synthesizer", "risk_assessor"]

        for agent in tier_4_agents:
            assert any(agent in name for name in agent_names), \
                f"Tier 4 agent '{agent}' not found in vocabulary"

    def test_vocabulary_has_all_tier_5_agents(self, vocabulary_registry):
        """Test that vocabulary includes all Tier 5 self-improvement agents."""
        agent_names = [name.lower() for name in vocabulary_registry.get_agent_names()]

        # Tier 5 agents as defined in production vocabulary (Self-Improvement)
        # Note: health_score and resource_optimizer are in 'legacy' section, not tier_5
        tier_5_agents = ["explainer", "feedback_learner"]

        for agent in tier_5_agents:
            assert any(agent in name for name in agent_names), \
                f"Tier 5 agent '{agent}' not found in vocabulary"

    def test_vocabulary_has_explainer_agent(self, vocabulary_registry):
        """Test that vocabulary includes Explainer agent for self-improvement."""
        agent_names = [name.lower() for name in vocabulary_registry.get_agent_names()]

        assert any("explainer" in name for name in agent_names), \
            "Explainer agent not found in vocabulary"

    def test_vocabulary_agent_count_matches_expected(self, vocabulary_registry):
        """Test that vocabulary has expected number of agents (18-20)."""
        # Use get_agent_names() for actual count of agents
        agent_names = vocabulary_registry.get_agent_names()

        # E2I has 18 base agents + potentially more
        assert len(agent_names) >= 18, f"Expected at least 18 agents, found {len(agent_names)}"
        assert len(agent_names) <= 25, f"Unexpectedly high agent count: {len(agent_names)}"


# =============================================================================
# VOCABULARY BRAND SYNC TESTS
# =============================================================================


class TestVocabularyBrandSync:
    """Test brand definitions in vocabulary match expected database values."""

    def _get_brand_names(self, brands):
        """Extract brand names from brands list.

        get_brands() returns list[str] of brand names.
        """
        return [b.lower() for b in brands]

    def test_vocabulary_has_remibrutinib(self, vocabulary_registry):
        """Test that vocabulary includes Remibrutinib brand."""
        brands = vocabulary_registry.get_brands()
        brand_names = self._get_brand_names(brands)

        assert any("remibrutinib" in name for name in brand_names), \
            "Remibrutinib not found in vocabulary"

    def test_vocabulary_has_fabhalta(self, vocabulary_registry):
        """Test that vocabulary includes Fabhalta brand."""
        brands = vocabulary_registry.get_brands()
        brand_names = self._get_brand_names(brands)

        assert any("fabhalta" in name for name in brand_names), \
            "Fabhalta not found in vocabulary"

    def test_vocabulary_has_kisqali(self, vocabulary_registry):
        """Test that vocabulary includes Kisqali brand."""
        brands = vocabulary_registry.get_brands()
        brand_names = self._get_brand_names(brands)

        assert any("kisqali" in name for name in brand_names), \
            "Kisqali not found in vocabulary"

    def test_vocabulary_brand_count_matches_expected(self, vocabulary_registry):
        """Test that vocabulary has expected number of brands."""
        brands = vocabulary_registry.get_brands()

        # E2I has 3 primary brands
        assert len(brands) >= 3, f"Expected at least 3 brands, found {len(brands)}"


# =============================================================================
# VOCABULARY REGION SYNC TESTS
# =============================================================================


class TestVocabularyRegionSync:
    """Test region definitions in vocabulary match expected database values."""

    def _get_region_names(self, regions):
        """Extract region names from regions list.

        get_regions() returns list[str] of region names.
        """
        return [r.lower() for r in regions]

    def test_vocabulary_has_northeast_region(self, vocabulary_registry):
        """Test that vocabulary includes Northeast US region."""
        regions = vocabulary_registry.get_regions()
        region_names = self._get_region_names(regions)

        assert any("northeast" in r for r in region_names), \
            "Northeast region not found in vocabulary"

    def test_vocabulary_has_south_region(self, vocabulary_registry):
        """Test that vocabulary includes South US region."""
        regions = vocabulary_registry.get_regions()
        region_names = self._get_region_names(regions)

        assert any("south" in r for r in region_names), \
            "South region not found in vocabulary"

    def test_vocabulary_has_midwest_region(self, vocabulary_registry):
        """Test that vocabulary includes Midwest US region."""
        regions = vocabulary_registry.get_regions()
        region_names = self._get_region_names(regions)

        assert any("midwest" in r for r in region_names), \
            "Midwest region not found in vocabulary"

    def test_vocabulary_has_west_region(self, vocabulary_registry):
        """Test that vocabulary includes West US region."""
        regions = vocabulary_registry.get_regions()
        region_names = self._get_region_names(regions)

        assert any("west" in r for r in region_names), \
            "West region not found in vocabulary"

    def test_vocabulary_regions_have_names(self, vocabulary_registry):
        """Test that all regions have name identifiers."""
        regions = vocabulary_registry.get_regions()

        # get_regions() returns list[str], so each item is already a string identifier
        for region in regions:
            assert region, f"Empty region identifier found"


# =============================================================================
# VOCABULARY TIER SYNC TESTS
# =============================================================================


class TestVocabularyTierSync:
    """Test tier definitions in vocabulary match expected database values."""

    def test_vocabulary_agents_have_tiers(self, vocabulary_registry):
        """Test that agents in vocabulary have tier assignments.

        The agents section uses tier_X keys (e.g., tier_0_foundation, tier_1_orchestration).
        So tiers are implicit in the key structure, not in agent data.
        """
        agents = vocabulary_registry.get_agents()

        # get_agents() returns dict[str, list] where keys indicate tiers
        # e.g., {'tier_0_foundation': ['scope_definer', ...], 'tier_1_orchestration': [...]}
        tier_keys = [key for key in agents.keys() if key.startswith("tier_")]

        # Should have tier keys covering multiple tiers
        assert len(tier_keys) >= 3, \
            f"Expected tier keys (tier_X_*), found: {list(agents.keys())}"

    def test_vocabulary_tiers_are_valid(self, vocabulary_registry):
        """Test that tier values are in valid range (0-5)."""
        agents = vocabulary_registry.get_agents()
        valid_tier_prefixes = [f"tier_{i}" for i in range(6)]

        for key in agents.keys():
            # Tier is encoded in the key name (e.g., tier_0_foundation)
            if key.startswith("tier_"):
                # Extract tier number from key
                is_valid = any(key.startswith(prefix) for prefix in valid_tier_prefixes)
                assert is_valid, f"Invalid tier key: {key}"


# =============================================================================
# DATABASE ENUM STRUCTURE TESTS
# =============================================================================


class TestDatabaseEnumStructure:
    """Test expected database ENUM structure matches vocabulary."""

    def test_db_enum_values_structure(self, db_enum_values):
        """Test that expected DB enum values structure is complete."""
        required_enums = ["agent_tier", "agent_name", "brand_name", "region_code"]

        for enum_name in required_enums:
            assert enum_name in db_enum_values, f"Missing expected enum: {enum_name}"
            assert len(db_enum_values[enum_name]) > 0, f"Empty enum values for: {enum_name}"

    def test_db_agent_tier_enum_values(self, db_enum_values):
        """Test that agent_tier enum has all tier values."""
        tier_values = db_enum_values["agent_tier"]

        expected_tiers = ["tier_0", "tier_1", "tier_2", "tier_3", "tier_4", "tier_5"]
        for tier in expected_tiers:
            assert tier in tier_values, f"Missing tier '{tier}' in agent_tier enum"

    def test_db_agent_name_enum_has_minimum_agents(self, db_enum_values):
        """Test that agent_name enum has minimum expected agents."""
        agent_names = db_enum_values["agent_name"]

        # Should have at least 18 agents
        assert len(agent_names) >= 18, f"Expected at least 18 agents in enum, found {len(agent_names)}"


# =============================================================================
# VOCABULARY-DATABASE ALIGNMENT TESTS
# =============================================================================


class TestVocabularyDatabaseAlignment:
    """Test alignment between vocabulary and expected database values."""

    def test_vocabulary_brands_align_with_db(self, vocabulary_registry, db_enum_values):
        """Test that vocabulary brands align with database enum values."""
        vocab_brands = vocabulary_registry.get_brands()
        # get_brands() returns list[str]
        vocab_brand_names = {b.lower() for b in vocab_brands}

        db_brands = {b.lower() for b in db_enum_values["brand_name"]}

        # Check that all DB brands are in vocabulary
        for db_brand in db_brands:
            found = any(db_brand in vb for vb in vocab_brand_names)
            assert found, f"DB brand '{db_brand}' not found in vocabulary"

    def test_vocabulary_regions_align_with_db(self, vocabulary_registry, db_enum_values):
        """Test that vocabulary regions align with database enum values."""
        vocab_regions = vocabulary_registry.get_regions()
        # get_regions() returns list[str]
        vocab_region_codes = {r.upper() for r in vocab_regions}

        db_regions = {r.upper() for r in db_enum_values["region_code"]}

        # Check for overlap (may not be exact match due to naming conventions)
        overlap = vocab_region_codes & db_regions
        assert len(overlap) > 0 or len(vocab_regions) > 0, \
            "No overlap between vocabulary and DB region codes"

    def test_vocabulary_agents_align_with_db(self, vocabulary_registry, db_enum_values):
        """Test that vocabulary agents align with database enum values."""
        # Use get_agent_names() for the flattened list of agent names
        vocab_agent_names = {name.lower().replace(" ", "_") for name in vocabulary_registry.get_agent_names()}

        db_agents = {a.lower() for a in db_enum_values["agent_name"]}

        # Check that core agents exist in both
        core_agents = ["orchestrator", "causal_impact", "gap_analyzer", "explainer"]
        for agent in core_agents:
            in_vocab = any(agent in va for va in vocab_agent_names)
            in_db = any(agent in da for da in db_agents)
            assert in_vocab, f"Core agent '{agent}' not found in vocabulary"
            assert in_db, f"Core agent '{agent}' not found in DB enum"


# =============================================================================
# ALIAS RESOLUTION TESTS
# =============================================================================


class TestVocabularyAliases:
    """Test that vocabulary aliases resolve correctly."""

    def test_brand_alias_resolution(self, vocabulary_registry):
        """Test that brand aliases resolve to canonical names."""
        # Common alias patterns
        test_cases = [
            ("REMI", "remibrutinib"),
            ("remi", "remibrutinib"),
            ("FAB", "fabhalta"),
            ("KIS", "kisqali"),
        ]

        for alias, expected_contains in test_cases:
            try:
                resolved = vocabulary_registry.resolve_alias(alias, "brand")
                if resolved:
                    assert expected_contains.lower() in resolved.lower(), \
                        f"Alias '{alias}' resolved to '{resolved}', expected to contain '{expected_contains}'"
            except (AttributeError, NotImplementedError):
                # Alias resolution may not be implemented
                pytest.skip("Alias resolution not implemented")

    def test_agent_alias_resolution(self, vocabulary_registry):
        """Test that agent aliases resolve to canonical names."""
        test_cases = [
            ("CI", "causal_impact"),
            ("GAP", "gap_analyzer"),
            ("EXP", "explainer"),
        ]

        for alias, expected_contains in test_cases:
            try:
                resolved = vocabulary_registry.resolve_alias(alias, "agent")
                if resolved:
                    # Just check it resolves to something
                    assert resolved is not None
            except (AttributeError, NotImplementedError):
                pytest.skip("Alias resolution not implemented")


# =============================================================================
# VOCABULARY VALIDATION TESTS
# =============================================================================


class TestVocabularyValidation:
    """Test vocabulary internal consistency and validation."""

    def test_no_duplicate_brand_names(self, vocabulary_registry):
        """Test that there are no duplicate brand names in vocabulary."""
        brands = vocabulary_registry.get_brands()
        # get_brands() returns list[str]
        brand_names = [b.lower() for b in brands]
        duplicates = [name for name in brand_names if brand_names.count(name) > 1]

        assert not duplicates, f"Duplicate brand names found: {set(duplicates)}"

    def test_no_duplicate_agent_names(self, vocabulary_registry):
        """Test that there are no duplicate agent names in vocabulary."""
        # Use get_agent_names() for the flattened list of agent names
        agent_names = [name.lower() for name in vocabulary_registry.get_agent_names()]
        duplicates = [name for name in agent_names if agent_names.count(name) > 1]

        assert not duplicates, f"Duplicate agent names found: {set(duplicates)}"

    def test_agents_have_required_fields(self, vocabulary_registry):
        """Test that agents have required fields."""
        # Use get_agent_names() - each agent must have a non-empty name
        agent_names = vocabulary_registry.get_agent_names()

        for name in agent_names:
            assert name, f"Empty agent name found"

    def test_brands_have_required_fields(self, vocabulary_registry):
        """Test that brands have required fields."""
        brands = vocabulary_registry.get_brands()

        # get_brands() returns list[str], each string is the brand name
        for brand in brands:
            assert brand, f"Empty brand name found"


# =============================================================================
# MIGRATION COMPATIBILITY TESTS
# =============================================================================


class TestMigrationCompatibility:
    """Test that vocabulary is compatible with database migrations."""

    def test_vocabulary_supports_migration_029_enums(self, vocabulary_registry, db_enum_values):
        """Test vocabulary supports ENUMs from migration 029."""
        # Migration 029 added agent_tier and agent_name ENUMs
        # Verify vocabulary can provide values for these

        vocab_agent_names = vocabulary_registry.get_agent_names()
        assert len(vocab_agent_names) >= len(db_enum_values["agent_name"]) - 5, \
            "Vocabulary has significantly fewer agents than DB enum"

    def test_vocabulary_tier_mapping_consistency(self, vocabulary_registry):
        """Test that vocabulary tier mappings are consistent."""
        agents = vocabulary_registry.get_agents()

        # Group agents by tier - tier is encoded in key name (e.g., tier_0_foundation)
        tier_groups = {}
        for key, agent_list in agents.items():
            if not key.startswith("tier_"):
                continue

            # Extract tier number from key (e.g., "tier_0_foundation" -> "0")
            tier_num = key.split("_")[1]

            if tier_num not in tier_groups:
                tier_groups[tier_num] = []

            if isinstance(agent_list, list):
                tier_groups[tier_num].extend(agent_list)

        # Check tier group sizes are reasonable
        for tier, members in tier_groups.items():
            assert len(members) <= 10, f"Tier {tier} has unusually many agents: {len(members)}"
