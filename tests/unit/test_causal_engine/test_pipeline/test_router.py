"""Unit tests for Library Router.

Tests the NLP-based question classification and library routing logic
per the Library Selection Decision Matrix.
"""

import pytest

from src.causal_engine.pipeline.router import (
    CausalLibrary,
    LibraryCapability,
    LibraryRouter,
    QuestionType,
    RoutingDecision,
)


class TestQuestionClassification:
    """Test question type classification."""

    @pytest.fixture
    def router(self) -> LibraryRouter:
        """Create a router instance."""
        return LibraryRouter()

    # === Causal Relationship Tests ===
    @pytest.mark.parametrize(
        "query",
        [
            "Does detailing cause prescription increases?",
            "Did the campaign affect sales?",
            "Is there a causal relationship between rep visits and TRx?",
            "What causes market share decline?",
            "Are speaker programs influencing prescriber behavior?",
            "Does increased frequency impact conversion rates?",
        ],
    )
    def test_causal_relationship_classification(self, router: LibraryRouter, query: str):
        """Test that causal relationship questions are classified correctly."""
        question_type, confidence, rationale = router.classify_question(query)
        assert question_type == QuestionType.CAUSAL_RELATIONSHIP
        assert confidence > 0.0
        assert rationale

    # === Effect Heterogeneity Tests ===
    @pytest.mark.parametrize(
        "query",
        [
            "How much does the treatment effect vary by region?",
            "Are there heterogeneous effects across specialties?",
            "How do treatment effects differ between segments?",
            "What is the CATE for oncologists vs cardiologists?",
            "Show segment-level treatment effects",
            "How does the impact change across deciles?",
        ],
    )
    def test_effect_heterogeneity_classification(self, router: LibraryRouter, query: str):
        """Test that effect heterogeneity questions are classified correctly."""
        question_type, confidence, rationale = router.classify_question(query)
        assert question_type == QuestionType.EFFECT_HETEROGENEITY
        assert confidence > 0.0

    # === Targeting Optimization Tests ===
    @pytest.mark.parametrize(
        "query",
        [
            "Who should we target for the next campaign?",
            "Which HCPs should we prioritize?",
            "Calculate the uplift score for each customer",
            "Optimize targeting for marketing spend",
            "Find the high responders for Kisqali",
            "Who would benefit most from increased detailing?",
        ],
    )
    def test_targeting_optimization_classification(self, router: LibraryRouter, query: str):
        """Test that targeting questions are classified correctly."""
        question_type, confidence, rationale = router.classify_question(query)
        assert question_type == QuestionType.TARGETING_OPTIMIZATION
        assert confidence > 0.0

    # === Impact Flow Tests ===
    @pytest.mark.parametrize(
        "query",
        [
            "How does the impact flow through the system?",
            "What is the path from investment to revenue?",
            "Show the causal graph for this process",
            "Where are the bottlenecks in the conversion funnel?",
            "Analyze the network effects of peer influence",
            "What are the multiplier effects of digital engagement?",
        ],
    )
    def test_impact_flow_classification(self, router: LibraryRouter, query: str):
        """Test that impact flow questions are classified correctly."""
        question_type, confidence, rationale = router.classify_question(query)
        assert question_type == QuestionType.IMPACT_FLOW
        assert confidence > 0.0

    # === Edge Cases ===
    def test_empty_query(self, router: LibraryRouter):
        """Test that empty queries return UNKNOWN."""
        question_type, confidence, rationale = router.classify_question("")
        assert question_type == QuestionType.UNKNOWN
        assert confidence == 0.0

    def test_whitespace_query(self, router: LibraryRouter):
        """Test that whitespace-only queries return UNKNOWN."""
        question_type, confidence, rationale = router.classify_question("   \t\n  ")
        assert question_type == QuestionType.UNKNOWN
        assert confidence == 0.0

    def test_unrelated_query(self, router: LibraryRouter):
        """Test that unrelated queries use fallback classification."""
        question_type, confidence, rationale = router.classify_question(
            "What is the weather today?"
        )
        # Should return UNKNOWN or low-confidence classification
        assert question_type in [QuestionType.UNKNOWN, QuestionType.CAUSAL_RELATIONSHIP]
        assert confidence <= 0.5


class TestLibraryRouting:
    """Test library routing decisions."""

    @pytest.fixture
    def router(self) -> LibraryRouter:
        """Create a router instance."""
        return LibraryRouter()

    def test_causal_relationship_routes_to_dowhy(self, router: LibraryRouter):
        """Test that causal relationship questions route to DoWhy."""
        decision = router.route("Does detailing cause prescription increases?")
        assert decision.primary_library == CausalLibrary.DOWHY
        assert decision.question_type == QuestionType.CAUSAL_RELATIONSHIP
        assert CausalLibrary.NETWORKX in decision.secondary_libraries

    def test_effect_heterogeneity_routes_to_econml(self, router: LibraryRouter):
        """Test that effect heterogeneity questions route to EconML."""
        decision = router.route("How much does the treatment effect vary by region?")
        assert decision.primary_library == CausalLibrary.ECONML
        assert decision.question_type == QuestionType.EFFECT_HETEROGENEITY
        assert CausalLibrary.CAUSALML in decision.secondary_libraries

    def test_targeting_routes_to_causalml(self, router: LibraryRouter):
        """Test that targeting questions route to CausalML."""
        decision = router.route("Who should we target for the next campaign?")
        assert decision.primary_library == CausalLibrary.CAUSALML
        assert decision.question_type == QuestionType.TARGETING_OPTIMIZATION
        assert CausalLibrary.ECONML in decision.secondary_libraries

    def test_impact_flow_routes_to_networkx(self, router: LibraryRouter):
        """Test that impact flow questions route to NetworkX."""
        decision = router.route("How does the impact flow through the system?")
        assert decision.primary_library == CausalLibrary.NETWORKX
        assert decision.question_type == QuestionType.IMPACT_FLOW
        assert CausalLibrary.DOWHY in decision.secondary_libraries


class TestSynergyPatterns:
    """Test multi-library synergy patterns."""

    @pytest.fixture
    def router(self) -> LibraryRouter:
        """Create a router instance."""
        return LibraryRouter()

    def test_end_to_end_synergy(self, router: LibraryRouter):
        """Test end-to-end synergy pattern."""
        decision = router.route("Any query", requested_synergy="end_to_end")
        assert decision.primary_library == CausalLibrary.NETWORKX
        assert len(decision.secondary_libraries) == 3
        assert CausalLibrary.DOWHY in decision.secondary_libraries
        assert CausalLibrary.ECONML in decision.secondary_libraries
        assert CausalLibrary.CAUSALML in decision.secondary_libraries
        assert decision.recommended_mode == "sequential"

    def test_fairness_aware_synergy(self, router: LibraryRouter):
        """Test fairness-aware targeting synergy pattern."""
        decision = router.route("Any query", requested_synergy="fairness_aware")
        assert decision.primary_library == CausalLibrary.ECONML
        assert CausalLibrary.CAUSALML in decision.secondary_libraries
        assert decision.recommended_mode == "parallel"

    def test_validated_experiments_synergy(self, router: LibraryRouter):
        """Test validated experiments synergy pattern."""
        decision = router.route("Any query", requested_synergy="validated_experiments")
        assert decision.primary_library == CausalLibrary.DOWHY
        assert CausalLibrary.CAUSALML in decision.secondary_libraries
        assert decision.recommended_mode == "validation_loop"

    def test_bottleneck_analysis_synergy(self, router: LibraryRouter):
        """Test bottleneck analysis synergy pattern."""
        decision = router.route("Any query", requested_synergy="bottleneck_analysis")
        assert decision.primary_library == CausalLibrary.NETWORKX
        assert CausalLibrary.DOWHY in decision.secondary_libraries
        assert decision.recommended_mode == "sequential"


class TestForcedLibraries:
    """Test forced library selection."""

    @pytest.fixture
    def router(self) -> LibraryRouter:
        """Create a router instance."""
        return LibraryRouter()

    def test_force_single_library(self, router: LibraryRouter):
        """Test forcing a single library."""
        decision = router.route("Any query", force_libraries=["econml"])
        assert decision.primary_library == CausalLibrary.ECONML
        assert len(decision.secondary_libraries) == 0
        assert decision.confidence == 1.0

    def test_force_multiple_libraries(self, router: LibraryRouter):
        """Test forcing multiple libraries."""
        decision = router.route("Any query", force_libraries=["dowhy", "causalml"])
        assert decision.primary_library == CausalLibrary.DOWHY
        assert CausalLibrary.CAUSALML in decision.secondary_libraries
        assert decision.recommended_mode == "parallel"

    def test_force_all_libraries(self, router: LibraryRouter):
        """Test forcing all libraries."""
        decision = router.route(
            "Any query",
            force_libraries=["networkx", "dowhy", "econml", "causalml"],
        )
        assert decision.primary_library == CausalLibrary.NETWORKX
        assert len(decision.secondary_libraries) == 3


class TestLibraryCapabilities:
    """Test library capability queries."""

    @pytest.fixture
    def router(self) -> LibraryRouter:
        """Create a router instance."""
        return LibraryRouter()

    def test_get_library_for_cate_capability(self, router: LibraryRouter):
        """Test finding library for CATE estimation."""
        library = router.get_library_for_capability(LibraryCapability.CATE_ESTIMATION)
        assert library == CausalLibrary.ECONML

    def test_get_library_for_uplift_capability(self, router: LibraryRouter):
        """Test finding library for uplift modeling."""
        library = router.get_library_for_capability(LibraryCapability.UPLIFT_MODELING)
        assert library == CausalLibrary.CAUSALML

    def test_get_library_for_graph_capability(self, router: LibraryRouter):
        """Test finding library for graph construction."""
        library = router.get_library_for_capability(LibraryCapability.GRAPH_CONSTRUCTION)
        assert library == CausalLibrary.NETWORKX

    def test_get_library_for_refutation_capability(self, router: LibraryRouter):
        """Test finding library for refutation testing."""
        library = router.get_library_for_capability(LibraryCapability.REFUTATION_TESTING)
        assert library == CausalLibrary.DOWHY


class TestRoutingDecision:
    """Test RoutingDecision dataclass."""

    def test_routing_decision_defaults(self):
        """Test RoutingDecision default values."""
        decision = RoutingDecision(
            question_type=QuestionType.CAUSAL_RELATIONSHIP,
            primary_library=CausalLibrary.DOWHY,
        )
        assert decision.secondary_libraries == []
        assert decision.confidence == 0.0
        assert decision.rationale == ""
        assert decision.required_capabilities == []
        assert decision.recommended_mode == "sequential"

    def test_routing_decision_with_all_fields(self):
        """Test RoutingDecision with all fields populated."""
        decision = RoutingDecision(
            question_type=QuestionType.TARGETING_OPTIMIZATION,
            primary_library=CausalLibrary.CAUSALML,
            secondary_libraries=[CausalLibrary.ECONML],
            confidence=0.95,
            rationale="Matched targeting pattern",
            required_capabilities=[LibraryCapability.UPLIFT_MODELING],
            recommended_mode="parallel",
        )
        assert decision.question_type == QuestionType.TARGETING_OPTIMIZATION
        assert decision.primary_library == CausalLibrary.CAUSALML
        assert len(decision.secondary_libraries) == 1
        assert decision.confidence == 0.95


class TestSynergyPatternRetrieval:
    """Test synergy pattern retrieval."""

    @pytest.fixture
    def router(self) -> LibraryRouter:
        """Create a router instance."""
        return LibraryRouter()

    def test_get_existing_synergy_pattern(self, router: LibraryRouter):
        """Test getting an existing synergy pattern."""
        pattern = router.get_synergy_pattern("end_to_end")
        assert pattern is not None
        assert len(pattern) == 4
        assert CausalLibrary.NETWORKX in pattern
        assert CausalLibrary.DOWHY in pattern
        assert CausalLibrary.ECONML in pattern
        assert CausalLibrary.CAUSALML in pattern

    def test_get_nonexistent_synergy_pattern(self, router: LibraryRouter):
        """Test getting a non-existent synergy pattern."""
        pattern = router.get_synergy_pattern("nonexistent_pattern")
        assert pattern is None


class TestQuestionTypeEnum:
    """Test QuestionType enum values."""

    def test_question_type_values(self):
        """Test that all question types have expected values."""
        assert QuestionType.CAUSAL_RELATIONSHIP.value == "causal_relationship"
        assert QuestionType.EFFECT_HETEROGENEITY.value == "effect_heterogeneity"
        assert QuestionType.TARGETING_OPTIMIZATION.value == "targeting_optimization"
        assert QuestionType.IMPACT_FLOW.value == "impact_flow"
        assert QuestionType.COMPREHENSIVE.value == "comprehensive"
        assert QuestionType.UNKNOWN.value == "unknown"


class TestCausalLibraryEnum:
    """Test CausalLibrary enum values."""

    def test_causal_library_values(self):
        """Test that all libraries have expected values."""
        assert CausalLibrary.NETWORKX.value == "networkx"
        assert CausalLibrary.DOWHY.value == "dowhy"
        assert CausalLibrary.ECONML.value == "econml"
        assert CausalLibrary.CAUSALML.value == "causalml"
