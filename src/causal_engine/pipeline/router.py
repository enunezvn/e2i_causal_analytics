"""Library Router for multi-library causal analysis.

This module provides question-type classification and routing logic to
determine which causal inference library(s) should handle a given query.

Library Selection Decision Matrix (from docs/E2I Causal Analytics KPI Framework.html):
- "Does X cause Y?" → DoWhy (causal validation)
- "How much does effect vary?" → EconML (heterogeneous effects)
- "Who should we target?" → CausalML (uplift modeling)
- "How does impact flow?" → NetworkX (graph analysis)
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import yaml  # type: ignore[import-untyped]


class QuestionType(str, Enum):
    """Classified question types for routing."""

    CAUSAL_RELATIONSHIP = "causal_relationship"  # "Does X cause Y?"
    EFFECT_HETEROGENEITY = "effect_heterogeneity"  # "How much does effect vary?"
    TARGETING_OPTIMIZATION = "targeting_optimization"  # "Who should we target?"
    IMPACT_FLOW = "impact_flow"  # "How does impact flow?"
    COMPREHENSIVE = "comprehensive"  # Complex query requiring multiple libraries
    UNKNOWN = "unknown"  # Cannot classify


class CausalLibrary(str, Enum):
    """Available causal inference libraries."""

    NETWORKX = "networkx"
    DOWHY = "dowhy"
    ECONML = "econml"
    CAUSALML = "causalml"


class LibraryCapability(str, Enum):
    """Capabilities of each library."""

    # NetworkX
    GRAPH_CONSTRUCTION = "graph_construction"
    PATH_ANALYSIS = "path_analysis"
    CENTRALITY_METRICS = "centrality_metrics"
    BOTTLENECK_DETECTION = "bottleneck_detection"

    # DoWhy
    CAUSAL_IDENTIFICATION = "causal_identification"
    EFFECT_ESTIMATION = "effect_estimation"
    REFUTATION_TESTING = "refutation_testing"
    SENSITIVITY_ANALYSIS = "sensitivity_analysis"

    # EconML
    CATE_ESTIMATION = "cate_estimation"
    HETEROGENEITY_ANALYSIS = "heterogeneity_analysis"
    POLICY_LEARNING = "policy_learning"
    FEATURE_IMPORTANCE = "feature_importance"

    # CausalML
    UPLIFT_MODELING = "uplift_modeling"
    TARGETING_OPTIMIZATION = "targeting_optimization"
    AUUC_CALCULATION = "auuc_calculation"
    SEGMENT_RANKING = "segment_ranking"


# Library capability mapping
LIBRARY_CAPABILITIES: Dict[CausalLibrary, List[LibraryCapability]] = {
    CausalLibrary.NETWORKX: [
        LibraryCapability.GRAPH_CONSTRUCTION,
        LibraryCapability.PATH_ANALYSIS,
        LibraryCapability.CENTRALITY_METRICS,
        LibraryCapability.BOTTLENECK_DETECTION,
    ],
    CausalLibrary.DOWHY: [
        LibraryCapability.CAUSAL_IDENTIFICATION,
        LibraryCapability.EFFECT_ESTIMATION,
        LibraryCapability.REFUTATION_TESTING,
        LibraryCapability.SENSITIVITY_ANALYSIS,
    ],
    CausalLibrary.ECONML: [
        LibraryCapability.CATE_ESTIMATION,
        LibraryCapability.HETEROGENEITY_ANALYSIS,
        LibraryCapability.POLICY_LEARNING,
        LibraryCapability.FEATURE_IMPORTANCE,
    ],
    CausalLibrary.CAUSALML: [
        LibraryCapability.UPLIFT_MODELING,
        LibraryCapability.TARGETING_OPTIMIZATION,
        LibraryCapability.AUUC_CALCULATION,
        LibraryCapability.SEGMENT_RANKING,
    ],
}


@dataclass
class RoutingDecision:
    """Result of the routing decision."""

    question_type: QuestionType
    primary_library: CausalLibrary
    secondary_libraries: List[CausalLibrary] = field(default_factory=list)
    confidence: float = 0.0  # 0.0-1.0
    rationale: str = ""
    required_capabilities: List[LibraryCapability] = field(default_factory=list)
    recommended_mode: str = "sequential"  # sequential, parallel, validation_loop


@dataclass
class QuestionPattern:
    """Pattern for question classification."""

    pattern: re.Pattern
    question_type: QuestionType
    weight: float = 1.0
    keywords: List[str] = field(default_factory=list)


class LibraryRouter:
    """Routes causal queries to appropriate library(s) based on question type.

    The router uses NLP-based pattern matching to classify questions and
    determine the optimal library or combination of libraries.

    Multi-Library Synergy Patterns:
    1. End-to-End Optimization: NetworkX → DoWhy → EconML → CausalML
    2. Fairness-Aware Targeting: EconML + CausalML
    3. Validated Experiments: DoWhy + CausalML
    4. System Bottleneck Analysis: NetworkX + DoWhy
    """

    # Question patterns for classification
    QUESTION_PATTERNS: List[QuestionPattern] = [
        # Causal Relationship patterns ("Does X cause Y?")
        QuestionPattern(
            pattern=re.compile(
                r"(does|do|did|is|are|was|were)\s+.+\s+(cause|causes|causing|caused|affect|affects|affecting|affected|impact|impacts|impacting|impacted|influence|influences|influencing|influenced)\s+",
                re.IGNORECASE,
            ),
            question_type=QuestionType.CAUSAL_RELATIONSHIP,
            weight=1.0,
            keywords=["cause", "causal", "effect", "impact", "influence", "relationship"],
        ),
        QuestionPattern(
            pattern=re.compile(
                r"(what|which)\s+.+\s+(causes|affects|impacts|influences)\s+",
                re.IGNORECASE,
            ),
            question_type=QuestionType.CAUSAL_RELATIONSHIP,
            weight=0.9,
            keywords=["what causes", "which affects"],
        ),
        QuestionPattern(
            pattern=re.compile(
                r"(causal|causally|causation)\s+(relationship|link|connection|path)",
                re.IGNORECASE,
            ),
            question_type=QuestionType.CAUSAL_RELATIONSHIP,
            weight=1.0,
            keywords=["causal relationship", "causal link"],
        ),
        # Effect Heterogeneity patterns ("How much does effect vary?")
        QuestionPattern(
            pattern=re.compile(
                r"(how\s+much|how\s+does|how\s+do)\s+.+\s+(vary|varies|varying|differ|differs|differing|change|changes|changing)\s+",
                re.IGNORECASE,
            ),
            question_type=QuestionType.EFFECT_HETEROGENEITY,
            weight=1.0,
            keywords=["vary", "differ", "heterogeneous", "segment"],
        ),
        QuestionPattern(
            pattern=re.compile(
                r"(heterogeneous|heterogeneity|different|differential)\s+(effect|effects|treatment|impact)",
                re.IGNORECASE,
            ),
            question_type=QuestionType.EFFECT_HETEROGENEITY,
            weight=1.0,
            keywords=["heterogeneous effect", "differential effect"],
        ),
        QuestionPattern(
            pattern=re.compile(
                r"(segment|subgroup|group)\s+(level|specific|analysis|effect)",
                re.IGNORECASE,
            ),
            question_type=QuestionType.EFFECT_HETEROGENEITY,
            weight=0.9,
            keywords=["segment level", "subgroup analysis"],
        ),
        QuestionPattern(
            pattern=re.compile(
                r"(cate|conditional\s+average\s+treatment\s+effect)",
                re.IGNORECASE,
            ),
            question_type=QuestionType.EFFECT_HETEROGENEITY,
            weight=1.0,
            keywords=["CATE", "conditional average treatment effect"],
        ),
        # Targeting Optimization patterns ("Who should we target?")
        QuestionPattern(
            pattern=re.compile(
                r"(who|which|what)\s+.+\s+(should|could|would)\s+.*(target|prioritize|focus|engage)",
                re.IGNORECASE,
            ),
            question_type=QuestionType.TARGETING_OPTIMIZATION,
            weight=1.0,
            keywords=["target", "prioritize", "focus", "engage"],
        ),
        QuestionPattern(
            pattern=re.compile(
                r"(uplift|lift|incremental)\s+(model|modeling|score|analysis)",
                re.IGNORECASE,
            ),
            question_type=QuestionType.TARGETING_OPTIMIZATION,
            weight=1.0,
            keywords=["uplift", "lift", "incremental"],
        ),
        QuestionPattern(
            pattern=re.compile(
                r"(optimize|optimization|optimal)\s+(targeting|allocation|marketing|engagement)",
                re.IGNORECASE,
            ),
            question_type=QuestionType.TARGETING_OPTIMIZATION,
            weight=1.0,
            keywords=["optimize targeting", "optimal allocation"],
        ),
        QuestionPattern(
            pattern=re.compile(
                r"(high|low|best|worst)\s+(responder|responders|performer|performers)",
                re.IGNORECASE,
            ),
            question_type=QuestionType.TARGETING_OPTIMIZATION,
            weight=0.9,
            keywords=["high responder", "low responder"],
        ),
        # Effect Heterogeneity - additional patterns
        QuestionPattern(
            pattern=re.compile(
                r"(show|display|present)\s+.*(segment|subgroup|group).*(effect|impact|treatment)",
                re.IGNORECASE,
            ),
            question_type=QuestionType.EFFECT_HETEROGENEITY,
            weight=1.2,  # Higher weight for specific pattern
            keywords=["segment effect", "group effect"],
        ),
        QuestionPattern(
            pattern=re.compile(
                r"(across|by|per)\s+(decile|quartile|segment|region|specialty)",
                re.IGNORECASE,
            ),
            question_type=QuestionType.EFFECT_HETEROGENEITY,
            weight=1.1,
            keywords=["across deciles", "by segment"],
        ),
        # Impact Flow patterns ("How does impact flow?")
        QuestionPattern(
            pattern=re.compile(
                r"(how\s+does|how\s+do)\s+.*(flow|propagate|spread|cascade)",
                re.IGNORECASE,
            ),
            question_type=QuestionType.IMPACT_FLOW,
            weight=1.5,  # Higher weight to avoid confusion with causal patterns
            keywords=["flow", "propagate", "spread", "cascade"],
        ),
        QuestionPattern(
            pattern=re.compile(
                r"(path|pathway|chain|network|graph)\s+(analysis|effect|flow|structure)",
                re.IGNORECASE,
            ),
            question_type=QuestionType.IMPACT_FLOW,
            weight=1.0,
            keywords=["path analysis", "network effect"],
        ),
        QuestionPattern(
            pattern=re.compile(
                r"(bottleneck|multiplier|mediator|mediating|indirect)\s+(effect|analysis|impact)",
                re.IGNORECASE,
            ),
            question_type=QuestionType.IMPACT_FLOW,
            weight=1.0,
            keywords=["bottleneck", "multiplier effect", "mediator"],
        ),
        QuestionPattern(
            pattern=re.compile(
                r"(dag|directed\s+acyclic\s+graph|causal\s+graph|causal\s+diagram)",
                re.IGNORECASE,
            ),
            question_type=QuestionType.IMPACT_FLOW,
            weight=1.0,
            keywords=["DAG", "causal graph"],
        ),
    ]

    # Question type to primary library mapping
    QUESTION_TO_LIBRARY: Dict[QuestionType, CausalLibrary] = {
        QuestionType.CAUSAL_RELATIONSHIP: CausalLibrary.DOWHY,
        QuestionType.EFFECT_HETEROGENEITY: CausalLibrary.ECONML,
        QuestionType.TARGETING_OPTIMIZATION: CausalLibrary.CAUSALML,
        QuestionType.IMPACT_FLOW: CausalLibrary.NETWORKX,
        QuestionType.COMPREHENSIVE: CausalLibrary.DOWHY,  # Default for comprehensive
        QuestionType.UNKNOWN: CausalLibrary.DOWHY,  # Default fallback
    }

    # Multi-library synergy patterns
    SYNERGY_PATTERNS: Dict[str, List[CausalLibrary]] = {
        "end_to_end": [
            CausalLibrary.NETWORKX,
            CausalLibrary.DOWHY,
            CausalLibrary.ECONML,
            CausalLibrary.CAUSALML,
        ],
        "fairness_aware": [CausalLibrary.ECONML, CausalLibrary.CAUSALML],
        "validated_experiments": [CausalLibrary.DOWHY, CausalLibrary.CAUSALML],
        "bottleneck_analysis": [CausalLibrary.NETWORKX, CausalLibrary.DOWHY],
    }

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the library router.

        Args:
            config_path: Optional path to routing configuration YAML
        """
        self.config: Dict[str, Any] = {}
        if config_path:
            self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        """Load routing configuration from YAML file."""
        try:
            with open(config_path) as f:
                self.config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            self.config = {}

    def classify_question(self, query: str) -> Tuple[QuestionType, float, str]:
        """Classify a natural language query into a question type.

        Args:
            query: Natural language causal query

        Returns:
            Tuple of (question_type, confidence, rationale)
        """
        if not query or not query.strip():
            return QuestionType.UNKNOWN, 0.0, "Empty query"

        query_lower = query.lower().strip()

        # Score each question type
        type_scores: Dict[QuestionType, float] = dict.fromkeys(QuestionType, 0.0)
        matched_patterns: List[str] = []

        for pattern_obj in self.QUESTION_PATTERNS:
            if pattern_obj.pattern.search(query_lower):
                type_scores[pattern_obj.question_type] += pattern_obj.weight
                matched_patterns.append(pattern_obj.question_type.value)

            # Also check keywords
            for keyword in pattern_obj.keywords:
                if keyword.lower() in query_lower:
                    type_scores[pattern_obj.question_type] += 0.3

        # Find the highest scoring type
        max_score = max(type_scores.values())

        if max_score == 0:
            # No patterns matched - use keyword heuristics
            return self._fallback_classification(query_lower)

        # Check for comprehensive (multiple high scores)
        # Only classify as comprehensive if multiple types have very similar high scores (>= 95%)
        high_score_types = [qt for qt, score in type_scores.items() if score >= max_score * 0.95]
        if len(high_score_types) > 1 and max_score >= 1.5:
            # Multiple types have similar high scores - comprehensive query
            confidence = min(1.0, max_score / 3.0)  # Normalize confidence
            rationale = f"Multiple patterns matched: {', '.join(matched_patterns)}"
            return QuestionType.COMPREHENSIVE, confidence, rationale

        # Single dominant type
        best_type = max(type_scores.items(), key=lambda x: x[1])[0]
        confidence = min(1.0, max_score / 2.0)  # Normalize confidence
        rationale = f"Matched pattern(s): {', '.join(set(matched_patterns))}"

        return best_type, confidence, rationale

    def _fallback_classification(self, query_lower: str) -> Tuple[QuestionType, float, str]:
        """Fallback classification using simple keyword matching."""
        # Simple keyword-based fallback
        causal_keywords = ["cause", "causal", "effect", "confound", "treatment"]
        hetero_keywords = ["vary", "heterogen", "segment", "differ", "cate"]
        targeting_keywords = ["target", "uplift", "who", "prioritize", "optimize"]
        flow_keywords = ["flow", "path", "network", "graph", "dag", "propagate"]

        scores = {
            QuestionType.CAUSAL_RELATIONSHIP: sum(1 for k in causal_keywords if k in query_lower),
            QuestionType.EFFECT_HETEROGENEITY: sum(1 for k in hetero_keywords if k in query_lower),
            QuestionType.TARGETING_OPTIMIZATION: sum(
                1 for k in targeting_keywords if k in query_lower
            ),
            QuestionType.IMPACT_FLOW: sum(1 for k in flow_keywords if k in query_lower),
        }

        max_score = max(scores.values())
        if max_score == 0:
            return QuestionType.UNKNOWN, 0.0, "No keywords matched"

        best_type = max(scores.items(), key=lambda x: x[1])[0]
        confidence = min(0.5, max_score / 4.0)  # Lower confidence for fallback
        return best_type, confidence, f"Fallback keyword match for {best_type.value}"

    def route(
        self,
        query: str,
        requested_synergy: Optional[str] = None,
        force_libraries: Optional[List[str]] = None,
    ) -> RoutingDecision:
        """Route a query to the appropriate library(s).

        Args:
            query: Natural language causal query
            requested_synergy: Optional synergy pattern to use
                ("end_to_end", "fairness_aware", "validated_experiments", "bottleneck_analysis")
            force_libraries: Optional list of libraries to force use

        Returns:
            RoutingDecision with library recommendations
        """
        # Handle forced libraries
        if force_libraries:
            libraries = [
                CausalLibrary(lib)
                for lib in force_libraries
                if lib in [e.value for e in CausalLibrary]
            ]
            if libraries:
                return RoutingDecision(
                    question_type=QuestionType.COMPREHENSIVE,
                    primary_library=libraries[0],
                    secondary_libraries=libraries[1:] if len(libraries) > 1 else [],
                    confidence=1.0,
                    rationale=f"Forced libraries: {', '.join(force_libraries)}",
                    recommended_mode="parallel" if len(libraries) > 1 else "sequential",
                )

        # Handle requested synergy pattern
        if requested_synergy and requested_synergy in self.SYNERGY_PATTERNS:
            libraries = self.SYNERGY_PATTERNS[requested_synergy]
            return RoutingDecision(
                question_type=QuestionType.COMPREHENSIVE,
                primary_library=libraries[0],
                secondary_libraries=libraries[1:],
                confidence=0.95,
                rationale=f"Using synergy pattern: {requested_synergy}",
                recommended_mode=self._get_mode_for_synergy(requested_synergy),
            )

        # Classify the question
        question_type, confidence, rationale = self.classify_question(query)

        # Get primary library
        primary_library = self.QUESTION_TO_LIBRARY.get(question_type, CausalLibrary.DOWHY)

        # Determine secondary libraries based on question type
        secondary_libraries = self._get_secondary_libraries(question_type, primary_library)

        # Determine required capabilities
        required_capabilities = self._get_required_capabilities(question_type)

        # Determine recommended mode
        recommended_mode = self._get_recommended_mode(question_type, secondary_libraries)

        return RoutingDecision(
            question_type=question_type,
            primary_library=primary_library,
            secondary_libraries=secondary_libraries,
            confidence=confidence,
            rationale=rationale,
            required_capabilities=required_capabilities,
            recommended_mode=recommended_mode,
        )

    def _get_secondary_libraries(
        self, question_type: QuestionType, primary: CausalLibrary
    ) -> List[CausalLibrary]:
        """Get secondary libraries based on question type."""
        if question_type == QuestionType.CAUSAL_RELATIONSHIP:
            # DoWhy primary → NetworkX for graph structure
            return [CausalLibrary.NETWORKX]

        elif question_type == QuestionType.EFFECT_HETEROGENEITY:
            # EconML primary → CausalML for uplift comparison
            return [CausalLibrary.CAUSALML]

        elif question_type == QuestionType.TARGETING_OPTIMIZATION:
            # CausalML primary → EconML for CATE validation
            return [CausalLibrary.ECONML]

        elif question_type == QuestionType.IMPACT_FLOW:
            # NetworkX primary → DoWhy for causal validation
            return [CausalLibrary.DOWHY]

        elif question_type == QuestionType.COMPREHENSIVE:
            # All libraries except primary
            return [lib for lib in CausalLibrary if lib != primary]

        return []

    def _get_required_capabilities(self, question_type: QuestionType) -> List[LibraryCapability]:
        """Get required capabilities for a question type."""
        capability_map = {
            QuestionType.CAUSAL_RELATIONSHIP: [
                LibraryCapability.CAUSAL_IDENTIFICATION,
                LibraryCapability.EFFECT_ESTIMATION,
                LibraryCapability.REFUTATION_TESTING,
            ],
            QuestionType.EFFECT_HETEROGENEITY: [
                LibraryCapability.CATE_ESTIMATION,
                LibraryCapability.HETEROGENEITY_ANALYSIS,
                LibraryCapability.FEATURE_IMPORTANCE,
            ],
            QuestionType.TARGETING_OPTIMIZATION: [
                LibraryCapability.UPLIFT_MODELING,
                LibraryCapability.TARGETING_OPTIMIZATION,
                LibraryCapability.SEGMENT_RANKING,
            ],
            QuestionType.IMPACT_FLOW: [
                LibraryCapability.GRAPH_CONSTRUCTION,
                LibraryCapability.PATH_ANALYSIS,
                LibraryCapability.CENTRALITY_METRICS,
            ],
        }
        return capability_map.get(question_type, [])

    def _get_recommended_mode(
        self, question_type: QuestionType, secondary_libraries: List[CausalLibrary]
    ) -> str:
        """Get recommended execution mode."""
        if not secondary_libraries:
            return "sequential"

        if question_type == QuestionType.COMPREHENSIVE:
            return "parallel"

        # For validation scenarios, use validation_loop
        if question_type in [QuestionType.CAUSAL_RELATIONSHIP, QuestionType.TARGETING_OPTIMIZATION]:
            return "validation_loop"

        return "sequential"

    def _get_mode_for_synergy(self, synergy: str) -> str:
        """Get execution mode for a synergy pattern."""
        mode_map = {
            "end_to_end": "sequential",
            "fairness_aware": "parallel",
            "validated_experiments": "validation_loop",
            "bottleneck_analysis": "sequential",
        }
        return mode_map.get(synergy, "sequential")

    def get_library_for_capability(self, capability: LibraryCapability) -> Optional[CausalLibrary]:
        """Get the library that provides a specific capability.

        Args:
            capability: The required capability

        Returns:
            The library that provides the capability, or None
        """
        for library, capabilities in LIBRARY_CAPABILITIES.items():
            if capability in capabilities:
                return library
        return None

    def get_synergy_pattern(self, pattern_name: str) -> Optional[List[CausalLibrary]]:
        """Get the libraries for a synergy pattern.

        Args:
            pattern_name: Name of the synergy pattern

        Returns:
            List of libraries for the pattern, or None if not found
        """
        return self.SYNERGY_PATTERNS.get(pattern_name)
