"""Intent classification node for orchestrator agent.

Fast intent classification optimized for <500ms:
- Pattern matching first (fastest)
- LLM fallback for ambiguous cases (Haiku)

Contract (issue #266 — invariant, enforced at import time):
    Any addition to ``IntentClassifierNode.INTENT_PATTERNS`` MUST also be
    ranked in ``INTENT_PRIORITY``. The import-time assertion at the bottom of
    this module verifies ``set(INTENT_PATTERNS) <= set(INTENT_PRIORITY)``. If
    you add a new pattern key without adding it to the priority tuple,
    ``AssertionError`` will fire on first import and name the missing intent.
    This is required to keep tie-break deterministic — the bug fixed by
    PR #247 (#254) was that dict-insertion order resolved ties silently.

    The reverse direction is intentionally *not* enforced: ``INTENT_PRIORITY``
    may declare future intents ahead of patterns being shipped.

Multi-faceted detection (issues #256 + #288):
    The ``multi_faceted`` regex tuple lives in ``src/agents/multi_faceted.py``
    as the single source of truth — see that module for context on why
    convergence is structural rather than semantic. The same module also
    hosts the boolean facet-scorer consumed by the chatbot routes;
    identity is asserted in
    ``tests/unit/test_agents/test_orchestrator/test_multi_faceted_ssot.py``.
"""

import logging
import re
import time
from typing import Literal, cast

from src.agents.multi_faceted import (
    MULTI_FACETED_PATTERNS,
    has_sequential_composition,
    is_multi_faceted_facet_score,
    is_multi_faceted_topic_count,
)
from src.utils.llm_factory import get_fast_llm, get_llm_provider
from src.utils.mock_llm import llm_or_marked_mock

from ..state import IntentClassification, OrchestratorState

logger = logging.getLogger(__name__)

# Type alias for intent types
IntentType = Literal[
    "causal_effect",
    "performance_gap",
    "segment_analysis",
    "experiment_design",
    "prediction",
    "resource_allocation",
    "explanation",
    "system_health",
    "drift_check",
    "feedback",
    "general",
]


# Tie-break priority for ``_pattern_classify``. When two intents tie on score,
# the one earlier in this tuple wins. Order: most-specific → least-specific.
# Documented + tested so behaviour cannot drift back to the dict-insertion-order
# arbitrary tie-breaks that commit 1dbc18fd papered over by dropping test
# queries. See issue #254.
INTENT_PRIORITY: tuple[str, ...] = (
    "experiment_monitor",  # most specific: active A/B experiment health
    "multi_faceted",  # multi-question composition signal (wins over component intents)
    "cohort_definition",  # patient-set construction
    "experiment_design",  # A/B planning
    "drift_check",  # data/model drift
    "segment_analysis",  # CATE / cohort effects
    "performance_gap",  # ROI / underperformance
    "resource_allocation",  # optimization
    "prediction",  # forecasts
    "feedback",  # learning from outcomes
    "causal_effect",  # general "why" — broader than the specific intents above
    "explanation",  # interpretation
    "system_health",  # operational status (catch-all for "health")
    "general",  # fallback
)


def _get_opik_connector():
    """Lazy import of OpikConnector to avoid circular imports."""
    try:
        from src.mlops.opik_connector import get_opik_connector

        return get_opik_connector()
    except ImportError:
        logger.debug("OpikConnector not available")
        return None
    except Exception as e:
        logger.warning(f"Failed to get OpikConnector: {e}")
        return None


class IntentClassifierNode:
    """Fast intent classification - optimized for <500ms.

    Uses pattern matching first, LLM only for ambiguous cases.
    """

    # Pattern-based classification for common queries
    INTENT_PATTERNS = {
        "causal_effect": [
            r"what.*(caus|impact|effect|driv|lead|result)",
            r"why.*(increase|decrease|change|drop|rise)",
            r"how does.*affect",
            r"what drives",
            r"attribution",
        ],
        "performance_gap": [
            r"(gap|opportunit|underperform|potential|improve)",
            r"roi.*(opportun|analys)",
            r"where.*underperform",
            r"untapped",
        ],
        "segment_analysis": [
            r"(segment|group|heterogen)",  # Removed "cohort" - handled by cohort_definition
            r"which.*(respond|perform).*(best|better)",
            r"\bcate\b|treatment effect.*by",
            r"differentiat.*strategy",
            r"subgroup.*analysis",
        ],
        "experiment_design": [
            r"(design|run|plan).*(experiment|test|trial)",
            r"a/b test",
            r"sample size",
            r"hypothesis.*test",
        ],
        "prediction": [
            r"predict|forecast|project",
            r"what will|expected",
            r"likelihood|probability",
        ],
        "resource_allocation": [
            r"(allocat|optimi|distribut).*(resource|budget|rep)",
            r"where.*invest",
            r"prioriti",
        ],
        "explanation": [
            r"explain|clarify|what does.*mean",
            r"help.*understand",
            r"break down",
        ],
        "system_health": [
            r"system.*(health|status)",
            r"model.*perform",
            r"pipeline.*status",
        ],
        "drift_check": [
            r"drift|shift|distribution.*change",
            r"data quality",
            r"model.*degrad",
        ],
        "feedback": [
            r"feedback|learn.*from",
            r"improve.*based on",
        ],
        "experiment_monitor": [
            r"(monitor|check|status).*(experiment|trial|a\/?b ?test)",
            r"sample ratio mismatch|\bsrm\b",
            r"interim analysis",
            r"(active|running).*(experiments?|trials?)",
            r"experiments?.*(health|status|issues)",
        ],
        # Single source of truth at ``src/agents/multi_faceted.py``
        # (issue #288). Identity-checked in test_multi_faceted_ssot.py.
        "multi_faceted": MULTI_FACETED_PATTERNS,
        "cohort_definition": [
            r"(define|create|build|construct).*(cohort|patient set|patient population)",
            r"cohort.*(definition|construction|criteria)",
            r"(inclusion|exclusion).*(criteria|rules)",
            r"patient.*eligib",
            r"eligible.*patient",
            r"filter.*patients",
            r"(remibrutinib|fabhalta|kisqali).*(cohort|patient)",
            r"(csu|pnh|breast cancer).*(cohort|patient|population)",
            r"\bcohort\b.*\bhcp",  # "cohort of HCPs" type queries
            r"\bhcp\b.*\bcohort",  # "HCP cohort" type queries
            r"(high.?value|high.?priority).*\b(hcp|physician|doctor)",  # high-value HCP queries
        ],
    }

    def __init__(self):
        """Initialize intent classifier with fast LLM for classification."""
        # Use fast LLM (Haiku or gpt-4o-mini based on LLM_PROVIDER env var).
        # In keyless contexts (Tier 1-5 harness, #606) fall back to an opt-in
        # MARKED mock (E2I_ALLOW_MOCK_LLM); prod stays fail-loud on a missing key.
        # _llm_classify already degrades to "general" on any parse error, so the
        # canned classification is a safe, parser-valid default.
        self.llm = llm_or_marked_mock(
            get_fast_llm,
            '{"primary_intent": "general", "confidence": 0.85, "requires_multi_agent": false}',
            max_tokens=256,
            timeout=2,
        )
        self._provider = get_llm_provider()

    async def execute(self, state: OrchestratorState) -> OrchestratorState:
        """Execute intent classification.

        Args:
            state: Current orchestrator state

        Returns:
            Updated state with intent classification
        """
        start_time = time.time()

        query = state.get("query", "").lower()

        # Try pattern matching first (fastest)
        pattern_result = self._pattern_classify(query)

        if pattern_result["confidence"] >= 0.8 and not self._needs_llm_disambiguation(
            query, pattern_result
        ):
            intent = pattern_result
        else:
            # Fall back to LLM for ambiguous OR borderline multi-part cases
            # (Fix 3b: soft multi-faceted signals that the deterministic layers
            # may have under-classified — see _needs_llm_disambiguation).
            intent = await self._llm_classify(state.get("query", ""))

        classification_time = int((time.time() - start_time) * 1000)

        return {
            **state,
            "intent": intent,
            "classification_latency_ms": classification_time,
            "current_phase": "routing",
        }

    def _pattern_classify(self, query: str) -> IntentClassification:
        """Fast pattern-based classification.

        Args:
            query: User query (lowercased)

        Returns:
            Intent classification result
        """
        scores = {}

        for intent, patterns in self.INTENT_PATTERNS.items():
            matched_count = 0
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    matched_count += 1
            # Any pattern match gives high confidence for that intent
            # More matches = higher confidence
            if matched_count > 0:
                scores[intent] = 0.8 + (0.2 * min(matched_count, 3) / 3)
            else:
                scores[intent] = 0.0

        if not scores or max(scores.values()) == 0:
            return IntentClassification(
                primary_intent="general",
                confidence=0.5,
                secondary_intents=[],
                requires_multi_agent=False,
            )

        # Deterministic tie-break: sort by (-score, INTENT_PRIORITY index). Lower
        # priority index = higher priority. Intents not in INTENT_PRIORITY lose
        # all ties to those that are. Issue #254.
        def _sort_key(item: tuple[str, float]) -> tuple[float, int]:
            name, score = item
            try:
                priority_idx = INTENT_PRIORITY.index(name)
            except ValueError:
                priority_idx = len(INTENT_PRIORITY)
            return (-score, priority_idx)

        ranked = sorted(scores.items(), key=_sort_key)
        primary = ranked[0][0]
        confidence = scores[primary]

        # Secondary intents in the same deterministic order.
        secondary = [k for k, v in ranked[1:] if v > 0]
        requires_multi_agent = len(secondary) > 0 and scores.get(secondary[0], 0) > 0.8

        # Fix 2 (audit C2/C3) — sequential-pipeline promotion. A dependency
        # marker ("then", "after that", "based on that", …) joining 2+ DISTINCT
        # strong intents signals a *dependent pipeline* the Tool Composer should
        # decompose — not a single intent and not a parallel pair. Guarded to an
        # explicit sequence marker so additive/parallel compounds (e.g. "what
        # causes A and what drives B and also explain C") stay single-agent,
        # preserving the locked near-miss negatives in
        # test_intent_classifier_multi_faceted.py.
        strong_components = [
            name for name, score in ranked if score >= 0.8 and name != "multi_faceted"
        ]
        if (
            primary != "multi_faceted"
            and len(strong_components) >= 2
            and has_sequential_composition(query)
        ):
            primary = "multi_faceted"
            confidence = max(confidence, scores.get("multi_faceted", 0.0), 0.85)
            secondary = strong_components
            requires_multi_agent = True

        return IntentClassification(
            primary_intent=cast(IntentType, primary),
            confidence=confidence,
            secondary_intents=secondary[:2],
            requires_multi_agent=requires_multi_agent,
        )

    def _needs_llm_disambiguation(self, query: str, pattern_result: IntentClassification) -> bool:
        """Decide whether a confidently pattern-classified *single* intent should
        still be escalated to the LLM (Fix 3b).

        The deterministic layers (regex patterns + sequential-pipeline promotion)
        catch the lexically clear multi-part queries. This predicate catches the
        residual: a query the pattern layer classified as a *non-multi_faceted*
        single intent at high confidence, yet which carries a soft multi-faceted
        signal — a sequence/dependency marker, or ≥2 facets, or ≥2 analytics
        topic groups. For those, the Haiku fallback (whose prompt already lists
        ``multi_faceted``) makes the final call rather than silently routing to
        one agent.

        Returns ``False`` when the result is already ``multi_faceted`` (the
        deterministic layers already decided) or when no soft signal is present.
        """
        if pattern_result.get("primary_intent") == "multi_faceted":
            return False
        q = query.lower()
        return bool(
            has_sequential_composition(q)
            or is_multi_faceted_facet_score(q)
            or is_multi_faceted_topic_count(q)
        )

    async def _llm_classify(self, query: str) -> IntentClassification:
        """LLM-based classification for ambiguous cases.

        Args:
            query: User query

        Returns:
            Intent classification result
        """
        prompt = f"""Classify this pharmaceutical analytics query into ONE primary intent.

Query: "{query}"

Intents:
- causal_effect: Questions about cause and effect, impact, attribution
- performance_gap: ROI opportunities, underperformance, potential improvements
- segment_analysis: Segment-specific effects, CATE, cohort analysis
- experiment_design: A/B tests, experiment planning, sample size
- experiment_monitor: Monitor running A/B experiments for SRM, interim, enrollment health
- prediction: Forecasting, projections, likelihood estimates
- resource_allocation: Budget/resource optimization, prioritization
- explanation: Clarifying results, interpreting findings
- system_health: Model/pipeline status, system performance
- drift_check: Data/model drift, distribution changes
- feedback: Learning from outcomes, improvement suggestions
- cohort_definition: Patient cohort construction, eligibility criteria, inclusion/exclusion rules
- multi_faceted: Multi-part questions combining 2+ distinct analyses (compare X and Y, then identify Z)
- general: Other/unclear

Respond with ONLY a JSON object:
{{"primary_intent": "<intent>", "confidence": <0.0-1.0>, "requires_multi_agent": <bool>}}"""

        try:
            # Get OpikConnector for LLM call tracing
            opik = _get_opik_connector()

            if opik and opik.is_enabled:
                # Trace the LLM call with dynamic provider info
                model_name = (
                    "gpt-4o-mini" if self._provider == "openai" else "claude-haiku-4-20250414"
                )
                async with opik.trace_llm_call(
                    model=model_name,
                    provider=self._provider,
                    prompt_template="intent_classification",
                    input_data={"query": query, "prompt": prompt},
                    metadata={"agent": "orchestrator", "operation": "intent_classification"},
                ) as llm_span:
                    response = await self.llm.ainvoke(prompt)
                    # Log tokens from response metadata
                    usage = response.response_metadata.get("usage", {})
                    llm_span.log_tokens(
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                    )
            else:
                # Fallback: no tracing
                response = await self.llm.ainvoke(prompt)

            import json

            result = json.loads(response.content)
            return IntentClassification(
                primary_intent=result.get("primary_intent", "general"),
                confidence=result.get("confidence", 0.5),
                secondary_intents=[],
                requires_multi_agent=result.get("requires_multi_agent", False),
            )
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return IntentClassification(
                primary_intent="general",
                confidence=0.3,
                secondary_intents=[],
                requires_multi_agent=False,
            )


# Import-time invariant (issue #266): every key in INTENT_PATTERNS must be
# ranked in INTENT_PRIORITY so tie-breaks remain deterministic. The reverse
# direction is intentionally NOT enforced — INTENT_PRIORITY may legitimately
# pre-declare future intents. If this fires, add the missing intent(s) to
# INTENT_PRIORITY in the correct specificity slot (most-specific to least).
_missing_priority_intents = set(IntentClassifierNode.INTENT_PATTERNS) - set(INTENT_PRIORITY)
assert not _missing_priority_intents, (
    f"INTENT_PATTERNS contains intents missing from INTENT_PRIORITY: "
    f"{sorted(_missing_priority_intents)}. "
    "Add them to INTENT_PRIORITY to preserve deterministic tie-break."
)
del _missing_priority_intents


# Export for use in graph
async def classify_intent(state: OrchestratorState) -> OrchestratorState:
    """Node function for intent classification.

    Args:
        state: Current state

    Returns:
        Updated state
    """
    classifier = IntentClassifierNode()
    return await classifier.execute(state)
