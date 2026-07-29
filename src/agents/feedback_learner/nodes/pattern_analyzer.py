"""
E2I Feedback Learner Agent - Pattern Analyzer Node
Version: 4.2
Purpose: Deep reasoning for pattern detection in feedback
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, cast
from uuid import uuid4

from src.utils.llm_content import normalize_llm_content

from ..rating_utils import (
    COPILOT_SURFACE,
    SURFACE_RATING_CEILINGS,
    rating_surface,
    rating_to_numeric,
)
from ..state import DetectedPattern, FeedbackLearnerState

logger = logging.getLogger(__name__)


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


# Shared normalizer (rating_utils): thumbs strings and numerics on one 1-5
# scale. Kept under the old private name — this module's detectors and its
# tests reference it directly.
_rating_to_numeric = rating_to_numeric


# Mirror of the ``DetectedPattern`` TypedDict Literals (state.py). The
# LLM/DSPy pattern paths emit free-form values into these enum-constrained
# contract fields; measured 2026-06-11: the LLM invented
# pattern_type='baseline_establishment' → pydantic literal_error at
# FeedbackLearnerOutput validation, failing the whole agent run.
_VALID_PATTERN_TYPES: frozenset = frozenset(
    {"accuracy_issue", "latency_issue", "relevance_issue", "format_issue", "coverage_gap"}
)
_VALID_SEVERITIES: frozenset = frozenset({"low", "medium", "high", "critical"})


def _sanitize_llm_pattern_enums(
    pattern_type: Any, severity: Any
) -> "Optional[tuple[Any, Any]]":  # values verified against the Literal sets below
    """Validate LLM-emitted enum fields against the DetectedPattern contract.

    Returns ``(pattern_type, severity)`` when usable, ``None`` when the
    pattern must be DROPPED. Uniform fail-closed (codex R4): BOTH an
    out-of-contract pattern_type AND an out-of-contract severity drop the
    whole pattern — the category is the semantic payload, and severity is
    LOAD-BEARING downstream (learning_extractor emits the model_retrain
    recommendation only for high/critical), so remapping either would
    fabricate a decision the model never made. Dropped patterns are
    surfaced via ``pattern_parse_anomalies`` on the node's state update,
    never silently discarded.
    """
    if pattern_type not in _VALID_PATTERN_TYPES:
        logger.warning(
            "Dropping LLM-emitted pattern with out-of-contract pattern_type=%r "
            "(allowed: %s) — the model invented a category outside the "
            "DetectedPattern Literal.",
            pattern_type,
            sorted(_VALID_PATTERN_TYPES),
        )
        return None
    if severity not in _VALID_SEVERITIES:
        logger.warning(
            "Dropping LLM-emitted pattern with out-of-contract severity=%r "
            "(allowed: %s) — severity gates the retrain path downstream; "
            "guessing a mapping would fabricate that decision.",
            severity,
            sorted(_VALID_SEVERITIES),
        )
        return None
    return (pattern_type, severity)


class PatternAnalyzerNode:
    """
    Deep reasoning for pattern detection in feedback.
    Identifies systematic issues requiring attention.
    """

    def __init__(
        self,
        use_llm: bool = False,
        llm: Optional[Any] = None,
        prefer_optimized: bool = True,
    ):
        """
        Initialize pattern analyzer.

        Args:
            use_llm: Whether to use LLM for analysis
            llm: Optional LLM instance
            prefer_optimized: When True, prefer the latest saved optimized DSPy
                module (feedback_learner_pattern) produced by the optimization
                loop — this is what closes the self-improvement loop. Falls back
                cleanly to LLM/deterministic when no artifact or LM is present.
        """
        self.use_llm = use_llm
        self.llm = llm
        self.prefer_optimized = prefer_optimized
        self._optimized_module: Optional[Any] = None
        self._optimized_meta: Dict[str, Any] = {}
        self._optimized_load_attempted = False

    async def execute(self, state: FeedbackLearnerState) -> FeedbackLearnerState:
        """Execute pattern analysis."""
        start_time = time.time()

        # Check if already failed
        if state.get("status") == "failed":
            return state

        try:
            feedback_items = state.get("feedback_items") or []

            if not feedback_items:
                return {
                    **state,
                    "detected_patterns": [],
                    "pattern_clusters": {},
                    "analysis_latency_ms": 0,
                    "status": "extracting",
                }

            # codex R4: per-run reset of the out-of-contract drop counter
            # (parse sites increment it; surfaced below — never silent).
            self._enum_drop_count = 0

            # Analyze patterns: prefer the optimized DSPy module (closes the
            # self-improvement loop); fall back to LLM, then deterministic.
            result = None
            if self.prefer_optimized:
                result = self._analyze_with_dspy(state)
            if result is None:
                if self.use_llm and self.llm:
                    result = await self._analyze_with_llm(state)
                else:
                    result = self._analyze_deterministic(state)

            analysis_time = int((time.time() - start_time) * 1000)

            logger.info(f"Pattern analysis complete: {len(result['patterns'])} patterns detected")

            out: FeedbackLearnerState = {
                **state,
                "detected_patterns": result["patterns"],
                "pattern_clusters": result["clusters"],
                "analysis_latency_ms": analysis_time,
                "model_used": result.get("model_used", "deterministic"),
                "status": "extracting",
            }
            # codex R4 fail-open guard: dropped out-of-contract patterns must
            # be VISIBLE. "0 patterns detected" after drops is a PARSE
            # anomaly, not a clean no-findings result — downstream and
            # observability can distinguish the two via this field.
            dropped = getattr(self, "_enum_drop_count", 0)
            if dropped:
                out["pattern_parse_anomalies"] = {"dropped_out_of_contract": dropped}
                if not result["patterns"]:
                    logger.error(
                        "ALL %d LLM-emitted patterns were out-of-contract and "
                        "dropped — this run's '0 patterns detected' is a parse "
                        "anomaly, not a clean no-findings result.",
                        dropped,
                    )
            return out

        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            return {
                **state,
                "errors": [{"node": "pattern_analyzer", "error": str(e)}],
                "status": "failed",
            }

    def _analyze_deterministic(self, state: FeedbackLearnerState) -> Dict[str, Any]:
        """Deterministic pattern analysis using heuristics."""
        feedback_items = state.get("feedback_items") or []
        summary: Dict[str, Any] = cast(Dict[str, Any], state.get("feedback_summary") or {})

        patterns: List[DetectedPattern] = []
        # #1256: ids must be unique ACROSS cycles, not just within one — the
        # API persists patterns with pattern_id as the upsert key, so a bare
        # positional "P1" collides with every previous cycle's first pattern
        # and inherits that row's created_at (insert-only DB default).
        run_tag = uuid4().hex[:8]
        pattern_id = 1

        # Analyze by feedback type
        summary.get("by_type", {})
        by_agent = summary.get("by_agent", {})

        # Check for low ratings pattern. F15 (audit): normalize numeric AND
        # string ratings (thumbs_up/down) so real chatbot feedback is not
        # silently dropped from pattern detection.
        scored_raw = [
            (fb, _rating_to_numeric(fb["user_feedback"]))
            for fb in feedback_items
            if fb["feedback_type"] == "rating"
        ]
        # Bind to a NEW variable (not a reassignment of ``scored_raw``) so mypy
        # infers the None-filtered element type as ``float`` rather than keeping
        # the original ``float | None`` declared type.
        scored = [(fb, num) for fb, num in scored_raw if num is not None]
        # #1251: group by reward surface BEFORE averaging. Surfaces have
        # different reward ceilings (copilot 0.8 vs cognitive 1.0, #1240), so
        # a pooled mean's distance-to-gate depends on source mix — a low
        # copilot pool hides behind a high cognitive pool (and vice versa).
        # The bottom-anchored < 3.0 gate itself is scale-agreeing across
        # surfaces (reward 0.5 = rating 3.0 on all), so each pool is gated
        # against the same threshold; single-surface pools behave exactly as
        # the old pooled gate did.
        pools: Dict[str, List[Any]] = {}
        # NB: loop var deliberately NOT named ``num`` — the per-agent detector
        # below walrus-binds ``num`` to ``float | None`` in this same function
        # scope, and a prior ``float`` binding here would flag that assignment.
        for fb, rated in scored:
            pools.setdefault(rating_surface(fb.get("metadata")), []).append((fb, rated))
        for surface in sorted(pools):
            pool = pools[surface]
            avg_rating = sum(num for _, num in pool) / len(pool)
            # #1258: min-pool guard — splitting by surface (#1251) made tiny
            # pools possible (one bad copilot turn in a low-traffic window =
            # an n=1 pool), and a single observation must not emit a persisted
            # pattern. Every sibling detector here has a count floor; 3 keeps
            # the smallest gated cohort the tests pin (n=3 pools still fire).
            if len(pool) >= 3 and avg_rating < 3.0:  # 1-5 scale, bottom-anchored
                affected_agents = list({fb["source_agent"] for fb, num in pool if num < 3})
                patterns.append(
                    DetectedPattern(
                        pattern_id=f"P{pattern_id}-{run_tag}",
                        pattern_type="accuracy_issue",
                        description=f"Low average user ratings detected (surface: {surface})",
                        frequency=len(pool),
                        severity="high" if avg_rating < 2.0 else "medium",
                        affected_agents=affected_agents,
                        example_feedback_ids=[fb["feedback_id"] for fb, _ in pool[:3]],
                        root_cause_hypothesis="Agent responses may not meet user expectations",
                    )
                )
                pattern_id += 1

        # Check for correction pattern
        corrections = [fb for fb in feedback_items if fb["feedback_type"] == "correction"]
        if len(corrections) > 5:
            affected_agents = list({fb["source_agent"] for fb in corrections})
            patterns.append(
                DetectedPattern(
                    pattern_id=f"P{pattern_id}-{run_tag}",
                    pattern_type="accuracy_issue",
                    description="Multiple user corrections submitted",
                    frequency=len(corrections),
                    severity="medium" if len(corrections) < 10 else "high",
                    affected_agents=affected_agents,
                    example_feedback_ids=[fb["feedback_id"] for fb in corrections[:3]],
                    root_cause_hypothesis="Agent may have knowledge gaps or outdated information",
                )
            )
            pattern_id += 1

        # Check for outcome errors
        outcomes = [fb for fb in feedback_items if fb["feedback_type"] == "outcome"]
        if outcomes:
            errors = []
            for fb in outcomes:
                if isinstance(fb["user_feedback"], dict):
                    error = fb["user_feedback"].get("error", 0)
                    if abs(error) > 0:
                        errors.append((fb, error))

            if len(errors) > 3:
                avg_error = sum(abs(e[1]) for e in errors) / len(errors)
                affected_agents = list({e[0]["source_agent"] for e in errors})
                patterns.append(
                    DetectedPattern(
                        pattern_id=f"P{pattern_id}-{run_tag}",
                        pattern_type="accuracy_issue",
                        description=f"Prediction errors detected (avg error: {avg_error:.2f})",
                        frequency=len(errors),
                        severity="high" if avg_error > 0.5 else "medium",
                        affected_agents=affected_agents,
                        example_feedback_ids=[e[0]["feedback_id"] for e in errors[:3]],
                        root_cause_hypothesis="Model predictions may be biased or outdated",
                    )
                )
                pattern_id += 1

        # Check for agent-specific issues
        for agent, count in by_agent.items():
            agent_feedback = [fb for fb in feedback_items if fb["source_agent"] == agent]
            # Normalize through _rating_to_numeric like the overall low-rating
            # detector — a bare isinstance gate here silently dropped every
            # thumbs_down string, hiding per-agent negative streaks.
            agent_negative = len(
                [
                    fb
                    for fb in agent_feedback
                    if (
                        fb["feedback_type"] == "correction"
                        or (
                            fb["feedback_type"] == "rating"
                            and (num := _rating_to_numeric(fb["user_feedback"])) is not None
                            and num < 3
                        )
                    )
                ]
            )

            if agent_negative > 3 and agent_negative / max(count, 1) > 0.3:
                patterns.append(
                    DetectedPattern(
                        pattern_id=f"P{pattern_id}-{run_tag}",
                        pattern_type="relevance_issue",
                        description=f"Agent '{agent}' has high negative feedback rate",
                        frequency=agent_negative,
                        severity="high",
                        affected_agents=[agent],
                        example_feedback_ids=[fb["feedback_id"] for fb in agent_feedback[:3]],
                        root_cause_hypothesis=f"Agent '{agent}' may need retraining or prompt updates",
                    )
                )
                pattern_id += 1

        # Cluster patterns by type
        clusters = self._cluster_patterns(patterns)

        return {
            "patterns": patterns,
            "clusters": clusters,
            "model_used": "deterministic",
        }

    def _load_optimized_pattern_module(self) -> Optional[Any]:
        """Load the latest optimized feedback_learner_pattern module, or None.

        An intentional miss (no artifact saved yet -> FileNotFoundError) is
        cached so we don't re-probe the filesystem every cycle. A transient
        error (import race, corrupt read) is NOT cached, so a later cycle can
        retry once the condition clears. Uses a zero-arg factory because
        load_optimized_module calls module_cls() (versioning.py).
        """
        if self._optimized_load_attempted:
            return self._optimized_module
        try:
            import dspy

            from src.optimization.gepa import load_optimized_module

            from ..dspy_integration import PatternDetectionSignature

            module, meta = load_optimized_module(
                lambda: dspy.ChainOfThought(PatternDetectionSignature),
                agent_name="feedback_learner_pattern",
            )
            self._optimized_module = module
            self._optimized_meta = meta
            self._optimized_load_attempted = True  # success -> cache it
            logger.info(
                "Loaded optimized pattern module version=%s",
                meta.get("version_id", "?"),
            )
        except FileNotFoundError:
            # Intentional miss: no artifact yet. Cache so we don't re-probe.
            # INFO, not DEBUG: prod ran on this fallback for 6 weeks while the
            # 2026-06-08 artifact sat unshipped and everyone believed the tuned
            # module was live (dspy_lane_ab_20260718.md §7). Logged once per
            # process (the miss is cached), so this stays quiet in volume.
            logger.info("No optimized pattern module saved yet; using base module")
            self._optimized_module = None
            self._optimized_load_attempted = True
        except Exception as e:  # noqa: BLE001
            # Transient: do NOT cache -> allow a later cycle to retry.
            logger.warning("Failed to load optimized pattern module (will retry): %s", e)
            self._optimized_module = None
        return self._optimized_module

    def _analyze_with_dspy(self, state: FeedbackLearnerState) -> Optional[Dict[str, Any]]:
        """Run the optimized DSPy module. Returns a result dict or None to fall back."""
        module = self._load_optimized_pattern_module()
        if module is None:
            return None

        import dspy

        from src.optimization.dspy_lm import ensure_dspy_configured

        if not ensure_dspy_configured():
            return None

        import json as _json

        # Bind the configured LM so the module runs regardless of thread context.
        lm = getattr(dspy.settings, "lm", None)
        if lm is not None and hasattr(module, "set_lm"):
            module.set_lm(lm)

        feedback_items = state.get("feedback_items") or []
        cognitive = cast(Dict[str, Any], state.get("cognitive_context") or {})
        try:
            prediction = module(
                feedback_batch=_json.dumps([dict(fb) for fb in feedback_items[:20]]),
                agent_baselines=_json.dumps(cognitive.get("agent_baselines", {})),
                historical_patterns=_json.dumps(cognitive.get("historical_patterns", [])),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Optimized DSPy pattern analysis failed; falling back: %s", e)
            return None

        raw_patterns = getattr(prediction, "patterns", []) or []
        patterns: List[DetectedPattern] = []
        run_tag = uuid4().hex[:8]  # #1256: see _analyze_deterministic
        for i, p in enumerate(raw_patterns, start=1):
            if not isinstance(p, dict):
                continue
            # LM output is free-form; the TypedDict fields are Literals.
            # Validate against the contract: drop out-of-contract pattern
            # types, clamp out-of-contract severities (see
            # _sanitize_llm_pattern_enums).
            sanitized = _sanitize_llm_pattern_enums(
                p.get("type") or p.get("pattern_type", "accuracy_issue"),
                p.get("severity") or "medium",
            )
            if sanitized is None:
                self._enum_drop_count = getattr(self, "_enum_drop_count", 0) + 1
                continue
            ptype, severity = sanitized
            patterns.append(
                DetectedPattern(
                    pattern_id=f"P{i}-{run_tag}",
                    pattern_type=ptype,
                    description=str(p.get("description", "")),
                    frequency=int(p.get("frequency", 0) or 0),
                    severity=severity,
                    affected_agents=list(p.get("affected_agents", []) or []),
                    example_feedback_ids=[
                        fb["feedback_id"] for fb in feedback_items[:3] if "feedback_id" in fb
                    ],
                    root_cause_hypothesis=str(p.get("root_cause_hypothesis", "")),
                )
            )

        version = self._optimized_meta.get("version_id", "unknown")
        return {
            "patterns": patterns,
            "clusters": self._cluster_patterns(patterns),
            "model_used": f"dspy_optimized:{version}",
        }

    async def _analyze_with_llm(self, state: FeedbackLearnerState) -> Dict[str, Any]:
        """Use LLM for sophisticated pattern analysis."""
        if not self.llm:
            return self._analyze_deterministic(state)

        try:
            prompt = self._build_analysis_prompt(state)

            # Get OpikConnector for LLM call tracing
            opik = _get_opik_connector()
            model_name = getattr(self.llm, "model", "claude")

            if opik and opik.is_enabled:
                # Trace the LLM call
                async with opik.trace_llm_call(
                    model=model_name,
                    provider="anthropic",
                    prompt_template="pattern_analysis",
                    input_data={"prompt": prompt[:500]},
                    metadata={"agent": "feedback_learner", "operation": "pattern_analysis"},
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

            # AIMessage.content is str | list of content blocks (#1358)
            patterns = self._parse_patterns(normalize_llm_content(response.content))
            clusters = self._cluster_patterns(patterns)

            return {
                "patterns": patterns,
                "clusters": clusters,
                "model_used": model_name,
            }
        except Exception as e:
            logger.warning(f"LLM analysis failed, using fallback: {e}")
            return self._analyze_deterministic(state)

    def _build_analysis_prompt(self, state: FeedbackLearnerState) -> str:
        """Build analysis prompt for LLM."""
        feedback_items = state.get("feedback_items") or []
        summary: Dict[str, Any] = cast(Dict[str, Any], state.get("feedback_summary") or {})

        # Sample feedback for analysis (avoid token limits)
        sample_size = min(50, len(feedback_items))
        sampled = feedback_items[:sample_size]

        feedback_str = "\n\n".join(
            [
                f"**Feedback {i + 1}** (Type: {fb['feedback_type']}, Agent: {fb['source_agent']})\n"
                f"Query: {fb['query'][:200]}\n"
                f"Response: {fb['agent_response'][:300]}\n"
                f"Feedback: {fb['user_feedback']}"
                for i, fb in enumerate(sampled)
            ]
        )

        return f"""Analyze feedback to identify systematic patterns.

## Summary
- Total: {summary.get("total_count", 0)}
- By type: {json.dumps(summary.get("by_type", {}))}
- By agent: {json.dumps(summary.get("by_agent", {}))}
- Avg rating: {summary.get("average_rating", "N/A")}
- Avg rating by source surface (ceilings differ — copilot tops out at {SURFACE_RATING_CEILINGS[COPILOT_SURFACE]:.1f}): {json.dumps(summary.get("average_rating_by_source", {}))}

## Sample Feedback

{feedback_str}

---

Identify patterns (accuracy_issue, latency_issue, relevance_issue, format_issue, coverage_gap).

Output JSON:
```json
{{
  "patterns": [
    {{
      "pattern_id": "P1",
      "pattern_type": "...",
      "description": "...",
      "frequency": <int>,
      "severity": "low|medium|high|critical",
      "affected_agents": ["..."],
      "example_feedback_ids": ["..."],
      "root_cause_hypothesis": "..."
    }}
  ]
}}
```"""

    def _parse_patterns(self, content: str) -> List[DetectedPattern]:
        """Parse detected patterns from response."""
        json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                patterns = []
                run_tag = uuid4().hex[:8]  # #1256: see _analyze_deterministic
                for p in data.get("patterns", []):
                    # LLM enum validation: drop out-of-contract patterns
                    # (see _sanitize_llm_pattern_enums); drops are counted
                    # and surfaced as pattern_parse_anomalies, never silent.
                    sanitized = _sanitize_llm_pattern_enums(
                        p.get("pattern_type", "accuracy_issue"),
                        p.get("severity", "medium"),
                    )
                    if sanitized is None:
                        self._enum_drop_count = getattr(self, "_enum_drop_count", 0) + 1
                        continue
                    ptype, severity = sanitized
                    patterns.append(
                        DetectedPattern(
                            pattern_id=f"{p.get('pattern_id', 'P?')}-{run_tag}",
                            pattern_type=ptype,
                            description=p.get("description", ""),
                            frequency=p.get("frequency", 1),
                            severity=severity,
                            affected_agents=p.get("affected_agents", []),
                            example_feedback_ids=p.get("example_feedback_ids", []),
                            root_cause_hypothesis=p.get("root_cause_hypothesis", ""),
                        )
                    )
                return patterns
            except (json.JSONDecodeError, TypeError):
                pass

        return []

    def _cluster_patterns(self, patterns: List[DetectedPattern]) -> Dict[str, List[str]]:
        """Cluster patterns by type."""
        clusters: Dict[str, List[str]] = {}

        for pattern in patterns:
            ptype = pattern["pattern_type"]
            if ptype not in clusters:
                clusters[ptype] = []
            clusters[ptype].append(pattern["pattern_id"])

        return clusters
