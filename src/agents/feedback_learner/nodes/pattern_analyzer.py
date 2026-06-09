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


def _rating_to_numeric(value: Any) -> Optional[float]:
    """Normalize a user rating to a 1-5 numeric scale.

    F15 (audit): the real feedback source (``chatbot_message_feedback``) stores
    ratings as enum STRINGS (``thumbs_up``/``thumbs_down``), but the low-rating
    pattern detector previously accepted only ``int``/``float`` — silently
    dropping every real string rating, so collected feedback produced zero
    patterns and ``update_effectiveness`` stayed pinned at 0.0. Handle both
    numeric ratings and the string forms; return None for genuinely unknown
    values (excluded, not coerced).
    """
    if isinstance(value, bool):
        return 5.0 if value else 1.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        v = value.strip().lower()
        positive = {"thumbs_up", "thumbsup", "up", "positive", "good", "helpful", "yes", "👍"}
        negative = {"thumbs_down", "thumbsdown", "down", "negative", "bad", "unhelpful", "no", "👎"}
        if v in positive:
            return 5.0
        if v in negative:
            return 1.0
        try:
            return float(v)  # numeric-as-string (e.g. "4")
        except ValueError:
            return None
    return None


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

            return {
                **state,
                "detected_patterns": result["patterns"],
                "pattern_clusters": result["clusters"],
                "analysis_latency_ms": analysis_time,
                "model_used": result.get("model_used", "deterministic"),
                "status": "extracting",
            }

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
        if scored:
            avg_rating = sum(num for _, num in scored) / len(scored)
            if avg_rating < 3.0:  # Assuming 1-5 scale
                affected_agents = list({fb["source_agent"] for fb, num in scored if num < 3})
                patterns.append(
                    DetectedPattern(
                        pattern_id=f"P{pattern_id}",
                        pattern_type="accuracy_issue",
                        description="Low average user ratings detected",
                        frequency=len(scored),
                        severity="high" if avg_rating < 2.0 else "medium",
                        affected_agents=affected_agents,
                        example_feedback_ids=[fb["feedback_id"] for fb, _ in scored[:3]],
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
                    pattern_id=f"P{pattern_id}",
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
                        pattern_id=f"P{pattern_id}",
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
            agent_negative = len(
                [
                    fb
                    for fb in agent_feedback
                    if (
                        fb["feedback_type"] == "correction"
                        or (
                            fb["feedback_type"] == "rating"
                            and isinstance(fb["user_feedback"], (int, float))
                            and fb["user_feedback"] < 3
                        )
                    )
                ]
            )

            if agent_negative > 3 and agent_negative / max(count, 1) > 0.3:
                patterns.append(
                    DetectedPattern(
                        pattern_id=f"P{pattern_id}",
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
            logger.debug("No optimized pattern module saved yet; using fallback")
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
        for i, p in enumerate(raw_patterns, start=1):
            if not isinstance(p, dict):
                continue
            patterns.append(
                DetectedPattern(
                    pattern_id=f"P{i}",
                    # LM output is free-form; the TypedDict fields are Literals.
                    # Mirror the LLM path's `.get(...)` (Any) so the best-effort
                    # value flows through without a false typeddict-item error.
                    pattern_type=p.get("type") or p.get("pattern_type", "accuracy_issue"),
                    description=str(p.get("description", "")),
                    frequency=int(p.get("frequency", 0) or 0),
                    severity=p.get("severity") or "medium",
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

            patterns = self._parse_patterns(response.content)
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
                for p in data.get("patterns", []):
                    patterns.append(
                        DetectedPattern(
                            pattern_id=p.get("pattern_id", "P?"),
                            pattern_type=p.get("pattern_type", "accuracy_issue"),
                            description=p.get("description", ""),
                            frequency=p.get("frequency", 1),
                            severity=p.get("severity", "medium"),
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
