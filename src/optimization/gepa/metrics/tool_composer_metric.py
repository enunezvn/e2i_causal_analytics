"""Tool Composer GEPA Metric for E2I Tier 1 Orchestration Agent.

This module provides the GEPA metric for optimizing the Tool Composer agent,
which dynamically composes analytical tools via a 4-phase pipeline:
    DECOMPOSE → PLAN → EXECUTE → SYNTHESIZE

The metric optimizes for:
- Decomposition quality (sub-question coverage, intent accuracy)
- Planning quality (tool selection, dependency ordering)
- Execution quality (success rate, error handling)
- Synthesis quality (answer relevance, confidence calibration)
"""

from dataclasses import dataclass, field
from typing import Optional, Union

from dspy import Example, Prediction

from src.optimization.gepa.metrics.base import DSPyTrace, ScoreWithFeedback


@dataclass
class ToolComposerGEPAMetric:
    """GEPA metric for Tool Composer agent (Tier 1).

    Integrates with:
    - tool_compositions table (V4.2)
    - 4-phase pipeline (decompose→plan→execute→synthesize)
    - Tool registry for validation

    Attributes:
        name: Metric name identifier
        description: Metric description for logging
        decomposition_weight: Weight for decomposition quality (default 0.25)
        planning_weight: Weight for planning quality (default 0.25)
        execution_weight: Weight for execution quality (default 0.25)
        synthesis_weight: Weight for synthesis quality (default 0.25)
        min_sub_questions: Minimum expected sub-questions for complex queries
        max_execution_time_ms: SLA threshold for total execution time
    """

    name: str = "tool_composer_gepa"
    description: str = (
        "GEPA metric for Tier 1 Tool Composer - decomposition, planning, execution, synthesis"
    )

    decomposition_weight: float = 0.25
    planning_weight: float = 0.25
    execution_weight: float = 0.25
    synthesis_weight: float = 0.25

    min_sub_questions: int = 2
    max_execution_time_ms: int = 30000  # 30s SLA

    # Intent types that should be recognized
    valid_intents: list[str] = field(
        default_factory=lambda: [
            "CAUSAL",
            "COMPARATIVE",
            "PREDICTIVE",
            "DESCRIPTIVE",
            "EXPERIMENTAL",
        ]
    )

    def __call__(
        self,
        gold: Example,
        pred: Prediction,
        trace: Optional[DSPyTrace] = None,
        pred_name: Optional[str] = None,
        pred_trace: Optional[DSPyTrace] = None,
    ) -> Union[float, ScoreWithFeedback]:
        """Compute score for tool composition.

        GEPA requires metrics to return a float score or ScoreWithFeedback.
        The score is a weighted combination of:
        - Decomposition quality (25%)
        - Planning quality (25%)
        - Execution quality (25%)
        - Synthesis quality (25%)

        Args:
            gold: Ground truth Example with expected outputs
            pred: Model Prediction with composition results
            trace: Full DSPy execution trace (optional)
            pred_name: Name of the predictor being optimized (optional)
            pred_trace: Execution trace for this specific predictor (optional)

        Returns:
            Float score between 0.0 and 1.0, or ScoreWithFeedback dict
        """
        scores = {}
        feedback_parts = []

        # Component 1: Decomposition Quality (Phase 1)
        scores["decomposition"], feedback = self._score_decomposition(pred, gold)
        feedback_parts.append(f"[DECOMPOSE] {feedback}")

        # Component 2: Planning Quality (Phase 2)
        scores["planning"], feedback = self._score_planning(pred, gold)
        feedback_parts.append(f"[PLAN] {feedback}")

        # Component 3: Execution Quality (Phase 3)
        scores["execution"], feedback = self._score_execution(pred)
        feedback_parts.append(f"[EXECUTE] {feedback}")

        # Component 4: Synthesis Quality (Phase 4)
        scores["synthesis"], feedback = self._score_synthesis(pred, gold)
        feedback_parts.append(f"[SYNTHESIZE] {feedback}")

        # Aggregate weighted score
        total_score = (
            self.decomposition_weight * scores["decomposition"]
            + self.planning_weight * scores["planning"]
            + self.execution_weight * scores["execution"]
            + self.synthesis_weight * scores["synthesis"]
        )

        # Return with feedback for GEPA reflection
        return {
            "score": total_score,
            "feedback": self._build_feedback(total_score, scores, feedback_parts, pred_name),
        }

    def _score_decomposition(self, pred: Prediction, gold: Example) -> tuple[float, str]:
        """Score decomposition quality from Phase 1.

        Evaluates:
        - Number of sub-questions generated
        - Intent classification accuracy
        - Entity coverage

        Args:
            pred: Prediction with decomposition result
            gold: Example with expected sub-questions

        Returns:
            Tuple of (score, feedback_message)
        """
        decomposition = getattr(pred, "decomposition", None)

        if not decomposition:
            return 0.0, "CRITICAL: No decomposition result. Check Phase 1 execution"

        sub_questions = getattr(decomposition, "sub_questions", [])
        question_count = len(sub_questions)

        if question_count == 0:
            return 0.0, "CRITICAL: Zero sub-questions generated"

        score = 0.0
        issues = []

        # Check minimum sub-questions
        if question_count >= self.min_sub_questions:
            score += 0.4
        else:
            issues.append(f"Only {question_count} sub-questions (min={self.min_sub_questions})")
            score += 0.2

        # Check intent classification
        intents_valid = 0
        for sq in sub_questions:
            intent = getattr(sq, "intent", None)
            if intent and intent in self.valid_intents:
                intents_valid += 1

        if question_count > 0:
            intent_ratio = intents_valid / question_count
            score += 0.3 * intent_ratio
            if intent_ratio < 1.0:
                issues.append(f"{question_count - intents_valid} invalid intents")

        # Check entity extraction
        extracted_entities = getattr(decomposition, "extracted_entities", [])
        expected_entities = getattr(gold, "expected_entities", [])

        if expected_entities and extracted_entities:
            overlap = len(set(extracted_entities) & set(expected_entities))
            entity_ratio = overlap / len(expected_entities)
            score += 0.3 * entity_ratio
            if entity_ratio < 1.0:
                missing = set(expected_entities) - set(extracted_entities)
                issues.append(f"Missing entities: {missing}")
        elif extracted_entities:
            score += 0.15  # Partial credit for any extraction

        if issues:
            return min(score, 1.0), f"{question_count} sub-questions. Issues: {'; '.join(issues)}"
        return min(score, 1.0), f"{question_count} sub-questions, all intents valid"

    def _score_planning(self, pred: Prediction, gold: Example) -> tuple[float, str]:
        """Score planning quality from Phase 2.

        Evaluates:
        - Tool selection appropriateness
        - Dependency ordering correctness
        - Parallel group efficiency

        Args:
            pred: Prediction with execution plan
            gold: Example with expected tool mappings

        Returns:
            Tuple of (score, feedback_message)
        """
        plan = getattr(pred, "plan", None)

        if not plan:
            return 0.0, "CRITICAL: No execution plan. Check Phase 2 execution"

        steps = getattr(plan, "steps", [])
        step_count = len(steps)

        if step_count == 0:
            return 0.0, "CRITICAL: Zero execution steps planned"

        score = 0.0
        issues = []

        # Check tool mappings exist
        tool_mappings = getattr(plan, "tool_mappings", [])
        if tool_mappings:
            score += 0.3

            # Check mapping confidence
            avg_confidence = sum(getattr(m, "confidence", 0.5) for m in tool_mappings) / len(
                tool_mappings
            )
            if avg_confidence >= 0.8:
                score += 0.2
            elif avg_confidence >= 0.6:
                score += 0.1
                issues.append(f"Avg tool confidence {avg_confidence:.2f} < 0.8")
        else:
            issues.append("No tool mappings")

        # Check dependency ordering
        parallel_groups = getattr(plan, "parallel_groups", [])
        if parallel_groups:
            # More parallel groups = better efficiency
            if len(parallel_groups) >= 2:
                score += 0.3
            else:
                score += 0.15
        else:
            # Check if any steps have dependencies
            has_deps = any(getattr(s, "depends_on_steps", []) for s in steps)
            if has_deps:
                score += 0.2  # Has dependencies but no parallel optimization
            else:
                score += 0.3  # All independent steps

        # Check expected tools if provided
        expected_tools = getattr(gold, "expected_tools", [])
        if expected_tools:
            actual_tools = [getattr(s, "tool_name", "") for s in steps]
            overlap = len(set(actual_tools) & set(expected_tools))
            tool_ratio = overlap / len(expected_tools) if expected_tools else 0
            score += 0.2 * tool_ratio
            if tool_ratio < 1.0:
                missing = set(expected_tools) - set(actual_tools)
                issues.append(f"Missing tools: {missing}")
        else:
            score += 0.1  # No expected tools to validate against

        if issues:
            return min(score, 1.0), f"{step_count} steps. Issues: {'; '.join(issues)}"
        return min(score, 1.0), f"{step_count} steps planned, optimized for parallel execution"

    def _score_execution(self, pred: Prediction) -> tuple[float, str]:
        """Score execution quality from Phase 3.

        Evaluates:
        - Tool success rate
        - Error handling
        - Execution time

        Args:
            pred: Prediction with execution trace

        Returns:
            Tuple of (score, feedback_message)
        """
        execution = getattr(pred, "execution", None)

        if not execution:
            return 0.0, "CRITICAL: No execution trace. Check Phase 3 execution"

        tools_executed = getattr(execution, "tools_executed", 0)
        tools_succeeded = getattr(execution, "tools_succeeded", 0)

        if tools_executed == 0:
            return 0.0, "CRITICAL: Zero tools executed"

        score = 0.0
        issues = []

        # Success rate (most important)
        success_rate = tools_succeeded / tools_executed
        score += 0.5 * success_rate
        if success_rate < 1.0:
            failed = tools_executed - tools_succeeded
            issues.append(f"{failed}/{tools_executed} tools failed")

        # Check execution time
        total_duration = getattr(pred, "total_duration_ms", 0)
        if total_duration > 0:
            if total_duration <= self.max_execution_time_ms:
                score += 0.3
            elif total_duration <= self.max_execution_time_ms * 1.5:
                score += 0.15
                issues.append(
                    f"Execution time {total_duration}ms > SLA {self.max_execution_time_ms}ms"
                )
            else:
                issues.append(
                    f"Execution time {total_duration}ms >> SLA {self.max_execution_time_ms}ms"
                )
        else:
            score += 0.1  # No timing info

        # Check for retries (error recovery)
        step_results = getattr(execution, "step_results", [])
        retry_count = sum(getattr(r, "retries", 0) for r in step_results if r)
        if retry_count > 0 and success_rate > 0.5:
            score += 0.2  # Recovered from errors
        elif success_rate == 1.0:
            score += 0.2  # No retries needed

        if issues:
            return min(
                score, 1.0
            ), f"{success_rate * 100:.0f}% success. Issues: {'; '.join(issues)}"
        return min(score, 1.0), f"All {tools_executed} tools succeeded in {total_duration}ms"

    def _score_synthesis(self, pred: Prediction, gold: Example) -> tuple[float, str]:
        """Score synthesis quality from Phase 4.

        Evaluates:
        - Answer relevance to query
        - Confidence calibration
        - Caveat appropriateness

        Args:
            pred: Prediction with synthesized response
            gold: Example with expected answer characteristics

        Returns:
            Tuple of (score, feedback_message)
        """
        response = getattr(pred, "response", None)

        if not response:
            return 0.0, "CRITICAL: No synthesized response. Check Phase 4 execution"

        answer = getattr(response, "answer", "")

        if not answer:
            return 0.0, "CRITICAL: Empty answer"

        score = 0.0
        issues = []

        # Check answer has content
        if len(answer) >= 50:  # Reasonable answer length
            score += 0.3
        else:
            score += 0.1
            issues.append(f"Short answer ({len(answer)} chars)")

        # Check confidence calibration
        confidence = getattr(response, "confidence", 0)
        execution = getattr(pred, "execution", None)
        if execution:
            success_rate = getattr(execution, "tools_succeeded", 0) / max(
                getattr(execution, "tools_executed", 1), 1
            )

            # Confidence should roughly match success rate
            confidence_delta = abs(confidence - success_rate)
            if confidence_delta <= 0.2:
                score += 0.3
            elif confidence_delta <= 0.4:
                score += 0.15
                issues.append(f"Confidence {confidence:.2f} vs success rate {success_rate:.2f}")
            else:
                issues.append(f"Overconfident: {confidence:.2f} vs {success_rate:.2f}")
        else:
            if 0.3 <= confidence <= 0.9:
                score += 0.2  # Reasonable confidence without execution info

        # Check caveats
        caveats = getattr(response, "caveats", [])
        failed_components = getattr(response, "failed_components", [])

        if failed_components:
            if caveats:
                score += 0.2  # Appropriately noted failures
            else:
                issues.append("Failed components not mentioned in caveats")
        else:
            score += 0.2  # No failures, caveats optional

        # Check supporting data
        supporting_data = getattr(response, "supporting_data", None)
        if supporting_data:
            score += 0.2
        else:
            score += 0.1  # Partial credit

        if issues:
            return min(score, 1.0), f"Confidence={confidence:.2f}. Issues: {'; '.join(issues)}"
        return min(score, 1.0), f"Answer synthesized with confidence={confidence:.2f}"

    def _build_feedback(
        self,
        total: float,
        scores: dict[str, float],
        parts: list[str],
        pred_name: Optional[str],
    ) -> str:
        """Build comprehensive feedback string for GEPA reflection.

        Args:
            total: Overall weighted score
            scores: Component scores dict
            parts: Feedback parts list
            pred_name: Optional predictor name

        Returns:
            Formatted feedback string
        """
        lines = [
            f"Overall: {total:.3f} "
            f"(decomp={scores['decomposition']:.2f}, plan={scores['planning']:.2f}, "
            f"exec={scores['execution']:.2f}, synth={scores['synthesis']:.2f})",
            "",
        ]
        lines.extend(parts)

        # Add optimization suggestions
        lowest_component = min(scores, key=scores.get)
        if scores[lowest_component] < 0.7:
            suggestions = {
                "decomposition": "Improve query decomposition: check intent classification and entity extraction",
                "planning": "Improve tool planning: optimize dependencies and parallel groups",
                "execution": "Improve execution: check error handling and retry logic",
                "synthesis": "Improve synthesis: calibrate confidence and add caveats",
            }
            lines.append(
                f"\n[SUGGESTION] {suggestions.get(lowest_component, 'Review lowest scoring component')}"
            )

        if pred_name:
            lines.append(f"\n[Optimizing predictor: {pred_name}]")

        return "\n".join(lines)


__all__ = ["ToolComposerGEPAMetric"]
