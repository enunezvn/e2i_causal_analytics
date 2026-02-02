"""
Rubric Evaluator for E2I Causal Analytics.

Evaluates chatbot responses against the causal analytics rubric
and determines appropriate improvement actions.

Integration points:
- Called by Reflector Node after response delivery
- Stores results in learning_signals table
- Triggers feedback_learner for improvements when needed

Configuration:
- Can load criteria and thresholds from config/self_improvement.yaml
- Or use programmatic overrides
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

# anthropic is optional - graceful degradation when not available
try:
    import anthropic

    _has_anthropic = True
except ImportError:
    anthropic = None  # type: ignore
    _has_anthropic = False

from src.agents.feedback_learner.evaluation.criteria import (
    DEFAULT_OVERRIDE_CONDITIONS,
    DEFAULT_THRESHOLDS,
    RubricCriterion,
    get_default_criteria,
)
from src.agents.feedback_learner.evaluation.models import (
    CriterionScore,
    EvaluationContext,
    ImprovementDecision,
    PatternFlag,
    RubricEvaluation,
)

logger = logging.getLogger(__name__)


def _load_criteria_from_config() -> Optional[List[RubricCriterion]]:
    """
    Load rubric criteria from config file.

    Returns:
        List of RubricCriterion objects or None if config unavailable.
    """
    try:
        from src.agents.feedback_learner.config import load_self_improvement_config

        config = load_self_improvement_config()

        criteria = []
        for name, criterion_config in config.rubric.criteria.items():
            criteria.append(
                RubricCriterion(
                    name=name,
                    weight=criterion_config.weight,
                    description=criterion_config.description,
                    scoring_guide=criterion_config.scoring_guide,
                )
            )

        logger.info(f"Loaded {len(criteria)} criteria from config")
        return criteria if criteria else None

    except Exception as e:
        logger.debug(f"Could not load criteria from config: {e}")
        return None


def _load_thresholds_from_config() -> Optional[Dict[str, float]]:
    """
    Load decision thresholds from config file.

    Returns:
        Dictionary of thresholds or None if config unavailable.
    """
    try:
        from src.agents.feedback_learner.config import load_self_improvement_config

        config = load_self_improvement_config()
        return config.decision_framework.thresholds

    except Exception as e:
        logger.debug(f"Could not load thresholds from config: {e}")
        return None


class RubricEvaluator:
    """
    Evaluates E2I chatbot responses against the causal analytics rubric.

    Uses AI-as-judge methodology with structured scoring across 5 criteria:
    - causal_validity (0.25)
    - actionability (0.25)
    - evidence_chain (0.20)
    - regulatory_awareness (0.15)
    - uncertainty_communication (0.15)

    Example:
        >>> evaluator = RubricEvaluator()
        >>> context = EvaluationContext(
        ...     user_query="Why did Kisqali adoption increase?",
        ...     agent_outputs={"causal_impact": {"effect": 0.23}},
        ...     final_response="Kisqali adoption increased by 23%..."
        ... )
        >>> result = await evaluator.evaluate(context)
        >>> print(f"Score: {result.weighted_score}, Decision: {result.decision}")
    """

    def __init__(
        self,
        criteria: Optional[List[RubricCriterion]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        override_conditions: Optional[List[Dict[str, Any]]] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1500,
        temperature: float = 0.3,
        use_config: bool = True,
    ):
        """
        Initialize evaluator with rubric configuration.

        Args:
            criteria: List of rubric criteria. If None and use_config=True,
                     loads from config/self_improvement.yaml. Falls back to defaults.
            thresholds: Decision thresholds. If None and use_config=True,
                       loads from config. Falls back to DEFAULT_THRESHOLDS.
            override_conditions: Override conditions. Defaults to DEFAULT_OVERRIDE_CONDITIONS.
            model: Anthropic model to use for evaluation.
            max_tokens: Max tokens for evaluation response.
            temperature: Temperature for evaluation (low for consistency).
            use_config: If True, attempt to load criteria/thresholds from config file.
        """
        # Load criteria: explicit > config > defaults
        if criteria is not None:
            self.criteria = criteria
        elif use_config:
            config_criteria = _load_criteria_from_config()
            self.criteria = config_criteria if config_criteria else get_default_criteria()
        else:
            self.criteria = get_default_criteria()

        # Load thresholds: explicit > config > defaults
        if thresholds is not None:
            self.thresholds = thresholds
        elif use_config:
            config_thresholds = _load_thresholds_from_config()
            self.thresholds = config_thresholds if config_thresholds else DEFAULT_THRESHOLDS.copy()
        else:
            self.thresholds = DEFAULT_THRESHOLDS.copy()

        self.override_conditions = (
            override_conditions
            if override_conditions is not None
            else DEFAULT_OVERRIDE_CONDITIONS.copy()
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Initialize Anthropic client (uses ANTHROPIC_API_KEY env var)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if _has_anthropic and api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            self.client = None
            if not _has_anthropic:
                logger.warning(
                    "anthropic package not installed, RubricEvaluator will use fallback scoring"
                )
            else:
                logger.warning(
                    "No ANTHROPIC_API_KEY found, RubricEvaluator will use fallback scoring"
                )

    async def evaluate(self, context: EvaluationContext) -> RubricEvaluation:
        """
        Evaluate a response against the causal analytics rubric.

        Args:
            context: Evaluation context with query, outputs, and response.

        Returns:
            RubricEvaluation with scores, decision, and suggestions.
        """
        logger.info(
            "Starting rubric evaluation for session=%s with agents=%s",
            context.session_id,
            context.agent_names,
        )

        # Get criterion scores (from AI or fallback)
        if self.client:
            criterion_scores, overall_analysis = await self._evaluate_with_ai(context)
        else:
            criterion_scores, overall_analysis = self._fallback_evaluation()

        # Calculate weighted score
        weighted_score = self._calculate_weighted_score(criterion_scores)

        # Determine decision
        decision = self._determine_decision(weighted_score, criterion_scores)

        # Detect patterns
        pattern_flags = self._detect_patterns(criterion_scores)

        # Generate improvement suggestion if needed
        improvement_suggestion = None
        if decision in [ImprovementDecision.SUGGESTION, ImprovementDecision.AUTO_UPDATE]:
            improvement_suggestion = await self._generate_improvement_suggestion(
                context, criterion_scores, decision
            )

        evaluation = RubricEvaluation(
            weighted_score=weighted_score,
            criterion_scores=criterion_scores,
            decision=decision,
            overall_analysis=overall_analysis,
            pattern_flags=pattern_flags,
            improvement_suggestion=improvement_suggestion,
        )

        logger.info(
            "Rubric evaluation complete: weighted_score=%.2f decision=%s patterns=%d",
            weighted_score,
            decision.value,
            len(pattern_flags),
        )

        return evaluation

    async def _evaluate_with_ai(
        self, context: EvaluationContext
    ) -> tuple[List[CriterionScore], str]:
        """Evaluate using AI-as-judge methodology."""
        eval_prompt = self._build_evaluation_prompt(context)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": eval_prompt}],
            )

            response_text = response.content[0].text
            return self._parse_evaluation_response(response_text)

        except Exception as e:
            # Handle anthropic.APIError specifically when available
            if _has_anthropic and isinstance(e, anthropic.APIError):
                logger.error("Anthropic API error during evaluation: %s", e)
            else:
                logger.error("Error during AI evaluation: %s", e)
            return self._fallback_evaluation()

    def _build_evaluation_prompt(self, context: EvaluationContext) -> str:
        """Build the AI-as-judge evaluation prompt."""
        criteria_text = "\n".join(
            [
                f"""
**{c.name.replace("_", " ").title()}** (Weight: {c.weight * 100:.0f}%)
{c.description}
Scoring Guide:
- 5: {c.scoring_guide.get(5, "Excellent")}
- 4: {c.scoring_guide.get(4, "Good")}
- 3: {c.scoring_guide.get(3, "Acceptable")}
- 2: {c.scoring_guide.get(2, "Poor")}
- 1: {c.scoring_guide.get(1, "Very Poor")}
"""
                for c in self.criteria
            ]
        )

        agent_info = ", ".join(context.agent_names) if context.agent_names else "Orchestrator"

        return f"""You are an expert evaluator for an E2I Causal Analytics chatbot used in pharmaceutical commercial operations.

Evaluate the following response against each criterion and provide scores (1-5) with reasoning.

## EVALUATION CRITERIA
{criteria_text}

## CONTEXT
**User Query:** {context.user_query}

**Agent Processing:** {agent_info}

**Final Response:**
{context.final_response}

## EVALUATION TASK
For each criterion, provide:
1. Score (1-5, can be decimal like 3.5)
2. Brief reasoning (1-2 sentences)
3. Specific evidence from the response

Format your response as JSON:
```json
{{
    "causal_validity": {{"score": X, "reasoning": "...", "evidence": "..."}},
    "actionability": {{"score": X, "reasoning": "...", "evidence": "..."}},
    "evidence_chain": {{"score": X, "reasoning": "...", "evidence": "..."}},
    "regulatory_awareness": {{"score": X, "reasoning": "...", "evidence": "..."}},
    "uncertainty_communication": {{"score": X, "reasoning": "...", "evidence": "..."}},
    "overall_analysis": "Brief summary of key strengths and weaknesses"
}}
```"""

    def _parse_evaluation_response(self, response_text: str) -> tuple[List[CriterionScore], str]:
        """Parse AI evaluation response into structured scores."""
        try:
            # Find JSON block
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            json_str = response_text[json_start:json_end]
            parsed = json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to parse evaluation response: %s", e)
            return self._fallback_evaluation()

        scores = []
        for criterion in self.criteria:
            if criterion.name in parsed:
                data = parsed[criterion.name]
                score_val = data.get("score", 3)
                # Clamp score to valid range
                score_val = max(1.0, min(5.0, float(score_val)))
                scores.append(
                    CriterionScore(
                        criterion=criterion.name,
                        score=score_val,
                        reasoning=data.get("reasoning", ""),
                        evidence=data.get("evidence"),
                    )
                )
            else:
                scores.append(
                    CriterionScore(
                        criterion=criterion.name,
                        score=3.0,
                        reasoning="Not evaluated",
                    )
                )

        overall_analysis = parsed.get("overall_analysis", self._summarize_evaluation(scores))

        return scores, overall_analysis

    def _fallback_evaluation(self) -> tuple[List[CriterionScore], str]:
        """Return neutral scores when AI evaluation is unavailable."""
        logger.info("Using fallback evaluation (neutral scores)")
        scores = [
            CriterionScore(
                criterion=c.name,
                score=3.0,
                reasoning="Fallback evaluation - AI unavailable",
            )
            for c in self.criteria
        ]
        return scores, "Fallback evaluation used - AI unavailable"

    def _calculate_weighted_score(self, scores: List[CriterionScore]) -> float:
        """Calculate weighted average score."""
        total = 0.0
        for score in scores:
            criterion = next((c for c in self.criteria if c.name == score.criterion), None)
            if criterion:
                total += score.score * criterion.weight
        return round(total, 2)

    def _determine_decision(
        self, weighted_score: float, scores: List[CriterionScore]
    ) -> ImprovementDecision:
        """Determine action based on scores and override conditions."""
        # Check override conditions first
        for condition in self.override_conditions:
            if condition["condition"] == "any_criterion_below":
                if any(s.score < condition["threshold"] for s in scores):
                    action = condition.get("action", "suggestion")
                    if action == "suggestion":
                        return ImprovementDecision.SUGGESTION
                    elif action == "escalate":
                        return ImprovementDecision.ESCALATE

        # Standard threshold-based decision
        if weighted_score >= self.thresholds["acceptable"]:
            return ImprovementDecision.ACCEPTABLE
        elif weighted_score >= self.thresholds["suggestion"]:
            return ImprovementDecision.SUGGESTION
        elif weighted_score >= self.thresholds["auto_update"]:
            return ImprovementDecision.AUTO_UPDATE
        else:
            return ImprovementDecision.ESCALATE

    def _detect_patterns(self, scores: List[CriterionScore]) -> List[PatternFlag]:
        """Detect recurring weakness patterns."""
        patterns = []

        for score in scores:
            if score.score < 3.0:
                patterns.append(
                    PatternFlag(
                        pattern_type=f"low_{score.criterion}",
                        score=score.score,
                        reasoning=score.reasoning,
                        criterion=score.criterion,
                    )
                )

        return patterns

    async def _generate_improvement_suggestion(
        self,
        context: EvaluationContext,
        scores: List[CriterionScore],
        decision: ImprovementDecision,
    ) -> Optional[str]:
        """Generate improvement suggestion based on weaknesses."""
        if not self.client:
            return None

        weak_criteria = [s for s in scores if s.score < 3.5]
        if not weak_criteria:
            return None

        weak_list = "\n".join(
            [f"- {s.criterion}: {s.score}/5 - {s.reasoning}" for s in weak_criteria]
        )

        prompt = f"""Based on this evaluation of an E2I Causal Analytics chatbot response:

**Weaknesses Identified:**
{weak_list}

**Original Query:** {context.user_query}

Generate a specific improvement to the system prompt that would address these weaknesses.

Focus on:
1. What specific instruction to add
2. What behavior to encourage
3. What pattern to avoid

Keep the improvement concise (2-3 sentences max)."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.5,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()

        except Exception as e:
            # Handle anthropic.APIError specifically when available
            if _has_anthropic and isinstance(e, anthropic.APIError):
                logger.error("Failed to generate improvement suggestion: %s", e)
            else:
                logger.error("Error generating improvement suggestion: %s", e)
            return None

    def _summarize_evaluation(self, scores: List[CriterionScore]) -> str:
        """Generate overall summary of evaluation."""
        strengths = [s for s in scores if s.score >= 4.0]
        weaknesses = [s for s in scores if s.score < 3.0]

        summary_parts = []

        if strengths:
            summary_parts.append(
                f"Strengths: {', '.join(s.criterion.replace('_', ' ') for s in strengths)}"
            )

        if weaknesses:
            summary_parts.append(
                f"Weaknesses: {', '.join(s.criterion.replace('_', ' ') for s in weaknesses)}"
            )

        return " | ".join(summary_parts) if summary_parts else "Adequate across all criteria"

    def evaluate_sync(self, context: EvaluationContext) -> RubricEvaluation:
        """
        Synchronous wrapper for evaluate().

        Use this when calling from synchronous code.
        """
        import asyncio

        return asyncio.run(self.evaluate(context))
