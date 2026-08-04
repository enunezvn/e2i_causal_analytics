"""
E2I Health Score Agent - Score Composer Node
Version: 4.2
Purpose: Compose overall health score from component scores
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, cast

from src.agents.feedback_learner.recipient_emit import emit_recipient_signal

from ..dspy_integration import get_health_score_dspy_integration
from ..metrics import DEFAULT_GRADES, DEFAULT_WEIGHTS, GradeThresholds, ScoreWeights
from ..state import HealthScoreState

logger = logging.getLogger(__name__)


# =============================================================================
# MODEL-QUALITY QUESTIONS (#1450)
#
# Demo 5.3 ("What is the ROC-AUC and calibration of the current Kisqali
# model?") routes correctly to health_score with check_scope="models" — the
# dispatcher's _derive_health_scope already maps the word "model" — and then
# gets a composite grade back, because health_summary is the only field the
# orchestrator's synthesizer reads (AGENT_RESPONSE_FIELDS["health_score"]).
# The reviewer asked for the MEASUREMENT, so when the question names a metric
# the summary leads with the metric values, their cohort and their as-of date;
# the composite line still follows.
#
# Ownership note: this belongs in health_score's models scope rather than a new
# bound chat tool because (a) health_score is the GOLD agent for 5.3 in
# benchmark_queries_gold.jsonl, (b) the /chat/stream path classifies straight to
# an AGENT and never sees the bound-tool set, so a tool would leave the
# reported surface unfixed, and (c) the codebase already assigns this ownership
# in KPI_SEMANTIC_NOTES["WS2-TR-001"]: "model telemetry lives with health_score".
# =============================================================================

# Query phrasings -> the ml_performance_metrics metric_name they name. "psi" has
# no such metric recorded; it is matched so the answer can name it as NOT
# RECORDED rather than silently answering a different question.
_METRIC_QUERY_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"\broc[\s\-_]?auc\b|\bauc[\s\-_]?roc\b|\bauroc\b|\bc[\-\s]statistic\b", "auc_roc"),
    (r"\bpr[\s\-_]?auc\b|\bprecision[\s\-]recall\s+auc\b|\baverage\s+precision\b", "pr_auc"),
    (r"\bcalibrat\w*", "calibration_slope"),
    (r"\bbrier\b", "brier_score"),
    (r"\bpsi\b|\bpopulation\s+stability\b", "psi"),
    (r"\baccuracy\b", "accuracy"),
    (r"\bf1\b|\bf[\s\-]1\s+score\b", "f1"),
    (r"\bprecision\b", "precision"),
    (r"\brecall\b|\bsensitivity\b", "recall"),
    # A bare "auc" last so the more specific ROC-AUC/PR-AUC patterns win first.
    (r"\bauc\b", "auc_roc"),
)

# Tokens that appear in EVERY registered model name (or are generic English) and
# therefore cannot identify which model a question is about. Everything else in a
# model_name — brand ("kisqali"), target ("initiation", "adoption") — can.
_GENERIC_MODEL_NAME_TOKENS = frozenset(
    {
        "goldstd",
        "gold",
        "standard",
        "model",
        "models",
        "logistic",
        "regression",
        "calibrated",
        "classifier",
        "baseline",
        "champion",
        "prod",
        "production",
        "staging",
        "test",
        "train",
    }
)

# Metric names that are ALSO ordinary business words. "What is the recall rate?"
# can mean a product recall, and "accuracy" / "precision" are used loosely about
# forecasts and targeting. These only count as a model-quality request when the
# question is visibly about a model, so an unrelated ask never gets a wall of ML
# metrics. The unambiguous ones (ROC-AUC, Brier, calibration, PSI, PR-AUC, F1)
# need no such qualifier.
_AMBIGUOUS_METRIC_KEYS = frozenset({"accuracy", "precision", "recall"})
_MODEL_CONTEXT_RE = re.compile(r"\bmodels?\b|\bclassifiers?\b|\bpredict\w*\b|\bml\b")

# Upper bound on models enumerated in one answer. A brand names up to 4 models
# today (initiation / persistence / discontinuation / hcp_adoption); the cap
# keeps an unscoped question from dumping the whole fleet into chat.
_MAX_MODEL_METRIC_LINES = 8


def _requested_eval_metrics(query: Optional[str]) -> List[str]:
    """Metric keys the question names, in the order the patterns match them."""
    lowered = (query or "").lower()
    if not lowered:
        return []
    model_context = bool(_MODEL_CONTEXT_RE.search(lowered))
    requested: List[str] = []
    for pattern, key in _METRIC_QUERY_PATTERNS:
        if key in requested or not re.search(pattern, lowered):
            continue
        if key in _AMBIGUOUS_METRIC_KEYS and not model_context:
            continue
        requested.append(key)
    return requested


def _identifying_tokens(model_name: str) -> set:
    """Tokens of a model name that can identify WHICH model a question means."""
    return {
        token
        for token in re.split(r"[^a-z0-9]+", (model_name or "").lower())
        if len(token) >= 4 and token not in _GENERIC_MODEL_NAME_TOKENS
    }


# #1461: these query words name a STAGE, not a model. They never appear in a
# registered model_name ("production"/"champion" are even generic name tokens),
# so token-overlap matching silently ignored them and a question about "the
# current Kisqali model" matched EVERY kisqali-named model, staging included
# (live-verified: demo 5.3 listed initiation_kisqali_goldstd_lr_v1 (staging)
# alongside the production champion).
#
# codex iter-1 HIGH (2026-08-04): a stage the question names EXPLICITLY must
# beat the ambient production reading of "current"/"live" — "the current
# staging Kisqali model" is a staging question, not a production one. Naming
# several stages (a comparison) constrains to that SET of stages — codex
# iter-2 MED: "production and staging" must not let an archived model leak in.
_EXPLICIT_STAGE_RES: Dict[str, "re.Pattern[str]"] = {
    "production": re.compile(r"\bproduction\b|\bprod\b"),
    "staging": re.compile(r"\bstaging\b"),
    "archived": re.compile(r"\barchived\b"),
}
# Currency words imply the production champion only when no explicit stage is
# named (the original #1461 incident: "the current Kisqali model").
_CURRENCY_PRODUCTION_RE = re.compile(r"\bcurrent\b|\blive\b|\bchampion\b")


def _stage_constraint(lowered: str) -> Optional[List[str]]:
    """The stage(s) the question constrains to, or None for no constraint."""
    named = [stage for stage, rx in _EXPLICIT_STAGE_RES.items() if rx.search(lowered)]
    if named:
        return named
    if _CURRENCY_PRODUCTION_RE.search(lowered):
        return ["production"]
    return None


def _models_matching_query(
    query: Optional[str], models: Sequence[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], bool, str]:
    """Models the question names, whether the match was real, and a caller note.

    Returns ``(models, matched, note)``. When no registered model name matches
    the question (a brand that has no model — Xolair is not in the data model at
    all), ``matched`` is False and ALL models are returned so the caller can
    disclose that it is answering about every registered model rather than
    silently attributing another brand's numbers to the one asked about (#1450 —
    unchanged).

    #1461: an explicit stage in the question ("production"/"staging"), or the
    currency words "current"/"live"/"champion" (production, unless an explicit
    stage overrides them), are applied as a STAGE CONSTRAINT before token
    matching. Naming several stages constrains to that SET (a comparison —
    codex iter-2 MED: an archived model must not leak into a
    production-vs-staging question). If any candidate in the constrained
    stage(s) matches the question's identifying tokens, only those are
    answered with. Several candidates in
    one stage for one brand (different prediction targets) are disambiguated
    on the target named in the question — a named target matches more
    identifying tokens than the brand alone; when the question names none, all
    are returned with a non-empty ``note`` for the caller to render, stating
    that several models in that stage exist. When NOTHING in the requested
    stage matches but another stage's model does, that model is returned with
    a note saying the requested stage had no match (codex iter-1 MED — the
    silent fall-through made a production question look answered by a staging
    model).
    """
    lowered = (query or "").lower()
    tokens = set(re.findall(r"[a-z0-9]+", lowered))

    def _overlap(model: Dict[str, Any]) -> set:
        return (
            _identifying_tokens(str(model.get("model_name") or model.get("model_id") or ""))
            & tokens
        )

    stages = _stage_constraint(lowered)
    if stages is not None:
        staged = [m for m in models if str(m.get("model_stage") or "").lower() in stages]
        staged_matched = [m for m in staged if _overlap(m)]
        if len(staged_matched) > 1:
            # Disambiguate on the prediction target: keep the candidate(s)
            # whose names match the MOST query tokens (brand + target beats
            # brand alone). In a multi-stage comparison this keeps one model
            # per compared stage for the named target.
            best = max(len(_overlap(m)) for m in staged_matched)
            staged_matched = [m for m in staged_matched if len(_overlap(m)) == best]
        if len(staged_matched) == 1 or (staged_matched and len(stages) > 1):
            # A multi-stage question is a comparison: several survivors (one
            # per compared stage) are the expected answer, not ambiguity.
            return staged_matched, True, ""
        if staged_matched:
            names = ", ".join(str(m.get("model_name") or m.get("model_id")) for m in staged_matched)
            note = (
                f"Several {stages[0]} models exist for this brand: {names}. The "
                "question does not name a single prediction target, so all of "
                "them are listed above; name the prediction target to narrow "
                "the answer."
            )
            return staged_matched, True, note
        # No candidate in the requested stage(s) matches the question's tokens
        # (e.g. the brand has only staging models): answer with the
        # unconstrained match, but SAY the requested stage had no match.
        matched = [m for m in models if _overlap(m)]
        if matched:
            label = "/".join(stages)
            note = (
                f"No {label}-stage model matches this question; the closest "
                "matching model(s) are listed with their actual stage."
            )
            return matched, True, note
        return list(models), False, ""

    matched = [m for m in models if _overlap(m)]
    if matched:
        return matched, True, ""
    return list(models), False, ""


def _signal_reward(output: str, inputs: Dict[str, Any]) -> float:
    """Deterministic heuristic reward in [0, 1] for an emitted summary signal.

    No randomness, no I/O — same (output, inputs) always yields the same value so
    the optimizer trains on a stable, reproducible reward. Rewards a well-formed,
    informative summary: it must be non-empty, carry the grade + score anchors,
    and resolve to a known status word (not the "unknown" fallback).
    """
    if not output:
        return 0.0
    score = 0.4  # base credit for producing any non-empty summary
    grade = str(inputs.get("grade", ""))
    if grade and f"Grade: {grade}" in output:
        score += 0.2
    if "/100" in output:
        score += 0.2
    # A resolvable status word (excellent/good/fair/poor/critical) signals the
    # grade mapped cleanly; the "unknown" fallback indicates a malformed grade.
    if "unknown" not in output.lower():
        score += 0.2
    return round(min(score, 1.0), 4)


class ScoreComposerNode:
    """
    Compose overall health score from component scores.
    Pure computation - no LLM calls.
    """

    def __init__(
        self,
        weights: Optional[ScoreWeights] = None,
        grades: Optional[GradeThresholds] = None,
    ):
        """
        Initialize score composer.

        Args:
            weights: Custom weights for score components
            grades: Custom grade thresholds
        """
        self.weights = weights or DEFAULT_WEIGHTS
        self.grades = grades or DEFAULT_GRADES

    async def execute(self, state: HealthScoreState) -> HealthScoreState:
        """Compose overall health score."""
        start_time = time.time()

        try:
            # F1 fail-closed: build the composite ONLY from dimensions a real
            # backend actually measured. An unmeasured dimension is excluded
            # entirely (NOT defaulted to a fail-open 1.0). Provenance discloses
            # how many of the four dimensions were measured.
            score_keys: Dict[str, str] = {
                "component": "component_health_score",
                "model": "model_health_score",
                "pipeline": "pipeline_health_score",
                "agent": "agent_health_score",
            }
            measured_flags: Dict[str, bool] = {
                "component": bool(state.get("component_health_measured", False)),
                "model": bool(state.get("model_health_measured", False)),
                "pipeline": bool(state.get("pipeline_health_measured", False)),
                "agent": bool(state.get("agent_health_measured", False)),
            }
            measured_dims = [dim for dim, ok in measured_flags.items() if ok]

            # `scores` is exposed to the diagnosis/analysis helpers below, which
            # treat a missing dimension as 1.0 (healthy) and therefore raise no
            # issues for it — correct, since an unmeasured dim has no findings.
            # cast: TypedDict.get(<dynamic str key>) widens to object; the value
            # is a measured float by construction (the node set it alongside the
            # measured flag).
            scores: Dict[str, float] = {
                dim: cast(float, state.get(score_keys[dim], 1.0)) for dim in measured_dims
            }

            measured_count = len(measured_dims)
            data_provenance: Literal["measured", "partial", "unknown"]
            if measured_count == 4:
                data_provenance = "measured"
            elif measured_count >= 1:
                data_provenance = "partial"
            else:
                data_provenance = "unknown"

            weights_dict = self.weights.to_dict()

            grade: Literal["A", "B", "C", "D", "F"]
            if measured_count == 0:
                # F1 anti-fabrication guard: with nothing measured we must NOT
                # claim a healthy grade-A/100 system. Report a clearly
                # non-healthy UNKNOWN state.
                overall_score = 0.0
                overall_score_100 = 0.0
                grade = "F"
            else:
                # Renormalize the weights over the measured dimensions so a
                # partial measurement is a faithful weighted average of only
                # what was measured (not diluted by absent dims).
                measured_weight_total = sum(weights_dict[dim] for dim in measured_dims)
                if measured_weight_total <= 0:
                    # Degenerate weights (e.g. all-zero) -> simple mean.
                    overall_score = sum(scores[dim] for dim in measured_dims) / measured_count
                else:
                    overall_score = (
                        sum(scores[dim] * weights_dict[dim] for dim in measured_dims)
                        / measured_weight_total
                    )
                overall_score_100 = overall_score * 100
                grade = self.grades.get_grade(overall_score)  # type: ignore[assignment]

            # Identify issues
            critical_issues, warnings = self._identify_issues(state)

            # F1: surface the unknown state as a critical issue so consumers see
            # WHY the score is 0 rather than mistaking it for a real F-grade.
            if measured_count == 0:
                critical_issues = [
                    "No health dimensions could be measured - no real health "
                    "backends are wired (component/model/pipeline/agent). Health "
                    "status is UNKNOWN, not healthy.",
                    *critical_issues,
                ]

            # Generate diagnostic reasoning
            diagnosis = self._generate_diagnosis(state, scores)

            # Generate enhanced summary with diagnosis (via the optimizable
            # template). #1447: provenance and scope are threaded in so an
            # UNMEASURED result is not narrated as a measured critical failure,
            # and a scoped check is not narrated as a whole-system verdict.
            summary = self._generate_summary(
                overall_score_100,
                grade,
                critical_issues,
                warnings,
                data_provenance=data_provenance,
                check_scope=state.get("check_scope"),
            )

            # Add diagnosis insights to summary if there are issues
            if diagnosis["root_causes"]:
                summary += "\n\nDiagnostic Analysis:"
                summary += f"\n- Health Trend: {diagnosis['health_trend'].upper()}"
                if diagnosis["priority_fixes"]:
                    top_fix = diagnosis["priority_fixes"][0]
                    summary += f"\n- Top Priority: {top_fix['action']} ({top_fix['component']})"

            # Emit a recipient training signal for the summary template
            # (best-effort). Deliberately emitted with the SUMMARY-template
            # output only: the #1450 model-quality block below is rendered from
            # its own (unoptimised) templates, so feeding it to the summary
            # signal would train the optimizer on text that template never
            # produced — and its honest "UNKNOWN"/"not recorded" wording would
            # dock ``_signal_reward``'s status-word credit for the wrong reason.
            await self._emit_summary_signal(
                overall_score=overall_score_100,
                grade=grade,
                scores=scores,
                critical_issues=critical_issues,
                summary=summary,
            )

            # #1450: a question naming a metric is answered with the METRIC.
            metrics_block = self._model_metrics_answer(state, measured_flags["model"])
            if metrics_block:
                summary = f"{metrics_block}\n\n{summary}"

            check_time = (state.get("total_latency_ms") or 0) + int(
                (time.time() - start_time) * 1000
            )

            logger.info(
                f"Score composition complete: score={overall_score_100:.1f}, "
                f"grade={grade}, issues={len(critical_issues)}, warnings={len(warnings)}"
            )

            # Ensure errors is always set (required field, v4.3 fix)
            errors = state.get("errors", [])

            return {
                **state,
                "overall_health_score": overall_score_100,
                "health_grade": grade,
                "data_provenance": data_provenance,
                "critical_issues": critical_issues,
                "warnings": warnings,
                "health_summary": summary,
                "health_diagnosis": diagnosis,
                "total_latency_ms": check_time,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "completed",
                "errors": errors,  # Contract-required field
            }

        except Exception as e:
            logger.error(f"Score composition failed: {e}")
            return {
                **state,
                "errors": [{"node": "score_composer", "error": str(e)}],
                "overall_health_score": 0.0,
                "health_grade": "F",
                "data_provenance": "unknown",
                "critical_issues": [f"Score composition failed: {e}"],
                "warnings": [],
                "health_summary": "Unable to compute health score due to an error.",
                "total_latency_ms": state.get("total_latency_ms") or 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "failed",
            }

    def _identify_issues(self, state: HealthScoreState) -> Tuple[List[str], List[str]]:
        """Identify critical issues and warnings."""
        critical = []
        warnings = []

        # Check components
        for comp in state.get("component_statuses") or []:
            if comp["status"] == "unhealthy":
                critical.append(f"Component '{comp['component_name']}' is unhealthy")
            elif comp["status"] == "degraded":
                warnings.append(f"Component '{comp['component_name']}' is degraded")
            elif comp["status"] == "unknown":
                warnings.append(f"Component '{comp['component_name']}' status is unknown")

        # Check models. Prefer the registry name over the UUID — the Alerts tab
        # renders these strings verbatim and "Model '2db8b0e0-...' is degraded"
        # tells a reader nothing.
        for model in state.get("model_metrics") or []:
            model_label = model.get("model_name") or model["model_id"]
            if model["status"] == "unhealthy":
                critical.append(f"Model '{model_label}' is unhealthy")
            elif model["status"] == "degraded":
                warnings.append(f"Model '{model_label}' is degraded")

        # Check pipelines
        for pipeline in state.get("pipeline_statuses") or []:
            if pipeline["status"] == "failed":
                critical.append(f"Pipeline '{pipeline['pipeline_name']}' has failed")
            elif pipeline["status"] == "stale":
                warnings.append(f"Pipeline '{pipeline['pipeline_name']}' data is stale")

        # Check agents
        for agent in state.get("agent_statuses") or []:
            sr = agent["success_rate"]
            if not agent["available"]:
                critical.append(f"Agent '{agent['agent_name']}' is unavailable")
            elif sr is not None and sr < 0.9:
                # Only warn on a MEASURED low rate. A None rate is unmeasured
                # (no recent telemetry) — not a low rate, so no warning.
                warnings.append(f"Agent '{agent['agent_name']}' has low success rate ({sr:.1%})")

        # Check accumulated errors
        for error in state.get("errors") or []:
            node = error.get("node", "unknown")
            msg = error.get("error", "Unknown error")
            warnings.append(f"Error in {node}: {msg}")

        return critical, warnings

    def _generate_summary(
        self,
        score: float,
        grade: str,
        issues: List[str],
        warnings: Optional[List[str]] = None,
        data_provenance: str = "measured",
        check_scope: Optional[str] = None,
    ) -> str:
        """Generate health summary via the optimizable summary template.

        Drop-in for the former inline construction: routes through
        ``HealthScoreDSPyIntegration.get_summary_prompt`` (the previously-dead
        getter) so the optimizable ``summary_template`` is actually consumed. The
        default template renders byte-identically to the historical string for a
        MEASURED full-scope check; ``components=""`` is passed since
        score_composer summaries do not enumerate component names.

        #1447 — the narration seam. The score/grade/counts alone cannot
        distinguish "nothing could be measured" (the deliberate fail-closed
        0.0/grade-"F" payload) from a genuinely measured grade-F catastrophe:
        both used to render as "System health is critical (Grade: F, Score:
        0.0/100). 1 critical issue(s) detected." Passing ``data_provenance``
        selects the UNKNOWN template and hands the ISSUE TEXT (not just its
        count) to the reader; passing ``check_scope`` names what was actually
        checked instead of always claiming "System".
        """
        integration = get_health_score_dspy_integration()
        return integration.get_summary_prompt(
            grade=grade,
            score=score,
            components="",
            critical_count=len(issues),
            warning_count=len(warnings or []),
            data_provenance=data_provenance,
            check_scope=check_scope,
            critical_issues=issues,
        )

    def _model_metrics_answer(self, state: HealthScoreState, model_measured: bool) -> str:
        """Answer a metric-naming question with the METRICS (#1450), or "".

        Returns the empty string when the question names no metric — the
        composite summary is then rendered exactly as before (regression pin:
        ``TestNonMetricQueriesAreUnchanged``).

        Honesty rules, in order:
          * model dimension NOT measured -> the unavailable template. No number
            is printed at all; the #1447 UNKNOWN summary follows it.
          * no registered model name matches the question -> every registered
            model is listed WITH that disclosure, so another brand's numbers are
            never silently attributed to the one asked about.
          * a requested metric that is not recorded -> named as not recorded.
        """
        requested = _requested_eval_metrics(state.get("query"))
        if not requested:
            return ""
        scope = state.get("check_scope")
        if scope not in ("full", "models"):
            # The models dimension was deliberately out of scope for this check;
            # it was not "not measured" so much as "not asked for".
            return ""

        integration = get_health_score_dspy_integration()
        models = list(state.get("model_metrics") or [])
        if not model_measured or not models:
            reason = (
                "the model health dimension was not measured (no metrics store is "
                "reachable), so no evaluation metric could be read."
            )
            return integration.get_model_metrics_prompt(requested, [], unavailable_reason=reason)

        matched, was_matched, stage_note = _models_matching_query(state.get("query"), models)
        truncated = len(matched) > _MAX_MODEL_METRIC_LINES
        block = integration.get_model_metrics_prompt(requested, matched[:_MAX_MODEL_METRIC_LINES])
        if stage_note:
            # #1461: several production models exist for the brand and the
            # question named no prediction target — say so explicitly.
            block += f"\n{stage_note}"
        if not was_matched:
            block += (
                "\nNo registered model name matched the question, so every registered "
                f"model is listed above ({len(models)} in total) rather than "
                "attributing another model's numbers to the one asked about."
            )
        if truncated:
            noun = "matching" if was_matched else "registered"
            block += (
                f"\nShowing {_MAX_MODEL_METRIC_LINES} of {len(matched)} {noun} models; "
                "name the brand and prediction target to narrow it."
            )
        return block

    async def _emit_summary_signal(
        self,
        overall_score: float,
        grade: str,
        scores: Dict[str, float],
        critical_issues: List[str],
        summary: str,
    ) -> None:
        """Emit ONE recipient training signal for the summary template.

        Best-effort: a persistence failure must never break score composition.
        ``signature_inputs`` is keyed by ``HealthSummarySignature.input_fields``
        (the explicit emit<->provider contract from
        ``recipient_required_input_keys('health_score')['summary_template']``):
        ``overall_score, grade, component_scores, critical_issues``. Only fields
        backed by real, fully-populated node data are emitted.

        #1447 corollary: a zero-measured run has an EMPTY ``scores`` dict, so
        ``component_scores`` is "" and the populated-fields guard below returns
        before emitting. The UNKNOWN narration therefore never reaches
        ``_signal_reward`` (whose heuristic docks a summary containing
        "unknown", a signal that only makes sense for the MEASURED template).
        """
        try:
            component_scores = ", ".join(f"{dim}={val:.2f}" for dim, val in scores.items())
            critical_repr = "; ".join(critical_issues) if critical_issues else "None"
            signature_inputs: Dict[str, Any] = {
                "overall_score": float(overall_score),
                "grade": grade,
                "component_scores": component_scores,
                "critical_issues": critical_repr,
            }
            # Emit only when every contract field is populated.
            if not all(v not in (None, "") for v in signature_inputs.values()):
                return
            reward = _signal_reward(summary, signature_inputs)
            await emit_recipient_signal(
                agent_name="health_score",
                signature_inputs=signature_inputs,
                generated_output=summary,
                reward=reward,
                template_field="summary_template",
            )
        except Exception as e:  # noqa: BLE001 - emission is best-effort, never fail the run
            logger.warning("health_score recipient signal emission skipped: %s", e)

    def _generate_diagnosis(self, state: HealthScoreState, scores: dict) -> dict:
        """Generate diagnostic reasoning for health issues.

        Provides deeper analysis of root causes, cascading effects,
        and prioritized remediation steps.

        Args:
            state: Current health score state
            scores: Component scores dictionary

        Returns:
            Diagnosis dictionary with root causes and remediation
        """
        diagnosis: dict[str, Any] = {
            "root_causes": [],
            "cascading_effects": [],
            "priority_fixes": [],
            "health_trend": "stable",
            "confidence": "high",
        }

        # Analyze each dimension for issues
        component_issues = self._analyze_component_health(state, scores)
        model_issues = self._analyze_model_health(state, scores)
        pipeline_issues = self._analyze_pipeline_health(state, scores)
        agent_issues = self._analyze_agent_health(state, scores)

        # Collect all root causes
        all_issues = component_issues + model_issues + pipeline_issues + agent_issues

        # Sort by impact (critical first)
        all_issues.sort(key=lambda x: x.get("impact_score", 0), reverse=True)
        diagnosis["root_causes"] = all_issues[:5]

        # Identify cascading effects
        diagnosis["cascading_effects"] = self._find_cascading_effects(all_issues)

        # Generate prioritized fixes
        diagnosis["priority_fixes"] = self._prioritize_fixes(all_issues)

        # Determine health trend
        if len(all_issues) > 3:
            diagnosis["health_trend"] = "degrading"
        elif len(all_issues) == 0:
            diagnosis["health_trend"] = "healthy"

        return diagnosis

    def _analyze_component_health(self, state: HealthScoreState, scores: dict) -> List[dict]:
        """Analyze component health issues."""
        issues = []
        component_score = scores.get("component", 1.0)

        # Analyze if score is below healthy threshold (0.8 aligns with quality gate validation)
        if component_score < 0.8:
            component_statuses = state.get("component_statuses") or []
            for comp in component_statuses:
                if comp["status"] in ("unhealthy", "degraded"):
                    issue = {
                        "dimension": "component",
                        "component": comp["component_name"],
                        "status": comp["status"],
                        "root_cause": self._infer_component_root_cause(dict(comp)),
                        "metrics": {
                            "latency_ms": comp.get("latency_ms"),
                            "error_message": comp.get("error_message"),
                        },
                        "impact_score": 1.0 if comp["status"] == "unhealthy" else 0.5,
                    }
                    issues.append(issue)

            # If score is degraded but no specific component issues found,
            # create a synthetic root cause for quality gate compliance
            if not issues:
                issues.append(
                    {
                        "dimension": "component",
                        "component": "system_aggregate",
                        "status": "degraded",
                        "root_cause": (
                            f"Component health score ({component_score:.1%}) below healthy threshold (80%). "
                            f"Multiple minor issues may be contributing to overall degradation."
                        ),
                        "metrics": {"aggregate_score": component_score},
                        "impact_score": 0.5,
                    }
                )

        return issues

    def _analyze_model_health(self, state: HealthScoreState, scores: dict) -> List[dict]:
        """Analyze model health issues."""
        issues = []
        model_score = scores.get("model", 1.0)

        if model_score < 0.8:
            for model in state.get("model_metrics") or []:
                if model["status"] in ("unhealthy", "degraded"):
                    accuracy = model.get("accuracy")
                    # error_rate may be None (UNMEASURED — the dashboard sources
                    # status but not error_rate). Guard the comparison/format so a
                    # null sub-field never crashes diagnosis (which would turn a
                    # real partial measurement into a failed composite).
                    error_rate = model.get("error_rate")

                    root_cause = "Unknown model issue"
                    if accuracy is not None and accuracy < 0.7:
                        root_cause = f"Model accuracy ({accuracy:.1%}) below threshold"
                    elif error_rate is not None and error_rate > 0.1:
                        root_cause = f"High error rate ({error_rate:.1%})"

                    issue = {
                        "dimension": "model",
                        "component": model.get("model_name") or model["model_id"],
                        "status": model["status"],
                        "root_cause": root_cause,
                        "metrics": {
                            "accuracy": accuracy,
                            "precision": model.get("precision"),
                            "recall": model.get("recall"),
                            "error_rate": error_rate,
                        },
                        "impact_score": 0.8 if model["status"] == "unhealthy" else 0.4,
                    }
                    issues.append(issue)

        return issues

    def _analyze_pipeline_health(self, state: HealthScoreState, scores: dict) -> List[dict]:
        """Analyze pipeline health issues."""
        issues = []
        pipeline_score = scores.get("pipeline", 1.0)

        if pipeline_score < 0.9:
            for pipeline in state.get("pipeline_statuses") or []:
                if pipeline["status"] in ("failed", "stale"):
                    freshness = pipeline.get("freshness_hours", 0)

                    if pipeline["status"] == "failed":
                        root_cause = "Pipeline execution failed"
                    elif freshness > 24:
                        root_cause = f"Data stale by {freshness:.1f} hours (>24h threshold)"
                    else:
                        root_cause = f"Data freshness degraded ({freshness:.1f} hours)"

                    issue = {
                        "dimension": "pipeline",
                        "component": pipeline["pipeline_name"],
                        "status": pipeline["status"],
                        "root_cause": root_cause,
                        "metrics": {
                            "freshness_hours": freshness,
                            "rows_processed": pipeline.get("rows_processed"),
                            "last_success": pipeline.get("last_success"),
                        },
                        "impact_score": 0.9 if pipeline["status"] == "failed" else 0.3,
                    }
                    issues.append(issue)

        return issues

    def _analyze_agent_health(self, state: HealthScoreState, scores: dict) -> List[dict]:
        """Analyze agent health issues."""
        issues = []
        agent_score = scores.get("agent", 1.0)

        if agent_score < 0.9:
            for agent in state.get("agent_statuses") or []:
                sr = agent["success_rate"]
                # A None success_rate is unmeasured (no recent telemetry), NOT a
                # measured low rate — it raises no issue on its own.
                if not agent["available"] or (sr is not None and sr < 0.9):
                    if not agent["available"]:
                        root_cause = "Agent unavailable - may be down or unreachable"
                    else:
                        root_cause = f"Low success rate ({sr:.1%})"

                    issue = {
                        "dimension": "agent",
                        "component": agent["agent_name"],
                        "status": "unavailable" if not agent["available"] else "degraded",
                        "root_cause": root_cause,
                        "metrics": {
                            "tier": agent.get("tier"),
                            "success_rate": agent.get("success_rate"),
                            "avg_latency_ms": agent.get("avg_latency_ms"),
                        },
                        "impact_score": 0.7 if not agent["available"] else 0.3,
                    }
                    issues.append(issue)

        return issues

    def _infer_component_root_cause(self, comp: dict) -> str:
        """Infer root cause for component issues."""
        component_name = comp.get("component_name", "").lower()
        error_msg = comp.get("error_message", "")
        latency = comp.get("latency_ms")

        if "database" in component_name or "db" in component_name:
            if latency and latency > 1000:
                return "Database connection slow - possible connection pool exhaustion"
            elif error_msg:
                return f"Database error: {error_msg[:100]}"
            return "Database connectivity issue"

        elif "cache" in component_name or "redis" in component_name:
            if error_msg and "connection" in error_msg.lower():
                return "Cache server connection refused - may need restart"
            return "Cache service degraded"

        elif "api" in component_name:
            if latency and latency > 5000:
                return "API response time critical - check downstream dependencies"
            return "API service degraded"

        elif "queue" in component_name or "message" in component_name:
            return "Message queue backlog or connectivity issue"

        return f"Component degraded: {error_msg[:100] if error_msg else 'Unknown cause'}"

    def _find_cascading_effects(self, issues: List[dict]) -> List[str]:
        """Identify cascading effects from root causes."""
        effects = []

        # Check for database issues affecting other components
        db_issues = [i for i in issues if "database" in i.get("component", "").lower()]
        if db_issues:
            effects.append(
                "Database issues may cause failures in agents, pipelines, and API endpoints"
            )

        # Check for pipeline issues affecting model freshness
        pipeline_issues = [i for i in issues if i.get("dimension") == "pipeline"]
        if pipeline_issues:
            effects.append(
                "Stale pipelines mean models are operating on outdated data - predictions may be unreliable"
            )

        # Check for model issues affecting agent reliability
        model_issues = [i for i in issues if i.get("dimension") == "model"]
        if model_issues:
            effects.append(
                "Degraded model accuracy affects all downstream agents relying on predictions"
            )

        # Multiple component failures
        if len(issues) > 3:
            effects.append(
                f"Multiple simultaneous issues ({len(issues)}) suggest potential infrastructure problem"
            )

        return effects[:3]

    def _prioritize_fixes(self, issues: List[dict]) -> List[dict]:
        """Generate prioritized list of fixes."""
        fixes = []

        # Define fix templates
        fix_templates: dict[str, Any] = {
            "component": {
                "database": "Check database connection pool and consider restart",
                "cache": "Verify Redis service status and memory usage",
                "api": "Review API logs and check downstream service health",
                "default": "Investigate component logs and restart if necessary",
            },
            "model": "Evaluate model on recent data and consider retraining",
            "pipeline": "Check pipeline logs, verify data sources, and re-run",
            "agent": "Check agent logs, verify dependencies, and restart service",
        }

        for i, issue in enumerate(issues[:5]):
            dimension = issue.get("dimension", "unknown")
            component = issue.get("component", "").lower()

            if dimension == "component":
                if "database" in component:
                    fix_action = fix_templates["component"]["database"]
                elif "cache" in component or "redis" in component:
                    fix_action = fix_templates["component"]["cache"]
                elif "api" in component:
                    fix_action = fix_templates["component"]["api"]
                else:
                    fix_action = fix_templates["component"]["default"]
            else:
                fix_action = fix_templates.get(dimension, "Investigate and remediate")

            fix = {
                "priority": i + 1,
                "component": issue.get("component"),
                "issue": issue.get("root_cause"),
                "action": fix_action,
                "estimated_impact": "high" if issue.get("impact_score", 0) > 0.7 else "medium",
            }
            fixes.append(fix)

        return fixes
