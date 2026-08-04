"""#1460 + #1461 — health narration honesty on the model-quality render path.

Four behaviors, one render path (``ScoreComposerNode`` ->
``HealthScoreDSPyIntegration``):

#1460 (A) — a SCOPED check must not narrate "All systems operational."
    ``get_summary_prompt`` derives ``issue_clause`` without consulting
    ``check_scope``: with zero critical issues it asserts whole-system health
    even when only ONE dimension was evaluated (``model_health.py`` measures
    only the model dimension for ``scope="models"``). Reachable from chat:
    "How healthy are the models?" routes ``scope="models"``.

#1460 (B) — a requested metric missing for ONE model must be disclosed on THAT
    model's line. ``get_model_metrics_prompt`` tracked ``reported`` globally
    across all models, so if model A records ``brier_score`` and model B does
    not, B's silent omission was indistinguishable from "fine".

#1461 — "current"/"production"/"live"/"champion" in the question are a STAGE
    CONSTRAINT, not ignorable noise: they never appear in model names, so
    token-overlap matching returned EVERY brand model including staging ones
    (live-verified demo 5.3: ``initiation_kisqali_goldstd_lr_v1 (staging)``
    listed alongside the production champion). When the constraint still
    leaves several production models for the brand, the answer disambiguates
    on the prediction target named in the question — or states explicitly
    that several production models exist and lists them.

The #1450 no-match disclosure path (``_models_matching_query`` returning
``(all_models, matched=False)`` + caller disclosure) is pinned UNCHANGED.

Offline only: real composer, real integration templates, real matcher —
``emit_recipient_signal`` patched exactly as the #1447/#1450 tests do.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.health_score.dspy_integration import HealthScoreDSPyIntegration
from src.agents.health_score.nodes.score_composer import (
    ScoreComposerNode,
    _models_matching_query,
)

EMIT_TARGET = "src.agents.health_score.nodes.score_composer.emit_recipient_signal"

DEMO_53 = "What is the ROC-AUC and calibration of the current Kisqali model?"


def _model(
    name: str,
    stage: str,
    eval_metrics: Dict[str, float],
    model_id: str = "00000000-0000-0000-0000-000000000000",
    eval_sample_size: Optional[int] = 1000,
    eval_as_of: Optional[str] = "2026-06-01T00:00:00+00:00",
) -> Dict[str, Any]:
    """One ``ModelMetrics``-shaped entry (mirrors the #1450 fixture shapes)."""
    return {
        "model_id": model_id,
        "model_name": name,
        "model_version": "1.0",
        "model_stage": stage,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "auc_roc": eval_metrics.get("auc_roc"),
        "prediction_latency_p50_ms": None,
        "prediction_latency_p99_ms": None,
        "predictions_last_24h": None,
        "error_rate": None,
        "status": "healthy",
        "eval_metrics": dict(eval_metrics),
        "eval_cohort": "holdout",
        "eval_sample_size": eval_sample_size,
        "eval_as_of": eval_as_of,
    }


# Live-verified shapes from demo 5.3 (2026-08-03): the production champion and
# the staging model that was wrongly listed alongside it (#1461).
KISQALI_PROD = _model(
    "hcp_adoption_kisqali_goldstd_lr_v1",
    "production",
    {"auc_roc": 0.767697, "calibration_slope": 0.925305, "brier_score": 0.190392},
    model_id="8d1244df-7c38-435e-820f-1f201e51af24",
)
KISQALI_STAGING = _model(
    "initiation_kisqali_goldstd_lr_v1",
    "staging",
    {"auc_roc": 0.851908, "calibration_slope": 1.049494, "brier_score": 0.145714},
    model_id="4ec55d13-46c8-4df4-9ec8-7723fad67fb3",
    eval_sample_size=1645,
    eval_as_of="2026-07-21T00:00:00+00:00",
)


def _models_state(query: str, models: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A models-scoped check where the model dimension WAS measured."""
    return {
        "query": query,
        "check_scope": "models",
        "model_metrics": models,
        "model_health_score": 1.0,
        "model_health_measured": True,
        "total_latency_ms": 0,
        "errors": [],
    }


async def _compose(state: Dict[str, Any]) -> Dict[str, Any]:
    node = ScoreComposerNode()
    with patch(EMIT_TARGET, new=AsyncMock(return_value=None)):
        return await node.execute(state)


def _line_naming(summary: str, model_name: str) -> str:
    """The single rendered model line for ``model_name``."""
    lines = [line for line in summary.splitlines() if model_name in line]
    assert len(lines) == 1, f"expected exactly one line for {model_name}: {summary!r}"
    return lines[0]


def _metric_segment(line: str) -> str:
    """The metric_list part of a model line (before the provenance clause,
    which legitimately says "not recorded" for absent cohort fields)."""
    return line.split("[")[0]


# =============================================================================
# #1460 (A) — scoped checks must not narrate "All systems operational."
# =============================================================================


class TestScopedNoIssueNarration1460A:
    @pytest.mark.parametrize(
        "scope,expected",
        [
            ("models", "No model health issues detected."),
            ("pipelines", "No pipeline health issues detected."),
            ("agents", "No agent health issues detected."),
            ("quick", "No component health issues detected."),
        ],
    )
    def test_scoped_no_issue_clause_names_the_dimension(self, scope: str, expected: str):
        """A scope="models" check measured ONLY the model dimension
        (model_health.py); asserting "All systems operational." claims health
        for dimensions never evaluated."""
        integration = HealthScoreDSPyIntegration()
        rendered = integration.get_summary_prompt(
            grade="A",
            score=100.0,
            components="",
            critical_count=0,
            warning_count=0,
            check_scope=scope,
        )
        assert "All systems operational" not in rendered, rendered
        assert expected in rendered, rendered

    @pytest.mark.parametrize("scope", ["full", None])
    def test_full_or_absent_scope_keeps_all_systems_operational(self, scope: Optional[str]):
        """Only a full (or absent-scope, historical-default) check evaluated
        every dimension, so only it may make the whole-system claim."""
        integration = HealthScoreDSPyIntegration()
        rendered = integration.get_summary_prompt(
            grade="A",
            score=100.0,
            components="",
            critical_count=0,
            warning_count=0,
            check_scope=scope,
        )
        assert "All systems operational." in rendered, rendered

    def test_critical_clause_unchanged_for_scoped_checks(self):
        """The with-issues clause was never the defect — pin it."""
        integration = HealthScoreDSPyIntegration()
        rendered = integration.get_summary_prompt(
            grade="F",
            score=30.0,
            components="",
            critical_count=2,
            warning_count=0,
            check_scope="models",
        )
        assert "2 critical issue(s) detected." in rendered, rendered

    @pytest.mark.asyncio
    async def test_models_scope_summary_is_scope_aware_end_to_end(self):
        """Chat-reachable shape: "How healthy are the models?" -> scope="models"
        via the orchestrator dispatcher; no metric named, so the plain summary
        renders — and must not assert whole-system health."""
        state = _models_state("How healthy are the models?", [KISQALI_PROD])
        summary = (await _compose(state))["health_summary"]
        assert summary.startswith("Model health is excellent"), summary
        assert "All systems operational" not in summary, summary
        assert "No model health issues detected." in summary, summary


# =============================================================================
# #1460 (B) — requested-metric omissions are disclosed PER MODEL LINE
# =============================================================================


class TestPerModelMissingMetricDisclosure1460B:
    KISQALI_NO_BRIER = _model(
        "initiation_kisqali_goldstd_lr_v1",
        "staging",
        {"auc_roc": 0.851908, "calibration_slope": 1.049494},
        model_id="4ec55d13-46c8-4df4-9ec8-7723fad67fb3",
        eval_sample_size=1645,
        eval_as_of="2026-07-21T00:00:00+00:00",
    )

    def test_model_without_the_requested_metric_discloses_it_on_its_line(self):
        """Model A records brier_score, model B does not: B's line must say so.
        A single global "missing" set hides B's omission entirely."""
        integration = HealthScoreDSPyIntegration()
        out = integration.get_model_metrics_prompt(
            ["brier_score"], [KISQALI_PROD, self.KISQALI_NO_BRIER]
        )
        line_b = _line_naming(out, "initiation_kisqali_goldstd_lr_v1")
        assert "Brier score not recorded" in _metric_segment(line_b), out

    def test_model_with_the_requested_metric_reports_the_value_not_a_disclosure(self):
        integration = HealthScoreDSPyIntegration()
        out = integration.get_model_metrics_prompt(
            ["brier_score"], [KISQALI_PROD, self.KISQALI_NO_BRIER]
        )
        line_a = _line_naming(out, "hcp_adoption_kisqali_goldstd_lr_v1")
        assert "Brier score 0.190" in line_a, out
        assert "not recorded" not in _metric_segment(line_a), out

    @pytest.mark.asyncio
    async def test_trigger_example_brier_of_the_kisqali_models(self):
        """The issue's trigger: "What is the Brier score of the Kisqali
        models?" with one matched model carrying brier_score and another
        carrying only auc_roc + calibration_slope."""
        state = _models_state(
            "What is the Brier score of the Kisqali models?",
            [KISQALI_PROD, self.KISQALI_NO_BRIER],
        )
        summary = (await _compose(state))["health_summary"]
        line_b = _line_naming(summary, "initiation_kisqali_goldstd_lr_v1")
        assert "Brier score not recorded" in _metric_segment(line_b), summary
        line_a = _line_naming(summary, "hcp_adoption_kisqali_goldstd_lr_v1")
        assert "Brier score 0.190" in line_a, summary

    def test_no_false_disclosure_when_every_model_records_it(self):
        """When all matched models record the requested metric there is nothing
        to disclose — no "not recorded" text may appear on any metric line."""
        integration = HealthScoreDSPyIntegration()
        out = integration.get_model_metrics_prompt(["brier_score"], [KISQALI_PROD, KISQALI_STAGING])
        for name in (
            "hcp_adoption_kisqali_goldstd_lr_v1",
            "initiation_kisqali_goldstd_lr_v1",
        ):
            assert "not recorded" not in _metric_segment(_line_naming(out, name)), out

    def test_metric_missing_from_every_model_is_still_disclosed(self):
        """PSI is recorded nowhere: the disclosure must survive the per-model
        rework (the #1450 anti-fabrication guarantee)."""
        integration = HealthScoreDSPyIntegration()
        out = integration.get_model_metrics_prompt(["psi"], [KISQALI_PROD, KISQALI_STAGING])
        lowered = out.lower()
        assert "psi" in lowered and "not recorded" in lowered, out


# =============================================================================
# #1461 — "current"/"production" is a stage constraint, not noise
# =============================================================================


class TestStageConstrainedMatching1461:
    @pytest.mark.parametrize("stage_word", ["current", "production", "live", "champion"])
    def test_stage_word_filters_to_production_candidates(self, stage_word: str):
        query = f"What is the ROC-AUC of the {stage_word} Kisqali model?"
        result = _models_matching_query(query, [KISQALI_PROD, KISQALI_STAGING])
        assert [m["model_name"] for m in result[0]] == ["hcp_adoption_kisqali_goldstd_lr_v1"], (
            f"{stage_word!r} must constrain to production-stage candidates: "
            f"{[m['model_name'] for m in result[0]]}"
        )
        assert result[1] is True

    def test_no_stage_word_still_matches_all_brand_models(self):
        """Without a stage word the brand matches every brand model (pin)."""
        result = _models_matching_query(
            "What is the ROC-AUC of the Kisqali models?", [KISQALI_PROD, KISQALI_STAGING]
        )
        assert {m["model_name"] for m in result[0]} == {
            "hcp_adoption_kisqali_goldstd_lr_v1",
            "initiation_kisqali_goldstd_lr_v1",
        }
        assert result[1] is True

    def test_no_match_disclosure_path_unchanged(self):
        """#1450 pin: an unregistered brand (with OR without a stage word)
        still returns (all_models, matched=False) for the caller disclosure."""
        result = _models_matching_query(
            "What is the ROC-AUC of the current Xolair model?",
            [KISQALI_PROD, KISQALI_STAGING],
        )
        assert result[1] is False
        assert len(result[0]) == 2

    @pytest.mark.asyncio
    async def test_demo_53_excludes_the_staging_model(self):
        """The live-verified defect: demo 5.3 returned
        initiation_kisqali_goldstd_lr_v1 (staging) alongside the production
        champion. "current" must exclude it."""
        state = _models_state(DEMO_53, [KISQALI_PROD, KISQALI_STAGING])
        summary = (await _compose(state))["health_summary"]
        assert "hcp_adoption_kisqali_goldstd_lr_v1" in summary, summary
        assert "initiation_kisqali_goldstd_lr_v1" not in summary, summary


class TestMultiProductionDisambiguation1461:
    KISQALI_PROD_INITIATION = _model(
        "initiation_kisqali_goldstd_lr_v1",
        "production",
        {"auc_roc": 0.851908, "calibration_slope": 1.049494, "brier_score": 0.145714},
        model_id="4ec55d13-46c8-4df4-9ec8-7723fad67fb3",
        eval_sample_size=1645,
        eval_as_of="2026-07-21T00:00:00+00:00",
    )

    def test_named_prediction_target_narrows_the_production_set(self):
        """Two production models for the brand (different prediction targets):
        a question naming the target gets THAT model only."""
        result = _models_matching_query(
            "What is the ROC-AUC of the current Kisqali initiation model?",
            [KISQALI_PROD, self.KISQALI_PROD_INITIATION],
        )
        assert [m["model_name"] for m in result[0]] == ["initiation_kisqali_goldstd_lr_v1"]
        assert result[1] is True

    @pytest.mark.asyncio
    async def test_composer_narrows_to_the_named_target(self):
        state = _models_state(
            "What is the ROC-AUC of the current Kisqali initiation model?",
            [KISQALI_PROD, self.KISQALI_PROD_INITIATION],
        )
        summary = (await _compose(state))["health_summary"]
        assert "initiation_kisqali_goldstd_lr_v1" in summary, summary
        assert "hcp_adoption_kisqali_goldstd_lr_v1" not in summary, summary

    @pytest.mark.asyncio
    async def test_unnamed_target_states_several_production_models_and_lists_them(self):
        """No prediction target in the question -> say explicitly that several
        production models exist for the brand, and list them."""
        state = _models_state(
            "What is the ROC-AUC of the current Kisqali model?",
            [KISQALI_PROD, self.KISQALI_PROD_INITIATION],
        )
        summary = (await _compose(state))["health_summary"]
        assert "hcp_adoption_kisqali_goldstd_lr_v1" in summary, summary
        assert "initiation_kisqali_goldstd_lr_v1" in summary, summary
        assert "everal production models" in summary, (
            "the answer must state explicitly that several production models "
            f"exist for the brand: {summary!r}"
        )


class TestExplicitStageOverride1461Iter2:
    """codex iter-1 findings (2026-08-04): an explicit stage the question names
    must beat the ambient production reading of "current"/"live" (HIGH — the
    staging model asked for was silently swapped for a production one), and a
    stage constraint matching nothing must say so instead of silently
    answering with another stage's model (MED)."""

    FABHALTA_STAGING = _model(
        "initiation_fabhalta_goldstd_lr_v1",
        "staging",
        {"auc_roc": 0.812345, "calibration_slope": 0.98, "brier_score": 0.15},
        model_id="11111111-1111-1111-1111-111111111111",
    )

    def test_explicit_staging_beats_current(self):
        """codex HIGH scenario: "current" must not veto an explicit "staging"."""
        result = _models_matching_query(
            "What is the current ROC-AUC of the staging Kisqali initiation model?",
            [KISQALI_PROD, KISQALI_STAGING],
        )
        assert [m["model_name"] for m in result[0]] == ["initiation_kisqali_goldstd_lr_v1"]
        assert result[1] is True

    def test_explicit_staging_alone_filters_to_staging(self):
        result = _models_matching_query(
            "What is the ROC-AUC of the staging Kisqali model?",
            [KISQALI_PROD, KISQALI_STAGING],
        )
        assert [m["model_name"] for m in result[0]] == ["initiation_kisqali_goldstd_lr_v1"]
        assert result[1] is True

    def test_two_stages_named_returns_models_from_both_stages(self):
        result = _models_matching_query(
            "Compare the production and staging Kisqali models",
            [KISQALI_PROD, KISQALI_STAGING],
        )
        assert {m["model_name"] for m in result[0]} == {
            "hcp_adoption_kisqali_goldstd_lr_v1",
            "initiation_kisqali_goldstd_lr_v1",
        }
        assert result[1] is True

    def test_comparison_filters_to_the_named_stage_set(self):
        """codex iter-2 MED: naming two stages must exclude models in a THIRD
        stage — "production and staging" is a constraint to that set, not the
        absence of a constraint."""
        archived = _model(
            "persistence_kisqali_goldstd_lr_v1",
            "archived",
            {"auc_roc": 0.7, "calibration_slope": 1.0, "brier_score": 0.2},
            model_id="22222222-2222-2222-2222-222222222222",
        )
        result = _models_matching_query(
            "Compare the production and staging Kisqali models",
            [KISQALI_PROD, KISQALI_STAGING, archived],
        )
        assert {m["model_name"] for m in result[0]} == {
            "hcp_adoption_kisqali_goldstd_lr_v1",
            "initiation_kisqali_goldstd_lr_v1",
        }, "the archived model must not leak into a production-vs-staging comparison"
        assert result[1] is True

    def test_comparison_still_narrows_on_a_named_target(self):
        """A comparison naming a prediction target keeps only that target's
        models (one per compared stage)."""
        initiation_prod = _model(
            "initiation_kisqali_goldstd_lr_v1",
            "production",
            {"auc_roc": 0.83, "calibration_slope": 1.0, "brier_score": 0.16},
            model_id="33333333-3333-3333-3333-333333333333",
        )
        adoption_staging = _model(
            "hcp_adoption_kisqali_goldstd_lr_v1",
            "staging",
            {"auc_roc": 0.75, "calibration_slope": 0.9, "brier_score": 0.2},
            model_id="44444444-4444-4444-4444-444444444444",
        )
        result = _models_matching_query(
            "Compare the production and staging Kisqali initiation models",
            [KISQALI_PROD, KISQALI_STAGING, initiation_prod, adoption_staging],
        )
        assert [m["model_name"] for m in result[0]] == [
            "initiation_kisqali_goldstd_lr_v1",
            "initiation_kisqali_goldstd_lr_v1",
        ], [m["model_name"] for m in result[0]]
        assert {m["model_stage"] for m in result[0]} == {"production", "staging"}
        assert result[1] is True

    def test_comparison_discloses_a_stage_with_no_matching_target(self):
        """codex iter-3 MED: if one compared stage has no model for the named
        target, the collapse to the other stage must be disclosed, not
        silent — production has only hcp_adoption here, so a
        production-vs-staging INITIATION comparison can only show staging."""
        result = _models_matching_query(
            "Compare the production and staging Kisqali initiation models",
            [KISQALI_PROD, KISQALI_STAGING],
        )
        assert [m["model_name"] for m in result[0]] == ["initiation_kisqali_goldstd_lr_v1"]
        assert result[1] is True
        assert "No production-stage model" in result[2], result[2]

    def test_no_model_in_requested_stage_discloses_it(self):
        """codex MED scenario: a production-constrained question over a brand
        with only a staging model must NOT silently answer as if the staging
        model were the requested one."""
        result = _models_matching_query(
            "What is the ROC-AUC of the production Fabhalta initiation model?",
            [KISQALI_PROD, self.FABHALTA_STAGING],
        )
        assert [m["model_name"] for m in result[0]] == ["initiation_fabhalta_goldstd_lr_v1"]
        assert result[1] is True
        assert "No production-stage model" in result[2], result[2]

    def test_unregistered_brand_with_stage_word_keeps_1450_disclosure(self):
        """#1450 pin: no name match at all still returns (all, False, "")."""
        result = _models_matching_query(
            "What is the ROC-AUC of the staging Xolair model?",
            [KISQALI_PROD, KISQALI_STAGING],
        )
        assert result[1] is False
        assert len(result[0]) == 2
        assert result[2] == ""

    @pytest.mark.asyncio
    async def test_composer_renders_the_no_stage_match_note(self):
        state = _models_state(
            "What is the ROC-AUC of the production Fabhalta initiation model?",
            [KISQALI_PROD, self.FABHALTA_STAGING],
        )
        summary = (await _compose(state))["health_summary"]
        assert "initiation_fabhalta_goldstd_lr_v1" in summary, summary
        assert "No production-stage model" in summary, summary

    @pytest.mark.asyncio
    async def test_composer_answers_staging_question_with_the_staging_model(self):
        state = _models_state(
            "What is the current ROC-AUC of the staging Kisqali initiation model?",
            [KISQALI_PROD, KISQALI_STAGING],
        )
        summary = (await _compose(state))["health_summary"]
        assert "initiation_kisqali_goldstd_lr_v1" in summary, summary
        assert "hcp_adoption_kisqali_goldstd_lr_v1" not in summary, summary
