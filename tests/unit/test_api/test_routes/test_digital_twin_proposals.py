"""``/digital-twin/proposed-experiments`` (#2206, owner item C: "surface proposed experiments").

A proposed experiment = a completed twin simulation whose recommendation is
deploy or refine and which is not yet linked to an experiment. The list is
brand-grant-filtered like every other twin read (H11); the draft action creates
ONE ``ml_experiments`` row with ``status='draft'`` (legal in the CHECK since
migration 061, never written by anything before) and links the simulation to
it. Nothing auto-runs: the owner promotes the draft.

Handlers are driven directly (the H11 test convention) with a
``SimpleNamespace`` repo whose client chain records the INSERT it receives.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException

VIEWER_KISQALI = {"app_metadata": {"role": "viewer", "brands": ["Kisqali"]}}
OPERATOR_KISQALI = {
    "email": "ops@example.com",
    "app_metadata": {"role": "operator", "brands": ["Kisqali"]},
}
ADMIN = {"id": "admin-id", "email": "admin@example.com", "app_metadata": {"role": "admin"}}


def _sim(brand="Kisqali", recommendation="deploy", ate=0.1, linked=None, **overrides):
    row = {
        "simulation_id": str(uuid4()),
        "model_id": str(uuid4()),
        "experiment_design_id": linked,
        "intervention_type": "email_campaign",
        "intervention_config": {"channel": "email", "duration_weeks": 8},
        "brand": brand,
        "simulated_ate": ate,
        "simulated_ci_lower": ate - 0.05,
        "simulated_ci_upper": ate + 0.05,
        "recommendation": recommendation,
        "recommendation_rationale": "CI excludes zero",
        "recommended_sample_size": 1200,
        "recommended_duration_weeks": 8,
        "simulation_confidence": 0.71,
        "simulation_status": "completed",
        "data_provenance": "cohort_estimated_synthetic_gold_v1",
        "created_at": datetime.now(timezone.utc),
    }
    row.update(overrides)
    return row


def _model_row(model_id, fidelity_score=None):
    return {
        "model_id": model_id,
        "fidelity_score": fidelity_score,
        "fidelity_sample_count": 0 if fidelity_score is None else 1,
    }


def _client(
    running_count=0, insert_rows=None, insert_error=None, measurable_count=0, delete_rows=None
):
    """repo.client: the count chains (running experiments, then real rows carrying the
    twin outcome — in the order the route issues them), the insert chain and the
    delete chain, all async."""
    chain = MagicMock()
    for name in ("select", "eq", "limit", "order", "delete"):
        getattr(chain, name).return_value = chain
    chain.not_.is_.return_value = chain

    # Count results by TABLE, not by call order: the route reads business_metrics
    # (measurability) and ml_experiments (running count) in whatever order.
    def _execute():
        table = client.table.call_args.args[0] if client.table.call_args else ""
        count = measurable_count if table == "business_metrics" else running_count
        return MagicMock(data=[], count=count)

    chain.execute = AsyncMock(side_effect=_execute)
    insert_chain = MagicMock()
    if insert_error is not None:
        insert_chain.execute = AsyncMock(side_effect=insert_error)
    else:
        insert_chain.execute = AsyncMock(return_value=MagicMock(data=insert_rows or []))
    chain.insert.return_value = insert_chain
    # The compensating delete after a lost claim returns the deleted row(s).
    delete_chain = MagicMock()
    delete_chain.eq.return_value = delete_chain
    delete_chain.execute = AsyncMock(
        return_value=MagicMock(data=delete_rows if delete_rows is not None else [{"id": "deleted"}])
    )
    chain.delete.return_value = delete_chain
    client = MagicMock()
    client.table.return_value = chain
    return client, chain


def _repo(proposed, *, linked=0, models=None, sim=None, link_ok=True, client=None):
    models = models or {}
    return SimpleNamespace(
        client=client or _client()[0],
        list_proposed_experiments=AsyncMock(return_value=proposed),
        count_linked_simulations=AsyncMock(return_value=linked),
        # The routes use the STRICT reads (codex r2 #2): None = no row; failures raise.
        require_model=AsyncMock(side_effect=lambda mid: models.get(str(mid))),
        require_simulation=AsyncMock(return_value=sim),
        claim_experiment_link=AsyncMock(
            side_effect=link_ok if isinstance(link_ok, Exception) else None,
            return_value=None if isinstance(link_ok, Exception) else link_ok,
        ),
    )


def _patched(repo):
    from src.api.routes import digital_twin as dt

    return patch.object(dt, "_get_twin_repo", AsyncMock(return_value=repo))


# =============================================================================
# GET /digital-twin/proposed-experiments
# =============================================================================


class TestListProposedExperiments:
    def test_out_of_grant_brand_is_403(self):
        from src.api.routes.digital_twin import BrandEnum
        from src.api.routes.digital_twin_proposals import list_proposed_experiments

        with pytest.raises(HTTPException) as ei:
            asyncio.run(
                list_proposed_experiments(brand=BrandEnum.REMIBRUTINIB, user=VIEWER_KISQALI)
            )
        assert ei.value.status_code == 403

    def test_non_admin_is_pinned_to_their_grant(self):
        from src.api.routes.digital_twin_proposals import list_proposed_experiments

        repo = _repo([])
        with _patched(repo):
            resp = asyncio.run(list_proposed_experiments(brand=None, user=VIEWER_KISQALI))

        assert resp.total_proposed == 0
        repo.list_proposed_experiments.assert_awaited_once()
        assert repo.list_proposed_experiments.call_args.kwargs["brand"] == "Kisqali"
        repo.count_linked_simulations.assert_awaited_once_with(brand="Kisqali")

    def test_items_carry_the_twin_parameters_and_the_models_fidelity_state(self):
        from src.api.routes.digital_twin_proposals import list_proposed_experiments

        sim = _sim()
        repo = _repo([sim], models={sim["model_id"]: _model_row(sim["model_id"])})
        with _patched(repo):
            resp = asyncio.run(list_proposed_experiments(brand=None, user=ADMIN))

        assert resp.total_proposed == 1
        item = resp.proposals[0]
        assert item.simulation_id == sim["simulation_id"]
        assert item.brand == "Kisqali"
        assert item.intervention_type == "email_campaign"
        assert item.intervention_config == {"channel": "email", "duration_weeks": 8}
        assert item.simulated_ate == 0.1
        assert item.simulated_ci_lower == pytest.approx(0.05)
        assert item.simulated_ci_upper == pytest.approx(0.15)
        assert item.recommendation == "deploy"
        assert item.recommendation_rationale == "CI excludes zero"
        assert item.recommended_sample_size == 1200
        assert item.recommended_duration_weeks == 8
        assert item.simulation_confidence == 0.71
        assert item.data_provenance == "cohort_estimated_synthetic_gold_v1"
        assert item.fidelity_status.value == "unvalidated"
        assert item.proposal_basis == "twin_simulation"
        # codex r1 #5: the ATE is an ABSOLUTE difference on the twin's outcome column,
        # never a percentage lift; the item names both.
        assert item.outcome_column == "cohort_conversion_outcome"
        assert item.effect_scale == "absolute"
        assert resp.outcome_column == "cohort_conversion_outcome"

    def test_a_measured_model_reads_validated(self):
        from src.api.routes.digital_twin_proposals import list_proposed_experiments

        sim = _sim()
        repo = _repo([sim], models={sim["model_id"]: _model_row(sim["model_id"], 0.9)})
        with _patched(repo):
            resp = asyncio.run(list_proposed_experiments(brand=None, user=ADMIN))
        assert resp.proposals[0].fidelity_status.value == "validated"

    def test_a_missing_model_row_reads_unvalidated_not_as_a_crash(self):
        from src.api.routes.digital_twin_proposals import list_proposed_experiments

        repo = _repo([_sim()], models={})
        with _patched(repo):
            resp = asyncio.run(list_proposed_experiments(brand=None, user=ADMIN))
        assert resp.proposals[0].fidelity_status.value == "unvalidated"

    def test_ordering_is_deploy_first_then_effect_desc(self):
        from src.api.routes.digital_twin_proposals import list_proposed_experiments

        rows = [
            _sim(recommendation="refine", ate=0.30),
            _sim(recommendation="deploy", ate=0.05),
            _sim(recommendation="deploy", ate=0.20),
            _sim(recommendation="refine", ate=0.01),
        ]
        repo = _repo(rows)
        with _patched(repo):
            resp = asyncio.run(list_proposed_experiments(brand=None, user=ADMIN))

        assert [(p.recommendation, p.simulated_ate) for p in resp.proposals] == [
            ("deploy", 0.20),
            ("deploy", 0.05),
            ("refine", 0.30),
            ("refine", 0.01),
        ]

    def test_envelope_reports_linked_and_real_running_counts_honestly(self):
        """Today: 0 real experiments running (every real ml_experiments row is
        pipeline lineage; the 360 running A/B rows are synthetic). The envelope
        says so with a real count, never a placeholder."""
        from src.api.routes.digital_twin_proposals import list_proposed_experiments

        client, chain = _client(running_count=0)
        repo = _repo([_sim()], linked=2, client=client)
        with _patched(repo):
            resp = asyncio.run(list_proposed_experiments(brand=None, user=ADMIN))

        assert resp.total_linked == 2
        assert resp.real_experiments_running == 0
        client.table.assert_any_call("ml_experiments")
        chain.select.assert_any_call("id", count="exact")
        chain.eq.assert_any_call("status", "running")
        chain.not_.is_.assert_any_call("intervention_channel", "null")
        # Real mode: the synthetic substrate is excluded from the count.
        chain.eq.assert_any_call("is_synthetic", False)

    def test_envelope_says_whether_the_twin_outcome_is_measurable_in_real_mode(self):
        """codex r1 #1 (measured live): cohort_conversion_outcome is recorded ONLY on the
        synthetic-gold per-HCP rows (13,797 rows, all is_synthetic=true); the real-mode
        outcome feed excludes them. A real draft's final analysis therefore cannot
        measure it today. The envelope states that from a real count, never assumes."""
        from src.api.routes.digital_twin_proposals import list_proposed_experiments

        client, chain = _client(running_count=0, measurable_count=0)
        repo = _repo([_sim()], client=client)
        with _patched(repo):
            resp = asyncio.run(list_proposed_experiments(brand=None, user=ADMIN))
        assert resp.outcome_measurable_in_real_mode is False
        client.table.assert_any_call("business_metrics")
        chain.eq.assert_any_call("metric_type", "per_hcp_rollup")
        chain.not_.is_.assert_any_call("cohort_conversion_outcome", "null")

        client2, _ = _client(running_count=0, measurable_count=42)
        repo2 = _repo([_sim()], client=client2)
        with _patched(repo2):
            resp2 = asyncio.run(list_proposed_experiments(brand=None, user=ADMIN))
        assert resp2.outcome_measurable_in_real_mode is True

    def test_real_running_count_is_scoped_to_the_callers_brand(self):
        """codex r1 #6: a Kisqali-only viewer must not receive the all-brand running count
        next to Kisqali-only proposal counts."""
        from src.api.routes.digital_twin_proposals import list_proposed_experiments

        client, chain = _client(running_count=0)
        repo = _repo([], client=client)
        with _patched(repo):
            asyncio.run(list_proposed_experiments(brand=None, user=VIEWER_KISQALI))
        assert ("brand", "Kisqali") in [c.args for c in chain.eq.call_args_list]

    def test_a_store_failure_is_a_500_not_an_empty_portfolio(self):
        """codex r1 #3: a query failure must never read as "no proposals"."""
        from src.api.routes.digital_twin_proposals import list_proposed_experiments

        repo = _repo([])
        repo.list_proposed_experiments = AsyncMock(side_effect=RuntimeError("42703"))
        with _patched(repo), pytest.raises(HTTPException) as ei:
            asyncio.run(list_proposed_experiments(brand=None, user=ADMIN))
        assert ei.value.status_code == 500

    def test_real_mode_counts_exclude_synthetic_even_on_a_showcase_deployment(self):
        """codex r2 #1: apply_provenance_filter deliberately drops the is_synthetic
        predicate when E2I_INCLUDE_SYNTHETIC is set. These two diagnostics are
        explicitly REAL-mode statements, so they filter is_synthetic=false themselves."""
        from src.api.routes.digital_twin_proposals import list_proposed_experiments

        client, chain = _client(running_count=0, measurable_count=0)
        repo = _repo([_sim()], client=client)
        with (
            _patched(repo),
            patch("src.repositories.provenance.deployment_includes_synthetic", return_value=True),
        ):
            resp = asyncio.run(list_proposed_experiments(brand=None, user=ADMIN))
        assert resp.real_experiments_running == 0
        assert resp.outcome_measurable_in_real_mode is False
        assert [c.args for c in chain.eq.call_args_list].count(("is_synthetic", False)) >= 2

    def test_a_failed_model_lookup_is_a_500_not_an_unvalidated_label(self):
        """codex r2 #2: a database failure must not read as 'unvalidated'."""
        from src.api.routes.digital_twin_proposals import list_proposed_experiments

        repo = _repo([_sim()])
        repo.require_model = AsyncMock(side_effect=RuntimeError("connection reset"))
        with _patched(repo), pytest.raises(HTTPException) as ei:
            asyncio.run(list_proposed_experiments(brand=None, user=ADMIN))
        assert ei.value.status_code == 500

    def test_one_model_lookup_per_distinct_model(self):
        from src.api.routes.digital_twin_proposals import list_proposed_experiments

        model_id = str(uuid4())
        rows = [_sim(model_id=model_id), _sim(model_id=model_id), _sim()]
        repo = _repo(rows)
        with _patched(repo):
            asyncio.run(list_proposed_experiments(brand=None, user=ADMIN))
        assert repo.require_model.await_count == 2


# =============================================================================
# POST /digital-twin/proposed-experiments/{simulation_id}/draft
# =============================================================================


class TestCreateDraftExperiment:
    def test_unknown_simulation_is_404(self):
        from src.api.routes.digital_twin_proposals import create_draft_experiment

        repo = _repo([], sim=None)
        with _patched(repo), pytest.raises(HTTPException) as ei:
            asyncio.run(create_draft_experiment(str(uuid4()), user=ADMIN))
        assert ei.value.status_code == 404

    def test_a_failed_simulation_lookup_is_a_500_not_a_404(self):
        """codex r2 #2: a database failure must not read as 'no such simulation'."""
        from src.api.routes.digital_twin_proposals import create_draft_experiment

        repo = _repo([], sim=None)
        repo.require_simulation = AsyncMock(side_effect=RuntimeError("connection reset"))
        with _patched(repo), pytest.raises(HTTPException) as ei:
            asyncio.run(create_draft_experiment(str(uuid4()), user=ADMIN))
        assert ei.value.status_code == 500

    def test_the_measurability_read_happens_before_the_insert(self):
        """codex r2 #5: a diagnostic read that fails AFTER the committed draft would
        turn a success into a 500 (and the retry into a 409). It runs first."""
        from src.api.routes.digital_twin_proposals import create_draft_experiment

        sim = _sim()
        client, _chain = _client(insert_rows=[{"id": str(uuid4())}])
        repo = _repo([], sim=sim, client=client)
        with _patched(repo):
            asyncio.run(create_draft_experiment(sim["simulation_id"], user=ADMIN))
        tables = [c.args[0] for c in client.table.call_args_list]
        assert tables.index("business_metrics") < tables.index("ml_experiments")
        assert tables.count("business_metrics") == 1

    def test_a_failing_measurability_read_is_a_500_before_any_mutation(self):
        from src.api.routes.digital_twin_proposals import create_draft_experiment

        sim = _sim()
        client, chain = _client(insert_rows=[{"id": str(uuid4())}])
        chain.execute = AsyncMock(side_effect=RuntimeError("connection reset"))
        repo = _repo([], sim=sim, client=client)
        with _patched(repo), pytest.raises(HTTPException) as ei:
            asyncio.run(create_draft_experiment(sim["simulation_id"], user=ADMIN))
        assert ei.value.status_code == 500
        chain.insert.assert_not_called()
        repo.claim_experiment_link.assert_not_awaited()

    def test_out_of_grant_simulation_is_404_not_403(self):
        """Do not leak another tenant's simulation."""
        from src.api.routes.digital_twin_proposals import create_draft_experiment

        sim = _sim(brand="Remibrutinib")
        repo = _repo([], sim=sim)
        with _patched(repo), pytest.raises(HTTPException) as ei:
            asyncio.run(create_draft_experiment(sim["simulation_id"], user=OPERATOR_KISQALI))
        assert ei.value.status_code == 404
        repo.client.table.assert_not_called()

    def test_already_linked_is_409_naming_the_experiment(self):
        from src.api.routes.digital_twin_proposals import create_draft_experiment

        existing = str(uuid4())
        sim = _sim(linked=existing)
        repo = _repo([], sim=sim)
        with _patched(repo), pytest.raises(HTTPException) as ei:
            asyncio.run(create_draft_experiment(sim["simulation_id"], user=ADMIN))
        assert ei.value.status_code == 409
        assert existing in str(ei.value.detail)
        repo.client.table.assert_not_called()

    @pytest.mark.parametrize(
        "overrides",
        [
            {"recommendation": "skip"},
            {"simulation_status": "failed"},
        ],
    )
    def test_only_a_completed_deploy_or_refine_run_can_become_a_draft(self, overrides):
        from src.api.routes.digital_twin_proposals import create_draft_experiment

        sim = _sim(**overrides)
        repo = _repo([], sim=sim)
        with _patched(repo), pytest.raises(HTTPException) as ei:
            asyncio.run(create_draft_experiment(sim["simulation_id"], user=ADMIN))
        assert ei.value.status_code == 422
        repo.client.table.assert_not_called()

    def test_creates_one_draft_row_from_the_twin_parameters_and_links_it(self):
        from src.api.routes.digital_twin_proposals import create_draft_experiment
        from src.data.per_hcp_cohort_columns import COHORT_OUTCOME_COLUMN

        sim = _sim()
        exp_id = str(uuid4())
        client, chain = _client(insert_rows=[{"id": exp_id}])
        repo = _repo([], sim=sim, client=client)
        with _patched(repo):
            resp = asyncio.run(create_draft_experiment(sim["simulation_id"], user=OPERATOR_KISQALI))

        client.table.assert_any_call("ml_experiments")
        chain.insert.assert_called_once()
        data = chain.insert.call_args.args[0]
        # The landmines: status must be EXPLICIT ('draft' — None inherits the DB
        # default 'running'); no MLflow id is invented (column nullable, UNIQUE).
        assert data["status"] == "draft"
        assert "mlflow_experiment_id" not in data or data["mlflow_experiment_id"] is None
        assert data["brand"] == "Kisqali"
        assert data["intervention_channel"] == "email_campaign"
        assert data["target_enrollment"] == 1200
        assert data["planned_duration_days"] == 56  # 8 weeks * 7
        # The outcome the twin predicted an effect ON — the same column the final
        # results feed measures, so the loop can close on the same quantity.
        assert data["prediction_target"] == COHORT_OUTCOME_COLUMN
        assert data["created_by"] == "ops@example.com"
        assert data["experiment_name"].startswith("twin_proposal_Kisqali_email_campaign_")
        assert sim["simulation_id"][:8] in data["experiment_name"]
        assert "0.1" in data["description"] and "unvalidated" in data["description"]
        # is_synthetic is never sent: the column's NOT NULL DEFAULT false makes the
        # row real-mode visible, which is what the fidelity loop needs.
        assert "is_synthetic" not in data

        repo.claim_experiment_link.assert_awaited_once_with(
            UUID(sim["simulation_id"]), UUID(exp_id)
        )
        assert resp.experiment_id == exp_id
        assert resp.simulation_id == sim["simulation_id"]
        assert resp.status == "draft"
        assert resp.linked is True
        assert resp.outcome_column == COHORT_OUTCOME_COLUMN
        # Measured live: the outcome is recorded only on synthetic-gold rows today.
        assert resp.outcome_measurable_in_real_mode is False
        assert "promot" in resp.next_step.lower()
        assert "synthetic-gold" in resp.next_step

    def test_a_lost_claim_deletes_the_orphan_draft_and_is_a_409(self):
        """codex r1 #2: two concurrent POSTs both pass the pre-check and both insert a
        draft; the link is a conditional CLAIM (experiment_design_id IS NULL), so exactly
        one wins. The loser removes the draft it just inserted and answers 409 naming
        the winner — never a second 201 hiding an orphan."""
        from src.api.routes.digital_twin_proposals import create_draft_experiment

        sim = _sim()
        exp_id = str(uuid4())
        winner = str(uuid4())
        client, chain = _client(insert_rows=[{"id": exp_id}])
        repo = _repo([], sim=sim, client=client, link_ok=False)
        # After the lost claim the route re-reads the simulation: it is linked now.
        repo.require_simulation = AsyncMock(
            side_effect=[sim, {**sim, "experiment_design_id": winner}]
        )
        with _patched(repo), pytest.raises(HTTPException) as ei:
            asyncio.run(create_draft_experiment(sim["simulation_id"], user=ADMIN))
        assert ei.value.status_code == 409
        assert winner in str(ei.value.detail)
        chain.delete.assert_called_once()
        chain.delete.return_value.eq.assert_any_call("id", exp_id)
        chain.delete.return_value.eq.assert_any_call("status", "draft")

    def test_a_lost_claim_whose_cleanup_removed_nothing_is_a_500_naming_the_draft(self):
        """codex r2 #3: the 409 must PROVE the orphan is gone; a delete that removed no
        row (or raised) leaves a draft the caller must be told about."""
        from src.api.routes.digital_twin_proposals import create_draft_experiment

        sim = _sim()
        exp_id = str(uuid4())
        client, _chain = _client(insert_rows=[{"id": exp_id}], delete_rows=[])
        repo = _repo([], sim=sim, client=client, link_ok=False)
        with _patched(repo), pytest.raises(HTTPException) as ei:
            asyncio.run(create_draft_experiment(sim["simulation_id"], user=ADMIN))
        assert ei.value.status_code == 500
        assert exp_id in str(ei.value.detail)

    def test_a_lost_claim_with_no_visible_winner_is_a_500_not_a_409_naming_none(self):
        from src.api.routes.digital_twin_proposals import create_draft_experiment

        sim = _sim()
        exp_id = str(uuid4())
        client, _chain = _client(insert_rows=[{"id": exp_id}])
        repo = _repo([], sim=sim, client=client, link_ok=False)
        repo.require_simulation = AsyncMock(
            side_effect=[sim, {**sim, "experiment_design_id": None}]
        )
        with _patched(repo), pytest.raises(HTTPException) as ei:
            asyncio.run(create_draft_experiment(sim["simulation_id"], user=ADMIN))
        assert ei.value.status_code == 500
        assert "None" not in str(ei.value.detail)

    def test_a_lost_claim_whose_reread_fails_still_reports_the_completed_removal(self):
        """codex r3 #1: a failing winner re-read must not relabel a completed delete as
        'could not be removed' — the caller would resolve by hand a draft that is gone."""
        from src.api.routes.digital_twin_proposals import create_draft_experiment

        sim = _sim()
        exp_id = str(uuid4())
        client, _chain = _client(insert_rows=[{"id": exp_id}])
        repo = _repo([], sim=sim, client=client, link_ok=False)
        repo.require_simulation = AsyncMock(side_effect=[sim, RuntimeError("connection reset")])
        with _patched(repo), pytest.raises(HTTPException) as ei:
            asyncio.run(create_draft_experiment(sim["simulation_id"], user=ADMIN))
        assert ei.value.status_code == 500
        detail = str(ei.value.detail)
        assert "was removed" in detail
        assert "could not be removed" not in detail
        assert "could not be read" in detail

    def test_a_claim_that_raises_is_a_500_naming_both_ids_never_a_200(self):
        from src.api.routes.digital_twin_proposals import create_draft_experiment

        sim = _sim()
        exp_id = str(uuid4())
        client, _chain = _client(insert_rows=[{"id": exp_id}])
        repo = _repo([], sim=sim, client=client, link_ok=RuntimeError("connection reset"))
        with _patched(repo), pytest.raises(HTTPException) as ei:
            asyncio.run(create_draft_experiment(sim["simulation_id"], user=ADMIN))
        assert ei.value.status_code == 500
        assert exp_id in str(ei.value.detail)
        assert sim["simulation_id"] in str(ei.value.detail)

    def test_an_insert_that_returns_no_row_is_a_500_not_a_dangling_link(self):
        from src.api.routes.digital_twin_proposals import create_draft_experiment

        sim = _sim()
        client, _chain = _client(insert_rows=[])
        repo = _repo([], sim=sim, client=client)
        with _patched(repo), pytest.raises(HTTPException) as ei:
            asyncio.run(create_draft_experiment(sim["simulation_id"], user=ADMIN))
        assert ei.value.status_code == 500
        repo.claim_experiment_link.assert_not_awaited()

    def test_malformed_simulation_id_is_422(self):
        from src.api.routes.digital_twin_proposals import create_draft_experiment

        with pytest.raises(HTTPException) as ei:
            asyncio.run(create_draft_experiment("not-a-uuid", user=ADMIN))
        assert ei.value.status_code == 422


# =============================================================================
# A draft never counts as running anywhere
# =============================================================================


class TestADraftNeverCountsAsRunning:
    """Every live 'active experiments' predicate selects ``status = 'running'``
    (the Home tile's /active-count, the enrollment sweep, the interim sweep).
    The draft writer writes ``'draft'``. Pinned at source level here; the
    faithful proof is the BEGIN/ROLLBACK rehearsal recorded in the PR (a planted
    draft row leaves all three counts unchanged)."""

    def test_the_writer_writes_draft(self):
        from src.api.routes import digital_twin_proposals as mod

        assert mod.DRAFT_STATUS == "draft"

    @pytest.mark.parametrize(
        "path,marker",
        [
            ("src/api/routes/experiments.py", '.eq("status", "running")'),
            ("src/tasks/ab_testing_tasks.py", '.eq("status", "running")'),
        ],
    )
    def test_every_live_running_predicate_selects_status_running(self, path, marker):
        from pathlib import Path

        root = Path(__file__).resolve().parents[4]
        source = (root / path).read_text()
        assert marker in source, f"{path}: the running predicate moved; re-verify draft exclusion"

    def test_the_route_is_mounted_under_the_digital_twin_prefix(self):
        from src.api.main import app

        paths = {r.path for r in app.routes}
        assert "/api/digital-twin/proposed-experiments" in paths
        assert "/api/digital-twin/proposed-experiments/{simulation_id}/draft" in paths


# =============================================================================
# experiment_design_id read-back on the simulation surfaces (item C.3)
# =============================================================================


class TestExperimentLinkReadBack:
    """SimulationResponse omitted the link entirely: a caller that linked a
    pre-screen got a 200 with no confirmation. The stored detail, the list and
    the history now carry ``experiment_design_id`` (null = a proposal)."""

    def test_stored_detail_carries_the_link(self):
        from src.api.routes import digital_twin as dt
        from src.api.routes.digital_twin import get_simulation

        exp_id = str(uuid4())
        row = _sim(linked=exp_id, twin_count=100, effect_heterogeneity={})
        repo = SimpleNamespace(
            get_simulation=AsyncMock(return_value=row), get_model=AsyncMock(return_value=None)
        )
        with patch.object(dt, "_get_twin_repo", AsyncMock(return_value=repo)):
            detail = asyncio.run(get_simulation(row["simulation_id"], user=ADMIN))
        assert detail.experiment_design_id == exp_id

        row["experiment_design_id"] = None
        with patch.object(dt, "_get_twin_repo", AsyncMock(return_value=repo)):
            detail = asyncio.run(get_simulation(row["simulation_id"], user=ADMIN))
        assert detail.experiment_design_id is None

    def test_history_and_list_rows_carry_the_link(self):
        from src.api.routes import digital_twin as dt
        from src.api.routes.digital_twin import get_simulation_history, list_simulations

        exp_id = str(uuid4())
        rows = [_sim(linked=exp_id, twin_count=100), _sim(twin_count=100)]
        repo = SimpleNamespace(
            simulations=SimpleNamespace(list_simulations=AsyncMock(return_value=rows))
        )
        with patch.object(dt, "_get_twin_repo", AsyncMock(return_value=repo)):
            history = asyncio.run(
                get_simulation_history(brand=None, limit=20, offset=0, user=ADMIN)
            )
            listing = asyncio.run(
                list_simulations(
                    brand=None, model_id=None, status=None, page=1, page_size=20, user=ADMIN
                )
            )
        assert [h.experiment_design_id for h in history.simulations] == [exp_id, None]
        assert [i.experiment_design_id for i in listing.simulations] == [exp_id, None]
