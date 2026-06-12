"""#889: four agent episodic readers post-filter raw_content the search RPC never returns.

The ``search_episodic_memory`` RPC (database/memory/035; re-verified live
2026-06-12: the RETURNS TABLE has no ``raw_content`` column) returns rows
WITHOUT ``raw_content``, while these readers post-filter on
``row.get("raw_content", {})`` — so every row hydrates to ``{}`` and the
filters silently drop or mis-classify ALL rows:

* heterogeneous_optimizer ``_get_episodic_context``  — treatment/outcome filter
  drops everything whenever the production caller passes the vars (it does).
* tool_composer ``find_similar_compositions``        — ``success`` filter →
  ``successful`` is permanently ``[]``, so the planner's G1/G2 episodic
  context (planner.py ``_check_episodic_memory`` → ``_format_episodic_context``)
  never fires.
* causal_impact ``get_prior_analyses``               — ``confidence`` filter →
  always ``[]``. DOUBLE bug: the write path never stored ``confidence`` in
  raw_content either (only ``confidence_interval``), so hydration alone could
  not fix it — the write now persists the gate confidence it already computes.
* causal_impact ``_get_episodic_context``            — same in-file family
  (issue table listed :629; :230 is the identical pattern, fixed together).
* scope_definer ``get_prior_scopes``                 — ``validation_passed``
  filter → always ``[]``.

Fix under test (the #888 gap_analyzer/cohort_constructor precedent): hydrate
``raw_content`` by memory_id via ONE batched PK select
(``episodic_memory.hydrate_raw_content``), post-filter on the hydrated stored
content, and over-fetch via ``content_filter_fetch_limit`` so the post-filter
cannot be starved by high-similarity non-matching rows.

RED on main @ 9bf176af (quoted per test): each filtered read returns ``[]``
for a row written seconds earlier through the agent's REAL write hook.

Each test writes through the real agent write path (no mocks), reads back
through the fixed reader with filters engaged, deletes its rows, and a
row-count guard asserts the table is bit-identical pre/post (baseline-bracket
pattern). Gated like the other faithful real-DB tests; run with the shared-DB
lock::

    flock -w 2400 /tmp/e2i_dbtest.lock -c \\
        'E2I_DB_INTEGRATION=1 PYTHONPATH=$PWD .venv/bin/pytest -n0 \\
         tests/integration/test_raw_content_hydration_reads_889.py'
"""

import os
import uuid

import pytest

_GATE = os.environ.get("E2I_DB_INTEGRATION") == "1"
_HAS_CREDS = bool(os.environ.get("OPENAI_API_KEY")) and bool(os.environ.get("SUPABASE_URL"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_GATE and _HAS_CREDS),
        reason="faithful real-DB memory-read test; set E2I_DB_INTEGRATION=1 + creds in .env",
    ),
]


def _cleanup_episodic(memory_id: str) -> None:
    from src.memory.episodic_memory import get_supabase_client

    get_supabase_client().table("episodic_memories").delete().eq("memory_id", memory_id).execute()


@pytest.fixture()
def episodic_rowcount_guard():
    """Baseline-bracket: episodic_memories row count must be identical pre/post.

    The integration tests share the docker DB with other agents (flock
    serializes access), so a leaked row is both a test bug and a pollution
    hazard — fail loudly instead of silently accumulating.
    """
    from src.memory.episodic_memory import get_supabase_client

    client = get_supabase_client()

    def _count() -> int:
        resp = (
            client.table("episodic_memories").select("memory_id", count="exact").limit(1).execute()
        )
        return int(resp.count or 0)

    before = _count()
    yield
    after = _count()
    assert after == before, f"test leaked episodic rows: {before} -> {after}"


# =============================================================================
# heterogeneous_optimizer — _get_episodic_context treatment/outcome filter
# =============================================================================


async def _store_cate_row(
    session_id: str, marker: str, treatment_var: str, outcome_var: str
) -> str | None:
    """Seed one real CATE episodic row through the #873-fixed write path."""
    from src.agents.heterogeneous_optimizer.memory_hooks import (
        HeterogeneousOptimizerMemoryHooks,
    )

    hooks = HeterogeneousOptimizerMemoryHooks()
    analysis_result = {
        "treatment_var": treatment_var,
        "outcome_var": outcome_var,
        "heterogeneity_score": 0.42,
        "overall_ate": 0.17,
        "high_responders": [{"segment": "northeast_high_decile"}],
        "low_responders": [{"segment": "southwest_low_decile"}],
        "status": "completed",
        "executive_summary": f"CATE heterogeneity probe ({marker})",
    }
    return await hooks.store_cate_analysis(
        session_id=session_id,
        analysis_result=analysis_result,
        brand="remibrutinib",
        region="northeast",
    )


@pytest.mark.asyncio
async def test_het_episodic_context_with_vars_returns_hydrated_row(episodic_rowcount_guard):
    """RED on main: ``_get_episodic_context(query, treatment_var=, outcome_var=)``
    returned ``[]`` for a row stored seconds earlier — the post-filter read
    ``raw_content`` off rows the search RPC returns WITHOUT it, so EVERY row
    failed the treatment_var check (#889 site 1). The production caller
    (get_memory_context) always passes the vars when the state has them, i.e.
    the het episodic context was permanently empty. GREEN: the row comes back
    with the hydrated stored dict, and the filters stay real in both
    directions."""
    from src.agents.heterogeneous_optimizer.memory_hooks import (
        HeterogeneousOptimizerMemoryHooks,
    )

    session_id = str(uuid.uuid4())
    marker = f"889-het-{uuid.uuid4().hex[:12]}"
    treatment_var = f"hcp_calls_{marker}"
    outcome_var = "trx_rate"
    query = f"CATE analysis: {treatment_var} -> {outcome_var} heterogeneity probe ({marker})"

    hooks = HeterogeneousOptimizerMemoryHooks()
    memory_id: str | None = None
    try:
        # Seed INSIDE try so a failed write/assert cannot leak the row
        # (codex R1 LOW on this suite).
        memory_id = await _store_cate_row(session_id, marker, treatment_var, outcome_var)
        assert memory_id, "store_cate_analysis failed — cannot exercise the filtered read"

        results = await hooks._get_episodic_context(
            query=query,
            treatment_var=treatment_var,
            outcome_var=outcome_var,
        )
        by_id = {str(r.get("memory_id")): r for r in results}
        assert str(memory_id) in by_id, (
            "_get_episodic_context(treatment_var/outcome_var) dropped the "
            "just-written row — the raw_content post-filter ran against rows "
            "the search RPC returns WITHOUT raw_content (#889 site 1)"
        )
        row = by_id[str(memory_id)]
        assert isinstance(row.get("raw_content"), dict), (
            "returned row must carry the HYDRATED raw_content dict"
        )
        assert row["raw_content"].get("treatment_var") == treatment_var
        assert row["raw_content"].get("heterogeneity_score") == 0.42

        # The filter must be REAL in both directions: a non-matching
        # treatment_var must exclude the row — hydration that ignores the
        # declared filter would be a fabricated pass.
        wrong_treatment = await hooks._get_episodic_context(
            query=query,
            treatment_var=f"other_treatment_{marker}",
            outcome_var=outcome_var,
        )
        assert str(memory_id) not in {str(r.get("memory_id")) for r in wrong_treatment}, (
            "treatment_var filter is not actually filtering"
        )
    finally:
        if memory_id:
            _cleanup_episodic(memory_id)


# =============================================================================
# tool_composer — find_similar_compositions success filter (+ planner G1/G2)
# =============================================================================


async def _store_composition_row(session_id: str, marker: str, success: bool) -> str | None:
    """Seed one real composition episodic row through the #876-fixed write path."""
    from src.agents.tool_composer.memory_hooks import ToolComposerMemoryHooks

    hooks = ToolComposerMemoryHooks()
    result = {
        "composition_id": marker,
        "query": f"Compare TRx conversion drivers across segments ({marker})",
        "status": "success" if success else "failed",
        "success": success,
        "total_duration_ms": 1234,
        "decomposition": {"sub_questions": [{"id": "sq1"}]},
        "execution": {"tools_executed": 2, "tools_succeeded": 2 if success else 0},
        "response": {"confidence": 0.9 if success else 0.1},
        "plan": {"steps": [{"tool_name": "kpi_query"}, {"tool_name": "causal_effect"}]},
    }
    return await hooks.store_composition(
        session_id=session_id,
        result=result,
        brand="remibrutinib",
        region="northeast",
    )


@pytest.mark.asyncio
async def test_tool_composer_find_similar_compositions_hydrates_and_filters_success(
    episodic_rowcount_guard,
):
    """RED on main: ``find_similar_compositions`` returned ``[]`` for a
    successful composition stored seconds earlier — the ``success`` post-filter
    read ``raw_content`` off rows that never carry it, so ``successful`` was
    permanently empty and the planner's G1/G2 episodic context never fired
    (#889 site 2 + the planner.py:351/:535 family). GREEN: the successful row
    comes back hydrated with the exact payload the planner reads
    (tool_sequence / confidence / total_duration_ms), a FAILED composition
    stays excluded, and ``_format_episodic_context`` renders the real values
    instead of the all-default zeros."""
    from src.agents.tool_composer.memory_hooks import ToolComposerMemoryHooks
    from src.agents.tool_composer.planner import ToolPlanner

    session_id = str(uuid.uuid4())
    marker = f"889-tc-{uuid.uuid4().hex[:12]}"
    query = f"Compare TRx conversion drivers across segments ({marker})"

    hooks = ToolComposerMemoryHooks()
    ok_id: str | None = None
    failed_id: str | None = None
    try:
        # Seed INSIDE try so a failed second write cannot leak the first row
        # (codex R1 LOW on this suite).
        ok_id = await _store_composition_row(session_id, marker, success=True)
        assert ok_id, "store_composition (success) failed — cannot exercise the read"
        failed_id = await _store_composition_row(session_id, f"{marker}-failed", success=False)
        assert failed_id, "store_composition (failed) failed — cannot exercise the read"

        results = await hooks.find_similar_compositions(query=query, limit=5)
        by_id = {str(r.get("memory_id")): r for r in results}
        assert str(ok_id) in by_id, (
            "find_similar_compositions dropped the just-written SUCCESSFUL "
            "composition — the success post-filter ran against rows the search "
            "RPC returns WITHOUT raw_content (#889 site 2)"
        )
        assert str(failed_id) not in by_id, (
            "success filter is not actually filtering (failed composition returned)"
        )
        row = by_id[str(ok_id)]
        rc = row.get("raw_content")
        assert isinstance(rc, dict), "returned row must carry the HYDRATED raw_content dict"
        # The exact keys planner.py:351/:535 reads — proves the G1/G2 episodic
        # context downstream of this hook now receives real data.
        assert rc.get("success") is True
        assert rc.get("tool_sequence") == ["kpi_query", "causal_effect"]
        assert rc.get("confidence") == 0.9
        assert rc.get("total_duration_ms") == 1234

        # planner._format_episodic_context uses no planner state — render the
        # hydrated rows through the REAL prompt formatter and assert the real
        # values (not the .get(..., 0) defaults) reach the LLM context.
        planner = object.__new__(ToolPlanner)
        prompt_block = planner._format_episodic_context([row])
        assert "kpi_query, causal_effect" in prompt_block
        assert "0.90" in prompt_block
        assert "1234ms" in prompt_block
    finally:
        if ok_id:
            _cleanup_episodic(ok_id)
        if failed_id:
            _cleanup_episodic(failed_id)


# =============================================================================
# causal_impact — get_prior_analyses confidence filter (+ write-side confidence)
#                 and _get_episodic_context treatment/outcome filter
# =============================================================================


async def _store_causal_row(
    session_id: str, marker: str, treatment_var: str, outcome_var: str, confidence: float
) -> str | None:
    """Seed one real causal episodic row through the #788-fixed write path."""
    from src.agents.causal_impact.memory_hooks import CausalImpactMemoryHooks

    hooks = CausalImpactMemoryHooks()
    result = {
        "ate_estimate": 0.18,
        "confidence": confidence,
        "confidence_interval": [0.05, 0.31],
        "refutation_passed": True,
        "gate_decision": "proceed",
        "effect_size": "moderate",
        "model_used": "linear_dml",
        "executive_summary": f"Causal probe ({marker})",
    }
    state = {
        "treatment_var": treatment_var,
        "outcome_var": outcome_var,
        "confounders": ["specialty", "decile"],
    }
    return await hooks.store_causal_analysis(
        session_id=session_id,
        result=result,
        state=state,
        brand="remibrutinib",
        region="northeast",
    )


@pytest.mark.asyncio
async def test_causal_impact_get_prior_analyses_confidence_filter_real(episodic_rowcount_guard):
    """RED on main — DOUBLE bug (#889 site 3): (a) the confidence post-filter
    read ``raw_content`` off rows the search RPC returns WITHOUT it, so
    ``get_prior_analyses`` was permanently ``[]``; (b) deeper, the WRITE path
    never stored ``confidence`` in raw_content (only ``confidence_interval``),
    so hydration alone could not make the filter pass — the writer must
    persist the confidence it already computes for the description. GREEN:
    a high-confidence row passes ``min_confidence=0.7`` with the hydrated
    payload; raising the threshold above the stored value excludes it; and the
    docstring-declared treatment/outcome filters are real."""
    from src.agents.causal_impact.memory_hooks import CausalImpactMemoryHooks

    session_id = str(uuid.uuid4())
    marker = f"889-ci-{uuid.uuid4().hex[:12]}"
    treatment_var = f"speaker_programs_{marker}"
    outcome_var = "nbrx_share"

    hooks = CausalImpactMemoryHooks()
    memory_id: str | None = None
    try:
        # Seed INSIDE try so a failed write/assert cannot leak the row.
        memory_id = await _store_causal_row(
            session_id, marker, treatment_var, outcome_var, confidence=0.91
        )
        assert memory_id, "store_causal_analysis failed — cannot exercise get_prior_analyses"

        rows = await hooks.get_prior_analyses(
            treatment_var=treatment_var,
            outcome_var=outcome_var,
            min_confidence=0.7,
        )
        by_id = {str(r.get("memory_id")): r for r in rows}
        assert str(memory_id) in by_id, (
            "get_prior_analyses dropped the just-written high-confidence row — "
            "either the raw_content post-filter ran against RPC rows without "
            "raw_content, or the write path never stored 'confidence' (#889 "
            "site 3, double bug)"
        )
        row = by_id[str(memory_id)]
        assert isinstance(row.get("raw_content"), dict)
        assert row["raw_content"].get("confidence") == 0.91, (
            "write path must persist the gate confidence it computes — "
            "without it the min_confidence filter can never pass"
        )
        assert row["raw_content"].get("ate_estimate") == 0.18

        # Threshold must be REAL: above the stored confidence -> excluded.
        too_strict = await hooks.get_prior_analyses(
            treatment_var=treatment_var,
            outcome_var=outcome_var,
            min_confidence=0.95,
        )
        assert str(memory_id) not in {str(r.get("memory_id")) for r in too_strict}, (
            "min_confidence filter is not actually filtering"
        )

        # Docstring-declared treatment filter must be real too.
        wrong_treatment = await hooks.get_prior_analyses(
            treatment_var=f"other_{marker}",
            outcome_var=outcome_var,
            min_confidence=0.7,
        )
        assert str(memory_id) not in {str(r.get("memory_id")) for r in wrong_treatment}, (
            "treatment_var filter is not actually filtering"
        )
    finally:
        if memory_id:
            _cleanup_episodic(memory_id)


@pytest.mark.asyncio
async def test_causal_impact_episodic_context_with_vars_returns_hydrated_row(
    episodic_rowcount_guard,
):
    """RED on main: same in-file family as site 3 — ``_get_episodic_context``
    post-filters treatment/outcome on ``raw_content`` the RPC never returns,
    so the filtered path (the production get_memory_context call shape)
    returned ``[]`` for a row stored seconds earlier. The issue table listed
    get_prior_analyses (:629); this is the identical pattern at :230, fixed
    together. GREEN: hydrated row returned, filter real both directions."""
    from src.agents.causal_impact.memory_hooks import CausalImpactMemoryHooks

    session_id = str(uuid.uuid4())
    marker = f"889-cic-{uuid.uuid4().hex[:12]}"
    treatment_var = f"sample_drops_{marker}"
    outcome_var = "trx_volume"
    query = f"Causal analysis: {treatment_var} -> {outcome_var} probe ({marker})"

    hooks = CausalImpactMemoryHooks()
    memory_id: str | None = None
    try:
        # Seed INSIDE try so a failed write/assert cannot leak the row.
        memory_id = await _store_causal_row(
            session_id, marker, treatment_var, outcome_var, confidence=0.85
        )
        assert memory_id, "store_causal_analysis failed — cannot exercise the filtered read"

        results = await hooks._get_episodic_context(
            query=query,
            treatment_var=treatment_var,
            outcome_var=outcome_var,
        )
        by_id = {str(r.get("memory_id")): r for r in results}
        assert str(memory_id) in by_id, (
            "_get_episodic_context(treatment_var/outcome_var) dropped the "
            "just-written row — raw_content post-filter vs RPC rows without "
            "raw_content (#889, causal_impact in-file family)"
        )
        assert isinstance(by_id[str(memory_id)].get("raw_content"), dict)

        wrong_outcome = await hooks._get_episodic_context(
            query=query,
            treatment_var=treatment_var,
            outcome_var="other_outcome",
        )
        assert str(memory_id) not in {str(r.get("memory_id")) for r in wrong_outcome}, (
            "outcome_var filter is not actually filtering"
        )
    finally:
        if memory_id:
            _cleanup_episodic(memory_id)


# =============================================================================
# scope_definer — get_prior_scopes validation_passed filter
# =============================================================================


async def _store_scope_row(
    session_id: str, marker: str, problem_type: str, validation_passed: bool
) -> str | None:
    """Seed one real scope episodic row through the #749-fixed legacy write path."""
    from src.agents.ml_foundation.scope_definer.memory_hooks import (
        ScopeDefinerMemoryHooks,
    )

    hooks = ScopeDefinerMemoryHooks()
    result = {
        "experiment_id": marker,
        "experiment_name": f"scope definition probe {marker}",
        "success_criteria": {"auc": 0.75},
    }
    state = {
        "inferred_problem_type": problem_type,
        "inferred_target_variable": "churn_90d",
        "business_objective": f"scope definition probe for {problem_type} ({marker})",
        "required_features": ["rx_history", "specialty"],
        "validation_passed": validation_passed,
        "use_case": "patient_retention",
    }
    return await hooks.store_scope_definition(
        session_id=session_id,
        result=result,
        state=state,
        brand="remibrutinib",
        region="northeast",
    )


@pytest.mark.asyncio
async def test_scope_definer_get_prior_scopes_validation_filter_real(episodic_rowcount_guard):
    """RED on main: ``get_prior_scopes`` returned ``[]`` for a validated scope
    stored seconds earlier — the ``validation_passed`` post-filter read
    ``raw_content`` off rows the search RPC returns WITHOUT it, so every row
    was excluded (#889 site 4). GREEN: the validated row comes back hydrated;
    a row written with ``validation_passed=False`` (possible via the direct
    store hook even though contribute_to_memory gates it) stays excluded; and
    the docstring-declared problem_type filter is real."""
    from src.agents.ml_foundation.scope_definer.memory_hooks import (
        ScopeDefinerMemoryHooks,
    )

    session_id = str(uuid.uuid4())
    marker = f"889-sd-{uuid.uuid4().hex[:12]}"
    problem_type = f"churn_prediction_{marker}"

    hooks = ScopeDefinerMemoryHooks()
    ok_id: str | None = None
    bad_id: str | None = None
    try:
        # Seed INSIDE try so a failed second write cannot leak the first row
        # (codex R1 LOW on this suite).
        ok_id = await _store_scope_row(session_id, marker, problem_type, validation_passed=True)
        assert ok_id, "store_scope_definition (validated) failed — cannot exercise the read"
        bad_id = await _store_scope_row(
            session_id, f"{marker}-unvalidated", problem_type, validation_passed=False
        )
        assert bad_id, "store_scope_definition (unvalidated) failed — cannot exercise the read"

        rows = await hooks.get_prior_scopes(problem_type=problem_type)
        by_id = {str(r.get("memory_id")): r for r in rows}
        assert str(ok_id) in by_id, (
            "get_prior_scopes dropped the just-written VALIDATED scope — the "
            "validation_passed post-filter ran against rows the search RPC "
            "returns WITHOUT raw_content (#889 site 4)"
        )
        assert str(bad_id) not in by_id, (
            "validation_passed filter is not actually filtering (unvalidated scope returned)"
        )
        row = by_id[str(ok_id)]
        assert isinstance(row.get("raw_content"), dict)
        assert row["raw_content"].get("validation_passed") is True
        assert row["raw_content"].get("problem_type") == problem_type

        # Docstring-declared problem_type filter must be real.
        wrong_type = await hooks.get_prior_scopes(problem_type=f"forecasting_{marker}")
        assert str(ok_id) not in {str(r.get("memory_id")) for r in wrong_type}, (
            "problem_type filter is not actually filtering"
        )
    finally:
        if ok_id:
            _cleanup_episodic(ok_id)
        if bad_id:
            _cleanup_episodic(bad_id)
