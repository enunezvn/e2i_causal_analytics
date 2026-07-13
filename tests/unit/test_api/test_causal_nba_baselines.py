# tests/unit/test_api/test_causal_nba_baselines.py
"""#1188: nba_triggers baseline (ANCOVA) loader + /variables surfacing.

The RCT grain gains an OPT-IN baseline join: triggers.patient_id ->
patient_journeys pre-treatment baselines (disease_severity, age_at_diagnosis,
academic_hcp, geographic_region one-hot). The join must be fail-closed
(allowlist, drop rows missing treatment/outcome/baseline) and NEVER activate
unless baselines were explicitly requested.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from src.api.routes import causal as causal_routes


def _fake_trigger_rows():
    return [
        {
            "control_group_flag": True,
            "action_taken": None,
            "patient_id": "pt_000001",
        },
        {
            "control_group_flag": False,
            "action_taken": "called_patient",
            "patient_id": "pt_000002",
        },
        {
            "control_group_flag": False,
            "action_taken": None,
            "patient_id": "pt_000003",
        },
        # Orphan trigger: no matching patient row -> dropped in baseline mode.
        {
            "control_group_flag": True,
            "action_taken": "sent_info",
            "patient_id": "pt_999999",
        },
    ]


def _fake_patient_baseline_rows():
    return [
        {
            "patient_id": "pt_000001",
            "disease_severity": 7.5,
            "age_at_diagnosis": 61,
            "academic_hcp": 1,
            "geographic_region": "south",
        },
        {
            "patient_id": "pt_000002",
            "disease_severity": 2.5,
            "age_at_diagnosis": 34,
            "academic_hcp": 0,
            "geographic_region": "west",
        },
        {
            "patient_id": "pt_000003",
            "disease_severity": 5.0,
            "age_at_diagnosis": 50,
            "academic_hcp": 0,
            "geographic_region": "south",
        },
    ]


def _patched_reads():
    return (
        patch.object(causal_routes, "get_async_supabase_client", AsyncMock(return_value=object())),
        patch.object(
            causal_routes,
            "_load_trigger_question_rows",
            AsyncMock(return_value=_fake_trigger_rows()),
        ),
        patch.object(
            causal_routes,
            "_load_patient_baseline_rows",
            AsyncMock(return_value=_fake_patient_baseline_rows()),
        ),
    )


@pytest.mark.asyncio
async def test_nba_baseline_join_builds_baseline_columns():
    p1, p2, p3 = _patched_reads()
    with p1, p2, p3:
        df, select_cols = await causal_routes._load_agent_estimation_frame(
            dataset="nba_triggers",
            treatment_var="control_group_flag",
            outcome_var="action_taken",
            covariates=[],
            limit=1500,
            brand=None,
            baseline_covariates=[
                "disease_severity",
                "age_at_diagnosis",
                "academic_hcp",
                "geographic_region",
            ],
        )
    # Orphan trigger dropped; the 3 joined rows remain.
    assert len(df) == 3
    assert "disease_severity" in df.columns
    assert "age_at_diagnosis" in df.columns
    assert "academic_hcp" in df.columns
    # geographic_region one-hot expanded (drop_first over sorted levels
    # {south, west} -> reference=south, dummy=west).
    assert "geographic_region=west" in df.columns
    assert "geographic_region" not in df.columns
    # Numeric coercion + designed-NULL outcome fill preserved from the
    # single-table path: NULL action_taken -> 0.0, presence -> 1.0.
    assert sorted(df["action_taken"].tolist()) == [0.0, 0.0, 1.0]
    assert set(select_cols) >= {
        "control_group_flag",
        "action_taken",
        "disease_severity",
        "age_at_diagnosis",
        "academic_hcp",
        "geographic_region=west",
    }


@pytest.mark.asyncio
async def test_nba_baseline_join_drops_rows_missing_baseline():
    rows = _fake_patient_baseline_rows()
    rows[0] = {**rows[0], "disease_severity": None}  # missing baseline
    p1, _, p3 = _patched_reads()
    with (
        p1,
        patch.object(
            causal_routes,
            "_load_trigger_question_rows",
            AsyncMock(return_value=_fake_trigger_rows()),
        ),
        patch.object(causal_routes, "_load_patient_baseline_rows", AsyncMock(return_value=rows)),
    ):
        df, _ = await causal_routes._load_agent_estimation_frame(
            dataset="nba_triggers",
            treatment_var="control_group_flag",
            outcome_var="action_taken",
            covariates=[],
            limit=1500,
            brand=None,
            baseline_covariates=["disease_severity", "age_at_diagnosis"],
        )
    # pt_000001 (missing severity) and the orphan are dropped.
    assert len(df) == 2
    assert df["disease_severity"].notna().all()


@pytest.mark.asyncio
async def test_nba_without_baselines_keeps_single_table_path():
    """Default (no baselines requested): the loader must NOT join patients —
    today's single-table read path is untouched."""
    import src.memory.services.factories as factories

    join_spy = AsyncMock(return_value=_fake_patient_baseline_rows())

    class _FakeQuery:
        def select(self, *_a, **_k):
            return self

        def eq(self, *_a, **_k):
            return self

        def limit(self, *_a, **_k):
            return self

        async def execute(self):
            class _R:
                data = [
                    {"control_group_flag": True, "action_taken": None},
                    {"control_group_flag": False, "action_taken": "sent_info"},
                ]

            return _R()

    class _FakeClient:
        def table(self, name):
            assert name == "triggers"
            return _FakeQuery()

    async def _fake_factory():
        return _FakeClient()

    original = factories.get_async_supabase_client
    factories.get_async_supabase_client = _fake_factory
    try:
        with patch.object(causal_routes, "_load_patient_baseline_rows", join_spy):
            df, select_cols = await causal_routes._load_agent_estimation_frame(
                dataset="nba_triggers",
                treatment_var="control_group_flag",
                outcome_var="action_taken",
                covariates=[],
                limit=1500,
                brand=None,
            )
    finally:
        factories.get_async_supabase_client = original
    join_spy.assert_not_awaited()
    assert list(select_cols) == ["control_group_flag", "action_taken"]
    assert len(df) == 2


@pytest.mark.asyncio
async def test_nba_baseline_join_rejects_disallowed_baseline():
    """Post-treatment columns can never ride in via the baseline channel."""
    p1, p2, p3 = _patched_reads()
    with p1, p2, p3, pytest.raises(HTTPException) as exc:
        await causal_routes._load_agent_estimation_frame(
            dataset="nba_triggers",
            treatment_var="control_group_flag",
            outcome_var="action_taken",
            covariates=[],
            limit=1500,
            brand=None,
            baseline_covariates=["adherence_rate"],
        )
    assert exc.value.status_code == 400


@pytest.mark.unit
def test_resolve_baselines_unsupported_dataset_fails_400():
    """adjust_baselines=True on a dataset with no curated baseline role must be
    an honest 400, not a silent no-op."""
    with pytest.raises(HTTPException) as exc:
        causal_routes._resolve_requested_baselines("patient_journeys", True)
    assert exc.value.status_code == 400


@pytest.mark.unit
def test_resolve_baselines_default_off_is_empty():
    assert causal_routes._resolve_requested_baselines("nba_triggers", False) == []


@pytest.mark.unit
def test_resolve_baselines_on_returns_curated_set():
    resolved = causal_routes._resolve_requested_baselines("nba_triggers", True)
    assert "disease_severity" in resolved
    assert "age_at_diagnosis" in resolved


@pytest.mark.asyncio
async def test_list_causal_variables_nba_includes_baseline_candidates():
    """/variables must surface the curated baseline role so the FE can offer
    the opt-in toggle (data-driven, no hardcoded FE list)."""
    import src.memory.services.factories as factories
    from src.api.dependencies.auth import TEST_USER

    class _FakeQuery:
        def select(self, *_a, **_k):
            return self

        def limit(self, *_a, **_k):
            return self

        async def execute(self):
            class _R:
                data = [
                    {
                        "control_group_flag": True,
                        "acceptance_status": "accepted",
                        "action_taken": None,
                        "conversion_flag": None,
                    }
                ]

            return _R()

    class _FakeClient:
        def table(self, _name):
            return _FakeQuery()

    async def _fake_factory():
        return _FakeClient()

    original = factories.get_async_supabase_client
    factories.get_async_supabase_client = _fake_factory
    try:
        resp = await causal_routes.list_causal_variables(dataset="nba_triggers", user=TEST_USER)
    finally:
        factories.get_async_supabase_client = original

    assert "disease_severity" in resp.baseline_candidates
    assert "age_at_diagnosis" in resp.baseline_candidates
    # The de-confounding covariate list stays empty for the RCT.
    assert resp.covariate_candidates == []


@pytest.mark.asyncio
async def test_trigger_question_rows_paged_beyond_single_limit():
    """codex iter-1 MED: the trigger read must PAGE the full table (post-join
    capping happens in the builder) — a single .limit(limit) read underfills
    the joined sample whenever orphans / missing baselines are dropped, and a
    20-page cap would silently truncate 37.5k live rows."""
    calls = []

    class _FakeQuery:
        def select(self, *_a, **_k):
            return self

        def eq(self, *_a, **_k):
            return self

        def range(self, lo, hi):
            calls.append((lo, hi))
            return self

        def limit(self, *_a, **_k):
            raise AssertionError("trigger read must page with .range(), not .limit()")

        async def execute(self):
            lo, hi = calls[-1]
            page_size = hi - lo + 1

            class _R:
                data = [{"control_group_flag": True, "action_taken": None, "patient_id": "p"}] * (
                    page_size if lo == 0 else 5
                )

            return _R()

    class _FakeClient:
        def table(self, _name):
            return _FakeQuery()

    rows = await causal_routes._load_trigger_question_rows(
        _FakeClient(), ["control_group_flag", "action_taken"], None
    )
    # First page full -> second page requested; second page short -> stop.
    assert len(calls) == 2
    assert len(rows) == (calls[0][1] - calls[0][0] + 1) + 5


@pytest.mark.asyncio
async def test_patient_baseline_rows_page_past_te_cap():
    """The patient read must cover the WHOLE patient_journeys table (25k+ rows
    live) — capping at _TE_MAX_PAGES(20)x1000 silently drops ~5k patients'
    triggers from the joined sample."""
    calls = []

    class _FakeQuery:
        def select(self, *_a, **_k):
            return self

        def range(self, lo, hi):
            calls.append((lo, hi))
            return self

        async def execute(self):
            lo, hi = calls[-1]
            page_size = hi - lo + 1
            # 25 full pages then a short one (~25k rows like live).
            n_full = 25

            class _R:
                data = (
                    [{"patient_id": f"p{lo}", "disease_severity": 5.0}] * page_size
                    if lo < n_full * page_size
                    else [{"patient_id": "plast", "disease_severity": 5.0}] * 18
                )

            return _R()

    class _FakeClient:
        def table(self, _name):
            return _FakeQuery()

    rows = await causal_routes._load_patient_baseline_rows(_FakeClient(), ["disease_severity"])
    page = calls[0][1] - calls[0][0] + 1
    assert len(rows) == 25 * page + 18, (
        f"paged read stopped early: {len(rows)} rows over {len(calls)} pages"
    )
