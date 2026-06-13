"""#894 faithful integration: the HAS_PROVENANCE family, live DB.

Migrations 063/067/069 tagged 26 tables with ``is_synthetic``, but 10
repository classes plus 3 non-repository read families never set
``HAS_PROVENANCE = True`` (or never applied ``apply_provenance_filter`` in
their bespoke reads), so synthetic substrate rows flowed into real-mode
reads. Live substrate at filing time (2026-06-12):

    ab_experiment_assignments  216000/216000 synthetic
    ab_experiment_results         360/360
    ml_model_registry             720/722 (the 2 real rows are the only
                                  artifact-bearing serving models — #840/#857)
    ml_training_runs              720/720
    ml_deployments                360/360
    user_sessions               10000/10000
    learning_signals              300/300
    causal_paths                  250/250
    ml_observability_spans        600/816 synthetic

These tests drive the REAL repositories / node helpers / module functions
against the live DB, red-first per family:

  * RED   — real-mode reads return the seeded synthetic row (or 42703 for the
            causal_path bespoke reads, which filter on columns that do not
            exist in the live schema).
  * GREEN — real-mode reads default-exclude synthetic rows while
            ``include_synthetic=True`` opt-in still reaches the substrate.

Self-cleaning: every test brackets the affected table's row count and deletes
its seeded rows; post-count must equal pre-count.

Run with the shared-DB lock::

    flock -w 2400 /tmp/e2i_dbtest.lock -c \\
        'E2I_DB_INTEGRATION=1 PYTHONPATH=$PWD .venv/bin/pytest -n0 \\
         tests/integration/test_has_provenance_family_894.py'
"""

import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest

_GATE = os.environ.get("E2I_DB_INTEGRATION") == "1"
_HAS_CREDS = bool(os.environ.get("SUPABASE_URL"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_GATE and _HAS_CREDS),
        reason="faithful real-DB HAS_PROVENANCE family test; "
        "set E2I_DB_INTEGRATION=1 + creds in .env",
    ),
]


# =============================================================================
# Client fixtures (the established realdb-suite idiom: reset the factory's
# cached async client around each test so the per-function event loop never
# reuses a pooled connection bound to a closed loop)
# =============================================================================


@pytest.fixture
async def async_client():
    import src.memory.services.factories as factories
    from src.memory.services.factories import get_async_supabase_client

    factories._async_supabase_client = None
    client = await get_async_supabase_client()
    assert client is not None, "async supabase client unavailable (creds?)"
    yield client
    factories._async_supabase_client = None


@pytest.fixture
def sync_client():
    from src.repositories import get_supabase_client

    client = get_supabase_client()
    assert client is not None, "sync supabase client unavailable (creds?)"
    return client


async def _acount(client, table: str, pk: str = "id") -> int:
    res = await client.table(table).select(pk, count="exact").limit(1).execute()
    return int(res.count)


def _scount(client, table: str, pk: str = "id") -> int:
    res = client.table(table).select(pk, count="exact").limit(1).execute()
    return int(res.count)


# =============================================================================
# Family: ab_experiment_assignments (ABExperimentRepository + monitor readers)
# =============================================================================


@pytest.fixture
def seeded_ab(sync_client):
    """A real parent experiment + one synthetic and one real assignment."""
    marker = uuid.uuid4().hex[:8]
    pre_exp = _scount(sync_client, "ml_experiments")
    pre_asg = _scount(sync_client, "ab_experiment_assignments")

    exp = (
        sync_client.table("ml_experiments")
        .insert(
            {
                "experiment_name": f"e2i894-ab-{marker}",
                "prediction_target": f"e2i894_{marker}",
                "status": "running",
                "is_synthetic": False,
            }
        )
        .execute()
    )
    exp_id = exp.data[0]["id"]

    rows = [
        {
            "experiment_id": exp_id,
            "unit_id": f"hcp-real-{marker}",
            "unit_type": "hcp",
            "variant": "control",
            "randomization_method": "simple",
            "is_synthetic": False,
        },
        {
            "experiment_id": exp_id,
            "unit_id": f"hcp-synth-{marker}",
            "unit_type": "hcp",
            "variant": "treatment",
            "randomization_method": "simple",
            "is_synthetic": True,
        },
    ]
    sync_client.table("ab_experiment_assignments").insert(rows).execute()
    try:
        yield {"experiment_id": exp_id, "marker": marker}
    finally:
        sync_client.table("ab_experiment_assignments").delete().eq(
            "experiment_id", exp_id
        ).execute()
        sync_client.table("ml_experiments").delete().eq("id", exp_id).execute()
        assert _scount(sync_client, "ml_experiments") == pre_exp
        assert _scount(sync_client, "ab_experiment_assignments") == pre_asg


async def test_ab_get_assignments_default_excludes_synthetic(sync_client, seeded_ab):
    """Real-mode get_assignments must not return the synthetic assignment."""
    from uuid import UUID

    from src.repositories.ab_experiment import ABExperimentRepository

    repo = ABExperimentRepository(supabase_client=sync_client)
    assignments = await repo.get_assignments(UUID(seeded_ab["experiment_id"]))

    units = {a.unit_id for a in assignments}
    assert f"hcp-synth-{seeded_ab['marker']}" not in units, (
        "real-mode get_assignments leaked a synthetic assignment"
    )
    assert units == {f"hcp-real-{seeded_ab['marker']}"}


async def test_ab_get_assignments_opt_in_returns_synthetic(sync_client, seeded_ab):
    from uuid import UUID

    from src.repositories.ab_experiment import ABExperimentRepository

    repo = ABExperimentRepository(supabase_client=sync_client)
    assignments = await repo.get_assignments(
        UUID(seeded_ab["experiment_id"]), include_synthetic=True
    )
    units = {a.unit_id for a in assignments}
    assert units == {
        f"hcp-real-{seeded_ab['marker']}",
        f"hcp-synth-{seeded_ab['marker']}",
    }


async def test_srm_variant_counts_exclude_synthetic(async_client, seeded_ab):
    """experiment_monitor's direct assignment read must default-exclude."""
    from src.agents.experiment_monitor.nodes.srm_detector import SRMDetectorNode

    node = SRMDetectorNode()
    counts = await node._get_variant_counts(async_client, seeded_ab["experiment_id"])

    # the synthetic row was the only 'treatment' assignment
    assert counts.get("treatment", 0) == 0, (
        f"SRM variant counts included a synthetic assignment: {counts}"
    )
    assert counts.get("control") == 1


# =============================================================================
# Family: ml_experiments (repository + health_checker enumeration + route)
# =============================================================================


@pytest.fixture
def seeded_experiments(sync_client):
    """One real running + one synthetic running experiment."""
    marker = uuid.uuid4().hex[:8]
    pre = _scount(sync_client, "ml_experiments")
    rows = [
        {
            "experiment_name": f"e2i894-real-{marker}",
            "prediction_target": f"e2i894_{marker}",
            "status": "running",
            "is_synthetic": False,
        },
        {
            "experiment_name": f"e2i894-synth-{marker}",
            "prediction_target": f"e2i894_{marker}",
            "status": "running",
            "is_synthetic": True,
        },
    ]
    res = sync_client.table("ml_experiments").insert(rows).execute()
    ids = {r["experiment_name"]: r["id"] for r in res.data}
    try:
        yield {"marker": marker, "ids": ids}
    finally:
        sync_client.table("ml_experiments").delete().in_("id", list(ids.values())).execute()
        assert _scount(sync_client, "ml_experiments") == pre


async def test_health_checker_enumeration_excludes_synthetic(async_client, seeded_experiments):
    """The AB sweep chain's upstream enumeration must not see synthetic ids."""
    from src.agents.experiment_monitor.nodes.health_checker import HealthCheckerNode

    node = HealthCheckerNode()
    experiments = await node._get_experiments(async_client, {"check_all_active": True})

    ids = {e["id"] for e in experiments}
    marker = seeded_experiments["marker"]
    assert seeded_experiments["ids"][f"e2i894-real-{marker}"] in ids
    assert seeded_experiments["ids"][f"e2i894-synth-{marker}"] not in ids, (
        "health_checker enumerated a synthetic experiment in real mode"
    )


async def test_active_experiment_count_excludes_synthetic(sync_client, seeded_experiments):
    """The Home QUICK_STATS active-count must not count synthetic experiments."""
    from src.api.routes.experiments import active_experiment_count

    result = await active_experiment_count()

    expected = (
        sync_client.table("ml_experiments")
        .select("id", count="exact")
        .eq("status", "running")
        .eq("is_synthetic", False)
        .limit(1)
        .execute()
    ).count
    unfiltered = (
        sync_client.table("ml_experiments")
        .select("id", count="exact")
        .eq("status", "running")
        .limit(1)
        .execute()
    ).count
    assert result["active_count"] == expected, (
        f"active_count={result['active_count']} != non-synthetic running={expected}"
    )
    # the live substrate carries synthetic running experiments, so the filter
    # must actually bite (guards against a vacuous pass on an empty table)
    assert unfiltered > expected


async def test_ml_experiment_get_by_name_excludes_synthetic(async_client, seeded_experiments):
    """A synthetic experiment name must not resolve in real mode (opt-in does)."""
    from src.repositories.ml_experiment import MLExperimentRepository

    repo = MLExperimentRepository(supabase_client=async_client)
    marker = seeded_experiments["marker"]

    assert await repo.get_by_name(f"e2i894-synth-{marker}") is None
    assert await repo.get_by_name(f"e2i894-real-{marker}") is not None
    opted = await repo.get_by_name(f"e2i894-synth-{marker}", include_synthetic=True)
    assert opted is not None


# =============================================================================
# Family: ml_model_registry (serving reads must stay loadable + never synthetic)
# =============================================================================


@pytest.fixture
def seeded_registry(sync_client):
    """A synthetic production-stage model WITH an artifact_path (the live hole:
    mlops_generator stamps stage='production'; only the NULL artifact_path kept
    it out of serving until now)."""
    marker = uuid.uuid4().hex[:8]
    pre_reg = _scount(sync_client, "ml_model_registry")
    pre_exp = _scount(sync_client, "ml_experiments")

    exp = (
        sync_client.table("ml_experiments")
        .insert(
            {
                "experiment_name": f"e2i894-reg-{marker}",
                "prediction_target": "csu_treatment_initiation",
                "status": "running",
                "is_synthetic": True,
            }
        )
        .execute()
    )
    exp_id = exp.data[0]["id"]
    reg = (
        sync_client.table("ml_model_registry")
        .insert(
            {
                "experiment_id": exp_id,
                "model_name": f"e2i894_synth_model_{marker}",
                "model_version": "1.0.0",
                "algorithm": "logistic_regression",
                "stage": "production",
                "is_champion": False,
                "artifact_path": f"/tmp/e2i894/{marker}.joblib",
                "is_synthetic": True,
            }
        )
        .execute()
    )
    reg_id = reg.data[0]["id"]
    try:
        yield {"marker": marker, "model_name": f"e2i894_synth_model_{marker}"}
    finally:
        sync_client.table("ml_model_registry").delete().eq("id", reg_id).execute()
        sync_client.table("ml_experiments").delete().eq("id", exp_id).execute()
        assert _scount(sync_client, "ml_model_registry") == pre_reg
        assert _scount(sync_client, "ml_experiments") == pre_exp


async def test_get_models_for_target_excludes_synthetic_and_keeps_real(
    async_client, sync_client, seeded_registry
):
    """Serving resolution must drop the synthetic prod row but keep the REAL
    registered champions (the #840/#857 AUC-0.83 pair) loadable."""
    from src.repositories.ml_experiment import MLModelRegistryRepository

    repo = MLModelRegistryRepository(supabase_client=async_client)
    names = await repo.get_models_for_target("csu_treatment_initiation")

    assert seeded_registry["model_name"] not in names, (
        "serving read returned a SYNTHETIC production model"
    )

    # the live real serving models must remain resolvable
    real_serving = (
        sync_client.table("ml_model_registry")
        .select("model_name")
        .eq("stage", "production")
        .eq("is_synthetic", False)
        .not_.is_("artifact_path", "null")
        .execute()
    ).data
    expected_names = {r["model_name"] for r in real_serving}
    assert expected_names, "live DB lost its real serving models — investigate before merging"
    assert expected_names.issubset(set(names))


async def test_registry_get_by_name_version_excludes_synthetic(async_client, seeded_registry):
    from src.repositories.ml_experiment import MLModelRegistryRepository

    repo = MLModelRegistryRepository(supabase_client=async_client)
    assert await repo.get_by_name_version(seeded_registry["model_name"], "1.0.0") is None
    opted = await repo.get_by_name_version(
        seeded_registry["model_name"], "1.0.0", include_synthetic=True
    )
    assert opted is not None


# =============================================================================
# Family: ml_observability_spans
# =============================================================================


@pytest.fixture
def seeded_spans(sync_client):
    marker = uuid.uuid4().hex[:8]
    pre = _scount(sync_client, "ml_observability_spans")
    now = datetime.now(timezone.utc)
    trace_id = f"e2i894-trace-{marker}"
    rows = [
        {
            "trace_id": trace_id,
            "span_id": f"e2i894-span-real-{marker}",
            "agent_name": "orchestrator",
            "agent_tier": "coordination",
            "started_at": now.isoformat(),
            "status": "success",
            "is_synthetic": False,
        },
        {
            "trace_id": trace_id,
            "span_id": f"e2i894-span-synth-{marker}",
            "agent_name": "orchestrator",
            "agent_tier": "coordination",
            "started_at": now.isoformat(),
            "status": "success",
            "is_synthetic": True,
        },
    ]
    sync_client.table("ml_observability_spans").insert(rows).execute()
    try:
        yield {"marker": marker, "trace_id": trace_id}
    finally:
        sync_client.table("ml_observability_spans").delete().eq("trace_id", trace_id).execute()
        assert _scount(sync_client, "ml_observability_spans") == pre


async def test_spans_by_trace_default_excludes_synthetic(sync_client, seeded_spans):
    from src.repositories.observability_span import ObservabilitySpanRepository

    repo = ObservabilitySpanRepository(supabase_client=sync_client)
    spans = await repo.get_spans_by_trace_id(seeded_spans["trace_id"])

    span_ids = {s.span_id for s in spans}
    assert span_ids == {f"e2i894-span-real-{seeded_spans['marker']}"}, (
        f"trace read leaked synthetic spans: {span_ids}"
    )


async def test_spans_by_trace_opt_in_returns_synthetic(sync_client, seeded_spans):
    from src.repositories.observability_span import ObservabilitySpanRepository

    repo = ObservabilitySpanRepository(supabase_client=sync_client)
    spans = await repo.get_spans_by_trace_id(seeded_spans["trace_id"], include_synthetic=True)
    assert {s.span_id for s in spans} == {
        f"e2i894-span-real-{seeded_spans['marker']}",
        f"e2i894-span-synth-{seeded_spans['marker']}",
    }


# =============================================================================
# Family: sentinel _eval_new_causal_path + causal_path bespoke reads
# =============================================================================


@pytest.fixture
def seeded_paths(sync_client):
    marker = uuid.uuid4().hex[:8]
    pre = _scount(sync_client, "causal_paths", pk="path_id")
    token = uuid.uuid4().hex[:10]
    start_node = f"e2i894s{marker}"
    common = {
        "discovery_date": "2026-06-12",
        "start_node": start_node,
        "end_node": "trx_growth",
        "causal_effect_size": 0.1,
        "confidence_level": 0.9,
        "brand": f"e2i894-{marker}",
        "data_split": "unassigned",
        "causal_chain": {"nodes": [start_node, "trx_growth"]},
    }
    synthetic = {**common, "path_id": f"T894{token}S", "is_synthetic": True}
    real = {**common, "path_id": f"T894{token}R", "is_synthetic": False}
    sync_client.table("causal_paths").insert([synthetic, real]).execute()
    try:
        yield {
            "marker": marker,
            "brand": common["brand"],
            "start_node": start_node,
            "synthetic_id": synthetic["path_id"],
            "real_id": real["path_id"],
        }
    finally:
        sync_client.table("causal_paths").delete().in_(
            "path_id", [synthetic["path_id"], real["path_id"]]
        ).execute()
        assert _scount(sync_client, "causal_paths", pk="path_id") == pre


async def test_sentinel_new_causal_path_excludes_synthetic(seeded_paths):
    """The 5-min sentinel scan must not fire on planted synthetic paths."""
    from src.memory.sentinels.registry import _eval_new_causal_path

    matches = await _eval_new_causal_path({}, seeded_paths["brand"], None)
    row_ids = {m["row_id"] for m in matches}
    assert seeded_paths["synthetic_id"] not in row_ids, (
        "sentinel evaluated a synthetic causal path in real mode"
    )
    assert seeded_paths["real_id"] in row_ids


async def test_sentinel_new_causal_path_opt_in(seeded_paths):
    from src.memory.sentinels.registry import _eval_new_causal_path

    matches = await _eval_new_causal_path({"include_synthetic": True}, seeded_paths["brand"], None)
    row_ids = {m["row_id"] for m in matches}
    assert {seeded_paths["synthetic_id"], seeded_paths["real_id"]} <= row_ids


async def test_causal_path_get_paths_for_cause_returns_rows(async_client, seeded_paths):
    """Pre-fix RED: filtered on a nonexistent ``cause`` column -> 42703."""
    from src.repositories.causal_path import CausalPathRepository

    repo = CausalPathRepository(supabase_client=async_client)
    rows = await repo.get_paths_for_cause(seeded_paths["start_node"])

    ids = {r["path_id"] for r in rows}
    assert ids == {seeded_paths["real_id"]}, (
        f"get_paths_for_cause should return exactly the real seeded row, got {ids}"
    )


async def test_causal_path_get_path_between_returns_row(async_client, seeded_paths):
    from src.repositories.causal_path import CausalPathRepository

    repo = CausalPathRepository(supabase_client=async_client)
    row = await repo.get_path_between(seeded_paths["start_node"], "trx_growth")
    assert row is not None and row["path_id"] == seeded_paths["real_id"]


async def test_causal_path_get_by_id_uses_path_id_pk(async_client, seeded_paths):
    """Pre-fix RED: filtered ``.eq('id', ...)`` but the PK is ``path_id``."""
    from src.repositories.causal_path import CausalPathRepository

    repo = CausalPathRepository(supabase_client=async_client)
    row = await repo.get_by_id(seeded_paths["real_id"])
    assert row is not None and row["path_id"] == seeded_paths["real_id"]

    # a synthetic id must not resolve in real mode; opt-in resolves it
    assert await repo.get_by_id(seeded_paths["synthetic_id"]) is None
    opted = await repo.get_by_id(seeded_paths["synthetic_id"], include_synthetic=True)
    assert opted is not None


async def test_ml_data_loader_sample_excludes_synthetic_causal_paths(sync_client, seeded_paths):
    """causal_paths belongs in PROVENANCE_TAGGED_TABLES (migration 063:18);
    the loader's stale pre-063 exclusion leaked synthetic rows into samples."""
    from src.repositories.ml_data_loader import MLDataLoader

    loader = MLDataLoader(supabase_client=sync_client)
    df = await loader.load_table_sample(
        "causal_paths", limit=400, columns=["path_id", "is_synthetic"]
    )
    assert not df.empty, "expected at least the seeded real row"
    assert not df["is_synthetic"].any(), (
        "MLDataLoader real-mode sample returned synthetic causal_paths"
    )

    df_opt = await loader.load_table_sample(
        "causal_paths", limit=400, columns=["path_id", "is_synthetic"], include_synthetic=True
    )
    assert df_opt["is_synthetic"].any()


# =============================================================================
# Family: learning_signals (procedural memory recall)
# =============================================================================


@pytest.fixture
def seeded_signals(sync_client):
    """One real + one synthetic signal. ``rated_agent`` is the e2i_agent_name
    ENUM, so isolation rides on the returned signal_ids, not a marker value."""
    agent = "tool_composer"
    pre = _scount(sync_client, "learning_signals", pk="signal_id")
    rows = [
        {
            "signal_type": "rating",
            "signal_value": 0.9,
            "rated_agent": agent,
            "is_training_example": True,
            "dspy_metric_value": 0.95,
            "is_synthetic": False,
        },
        {
            "signal_type": "rating",
            "signal_value": 0.9,
            "rated_agent": agent,
            "is_training_example": True,
            "dspy_metric_value": 0.95,
            "is_synthetic": True,
        },
    ]
    res = sync_client.table("learning_signals").insert(rows).execute()
    ids = [r["signal_id"] for r in res.data]
    try:
        yield {
            "agent": agent,
            "real_id": res.data[0]["signal_id"],
            "synthetic_id": res.data[1]["signal_id"],
        }
    finally:
        sync_client.table("learning_signals").delete().in_("signal_id", ids).execute()
        assert _scount(sync_client, "learning_signals", pk="signal_id") == pre


async def test_recent_signals_exclude_synthetic(seeded_signals):
    from src.memory.procedural_memory import get_recent_signals

    signals = await get_recent_signals(limit=500, agent_name=seeded_signals["agent"])
    ids = {s["signal_id"] for s in signals}
    assert seeded_signals["real_id"] in ids
    assert seeded_signals["synthetic_id"] not in ids, (
        "get_recent_signals returned the seeded synthetic signal in real mode"
    )
    leaked = [s["signal_id"] for s in signals if s.get("is_synthetic")]
    assert leaked == [], f"real-mode recall leaked {len(leaked)} synthetic signals"

    opted = await get_recent_signals(
        limit=500, agent_name=seeded_signals["agent"], include_synthetic=True
    )
    opted_ids = {s["signal_id"] for s in opted}
    assert {seeded_signals["real_id"], seeded_signals["synthetic_id"]} <= opted_ids


async def test_training_examples_exclude_synthetic(seeded_signals):
    """Synthetic-signal recall must not feed DSPy self-learning by default."""
    from src.memory.procedural_memory import get_training_examples_for_agent

    examples = await get_training_examples_for_agent(seeded_signals["agent"], limit=500)
    ids = {e["signal_id"] for e in examples}
    assert seeded_signals["real_id"] in ids
    assert seeded_signals["synthetic_id"] not in ids
    assert all(not e.get("is_synthetic") for e in examples)

    opted = await get_training_examples_for_agent(
        seeded_signals["agent"], limit=500, include_synthetic=True
    )
    opted_ids = {e["signal_id"] for e in opted}
    assert {seeded_signals["real_id"], seeded_signals["synthetic_id"]} <= opted_ids


# =============================================================================
# Family: user_sessions
# =============================================================================


@pytest.fixture
def seeded_sessions(sync_client):
    marker = f"e2i894-user-{uuid.uuid4().hex[:8]}"
    pre = _scount(sync_client, "user_sessions", pk="session_id")
    now = datetime.now(timezone.utc)
    rows = [
        {"user_id": marker, "session_start": now.isoformat(), "is_synthetic": False},
        {"user_id": marker, "session_start": now.isoformat(), "is_synthetic": True},
    ]
    res = sync_client.table("user_sessions").insert(rows).execute()
    ids = [r["session_id"] for r in res.data]
    try:
        yield {"marker": marker, "now": now}
    finally:
        sync_client.table("user_sessions").delete().in_("session_id", ids).execute()
        assert _scount(sync_client, "user_sessions", pk="session_id") == pre


async def test_user_sessions_default_exclude_synthetic(async_client, seeded_sessions):
    from src.repositories.user_session import UserSessionRepository

    repo = UserSessionRepository(supabase_client=async_client)
    rows = await repo.get_by_user(seeded_sessions["marker"])
    assert len(rows) == 1 and not rows[0].get("is_synthetic")

    window = await repo.get_sessions_in_range(
        seeded_sessions["now"] - timedelta(minutes=5),
        seeded_sessions["now"] + timedelta(minutes=5),
    )
    leaked = [r for r in window if r.get("is_synthetic")]
    assert leaked == [], f"get_sessions_in_range leaked {len(leaked)} synthetic sessions"
