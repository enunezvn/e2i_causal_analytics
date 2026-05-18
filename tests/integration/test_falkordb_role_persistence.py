"""Phase 6 integration tests for FalkorDB causal-role persistence + KG enrichment.

Plan: ``.claude/plans/causal_role_propagation_FINAL.md`` §6.5.

Five cases enforce the Phase 6 contract:

  1. Write a role to FalkorDB; read back via
     ``ensemble_voter.layer_2_kg_signal(feature)``.
  2. KG corroborates LLM (same role) → ``kg_role_enrichment`` node sets
     ``source="kg"`` (KG corroboration promotes trust).
  3. KG contradicts LLM (different role) → ``kg_role_enrichment`` keeps
     ``source="llm"`` AND sets ``evaluator_satisfied=False``
     (Phase 2 C1 then gates the attribution out).
  4. KG silent (no Feature node) → ``kg_role_enrichment`` leaves
     attribution unchanged.
  5. Manifest-source attribution → ``kg_role_enrichment`` never queries
     the KG (manifest is already verification-grade per C1).

Falsifiability anchors:
  - Revert the corroborate branch in ``kg_role_enrichment`` → case 2
    trips (``source`` stays ``"llm"``).
  - Revert the contradict branch → case 3 trips
    (``evaluator_satisfied`` stays ``True``).
  - Skip the manifest short-circuit → case 5 trips (the spy detects a
    KG read for a manifest-source feature).

All tests patch ``FalkorDBClient`` at the module-level seam so no live
KG is required.
"""

from __future__ import annotations

from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeGraphResult:
    """Mimics the falkordb result-set shape: ``.result_set`` is a list of rows."""

    def __init__(self, result_set: list[list[Any]]) -> None:
        self.result_set = result_set


class _FakeGraph:
    """In-memory fake of ``FalkorDB.select_graph()`` — answers Cypher MATCHes from a dict.

    Stores nodes keyed by ``(feature, experiment_id)`` and serves
    ``layer_2_kg_signal`` MATCH queries against that store. Also records
    every ``query()`` call on ``calls`` so falsifiability tests can
    detect unexpected reads.
    """

    def __init__(self, store: dict[tuple[str, str], dict[str, Any]] | None = None) -> None:
        self.store: dict[tuple[str, str], dict[str, Any]] = store or {}
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> _FakeGraphResult:
        self.calls.append((cypher, params))
        # CREATE/MERGE writes — record into store.
        if "MERGE" in cypher or "CREATE" in cypher:
            if params and "feature" in params and "experiment_id" in params:
                key = (params["feature"], params["experiment_id"])
                self.store[key] = dict(params)
            return _FakeGraphResult([])
        # MATCH reads — look up by (feature, experiment_id).
        if "MATCH" in cypher and params:
            feature = params.get("feature")
            experiment_id = params.get("experiment_id")
            if feature is not None and experiment_id is not None:
                hit = self.store.get((feature, experiment_id))
                if hit is None:
                    return _FakeGraphResult([])
                # Return shape mimicking the layer_2_kg_signal query:
                #   [causal_role, causal_role_source, evaluator_model]
                return _FakeGraphResult(
                    [
                        [
                            hit.get("causal_role"),
                            hit.get("causal_role_source"),
                            hit.get("evaluator_model"),
                        ]
                    ]
                )
        return _FakeGraphResult([])


# ---------------------------------------------------------------------------
# Case 1 — Write then read back via layer_2_kg_signal
# ---------------------------------------------------------------------------


def test_case_1_write_then_read_via_layer_2_kg_signal() -> None:
    """Upsert a Feature node; ``layer_2_kg_signal(feature, experiment_id)``
    returns the persisted role + source + evaluator_model."""
    from src.data.kg.ensemble_voter import (
        layer_2_kg_signal,
        upsert_feature_role_node,
    )

    graph = _FakeGraph()
    upsert_feature_role_node(
        graph,
        feature="age",
        experiment_id="exp-1",
        causal_role="confounder",
        causal_role_source="llm",
        evaluator_model="anthropic/claude-haiku-4-5-20251001",
        brand="dupixent",
    )

    result = layer_2_kg_signal(graph, feature="age", experiment_id="exp-1")
    assert result is not None
    assert result["causal_role"] == "confounder"
    assert result["causal_role_source"] == "llm"
    assert result["evaluator_model"] == "anthropic/claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Case 2 — KG corroborates LLM → enrichment node promotes source="kg"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_case_2_kg_corroborates_llm_promotes_to_kg() -> None:
    """LLM attribution says ``age=confounder``; KG also says ``confounder``
    → ``kg_role_enrichment`` mutates ``source`` from ``"llm"`` to ``"kg"``."""
    from src.agents.ml_foundation.data_preparer.nodes.kg_role_enrichment import (
        kg_role_enrichment,
    )

    graph = _FakeGraph(
        store={
            ("age", "exp-2"): {
                "feature": "age",
                "experiment_id": "exp-2",
                "causal_role": "confounder",
                "causal_role_source": "llm",
                "evaluator_model": "kg:falkordb",
            }
        }
    )
    state = {
        "experiment_id": "exp-2",
        "role_attributions": [
            {
                "feature": "age",
                "causal_role": "confounder",
                "source": "llm",
                "evaluator_satisfied": True,
                "evaluator_model": "anthropic/claude-haiku-4-5-20251001",
            }
        ],
    }

    updated = await kg_role_enrichment(state, _graph_override=graph)  # type: ignore[arg-type]
    out = updated["role_attributions"]
    assert len(out) == 1
    assert out[0]["source"] == "kg"
    assert out[0]["causal_role"] == "confounder"
    assert out[0]["evaluator_satisfied"] is True
    # KG corroborates → flip source to "kg" but PRESERVE the upstream LLM's
    # evaluator_model. KG is a corroborating store, not an evaluator; clobbering
    # evaluator_model would corrupt audit provenance ("which model produced
    # this verdict?").
    assert out[0]["evaluator_model"] == "anthropic/claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Case 3 — KG contradicts LLM → keep llm but evaluator_satisfied=False
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_case_3_kg_contradicts_llm_downgrades_satisfied() -> None:
    """LLM says ``age=confounder``; KG says ``age=collider``
    → ``kg_role_enrichment`` keeps ``source="llm"`` but flips
    ``evaluator_satisfied`` to ``False``. Phase 2 C1 then gates it out."""
    from src.agents.ml_foundation.data_preparer.nodes.kg_role_enrichment import (
        kg_role_enrichment,
    )

    graph = _FakeGraph(
        store={
            ("age", "exp-3"): {
                "feature": "age",
                "experiment_id": "exp-3",
                "causal_role": "collider",  # disagrees with LLM's confounder
                "causal_role_source": "llm",
                "evaluator_model": "kg:falkordb",
            }
        }
    )
    state = {
        "experiment_id": "exp-3",
        "role_attributions": [
            {
                "feature": "age",
                "causal_role": "confounder",
                "source": "llm",
                "evaluator_satisfied": True,
                "evaluator_model": "anthropic/claude-haiku-4-5-20251001",
            }
        ],
    }

    updated = await kg_role_enrichment(state, _graph_override=graph)  # type: ignore[arg-type]
    out = updated["role_attributions"]
    assert len(out) == 1
    assert out[0]["source"] == "llm"  # NOT promoted to kg
    assert out[0]["evaluator_satisfied"] is False  # downgraded
    assert out[0]["causal_role"] == "confounder"  # LLM's claim preserved


# ---------------------------------------------------------------------------
# Case 4 — KG silent (no Feature node) → attribution unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_case_4_kg_silent_leaves_attribution_unchanged() -> None:
    """No Feature node in KG for ``age``/``exp-4`` → enrichment node
    leaves the LLM attribution byte-identical."""
    from src.agents.ml_foundation.data_preparer.nodes.kg_role_enrichment import (
        kg_role_enrichment,
    )

    graph = _FakeGraph(store={})  # empty
    original = {
        "feature": "age",
        "causal_role": "confounder",
        "source": "llm",
        "evaluator_satisfied": True,
        "evaluator_model": "anthropic/claude-haiku-4-5-20251001",
    }
    state = {
        "experiment_id": "exp-4",
        "role_attributions": [dict(original)],
    }

    updated = await kg_role_enrichment(state, _graph_override=graph)  # type: ignore[arg-type]
    out = updated["role_attributions"]
    assert len(out) == 1
    assert out[0] == original  # byte-identical


# ---------------------------------------------------------------------------
# Case 5 — Manifest source short-circuits; KG never queried
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_case_5_manifest_source_never_queries_kg() -> None:
    """Manifest attributions bypass the KG entirely — codex-2 fix per §6.3.

    Falsifiability anchor: a spy on ``graph.query`` records zero MATCH
    calls for ``age`` since manifest is already verification-grade. If
    the manifest short-circuit is reverted, ``graph.calls`` will contain
    at least one read for ``age``.
    """
    from src.agents.ml_foundation.data_preparer.nodes.kg_role_enrichment import (
        kg_role_enrichment,
    )

    graph = _FakeGraph(
        store={
            # KG happens to disagree — but we should NEVER consult it
            # for a manifest source.
            ("age", "exp-5"): {
                "feature": "age",
                "experiment_id": "exp-5",
                "causal_role": "collider",
                "causal_role_source": "llm",
                "evaluator_model": "kg:falkordb",
            }
        }
    )
    original = {
        "feature": "age",
        "causal_role": "confounder",
        "source": "manifest",
        "evaluator_satisfied": True,
        "evaluator_model": "n/a",
    }
    state = {
        "experiment_id": "exp-5",
        "role_attributions": [dict(original)],
    }

    updated = await kg_role_enrichment(state, _graph_override=graph)  # type: ignore[arg-type]
    out = updated["role_attributions"]
    # Manifest attribution preserved byte-for-byte.
    assert out[0] == original
    # Spy: no MATCH calls for the manifest feature.
    feature_match_calls = [
        c for c in graph.calls if "MATCH" in c[0] and c[1] and c[1].get("feature") == "age"
    ]
    assert feature_match_calls == [], (
        f"manifest short-circuit reverted: KG was queried for manifest feature: "
        f"{feature_match_calls!r}"
    )


# ---------------------------------------------------------------------------
# Schema pin: kg_role_enrichment node sits between baseline_computer
# and finalize_output (codex-2 placement guard).
# ---------------------------------------------------------------------------


def test_graph_inserts_kg_role_enrichment_between_baseline_and_finalize() -> None:
    """The graph edge from ``compute_baseline_metrics`` now routes to
    ``kg_role_enrichment`` (not directly to ``finalize_output``);
    ``kg_role_enrichment`` then routes to ``finalize_output``.

    Falsifiability anchor: revert the §6.3 graph rewiring to the prior
    ``compute_baseline_metrics → finalize_output`` edge → this assertion
    trips.
    """
    from src.agents.ml_foundation.data_preparer.graph import create_data_preparer_graph

    graph = create_data_preparer_graph()
    # LangGraph stores edges on ``.edges`` as a set of (source, target) tuples.
    edges = {(src, tgt) for src, tgt in graph.edges}
    assert ("compute_baseline_metrics", "kg_role_enrichment") in edges, (
        f"missing edge compute_baseline_metrics→kg_role_enrichment; edges={sorted(edges)}"
    )
    assert ("kg_role_enrichment", "finalize_output") in edges, (
        f"missing edge kg_role_enrichment→finalize_output; edges={sorted(edges)}"
    )
    # Old direct edge must be gone.
    assert ("compute_baseline_metrics", "finalize_output") not in edges, (
        "old direct edge compute_baseline_metrics→finalize_output still present; "
        "kg_role_enrichment was not inserted in-between"
    )


# ---------------------------------------------------------------------------
# Mirror-script smoke: dry-run on a mock cursor returns rows-read count
# without writing to FalkorDB.
# ---------------------------------------------------------------------------


def test_mirror_script_dry_run_collects_feature_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """``mirror_role_attributions_to_falkordb`` in --dry-run mode reads
    rows from the cursor and reports the upsert count without
    touching the graph. Verifies the read path + payload shape."""
    from scripts import mirror_role_attributions_to_falkordb as mod

    # Fake psycopg cursor: yields two rows, one valid + one with NULL role.
    rows = [
        ("exp-A", "age", "confounder", "llm", "anthropic/claude-haiku-4-5-20251001"),
        ("exp-A", "bmi", None, None, None),  # skipped — no attribution
    ]

    class _FakeCur:
        def __init__(self) -> None:
            self._rows: list[tuple] = list(rows)

        def execute(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def fetchall(self) -> list[tuple]:
            return list(self._rows)

        def __enter__(self) -> "_FakeCur":
            return self

        def __exit__(self, *_exc: Any) -> None:
            pass

    class _FakeConn:
        def cursor(self) -> _FakeCur:
            return _FakeCur()

        def __enter__(self) -> "_FakeConn":
            return self

        def __exit__(self, *_exc: Any) -> None:
            pass

    fake_conn = _FakeConn()
    fake_graph = _FakeGraph()

    written = mod.mirror_role_attributions(
        conn=fake_conn,  # type: ignore[arg-type]
        graph=fake_graph,
        brand="dupixent",
        dry_run=True,
    )
    # 1 row eligible (the bmi row is skipped due to NULL columns).
    assert written == 1
    # Dry-run must not have called CREATE/MERGE.
    write_calls = [c for c in fake_graph.calls if "MERGE" in c[0] or "CREATE" in c[0]]
    assert write_calls == []


def test_mirror_script_live_upserts_to_graph() -> None:
    """``mirror_role_attributions_to_falkordb`` in NON-dry-run mode emits
    one MERGE per eligible row. Pin the Cypher to use ``Feature`` node
    + ``FOR_BRAND`` edge (codex-2 §6.1 schema decision)."""
    from scripts import mirror_role_attributions_to_falkordb as mod

    rows = [
        ("exp-A", "age", "confounder", "llm", "anthropic/claude-haiku-4-5-20251001"),
    ]

    class _FakeCur:
        def execute(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def fetchall(self) -> list[tuple]:
            return list(rows)

        def __enter__(self) -> "_FakeCur":
            return self

        def __exit__(self, *_exc: Any) -> None:
            pass

    class _FakeConn:
        def cursor(self) -> _FakeCur:
            return _FakeCur()

    fake_conn = _FakeConn()
    fake_graph = _FakeGraph()

    written = mod.mirror_role_attributions(
        conn=fake_conn,  # type: ignore[arg-type]
        graph=fake_graph,
        brand="dupixent",
        dry_run=False,
    )
    assert written == 1
    write_calls = [c for c in fake_graph.calls if "MERGE" in c[0] or "CREATE" in c[0]]
    assert len(write_calls) >= 1, f"expected at least one MERGE call; got {fake_graph.calls!r}"
    # Codex-2 §6.1: schema MUST use FOR_BRAND (not BELONGS_TO, which model_trainer
    # already uses for (:Model)-[:BELONGS_TO]->(:Experiment)).
    cypher_blob = " ".join(c[0] for c in write_calls)
    assert "FOR_BRAND" in cypher_blob, (
        f"mirror script must use FOR_BRAND edge per §6.1 schema decision; cypher={cypher_blob!r}"
    )
    assert "BELONGS_TO" not in cypher_blob, (
        f"mirror script must NOT use BELONGS_TO (overloads model_trainer:367); "
        f"cypher={cypher_blob!r}"
    )
    # Feature node label pinned.
    assert ":Feature" in cypher_blob, f"missing :Feature label; cypher={cypher_blob!r}"
