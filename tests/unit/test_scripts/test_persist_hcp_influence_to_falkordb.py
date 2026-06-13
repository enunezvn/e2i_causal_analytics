"""Issue #169: unit tests for the FalkorDB HCP-influence persistence script.

Covers the acceptance criteria from the issue body:

1. **Round-trip parity**: build a converter-style graph for a synthetic
   50-HCP fixture, persist it through a mocked FalkorDB graph, re-derive
   ``influence_network_size`` + ``peer_influence_score`` from the
   readback, and assert they match the values the converter would
   compute on the same input (modulo float rounding).
2. **Cohort tagging**: ingest two distinct synthetic cohorts back-to-back
   and assert each is queryable independently (no cross-cohort
   contamination on the property predicate).
3. **Idempotency**: ingest the same fixture twice via the real ``MERGE``
   Cypher; assert node + edge counts in the mock graph don't double.
4. **--replace flag**: assert it deletes only the prior ``cohort_id``
   rows before reloading.

Real-FalkorDB validation is deferred (no instance reachable in the
worktree); the in-process ``_FakeFalkorGraph`` below executes the
script's Cypher against a dict-backed store with sufficient semantics to
cover MERGE / MATCH / DETACH DELETE on the schema this script emits.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd
import pytest

from scripts.convert_optum_rwd import (
    PEER_INFLUENCE_SCALE,
    build_hcp_influence_graph,
    score_hcp_influence_graph,
)
from scripts.persist_hcp_influence_to_falkordb import (
    delete_cohort_rows,
    persist_graph_to_falkordb,
    read_influence_from_falkordb,
)

# --------------------------------------------------------------------------- #
# In-process FalkorDB-compatible fake                                         #
# --------------------------------------------------------------------------- #


class _Result:
    def __init__(
        self,
        result_set: list[list[Any]] | None = None,
        nodes_deleted: int = 0,
        relationships_deleted: int = 0,
    ) -> None:
        self.result_set = result_set or []
        self.nodes_deleted = nodes_deleted
        self.relationships_deleted = relationships_deleted


class _FakeFalkorGraph:
    """Minimal in-process executor for the Cypher this script emits.

    Implements just enough behaviour to exercise round-trip parity +
    idempotency + cohort isolation:

    - HCP nodes keyed on ``(id, cohort_id)``; MERGE upserts.
    - SHARED_PATIENTS edges keyed on ``(src_key, dst_key, cohort_id)``;
      MERGE upserts.
    - DETACH DELETE on ``cohort_id``-matched HCPs drops both nodes and
      their incident edges.
    - Readback MATCH queries return list-of-list result rows shaped
      the way the production FalkorDB client does.

    NOTE: this is NOT a general-purpose Cypher engine. It dispatches on
    distinctive substrings of the Cypher this script emits. Updating the
    persistence script's Cypher requires updating the fake — by design,
    so the contract stays explicit.
    """

    def __init__(self) -> None:
        # Node key = (cohort_id, id) → node prop dict
        self.nodes: dict[tuple[str, str], dict[str, Any]] = {}
        # Edge key = (cohort_id, src, dst) → edge prop dict
        self.edges: dict[tuple[str, str, str], dict[str, Any]] = {}
        # Call log for assertions
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> _Result:
        params = params or {}
        self.calls.append((cypher, dict(params)))
        c = cypher.strip()

        # Node MERGE batch
        if "MERGE (h:HCP {id: row.id" in c:
            cohort = params["cohort_id"]
            for row in params["rows"]:
                key = (cohort, row["id"])
                self.nodes[key] = {
                    "id": row["id"],
                    "npi": row["id"],
                    "cohort_id": cohort,
                    "ingested_at": params["ingested_at"],
                    "e2i_entity_type": "hcp",
                }
            return _Result()

        # Edge MERGE batch
        if "MERGE (a)-[r:SHARED_PATIENTS" in c:
            cohort = params["cohort_id"]
            for row in params["rows"]:
                src_key = (cohort, row["src"])
                dst_key = (cohort, row["dst"])
                if src_key not in self.nodes or dst_key not in self.nodes:
                    # MATCH would fail — skip (cohort isolation guard).
                    continue
                edge_key = (cohort, row["src"], row["dst"])
                self.edges[edge_key] = {
                    "weight": int(row["weight"]),
                    "cohort_id": cohort,
                    "ingested_at": params["ingested_at"],
                }
            return _Result()

        # DETACH DELETE cohort
        if "DETACH DELETE h" in c and "cohort_id" in c:
            cohort = params["cohort_id"]
            doomed = [k for k in self.nodes if k[0] == cohort]
            doomed_edges = [k for k in self.edges if k[0] == cohort]
            for k in doomed:
                del self.nodes[k]
            for k in doomed_edges:
                del self.edges[k]
            return _Result(
                nodes_deleted=len(doomed),
                relationships_deleted=len(doomed_edges),
            )

        # Read nodes
        if "RETURN h.id AS id" in c:
            cohort = params["cohort_id"]
            rows = [[k[1]] for k in self.nodes if k[0] == cohort]
            return _Result(result_set=rows)

        # Read edges. The production Cypher AFTER codex pass-2 MEDIUM
        # uses an UNDIRECTED match (`-` no arrow) with `WHERE a.id < b.id`
        # AND `WITH ... max(r.weight)` aggregation (pass-3 MEDIUM) so
        # duplicate legacy rows are deduplicated. The fake mirrors that
        # aggregation by collapsing on the canonical key and taking max
        # weight.
        if "max(r.weight) AS weight" in c or "r.weight AS weight" in c:
            cohort = params["cohort_id"]
            agg: dict[tuple[str, str], int] = {}
            for k, v in self.edges.items():
                if k[0] != cohort:
                    continue
                a_id, b_id = k[1], k[2]
                # Mirror the production `WHERE a.id < b.id` filter +
                # the canonical-key aggregation.
                if a_id < b_id:
                    key = (a_id, b_id)
                elif b_id < a_id:
                    key = (b_id, a_id)
                else:
                    continue  # self-loop with `a.id < b.id` predicate
                w = int(v["weight"])
                agg[key] = max(agg.get(key, 0), w)
            rows = [[src, dst, w] for (src, dst), w in agg.items()]
            return _Result(result_set=rows)

        # Cohort-scoped influence-network traversal (depth 1 only for
        # the cohort-isolation test). The production Cypher AFTER
        # codex pass-1 MEDIUM-1 is path-bound and SHARED_PATIENTS-typed:
        #
        #   MATCH path = (h:HCP {id, cohort_id})
        #     -[:SHARED_PATIENTS*1..N]-(connected)
        #   WHERE connected.cohort_id = $cohort_id
        #     AND all(r IN relationships(path) WHERE r.cohort_id = $cohort_id)
        #
        # The fake honours that contract by ONLY traversing edges whose
        # cohort_id matches AND only respecting SHARED_PATIENTS edges
        # (which is the only edge type the fake stores — extra edge
        # types would never enter via persist_graph_to_falkordb, but
        # the test_cross_relationship_leak test below seeds a fake
        # cross-cohort/relationship edge to verify the predicate.
        if (
            "WHERE connected.cohort_id" in c
            and ":SHARED_PATIENTS" in c
            and "relationships(path)" in c
        ):
            hcp_id = params["hcp_id"]
            cohort = params["cohort_id"]
            neighbours: set[str] = set()
            for ec, src, dst in self.edges:
                if ec != cohort:
                    continue
                if src == hcp_id:
                    neighbours.add(dst)
                elif dst == hcp_id:
                    neighbours.add(src)
            if "count(DISTINCT connected)" in c:
                return _Result(result_set=[[len(neighbours)]])
            return _Result(result_set=[[n] for n in sorted(neighbours)])

        # 1-hop SHARED_PATIENTS degree count (count_hcp_influence_degree).
        if "-[:SHARED_PATIENTS]-(neighbor:HCP" in c:
            hcp_id = params["hcp_id"]
            cohort = params["cohort_id"]
            neighbours = set()
            for ec, src, dst in self.edges:
                if ec != cohort:
                    continue
                if src == hcp_id:
                    neighbours.add(dst)
                elif dst == hcp_id:
                    neighbours.add(src)
            return _Result(result_set=[[len(neighbours)]])

        raise AssertionError(f"_FakeFalkorGraph saw unexpected Cypher:\n{c[:200]}")


# --------------------------------------------------------------------------- #
# Fixture builders                                                            #
# --------------------------------------------------------------------------- #


def _make_synthetic_cohort(
    n_hcps: int = 50,
    n_patients: int = 200,
    cluster_size: int = 5,
    seed: int = 0,
) -> tuple[set[int], dict[int, pd.Timestamp], pd.DataFrame, pd.DataFrame]:
    """Build a deterministic synthetic cohort for the parity tests.

    Each patient is assigned a contiguous cluster of ``cluster_size``
    HCPs forming a clique on the patient's HCP set. With 50 HCPs and
    cluster_size=5 we get 10 disjoint cliques; with overlapping patient
    assignments edges accumulate weight > 1, which lets the parity test
    exercise the int-weight path end-to-end.
    """
    import random

    rnd = random.Random(seed)
    base_date = pd.Timestamp("2024-01-15")

    npis = [f"NPI_{i:04d}" for i in range(n_hcps)]
    med_rows: list[dict[str, Any]] = []
    proc_rows: list[dict[str, Any]] = []
    idx_by_patid: dict[int, pd.Timestamp] = {}
    kept: set[int] = set()

    for pid in range(n_patients):
        # Each patient gets a contiguous window of HCPs offset by a
        # random starting position. With overlap, edge weights climb.
        start = rnd.randrange(0, n_hcps - cluster_size + 1)
        patient_hcps = npis[start : start + cluster_size]
        pat_idx = base_date + pd.Timedelta(days=pid)
        idx_by_patid[pid] = pat_idx
        kept.add(pid)
        # Half the contacts go to medication, half to procedure to
        # exercise both data sources in the helper.
        for k, npi in enumerate(patient_hcps):
            fill_date = pat_idx - pd.Timedelta(days=10 + k)
            if k % 2 == 0:
                med_rows.append(
                    {
                        "patid": pid,
                        "npi": npi,
                        "medication_date": fill_date,
                    }
                )
            else:
                proc_rows.append(
                    {
                        "patid": pid,
                        "npi": npi,
                        "proc_date": fill_date,
                    }
                )

    med = pd.DataFrame(med_rows)
    proc = pd.DataFrame(proc_rows)
    return kept, idx_by_patid, med, proc


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


class TestRoundTripParity:
    """Acceptance criterion 1: Parquet artifact == FalkorDB readback."""

    def test_50_hcp_fixture_roundtrip(self) -> None:
        kept, idx, med, proc = _make_synthetic_cohort(n_hcps=50, n_patients=200, seed=42)
        graph = build_hcp_influence_graph(kept_patids=kept, med=med, proc=proc, idx_by_patid=idx)
        assert graph is not None
        # Sanity: non-trivial graph
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0

        # Converter-path scoring (== Parquet artifact)
        parquet_deg, parquet_score = score_hcp_influence_graph(graph)

        # Persist + read back via the fake FalkorDB
        fake = _FakeFalkorGraph()
        persist_graph_to_falkordb(graph, fake, cohort_id="test_cohort")
        readback_deg, readback_score = read_influence_from_falkordb(fake, "test_cohort")

        assert readback_deg == parquet_deg
        # Score is post-rounding-to-2-places so equality is acceptable;
        # but pin a small tolerance for safety against future
        # round-mode drift.
        assert set(readback_score) == set(parquet_score)
        for npi, score in parquet_score.items():
            assert abs(readback_score[npi] - score) < 1e-9

    def test_empty_graph_roundtrip(self) -> None:
        """0 patients → 0 nodes → empty readback (no spurious writes)."""
        fake = _FakeFalkorGraph()
        graph = build_hcp_influence_graph(
            kept_patids=set(),
            med=pd.DataFrame(columns=["patid", "npi", "medication_date"]),
            proc=pd.DataFrame(columns=["patid", "npi", "proc_date"]),
        )
        assert graph is not None
        n, e = persist_graph_to_falkordb(graph, fake, cohort_id="empty")
        assert (n, e) == (0, 0)
        deg, score = read_influence_from_falkordb(fake, "empty")
        assert deg == {} and score == {}

    def test_edge_weight_preserved(self) -> None:
        """Patient cliques contribute integer weights that must survive the round trip."""
        # Two patients both share HCPs A+B → edge weight 2
        idx = {1: pd.Timestamp("2024-06-01"), 2: pd.Timestamp("2024-06-02")}
        med = pd.DataFrame(
            [
                {"patid": 1, "npi": "A", "medication_date": pd.Timestamp("2024-05-15")},
                {"patid": 1, "npi": "B", "medication_date": pd.Timestamp("2024-05-16")},
                {"patid": 2, "npi": "A", "medication_date": pd.Timestamp("2024-05-15")},
                {"patid": 2, "npi": "B", "medication_date": pd.Timestamp("2024-05-16")},
            ]
        )
        proc = pd.DataFrame(columns=["patid", "npi", "proc_date"])
        graph = build_hcp_influence_graph(kept_patids={1, 2}, med=med, proc=proc, idx_by_patid=idx)
        assert graph is not None
        assert graph["A"]["B"]["weight"] == 2
        fake = _FakeFalkorGraph()
        persist_graph_to_falkordb(graph, fake, cohort_id="wts")
        # Inspect fake's stored edge weight.
        edge_keys = list(fake.edges.keys())
        assert len(edge_keys) == 1
        assert fake.edges[edge_keys[0]]["weight"] == 2


class TestCohortTagging:
    """Acceptance criterion 2: two cohorts back-to-back stay independent."""

    def test_two_cohorts_independently_queryable(self) -> None:
        fake = _FakeFalkorGraph()
        # Two completely disjoint synthetic cohorts.
        kept_a, idx_a, med_a, proc_a = _make_synthetic_cohort(n_hcps=20, n_patients=30, seed=1)
        kept_b, idx_b, med_b, proc_b = _make_synthetic_cohort(n_hcps=20, n_patients=30, seed=2)
        # Disambiguate NPI namespaces so each cohort's HCPs are distinct.
        med_b["npi"] = med_b["npi"].apply(lambda x: f"COHORT_B_{x}")
        proc_b["npi"] = proc_b["npi"].apply(lambda x: f"COHORT_B_{x}")
        graph_a = build_hcp_influence_graph(
            kept_patids=kept_a, med=med_a, proc=proc_a, idx_by_patid=idx_a
        )
        graph_b = build_hcp_influence_graph(
            kept_patids=kept_b, med=med_b, proc=proc_b, idx_by_patid=idx_b
        )
        assert graph_a is not None and graph_b is not None

        persist_graph_to_falkordb(graph_a, fake, cohort_id="cohort_a")
        persist_graph_to_falkordb(graph_b, fake, cohort_id="cohort_b")

        # Each cohort's readback recovers its own graph, nothing else.
        deg_a, _ = read_influence_from_falkordb(fake, "cohort_a")
        deg_b, _ = read_influence_from_falkordb(fake, "cohort_b")

        assert set(deg_a.keys()).isdisjoint(set(deg_b.keys()))
        assert set(deg_a.keys()) == set(graph_a.nodes())
        assert set(deg_b.keys()) == set(graph_b.nodes())

    def test_same_npi_in_both_cohorts_stays_separate(self) -> None:
        """An NPI appearing in BOTH cohorts must be addressable independently."""
        fake = _FakeFalkorGraph()
        # Both cohorts share NPI "X" with totally different neighbours.
        med_a = pd.DataFrame(
            [
                {"patid": 1, "npi": "X", "medication_date": pd.Timestamp("2024-05-15")},
                {"patid": 1, "npi": "A", "medication_date": pd.Timestamp("2024-05-16")},
            ]
        )
        med_b = pd.DataFrame(
            [
                {"patid": 2, "npi": "X", "medication_date": pd.Timestamp("2024-05-15")},
                {"patid": 2, "npi": "B", "medication_date": pd.Timestamp("2024-05-16")},
            ]
        )
        proc_empty = pd.DataFrame(columns=["patid", "npi", "proc_date"])
        g_a = build_hcp_influence_graph(
            kept_patids={1},
            med=med_a,
            proc=proc_empty,
            idx_by_patid={1: pd.Timestamp("2024-06-01")},
        )
        g_b = build_hcp_influence_graph(
            kept_patids={2},
            med=med_b,
            proc=proc_empty,
            idx_by_patid={2: pd.Timestamp("2024-06-01")},
        )
        persist_graph_to_falkordb(g_a, fake, cohort_id="A")
        persist_graph_to_falkordb(g_b, fake, cohort_id="B")
        # Each cohort sees only its own X-neighbour.
        deg_a, _ = read_influence_from_falkordb(fake, "A")
        deg_b, _ = read_influence_from_falkordb(fake, "B")
        assert "A" in deg_a and "B" not in deg_a
        assert "B" in deg_b and "A" not in deg_b


class TestIdempotency:
    """Acceptance criterion 3: rerun is a no-op via MERGE."""

    def test_canonical_edge_direction_survives_swapped_iteration(self) -> None:
        """Codex pass-2 MEDIUM: persistence canonicalises (a, b) by sorted order.

        networkx.Graph.edges() iteration is implementation-defined.
        On a rerun (different process / different hash seed) the same
        logical undirected edge can come out as (B, A) instead of
        (A, B). The persistence step must canonicalise so the FalkorDB
        MERGE doesn't create a second directed row.
        """
        import networkx as nx

        # Hand-build a graph that exercises BOTH endpoint orders for the
        # same logical undirected edge.
        g1: Any = nx.Graph()
        g1.add_edge("A", "B", weight=3)
        # Second graph swaps the addition order — networkx still treats
        # the edge as undirected, but `.edges()` iteration order may
        # differ across hash seeds.
        g2: Any = nx.Graph()
        g2.add_edge("B", "A", weight=5)  # weight changed too

        fake = _FakeFalkorGraph()
        persist_graph_to_falkordb(g1, fake, cohort_id="dir")
        assert len(fake.edges) == 1
        # Persisted endpoints are sorted: ('A', 'B').
        key = next(iter(fake.edges))
        assert key[1:] == ("A", "B")

        # Second ingest with reversed order + different weight must
        # MERGE the same edge (canonical key), not create a sibling.
        persist_graph_to_falkordb(g2, fake, cohort_id="dir")
        assert len(fake.edges) == 1
        # ON MATCH SET r.weight = row.weight → the new weight wins.
        assert fake.edges[key]["weight"] == 5

    def test_readback_dedupes_legacy_duplicate_directed_edges(self) -> None:
        """Codex pass-3 MEDIUM: legacy duplicate edges aggregate to one row.

        If a pre-canonicalisation run left both ``(A)->(B)`` and
        ``(B)->(A)`` for the same logical pair, the undirected readback
        would surface two rows with src=A, dst=B. The Cypher uses
        ``WITH ... max(r.weight) AS weight`` to collapse them. The
        fake mirrors that aggregation; this test seeds the legacy
        artifact and asserts the readback returns ONE edge with the
        max weight.
        """
        fake = _FakeFalkorGraph()
        # Pretend pre-canonicalisation persisted both directions with
        # divergent weights — a legacy artifact.
        fake.nodes[("legacy", "A")] = {"id": "A", "cohort_id": "legacy"}
        fake.nodes[("legacy", "B")] = {"id": "B", "cohort_id": "legacy"}
        fake.edges[("legacy", "A", "B")] = {"weight": 3, "cohort_id": "legacy"}
        fake.edges[("legacy", "B", "A")] = {"weight": 7, "cohort_id": "legacy"}

        deg, _ = read_influence_from_falkordb(fake, "legacy")
        # Both A and B see exactly one neighbour (each other).
        assert deg == {"A": 1, "B": 1}

        # Codex pass-4 LOW: also assert max(weight) semantics survive
        # the aggregation. Read the underlying readback's edge result
        # directly: one row, with the LARGER of the two legacy weights.
        edges_result = fake.query(
            "MATCH (a:HCP {cohort_id: $cohort_id})-[r:SHARED_PATIENTS {cohort_id: $cohort_id}]-"
            "(b:HCP {cohort_id: $cohort_id}) "
            "WHERE a.id < b.id "
            "WITH a.id AS src, b.id AS dst, max(r.weight) AS weight "
            "RETURN src, dst, weight",
            {"cohort_id": "legacy"},
        )
        assert len(edges_result.result_set) == 1
        assert edges_result.result_set[0] == ["A", "B", 7]

    def test_double_ingest_no_doubling(self) -> None:
        fake = _FakeFalkorGraph()
        kept, idx, med, proc = _make_synthetic_cohort(n_hcps=20, n_patients=40, seed=7)
        graph = build_hcp_influence_graph(kept_patids=kept, med=med, proc=proc, idx_by_patid=idx)
        persist_graph_to_falkordb(graph, fake, cohort_id="idem")
        node_count_first = len(fake.nodes)
        edge_count_first = len(fake.edges)
        # Second ingest — MERGE must upsert, not duplicate.
        persist_graph_to_falkordb(graph, fake, cohort_id="idem")
        assert len(fake.nodes) == node_count_first
        assert len(fake.edges) == edge_count_first


class TestReplaceFlag:
    """Acceptance criterion 4: --replace deletes prior cohort_id rows only."""

    def test_delete_only_target_cohort(self) -> None:
        fake = _FakeFalkorGraph()
        kept_a, idx_a, med_a, proc_a = _make_synthetic_cohort(seed=1)
        kept_b, idx_b, med_b, proc_b = _make_synthetic_cohort(seed=2)
        med_b["npi"] = med_b["npi"].apply(lambda x: f"B_{x}")
        proc_b["npi"] = proc_b["npi"].apply(lambda x: f"B_{x}")
        g_a = build_hcp_influence_graph(kept_a, med_a, proc_a, idx_a)
        g_b = build_hcp_influence_graph(kept_b, med_b, proc_b, idx_b)
        persist_graph_to_falkordb(g_a, fake, cohort_id="A")
        persist_graph_to_falkordb(g_b, fake, cohort_id="B")
        nodes_b_before = sum(1 for k in fake.nodes if k[0] == "B")
        edges_b_before = sum(1 for k in fake.edges if k[0] == "B")

        nodes_deleted, edges_deleted = delete_cohort_rows(fake, "A")
        assert nodes_deleted == sum(1 for k in fake.nodes if k[0] == "A") + nodes_deleted
        # B untouched
        assert sum(1 for k in fake.nodes if k[0] == "B") == nodes_b_before
        assert sum(1 for k in fake.edges if k[0] == "B") == edges_b_before
        # A wiped
        assert sum(1 for k in fake.nodes if k[0] == "A") == 0
        assert sum(1 for k in fake.edges if k[0] == "A") == 0
        # Edge counter is best-effort
        assert edges_deleted >= 0

    def test_replace_then_reload_lands_clean(self) -> None:
        fake = _FakeFalkorGraph()
        kept, idx, med, proc = _make_synthetic_cohort(n_hcps=10, n_patients=20, seed=9)
        g1 = build_hcp_influence_graph(kept, med, proc, idx)
        persist_graph_to_falkordb(g1, fake, cohort_id="X")
        # Sanity check: pre-replace the bigger graph is loaded.
        assert sum(1 for k in fake.nodes if k[0] == "X") == g1.number_of_nodes()

        # Build a DIFFERENT graph (fewer HCPs) and replace
        kept2, idx2, med2, proc2 = _make_synthetic_cohort(n_hcps=5, n_patients=10, seed=10)
        g2 = build_hcp_influence_graph(kept2, med2, proc2, idx2)
        delete_cohort_rows(fake, "X")
        persist_graph_to_falkordb(g2, fake, cohort_id="X")

        second_nodes = sum(1 for k in fake.nodes if k[0] == "X")
        # If replace had a bug (e.g. left stale rows), second_nodes would
        # include the larger first-graph's extra HCPs.
        assert second_nodes == g2.number_of_nodes()
        # We DON'T require strict inequality — small fixtures might
        # coincidentally yield the same count — but we DO require that
        # the readback IDs match g2 (not g1).
        deg, _ = read_influence_from_falkordb(fake, "X")
        assert set(deg.keys()) == set(g2.nodes())


class TestSemanticMemoryCohortFilter:
    """Verify the semantic_memory.py extension accepts ``cohort_id``."""

    @staticmethod
    def _wired_semantic_memory(fake: "_FakeFalkorGraph") -> Any:
        from unittest.mock import MagicMock, patch

        from src.memory.semantic_memory import FalkorDBSemanticMemory

        client = MagicMock()
        client.select_graph.return_value = fake
        cfg = MagicMock()
        cfg.semantic.graph_name = "e2i_semantic"
        with (
            patch("src.memory.semantic_memory.get_config", return_value=cfg),
            patch(
                "src.memory.semantic_memory.get_falkordb_client",
                return_value=client,
            ),
        ):
            sm = FalkorDBSemanticMemory()
            _ = sm.client
            _ = sm.graph
            return sm

    def test_cohort_filter_param_threaded(self) -> None:
        """get_hcp_influence_network + count_hcp_influence_network accept cohort_id."""
        fake = _FakeFalkorGraph()
        med_a = pd.DataFrame(
            [
                {"patid": 1, "npi": "X", "medication_date": pd.Timestamp("2024-05-15")},
                {"patid": 1, "npi": "A1", "medication_date": pd.Timestamp("2024-05-16")},
            ]
        )
        med_b = pd.DataFrame(
            [
                {"patid": 2, "npi": "X", "medication_date": pd.Timestamp("2024-05-15")},
                {"patid": 2, "npi": "B1", "medication_date": pd.Timestamp("2024-05-16")},
            ]
        )
        empty = pd.DataFrame(columns=["patid", "npi", "proc_date"])
        idx_a = {1: pd.Timestamp("2024-06-01")}
        idx_b = {2: pd.Timestamp("2024-06-01")}
        ga = build_hcp_influence_graph({1}, med_a, empty, idx_a)
        gb = build_hcp_influence_graph({2}, med_b, empty, idx_b)
        persist_graph_to_falkordb(ga, fake, cohort_id="cohort_a")
        persist_graph_to_falkordb(gb, fake, cohort_id="cohort_b")

        sm = self._wired_semantic_memory(fake)
        count_a = sm.count_hcp_influence_network("X", max_depth=1, cohort_id="cohort_a")
        count_b = sm.count_hcp_influence_network("X", max_depth=1, cohort_id="cohort_b")
        assert count_a == 1  # X is connected to A1 only in cohort_a
        assert count_b == 1  # X is connected to B1 only in cohort_b

    def test_count_hcp_influence_degree_matches_parquet(self) -> None:
        """Codex pass-1 MEDIUM-2: count_hcp_influence_degree == Parquet degree."""
        fake = _FakeFalkorGraph()
        kept, idx, med, proc = _make_synthetic_cohort(n_hcps=15, n_patients=30, seed=11)
        graph = build_hcp_influence_graph(kept_patids=kept, med=med, proc=proc, idx_by_patid=idx)
        persist_graph_to_falkordb(graph, fake, cohort_id="C")
        parquet_deg, _ = score_hcp_influence_graph(graph)
        sm = self._wired_semantic_memory(fake)
        for npi, expected in parquet_deg.items():
            got = sm.count_hcp_influence_degree(npi, cohort_id="C")
            assert got == expected, f"degree mismatch for {npi}: got={got} expected={expected}"

    def test_cross_cohort_edge_traversal_blocked(self) -> None:
        """Codex pass-1 LOW-1 + MEDIUM-1: terminal-only predicate would leak.

        A node tagged with the requested ``cohort_id`` that is ONLY
        reachable via an edge tagged with a DIFFERENT cohort must NOT
        appear in the traversal. The post-fix Cypher constrains every
        relationship in the path via
        ``all(r IN relationships(path) WHERE r.cohort_id = $cohort_id)``.
        """
        fake = _FakeFalkorGraph()
        med_a = pd.DataFrame(
            [
                {"patid": 1, "npi": "X", "medication_date": pd.Timestamp("2024-05-15")},
                {"patid": 1, "npi": "A", "medication_date": pd.Timestamp("2024-05-16")},
            ]
        )
        empty = pd.DataFrame(columns=["patid", "npi", "proc_date"])
        ga = build_hcp_influence_graph({1}, med_a, empty, {1: pd.Timestamp("2024-06-01")})
        persist_graph_to_falkordb(ga, fake, cohort_id="A")

        # Manually inject a cohort-mismatched rogue edge: node Z is
        # tagged cohort_id="A" but the X-Z edge is tagged cohort_id="B".
        # The path-bound Cypher must reject the traversal.
        fake.nodes[("A", "Z")] = {
            "id": "Z",
            "npi": "Z",
            "cohort_id": "A",
            "ingested_at": "test",
            "e2i_entity_type": "hcp",
        }
        fake.edges[("B", "X", "Z")] = {
            "weight": 1,
            "cohort_id": "B",
            "ingested_at": "test",
        }

        sm = self._wired_semantic_memory(fake)
        count_a = sm.count_hcp_influence_network("X", max_depth=2, cohort_id="A")
        # Only the legitimate cohort-A neighbour (A) should be reachable.
        assert count_a == 1


class TestLoadCohortInputs:
    """Parquet loader contract: schema flexibility + missing-file errors."""

    def _write_journeys(self, tmp_path: Any, pid_col: str = "patient_id") -> None:
        cohort_dir = tmp_path / "cohort"
        cohort_dir.mkdir()
        journeys = pd.DataFrame(
            [
                {pid_col: 1, "index_date": pd.Timestamp("2024-06-01")},
                {pid_col: 2, "index_date": pd.Timestamp("2024-06-15")},
            ]
        )
        journeys.to_parquet(cohort_dir / "e2i_ml_v3_patient_journeys.parquet")
        return cohort_dir

    def _write_raw(self, tmp_path: Any) -> Any:
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        pd.DataFrame(
            [{"patid": 1, "npi": "A", "medication_date": pd.Timestamp("2024-05-15")}]
        ).to_parquet(raw_dir / "medication.parquet")
        pd.DataFrame(
            [{"patid": 1, "npi": "B", "proc_date": pd.Timestamp("2024-05-16")}]
        ).to_parquet(raw_dir / "procedure.parquet")
        return raw_dir

    def test_accepts_patient_id_column(self, tmp_path: Any) -> None:
        from scripts.persist_hcp_influence_to_falkordb import load_cohort_inputs

        cohort_dir = self._write_journeys(tmp_path, pid_col="patient_id")
        raw_dir = self._write_raw(tmp_path)
        kept, idx, med, proc = load_cohort_inputs(raw_dir, cohort_dir)
        assert kept == {1, 2}
        assert idx[1] == pd.Timestamp("2024-06-01")
        assert not med.empty and not proc.empty

    def test_accepts_patid_column(self, tmp_path: Any) -> None:
        from scripts.persist_hcp_influence_to_falkordb import load_cohort_inputs

        cohort_dir = self._write_journeys(tmp_path, pid_col="patid")
        raw_dir = self._write_raw(tmp_path)
        kept, idx, _, _ = load_cohort_inputs(raw_dir, cohort_dir)
        assert kept == {1, 2}

    def test_missing_raw_parquet_raises(self, tmp_path: Any) -> None:
        from scripts.persist_hcp_influence_to_falkordb import load_cohort_inputs

        cohort_dir = self._write_journeys(tmp_path)
        raw_dir = tmp_path / "missing"
        raw_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            load_cohort_inputs(raw_dir, cohort_dir)

    def test_missing_index_date_raises(self, tmp_path: Any) -> None:
        from scripts.persist_hcp_influence_to_falkordb import load_cohort_inputs

        cohort_dir = tmp_path / "cohort"
        cohort_dir.mkdir()
        pd.DataFrame([{"patient_id": 1}]).to_parquet(
            cohort_dir / "e2i_ml_v3_patient_journeys.parquet"
        )
        raw_dir = self._write_raw(tmp_path)
        with pytest.raises(ValueError, match="index_date"):
            load_cohort_inputs(raw_dir, cohort_dir)


class TestCLI:
    """End-to-end CLI smoke: dry-run path exercises arg parsing + graph build."""

    def test_dry_run_exits_0(self, tmp_path: Any, capsys: Any) -> None:
        from scripts.persist_hcp_influence_to_falkordb import main

        cohort_dir = tmp_path / "cohort"
        cohort_dir.mkdir()
        pd.DataFrame(
            [
                {"patient_id": 1, "index_date": pd.Timestamp("2024-06-01")},
                {"patient_id": 2, "index_date": pd.Timestamp("2024-06-15")},
            ]
        ).to_parquet(cohort_dir / "e2i_ml_v3_patient_journeys.parquet")
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        pd.DataFrame(
            [
                {"patid": 1, "npi": "A", "medication_date": pd.Timestamp("2024-05-15")},
                {"patid": 1, "npi": "B", "medication_date": pd.Timestamp("2024-05-16")},
            ]
        ).to_parquet(raw_dir / "medication.parquet")
        pd.DataFrame(
            [{"patid": 1, "npi": "C", "proc_date": pd.Timestamp("2024-05-17")}]
        ).to_parquet(raw_dir / "procedure.parquet")

        exit_code = main(
            [
                "--parquet-dir",
                str(raw_dir),
                "--cohort-dir",
                str(cohort_dir),
                "--cohort-id",
                "smoke",
                "--dry-run",
            ]
        )
        assert exit_code == 0

    def test_missing_input_returns_1(self, tmp_path: Any) -> None:
        from scripts.persist_hcp_influence_to_falkordb import main

        exit_code = main(
            [
                "--parquet-dir",
                str(tmp_path / "nope"),
                "--cohort-dir",
                str(tmp_path / "nope"),
                "--cohort-id",
                "x",
                "--dry-run",
            ]
        )
        assert exit_code == 1


class TestBuildGraphContract:
    """Lock the shared helper's contract so the persistence path stays in sync."""

    def test_returns_none_without_networkx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If networkx isn't importable, the helper logs + returns None."""
        import builtins as _builtins

        real_import = _builtins.__import__

        def bad_import(name: str, *args: Any, **kw: Any) -> Any:
            if name == "networkx":
                raise ImportError("simulated missing networkx")
            return real_import(name, *args, **kw)

        monkeypatch.setattr(_builtins, "__import__", bad_import)
        graph = build_hcp_influence_graph(
            kept_patids={1},
            med=pd.DataFrame({"patid": [1], "npi": ["A"]}),
            proc=pd.DataFrame(columns=["patid", "npi", "proc_date"]),
        )
        assert graph is None

    def test_lookback_gate_excludes_post_index_rows(self) -> None:
        """PR #168 MEDIUM-2 contract preserved by the factored helper."""
        idx = {1: pd.Timestamp("2024-06-01")}
        # Within window — kept
        # Outside window (post-index) — dropped
        # Outside window (too-old) — dropped
        med = pd.DataFrame(
            [
                {"patid": 1, "npi": "A", "medication_date": pd.Timestamp("2024-05-15")},
                {"patid": 1, "npi": "B", "medication_date": pd.Timestamp("2024-07-01")},
                {"patid": 1, "npi": "C", "medication_date": pd.Timestamp("2023-01-01")},
            ]
        )
        proc = pd.DataFrame(columns=["patid", "npi", "proc_date"])
        graph = build_hcp_influence_graph(kept_patids={1}, med=med, proc=proc, idx_by_patid=idx)
        assert graph is not None
        # Only A survives → no edges
        assert set(graph.nodes()) == {"A"}
        assert graph.number_of_edges() == 0

    def test_score_scale_clamped_to_999(self) -> None:
        """``peer_influence_score`` is bounded to fit DECIMAL(3,2)."""
        import networkx as nx

        g = nx.Graph()
        g.add_edge("A", "B", weight=1)
        deg, score = score_hcp_influence_graph(g, scale=PEER_INFLUENCE_SCALE)
        for v in score.values():
            assert 0.0 <= v <= 9.99


# Suppress unused-import lint by exporting; keeps `defaultdict` from
# being eagerly imported if test runner trims (not strictly needed).
_ = defaultdict


class TestTargetGraphResolution:
    """The writer must target the SAME graph the semantic-memory readers use (#890).

    #169's intent: persist the HCP influence graph so that
    FalkorDBSemanticMemory.get_hcp_influence_network() (which resolves
    get_config().semantic.graph_name, deployed: e2i_causal per #749) returns
    data. The hardcoded "e2i_semantic" target re-created the very split-brain
    #169 was filed to close: writer writes an orphan graph, readers read empty.
    """

    def test_resolve_target_graph_name_uses_semantic_config(self) -> None:
        from unittest.mock import MagicMock, patch

        from scripts.persist_hcp_influence_to_falkordb import _resolve_target_graph_name

        cfg = MagicMock()
        cfg.semantic.graph_name = "configured_graph_xyz"
        with patch("src.memory.services.config.get_config", return_value=cfg):
            assert _resolve_target_graph_name() == "configured_graph_xyz"

    def test_main_selects_configured_graph(self, tmp_path: Any) -> None:
        from unittest.mock import MagicMock, patch

        import scripts.persist_hcp_influence_to_falkordb as mod

        fake_nx_graph = MagicMock()
        fake_nx_graph.number_of_nodes.return_value = 1
        fake_nx_graph.number_of_edges.return_value = 0

        client = MagicMock()
        cfg = MagicMock()
        cfg.semantic.graph_name = "configured_graph_xyz"

        with (
            patch.object(mod, "load_cohort_inputs", return_value=({1}, {}, None, None)),
            patch.object(mod, "build_hcp_influence_graph", return_value=fake_nx_graph),
            patch.object(mod, "persist_graph_to_falkordb", return_value=(1, 0)),
            patch("src.memory.services.factories.get_falkordb_client", return_value=client),
            patch("src.memory.services.config.get_config", return_value=cfg),
        ):
            exit_code = mod.main(
                [
                    "--parquet-dir",
                    str(tmp_path),
                    "--cohort-dir",
                    str(tmp_path),
                    "--cohort-id",
                    "smoke",
                ]
            )

        assert exit_code == 0
        client.select_graph.assert_called_once_with("configured_graph_xyz")
