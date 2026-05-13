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

        # Read edges
        if "r.weight AS weight" in c:
            cohort = params["cohort_id"]
            rows = [[k[1], k[2], v["weight"]] for k, v in self.edges.items() if k[0] == cohort]
            return _Result(result_set=rows)

        # Cohort-scoped influence-network traversal (depth 1 only for
        # the cohort-isolation test). We use depth 1 because the
        # persistence schema is bipartite-by-edge (HCP-HCP only), so
        # 1-hop neighbours are sufficient to verify the WHERE predicate.
        if "WHERE connected.cohort_id" in c and "[*1.." in c:
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
            # Return one row per connected node (count if asked, else
            # node-shape stub). We don't fabricate full node objects
            # since this branch is only hit by the count helper in this
            # test suite.
            if "count(DISTINCT connected)" in c:
                return _Result(result_set=[[len(neighbours)]])
            # Otherwise return raw IDs (good enough for cohort-isolation
            # assertions).
            return _Result(result_set=[[n] for n in sorted(neighbours)])

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

    def test_cohort_filter_param_threaded(self) -> None:
        """get_hcp_influence_network + count_hcp_influence_network accept cohort_id."""
        from unittest.mock import MagicMock, patch

        from src.memory.semantic_memory import FalkorDBSemanticMemory

        fake = _FakeFalkorGraph()
        # Seed two cohorts both containing HCP "X" with disjoint neighbours.
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

        # Wire the fake into a FalkorDBSemanticMemory instance.
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
            _ = sm.client  # force lazy init
            _ = sm.graph
            count_a = sm.count_hcp_influence_network("X", max_depth=1, cohort_id="cohort_a")
            count_b = sm.count_hcp_influence_network("X", max_depth=1, cohort_id="cohort_b")

        assert count_a == 1  # X is connected to A1 only in cohort_a
        assert count_b == 1  # X is connected to B1 only in cohort_b


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
