"""Structural-recovery benchmark for guided causal discovery.

WHY THIS EXISTS
---------------
Every other test in this directory verifies MECHANISM — priors get seeded, cycles
get broken, confounders get placed atomically, confidence divides by the converged
count. None of them verified RECOVERY: given data generated from a DAG we know,
does the pipeline return that DAG, and does it return the right backdoor set?

``background_knowledge.py`` claims guided PC was "validated on the patient_journeys
gold standard", but no committed test carried that claim. This benchmark does. It
drives the REAL ``GraphBuilderNode.execute`` end-to-end (discovery -> gate -> DAG ->
backdoor search) against a data-generating process whose true DAG is known by
construction, and measures edge precision / recall / F1 and Structural Hamming
Distance against it.

THE DGP (patient_journeys-shaped, see ``_make_frame``)
------------------------------------------------------
    disease_severity -> treatment_arm,  disease_severity -> persistent_180d   (confounder)
    academic_hcp     -> treatment_arm,  academic_hcp     -> persistent_180d   (confounder)
    region_south     -> treatment_arm                                         (instrument)
    prognostic_only  -> persistent_180d                                       (precision cov)
    noise_cov        -> (nothing)                                             (irrelevant)
    treatment_arm    -> persistent_180d                                       (the estimand)

True minimal backdoor set = {disease_severity, academic_hcp}. ``region_south`` is a
treatment-only predictor and ``prognostic_only`` an outcome-only predictor: neither
is a confounder, and a correct structure learner must NOT place them as common causes.

WHAT THE BENCHMARK ESTABLISHED (2026-09-01, seeds pinned below)
---------------------------------------------------------------
Measured under the requirements.txt pins — causal-learn 0.1.4.3, numpy 2.3.5,
pandas 2.3.3, networkx 3.6.1. The bands below are properties of THOSE versions:
a dependency bump that shifts them should update these numbers, not loosen the
assertions.
1. CAPABILITY (asserted, ``TestGuidedRecoveryWithHonestPriors``): when the declared
   confounders are the true ones, guided PC recovers the DAG essentially correctly.
   Over a 20-run sweep (n = 500 and n = 2000, seeds 1-10):
       exact recovery      12/20        (so exactness is NOT asserted — it is seed-dependent)
       SHD                 max 1, mean 0.40
       reversed edges      0/20         (never inverts an arrow)
       non-confounder placed as a common cause
                           0/20         (never invents confounding)
       true confounders present in the backdoor set
                           20/20        (one run added prognostic_only, a harmless
                                         precision covariate, making a valid superset)
   The invariants that hold in every run — no reversed edge, no invented common cause,
   no omitted true confounder, SHD <= 1 — are what this class asserts. The residual
   errors are single spurious covariate-covariate edges or a missed instrument edge,
   neither of which reaches the backdoor set. This is the claim to protect against
   regression; a tighter assertion would be overfitted to the seeds.

2. KNOWN GAP (pinned as CHARACTERIZATION, ``TestProductionWiringIsPriorDetermined``):
   the API declares EVERY covariate a confounder (``modeled_confounders=covariates``,
   ``src/api/routes/causal.py``), which seeds ``conf->treatment`` AND ``conf->outcome``
   as REQUIRED edges for all of them. Measured consequence: F1 drops to 0.78, SHD 4,
   and the shipped DAG is IDENTICAL whether the frame contains real causal structure
   or is pure noise. Under that wiring the data cannot change the graph.

   The LABELLING half of this gap is now fixed: ``dag_source`` reports
   'prior_asserted' rather than 'discovered' for a prior-implied DAG, and
   ``discovered_confounders`` no longer echoes the declared covariates back
   (tests/unit/test_api/test_causal_agent_analyze.py). The WIRING half stands, and
   the obvious remedy was measured and REJECTED — dropping the required confounder
   edges (tiers only, letting the data select) recovers cleaner STRUCTURE but drops
   a true confounder from the backdoor set in 7/20 runs on this DGP and 3/20 on an
   all-binary variant, i.e. it trades a labelling problem for a CONFOUNDED estimate.
   Adding the curated set back after discovery is not a middle path either: it
   reproduces the prior-implied DAG exactly, because ``_add_curated_confounder_edges``
   draws the same two edges on the ACCEPT path. Over-declaration costs precision,
   not bias (ATE +0.1439 all-covariates vs +0.1420 true-confounders-only, true
   0.1586), so the wiring is deliberately LEFT AS IS pending a design that reports
   per-edge provenance rather than one that removes the adjustment guarantee.

3. KNOWN GAP (pinned, ``TestPostTreatmentCovariateIsNotRejected``): a post-treatment
   MEDIATOR declared as a confounder is forced in with its edge reversed, gate-ACCEPTed,
   and shipped in the adjustment set. Measured ATE consequence on the mediator DGP:
   true total effect +0.2925, correct set +0.2887 (-1%), pipeline set +0.1182 (-60%).

4. KNOWN GAP (pinned, ``TestGateCannotRejectSingleAlgorithmRuns``): with one converged
   algorithm — which is what guided mode runs — agreement and edge confidence are both
   1.0 by construction, so 0.4 + 0.4 already meets ``accept_threshold``; ACCEPT is
   reached before ``structure_score`` is consulted.

5. KNOWN GAP (pinned, ``TestBinaryFramesGetAGaussianTest``): ``_select_independence_test``
   returns ``fisherz`` for an all-binary numeric frame; the ``chisq`` branch needs a
   non-numeric dtype, which PC's ``data.values`` path cannot consume.

The characterization tests (2-5) PIN CURRENT BEHAVIOUR SO A FIX IS NOTICED. They are
not an endorsement of it. If one fails because someone corrected the wiring, the gate,
or the test selection: that is the fix landing — update the test to assert the new,
better behaviour and delete the corresponding gap note above.

SCOPE / FAITHFULNESS: this is a synthetic linear-logistic DGP with Gaussian
confounders, no missingness and n <= 2000 — a faithful test of the ALGORITHM AND ITS
WIRING, not of the real patient_journeys distribution. Running the same harness
against the live frame remains the stronger check.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple, cast

import numpy as np
import pandas as pd
import pytest

from src.agents.causal_impact.nodes.graph_builder import GraphBuilderNode
from src.agents.causal_impact.state import CausalImpactState
from src.causal_engine.discovery.algorithms.pc_wrapper import PCAlgorithm
from src.causal_engine.discovery.base import (
    AlgorithmResult,
    DiscoveredEdge,
    DiscoveryAlgorithmType,
    DiscoveryConfig,
    DiscoveryResult,
    EdgeType,
    GateDecision,
)
from src.causal_engine.discovery.gate import DiscoveryGate

TREATMENT = "treatment_arm"
OUTCOME = "persistent_180d"

TRUE_CONFOUNDERS = ["academic_hcp", "disease_severity"]
NON_CONFOUNDERS = ["noise_cov", "prognostic_only", "region_south"]
ALL_COVARIATES = sorted(TRUE_CONFOUNDERS + NON_CONFOUNDERS)

TRUE_EDGES: Set[Tuple[str, str]] = {
    ("disease_severity", TREATMENT),
    ("disease_severity", OUTCOME),
    ("academic_hcp", TREATMENT),
    ("academic_hcp", OUTCOME),
    ("region_south", TREATMENT),
    ("prognostic_only", OUTCOME),
    (TREATMENT, OUTCOME),
}


def _make_frame(n: int, seed: int) -> pd.DataFrame:
    """Sample the DGP documented in the module docstring."""
    rng = np.random.default_rng(seed)
    severity = rng.normal(0.0, 1.0, n)
    academic = rng.binomial(1, 0.35, n).astype(float)
    region = rng.binomial(1, 0.40, n).astype(float)
    prognostic = rng.normal(0.0, 1.0, n)
    noise = rng.normal(0.0, 1.0, n)

    logit_t = -0.2 + 0.9 * severity + 0.8 * academic + 0.7 * region
    treatment = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit_t)), n).astype(float)

    logit_y = -0.3 + 0.8 * treatment + 0.8 * severity + 0.7 * academic + 0.6 * prognostic
    outcome = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit_y)), n).astype(float)

    return pd.DataFrame(
        {
            TREATMENT: treatment,
            OUTCOME: outcome,
            "disease_severity": severity,
            "academic_hcp": academic,
            "region_south": region,
            "prognostic_only": prognostic,
            "noise_cov": noise,
        }
    )


def _make_noise_frame(n: int, seed: int) -> pd.DataFrame:
    """Same columns, ZERO causal structure — every variable independent."""
    rng = np.random.default_rng(seed)
    frame: Dict[str, Any] = {
        TREATMENT: rng.binomial(1, 0.5, n).astype(float),
        OUTCOME: rng.binomial(1, 0.5, n).astype(float),
    }
    for column in ALL_COVARIATES:
        frame[column] = rng.normal(0.0, 1.0, n)
    return pd.DataFrame(frame)


def _make_mediator_frame(n: int, seed: int) -> pd.DataFrame:
    """DGP where ``adherence_90d`` is a POST-TREATMENT mediator: T -> M -> Y."""
    rng = np.random.default_rng(seed)
    severity = rng.normal(0.0, 1.0, n)
    treatment = rng.binomial(1, 1.0 / (1.0 + np.exp(-(0.9 * severity))), n).astype(float)
    mediator = 1.2 * treatment + 0.5 * severity + rng.normal(0.0, 1.0, n)
    logit_y = 0.8 * treatment + 0.9 * mediator + 0.8 * severity
    outcome = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit_y)), n).astype(float)
    return pd.DataFrame(
        {
            TREATMENT: treatment,
            OUTCOME: outcome,
            "disease_severity": severity,
            "adherence_90d": mediator,
        }
    )


async def _build_dag(
    data: pd.DataFrame,
    declared_confounders: Sequence[str],
    *,
    guided: bool = True,
) -> Dict[str, Any]:
    """Drive the real graph_builder node the way the agent API drives it."""
    state: Dict[str, Any] = {
        "query": f"What is the causal effect of {TREATMENT} on {OUTCOME}?",
        "treatment_var": TREATMENT,
        "outcome_var": OUTCOME,
        "confounders": list(declared_confounders),
        "modeled_confounders": list(declared_confounders),
        "data_cache": {"estimation_data": data},
        "auto_discover": True,
        "discovery_guided": guided,
    }
    result = await GraphBuilderNode().execute(cast(CausalImpactState, state))
    assert result.get("status") != "failed", result.get("error_message")
    graph = result["causal_graph"]
    adjustment_sets = graph.get("adjustment_sets") or [[]]
    return {
        "edges": {(u, v) for u, v in graph["edges"]},
        "adjustment_set": set(adjustment_sets[0]),
        "gate_decision": graph.get("discovery_gate_decision"),
        "dag_overridden": bool(graph.get("discovery_dag_overridden")),
        "n_discovered_edges": graph.get("discovery_n_edges"),
        "skip_reason": result.get("discovery_skip_reason"),
    }


def _structural_metrics(edges: Iterable[Tuple[str, str]]) -> Dict[str, float]:
    """Edge precision / recall / F1 and Structural Hamming Distance vs TRUE_EDGES.

    A reversed edge costs 2 SHD (one false positive plus one false negative), which
    is the intent: an inverted arrow is a worse error than a missing one.
    """
    found = set(edges)
    true_positives = len(found & TRUE_EDGES)
    false_positives = len(found - TRUE_EDGES)
    false_negatives = len(TRUE_EDGES - found)
    precision = true_positives / len(found) if found else 0.0
    recall = true_positives / len(TRUE_EDGES)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "shd": float(false_positives + false_negatives),
    }


class TestGuidedRecoveryWithHonestPriors:
    """CAPABILITY. Guided PC recovers the true DAG when the declared confounders are
    the true confounders. Asserted as a seed sweep with measured bands rather than a
    single lucky seed: exact recovery is seed-dependent (12/20), but the invariants
    below held in every one of the 20 runs. A regression here means the guided
    discovery method itself has broken."""

    SWEEP = [(n, seed) for n in (500, 2000) for seed in range(1, 11)]

    @pytest.mark.asyncio
    async def test_never_reverses_an_edge_or_invents_a_common_cause(self) -> None:
        """The two error classes that would corrupt the adjustment set: an inverted
        arrow, or a non-confounder placed as a common cause of treatment AND outcome."""
        for n_rows, seed in self.SWEEP:
            edges = (await _build_dag(_make_frame(n_rows, seed), TRUE_CONFOUNDERS))["edges"]
            reversed_edges = {(u, v) for u, v in edges if (v, u) in TRUE_EDGES} - TRUE_EDGES
            assert not reversed_edges, f"n={n_rows} seed={seed} reversed {sorted(reversed_edges)}"
            for covariate in NON_CONFOUNDERS:
                assert not ((covariate, TREATMENT) in edges and (covariate, OUTCOME) in edges), (
                    f"n={n_rows} seed={seed}: {covariate} invented as a common cause"
                )

    @pytest.mark.asyncio
    async def test_structural_error_stays_within_one_edge(self) -> None:
        """Measured band: SHD <= 1 in every run, mean 0.40. Exactness is NOT asserted
        (12/20) — pinning it would overfit the seed."""
        distances = []
        for n_rows, seed in self.SWEEP:
            edges = (await _build_dag(_make_frame(n_rows, seed), TRUE_CONFOUNDERS))["edges"]
            shd = _structural_metrics(edges)["shd"]
            assert shd <= 1.0, (
                f"n={n_rows} seed={seed} SHD={shd}: "
                f"spurious={sorted(edges - TRUE_EDGES)} missing={sorted(TRUE_EDGES - edges)}"
            )
            distances.append(shd)
        assert sum(distances) / len(distances) <= 0.6

    @pytest.mark.asyncio
    async def test_backdoor_set_never_omits_a_true_confounder(self) -> None:
        """The assertion that actually protects the estimate. A superset is admissible
        (it costs precision, not bias); a MISSING confounder leaves it confounded."""
        for n_rows, seed in self.SWEEP:
            result = await _build_dag(_make_frame(n_rows, seed), TRUE_CONFOUNDERS)
            assert set(TRUE_CONFOUNDERS) <= result["adjustment_set"], (
                f"n={n_rows} seed={seed} dropped a true confounder: "
                f"{sorted(result['adjustment_set'])}"
            )
            # Nothing that would BIAS the estimate may enter: no descendant of
            # treatment, and no variable the DAG does not license.
            assert "noise_cov" not in result["adjustment_set"]

    @pytest.mark.asyncio
    async def test_returns_the_true_minimal_backdoor_set_on_the_pinned_seed(self) -> None:
        result = await _build_dag(_make_frame(2000, 2000), TRUE_CONFOUNDERS)
        assert result["adjustment_set"] == set(TRUE_CONFOUNDERS)
        assert result["edges"] == TRUE_EDGES


class TestProductionWiringIsPriorDetermined:
    """KNOWN GAP (see docstring item 2), pinned as characterization.

    Declaring every covariate a confounder — what the agent API does today — forces
    ``conf->treatment`` and ``conf->outcome`` for all of them, so the shipped DAG no
    longer depends on the data. Delete this class and fix the assertions in
    ``TestGuidedRecoveryWithHonestPriors`` if the wiring is narrowed."""

    @pytest.mark.asyncio
    async def test_real_data_and_pure_noise_produce_the_same_dag(self) -> None:
        real = await _build_dag(_make_frame(2000, 2000), ALL_COVARIATES)
        noise = await _build_dag(_make_noise_frame(2000, 1), ALL_COVARIATES)

        # Discovery genuinely found nothing in the noise frame...
        assert noise["n_discovered_edges"] == 0
        assert noise["gate_decision"] == GateDecision.REJECT.value
        # ...and the gate correctly rejected it. It changes nothing: the manual DAG
        # asserts the same declared common causes, so both frames ship one graph.
        assert real["edges"] == noise["edges"]
        assert real["adjustment_set"] == noise["adjustment_set"]

    @pytest.mark.asyncio
    async def test_non_confounders_enter_the_adjustment_set(self) -> None:
        result = await _build_dag(_make_frame(2000, 2000), ALL_COVARIATES)
        assert result["adjustment_set"] == set(ALL_COVARIATES)
        assert _structural_metrics(result["edges"])["shd"] == 4.0

    @pytest.mark.asyncio
    async def test_gate_accepts_the_prior_determined_dag(self) -> None:
        """The gate ACCEPTs and nothing is overridden, so the DAG ships as-is even
        though the identical graph is produced from noise. The API layer no longer
        calls this combination 'discovered' — it reports 'prior_asserted' when every
        shipped edge is prior-implied (see test_causal_agent_analyze.py) — but the
        gate decision itself is unchanged, which is what this pins."""
        result = await _build_dag(_make_frame(2000, 2000), ALL_COVARIATES)
        assert result["gate_decision"] == GateDecision.ACCEPT.value
        assert result["dag_overridden"] is False


class TestPostTreatmentCovariateIsNotRejected:
    """KNOWN GAP (see docstring item 3), pinned as characterization.

    Nothing validates that a declared confounder is pre-treatment. A mediator is
    forced in with its true edge reversed and shipped in the adjustment set;
    conditioning on it attenuated the measured ATE by 60%."""

    @pytest.mark.asyncio
    async def test_mediator_declared_as_confounder_is_shipped_in_adjustment_set(self) -> None:
        result = await _build_dag(
            _make_mediator_frame(5000, 3), ["disease_severity", "adherence_90d"]
        )
        assert "adherence_90d" in result["adjustment_set"]
        # The prior wins over the data: the true direction is treatment -> mediator.
        assert ("adherence_90d", TREATMENT) in result["edges"]
        assert (TREATMENT, "adherence_90d") not in result["edges"]
        assert result["gate_decision"] == GateDecision.ACCEPT.value


class TestGateCannotRejectSingleAlgorithmRuns:
    """KNOWN GAP (see docstring item 4), pinned as characterization.

    Guided mode restricts discovery to PC alone. With one converged algorithm every
    edge scores confidence 1.0 and agreement 1.0 by construction, so the weighted
    score reaches ``accept_threshold`` (0.4 + 0.4) before ``structure_score`` is
    added — no single-algorithm result with at least one edge can be rejected."""

    @staticmethod
    def _single_algorithm_result(edges: List[Tuple[str, str]]) -> DiscoveryResult:
        discovered = [
            DiscoveredEdge(
                source=source,
                target=target,
                edge_type=EdgeType.DIRECTED,
                confidence=1.0,
                algorithm_votes=1,
                algorithms=["pc"],
            )
            for source, target in edges
        ]
        import networkx as nx

        dag = nx.DiGraph()
        dag.add_edges_from(edges)
        dag.add_nodes_from(f"isolated_{i}" for i in range(8))
        return DiscoveryResult(
            success=True,
            config=DiscoveryConfig(algorithms=[DiscoveryAlgorithmType.PC]),
            ensemble_dag=dag,
            edges=discovered,
            algorithm_results=[
                AlgorithmResult(
                    algorithm=DiscoveryAlgorithmType.PC,
                    adjacency_matrix=np.zeros((2, 2), dtype=int),
                    edge_list=list(edges),
                    runtime_seconds=0.01,
                    converged=True,
                )
            ],
        )

    def test_structurally_poor_result_is_still_accepted(self) -> None:
        """One unrelated edge among eight isolated nodes: structure score ~0.1,
        expected-edge recall 0 — and the gate still ACCEPTs."""
        result = self._single_algorithm_result([("unrelated_a", "unrelated_b")])
        evaluation = DiscoveryGate().evaluate(result, expected_edges=[(TREATMENT, OUTCOME)])

        assert evaluation.decision == GateDecision.ACCEPT
        assert evaluation.metadata["structure_score"] < 0.5
        assert evaluation.metadata["agreement_score"] == 1.0
        assert evaluation.metadata["edge_confidence_score"] == 1.0

    def test_reversed_estimand_edge_is_still_accepted(self) -> None:
        result = self._single_algorithm_result([(OUTCOME, TREATMENT)])
        assert DiscoveryGate().evaluate(result).decision == GateDecision.ACCEPT


class TestBinaryFramesGetAGaussianTest:
    """KNOWN GAP (see docstring item 5), pinned as characterization.

    ``chisq`` is unreachable for any frame PC can actually run on: the branch needs a
    non-numeric dtype, but PC consumes ``data.values``. Binary treatment/outcome
    columns are therefore tested with Fisher's z, a linear-Gaussian test."""

    def test_all_binary_frame_selects_fisherz(self) -> None:
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(
            {name: rng.binomial(1, 0.5, 500).astype(float) for name in ("a", "b", "c")}
        )
        config = DiscoveryConfig(algorithms=[DiscoveryAlgorithmType.PC])
        assert PCAlgorithm()._select_independence_test(frame, config) == "fisherz"
