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
   Over a 20-run sweep (n = 500 and n = 2000, seeds 1-10), re-measured 2026-09-02
   through the corroboration gate at the shipped guided default B=20, with the
   AUGMENT fix (item 4) also in effect:
       gate ACCEPTed       18/20        (n=500 seed 2 REVIEW, seed 8 AUGMENT)
       gate shipped a discovered edge
                           19/20        (the 18 ACCEPTs plus seed 8 AUGMENT, which
                                         now actually augments — see item 4)
       exact recovery      11/20        (so exactness is NOT asserted — it is seed-dependent)
       SHD                 max 1 and mean 0.39 over the 18 ACCEPTed runs;
                           max 2 and mean 0.5 over all 20 shipped DAGs
       reversed edges      0/20         (never inverts an arrow)
       non-confounder placed as a common cause
                           0/20         (never invents confounding)
       true confounders present in the backdoor set
                           20/20        (exactly {disease_severity, academic_hcp} in
                                         every run — no superset, no omission)
   The invariants that hold in every run are what this class asserts: no reversed
   edge, no invented common cause, no omitted true confounder, and SHD <= 1 whenever
   the gate ACCEPTs or AUGMENTs. The residual errors on accepted/augmented runs are
   single spurious covariate-covariate edges or a missed instrument edge, neither of
   which reaches the backdoor set. On the one withheld run (n=500 seed 2, REVIEW) the
   bare manual DAG ships instead, a strict SUBSET of the true DAG missing both
   beyond-prior true edges — a recovery loss, not a correctness one, which is why the
   assertion there is subset-safety rather than SHD. This is the claim to protect
   against regression; a tighter assertion would be overfitted to the seeds.

2. FIXED (2026-09-02, was KNOWN GAP "production wiring is prior-determined"):
   the API used to declare EVERY covariate a confounder
   (``modeled_confounders=covariates``), seeding ``conf->treatment`` AND
   ``conf->outcome`` as REQUIRED edges for all of them — the shipped DAG was
   IDENTICAL for real structure and pure noise (F1 0.78, SHD 4). The naive
   remedy (tiers only, data selects, backdoor from the DAG) was measured and
   REJECTED at the time: it dropped a true confounder from the backdoor set in
   7/20 runs — a CONFOUNDED estimate. The fix SPLITS the fused channels:

     - ``anchored_confounders`` (STRUCTURAL prior): only these seed required
       edges. The agent API passes [] — its dataset-spec covariate list is a
       role allowlist, not a per-question confounder assertion — so prod runs
       tiers + estimand-edge priors and the data selects the confounder edges.
     - ``modeled_confounders`` (ADJUSTMENT GUARANTEE): every declared covariate
       present in the shipped DAG (and not a treatment descendant) is UNIONED
       into the final adjustment sets (``_apply_adjustment_guarantee``), so the
       7/20-class structural misses are harmless: the conditioning set stays
       exactly the declared covariates BY CONSTRUCTION (over-declaration costs
       precision, not bias — ATE +0.1439 all-covariates vs +0.1420 true-only,
       true +0.1586). The ACCEPT path re-asserts only ANCHORED confounders on
       the discovered DAG; legacy callers without the anchored key keep the old
       full re-add.
     - Per-edge provenance ships on the graph (``edge_provenance``:
       required_prior / discovered / curated) and on the API DAG model.

   Measured under the prod shape (anchored=[], declared=ALL, B=20, seeds 1-10):
   real frames 19/20 ACCEPT + 1 AUGMENT, ALL on the bootstrap_stability basis
   (prior_determined no longer occurs); shipped-DAG F1 mean 0.93 (n=2000:
   mean 0.98, SHD <= 1, 7/10 exact; n=500: 0.78-1.00); noise frames 0/20
   ACCEPT (16 reject / 4 review) shipping the all-curated manual assertion;
   final adjustment set == declared covariates in 40/40 runs. Real-vs-noise
   DAGs now DIFFER (``TestProductionWiringIsDataResponsive``). ``dag_source``
   tracks the anchored channel, so honest prod ACCEPTs read 'discovered'
   (prior_asserted still applies to genuinely anchored priors). The mediator
   DGP (item 3) is byte-identical under both shapes — tiers, not the required
   edges, force the reversed direction — so its pins stand unchanged.

3. KNOWN GAP (pinned, ``TestPostTreatmentCovariateIsNotRejected``): a post-treatment
   MEDIATOR declared as a confounder is forced in with its edge reversed, gate-ACCEPTed,
   and shipped in the adjustment set. Measured ATE consequence on the mediator DGP:
   true total effect +0.2925, correct set +0.2887 (-1%), pipeline set +0.1182 (-60%).

4. FIXED (2026-09-02, was KNOWN GAP "gate cannot reject single-algorithm runs"):
   the gate now scores CORROBORATION — bootstrap resample stability over
   beyond-prior edges for single-algorithm runs (``DiscoveryConfig.
   bootstrap_resamples``, guided default 20) — instead of self-agreement.
   Measured at B=20 under the same pins: honest-priors sweep 18/20 ACCEPT
   (n=2000: 10/10; n=500: 8/10 — seed 2 REVIEW, seed 8 AUGMENT; band
   user-accepted). Noise frames with no declared confounders: 0/20 ACCEPT
   (n=2000: 7 reject / 3 review; n=500: 9 reject / 1 review). A run with
   no stability data is uncorroborated (scores 0.0) and cannot be accepted.
   Prior-determined runs (every edge required) renormalize over edge
   confidence + structure and still ACCEPT — the wiring itself is gap 2.

   Also fixed in this branch (2026-09-02): AUGMENT used to be inert.
   ``GateEvaluation.to_dict`` serialized only ``n_high_confidence_edges`` (a
   count) but not the edges themselves, so graph_builder's AUGMENT branch
   iterated an empty list and shipped the bare manual DAG — indistinguishable
   from REVIEW. n=500 seed 8 is the first honest-priors case that reaches
   AUGMENT at all, and before this fix it reached that decision without ever
   actually augmenting the shipped DAG. ``to_dict`` now also serializes
   ``high_confidence_edges``, so AUGMENT ships the manual DAG plus its
   corroborated beyond-prior edges:
   n=500 seed 8 now ships the 5-edge prior DAG plus the corroborated
   ``prognostic_only -> persistent_180d`` edge, at SHD 1 (previously SHD 2 as a
   bare manual DAG, and counted as withheld). Of the honest-priors sweep's two
   non-ACCEPT runs, only n=500 seed 2 (REVIEW) still withholds discovery
   entirely — the withheld count for that sweep is now 1, not 2.

5. KNOWN GAP (pinned, ``TestBinaryFramesGetAGaussianTest``): ``_select_independence_test``
   returns ``fisherz`` for an all-binary numeric frame; the ``chisq`` branch needs a
   non-numeric dtype, which PC's ``data.values`` path cannot consume. The obvious
   remedy — cardinality-based selection routing all-binary frames to ``chisq`` — was
   measured 2026-09-02 and REJECTED: causal-learn's ``chisq`` DOES accept float-coded
   0/1 data (``Chisq_or_Gsq.__init__`` re-encodes every column to 0..k-1 via
   ``np.unique`` itself, so the dtype guard protects no input-capability
   constraint of causal-learn's), but on an all-binary
   variant of this DGP (every covariate Bernoulli, same structure and coefficients),
   driven through the real ``GraphBuilderNode`` at the guided default B=20 over the
   same 20-point sweep, chisq did NOT improve recovery: mean F1 0.943 vs 0.953,
   mean SHD 0.70 vs 0.60, one more withheld run (5 REVIEW vs fisherz's 4), with the
   backdoor set correct 20/20, zero reversed edges and zero invented common causes
   under BOTH tests (paired per-seed: fisherz strictly better on 3 sweep points,
   chisq on 1 — within seed noise). Selection is deliberately LEFT AS IS: the dead
   ``chisq`` branch is a code-shape defect, not a measured recovery loss, and a
   selection change would re-open the fix-1/fix-2 measured bands for no gain.

6. NEW CAPABILITY (2026-09-02, ``TestLatentConfounderProducesAFlag``): a
   latent-confounding DIAGNOSTIC. Guided discovery is PC-only and PC assumes
   causal sufficiency, so before this nothing in the pipeline could NOTICE a
   latent confounder. One UNGUIDED FCI run now rides along
   (``DiscoveryConfig.latent_diagnostic``; graph_builder defaults it ON for
   guided runs via state key ``discovery_latent_diagnostic``), surfacing as
   ``causal_graph["latent_diagnostic"]`` — converged, runtime, bidirected pairs
   by column name, estimand, flag — carried through gate-evaluation metadata as
   a pass-through (never a gate input: item 4's calibration is untouched,
   pinned by an on-vs-off equality test) plus a warning through the state accumulator
   into the analyze response when the flag is up.

   Flag predicate (MEASURED, alpha=0.05/fisherz, seeds 1-10): the ESTIMAND
   PAIR carries a bidirected mark in FCI's PAG. Operating point:
       null-effect latent DGP (treatment_effect=0, disease_severity dropped —
       the whole T-Y dependence is an unmeasured common cause):
                                        flag 10/10 at n=2000, 6/10 at n=500
       same DGP, severity OBSERVED:     0/10 (FCI separates T from Y)
       pure-noise frames:               0/10 (covariate-level bidirected marks
                                        occur — n=2000 seed 7 — and do NOT flag)
   Covariate-level bidirected pairs stay in the payload as data but never
   raise the flag: they appeared on the observed-confounder control (2/10 at
   n=2000, 4/10 at n=500 across alpha 0.05) and on noise, i.e. they are
   orientation noise on this DGP family, not latent evidence.

   SURFACING POLICY (2026-09-02, measured before designed): a full alpha
   sweep through the production seam (``_run_latent_diagnostic`` + the
   estimand-pair predicate; alpha 0.005/0.01/0.02/0.05/0.1/0.2, n=2000,
   seeds 1-10) put the shipped alpha=0.05 at the knee — detection 10/10 at
   EVERY level, specificity 0/10 through 0.05 (then 1/10 at 0.1, 3/10 at
   0.2: false alarms where a MEASURED confounder explains the dependence),
   and the effectful-frame mark rate MONOTONE DECREASING in alpha (10/10 at
   0.005 down to 6/10 at 0.2) with latent==observed at every level. No
   discovery-side knob reduces the fatigue rate without buying misleading
   alarms, so fatigue is handled as POLICY: graph_builder only annotates the
   payload and logs the flag (base-rate observability; MLflow keeps
   latent_diagnostic_ran/flag per run), and ``InterpretationNode`` surfaces
   the warning iff the flag is corroborated by the E-value
   (robust_to_confounding False, or sensitivity unavailable — fail-open).
   A mark on an estimate the E-value calls robust is, per the table above,
   indistinguishable from orientation noise.

   KNOWN LIMIT (measured, then accepted): latent confounding ON TOP of a real
   direct effect is NOT detectable and is not claimed. With T an ancestor of Y
   the true MAG orients T -> Y — there is no bidirected mark to find — and the
   T <-> Y marks that DO appear on effectful frames (7/10 at n=2000) are
   fisherz-on-binary orientation noise appearing at the SAME rate whether the
   confounder is latent or observed (7/10 vs 7/10; n=500: 6/10 vs 6/10). No
   measured statistic separates them: alpha in {0.01, 0.05, 0.1, 0.2} (flag
   rates track within 1/10 of each other at every level), bootstrap stability
   of the mark over B=20 (true-signal 0.65-1.00 vs artifact 0.35-0.95 —
   overlapping), and requiring a near-zero resample directed-rate (effectful
   control still 6/10 at theta=0.05) all fail to discriminate. So the flag on
   an effectful frame means exactly what the warning says — FCI could not
   attribute the T-Y dependence to the treatment — and quantitative latent-
   confounding robustness stays where it belongs, with the E-value sensitivity
   analysis.

The remaining characterization tests (3, 5) PIN CURRENT BEHAVIOUR SO A FIX IS
NOTICED. They are not an endorsement of it. If one fails because someone corrected the
wiring or the test selection: that is the fix landing — update the test to assert the
new, better behaviour and delete the corresponding gap note above. Items 2 and 4 are
what that looks like once done: each gap note became a fix record, and
``TestGateRejectsUncorroboratedSingleAlgorithmRuns`` /
``TestProductionWiringIsDataResponsive`` now assert the corrected behaviour.

SCOPE / FAITHFULNESS: this is a synthetic linear-logistic DGP with Gaussian
confounders, no missingness and n <= 2000 — a faithful test of the ALGORITHM AND ITS
WIRING, not of the real patient_journeys distribution. Running the same harness
against the live frame remains the stronger check.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, cast

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


def _make_frame(n: int, seed: int, treatment_effect: float = 0.8) -> pd.DataFrame:
    """Sample the DGP documented in the module docstring.

    ``treatment_effect`` scales the treatment coefficient in the outcome logit
    (default 0.8, the documented DGP). ``0.0`` gives the NULL-EFFECT variant
    used by the latent-confounding diagnostic benchmark (item 6): identical
    draws and structure, but the entire T-Y dependence flows through the
    confounders.
    """
    rng = np.random.default_rng(seed)
    severity = rng.normal(0.0, 1.0, n)
    academic = rng.binomial(1, 0.35, n).astype(float)
    region = rng.binomial(1, 0.40, n).astype(float)
    prognostic = rng.normal(0.0, 1.0, n)
    noise = rng.normal(0.0, 1.0, n)

    logit_t = -0.2 + 0.9 * severity + 0.8 * academic + 0.7 * region
    treatment = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit_t)), n).astype(float)

    logit_y = (
        -0.3 + treatment_effect * treatment + 0.8 * severity + 0.7 * academic + 0.6 * prognostic
    )
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
    anchored: Optional[Sequence[str]] = None,
    guided: bool = True,
    bootstrap_resamples: Optional[int] = None,
    latent_diagnostic: Optional[bool] = None,
) -> Dict[str, Any]:
    """Drive the real graph_builder node the way the agent API drives it.

    ``bootstrap_resamples=None`` leaves the state key OUT, so the node applies
    its own guided default (20) — that is the wiring the agent API ships, and
    the sweeps below measure the system AT that default. Pass an int to pin a
    resample count explicitly (0 turns stability off). ``latent_diagnostic``
    mirrors that idiom for the FCI diagnostic's ``discovery_latent_diagnostic``
    state key (guided node default: ON).

    Fix 4 split the confounder wiring into two channels. ``declared_confounders``
    always fills the ADJUSTMENT-GUARANTEE channel (``modeled_confounders`` — these
    are unioned into the final adjustment sets no matter what the DAG shows).
    ``anchored=None`` (default) also anchors them as STRUCTURAL priors
    (``anchored_confounders`` — required conf->treatment/conf->outcome edges),
    which is the honest-priors benchmark shape the capability bands were measured
    under. ``anchored=[]`` is the agent API's PRODUCTION shape: no structural
    priors, tiers + the estimand edge only, the data selects the confounder edges.
    """
    state: Dict[str, Any] = {
        "query": f"What is the causal effect of {TREATMENT} on {OUTCOME}?",
        "treatment_var": TREATMENT,
        "outcome_var": OUTCOME,
        "confounders": list(declared_confounders),
        "modeled_confounders": list(declared_confounders),
        "anchored_confounders": list(declared_confounders if anchored is None else anchored),
        "data_cache": {"estimation_data": data},
        "auto_discover": True,
        "discovery_guided": guided,
    }
    if bootstrap_resamples is not None:
        state["discovery_bootstrap_resamples"] = bootstrap_resamples
    if latent_diagnostic is not None:
        state["discovery_latent_diagnostic"] = latent_diagnostic
    result = await GraphBuilderNode().execute(cast(CausalImpactState, state))
    assert result.get("status") != "failed", result.get("error_message")
    graph = result["causal_graph"]
    adjustment_sets = graph.get("adjustment_sets") or [[]]
    gate_evaluation = result.get("discovery_gate_evaluation") or {}
    return {
        "edges": {(u, v) for u, v in graph["edges"]},
        "adjustment_set": set(adjustment_sets[0]),
        "gate_decision": graph.get("discovery_gate_decision"),
        "corroboration_basis": (gate_evaluation.get("metadata") or {}).get("corroboration_basis"),
        "dag_overridden": bool(graph.get("discovery_dag_overridden")),
        "n_discovered_edges": graph.get("discovery_n_edges"),
        "edge_provenance": {
            (e["source"], e["target"]): e["provenance"]
            for e in (graph.get("edge_provenance") or [])
        },
        "skip_reason": result.get("discovery_skip_reason"),
        "latent_diagnostic": graph.get("latent_diagnostic"),
        # Direct node call (no LangGraph accumulator): exactly the NEW warning
        # entries this execute() returned.
        "warnings": result.get("warnings") or [],
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
    single lucky seed. A regression here means the guided discovery method itself has
    broken.

    These invariants are measured on the discovery + GATE system at the wiring the
    agent API ships — ``bootstrap_resamples`` at graph_builder's guided default of 20
    (``_build_dag`` deliberately omits the state key). The gate is therefore part of
    what is under test: it decides whether the DISCOVERED DAG ships at all, an AUGMENT
    decision ships the manual DAG plus its corroborated beyond-prior edges, and a
    withheld (REVIEW/REJECT) discovery ships the bare manual/prior DAG instead.
    Measured band at B=20: 18/20 ACCEPT (n=2000: 10/10; n=500: 8/10 — seed 2 REVIEW,
    seed 8 AUGMENT). That band is user-accepted; do not chase 20/20. Both non-ACCEPT
    runs are the same two beyond-prior true edges going bootstrap-unstable at n=500;
    seed 8's edge (``prognostic_only -> persistent_180d``) is stable enough to clear
    AUGMENT's own threshold and ships anyway, while seed 2's is not and the fallback
    (a strict subset of the true DAG) ships instead — only seed 2 is actually
    withheld."""

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
        """Gate-aware. Where the gate ACCEPTs or AUGMENTs, a discovered DAG actually
        ships — ACCEPT ships it directly, AUGMENT ships the manual DAG plus its
        corroborated beyond-prior edges (fixed 2026-09-02: ``GateEvaluation.to_dict``
        now serializes the edges, not just a count) — and the old bar holds: SHD <= 1.
        Where the gate withholds entirely (REVIEW/REJECT), the bare manual DAG ships,
        and the bar that matters is that the fallback is SAFE — a non-empty strict
        subset of the true DAG, never spurious structure. Measured at B=20: 19/20
        ship at SHD <= 1 (18 ACCEPT + 1 AUGMENT at n=500 seed 8, which now ships the
        5-edge prior DAG plus the corroborated ``prognostic_only -> persistent_180d``
        edge, SHD 1), 1 withheld (n=500 seed 2, REVIEW) shipping the bare 5-edge prior
        DAG at SHD 2 because both beyond-prior true edges are bootstrap-unstable there.
        """
        distances = []
        withheld = []
        for n_rows, seed in self.SWEEP:
            result = await _build_dag(_make_frame(n_rows, seed), TRUE_CONFOUNDERS)
            edges = result["edges"]
            shd = _structural_metrics(edges)["shd"]
            if result["gate_decision"] in (GateDecision.ACCEPT.value, GateDecision.AUGMENT.value):
                assert shd <= 1.0, (
                    f"n={n_rows} seed={seed} gate={result['gate_decision']} SHD={shd}: "
                    f"spurious={sorted(edges - TRUE_EDGES)} missing={sorted(TRUE_EDGES - edges)}"
                )
            else:
                withheld.append((n_rows, seed, result["gate_decision"]))
                assert edges, (
                    f"n={n_rows} seed={seed} gate={result['gate_decision']}: empty fallback DAG"
                )
                assert edges <= TRUE_EDGES, (
                    f"n={n_rows} seed={seed} gate={result['gate_decision']}: the fallback "
                    f"DAG invented structure {sorted(edges - TRUE_EDGES)}"
                )
            distances.append((n_rows, seed, shd))
        shds = [d for _, _, d in distances]
        assert len(withheld) <= 1, (
            f"gate withheld more than the accepted+augmented band: {withheld}"
        )
        mean_shd = sum(shds) / len(shds)
        # Measured 2026-09-02 (post-fix): 0.5 (19/20 shipped runs at SHD <= 1, one
        # withheld run at SHD 2). Bound carries modest headroom over the measurement.
        assert mean_shd <= 0.55, f"mean SHD {mean_shd} over per-run SHDs {distances}"

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


class TestProductionWiringIsDataResponsive:
    """FIXED GAP (was docstring item 2). The agent API now declares every
    covariate into the adjustment-GUARANTEE channel (``modeled_confounders``)
    and NOTHING into the structural-prior channel (``anchored_confounders=[]``,
    the ``anchored=[]`` shape here), so guided discovery runs with tiers + the
    required estimand edge only: the DATA selects the confounder edges, while
    the guarantee unions every declared covariate into the final adjustment set
    — the estimate's conditioning set is unchanged BY CONSTRUCTION, and a
    structural miss by discovery cannot silently unadjust it."""

    @pytest.mark.asyncio
    async def test_real_data_and_pure_noise_produce_different_dags(self) -> None:
        real = await _build_dag(_make_frame(2000, 2000), ALL_COVARIATES, anchored=[])
        noise = await _build_dag(_make_noise_frame(2000, 1), ALL_COVARIATES, anchored=[])

        # Discovery genuinely finds nothing in the noise frame and the gate
        # rejects it; the manual fallback ships the declared covariates as
        # ASSERTED common causes (labeled curated, never discovered).
        assert noise["n_discovered_edges"] == 0
        assert noise["gate_decision"] == GateDecision.REJECT.value
        assert set(noise["edge_provenance"].values()) == {"curated"}
        # The real frame ships a data-selected DAG — the graphs now DIFFER
        # (measured: real recovers TRUE_EDGES exactly on this seed; noise ships
        # the 11-edge all-covariate assertion).
        assert real["edges"] != noise["edges"]
        assert real["edges"] == TRUE_EDGES
        # The adjustment GUARANTEE holds on both: conditioning set == declared.
        assert real["adjustment_set"] == set(ALL_COVARIATES)
        assert noise["adjustment_set"] == set(ALL_COVARIATES)

    @pytest.mark.asyncio
    async def test_declared_covariates_stay_adjusted_while_structure_recovers(self) -> None:
        """Non-confounders still enter the adjustment set — deliberately: that is
        the guarantee channel keeping the ATE's conditioning set exactly the
        declared covariates (over-declaration costs precision, not bias —
        measured ATE +0.1439 all-covariates vs +0.1420 true-only, true +0.1586).
        What changed is the DAG: SHD 4.0 -> 0.0 on this pinned seed (sweep band:
        F1 mean 0.93, SHD <= 1 in 17/20 runs), and every covariate edge carries
        honest 'discovered' provenance instead of being prior-forced."""
        result = await _build_dag(_make_frame(2000, 2000), ALL_COVARIATES, anchored=[])
        assert result["adjustment_set"] == set(ALL_COVARIATES)
        assert _structural_metrics(result["edges"])["shd"] == 0.0
        assert result["edge_provenance"][(TREATMENT, OUTCOME)] == "required_prior"
        covariate_labels = {
            label
            for edge, label in result["edge_provenance"].items()
            if edge != (TREATMENT, OUTCOME)
        }
        assert covariate_labels == {"discovered"}

    @pytest.mark.asyncio
    async def test_gate_scores_prod_runs_on_bootstrap_stability(self) -> None:
        """With beyond-prior edges now existing in every prod-shaped run, the
        fix-2 gate scores real evidence — bootstrap resample stability — instead
        of renormalizing over a prior-determined graph. This is the decision
        basis that lets it genuinely REJECT the noise frame above."""
        result = await _build_dag(_make_frame(2000, 2000), ALL_COVARIATES, anchored=[])
        assert result["gate_decision"] == GateDecision.ACCEPT.value
        assert result["corroboration_basis"] == "bootstrap_stability"
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


class TestGateRejectsUncorroboratedSingleAlgorithmRuns:
    """FIXED GAP (was docstring item 4): a single-algorithm result used to
    self-agree at 1.0 and reach ACCEPT unconditionally. The gate now scores
    corroboration — bootstrap stability for single-algorithm runs — and a run
    with NO stability evidence scores 0.0 and cannot be accepted, nor can its
    uncorroborated edges qualify for AUGMENT."""

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

    def test_structurally_poor_uncorroborated_result_is_rejected(self) -> None:
        result = self._single_algorithm_result([("unrelated_a", "unrelated_b")])
        evaluation = DiscoveryGate().evaluate(result, expected_edges=[(TREATMENT, OUTCOME)])
        assert evaluation.decision == GateDecision.REJECT
        assert evaluation.metadata["corroboration_score"] == 0.0
        assert evaluation.metadata["corroboration_basis"] == "uncorroborated_single_run"

    def test_reversed_estimand_edge_is_rejected(self) -> None:
        result = self._single_algorithm_result([(OUTCOME, TREATMENT)])
        assert DiscoveryGate().evaluate(result).decision == GateDecision.REJECT


class TestBootstrapStabilityGatesNoiseAtTheDataLevel:
    """End-to-end through the real GraphBuilderNode: with bootstrap on and no
    declared confounders, spurious noise-frame edges are unstable, so the
    gate no longer ACCEPTs the discovered DAG — the manual fallback ships
    instead. Pinned to one seed with the prod-default B to match the fix-2
    calibration; the full measured sweep lives in the module docstring
    (item 4)."""

    @pytest.mark.asyncio
    async def test_noise_discoveries_are_not_accepted(self) -> None:
        result = await _build_dag(_make_noise_frame(500, 7), [], bootstrap_resamples=20)
        # Non-vacuity: this seed must keep REACHING the corroboration scoring, not
        # short-circuit on the gate's min-edges check. Measured: 3 edges, REVIEW.
        assert result["n_discovered_edges"], "pin degraded to a vacuous min-edges reject"
        assert result["gate_decision"] != GateDecision.ACCEPT.value

    @pytest.mark.asyncio
    async def test_honest_priors_still_accept_with_bootstrap_on(self) -> None:
        """Positive control: the gate's new muscle must not reject genuine
        structure. Same pinned seed family as the capability sweep."""
        result = await _build_dag(_make_frame(500, 3), TRUE_CONFOUNDERS, bootstrap_resamples=20)
        assert result["gate_decision"] == GateDecision.ACCEPT.value
        assert result["edges"]  # discovered DAG shipped


class TestBinaryFramesGetAGaussianTest:
    """KNOWN GAP (see docstring item 5), pinned as characterization.

    ``chisq`` is unreachable for any frame PC can actually run on: the branch needs a
    non-numeric dtype, but PC consumes ``data.values``. Binary treatment/outcome
    columns are therefore tested with Fisher's z, a linear-Gaussian test.

    Measured 2026-09-02 (item 5): routing all-binary frames to chisq does not
    improve recovery on this benchmark, so the selection — and this pin — stand."""

    def test_all_binary_frame_selects_fisherz(self) -> None:
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(
            {name: rng.binomial(1, 0.5, 500).astype(float) for name in ("a", "b", "c")}
        )
        config = DiscoveryConfig(algorithms=[DiscoveryAlgorithmType.PC])
        assert PCAlgorithm()._select_independence_test(frame, config) == "fisherz"


def _null_effect_latent_frame(n: int, seed: int) -> pd.DataFrame:
    """Latent DGP for the diagnostic benchmark (docstring item 6): the true
    treatment effect is ZERO and ``disease_severity`` — a genuine confounder of
    treatment and outcome — is dropped from the frame, so the entire T-Y
    dependence flows through an unmeasured common cause. This is the class FCI
    can actually detect (measured 10/10 at n=2000): with no direct effect, the
    true MAG edge is T <-> Y. Dropping severity while KEEPING the direct
    effect is not a usable flag case — see the item-6 KNOWN LIMIT."""
    return _make_frame(n, seed, treatment_effect=0.0).drop(columns=["disease_severity"])


class TestLatentConfounderProducesAFlag:
    """CAPABILITY (docstring item 6): the FCI latent-confounding diagnostic,
    driven through the real ``GraphBuilderNode.execute``. All pins at n=2000,
    seeds measured in the item-6 table. ``bootstrap_resamples=0`` keeps each
    run inside CI's 30s thread cap: the diagnostic is orthogonal to gate
    corroboration (the gate never reads it), so the pins lose nothing by
    turning resampling off."""

    @pytest.mark.asyncio
    async def test_null_effect_latent_dgp_raises_the_flag(self) -> None:
        """Measured 10/10 over seeds 1-10 at n=2000; pinned at seed 1. The
        payload must name real columns (the pre-fix ``get_bidirected_edges``
        returned node-INDEX strings). Surfacing policy (item 6): this node
        only annotates — the human-readable warning is raised downstream by
        ``InterpretationNode`` iff the E-value corroborates, so graph_builder
        must NOT put it in the warnings accumulator."""
        result = await _build_dag(
            _null_effect_latent_frame(2000, 1), ["academic_hcp"], bootstrap_resamples=0
        )
        payload = result["latent_diagnostic"]
        assert payload is not None and payload["ran"] and payload["converged"]
        assert payload["flag"] is True
        assert payload["treatment"] == TREATMENT and payload["outcome"] == OUTCOME
        pairs = payload["bidirected_edges"]
        assert any({u, v} == {TREATMENT, OUTCOME} for u, v in pairs), pairs
        flat = {name for pair in pairs for name in pair}
        assert flat <= {
            TREATMENT,
            OUTCOME,
            "academic_hcp",
            "region_south",
            "prognostic_only",
            "noise_cov",
        }
        assert not any("Latent-confounding diagnostic" in w for w in result["warnings"]), result[
            "warnings"
        ]

    @pytest.mark.asyncio
    async def test_diagnostic_does_not_change_the_gate_decision_or_the_dag(self) -> None:
        """Constraint pin: fix 2 calibrated the gate's accept/reject bands; the
        diagnostic must be a pure annotation. Identical frame and config with
        the diagnostic ON vs OFF must ship the identical gate decision, DAG,
        and adjustment set."""
        frame = _null_effect_latent_frame(2000, 1)
        on = await _build_dag(frame, ["academic_hcp"], bootstrap_resamples=0)
        off = await _build_dag(
            frame, ["academic_hcp"], bootstrap_resamples=0, latent_diagnostic=False
        )
        assert off["latent_diagnostic"] is None  # opt-out actually opts out
        assert on["gate_decision"] == off["gate_decision"]
        assert on["edges"] == off["edges"]
        assert on["adjustment_set"] == off["adjustment_set"]

    @pytest.mark.asyncio
    async def test_observed_confounders_do_not_flag(self) -> None:
        """Specificity control, measured 0/10 at n=2000: same null-effect DGP
        with ``disease_severity`` OBSERVED — FCI separates T from Y given the
        observed confounders, so there is no estimand mark and no warning."""
        result = await _build_dag(
            _make_frame(2000, 1, treatment_effect=0.0),
            TRUE_CONFOUNDERS,
            bootstrap_resamples=0,
        )
        payload = result["latent_diagnostic"]
        assert payload is not None and payload["ran"] and payload["converged"]
        assert payload["flag"] is False
        assert not any("Latent-confounding diagnostic" in w for w in result["warnings"])

    @pytest.mark.asyncio
    async def test_covariate_level_bidirected_pairs_do_not_flag(self) -> None:
        """Noise control with teeth: at n=2000 seed 7 FCI marks a COVARIATE
        pair bidirected (persistent_180d <-> prognostic_only, measured) — the
        payload must report it (positive control that the diagnostic saw it)
        while the flag stays down: covariate-level marks false-alarm on every
        measured control, so only the estimand pair may raise the flag."""
        result = await _build_dag(_make_noise_frame(2000, 7), [], bootstrap_resamples=0)
        payload = result["latent_diagnostic"]
        assert payload is not None and payload["ran"] and payload["converged"]
        assert payload["bidirected_edges"], "expected the measured covariate-level mark"
        assert not any({u, v} == {TREATMENT, OUTCOME} for u, v in payload["bidirected_edges"])
        assert payload["flag"] is False
        assert not any("Latent-confounding diagnostic" in w for w in result["warnings"])
