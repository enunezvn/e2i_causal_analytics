"""Layer-4 multi-model worker ensemble (Issue #242).

Runs the existing single-model :class:`CausalRoleClassifier` (a DSPy module)
under THREE independent LMs — Sonnet 4.6 + Opus 4.7 + GPT-5 — over the same
feature, then fuses the three opinions with **agreement-or-escalate** logic:

* 3-of-3 agree on ``causal_role``     → ``full``     (auto-verdict)
* 2-of-3 agree, 3rd dissents          → ``majority``  (majority verdict + split audit)
* all disagree, OR <=1 healthy vote   → ``split``     (escalate / ``unknown``)

A model that errored/timed out is a **non-vote** (degrade-to-healthy), not a
disagreement: if the 2 surviving models agree the verdict is still ``full``.

**Why a multi-MODEL ensemble (vs. the existing multi-SIGNAL EnsembleVoter).**
The #240 severity-gate's trust input historically came from a single Haiku
evaluator — an Anthropic sibling of the Sonnet worker, so their failures
correlate. The ensemble's agreement-state is an *independent-provider* signal
(GPT-5 is non-Anthropic): ``full`` => safe to trust; ``majority``/``split`` =>
the vendors disagree, which is exactly the asymmetric-failure-mode signal the
gate needs. See ``docs/plans/242-multi-model-ensemble.md`` and #240 AC3.5.

**Offline-first.** This module is wired into the offline precision harness
(``scripts/measure_layer4_precision.py --ensemble``) and the curation surfacing
only. The live ``adaptive_validity_check`` node stays single-Sonnet — wiring a
3x LLM call into a live LangGraph path is deferred to a #240 follow-up so we do
not triple cost or feed unvalidated fused roles into production before the gate
ACs (FP-rate, kappa, sign-off) are measured.

The agreement logic (:func:`_fuse_votes`) is a PURE function with no I/O so it
is exhaustively unit-tested without an LM.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence

from src.data.kg.types import (
    CausalRole,
    EnsembleAgreement,
    EnsembleClassification,
    EnsembleModelVote,
)


def _aggregate_telemetry(
    votes: Sequence[EnsembleModelVote],
) -> tuple[Optional[float], Optional[float]]:
    """Sum per-provider cost and take the slowest latency.

    The ensemble runs the models in parallel, so wall-time ~= the slowest
    model; total spend is the sum across providers. ``None`` when no vote
    surfaced the corresponding telemetry (e.g. all models errored, or usage
    blocks were absent).
    """
    costs = [v.cost_usd for v in votes if v.cost_usd is not None]
    latencies = [v.latency_ms for v in votes if v.latency_ms is not None]
    total_cost = sum(costs) if costs else None
    max_latency = max(latencies) if latencies else None
    return total_cost, max_latency


def _fuse_votes(
    feature_name: str,
    votes: Sequence[EnsembleModelVote],
) -> EnsembleClassification:
    """Fuse per-model votes into a single :class:`EnsembleClassification`.

    Pure function — agreement is computed over ``causal_role`` only. Tie policy:
    a *strict* majority is required (top role count * 2 > number of healthy
    votes); any tie (e.g. 1-1 among two healthy votes) is a ``split``.
    """
    votes = tuple(votes)
    healthy = [v for v in votes if v.causal_role is not None]
    n_healthy = len(healthy)
    total_cost, max_latency = _aggregate_telemetry(votes)

    agreement: EnsembleAgreement
    fused_role: Optional[CausalRole]

    # <=1 healthy vote: not enough independent signal to trust — escalate.
    if n_healthy <= 1:
        agreement = "split"
        fused_role = None
    else:
        counts: Counter[Optional[CausalRole]] = Counter(v.causal_role for v in healthy)
        top_role, top_count = counts.most_common(1)[0]
        if len(counts) == 1:
            agreement = "full"
            fused_role = top_role
        elif top_count * 2 > n_healthy:  # strict majority
            agreement = "majority"
            fused_role = top_role
        else:  # tie / all-distinct: no majority
            agreement = "split"
            fused_role = None

    if fused_role is not None:
        rep = next(v for v in healthy if v.causal_role == fused_role)
        fused_mechanism = rep.mechanism
        fused_remediation = rep.recommended_remediation
    else:
        fused_mechanism = ""
        fused_remediation = None

    return EnsembleClassification(
        feature_name=feature_name,
        agreement=agreement,
        fused_role=fused_role,
        fused_mechanism=fused_mechanism,
        fused_remediation=fused_remediation,
        votes=votes,
        healthy_votes=n_healthy,
        total_cost_usd=total_cost,
        max_latency_ms=max_latency,
    )
