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
    LLMEvaluatorAudit,
    LLMVerdict,
)

# --- Per-provider pricing (USD per million tokens) ---------------------------
# Mirrors the documented ``HAIKU_*_USD_PER_MTOK`` constants in
# ``src/data/causal_role_evaluator.py`` (#241 telemetry). These are list-price
# ESTIMATES for the ensemble's three members and the single place to update on
# a price change; ``_cost_for`` is the only consumer. Per-provider so the
# ensemble can report cost for Sonnet / Opus / GPT-5 independently (#242 AC4).
SONNET_INPUT_USD_PER_MTOK = 3.00
SONNET_OUTPUT_USD_PER_MTOK = 15.00
OPUS_INPUT_USD_PER_MTOK = 15.00
OPUS_OUTPUT_USD_PER_MTOK = 75.00
GPT5_INPUT_USD_PER_MTOK = 1.25
GPT5_OUTPUT_USD_PER_MTOK = 10.00


def _rates_for_model(model: str) -> Optional[tuple[float, float]]:
    """Map a provider-prefixed model string to its ``(input, output)`` per-MTok
    rates, or ``None`` for a model the ensemble does not price.

    Matched by substring so minor version suffixes (``-4-6``, dated variants)
    resolve to the same family. ``opus`` is checked before ``sonnet`` only for
    clarity; the two never co-occur in one string.
    """
    m = model.lower()
    if "opus" in m:
        return (OPUS_INPUT_USD_PER_MTOK, OPUS_OUTPUT_USD_PER_MTOK)
    if "sonnet" in m:
        return (SONNET_INPUT_USD_PER_MTOK, SONNET_OUTPUT_USD_PER_MTOK)
    if "gpt-5" in m or "gpt5" in m:
        return (GPT5_INPUT_USD_PER_MTOK, GPT5_OUTPUT_USD_PER_MTOK)
    return None


def _cost_for(
    model: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
) -> Optional[float]:
    """Cost in USD for one model call, from its per-MTok rates.

    ``None`` when token counts were not surfaced (cache hit / missing usage)
    or when the model family is not priced — never guesses a cost.
    """
    if input_tokens is None or output_tokens is None:
        return None
    rates = _rates_for_model(model)
    if rates is None:
        return None
    input_rate, output_rate = rates
    return input_tokens / 1e6 * input_rate + output_tokens / 1e6 * output_rate


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


def _model_basename(model: str) -> str:
    """Drop the provider prefix for compact audit labels (``openai/gpt-5`` →
    ``gpt-5``)."""
    return model.split("/")[-1]


def _ensemble_to_llm_verdict(clf: EnsembleClassification) -> Optional[LLMVerdict]:
    """Adapt a fused :class:`EnsembleClassification` to the ``LLMVerdict`` shape
    the existing ``EnsembleVoter`` (and the #240 severity-gate) already consume
    — so the ensemble plugs in with ZERO voter changes (#242 AC3).

    The ensemble's agreement-state is packaged as the ``LLMEvaluatorAudit``
    sidecar the gate reads (``evaluate_r1`` consumes ``satisfied`` +
    ``missed_considerations``): ``full`` => ``satisfied=True`` (the gate may
    trust the multi-vendor verdict); ``majority`` => ``satisfied=False`` with
    the dissenting model(s) listed in ``missed_considerations``.

    ``split`` (``fused_role is None``) returns ``None`` — no confident verdict,
    so the voter abstains / escalates to review (``unknown``), matching the
    single-model ``classify_feature`` "no confident verdict → None" contract.
    """
    if clf.fused_role is None:
        return None

    members = "+".join(_model_basename(v.model) for v in clf.votes)
    evaluator_model = f"ensemble:{members}"
    satisfied = clf.agreement == "full"

    if satisfied:
        missed: tuple[str, ...] = ()
        notes = (
            f"ensemble full agreement ({clf.healthy_votes}/{len(clf.votes)} models) "
            f"on {clf.fused_role}"
        )
    else:
        # majority: surface each dissenting HEALTHY vote as "<model>:<role>"
        # (<=5 items, each <=80 chars per the LLMEvaluatorAudit contract).
        missed = tuple(
            f"{_model_basename(v.model)}:{v.causal_role}"[:80]
            for v in clf.votes
            if v.causal_role is not None and v.causal_role != clf.fused_role
        )[:5]
        notes = (
            f"ensemble majority ({clf.healthy_votes} healthy) on {clf.fused_role}; "
            f"dissent: {', '.join(missed) or 'none'}"
        )

    in_toks = [v.input_tokens for v in clf.votes if v.input_tokens is not None]
    out_toks = [v.output_tokens for v in clf.votes if v.output_tokens is not None]

    audit = LLMEvaluatorAudit(
        satisfied=satisfied,
        rationale_complete=satisfied,
        missed_considerations=missed,
        notes=notes[:500],
        evaluator_model=evaluator_model,
        latency_ms=clf.max_latency_ms,
        input_tokens=sum(in_toks) if in_toks else None,
        output_tokens=sum(out_toks) if out_toks else None,
        cost_usd=clf.total_cost_usd,
    )

    return LLMVerdict(
        causal_role=clf.fused_role,
        mechanism=clf.fused_mechanism,
        # Match the single-model loader's convention for a missing remediation.
        recommended_remediation=clf.fused_remediation or "keep_with_caveat",
        evaluator_audit=audit,
    )
