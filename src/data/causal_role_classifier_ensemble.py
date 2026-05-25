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

import logging
import os
import time
import typing
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, Optional, Sequence

import dspy

from src.data.causal_role_classifier import CausalRoleClassifier

# Single source of truth for the #241 dual OpenAI/Anthropic usage-shape
# extractor (reused, not reimplemented, so a future provider-shape change
# reaches both the single-model loader and this ensemble with one diff) and
# the compiled-artifact loader (all 3 ensemble members share the same compiled
# few-shot demos).
from src.data.causal_role_classifier_loader import (
    _PROVIDER_TO_ENV_VARS,
    _env_value_is_usable,
    _extract_lm_usage,
    _model_provider,
    load_compiled_classifier,
)
from src.data.kg.types import (
    CausalRole,
    EnsembleAgreement,
    EnsembleClassification,
    EnsembleModelVote,
    LLMEvaluatorAudit,
    LLMVerdict,
    Remediation,
)

logger = logging.getLogger(__name__)

# Default ensemble members (#242: Sonnet 4.6 + Opus 4.7 + GPT-5). Provider-
# prefixed litellm/DSPy form. Overridable per-deploy via env (read at CALL
# time, never import time) so model drift is a config change, not a code edit.
_DEFAULT_SONNET = "anthropic/claude-sonnet-4-6"
_DEFAULT_OPUS = "anthropic/claude-opus-4-7"
_DEFAULT_GPT = "openai/gpt-5"

_VALID_ROLES = frozenset(typing.get_args(CausalRole))
_VALID_REMEDIATIONS = frozenset(typing.get_args(Remediation))


def _coerce_role(value: object) -> Optional[CausalRole]:
    """Return ``value`` iff it is in the ``CausalRole`` vocabulary, else None.

    Reads the Literal via ``typing.get_args`` so the vocabulary has one
    definition (``types.py``) and cannot drift.
    """
    return value if value in _VALID_ROLES else None  # type: ignore[return-value]


def _coerce_remediation(value: object) -> Optional[Remediation]:
    return value if value in _VALID_REMEDIATIONS else None  # type: ignore[return-value]


def _resolve_models() -> tuple[str, str, str]:
    """The three ensemble member model strings, from env with defaults."""
    return (
        os.environ.get("ENSEMBLE_SONNET_MODEL", _DEFAULT_SONNET),
        os.environ.get("ENSEMBLE_OPUS_MODEL", _DEFAULT_OPUS),
        os.environ.get("ENSEMBLE_GPT_MODEL", _DEFAULT_GPT),
    )


class EnsemblePreflightError(RuntimeError):
    """Raised when an ensemble member's provider API key is absent.

    A *missing key* is a configuration error, NOT a runtime outage. We fail
    loudly here rather than letting that member error at call time and degrade
    to a non-vote — silently running a 2-of-3 "ensemble" of two Anthropic
    siblings would throw away the independent-provider property that is the
    entire reason #242 exists (see module docstring + #240 AC3.5).
    """


def _preflight_models(models: Sequence[str]) -> None:
    """Verify every member is addressable AND its provider key is usable; raise
    listing every problem. Reuses the loader's provider→env-var map and its
    non-whitespace key guard so the two stay in sync.

    A member fails preflight when it has no recognised provider prefix (a bare
    name or a typo like ``opnai/gpt-5`` could never authenticate) OR when none
    of its provider's keys are set to a non-empty/non-whitespace value. Both
    would otherwise pass to call time and surface as a silent runtime non-vote,
    collapsing the ensemble to fewer vendors — the exact failure preflight
    exists to prevent.
    """
    problems: list[str] = []
    for model in models:
        provider = _model_provider(model)
        if provider is None or provider not in _PROVIDER_TO_ENV_VARS:
            problems.append(
                f"{model} has no recognised provider prefix "
                f"(expected one of {sorted(_PROVIDER_TO_ENV_VARS)})"
            )
            continue
        env_vars = _PROVIDER_TO_ENV_VARS[provider]
        if not any(_env_value_is_usable(var) for var in env_vars):
            problems.append(f"{model} needs one of {list(env_vars)} set (non-empty)")
    if problems:
        raise EnsemblePreflightError("ensemble preflight failed — " + "; ".join(problems))


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


# --- Per-model execution (the only code that touches an LM) ------------------
# ``_make_lm`` and ``_predict_under_lm`` are deliberately tiny indirections so
# unit tests can stub them and exercise _classify_one / the orchestration with
# NO live API call. Production wires them to real DSPy.


def _make_lm(model: str) -> Any:
    """Construct a per-model DSPy LM. Construction is lazy (no API call until
    the LM is invoked), so a missing key surfaces at call time, not here."""
    return dspy.LM(model=model)


def _predict_under_lm(
    classifier: Any,
    lm: Any,
    *,
    feature_name: str,
    derivation_pseudocode: str,
    dataset_context: str,
) -> Any:
    """Run the shared classifier under ``lm`` via the established per-call
    ``dspy.settings.context`` override (mirrors the loader's evaluator path)."""
    with dspy.settings.context(lm=lm):
        return classifier(
            feature_name=feature_name,
            derivation_pseudocode=derivation_pseudocode,
            dataset_context=dataset_context,
        )


def _classify_one(
    model: str,
    *,
    feature_name: str,
    derivation_pseudocode: str,
    dataset_context: str,
    classifier: Any,
    prompt_mode: Literal["compiled", "zeroshot"] = "compiled",
) -> EnsembleModelVote:
    """Run ONE model over the feature, returning a healthy vote or — on any
    failure / invalid role — a NON-vote (``causal_role=None`` + ``error``).

    A non-vote is degrade-to-healthy fuel for :func:`_fuse_votes`, never a
    raise: one provider's outage must not sink the ensemble. Timing is recorded
    even on the failure path (operators can see how long a rate-limited call
    took). Telemetry mirrors the #241 evaluator path.

    ``prompt_mode="zeroshot"`` runs a FRESH (uncompiled) CausalRoleClassifier
    with no few-shot demos instead of the shared compiled artifact, so each
    vendor reasons from the bare signature only (#242 de-confound ablation).
    ``prompt_mode="compiled"`` (default) preserves today's behaviour exactly.
    """
    # De-confound: in zeroshot mode, each model gets a fresh uncompiled
    # CausalRoleClassifier so no Sonnet-optimised demos are injected into
    # Opus or GPT-5.  This is the cheap, valid ablation — no per-vendor
    # compile step required.
    #
    # TODO: per-vendor compiled artifacts (Opus-compiled + GPT-5-compiled)
    # would be a stronger ablation but require separate compile runs and
    # spend.  Leave as a documented future step.
    effective_classifier: Any
    if prompt_mode == "zeroshot":
        effective_classifier = CausalRoleClassifier()
    else:
        effective_classifier = classifier

    start = time.perf_counter()
    try:
        lm = _make_lm(model)
        prediction = _predict_under_lm(
            effective_classifier,
            lm,
            feature_name=feature_name,
            derivation_pseudocode=derivation_pseudocode,
            dataset_context=dataset_context,
        )
    except Exception as exc:  # noqa: BLE001 — best-effort: any failure = non-vote
        latency_ms = (time.perf_counter() - start) * 1000.0
        logger.warning("ensemble: model=%s raised: %s — recording non-vote.", model, exc)
        # Keep the FULL provider message: the A/B harness inspects vote.error for
        # credit/quota exhaustion to stop cleanly, and the matchable phrase often
        # sits >80 chars into a litellm error (e.g. Anthropic's "credit balance is
        # too low" lands ~char 118). Truncating here hides it and silently defeats
        # the graceful stop — the run then pollutes every remaining row.
        return EnsembleModelVote(
            model=model,
            causal_role=None,
            latency_ms=latency_ms,
            error=(str(exc) or type(exc).__name__),
        )

    latency_ms = (time.perf_counter() - start) * 1000.0
    role = _coerce_role(getattr(prediction, "causal_role", None))
    if role is None:
        logger.warning(
            "ensemble: model=%s returned causal_role=%r outside vocabulary — non-vote.",
            model,
            getattr(prediction, "causal_role", None),
        )
        return EnsembleModelVote(
            model=model, causal_role=None, latency_ms=latency_ms, error="invalid_role"
        )

    remediation = _coerce_remediation(getattr(prediction, "recommended_remediation", None))
    raw_mechanism = getattr(prediction, "mechanism", "")
    mechanism = raw_mechanism if isinstance(raw_mechanism, str) else ""
    input_tokens, output_tokens = _extract_lm_usage(lm)
    cost_usd = _cost_for(model, input_tokens, output_tokens)
    return EnsembleModelVote(
        model=model,
        causal_role=role,
        mechanism=mechanism,
        recommended_remediation=remediation,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        error=None,
    )


def run_ensemble_classification(
    *,
    feature_name: str,
    derivation_pseudocode: str,
    dataset_context: str,
    models: Optional[Sequence[str]] = None,
    classifier: Any = None,
    max_workers: int = 3,
    preflight: bool = True,
    prompt_mode: Literal["compiled", "zeroshot"] = "compiled",
) -> EnsembleClassification:
    """Run all members in parallel and fuse. Returns the rich
    :class:`EnsembleClassification` (per-provider votes + telemetry) for the
    offline harness / curation consumers.

    ``preflight`` (default True) checks every member's provider key up front and
    raises :class:`EnsemblePreflightError` if one is missing — so a config gap
    fails loudly instead of silently collapsing the ensemble to one vendor.

    ``prompt_mode`` controls which classifier each member uses:

    * ``"compiled"`` (default) — shared compiled artifact (Sonnet-optimised few-shot
      demos); preserves today's production behaviour exactly.
    * ``"zeroshot"`` — each member gets a fresh :class:`CausalRoleClassifier` with
      NO demos so each vendor reasons from the bare signature only.  This is the
      #242 de-confound ablation: Sonnet-bias correlation eliminated.
    """
    model_tuple = tuple(models) if models is not None else _resolve_models()
    if preflight:
        _preflight_models(model_tuple)
    if classifier is None:
        classifier = load_compiled_classifier()
        if classifier is None:
            logger.warning(
                "ensemble: no compiled classifier artifact found — falling back "
                "to an uncompiled CausalRoleClassifier() for all members."
            )
            classifier = CausalRoleClassifier()

    workers = max(1, min(max_workers, len(model_tuple)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_by_model = {
            executor.submit(
                _classify_one,
                model,
                feature_name=feature_name,
                derivation_pseudocode=derivation_pseudocode,
                dataset_context=dataset_context,
                classifier=classifier,
                prompt_mode=prompt_mode,
            ): model
            for model in model_tuple
        }
        votes_by_model = {future_by_model[f]: f.result() for f in future_by_model}

    votes = tuple(votes_by_model[m] for m in model_tuple)  # stable member order
    return _fuse_votes(feature_name, votes)


def classify_feature_ensemble(
    *,
    feature_name: str,
    derivation_pseudocode: str,
    dataset_context: str,
    models: Optional[Sequence[str]] = None,
    classifier: Any = None,
    max_workers: int = 3,
    preflight: bool = True,
    prompt_mode: Literal["compiled", "zeroshot"] = "compiled",
) -> Optional[LLMVerdict]:
    """Public entry mirroring the single-model ``classify_feature`` contract.

    Returns an ``LLMVerdict`` (with the ensemble agreement packaged as its
    ``evaluator_audit`` sidecar) on ``full``/``majority``; ``None`` on ``split``
    (the voter then abstains / escalates). The richer per-provider breakdown is
    available via :func:`run_ensemble_classification`.

    ``prompt_mode`` is forwarded to :func:`run_ensemble_classification`; see
    that function's docstring for details.
    """
    classification = run_ensemble_classification(
        feature_name=feature_name,
        derivation_pseudocode=derivation_pseudocode,
        dataset_context=dataset_context,
        models=models,
        classifier=classifier,
        max_workers=max_workers,
        preflight=preflight,
        prompt_mode=prompt_mode,
    )
    return _ensemble_to_llm_verdict(classification)
