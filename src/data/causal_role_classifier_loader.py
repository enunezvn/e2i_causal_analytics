"""Phase 2.5 — loader + verdict adapter for the compiled Layer 4 classifier.

Wraps :class:`src.data.causal_role_classifier.CausalRoleClassifier` so call
sites can:

1. Lazily load the persisted compiled program from
   ``artifacts/dspy/causal_role_classifier.json`` (or a custom path).
2. Run the classifier on a single feature (returning a typed
   :class:`src.data.kg.types.LLMVerdict` for the Phase 2.7 ``EnsembleVoter``).
3. Fall back gracefully when no LM endpoint is configured *or* the persisted
   artifact is missing — returns ``None`` so the caller emits a non-LLM
   verdict instead of raising.

Decision policy at call sites: any caller that wants the LLM-verdict path
asks :func:`classify_feature` for an ``LLMVerdict``. ``None`` means the LLM
path is unavailable for this run — the caller proceeds with whatever Layer 1
/ Layer 3 / KG signals it has, falling back to the non-LLM bypass paths in
``adaptive_validity_check._compose_legacy_verdict``.

Why a loader rather than instantiating ``CausalRoleClassifier`` directly:
the persisted JSON contains the BootstrapFewShot-curated few-shot demos. A
fresh ``CausalRoleClassifier()`` has zero demos, so its predictions are
prompt-driven only (no in-context examples). Loading the compiled program is
what turns Phase 2.5 from "we have the signature" into "we have a teacher-
optimised classifier."
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

import dspy

from src.data.causal_role_classifier import CausalRoleClassifier
from src.data.causal_role_evaluator import (
    CausalRoleEvaluator,
    _evaluator_lm_is_configured,
    compute_haiku_cost_usd,
    evaluator_is_enabled,
    resolve_evaluator_model,
)
from src.data.kg.types import CausalRole, LLMEvaluatorAudit, LLMVerdict, Remediation

logger = logging.getLogger(__name__)

# Project-root anchored default path. Mirrors the default in
# ``scripts/compile_causal_role_classifier.py`` so a default-args compile run
# writes to the same place a default-args load reads from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "dspy" / "causal_role_classifier.json"

# Valid roles + remediations sets — used to sanitise LLM output. The
# Phase 2.7 ``EnsembleVoter`` itself sanitises (see
# ``ensemble_voter.VALID_LLM_ROLES``) but the loader does a first pass so
# downstream consumers can rely on a clean ``LLMVerdict`` shape.
_VALID_ROLES: frozenset[str] = frozenset(
    {"ancestor", "confounder", "instrument", "mediator", "collider", "descendant"}
)
_VALID_REMEDIATIONS: frozenset[str] = frozenset({"drop", "window", "transform", "keep_with_caveat"})


def _lm_is_configured() -> bool:
    """Return True iff a default DSPy LM is currently registered.

    Used by :func:`classify_feature` to skip the call entirely when no LM
    endpoint is configured — the loader silently no-ops in that case so a
    bare pipeline run on a developer laptop without API keys doesn't
    raise. The audit trail still records that the LLM path was not taken
    (caller's responsibility — the loader returns ``None`` so the caller
    sees the absence explicitly).
    """
    try:
        lm = getattr(dspy.settings, "lm", None)
    except Exception:
        return False
    return lm is not None


# Default model. Matches the convention in
# ``src/api/routes/chatbot_dspy.py:63`` and ``src/rag/causal_rag.py:186``.
_DEFAULT_LM_MODEL = "anthropic/claude-sonnet-4-6"

# Provider → env-var mapping. The DSPy / LiteLLM model string carries the
# provider prefix (``anthropic/``, ``openai/``, ``azure/``); we use that
# prefix to gate on the matching env var so a key for the WRONG provider
# does not green-light configuration (codex pass-2 MEDIUM: an env with
# only ``OPENAI_API_KEY`` set must NOT auth-configure an Anthropic model
# and then silently fail every LM call inside classify_feature).
_PROVIDER_TO_ENV_VARS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "azure": ("AZURE_API_KEY", "AZURE_OPENAI_API_KEY"),
}


def _env_value_is_usable(var_name: str) -> bool:
    """Return True iff env var is set and non-whitespace.

    Codex pass-2 MEDIUM: whitespace-only key values previously passed the
    truthy check (``os.environ.get(v)`` returns the string, which is
    truthy if length > 0) and would let ``dspy.LM`` configure with a
    provider-rejected credential.
    """
    raw = os.environ.get(var_name)
    if raw is None:
        return False
    return bool(raw.strip())


def _model_provider(model: str) -> Optional[str]:
    """Extract the LiteLLM provider prefix from a model string.

    Returns the lower-cased provider (``"anthropic"``, ``"openai"``, etc.)
    when the model is ``provider/path`` shaped; returns ``None`` for
    bare model names (which LiteLLM would have to guess at).
    """
    if "/" not in model:
        return None
    return model.split("/", 1)[0].strip().lower()


def ensure_dspy_lm_configured(
    *,
    model: str = _DEFAULT_LM_MODEL,
    require_api_key: bool = True,
) -> bool:
    """Idempotent DSPy LM configuration for the runtime production path.

    Codex pass-1 HIGH-1 (issue #193): the loader previously checked
    ``dspy.settings.lm is None`` and returned ``None`` from
    ``classify_feature`` when no LM was configured — but the
    orchestrator never instantiated one. The production pipeline path
    with ``ANTHROPIC_API_KEY`` set in env therefore silently no-op'd
    every Layer 4 invocation, effectively disabling the Stage 3 wiring.

    This helper bridges the gap: when called at orchestrator entry it
    configures a default DSPy LM (matching the convention used in
    ``src/api/routes/chatbot_dspy.py`` and ``src/rag/causal_rag.py``) iff
    (a) no LM is already configured AND (b) a usable API key is present
    in the environment FOR THE TARGET PROVIDER. Returns ``True`` when an
    LM is configured after the call (either pre-existing or freshly
    configured), ``False`` when no LM is or can be configured.

    Idempotent: a second call with an LM already configured returns
    ``True`` without re-configuring (this matches the chatbot_dspy
    module's ``_dspy_lm_configured`` once-flag pattern but is stateless
    so test code can rely on ``dspy.settings.lm = None`` to force
    re-configuration).

    Codex pass-2 MEDIUM (issue #193): the credential gate is now
    provider-aware. The model string's provider prefix (``anthropic/``,
    ``openai/``) is mapped to the corresponding env var(s); only that
    var(s) are checked. An env with only ``OPENAI_API_KEY`` set will NOT
    green-light configuration of an Anthropic model. Whitespace-only key
    values are rejected (``.strip() == ""``).

    Args:
        model: Default LM model string in LiteLLM-shape ``provider/path``.
            Default matches the documented convention (Anthropic Claude
            Sonnet 4). Unknown providers (model not in
            ``_PROVIDER_TO_ENV_VARS``) fall back to "any recognised key"
            so the helper stays permissive for new providers, with a
            warning logged.
        require_api_key: When ``True`` (default) the configuration step is
            skipped unless a usable provider-matching API key env var
            is present. When ``False``, ``dspy.LM(model)`` is invoked
            unconditionally — useful for tests that supply DummyLM via a
            different mechanism. Tests of this function itself must set
            ``require_api_key=True`` and prepare the env so the branching
            is exercised.

    Returns:
        ``True`` iff a DSPy LM is configured at end of the call. ``False``
        means Layer 4 will skip — either because no usable provider-
        matching key is present (the documented CI / developer-laptop
        path) OR because configuration raised (logged as a warning).
    """
    if _lm_is_configured():
        return True
    if require_api_key:
        provider = _model_provider(model)
        if provider is not None and provider not in _PROVIDER_TO_ENV_VARS:
            # Codex pass-3 MEDIUM (issue #193): typoed provider prefixes
            # (e.g. ``antropic/claude-sonnet-4-6`` missing the
            # ``h``) used to fall back to a permissive any-recognised-
            # key check, which would let an env with only the wrong
            # provider's key green-light an unusable LM. The previous
            # pass-2 fix specifically targeted this class of silent
            # disablement; the typo path was the remaining hole.
            # Fail closed: when the operator wrote a slash-shaped model
            # string with an unrecognised provider, refuse to guess.
            logger.warning(
                "ensure_dspy_lm_configured: model=%r has slash-shaped "
                "provider prefix %r which is not in %s; refusing to "
                "configure (fail-closed against typoed prefix). Set the "
                "model to a recognised provider or update "
                "_PROVIDER_TO_ENV_VARS to add the new provider.",
                model,
                provider,
                sorted(_PROVIDER_TO_ENV_VARS.keys()),
            )
            return False
        expected_vars = _PROVIDER_TO_ENV_VARS.get(provider) if provider else None
        if expected_vars is None:
            # Bare model (no slash): keep the permissive any-key fallback
            # so LiteLLM's auto-provider-detection path still works for
            # dev environments. We only refuse on the typoed-slash path
            # above (where the operator's intent was clearly "use
            # provider X" but X is unknown).
            expected_vars = tuple(v for vs in _PROVIDER_TO_ENV_VARS.values() for v in vs)
            logger.warning(
                "ensure_dspy_lm_configured: model=%r has no provider prefix "
                "(bare model name); falling back to permissive any-key check "
                "over %s. Prefer slash-shaped provider/path strings.",
                model,
                list(expected_vars),
            )
        if not any(_env_value_is_usable(v) for v in expected_vars):
            logger.info(
                "ensure_dspy_lm_configured: no usable provider-matching API key "
                "env var found (model=%s, checked %s) — Layer 4 will skip this run.",
                model,
                list(expected_vars),
            )
            return False
    try:
        lm = dspy.LM(model)
        dspy.configure(lm=lm)
        logger.info(
            "ensure_dspy_lm_configured: configured default DSPy LM with model=%s",
            model,
        )
        return True
    except Exception as exc:
        logger.warning(
            "ensure_dspy_lm_configured: failed to configure DSPy LM (%s); "
            "Layer 4 will skip this run.",
            exc,
        )
        return False


def load_compiled_classifier(
    *,
    artifact_path: Path | None = None,
    strict: bool = False,
) -> Optional[CausalRoleClassifier]:
    """Load the persisted compiled program.

    Args:
        artifact_path: Path to the compiled JSON. Defaults to
            :data:`DEFAULT_ARTIFACT_PATH` (``artifacts/dspy/causal_role_classifier.json``).
        strict: When ``True``, raises ``FileNotFoundError`` if the artifact
            is missing. When ``False`` (default), returns ``None`` and logs
            a warning — the standard "no Layer 4 today" path.

    Returns:
        Loaded :class:`CausalRoleClassifier` with the bootstrapped few-shot
        demos populated, or ``None`` if the artifact is missing and
        ``strict=False``.

    Note: ``dspy.Module.load`` is invoked with ``allow_pickle=False`` so a
    corrupted-or-tampered artifact cannot execute arbitrary code at load
    time (DSPy 3.1's default is also ``False``, but we set it explicitly
    so a future default flip does not silently widen the trust boundary).
    """
    path = Path(artifact_path) if artifact_path is not None else DEFAULT_ARTIFACT_PATH
    if not path.exists():
        msg = f"Compiled classifier artifact not found at {path}"
        if strict:
            raise FileNotFoundError(msg)
        logger.warning(
            "%s — Layer 4 will be skipped for this run. Generate via "
            "`python scripts/compile_causal_role_classifier.py`.",
            msg,
        )
        return None

    classifier = CausalRoleClassifier()
    classifier.load(str(path), allow_pickle=False)
    logger.info("load_compiled_classifier: loaded %s", path)
    return classifier


def _extract_pmids(mechanism: str) -> tuple[str, ...]:
    """Extract PMID-shaped tokens from the LLM's free-text mechanism.

    The compile-set ``mechanism`` strings do not contain citations (Phase 2.5
    deferred citation generation to Phase 2.7's CitationResolver workflow).
    The loader's extractor is a defensive stub for the day a future compile
    set teaches the LLM to cite — returning ``()`` for the current vocabulary
    is the correct behaviour.
    """
    if not mechanism:
        return ()
    # PMIDs are decimal integers (typically 4-9 digits) often introduced with
    # ``PMID:`` / ``pmid:`` / ``[PMID 12345]`` patterns. The conservative
    # regex below requires at least 4 digits to avoid false-positives on
    # short integers that appear in derivation pseudocode.
    matches = re.findall(r"\bPMID[:\s]?\s*(\d{4,9})\b", mechanism, flags=re.IGNORECASE)
    # Dedup while preserving first-occurrence order.
    seen: dict[str, None] = {}
    for m in matches:
        seen.setdefault(m, None)
    return tuple(seen.keys())


def _coerce_role(value: object) -> Optional[CausalRole]:
    if isinstance(value, str) and value in _VALID_ROLES:
        return value  # type: ignore[return-value]
    return None


def _coerce_remediation(value: object) -> Optional[Remediation]:
    if isinstance(value, str) and value in _VALID_REMEDIATIONS:
        return value  # type: ignore[return-value]
    return None


def _build_evaluator() -> Optional[CausalRoleEvaluator]:
    """Construct a CausalRoleEvaluator when the operator has opted in.

    Returns ``None`` when the evaluator is disabled or Haiku is
    unconfigured. The returned evaluator carries no LM binding; the LM
    is set per-call via ``dspy.settings.context`` inside
    :func:`_run_evaluator`.

    Plan: ``.claude/plans/layer4_evaluator_audit_signal.md``.
    """
    if not evaluator_is_enabled():
        return None
    if not _evaluator_lm_is_configured():
        # INFO-level so operators see the explicit "I enabled this but
        # nothing is happening" cause. Mirrors the loader's missing-key
        # diagnostic pattern.
        logger.info(
            "_build_evaluator: ADAPTIVE_VALIDITY_EVALUATOR_ENABLED=1 but "
            "ANTHROPIC_API_KEY missing — evaluator skipped. Set the "
            "Anthropic key to enable the Layer-4 audit evaluator."
        )
        return None
    try:
        return CausalRoleEvaluator()
    except Exception as exc:
        logger.warning(
            "_build_evaluator: CausalRoleEvaluator construction raised "
            "%s: %s — evaluator skipped, worker verdict preserved.",
            type(exc).__name__,
            exc,
        )
        return None


def _run_evaluator(
    evaluator: CausalRoleEvaluator,
    *,
    feature_name: str,
    derivation_pseudocode: str,
    dataset_context: str,
    worker_verdict: LLMVerdict,
) -> Optional[LLMEvaluatorAudit]:
    """Call the evaluator inside a Haiku LM context. Returns None on failure.

    The evaluator may raise on rate-limits, malformed outputs, or
    transient network errors. In all cases we log and return None so
    the worker's verdict is preserved.

    Issue #241: captures per-call telemetry — wall-clock latency,
    Haiku ``usage.prompt_tokens`` / ``usage.completion_tokens`` pulled
    from the DSPy LM ``.history`` after the call, and a computed
    USD cost (using the constants in ``causal_role_evaluator``). The
    telemetry is attached to the returned ``LLMEvaluatorAudit``. When
    usage extraction fails (empty history, missing keys), latency is
    still recorded but the token / cost fields are ``None`` —
    partial-telemetry is better than dropping the audit.

    The WARNING log on exceptions includes the timing so operators can
    see how long a rate-limited / failed call took before the worker
    verdict falls through.
    """
    model = resolve_evaluator_model()
    evaluator_lm = None
    start = time.perf_counter()
    try:
        evaluator_lm = dspy.LM(model=model)
        with dspy.settings.context(lm=evaluator_lm):
            audit = evaluator.evaluate(
                feature_name=feature_name,
                derivation_pseudocode=derivation_pseudocode,
                dataset_context=dataset_context,
                worker_verdict=worker_verdict,
                evaluator_model=model,
            )
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        logger.warning(
            "Layer-4 evaluator raised for feature=%s after latency_ms=%.2f: %s — "
            "returning verdict with evaluator_audit=None.",
            feature_name,
            latency_ms,
            exc,
        )
        return None

    latency_ms = (time.perf_counter() - start) * 1000.0
    input_tokens, output_tokens = _extract_lm_usage(evaluator_lm)
    cost_usd: Optional[float]
    if input_tokens is None and output_tokens is None:
        cost_usd = None
    else:
        cost_usd = compute_haiku_cost_usd(input_tokens=input_tokens, output_tokens=output_tokens)
    return dataclasses.replace(
        audit,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
    )


def _extract_lm_usage(
    evaluator_lm: object,
) -> tuple[Optional[int], Optional[int]]:
    """Return ``(input_tokens, output_tokens)`` from the most recent
    LM call recorded on ``evaluator_lm.history``.

    Issue #241. Defensive: returns ``(None, None)`` when the LM has no
    history, when the latest entry has no ``usage`` block, or when
    the usage block has neither OpenAI-style (``prompt_tokens`` /
    ``completion_tokens``) nor Anthropic-native (``input_tokens`` /
    ``output_tokens``) keys.

    The DSPy LM stores each call as
    ``{"usage": dict(response.usage), ...}`` in ``self.history``. For
    Anthropic via litellm the usage block is the OpenAI shape
    (litellm normalizes the response); we accept both shapes so a
    future provider change doesn't silently drop telemetry.
    """
    history = getattr(evaluator_lm, "history", None)
    if not history:
        return (None, None)
    try:
        usage = history[-1].get("usage")
    except (AttributeError, IndexError, TypeError):
        return (None, None)
    if not isinstance(usage, dict):
        return (None, None)
    # Accept OpenAI shape first (litellm normalizes Anthropic to this).
    in_t = usage.get("prompt_tokens")
    out_t = usage.get("completion_tokens")
    # Fall back to Anthropic-native field names if the OpenAI shape is
    # absent (e.g. provider drift, direct Anthropic SDK in future).
    if in_t is None:
        in_t = usage.get("input_tokens")
    if out_t is None:
        out_t = usage.get("output_tokens")
    in_t = int(in_t) if isinstance(in_t, (int, float)) and not isinstance(in_t, bool) else None
    out_t = int(out_t) if isinstance(out_t, (int, float)) and not isinstance(out_t, bool) else None
    return (in_t, out_t)


def classify_feature(
    *,
    feature_name: str,
    derivation_pseudocode: str,
    dataset_context: str,
    classifier: Optional[CausalRoleClassifier] = None,
) -> Optional[LLMVerdict]:
    """Run the compiled classifier on one feature and return an LLMVerdict.

    Args:
        feature_name: Feature being classified.
        derivation_pseudocode: Plain-English or pseudo-code describing how the
            feature is derived (mirrors the compile-set ``Example`` field).
        dataset_context: Target + cohort + prediction-anchor context.
        classifier: Pre-loaded classifier instance. When ``None``, the loader
            calls :func:`load_compiled_classifier` itself with default args
            (so most callers can just pass the three input fields).

    Returns:
        :class:`LLMVerdict` carrying the LLM's role, mechanism, remediation,
        and extracted PMIDs. Returns ``None`` when:

        - No DSPy LM endpoint is configured (the loader silently no-ops so
          developer laptops without API keys don't fail the run).
        - The compiled artifact is missing (the load step warned).
        - The LLM call itself raised (the exception is logged but not
          re-raised — Layer 4 is best-effort: a network blip falls through
          to the non-LLM verdict path).
        - The LLM returned a malformed role (sanitised to ``None`` so the
          voter's downstream sanitiser doesn't see a confident-but-wrong
          role).

    The voter receives ``None`` as "LLM did not run / did not produce a
    valid verdict" and falls through to the non-LLM precedence rules.
    """
    if not _lm_is_configured():
        logger.debug(
            "classify_feature(%s): no DSPy LM configured — returning None.",
            feature_name,
        )
        return None

    if classifier is None:
        classifier = load_compiled_classifier()
        if classifier is None:
            return None

    try:
        prediction = classifier(
            feature_name=feature_name,
            derivation_pseudocode=derivation_pseudocode,
            dataset_context=dataset_context,
        )
    except Exception as exc:
        # Best-effort: log and return None so the caller falls through to
        # the non-LLM path. This is the documented escape hatch for the
        # "LM call rate-limited / transient" case.
        logger.warning(
            "classify_feature(%s) raised: %s — returning None (Layer 4 skipped).",
            feature_name,
            exc,
        )
        return None

    role = _coerce_role(getattr(prediction, "causal_role", None))
    if role is None:
        logger.warning(
            "classify_feature(%s): LLM returned causal_role=%r outside vocabulary; returning None.",
            feature_name,
            getattr(prediction, "causal_role", None),
        )
        return None

    remediation = _coerce_remediation(getattr(prediction, "recommended_remediation", None))
    if remediation is None:
        # The voter accepts a sane default when remediation is missing/invalid;
        # picking "keep_with_caveat" for pre-index roles and "drop" for
        # post-index roles is downstream's job. We pass through what the LLM
        # returned but coerce to a safe-ish default so the dataclass holds.
        logger.warning(
            "classify_feature(%s): LLM returned recommended_remediation=%r outside "
            "vocabulary; coercing to 'keep_with_caveat'.",
            feature_name,
            getattr(prediction, "recommended_remediation", None),
        )
        remediation = "keep_with_caveat"

    # Codex pass-1 MEDIUM-1 (issue #193): coerce ``mechanism`` to ``str``
    # at the loader boundary. A malformed LLM that returns a list/dict for
    # the mechanism field would otherwise propagate into ``_extract_pmids``
    # (which calls ``re.findall``), raising ``TypeError`` outside the
    # ``classify_feature`` try/except in this function. The orchestrator's
    # outer try/except would swallow it as "Layer 4 failed", but direct
    # loader callers (tests, scripts) would see an exception in violation
    # of the documented "best-effort: malformed output → None" contract.
    raw_mechanism = getattr(prediction, "mechanism", None)
    if raw_mechanism is None or raw_mechanism == "":
        mechanism = ""
    elif isinstance(raw_mechanism, str):
        mechanism = raw_mechanism
    else:
        logger.warning(
            "classify_feature(%s): LLM returned non-string mechanism=%r "
            "(type=%s); coercing to empty string.",
            feature_name,
            raw_mechanism,
            type(raw_mechanism).__name__,
        )
        mechanism = ""

    worker_verdict = LLMVerdict(
        causal_role=role,
        mechanism=mechanism,
        recommended_remediation=remediation,
        cited_pmids=_extract_pmids(mechanism),
    )

    evaluator = _build_evaluator()
    if evaluator is None:
        return worker_verdict

    audit = _run_evaluator(
        evaluator,
        feature_name=feature_name,
        derivation_pseudocode=derivation_pseudocode,
        dataset_context=dataset_context,
        worker_verdict=worker_verdict,
    )
    if audit is None:
        return worker_verdict
    return dataclasses.replace(worker_verdict, evaluator_audit=audit)


def is_lm_configured_for_classification() -> bool:
    """Public wrapper for tests / callers that want to log whether Layer 4
    will fire on the current run.

    Returns True iff (a) a DSPy LM is registered AND (b) the compiled
    artifact is loadable. False otherwise — Layer 4 will silently skip.
    """
    if not _lm_is_configured():
        return False
    return DEFAULT_ARTIFACT_PATH.exists() or "DSPY_CLASSIFIER_ARTIFACT" in os.environ
