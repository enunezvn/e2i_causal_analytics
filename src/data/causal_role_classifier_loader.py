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

import logging
import os
import re
from pathlib import Path
from typing import Optional

import dspy

from src.data.causal_role_classifier import CausalRoleClassifier
from src.data.kg.types import CausalRole, LLMVerdict, Remediation

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

    mechanism = getattr(prediction, "mechanism", None) or ""
    return LLMVerdict(
        causal_role=role,
        mechanism=mechanism,
        recommended_remediation=remediation,
        cited_pmids=_extract_pmids(mechanism),
    )


def is_lm_configured_for_classification() -> bool:
    """Public wrapper for tests / callers that want to log whether Layer 4
    will fire on the current run.

    Returns True iff (a) a DSPy LM is registered AND (b) the compiled
    artifact is loadable. False otherwise — Layer 4 will silently skip.
    """
    if not _lm_is_configured():
        return False
    return DEFAULT_ARTIFACT_PATH.exists() or "DSPY_CLASSIFIER_ARTIFACT" in os.environ
