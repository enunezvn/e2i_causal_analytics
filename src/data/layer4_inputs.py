"""Layer-4 input assembly and citation-verdict serialisation.

Extracted from ``adaptive_validity_check`` (Lane E, 2026-09-22) under the
module-size ratchet: these are pure helpers with no node state. The node
imports them under its private names (``_build_layer_4_inputs``,
``_citation_verdict_to_dict``) so its call sites and tests are unchanged.
No ``dspy`` import here — the node keeps DSPy off its import-time surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from src.data.feature_contract import FeatureContract
    from src.data.kg.types import CitationVerdict


def build_layer_4_inputs(
    feature: str,
    contract: Optional["FeatureContract"],
    target: str,
    manifest_source: Optional[str],
    treatment: Optional[str] = None,
) -> tuple[str, str]:
    """Build the (derivation_pseudocode, dataset_context) pair for Layer 4.

    ``treatment`` (Lane E, real-data causal estimation): the causal treatment
    column when the node runs over a CAUSAL frame via
    ``src.causal_engine.feature_role_panel`` (read from
    ``scope_spec["causal_treatment"]``). The classifier's roles (confounder /
    instrument / mediator / collider) are defined relative to a treatment, so
    the dataset_context names T and the causal question. ``None`` — the
    prediction path — leaves the context byte-identical to before.

    The compiled :class:`src.data.causal_role_classifier.CausalRoleClassifier`
    expects three input fields: ``feature_name`` (provided by caller),
    ``derivation_pseudocode``, and ``dataset_context``. This helper assembles
    the latter two from the feature's manifest contract (when available) and
    the scope_spec target metadata.

    When ``contract`` is None (feature has no manifest entry — e.g. a numeric
    column the runner pre-cleaned), the derivation pseudocode falls back to a
    "no manifest contract on file" sentinel string. The LLM is still able to
    classify based on the feature name alone, but with reduced confidence;
    the audit trail records the absence so an operator can extend the
    manifest later.
    """
    if contract is not None:
        derivation = (
            f"source={contract.source}; "
            f"derivation_inputs={list(contract.derivation_inputs)}; "
            f"aggregation={contract.aggregation}; "
            f"window_days={contract.window_days}; "
            f"knowable_at={contract.knowable_at}"
        )
    else:
        derivation = (
            f"No manifest contract on file for {feature!r} "
            "(LLM is classifying from feature name + dataset context only)"
        )

    cohort = manifest_source or "unspecified"
    dataset_context = f"cohort={cohort}; target={target}; prediction_anchor=index_date"
    if treatment:
        dataset_context += (
            f"; treatment={treatment}; causal_question=effect of {treatment} on {target}"
        )
    return derivation, dataset_context


def citation_verdict_to_dict(verdict: "CitationVerdict", *, verified: bool) -> dict[str, Any]:
    """Serialise one ``CitationVerdict`` for the legacy dict / sidecar (JSON-safe)."""
    return {
        "identifier": verdict.identifier,
        "identifier_kind": verdict.identifier_kind,
        "verified": bool(verified),
        "abstract_resolved": bool(verdict.abstract_resolved),
        "entities_found": list(verdict.entities_found or ()),
        "causal_cue_found": verdict.causal_cue_found,
        "overall_confidence": float(verdict.overall_confidence),
        "error": verdict.error,
    }
