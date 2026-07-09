"""Predictive-cohort strategic insight: who to target and why (cohort + SHAP)."""

from __future__ import annotations

import logging
from typing import Any

from src.insights.common import normalize_list, run_signature

logger = logging.getLogger(__name__)

try:
    import dspy

    class PredictiveCohortInsightSignature(dspy.Signature):
        """Turn a scored out-of-sample cohort into a targeting read for a commercial
        analyst, STRICTLY grounded in the provided numbers. Use ONLY the score
        distribution, named top targets, and SHAP driver importances given; NEVER
        invent entities, probabilities, or features. Refer to the scored rows by
        their stated kind (patients or prescribers), not as "entities". Say who
        to prioritise, what drives their scores, and how confident the ranking
        is. If drivers_summary says drivers were not computed at cohort level,
        do NOT claim the model lacks feature importances — state that per-target
        SHAP contributions are available in the drill-down. The registry
        context lists curated directional chains around the predicted outcome —
        SEPARATE domain knowledge, not model output: you may use it
        qualitatively to frame why targeting matters, but NEVER present it as
        a prediction driver and NEVER attribute numbers to it."""

        model_version: str = dspy.InputField(desc="Scoring model version")
        distribution_summary: str = dspy.InputField(
            desc="n scored (with entity kind), mean probability, predicted outcome"
        )
        top_targets_summary: str = dspy.InputField(desc="Top-ranked targets with probabilities")
        drivers_summary: str = dspy.InputField(
            desc="Cohort-level SHAP drivers (mean |SHAP| over the top targets) + importances"
        )
        registry_context: str = dspy.InputField(
            desc="Curated registry chains around the predicted outcome (directional, no figures)"
        )

        interpretation: str = dspy.OutputField(desc="Targeting read grounded in the numbers")
        key_takeaways: list = dspy.OutputField(desc="3-5 grounded targeting takeaways")

    class PredictiveWhatIfInsightSignature(dspy.Signature):
        """Explain ONE hypothetical "what-if" prediction to a commercial analyst in
        plain language, STRICTLY grounded in the provided numbers. The analyst
        hand-built a hypothetical profile (a patient or prescriber that may not
        exist) and the model scored it. Cover, in order: (1) what was asked —
        restate the entered profile briefly; (2) what the score means — the
        predicted probability of the stated outcome for a profile like this,
        compared against the scored cohort's mean when given; (3) what drives it —
        the SHAP contributions given, naming which attributes push the score up
        or down; (4) how to USE it — concrete next steps such as qualifying or
        prioritising real patients/prescribers matching this profile, or testing
        how changing one attribute moves the score. NEVER invent numbers or
        features. NEVER make causal claims: this is a predictive score, so
        changing an input in the form does NOT mean changing that attribute in
        the real world would change the outcome — say so when relevant. Define
        any technical term (e.g. SHAP) in a brief plain-English aside. The
        registry context is separate domain knowledge — qualitative framing
        only, never a prediction driver, never a source of numbers."""

        model_version: str = dspy.InputField(desc="Scoring model version")
        profile_summary: str = dspy.InputField(desc="The hypothetical profile's entered inputs")
        result_summary: str = dspy.InputField(
            desc="Predicted probability + outcome + cohort-mean comparison when available"
        )
        drivers_summary: str = dspy.InputField(desc="Per-row SHAP contributions for this profile")
        registry_context: str = dspy.InputField(
            desc="Curated registry chains around the predicted outcome (directional, no figures)"
        )

        interpretation: str = dspy.OutputField(
            desc="Plain-language read of the what-if result + how to use it"
        )
        key_takeaways: list = dspy.OutputField(desc="3-5 grounded, actionable takeaways")

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    PredictiveCohortInsightSignature = None  # type: ignore[assignment,misc]
    PredictiveWhatIfInsightSignature = None  # type: ignore[assignment,misc]


# Gold-standard model names are ``<cohort>_<brand>_goldstd_lr_v1`` (see
# predictions.build_curated_input_fields). Each cohort maps to the registry
# query terms whose 6-char token prefixes match the relevant chain nodes:
# persistence -> persistent_180d, initiation -> treatment_initiated,
# discontinuation -> discontinued_180d; HCP adoption feeds prescriber-side
# outcomes (intent_to_prescribe chains + new-to-brand volume).
_COHORT_OUTCOME_TERMS: dict[str, tuple[str, ...]] = {
    "hcp_adoption": ("intent to prescribe", "NBRx"),
    "persistence": ("persistence",),
    "initiation": ("initiation",),
    "discontinuation": ("discontinuation",),
}
_BRAND_PROPER = {"remibrutinib": "Remibrutinib", "kisqali": "Kisqali", "fabhalta": "Fabhalta"}

# What the scored rows ARE and what the model predicts, per cohort. The raw
# entity ids (scvpt_*/scvhcp_*) don't say; the model name's cohort prefix does.
_COHORT_ENTITY_KIND: dict[str, str] = {
    "hcp_adoption": "prescribers (HCPs)",
    "persistence": "patients",
    "initiation": "patients",
    "discontinuation": "patients",
}
_COHORT_OUTCOME_LABEL: dict[str, str] = {
    "hcp_adoption": "adopting the brand (intent to prescribe)",
    "persistence": "staying on therapy at 180 days",
    "initiation": "starting treatment",
    "discontinuation": "discontinuing therapy within 180 days",
}


def cohort_facets_for_model(model_version: str) -> tuple[str | None, str | None]:
    """(entity_kind, outcome_label) from a gold-standard model name.

    Unrecognizable names yield ``(None, None)`` — callers fall back to neutral
    wording ("entities", "the targeted outcome") rather than guessing.
    """
    low = str(model_version or "").lower()
    cohort = next((c for c in _COHORT_ENTITY_KIND if low.startswith(c)), None)
    if cohort is None:
        return None, None
    return _COHORT_ENTITY_KIND[cohort], _COHORT_OUTCOME_LABEL[cohort]


def outcome_terms_for_model(model_version: str) -> tuple[str | None, tuple[str, ...]]:
    """(brand, registry query terms) derived from a gold-standard model name.

    Unrecognizable names yield ``(None, ())`` — an honest empty context, never
    a generic commercial fetch pretending relevance to a patient-level model.
    """
    low = str(model_version or "").lower()
    brand = next((proper for key, proper in _BRAND_PROPER.items() if key in low), None)
    terms = next(
        (terms for cohort, terms in _COHORT_OUTCOME_TERMS.items() if low.startswith(cohort)),
        (),
    )
    return brand, terms


def build_grounding(
    model_version: str,
    n_scored: int,
    mean_prob: float,
    top_targets: list[dict[str, Any]],
    top_drivers: list[dict[str, Any]],
    causal_drivers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    entity_kind, outcome_label = cohort_facets_for_model(model_version)
    distribution_summary = (
        f"{n_scored} {entity_kind or 'entities'} scored, mean probability {mean_prob:.3f}"
        + (f" of {outcome_label}" if outcome_label else "")
    )
    top_targets_summary = (
        "; ".join(
            f"{t.get('entity_id')} ({float(t.get('probability') or 0):.2f})"
            for t in top_targets[:5]
        )
        or "none"
    )
    drivers_summary = (
        "; ".join(
            f"{d.get('feature')} ({float(d.get('importance') or 0):.2f})" for d in top_drivers[:5]
        )
        # Honest absence: cohort scoring may skip the driver aggregation, but
        # per-target SHAP always exists in the drill-down — never let the LM
        # read an empty list as "the model has no feature importances".
        or "not computed at cohort level (per-target SHAP contributions are "
        "available in the entity drill-down)"
    )
    # Digit-free registry chains around the predicted outcome: SHAP explains
    # the model's scores; the registry adds directional domain context and the
    # two must never blur (no figures for the LM to launder into importances).
    from src.insights.causal_context import format_driver_names, format_qualitative_context

    drivers = causal_drivers or []
    named = format_driver_names(drivers)
    chips = [
        {"label": "Scored", "value": str(n_scored)},
        {"label": "Mean p", "value": f"{mean_prob:.3f}"},
        {"label": "Top targets", "value": str(min(len(top_targets), 5))},
    ]
    if named:
        chips.append({"label": "Registry chains", "value": str(len(named))})
    return {
        "model_version": model_version,
        "distribution_summary": distribution_summary,
        "top_targets_summary": top_targets_summary,
        "drivers_summary": drivers_summary,
        "registry_context": format_qualitative_context(drivers),
        "has_registry_context": bool(named),
        "grounding": chips,
    }


def _fallback(g: dict[str, Any]) -> dict[str, Any]:
    insight = (
        f"Model {g['model_version']}: {g['distribution_summary']}. "
        f"Highest-probability targets: {g['top_targets_summary']}. "
        f"Main drivers: {g['drivers_summary']}. "
        + (f"{g['registry_context']} " if g.get("has_registry_context") else "")
        + "(Factual summary — LLM interpretation unavailable.)"
    )
    return {
        "insight": insight,
        "key_takeaways": [g["distribution_summary"], f"Drivers: {g['drivers_summary']}"],
        "grounding": g["grounding"],
        "is_fallback": True,
    }


def generate_insight(g: dict[str, Any]) -> dict[str, Any]:
    pred = run_signature(
        PredictiveCohortInsightSignature,
        model_version=g["model_version"],
        distribution_summary=g["distribution_summary"],
        top_targets_summary=g["top_targets_summary"],
        drivers_summary=g["drivers_summary"],
        registry_context=g["registry_context"],
    )
    if pred is None:
        return _fallback(g)
    interpretation = str(getattr(pred, "interpretation", "")).strip()
    if not interpretation:
        return _fallback(g)
    return {
        "insight": interpretation,
        "key_takeaways": normalize_list(getattr(pred, "key_takeaways", [])),
        "grounding": g["grounding"],
        "is_fallback": False,
    }


def build_whatif_grounding(
    model_version: str,
    features: dict[str, Any],
    probability: float,
    confidence: float | None,
    cohort_mean: float | None,
    n_scored: int | None,
    top_drivers: list[dict[str, Any]],
    causal_drivers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Grounding for ONE hypothetical what-if prediction (single scored row)."""
    entity_kind, outcome_label = cohort_facets_for_model(model_version)
    kind = entity_kind or "entity"
    outcome = outcome_label or "the targeted outcome"

    profile_summary = (
        f"hypothetical {kind.rstrip('s') if kind != 'prescribers (HCPs)' else 'prescriber (HCP)'}: "
        + ("; ".join(f"{k}={features[k]}" for k in sorted(features)) or "no inputs entered")
    )
    result_summary = f"predicted probability {probability:.2f} of {outcome}"
    if cohort_mean is not None:
        result_summary += f" vs cohort mean {cohort_mean:.2f}"
        if n_scored:
            result_summary += f" across {n_scored} scored {kind}"
    if confidence is not None:
        result_summary += f"; model confidence {confidence:.2f}"
    drivers_summary = (
        "; ".join(
            f"{d.get('feature')} ({float(d.get('importance') or 0):+.2f})" for d in top_drivers[:8]
        )
        or "not returned for this row (SHAP unavailable)"
    )

    from src.insights.causal_context import format_driver_names, format_qualitative_context

    drivers = causal_drivers or []
    named = format_driver_names(drivers)
    chips = [
        {"label": "Predicted p", "value": f"{probability:.2f}"},
        {"label": "Inputs", "value": str(len(features))},
    ]
    if cohort_mean is not None:
        chips.append({"label": "Cohort mean", "value": f"{cohort_mean:.2f}"})
    if named:
        chips.append({"label": "Registry chains", "value": str(len(named))})
    return {
        "model_version": model_version,
        "profile_summary": profile_summary,
        "result_summary": result_summary,
        "drivers_summary": drivers_summary,
        "registry_context": format_qualitative_context(drivers),
        "has_registry_context": bool(named),
        "grounding": chips,
    }


def _whatif_fallback(g: dict[str, Any]) -> dict[str, Any]:
    insight = (
        f"What-if against model {g['model_version']}: {g['profile_summary']}. "
        f"Result: {g['result_summary']}. SHAP contributions: {g['drivers_summary']}. "
        "This is a predictive score for the entered profile, not a causal estimate. "
        "(Factual summary — LLM interpretation unavailable.)"
    )
    return {
        "insight": insight,
        "key_takeaways": [g["result_summary"], f"Drivers: {g['drivers_summary']}"],
        "grounding": g["grounding"],
        "is_fallback": True,
    }


def generate_whatif_insight(g: dict[str, Any]) -> dict[str, Any]:
    pred = run_signature(
        PredictiveWhatIfInsightSignature,
        model_version=g["model_version"],
        profile_summary=g["profile_summary"],
        result_summary=g["result_summary"],
        drivers_summary=g["drivers_summary"],
        registry_context=g["registry_context"],
    )
    if pred is None:
        return _whatif_fallback(g)
    interpretation = str(getattr(pred, "interpretation", "")).strip()
    if not interpretation:
        return _whatif_fallback(g)
    return {
        "insight": interpretation,
        "key_takeaways": normalize_list(getattr(pred, "key_takeaways", [])),
        "grounding": g["grounding"],
        "is_fallback": False,
    }
