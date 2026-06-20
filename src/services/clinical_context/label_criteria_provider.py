"""LabelCriteriaProvider — derive an indication-scoped IndicatedPopulation for a
brand by confirming reviewed candidate criteria against the LIVE OpenFDA label.

Design (codex HIGH#1, label-PRIMARY): the reviewed cohort_constructor ``CohortConfig``
supplies the *candidate* criteria (column + operator + value — the reviewed,
column-bound reconciliation cache), and the **live OpenFDA label is the gating
authority**: a deterministic, brand-agnostic evidence-matcher confirms each
candidate against the live label text (disease name, "adult", status tokens,
prior-therapy phrases, stated thresholds). Each criterion is tagged
``label_evidenced`` vs config-unconfirmed. The label decides which criteria are
active — so this derives the indicated population FROM the API — without the brittle
free-text *value* extraction the prototype disproved (it reuses reviewed values).

Indication-scoped (codex HIGH#2): ``derive(brand, indication)`` — Fabhalta must not
silently default to PNH; callers pass the data-resolved indication.

Fail-open: live label unreachable -> ``source="unavailable"`` with all criteria
unconfirmed (the gate then returns indeterminate — no hard flag without label support).
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from src.agents.cohort_constructor.configs import get_brand_config, list_available_configs
from src.agents.cohort_constructor.types import CriterionType, Operator
from src.services.clinical_context.brand_map import resolve_brand_profile
from src.services.clinical_context.clients import _OpenFDAClient
from src.services.clinical_context.label_gate import GateCriterion, IndicatedPopulation

logger = logging.getLogger(__name__)

# General CLINICAL-CONCEPT lexicon keyed by DATA COLUMN (the existing SSOT field
# names). Brand-agnostic: each entry is the set of token patterns that, if present
# in the live label text, EVIDENCE that column's criterion. This is the
# "match-to-existing-data" binding — not per-brand rules. Value-bearing categorical
# fields encode the expected value in the pattern.
_FIELD_TOKENS: dict[str, List[str]] = {
    "age_at_diagnosis": [r"\badult", r"\b18 years", r"≥\s*18", r">=\s*18"],
    "prior_antihistamine_therapy": [
        r"(despite|inadequately controlled|symptomatic despite|following|after).{0,40}antihistamine",
        r"antihistamine.{0,40}(treatment|therapy)",
    ],
    "hr_status": [r"hr[- ]?positive", r"hormone receptor.{0,14}positive"],
    "her2_status": [r"her2.{0,4}negative", r"human epidermal growth factor receptor 2.{0,8}negative"],
    "disease_stage": [r"advanced", r"metastatic", r"stage ii", r"stage iii"],
    "ecog_performance_status": [r"\becog\b", r"performance status"],
    "urticaria_severity_uas7": [r"uas7", r"urticaria activity score", r"moderate.to.severe"],
    "ldh_ratio": [r"\bldh\b", r"lactate dehydrogenase"],
    "proteinuria_g_day": [r"proteinuria", r"\bupcr\b", r"urine protein"],
    "egfr": [r"egfr", r"glomerular filtration"],
    "complement_inhibitor_status": [r"complement inhibitor"],
    # exclusions (matched against the broader label incl. boxed warning)
    "active_serious_infection": [r"serious infection", r"encapsulated bacteria"],
    "meningococcal_vaccination_current": [r"meningococc", r"vaccinat"],
}

_SNIPPET_PAD = 60


def _snippet(text: str, match: re.Match) -> str:
    start = max(0, match.start() - _SNIPPET_PAD)
    end = min(len(text), match.end() + _SNIPPET_PAD)
    return ("…" if start else "") + text[start:end].strip() + ("…" if end < len(text) else "")


def _evidence_for_criterion(field: str, label_text: str, disease: str) -> Optional[str]:
    """Return a live-label snippet evidencing ``field``'s criterion, or None.

    diagnosis_code is evidenced by the disease NAME appearing in the label; all
    other fields by their concept lexicon."""
    if field == "diagnosis_code":
        for token in (w for w in disease.lower().split() if len(w) > 4):
            m = re.search(re.escape(token), label_text)
            if m:
                return _snippet(label_text, m)
        return None
    for pat in _FIELD_TOKENS.get(field, []):
        m = re.search(pat, label_text, flags=re.IGNORECASE)
        if m:
            return _snippet(label_text, m)
    return None


_CATEGORICAL_OPS = (Operator.EQUAL, Operator.NOT_EQUAL, Operator.IN, Operator.NOT_IN)


def label_segment_columns(brand: str, indication: Optional[str] = None) -> List[str]:
    """The brand's label-relevant CATEGORICAL inclusion-criterion columns — the
    dimensions worth segmenting on so on/off-label bands form (codex HIGH#3). Excludes
    diagnosis_code (constant within a brand) and continuous-threshold criteria (which
    segment poorly per exact value and are label-unconfirmed anyway)."""
    try:
        cfg = get_brand_config(brand, indication)
    except ValueError:
        return []
    cols: List[str] = []
    for c in cfg.inclusion_criteria:
        if c.field == "diagnosis_code":
            continue
        if c.operator in _CATEGORICAL_OPS and c.field not in cols:
            cols.append(c.field)
    return cols


def resolve_indication(brand: str, diagnosis_codes) -> Optional[str]:
    """Resolve the indication from the frame's diagnosis distribution (codex HIGH#2 —
    Fabhalta must not silently default to PNH). Returns the brand indication whose
    diagnosis_code set covers the frame's codes when UNAMBIGUOUS; else None (caller
    falls back to the brand default, logged)."""
    codes = {str(c).strip() for c in (diagnosis_codes or []) if c is not None and str(c).strip()}
    if not codes:
        return None
    brand_lower = brand.lower()
    matches: set[str] = set()
    for key, summary in list_available_configs().items():
        if not key.startswith(brand_lower):
            continue
        cfg = get_brand_config(brand, summary["indication"])
        code_sets = [
            set(c.value)
            for c in cfg.inclusion_criteria
            if c.field == "diagnosis_code" and isinstance(c.value, (list, tuple, set))
        ]
        if code_sets and codes & set().union(*code_sets):
            matches.add(summary["indication"])
    return next(iter(matches)) if len(matches) == 1 else None


class LabelCriteriaProvider:
    """Derive an IndicatedPopulation: reviewed candidate criteria confirmed against
    the live OpenFDA label. ``openfda_client`` is injectable (faithful fixtures /
    live); defaults to the real client."""

    def __init__(self, openfda_client: Optional[object] = None) -> None:
        self._client = openfda_client if openfda_client is not None else _OpenFDAClient()

    def derive(self, brand: str, indication: Optional[str] = None) -> IndicatedPopulation:
        cfg = get_brand_config(brand, indication)
        try:
            profile = resolve_brand_profile(brand)
            drug_name, disease = profile.drug_name, profile.disease
        except KeyError:
            drug_name, disease = brand, cfg.indication
        inclusion_text, full_text, fetched = self._label_text(drug_name)

        gate_criteria: List[GateCriterion] = []
        for crit in list(cfg.inclusion_criteria) + list(cfg.exclusion_criteria):
            # Inclusion criteria define the indicated POPULATION -> evidenced only by
            # the indications text. Exclusions are safety-driven -> may be evidenced
            # by the boxed warning too. (Without this split, Fabhalta's boxed-warning
            # self-description "a complement inhibitor" would falsely evidence the
            # patient-side complement_inhibitor_status inclusion criterion.)
            search_text = (
                full_text if crit.criterion_type is CriterionType.EXCLUSION else inclusion_text
            )
            evidence = (
                _evidence_for_criterion(crit.field, search_text, disease) if fetched else None
            )
            gate_criteria.append(
                GateCriterion(
                    criterion=crit,
                    label_evidenced=bool(evidence),
                    label_evidence=evidence,
                )
            )

        if not fetched:
            source = "unavailable"
        elif any(gc.label_evidenced for gc in gate_criteria):
            source = "openfda_evidenced"
        else:
            source = "config_unconfirmed"

        return IndicatedPopulation(
            brand=brand, indication=cfg.indication, criteria=gate_criteria, source=source
        )

    def _label_text(self, drug_name: str) -> Tuple[str, str, bool]:
        """Return (inclusion_text, full_text, fetched). ``inclusion_text`` =
        indications + LoU (where the indicated population is described);
        ``full_text`` adds the boxed warning (safety — for exclusion evidence).
        Fail-open: any failure -> ('', '', False)."""
        try:
            label = self._client.fetch_label(drug_name)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 — best-effort; any failure => unavailable
            logger.warning("label-criteria: OpenFDA fetch failed for %s: %s", drug_name, exc)
            label = None
        if not label:
            return "", "", False
        inclusion_parts: List[str] = []
        boxed: Optional[str] = None
        try:
            inclusion_parts.extend(self._client.approved_indications(label))  # type: ignore[attr-defined]
            lou = self._client.limitations_of_use(label)  # type: ignore[attr-defined]
            if lou:
                inclusion_parts.append(lou)
            boxed = self._client.boxed_warning(label)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            logger.warning("label-criteria: extraction failed for %s: %s", drug_name, exc)
            if not inclusion_parts:
                return "", "", False
        inclusion_text = " ".join(inclusion_parts).lower()
        full_text = (inclusion_text + " " + boxed.lower()) if boxed else inclusion_text
        return inclusion_text, full_text, True
