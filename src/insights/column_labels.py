"""Display labels + plain-language definitions for the gold-standard columns.

ONE leaf SSOT for every surface that prints a treatment / outcome column:
``GET /causal/variables`` and ``GET /segments/datasets`` serve ``labels`` (and
``definitions``) per offered column, the discover-effects summary prose and
the strategic-insight builders (src/insights/hte.py, treatment_effect.py,
clinical_narrative.py, causal_discovery.py) interpolate ``column_label``, and
the frontend's shared ``columnLabel`` (frontend/src/lib/column-labels.ts)
mirrors the fallback byte-for-byte.

History: this lived in src/api/routes/causal.py (``_COLUMN_LABELS``). The
route still re-exports it under the historical names. Moved here 2026-09-05
(#1895) because the insight builders could not import a route module without
an insights -> api.routes inversion (and the route's heavy import graph), so
they rendered raw column names ("For treatment_arm -> persistent_180d ...")
under headers that already read "Treatment arm -> Persistent at 180d".

Leaf by design: NO imports from src.api / src.insights.* / anything heavy —
``tests/insights/test_column_labels.py`` pins that.
"""

from __future__ import annotations

# Human-readable display labels for the curated columns (data-driven FE; keeps
# the frontend free of a humanizer). Columns absent here fall back to the raw
# name title-cased by the caller.
COLUMN_LABELS: dict[str, str] = {
    "treatment_arm": "Treatment arm",
    "treatment_initiated": "Treatment initiated",
    "persistent_180d": "Persistent at 180d",
    "discontinued_180d": "Discontinued at 180d",
    "adherent_180d": "Adherent at 180d",
    "low_gap_180d": "Low refill gap (≤30d)",
    "adherence_rate": "Adherence rate (PDC)",
    "gap_days": "Refill gap (days)",
    # #1188 RCT baseline covariates (joined from patient_journeys).
    # disease_severity is a UNIVERSAL confounder: one generic normal(5,2)-clipped
    # 0–10 index applied identically to every brand (CSU / breast cancer / PNH), NOT
    # a per-disease clinical instrument like UAS7 / ECOG / eGFR. The label states the
    # cross-indication nature so it never reads as an indication-specific severity
    # score next to the real per-brand biomarkers (Part C, 2026-07-13).
    "disease_severity": "Disease severity (cross-indication 0–10 index)",
    "age_at_diagnosis": "Age at diagnosis",
    "academic_hcp": "Academic HCP",
    "geographic_region": "Geographic region",
    "biologic_experienced": "Biologic-experienced (prior anti-IgE)",
    # #1321 Fabhalta pilot: prior C5-inhibitor switch (eculizumab/ravulizumab).
    "complement_inhibitor_status": "Prior C5-inhibitor (switch)",
    # #1321 rollout: the two brand-distinct axes' KG-node labels (the node name is the
    # raw eligibility column; the label states the derived contrast).
    "disease_stage": "Advanced line (metastatic / stage IV)",
    "urticaria_severity_uas7": "Uncontrolled CSU (UAS7 ≥ 28)",
    "copay_support": "Copay support",
    "psp_enrolled": "Patient support program",
    # COMM-ARMS Phase 3/4: the initiation-latent commercial arms. "Sample
    # dropped" (rep jargon for leaving product samples) read as "excluded from
    # the sample" on the /segment-analysis dropdown (2026-09-04), so the label
    # names the lever. trigger_accepted previously had NO curated label and fell
    # back to the auto-capitalized "Trigger accepted" while the clinical-context
    # panel (brand_map._COMMERCIAL_TREATMENT_CONTEXT) said "NBA trigger accepted";
    # the two SSOTs are now pinned equal (test_commercial_arm_labels_agree_...).
    "rep_detailing_high": "High rep detailing",
    "sample_dropped": "Product samples provided (rep sample drop)",
    "trigger_accepted": "NBA trigger accepted",
    # NOT a measured payer metric: a deterministic access gradient derived from
    # insurance_type (range approx -0.35..+0.45, higher = better access). The label
    # says "derived" for the same reason disease_severity's label says
    # "cross-indication" — an analyst must not read a synthetic index as a real
    # instrument.
    "insurance_access_score": "Insurance access (derived from insurance type)",
}


def column_label(col: str) -> str:
    """Display label for a gold-standard column: the curated ``COLUMN_LABELS``
    entry, else the auto-label (underscores -> spaces, first letter capitalised).

    The ONE label path for every user-facing surface — ``GET /causal/variables``
    and ``GET /segments/datasets`` serve it per offered column, and the
    discover-effects summary prose interpolates it. 2026-09-05: the
    /segment-analysis relabel (#1893, "Product samples provided (rep sample
    drop)") never reached the /causal-analysis leaderboard text, which still
    interpolated the raw ``sample_dropped``."""
    if not col:
        return ""
    return COLUMN_LABELS.get(col, col.replace("_", " ").capitalize())


# Plain-language definitions of the curated treatment/outcome columns — what a
# 1 vs 0 means — served next to COLUMN_LABELS (GET /segments/datasets
# `definitions`) so the config dropdowns can explain the selected option
# (2026-09-04 /segment-analysis review: "Sample dropped" was unreadable and no
# option said what it measured). Worded from the DGP that generates the columns
# (src/ml/synthetic/dgp/*, treatment_arm.ARM_REGISTRY, adherence_outcomes,
# cohort_outcomes). Every patient_journeys treatment + outcome MUST have an entry
# (test_every_patient_grain_option_has_a_definition); a column absent here is
# simply served without a definition — the FE never invents one.
COLUMN_DEFINITIONS: dict[str, str] = {
    # --- patient_journeys treatments ---
    "treatment_arm": (
        "Patient is on the brand's therapy (1) vs not (0). Observational: assignment "
        "depends on disease severity and academic HCP, which the analysis adjusts for."
    ),
    "copay_support": "Patient received copay assistance (1) vs not (0).",
    "psp_enrolled": "Patient enrolled in the brand's patient support program (1) vs not (0).",
    "rep_detailing_high": (
        "The patient's prescriber received high-frequency sales-rep detailing (1) vs not (0)."
    ),
    "sample_dropped": (
        "The patient's prescriber received product samples from the sales rep (1) vs "
        "not (0). A promotional lever, not a data exclusion."
    ),
    "trigger_accepted": (
        "The prescriber acted on a next-best-action (NBA) trigger (1) vs not (0)."
    ),
    "complement_inhibitor_status": (
        "Fabhalta only: patient switched from a prior C5 inhibitor "
        "(eculizumab / ravulizumab) (1) vs C5-inhibitor-naive (0)."
    ),
    "disease_stage": (
        "Kisqali only: advanced-line disease (metastatic / stage IV) (1) vs earlier line (0)."
    ),
    "urticaria_severity_uas7": (
        "Remibrutinib only: uncontrolled CSU, UAS7 ≥ 28 (1) vs below 28 (0)."
    ),
    # --- patient_journeys outcomes (treatment_initiated is also the target of the
    # initiation-latent arms, so it sits on both sides of the question) ---
    "treatment_initiated": "Patient started therapy (1) vs did not (0).",
    # persistent/discontinued are drawn for EVERY row (the DGP conditions on
    # treatment_arm, not treatment_initiated) and the loader applies no initiator
    # filter, so the wording must not promise a cohort restriction that is not
    # applied (codex iter-1, PR #1893; measured on prod 2026-09-04).
    "persistent_180d": (
        "Still on therapy at day 180 of the window (1) vs not (0). "
        "The complement of Discontinued at 180d."
    ),
    "discontinued_180d": (
        "Stopped therapy within the 180-day window (1) vs still on therapy (0). "
        "The complement of Persistent at 180d."
    ),
    "adherent_180d": (
        "Proportion of days covered (PDC) ≥ 0.80 over the 180-day window (1) vs below (0)."
    ),
    "low_gap_180d": (
        "Refill gap of 30 days or less over the 180-day window (1) vs a longer gap (0). "
        "A subset of Adherent at 180d."
    ),
}
