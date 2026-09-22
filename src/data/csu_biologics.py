"""CSU escalation-therapy vocabulary: the biologics / BTK inhibitor the Optum
converters match, in ONE place (Lane C, remibrutinib pre-wiring, spec
2026-09-22 §3C.1).

Two converters read the same vocabulary so a drug cannot be recognised by one
and dropped by the other:

* ``scripts/convert_optum_rwd.py`` (the claim-level CSU real-drop converter):
  ``_csu_biologic_mask`` ORs every signal here over a medication frame's
  ``code`` / ``Brand_Name`` / ``Generic_Name`` columns, and
  ``_classify_biologic_brand`` returns the ``key`` of the matched drug.
* ``scripts/convert_optum_mart.py`` (the enriched patient-grain drop): the
  vendor labels ``index_biologic_brand`` with the uppercase brand
  (``XOLAIR`` / ``DUPIXENT``) and ``index_biologic_molecule`` with the
  molecule; :func:`canonical_csu_arm` folds either onto the ``arm_label`` a
  causal cohort export contrasts on.

Remibrutinib (Rhapsido, oral BTK inhibitor; FDA CSU approval 2025-09-30) is
absent from every real drop to date (the drops end 2025-09-30). Its entry is
pre-wiring: the brand and generic names are the marketed ones and the NDC
prefixes carry BOTH the marketed product and the synthetic placeholder:

* marketed: openFDA product_ndc ``0078-1483`` (RHAPSIDO, Novartis
  Pharmaceuticals Corporation, 25 mg tablet; packages ``0078-1483-20`` /
  ``-92`` / ``-93``; verified 2026-09-22), listed in every form a claim code
  can take -- 11-digit dashless ``000781483`` (the drop's form), dashed 5-4
  ``00078-1483``, dashed 4-4 ``0078-1483`` (openFDA's print form) and the raw
  dashless 10-digit 4-4-2 ``00781483``;
* placeholder: ``src.ml.synthetic.clinical_codes.BRAND_NDC["Remibrutinib"]``
  (00078-1100-30), which the synthetic claims generator emits, in the two
  forms the generator produces (openFDA has no product ``0078-1100``, so it
  cannot misclassify a real fill).

Measured on the April 2026 raw drop (``data/rwd/csu/csu_data.xlsx`` medication
sheet, 10,000 rows): Brand_Name and Generic_Name are populated on every row
and no row is code-only, so on real data the names decide the match; the NDC
prefixes are the code-only fallback for a future drop that changes shape. The
bare Novartis labeler (00078 / 0078) is deliberately NOT a prefix — Kisqali
(00078-0903) and Fabhalta (00078-1175) share it. Known and accepted: an 8-digit
raw prefix (``00781483``) would also match an 11-digit code on labeler 00781
(Sandoz) with product code 483x; openFDA lists no such product (2026-09-22),
and Dupixent's ``0024`` prefix has carried the same ambiguity since #157.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Optional

import pandas as pd

# ``patient_journeys.brand`` / ``treatment_events.brand`` are ``brand_type``
# (database/core/e2i_ml_complete_v3_schema.sql: Remibrutinib / Fabhalta / Kisqali /
# competitor / other). The CSU competitors collapse to ``competitor``; the
# platform's own CSU brand carries its enum label.
COMPETITOR_JOURNEY_BRAND: Final[str] = "competitor"
REMIBRUTINIB_JOURNEY_BRAND: Final[str] = "Remibrutinib"


@dataclass(frozen=True)
class CsuBiologic:
    """One CSU escalation-therapy drug the converters recognise.

    ``key`` is what ``_classify_biologic_brand`` returns; ``arm_label`` is the
    canonical ``index_biologic_brand`` value (the vendor's uppercase brand);
    ``brand_names`` / ``generic_names`` are case-insensitive SUBSTRING matches on
    the claim's Brand_Name / Generic_Name; ``ndc_prefixes`` are ``startswith``
    matches on the claim code (the drop carries 11-digit dashless NDCs, tests
    also pass the dashed 5-4-2 form); ``hcpcs`` are exact code matches.
    """

    key: str
    arm_label: str
    brand_names: tuple[str, ...]
    generic_names: tuple[str, ...]
    ndc_prefixes: tuple[str, ...]
    hcpcs: frozenset[str]
    journey_brand: str


CSU_BIOLOGICS: Final[tuple[CsuBiologic, ...]] = (
    # Xolair (omalizumab): NDC labeler 50242 (Genentech), HCPCS J2357.
    CsuBiologic(
        key="xolair",
        arm_label="XOLAIR",
        brand_names=("XOLAIR",),
        generic_names=("omalizumab",),
        ndc_prefixes=("50242",),
        hcpcs=frozenset({"J2357"}),
        journey_brand=COMPETITOR_JOURNEY_BRAND,
    ),
    # Dupixent (dupilumab): NDC labeler 00024 / 0024 (Sanofi). J0517 is
    # canonically eculizumab, but the analyst spec assigns it to Dupixent and
    # the converter has followed the spec since #157 -- kept as-is.
    CsuBiologic(
        key="dupixent",
        arm_label="DUPIXENT",
        brand_names=("DUPIXENT",),
        generic_names=("dupilumab",),
        ndc_prefixes=("00024", "0024"),
        hcpcs=frozenset({"J0517"}),
        journey_brand=COMPETITOR_JOURNEY_BRAND,
    ),
    # Remibrutinib (Rhapsido): oral BTK inhibitor, no J-code. Product-level NDC
    # prefixes: the marketed 0078-1483 in every claim-code form, then the
    # synthetic placeholder 0078-1100 (see the module docstring).
    CsuBiologic(
        key="remibrutinib",
        arm_label="RHAPSIDO",
        brand_names=("RHAPSIDO",),
        generic_names=("remibrutinib",),
        ndc_prefixes=(
            "000781483",
            "00078-1483",
            "0078-1483",
            "00781483",
            "000781100",
            "00078-1100",
        ),
        hcpcs=frozenset(),
        journey_brand=REMIBRUTINIB_JOURNEY_BRAND,
    ),
)

_BY_KEY: Final[dict[str, CsuBiologic]] = {b.key: b for b in CSU_BIOLOGICS}
REMIBRUTINIB_ARM_LABEL: Final[str] = _BY_KEY["remibrutinib"].arm_label
# Every label (brand or molecule, uppercased) that folds onto an arm label.
_ARM_LABEL_BY_ALIAS: Final[dict[str, str]] = {
    **{alias.upper(): b.arm_label for b in CSU_BIOLOGICS for alias in b.brand_names},
    **{alias.upper(): b.arm_label for b in CSU_BIOLOGICS for alias in b.generic_names},
    **{b.arm_label.upper(): b.arm_label for b in CSU_BIOLOGICS},
}

# Flat views the claim-level converter re-exports under its historical names.
CSU_BIOLOGIC_HCPCS: Final[frozenset[str]] = frozenset().union(*(b.hcpcs for b in CSU_BIOLOGICS))
CSU_BIOLOGIC_NDC_PREFIXES: Final[tuple[str, ...]] = tuple(
    p for b in CSU_BIOLOGICS for p in b.ndc_prefixes
)
CSU_BIOLOGIC_GENERICS: Final[tuple[str, ...]] = tuple(
    g for b in CSU_BIOLOGICS for g in b.generic_names
)
CSU_BIOLOGIC_BRANDS: Final[tuple[str, ...]] = tuple(n for b in CSU_BIOLOGICS for n in b.brand_names)


def csu_biologic_by_key(key: str) -> CsuBiologic:
    """The vocabulary entry for ``key`` (``xolair`` / ``dupixent`` / ``remibrutinib``)."""
    return _BY_KEY[key]


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


def classify_csu_biologic(
    *,
    brand_name: object = None,
    generic_name: object = None,
    code: object = None,
) -> Optional[str]:
    """Return the matched drug's ``key`` for one medication row, or ``None``.

    Precedence is the converter's historical one: Brand_Name, then
    Generic_Name, then the claim code (NDC prefix or HCPCS); within a signal the
    vocabulary order (Xolair, Dupixent, remibrutinib) decides.
    """
    if not _is_blank(brand_name):
        b = str(brand_name).strip().upper()
        for drug in CSU_BIOLOGICS:
            if any(name in b for name in drug.brand_names):
                return drug.key
    if not _is_blank(generic_name):
        g = str(generic_name).strip().lower()
        for drug in CSU_BIOLOGICS:
            if any(name in g for name in drug.generic_names):
                return drug.key
    if not _is_blank(code):
        c = str(code).strip().upper()
        for drug in CSU_BIOLOGICS:
            if c in drug.hcpcs or any(c.startswith(p) for p in drug.ndc_prefixes):
                return drug.key
    return None


def canonical_csu_arm(label: object) -> Optional[str]:
    """Fold a vendor brand or molecule label onto the canonical arm label.

    ``XOLAIR`` / ``omalizumab`` -> ``XOLAIR``; ``DUPIXENT`` / ``DUPILUMAB`` ->
    ``DUPIXENT``; ``RHAPSIDO`` / ``remibrutinib`` -> ``RHAPSIDO``. Anything else
    (``no_treatment``, blanks, NaN, another product) -> ``None``.
    """
    if _is_blank(label):
        return None
    return _ARM_LABEL_BY_ALIAS.get(str(label).strip().upper())


def journey_brand_for(key: Optional[str]) -> str:
    """The ``brand_type`` label a journey / treatment event carries for a
    matched drug key; ``None`` (no CSU biologic) stays ``competitor`` -- the
    converter's historical value for every CSU row."""
    if key is None:
        return COMPETITOR_JOURNEY_BRAND
    return _BY_KEY[key].journey_brand
