#!/usr/bin/env python3
"""Reconcile output schemas of ``convert_csu_rwd.py`` and ``convert_optum_rwd.py``.

This script emits a mapping table for overlapping concepts (age, brand,
indication, comorbidity flags, time-window aggregates) between the two
RWD converters. It is the executable companion to
``docs/lineage/csu_field_audit.md``.

Modes
-----
* ``--mode synthetic`` (default): build minimal in-memory fixtures that
  exercise the schemas without touching real data. Good for CI.
* ``--mode files``: read the most recent CSU/Optum outputs from
  ``data/rwd/csu/*.json`` and ``data/rwd/optum/*/*.parquet`` and reconcile
  their actual schemas. Requires both data trees to exist.

Outputs
-------
* ``docs/lineage/converter_schema_reconciliation.json`` — machine-readable
  mapping consumed downstream.
* stdout — human-readable summary table.

Exit codes
----------
* ``0`` if every overlapping concept has matching dtypes AND
  ``semantic_match=True``.
* Non-zero if any overlap mismatches (so this can be wired into CI later).

Usage
-----
::

    python scripts/converter_schema_reconciliation.py
    python scripts/converter_schema_reconciliation.py --mode files
    python scripts/converter_schema_reconciliation.py --output /tmp/recon.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "lineage" / "converter_schema_reconciliation.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Reconciliation primitives                                                   #
# --------------------------------------------------------------------------- #


@dataclass
class FieldMapping:
    """A single overlapping concept across the two converters."""

    concept: str
    csu_column: str | None
    csu_dtype: str | None
    optum_column: str | None
    optum_dtype: str | None
    semantic_match: bool
    notes: str = ""

    @property
    def dtype_match(self) -> bool:
        """Pandas-style dtype family equivalence (int64 ≡ Int64, etc.)."""
        if self.csu_dtype is None or self.optum_dtype is None:
            return False
        return _dtype_family(self.csu_dtype) == _dtype_family(self.optum_dtype)

    @property
    def is_clean(self) -> bool:
        """Concept is reconciled iff present on both sides AND dtype/semantic match."""
        if self.csu_column is None or self.optum_column is None:
            return False
        return self.dtype_match and self.semantic_match


@dataclass
class ReconciliationReport:
    """Top-level reconciliation result, written to JSON for downstream consumers."""

    mode: str
    csu_columns_total: int
    optum_columns_total: int
    overlapping_concepts_total: int
    overlapping_concepts_clean: int
    field_mappings: list[FieldMapping] = field(default_factory=list)
    csu_only_columns: list[str] = field(default_factory=list)
    optum_only_columns: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def has_mismatches(self) -> bool:
        """True iff at least one overlap is not clean."""
        return self.overlapping_concepts_clean < self.overlapping_concepts_total

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["has_mismatches"] = self.has_mismatches
        return d


# --------------------------------------------------------------------------- #
# Concept catalogue — overlapping fields between CSU and Optum journeys       #
# --------------------------------------------------------------------------- #
#
# Each entry maps a logical concept to the column name each converter emits
# for that concept. The semantic_match flag captures whether the two sides
# represent the *same* notion (e.g. age-at-index vs binned age-group are
# both "age" but not semantically equal — semantic_match=False).
#
# Source for column lists:
#   * CSU: scripts/convert_csu_rwd.py — _build_patient_journeys / _build_hcp_profiles
#   * Optum: scripts/convert_optum_rwd.py — _build_journey_record / _build_hcp_profiles
#
# This catalogue is intentionally maintained in code (not config) so that
# unit tests can lock the contract.

CONCEPT_CATALOGUE: list[dict[str, Any]] = [
    {
        "concept": "patient_id",
        "csu_column": "patient_id",
        "optum_column": "patient_id",
        "semantic_match": True,
        "notes": "Both use PAT_-prefixed string IDs.",
    },
    {
        "concept": "patient_journey_id",
        "csu_column": "patient_journey_id",
        "optum_column": "patient_journey_id",
        "semantic_match": True,
        "notes": "Both use PJ_-prefixed string IDs.",
    },
    {
        "concept": "patient_hash",
        "csu_column": "patient_hash",
        "optum_column": "patient_hash",
        "semantic_match": True,
        "notes": "SHA-256 prefix of patid.",
    },
    {
        "concept": "index_date",
        "csu_column": "journey_start_date",
        "optum_column": "index_date",
        "semantic_match": False,
        "notes": (
            "CSU: vendor-assigned indexdt (or earliest clinical date fallback). "
            "Optum: claim-anchored qualifying-dx date per analyst spec §3. "
            "Different anchors — NOT comparable as features."
        ),
    },
    {
        "concept": "journey_start_date",
        "csu_column": "journey_start_date",
        "optum_column": "journey_start_date",
        "semantic_match": True,
        "notes": "Both ISO-format date strings; semantically the index date.",
    },
    {
        "concept": "lookback_window_start",
        "csu_column": None,
        "optum_column": "lookback_start_date",
        "semantic_match": False,
        "notes": (
            "Optum exposes the lookback window explicitly; CSU does not "
            "(no lookback masking applied)."
        ),
    },
    {
        "concept": "prediction_window_end",
        "csu_column": None,
        "optum_column": "prediction_end_date",
        "semantic_match": False,
        "notes": "Optum-only; CSU has no prediction window concept.",
    },
    {
        "concept": "age",
        "csu_column": "age_continuous",
        "optum_column": "age_at_index",
        "semantic_match": True,
        "notes": (
            "Both raw float age. CSU also emits age_group (binned); Optum emits "
            "age_group separately. Reconciled at the continuous-age concept."
        ),
    },
    {
        "concept": "age_group",
        "csu_column": "age_group",
        "optum_column": "age_group",
        "semantic_match": True,
        "notes": "Both use the same bins via the shared rwd_common helper logic.",
    },
    {
        "concept": "gender",
        "csu_column": "gender",
        "optum_column": "gender",
        "semantic_match": True,
        "notes": (
            "Both M/F strings. Optum uses 'U' for unknown; CSU uses None — see "
            "downstream null-handling in data_preparer."
        ),
    },
    {
        "concept": "geographic_region",
        "csu_column": "geographic_region",
        "optum_column": "geographic_region",
        "semantic_match": True,
        "notes": "Both use Census 4-region from 3-digit ZIP.",
    },
    {
        "concept": "zip_code_5",
        "csu_column": "zip_code",
        "optum_column": "zip5",
        "semantic_match": True,
        "notes": "Both 5-digit ZIP strings. Naming difference: zip_code vs zip5.",
    },
    {
        "concept": "insurance_product",
        "csu_column": "insurance_type",
        "optum_column": "insurance_product",
        "semantic_match": True,
        "notes": "Both Commercial/Medicare/Medicaid/Other strings.",
    },
    {
        "concept": "primary_diagnosis_code",
        "csu_column": "primary_diagnosis_code",
        "optum_column": "primary_diagnosis_code",
        "semantic_match": True,
        "notes": "Both ICD-10 dx code with dot inserted via _format_diagcode.",
    },
    {
        "concept": "brand",
        "csu_column": "brand",
        "optum_column": "brand",
        "semantic_match": True,
        "notes": (
            "CSU writes 'competitor' iff treatment_initiated, None otherwise — "
            "POST-INDEX. Optum writes 'competitor' as a constant tag — "
            "OBSERVABLE. Same column name, different semantics; flag for "
            "downstream review."
        ),
    },
    {
        "concept": "comorbidity_flags",
        "csu_column": "comorbidities",
        "optum_column": "comorbidities",
        "semantic_match": False,
        "notes": (
            "CSU writes constant []; Optum also writes [] but emits per-comorbidity "
            "flag columns (has_atopic_dermatitis, has_asthma, ...). The list-typed "
            "column is a vestigial schema field on both sides."
        ),
    },
    {
        "concept": "data_source",
        "csu_column": "data_source",
        "optum_column": "data_source",
        "semantic_match": True,
        "notes": "Both 'RWD_Claims' string constant.",
    },
    {
        "concept": "data_quality_score",
        "csu_column": "data_quality_score",
        "optum_column": "data_quality_score",
        "semantic_match": False,
        "notes": (
            "CSU: random uniform in archetype-dependent band (synthetic noise). "
            "Optum: fraction of §7 features non-null. Same column, different "
            "semantics."
        ),
    },
    {
        "concept": "data_split",
        "csu_column": "data_split",
        "optum_column": "data_split",
        "semantic_match": True,
        "notes": "Both train/validation/test/holdout enum.",
    },
    {
        "concept": "treatment_initiated",
        "csu_column": "treatment_initiated",
        "optum_column": "treatment_initiated",
        "semantic_match": False,
        "notes": (
            "CSU: '1 if patient appears in medication sheet'. Optum: "
            "initiated_biologic_180d aliased to treatment_initiated for "
            "tier-0 backward compat. Different temporal semantics — "
            "CSU is POST-INDEX (entire med history), Optum is windowed."
        ),
    },
    {
        "concept": "discontinuation_flag",
        "csu_column": "discontinuation_flag",
        "optum_column": "discontinuation_flag",
        "semantic_match": False,
        "notes": (
            "CSU: gap > 90d derived over entire med history. Optum: "
            "discontinued_180d windowed to [init_date, init_date+180]. "
            "Same column name, different semantics."
        ),
    },
    {
        "concept": "days_on_therapy",
        "csu_column": "days_on_therapy",
        "optum_column": None,
        "semantic_match": False,
        "notes": (
            "CSU-only feature; Optum does not emit a 'days_on_therapy' column "
            "because it would be POST-INDEX (matches §3 row 38 in audit)."
        ),
    },
    {
        "concept": "hcp_visits",
        "csu_column": "hcp_visits",
        "optum_column": None,
        "semantic_match": False,
        "notes": (
            "CSU-only feature. Optum offers office_visits_total / "
            "office_visits_allergist / office_visits_dermatology / "
            "office_visits_pcp instead, all with explicit lookback windows."
        ),
    },
    {
        "concept": "medication_claim_count",
        "csu_column": "medication_claim_count",
        "optum_column": None,
        "semantic_match": False,
        "notes": (
            "CSU-only feature; trivially equivalent to treatment_initiated > 0 "
            "(POST-INDEX). Optum has no analogue."
        ),
    },
    {
        "concept": "engagement_score",
        "csu_column": "engagement_score",
        "optum_column": None,
        "semantic_match": False,
        "notes": (
            "CSU-only synthetic composite. Optum decomposes into "
            "office_visits_*, unique_providers, specialist_concentration."
        ),
    },
    {
        "concept": "disease_severity",
        "csu_column": "disease_severity",
        "optum_column": None,
        "semantic_match": False,
        "notes": (
            "CSU-only synthetic composite. Optum decomposes into dx_l50_*_count, "
            "dx_total_csu, charlson_score, elixhauser_score (all lookback-windowed)."
        ),
    },
    {
        "concept": "hcp_id",
        "csu_column": "hcp_id",
        "optum_column": "hcp_id",
        "semantic_match": True,
        "notes": "HCP profiles: both use HCP_-prefixed string IDs.",
    },
    {
        "concept": "hcp_specialty",
        "csu_column": "specialty",
        "optum_column": "specialty",
        "semantic_match": False,
        "notes": (
            "CSU: hard-coded 'Allergy/Immunology' for all HCPs. Optum: derived "
            "from provider.taxonomy1 (207K → Allergy/Immunology, 207N → "
            "Dermatology, else Other). Same column, different fidelity."
        ),
    },
    {
        "concept": "hcp_practice_type",
        "csu_column": "practice_type",
        "optum_column": "practice_type",
        "semantic_match": True,
        "notes": "Both Hospital/Group/Solo derived from patient volume thresholds.",
    },
    {
        "concept": "hcp_adoption_category",
        "csu_column": "adoption_category",
        "optum_column": "adoption_category",
        "semantic_match": True,
        "notes": "Both derived from prescribing-volume quartiles.",
    },
]


# --------------------------------------------------------------------------- #
# Schema extraction                                                           #
# --------------------------------------------------------------------------- #


def _dtype_family(dtype: str) -> str:
    """Coarse-grained dtype equivalence for cross-format reconciliation.

    pyarrow / pandas / json-derived dtypes can express the same concept many
    ways (``int64`` vs ``Int64`` vs ``int``). We collapse to a small family
    set so the reconciliation report doesn't trip on representation noise.
    """
    d = str(dtype).lower().strip()
    if d.startswith("int") or d in {"long", "smallint", "bigint"}:
        return "int"
    if d.startswith("float") or d in {"double", "real", "decimal"}:
        return "float"
    if d.startswith(("bool", "boolean")):
        return "bool"
    if d.startswith(("datetime", "timestamp", "date")):
        return "datetime"
    # Order matters: check struct/dict family BEFORE string family because
    # 'struct' literally starts with 'str' and would otherwise be miscategorised.
    if d.startswith(("dict", "struct", "map")):
        return "dict"
    if d.startswith(("list", "array")):
        return "list"
    if d.startswith(("string", "object", "utf8", "large_string", "str")):
        return "string"
    return d


def _build_synthetic_csu_schema() -> dict[str, str]:
    """Synthetic schema mirroring ``CSUDataConverter._build_patient_journeys``.

    Used in ``--mode synthetic`` so the script can reconcile contracts
    without requiring the Excel workbook or any real data files. The dtype
    map below was derived from code-reading (the same source as the audit).
    """
    return {
        # IDs
        "patient_journey_id": "string",
        "patient_id": "string",
        "patient_hash": "string",
        # Dates
        "journey_start_date": "string",  # ISO date string
        "journey_end_date": "string",
        "journey_duration_days": "int",
        # Journey
        "journey_stage": "string",
        "journey_status": "string",
        "primary_diagnosis_code": "string",
        "primary_diagnosis_desc": "string",
        "secondary_diagnosis_codes": "list",
        "brand": "string",
        # Demo
        "age_group": "string",
        "gender": "string",
        "geographic_region": "string",
        "state": "string",
        "zip_code": "string",
        "insurance_type": "string",
        # Quality
        "data_quality_score": "float",
        "comorbidities": "list",
        "risk_score": "float",
        # Provenance
        "data_source": "string",
        "data_sources_matched": "list",
        "source_match_confidence": "float",
        "source_stacking_flag": "bool",
        "source_combination_method": "string",
        "source_timestamp": "string",
        "ingestion_timestamp": "string",
        "data_lag_hours": "float",
        "data_split": "string",
        "split_config_id": "string",
        "created_at": "string",
        "updated_at": "string",
        # Targets / leaky features
        "treatment_initiated": "int",
        "discontinuation_flag": "int",
        "disease_severity": "float",
        "engagement_score": "float",
        "days_on_therapy": "int",
        "hcp_visits": "int",
        "prior_treatments": "int",
        # Pass-through extras
        "age_continuous": "float",
        "eligibility_duration_days": "int",
        "medication_claim_count": "int",
        "procedure_claim_count": "int",
        "lab_claim_count": "int",
        # HCP profile columns (audit catalogue spans HCP fields too — include
        # them here so reconciliation lines up at the concept level).
        "hcp_id": "string",
        "specialty": "string",
        "practice_type": "string",
        "adoption_category": "string",
    }


def _build_synthetic_optum_schema() -> dict[str, str]:
    """Synthetic schema mirroring ``OptumDataConverter._build_journey_record``."""
    return {
        # IDs
        "patient_journey_id": "string",
        "patient_id": "string",
        "patient_hash": "string",
        # Dates (Optum-specific)
        "index_date": "string",
        "lookback_start_date": "string",
        "prediction_end_date": "string",
        "journey_start_date": "string",
        "journey_end_date": "string",
        "journey_duration_days": "int",
        # Journey
        "journey_stage": "string",
        "journey_status": "string",
        "primary_diagnosis_code": "string",
        "primary_diagnosis_desc": "string",
        "secondary_diagnosis_codes": "list",
        "brand": "string",
        # Geography
        "state": "string",
        "zip_code": "string",
        "comorbidities": "list",
        "risk_score": "float",
        # Provenance
        "data_source": "string",
        "data_sources_matched": "list",
        "source_match_confidence": "float",
        "source_stacking_flag": "bool",
        "source_combination_method": "string",
        "source_timestamp": "string",
        "ingestion_timestamp": "string",
        "data_lag_hours": "float",
        "data_split": "string",
        "created_at": "string",
        "updated_at": "string",
        "data_quality_score": "float",
        # Targets
        "initiated_biologic_180d": "int",
        "discontinued_180d": "int",
        "persistent_at_180d": "int",
        "treatment_initiated": "int",
        "discontinuation_flag": "int",
        # Demo (from _compute_features 7.1)
        "age_at_index": "float",
        "age_group": "string",
        "gender": "string",
        "zip5": "string",
        "zip3": "string",
        "geographic_region": "string",
        "insurance_product": "string",
        "plan_type": "string",
        "urban_rural_code": "string",
        # Disease (7.2)
        "dx_l50_1_count": "int",
        "dx_l50_8_count": "int",
        "dx_l50_9_count": "int",
        "dx_total_csu": "int",
        "dx_angioedema_count": "int",
        "months_since_first_dx": "int",
        "csu_chronicity": "string",
        # Comorbidities (7.3) — has_<name> + <name>_claim_count for each in COMORBIDITY_CODES
        "has_atopic_dermatitis": "int",
        "atopic_dermatitis_claim_count": "int",
        "has_asthma": "int",
        "asthma_claim_count": "int",
        "has_allergic_rhinitis": "int",
        "allergic_rhinitis_claim_count": "int",
        "has_anxiety": "int",
        "anxiety_claim_count": "int",
        "has_depression": "int",
        "depression_claim_count": "int",
        "has_thyroid_autoimmune": "int",
        "thyroid_autoimmune_claim_count": "int",
        "has_nsaid_hypersensitivity": "int",
        "nsaid_hypersensitivity_claim_count": "int",
        "has_angioedema": "int",
        "angioedema_claim_count": "int",
        "atopy_score": "int",
        "mental_health_flag": "int",
        "elixhauser_score": "int",
        "charlson_score": "int",
        # Utilization (7.4)
        "office_visits_total": "int",
        "office_visits_allergist": "int",
        "office_visits_dermatology": "int",
        "office_visits_pcp": "int",
        "ed_visits_total": "int",
        "ed_visits_urticaria_angio": "int",
        "hospitalizations_total": "int",
        "unique_providers": "int",
        # Provider mix (7.7)
        "primary_specialist_type": "string",
        "saw_allergist_flag": "int",
        "saw_dermatologist_flag": "int",
        "specialist_concentration": "float",
        # HCP profile columns (audit catalogue spans HCP fields too).
        "hcp_id": "string",
        "specialty": "string",
        "practice_type": "string",
        "adoption_category": "string",
    }


def _read_actual_csu_schema(path: Path) -> dict[str, str]:
    """Inspect the first record of the CSU patient_journeys JSON output."""
    if not path.exists():
        raise FileNotFoundError(path)
    import json as _json

    with path.open("r", encoding="utf-8") as f:
        records = _json.load(f)
    if not records:
        raise ValueError(f"Empty records in {path}")
    sample = records[0]
    return {col: _python_type_to_family(value) for col, value in sample.items()}


def _read_actual_optum_schema(path: Path) -> dict[str, str]:
    """Inspect the parquet schema of an Optum cohort output."""
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        import pandas as pd  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - pandas is a hard dep elsewhere
        raise RuntimeError("pandas required for --mode files") from exc

    df = pd.read_parquet(path)
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


def _python_type_to_family(value: Any) -> str:
    if value is None:
        return "string"  # default — JSON nulls are untyped
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return "string"


# --------------------------------------------------------------------------- #
# Reconciliation                                                              #
# --------------------------------------------------------------------------- #


def reconcile(
    csu_schema: dict[str, str],
    optum_schema: dict[str, str],
    catalogue: list[dict[str, Any]] | None = None,
) -> ReconciliationReport:
    """Apply the concept catalogue to the two schemas and return a report."""
    if catalogue is None:
        catalogue = CONCEPT_CATALOGUE

    mappings: list[FieldMapping] = []
    overlapping = 0
    clean = 0

    csu_columns_used: set[str] = set()
    optum_columns_used: set[str] = set()

    for entry in catalogue:
        csu_col = entry.get("csu_column")
        optum_col = entry.get("optum_column")

        csu_dtype = csu_schema.get(csu_col) if csu_col else None
        optum_dtype = optum_schema.get(optum_col) if optum_col else None

        mapping = FieldMapping(
            concept=entry["concept"],
            csu_column=csu_col,
            csu_dtype=csu_dtype,
            optum_column=optum_col,
            optum_dtype=optum_dtype,
            semantic_match=entry["semantic_match"],
            notes=entry.get("notes", ""),
        )
        mappings.append(mapping)

        if csu_col is not None and optum_col is not None:
            overlapping += 1
            if mapping.is_clean:
                clean += 1

        if csu_col:
            csu_columns_used.add(csu_col)
        if optum_col:
            optum_columns_used.add(optum_col)

    csu_only = sorted(set(csu_schema.keys()) - csu_columns_used)
    optum_only = sorted(set(optum_schema.keys()) - optum_columns_used)

    return ReconciliationReport(
        mode="synthetic",  # caller may override
        csu_columns_total=len(csu_schema),
        optum_columns_total=len(optum_schema),
        overlapping_concepts_total=overlapping,
        overlapping_concepts_clean=clean,
        field_mappings=mappings,
        csu_only_columns=csu_only,
        optum_only_columns=optum_only,
        notes="See docs/lineage/csu_field_audit.md for per-field temporal alignment.",
    )


# --------------------------------------------------------------------------- #
# Output                                                                      #
# --------------------------------------------------------------------------- #


def _print_summary(report: ReconciliationReport) -> None:
    print("=" * 72)
    print(f"Converter Schema Reconciliation — mode={report.mode}")
    print("=" * 72)
    print(f"  CSU columns total:                {report.csu_columns_total}")
    print(f"  Optum columns total:              {report.optum_columns_total}")
    print(f"  Overlapping concepts:             {report.overlapping_concepts_total}")
    print(f"  Overlapping concepts CLEAN:       {report.overlapping_concepts_clean}")
    print(
        f"  Mismatches:                       "
        f"{report.overlapping_concepts_total - report.overlapping_concepts_clean}"
    )
    print()
    print(
        f"{'Concept':<28} {'CSU col':<28} {'Optum col':<28} "
        f"{'dtype':<10} {'semantic':<10} {'clean'}"
    )
    print("-" * 116)
    for m in report.field_mappings:
        if m.csu_column is None or m.optum_column is None:
            dtype_str = "n/a"
        elif m.csu_dtype is None or m.optum_dtype is None:
            # Concept references columns not present in either input schema
            # (e.g., HCP-profile fields when only journey schemas are loaded).
            dtype_str = "absent"
        elif m.dtype_match:
            dtype_str = "match"
        else:
            dtype_str = "MISMATCH"
        print(
            f"{m.concept:<28} "
            f"{(m.csu_column or '-'):<28} "
            f"{(m.optum_column or '-'):<28} "
            f"{dtype_str:<10} "
            f"{('match' if m.semantic_match else 'NO'):<10} "
            f"{'OK' if m.is_clean else '--'}"
        )
    print()
    print(
        f"  CSU-only columns ({len(report.csu_only_columns)}):  "
        f"{', '.join(report.csu_only_columns[:6])}"
        f"{'...' if len(report.csu_only_columns) > 6 else ''}"
    )
    print(
        f"  Optum-only columns ({len(report.optum_only_columns)}): "
        f"{', '.join(report.optum_only_columns[:6])}"
        f"{'...' if len(report.optum_only_columns) > 6 else ''}"
    )
    print("=" * 72)


def write_report(report: ReconciliationReport, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    logger.info("Wrote %s", output)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconcile output schemas of convert_csu_rwd.py and convert_optum_rwd.py."
    )
    parser.add_argument(
        "--mode",
        choices=("synthetic", "files"),
        default="synthetic",
        help="synthetic: in-memory schemas; files: read from data/rwd/* outputs",
    )
    parser.add_argument(
        "--csu-input",
        type=Path,
        default=PROJECT_ROOT / "data" / "rwd" / "csu" / "e2i_ml_v3_patient_journeys.json",
        help="(--mode files) CSU patient_journeys JSON path",
    )
    parser.add_argument(
        "--optum-input",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "rwd"
            / "optum"
            / "initiation"
            / "e2i_ml_v3_patient_journeys.parquet"
        ),
        help="(--mode files) Optum patient_journeys parquet path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--no-fail-on-mismatch",
        action="store_true",
        help="Always exit 0, even if overlaps mismatch (for development).",
    )
    args = parser.parse_args(argv)

    if args.mode == "files":
        try:
            csu_schema = _read_actual_csu_schema(args.csu_input)
            optum_schema = _read_actual_optum_schema(args.optum_input)
        except FileNotFoundError as exc:
            logger.error("Input file not found: %s", exc)
            logger.error(
                "Falling back to --mode synthetic. Pass --mode synthetic explicitly to suppress."
            )
            return 2
    else:
        csu_schema = _build_synthetic_csu_schema()
        optum_schema = _build_synthetic_optum_schema()

    report = reconcile(csu_schema, optum_schema)
    report.mode = args.mode

    _print_summary(report)
    write_report(report, args.output)

    if report.has_mismatches and not args.no_fail_on_mismatch:
        logger.warning(
            "Reconciliation found %d mismatched overlapping concept(s). "
            "See %s and docs/lineage/csu_field_audit.md.",
            report.overlapping_concepts_total - report.overlapping_concepts_clean,
            args.output,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
