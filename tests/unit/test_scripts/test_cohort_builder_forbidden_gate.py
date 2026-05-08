"""Unit tests for the cohort-builder forbidden-column gate (Item C).

The CSU and Optum converters share a per-record drop helper that filters
post-index columns from the OUTPUT records before they hit disk. The
gate must:

1. Drop every column in the supplied ``forbidden`` list.
2. Preserve every other column unchanged.
3. NEVER mutate the input records (hash-stable for deterministic tests).
4. Preserve targets when the caller passes ``*_FORBIDDEN_NON_TARGET``
   (which excludes ``*_TARGETS``).
"""

from __future__ import annotations


def test_csu_drop_filters_forbidden_columns():
    from scripts.convert_csu_rwd import _drop_forbidden_columns
    from src.data.manifests import CSU_FORBIDDEN_NON_TARGET

    records = [
        {
            "patient_id": "P0001",
            "age_continuous": 50,
            "journey_duration_days": 120,
            "journey_status": "active",
            "journey_stage": "initial_treatment",
            "journey_start_date": "2024-01-01",
            "journey_end_date": "2024-04-30",
            "brand": "competitor",
            # Targets — must be preserved
            "treatment_initiated": 1,
            "discontinuation_flag": 0,
        }
    ]
    [out] = _drop_forbidden_columns(records, CSU_FORBIDDEN_NON_TARGET)

    # Forbidden non-target keys removed
    for key in (
        "journey_duration_days",
        "journey_status",
        "journey_stage",
        "journey_start_date",
        "journey_end_date",
        "brand",
    ):
        assert key not in out, f"{key} should have been dropped"

    # SAFE + targets preserved
    assert out["patient_id"] == "P0001"
    assert out["age_continuous"] == 50
    assert out["treatment_initiated"] == 1
    assert out["discontinuation_flag"] == 0


def test_csu_drop_does_not_mutate_input():
    from scripts.convert_csu_rwd import _drop_forbidden_columns

    original = [{"a": 1, "b": 2, "c": 3}]
    snapshot = [dict(r) for r in original]
    _drop_forbidden_columns(original, ["b"])
    assert original == snapshot, "Input was mutated"


def test_csu_drop_preserves_all_targets():
    """Every name in CSU_TARGETS survives the gate."""
    from scripts.convert_csu_rwd import _drop_forbidden_columns
    from src.data.manifests import CSU_FORBIDDEN_NON_TARGET, CSU_TARGETS

    record = {tgt: f"value_{tgt}" for tgt in CSU_TARGETS}
    record["journey_duration_days"] = 999  # forbidden
    [out] = _drop_forbidden_columns([record], CSU_FORBIDDEN_NON_TARGET)
    for tgt in CSU_TARGETS:
        assert tgt in out, f"target {tgt} dropped — gate is over-aggressive"
    assert "journey_duration_days" not in out


def test_optum_drop_filters_forbidden_columns():
    from scripts.convert_optum_rwd import _drop_forbidden_columns
    from src.data.manifests import OPTUM_FORBIDDEN_NON_TARGET

    records = [
        {
            "patient_id": "P0001",
            "age_at_index": 50,
            "journey_duration_days": 365,
            "journey_status": "active",
            "prediction_end_date": "2024-12-31",
            # Targets across cohorts — all preserved
            "treatment_initiated": 1,
            "initiated_biologic_180d": 1,
            "discontinued_180d": 0,
            "persistent_at_180d": 1,
        }
    ]
    [out] = _drop_forbidden_columns(records, OPTUM_FORBIDDEN_NON_TARGET)
    for key in ("journey_duration_days", "journey_status", "prediction_end_date"):
        assert key not in out
    for tgt in (
        "treatment_initiated",
        "initiated_biologic_180d",
        "discontinued_180d",
        "persistent_at_180d",
    ):
        assert out[tgt] == 1 or out[tgt] == 0, f"target {tgt} dropped"


def test_optum_drop_preserves_all_targets():
    from scripts.convert_optum_rwd import _drop_forbidden_columns
    from src.data.manifests import OPTUM_FORBIDDEN_NON_TARGET, OPTUM_TARGETS

    record = {tgt: f"value_{tgt}" for tgt in OPTUM_TARGETS}
    record["journey_duration_days"] = 999
    [out] = _drop_forbidden_columns([record], OPTUM_FORBIDDEN_NON_TARGET)
    for tgt in OPTUM_TARGETS:
        assert tgt in out, f"target {tgt} dropped"


def test_csu_drop_handles_empty_record_list():
    from scripts.convert_csu_rwd import _drop_forbidden_columns

    assert _drop_forbidden_columns([], ["x"]) == []


def test_csu_drop_handles_missing_forbidden_key_in_record():
    """If a record doesn't contain a forbidden key, the helper is a no-op
    on that key — no KeyError, no spurious mutation."""
    from scripts.convert_csu_rwd import _drop_forbidden_columns

    [out] = _drop_forbidden_columns([{"a": 1}], ["nonexistent"])
    assert out == {"a": 1}


# Codex Q9 — explicit coverage assertion: every column the gate is
# INTENDED to drop must appear in the corresponding FORBIDDEN_NON_TARGET
# list. Without this, a future converter change that introduces a new
# post-index column would silently bypass the gate (the manifest
# wouldn't list it; the gate's drop set wouldn't include it).

CSU_INTENDED_DROP = {
    "journey_start_date",
    "journey_end_date",
    "journey_duration_days",
    "journey_stage",
    "journey_status",
    "brand",
}

OPTUM_INTENDED_DROP = {
    "prediction_end_date",
    "journey_start_date",
    "journey_end_date",
    "journey_duration_days",
    "journey_stage",
    "journey_status",
    "brand",
}


def test_csu_intended_drop_columns_all_in_non_target():
    """Every column the CSU converter is supposed to drop at the
    boundary must appear in CSU_FORBIDDEN_NON_TARGET. A new post-index
    column added to ``_build_patient_journeys`` MUST be added to the
    manifest's ``_POST_INDEX_FORBIDDEN`` list (which feeds
    ``CSU_FORBIDDEN_AS_FEATURES`` and thus ``CSU_FORBIDDEN_NON_TARGET``)
    — otherwise the gate becomes a no-op on that new column."""
    from src.data.manifests import CSU_FORBIDDEN_NON_TARGET

    forbidden = set(CSU_FORBIDDEN_NON_TARGET)
    missing = CSU_INTENDED_DROP - forbidden
    assert not missing, (
        f"Columns the gate intends to drop are not in CSU_FORBIDDEN_NON_TARGET: "
        f"{sorted(missing)}. Add them to CSU manifest's _POST_INDEX_FORBIDDEN "
        f"FeatureContract list, or remove them from CSU_INTENDED_DROP if no "
        f"longer applicable."
    )


def test_optum_intended_drop_columns_all_in_non_target():
    """Every column the Optum converter is supposed to drop at the
    boundary must appear in OPTUM_FORBIDDEN_NON_TARGET."""
    from src.data.manifests import OPTUM_FORBIDDEN_NON_TARGET

    forbidden = set(OPTUM_FORBIDDEN_NON_TARGET)
    missing = OPTUM_INTENDED_DROP - forbidden
    assert not missing, (
        f"Columns the gate intends to drop are not in OPTUM_FORBIDDEN_NON_TARGET: "
        f"{sorted(missing)}. Add them to Optum manifest's _POST_INDEX_FORBIDDEN "
        f"FeatureContract list, or remove them from OPTUM_INTENDED_DROP if no "
        f"longer applicable."
    )
