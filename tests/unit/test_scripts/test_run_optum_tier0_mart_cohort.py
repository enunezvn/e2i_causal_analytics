"""Wiring test: the mart-sourced initiation cohort is a first-class runner cohort.

The Optum mart adapter writes to ``data/rwd/mart/initiation`` (a non-``optum``
path so the ``optum_mart`` feature-manifest override resolves without an M2
autodetect conflict). This pins that wiring.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import scripts.run_optum_tier0_test as wrapper  # noqa: E402
from src.data.manifests.resolution import resolve_manifest_source  # noqa: E402


def test_initiation_mart_cohort_registered():
    assert wrapper.COHORT_TARGETS["initiation_mart"] == "initiated_biologic_180d"
    assert wrapper.COHORT_DIR["initiation_mart"] == "data/rwd/mart/initiation"


def test_discontinuation_mart_cohort_registered():
    assert wrapper.COHORT_TARGETS["discontinuation_mart"] == "discontinued_180d"
    assert wrapper.COHORT_DIR["discontinuation_mart"] == "data/rwd/mart/discontinuation"


def test_persistence_mart_cohort_registered():
    assert wrapper.COHORT_TARGETS["persistence_mart"] == "persistent_at_180d"
    assert wrapper.COHORT_DIR["persistence_mart"] == "data/rwd/mart/persistence"


@pytest.mark.parametrize(
    ("cohort", "expected_target"),
    [
        ("initiation_mart", "initiated_biologic_180d"),
        ("discontinuation_mart", "discontinued_180d"),
        ("persistence_mart", "persistent_at_180d"),
    ],
)
def test_apply_overrides_sets_mart_target_outcome(cohort, expected_target):
    """apply_overrides must push the per-cohort target into tier0.CONFIG so the
    pipeline trains/evaluates on the RIGHT label (the runner-target footgun)."""
    import scripts.run_tier0_test as tier0

    wrapper.apply_overrides(cohort, wrapper.OptumTestConfig())
    assert tier0.CONFIG.target_outcome == expected_target


@pytest.mark.parametrize("cohort", ["discontinuation_mart", "persistence_mart"])
def test_mart_disc_persist_resolve_optum_mart_manifest(cohort):
    """The disc/persist mart dirs are non-'optum' paths, so the explicit
    optum_mart override resolves without an autodetect conflict (as for initiation)."""
    data_dir = wrapper.COHORT_DIR[cohort]
    assert resolve_manifest_source(data_dir, "optum_mart") == "optum_mart"


def test_convert_hint_points_to_mart_converter_for_mart_cohorts():
    """Footgun fix: a missing mart-cohort dir must suggest convert_optum_mart.py
    with the BASE cohort name + the exact output dir — NOT convert_optum_rwd.py
    with the (nonexistent) '*_mart' cohort name."""
    hint = wrapper._convert_hint("discontinuation_mart")
    assert "convert_optum_mart.py" in hint
    assert "--cohort discontinuation" in hint  # base name, no _mart suffix
    assert "--output data/rwd/mart/discontinuation" in hint
    # legacy optum cohorts still point at the raw-claims converter
    legacy = wrapper._convert_hint("discontinuation")
    assert "convert_optum_rwd.py" in legacy
    assert "--cohort discontinuation" in legacy


def test_mart_cohort_without_manifest_warns():
    """Running a *_mart cohort with no resolved feature manifest loses the Layer-5
    leakage defense-in-depth silently — the runner must warn loudly. (The
    converter's positive-enumeration is the PRIMARY defense, so this is a warning,
    not a fail-close.)"""
    warn = wrapper._mart_manifest_warning("discontinuation_mart", None)
    assert warn is not None
    assert "optum_mart" in warn and "Layer 5" in warn


def test_mart_cohort_with_manifest_no_warning():
    assert wrapper._mart_manifest_warning("discontinuation_mart", "optum_mart") is None


def test_non_mart_cohort_no_manifest_warning():
    """Legacy optum cohorts autodetect their manifest, so a None here is the
    pre-resolution state, not the mart silent-no-op trap — no mart warning."""
    assert wrapper._mart_manifest_warning("discontinuation", None) is None


def test_single_class_preflight_fails_closed(tmp_path):
    """A single-class target must fail closed with an actionable message BEFORE
    tier0's stratified split crashes cryptically downstream."""
    data_dir = tmp_path / "disc"
    data_dir.mkdir()
    pd.DataFrame({"discontinued_180d": [0, 0, 0, 0]}).to_parquet(
        data_dir / "e2i_ml_v3_patient_journeys.parquet"
    )
    err = wrapper._single_class_error(data_dir, "discontinuation_mart", "discontinued_180d")
    assert err is not None
    assert "1 class" in err and "discontinued_180d" in err


def test_two_class_preflight_passes(tmp_path):
    """A usable (>=2 class) target returns None (no error)."""
    data_dir = tmp_path / "disc"
    data_dir.mkdir()
    pd.DataFrame({"discontinued_180d": [0, 1, 0, 1]}).to_parquet(
        data_dir / "e2i_ml_v3_patient_journeys.parquet"
    )
    assert (
        wrapper._single_class_error(data_dir, "discontinuation_mart", "discontinued_180d") is None
    )


def test_preflight_defers_when_file_or_column_absent(tmp_path):
    """Missing journeys file or target column -> defer to downstream gates (None),
    honoring the converter's non-raising empty/zero-positive contract."""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert wrapper._single_class_error(empty, "discontinuation_mart", "discontinued_180d") is None
    # file present but target column absent
    pd.DataFrame({"other": [1, 2]}).to_parquet(empty / "e2i_ml_v3_patient_journeys.parquet")
    assert wrapper._single_class_error(empty, "discontinuation_mart", "discontinued_180d") is None


def test_mart_path_resolves_to_optum_mart_manifest_without_conflict():
    # autodetect finds no source on the 'mart' path, so the explicit override wins.
    assert resolve_manifest_source("data/rwd/mart/initiation", "optum_mart") == "optum_mart"
    # the legacy optum path would CONFLICT with an optum_mart override (sanity).
    import pytest

    with pytest.raises(ValueError):
        resolve_manifest_source("data/rwd/optum/initiation", "optum_mart")


def test_build_cohort_config_field_adaptive_quality_gate():
    """The harness cohort quality gate is CONFIG-driven AND field-adaptive:
    it gates on the configured threshold when ``data_quality_score`` is present,
    and is a NO-OP (no quality criterion, field not required) when the column is
    absent — so a cohort already constructed upstream cannot be zeroed out."""
    import pandas as pd

    import scripts.run_tier0_test as tier0

    # present -> gate on the configured (non-default) threshold
    df_q = pd.DataFrame({"patient_journey_id": ["PJ_1"], "data_quality_score": [0.9]})
    cfg = tier0._build_cohort_config(df_q, min_data_quality=0.42)
    dq = [c for c in cfg.inclusion_criteria if c.field == "data_quality_score"]
    assert len(dq) == 1 and dq[0].value == 0.42
    assert "data_quality_score" in cfg.required_fields

    # absent -> no-op quality gate (retains the upstream-constructed cohort)
    df_noq = pd.DataFrame({"patient_journey_id": ["PJ_1"]})
    cfg2 = tier0._build_cohort_config(df_noq, min_data_quality=0.42)
    assert all(c.field != "data_quality_score" for c in cfg2.inclusion_criteria)
    assert "data_quality_score" not in cfg2.required_fields
