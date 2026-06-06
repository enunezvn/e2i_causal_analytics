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

import scripts.run_optum_tier0_test as wrapper  # noqa: E402
from src.data.manifests.resolution import resolve_manifest_source  # noqa: E402


def test_initiation_mart_cohort_registered():
    assert wrapper.COHORT_TARGETS["initiation_mart"] == "initiated_biologic_180d"
    assert wrapper.COHORT_DIR["initiation_mart"] == "data/rwd/mart/initiation"


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
