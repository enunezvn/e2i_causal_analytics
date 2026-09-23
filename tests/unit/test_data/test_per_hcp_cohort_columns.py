"""The per-HCP cohort column contract has ONE light home.

``business_metrics`` ``per_hcp_rollup`` rows carry the Digital Twin's planted DGP: eight
treatment channels and one outcome (migrations 099 / 147). Three code paths must agree on
those names -- the plant (``scripts/backfill_segment_engagement.py``), the twin's cohort
reader (``src/digital_twin/effect``) and the per-HCP ETL's preview, which reports the
obsolete rows that still carry them (2026-09-21: a full-window backfill's reconcile deleted
such rows and the twin went dark for every brand).

The ETL cannot import the twin package: ``src.digital_twin.__init__`` pulls sklearn, dowhy and
shap (15.9 s and +507 MB, measured 2026-09-22). So the contract lives in ``src.data``, whose
modules are side-effect-free by charter, and the provider re-exports it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.data import per_hcp_cohort_columns as columns

_REPO = Path(__file__).resolve().parents[3]
_WATCHED = ("src.digital_twin", "sklearn", "pandas", "numpy", "dowhy", "shap")


def _modules_loaded_by(import_line: str) -> str:
    code = f"import sys; {import_line}; print(sorted(m for m in sys.modules if m in {_WATCHED!r}))"
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=_REPO
    ).stdout.strip()


def test_the_contract_module_imports_without_the_twin_or_any_numeric_stack():
    assert _modules_loaded_by("import src.data.per_hcp_cohort_columns") == "[]"


def test_planted_columns_are_the_eight_channels_plus_the_outcome_in_a_fixed_order():
    assert columns.COHORT_OUTCOME_COLUMN == "cohort_conversion_outcome"
    treatments = tuple(sorted(set(columns.INTERVENTION_TREATMENT_MAP.values())))
    assert len(treatments) == 8 == len(columns.INTERVENTION_TREATMENT_MAP)
    assert columns.PLANTED_COLUMNS == (*treatments, columns.COHORT_OUTCOME_COLUMN)
    assert len(set(columns.PLANTED_COLUMNS)) == len(columns.PLANTED_COLUMNS)


def test_the_provider_re_exports_the_same_objects():
    from src.digital_twin.effect import provider

    assert provider.INTERVENTION_TREATMENT_MAP is columns.INTERVENTION_TREATMENT_MAP
    assert provider.COHORT_OUTCOME_COLUMN == columns.COHORT_OUTCOME_COLUMN
    assert provider.COHORT_ESTIMABLE_INTERVENTIONS == frozenset(columns.INTERVENTION_TREATMENT_MAP)


# --------------------------------------------------------------------------- lane T1
# The adoption DGP's planted channel effects (owner decision 2026-09-23: "approve DGP
# extension, we need to recover statistical, not structural effects").


def test_adoption_channel_constants_share_the_intervention_key_set():
    assert set(columns.ADOPTION_CHANNEL_LOGIT_BETA) == set(columns.INTERVENTION_TREATMENT_MAP)
    assert set(columns.ADOPTION_CHANNEL_PLANTED_RD) == set(columns.INTERVENTION_TREATMENT_MAP)
    assert columns.ADOPTION_NULL_CHANNEL in columns.INTERVENTION_TREATMENT_MAP


def test_adoption_channel_ordering_mirrors_the_business_metrics_plant_with_one_null():
    expected_order = (
        "digital_engagement",  # engagement_score
        "speaker_program_invitation",  # speaker_program_count
        "peer_influence_activation",  # peer_influence_score
        "patient_support_program",  # patient_support_enrollment
        "email_campaign",  # email_campaign_count
        "call_frequency_increase",  # call_frequency
        "sample_distribution",  # sample_volume
        "rep_training_quality",  # rep_training_score -- the honest null
    )
    by_beta = tuple(
        sorted(
            columns.ADOPTION_CHANNEL_LOGIT_BETA,
            key=columns.ADOPTION_CHANNEL_LOGIT_BETA.get,
            reverse=True,
        )
    )
    by_rd = tuple(
        sorted(
            columns.ADOPTION_CHANNEL_PLANTED_RD,
            key=columns.ADOPTION_CHANNEL_PLANTED_RD.get,
            reverse=True,
        )
    )
    assert by_beta == expected_order
    assert by_rd == expected_order
    betas = [columns.ADOPTION_CHANNEL_LOGIT_BETA[k] for k in expected_order]
    rds = [columns.ADOPTION_CHANNEL_PLANTED_RD[k] for k in expected_order]
    assert betas[:-1] == sorted(betas[:-1], reverse=True) and len(set(betas[:-1])) == 7
    assert rds[:-1] == sorted(rds[:-1], reverse=True) and len(set(rds[:-1])) == 7
    nulls = [k for k, v in columns.ADOPTION_CHANNEL_LOGIT_BETA.items() if v == 0.0]
    assert nulls == [columns.ADOPTION_NULL_CHANNEL] == ["rep_training_quality"]
    assert columns.ADOPTION_CHANNEL_PLANTED_RD[columns.ADOPTION_NULL_CHANNEL] == 0.0


def test_adoption_channel_planted_rd_is_the_measured_0_155_per_logit_unit():
    # explore_adoption_dgp.md section 3: realised RD ~ 0.155 * beta on this DGP (0.148-0.157),
    # not the first-order 0.24 * beta. A constant edited on one side only breaks this.
    for k, beta in columns.ADOPTION_CHANNEL_LOGIT_BETA.items():
        rd = columns.ADOPTION_CHANNEL_PLANTED_RD[k]
        if beta == 0.0:
            assert rd == 0.0
        else:
            assert 0.145 <= rd / beta <= 0.16, f"{k}: rd/beta={rd / beta:.3f}"
