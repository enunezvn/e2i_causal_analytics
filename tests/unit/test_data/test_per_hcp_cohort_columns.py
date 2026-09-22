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
