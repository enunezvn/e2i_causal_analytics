"""Lane T2: the twin's outcome is ``hcp_brand_adoption.adopted``, a DIFFERENT TABLE from
the ``business_metrics`` plant column — so it needs its own constant, not a rename.

Task 1 measured this live (docs/demos/results/2026-09-23_t2_premise_probe/): flipping
``COHORT_OUTCOME_COLUMN`` in place makes three PostgREST call sites raise
``42703 column business_metrics.adopted does not exist`` — ``cohort_loader:252``,
``digital_twin_proposals:134``, and the ``has_outcome``/``required`` reads at
``cohort_loader:197,212,230``. ``COHORT_TABLE`` is ``business_metrics``; ``adopted``
lives in ``hcp_brand_adoption`` (migration 076).

So the two names must coexist: ``COHORT_OUTCOME_COLUMN`` stays the plant column the ETL
writes (``PLANTED_COLUMNS`` -> ``business_metrics_per_hcp_etl.py:86``), and the twin reads
its own outcome through ``TWIN_OUTCOME_COLUMN`` on ``TWIN_OUTCOME_TABLE``.
"""

from __future__ import annotations

from src.data import per_hcp_cohort_columns as columns


def test_the_twin_outcome_is_its_own_constant_on_its_own_table():
    """The twin outcome is declared, and declared WITH the table it lives on — a bare
    column name would let a caller query it against business_metrics and get a 42703."""
    assert columns.TWIN_OUTCOME_COLUMN == "adopted"
    assert columns.TWIN_OUTCOME_TABLE == "hcp_brand_adoption"


def test_the_plant_column_and_the_twin_outcome_are_distinct():
    """The single-writer guard. If these ever collapse to one value, the ETL's
    PLANTED_COLUMNS starts writing the backfill-owned column."""
    assert columns.COHORT_OUTCOME_COLUMN == "cohort_conversion_outcome"
    assert columns.TWIN_OUTCOME_COLUMN != columns.COHORT_OUTCOME_COLUMN


def test_the_plant_never_writes_the_twin_outcome():
    """``adopted`` is owned by scripts/backfill_hcp_treatment_arm.py. The per-HCP ETL
    must not list it, or the two writers race on one column (the #147 failure mode)."""
    assert columns.TWIN_OUTCOME_COLUMN not in columns.PLANTED_COLUMNS
    assert columns.COHORT_OUTCOME_COLUMN in columns.PLANTED_COLUMNS


def test_the_twin_outcome_table_is_not_the_cohort_table():
    """The whole reason this is a second constant and not a rename."""
    from src.digital_twin.effect.cohort_loader import COHORT_TABLE

    assert COHORT_TABLE == "business_metrics"
    assert columns.TWIN_OUTCOME_TABLE != COHORT_TABLE
