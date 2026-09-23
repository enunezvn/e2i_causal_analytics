"""Migration 154: goldstd ``ml_experiments.brand`` is the cohort's brand (#2256).

Hermetic — reads the migration FILES, never a database (the BEGIN/apply/ROLLBACK
rehearsal on the lane proves the live shape: 8 rows -> Kisqali/Fabhalta, 2 -> NULL,
5 unchanged, re-apply = 0 rows). These tests pin the file to the code that wrote the
rows, so the derivation cannot drift from ``cohort_spec``:

* the per-brand name pattern must match every ``goldstd_experiment_name(cohort, brand)``
  the pipeline can write, and its captured token must be that brand (lower-cased), with
  the name-derived target equal to the spec's ``prediction_target``;
* the rows set to NULL must be exactly the specs with ``brand=None`` (all-brand
  cohorts), under the experiment names their runner used;
* every UPDATE is scoped to ``created_by = 'gold_standard_eval'`` and is a
  compare-and-set on the wrong constant ``'Remibrutinib'`` (idempotent, never
  overwrites a healed row);
* the rollback restores ``'Remibrutinib'`` on the same scope.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.mlops.gold_standard_eval.cohort_deployer import GOLDSTD_EXPERIMENT_NAME
from src.mlops.gold_standard_eval.cohort_spec import (
    BRANDS,
    DISCONTINUATION,
    HCP_ADOPTION_COHORT,
    INITIATION,
    PATIENT_COHORTS,
    PERSISTENCE,
    goldstd_experiment_name,
    make_hcp_spec,
    make_patient_spec,
)
from src.mlops.gold_standard_eval.run_persistence_eval import (
    DISCONTINUATION_EXPERIMENT_NAME,
    PERSISTENCE_EXPERIMENT_NAME,
)

_MIGRATIONS = Path(__file__).resolve().parents[3] / "database" / "migrations"
_MIGRATION = _MIGRATIONS / "154_goldstd_experiment_brand.sql"
_ROLLBACK = _MIGRATIONS / "rollback_154_goldstd_experiment_brand.sql"


def _code(path: Path) -> str:
    """SQL with ``--`` comment lines removed (the header prose is not executable)."""
    assert path.exists(), f"missing {path}"
    return "\n".join(
        line for line in path.read_text().splitlines() if not line.lstrip().startswith("--")
    )


def _updates(sql: str) -> list[str]:
    return re.findall(r"UPDATE\s+ml_experiments\b.*?;", sql, re.S)


def _per_brand_pattern() -> str:
    patterns = set(re.findall(r"FROM '(\^\(\?:[^']+)'\)", _code(_MIGRATION)))
    assert len(patterns) == 1, f"pre-check and UPDATE must share one name pattern: {patterns}"
    return patterns.pop()


def _null_pairs() -> set[tuple[str, str]]:
    stmt = [u for u in _updates(_code(_MIGRATION)) if re.search(r"SET\s+brand\s*=\s*NULL", u)]
    assert len(stmt) == 1, stmt
    return set(re.findall(r"\('([^']+)',\s*'([^']+)'\)", stmt[0]))


def test_per_brand_pattern_derives_every_pipeline_slot_brand_and_target():
    pattern = re.compile(_per_brand_pattern())
    slots = [(c, b, make_patient_spec(c, b)) for c in PATIENT_COHORTS for b in BRANDS]
    slots += [(HCP_ADOPTION_COHORT, b, make_hcp_spec(b)) for b in BRANDS]
    assert len(slots) == 12

    for cohort, brand, spec in slots:
        name = goldstd_experiment_name(cohort, brand)
        m = pattern.match(name)
        assert m, f"{name} not matched by the migration pattern"
        assert m.group(1) == brand.lower() == spec.brand.lower()
        # the UPDATE also requires prediction_target == name minus '_goldstd_eval_v1'
        assert name.removesuffix("_goldstd_eval_v1") == spec.target


def test_per_brand_pattern_does_not_match_the_all_brand_or_initiation_legacy_rows():
    pattern = re.compile(_per_brand_pattern())
    for name in (
        GOLDSTD_EXPERIMENT_NAME,
        PERSISTENCE_EXPERIMENT_NAME,
        DISCONTINUATION_EXPERIMENT_NAME,
    ):
        assert not pattern.match(name), name


def test_null_rows_are_exactly_the_all_brand_specs():
    expected = {
        (PERSISTENCE_EXPERIMENT_NAME, PERSISTENCE.target),
        (DISCONTINUATION_EXPERIMENT_NAME, DISCONTINUATION.target),
    }
    assert PERSISTENCE.brand is None and DISCONTINUATION.brand is None
    assert _null_pairs() == expected
    # INITIATION is single-brand Remibrutinib: its legacy row is already correct.
    assert INITIATION.brand == "Remibrutinib"
    assert GOLDSTD_EXPERIMENT_NAME not in {n for n, _t in _null_pairs()}


def test_every_update_is_scoped_and_compare_and_set_on_the_wrong_constant():
    updates = _updates(_code(_MIGRATION))
    assert len(updates) == 2, updates
    for u in updates:
        assert re.search(r"created_by\s*=\s*'gold_standard_eval'", u), u
        assert re.search(r"\bbrand\s*=\s*'Remibrutinib'", u.split("WHERE", 1)[1]), u


def test_contract_conflict_precheck_raises():
    sql = _code(_MIGRATION)
    assert "cohort_data_source" in sql and "RAISE EXCEPTION" in sql


def test_rollback_restores_remibrutinib_on_the_same_scope():
    sql = _code(_ROLLBACK)
    updates = _updates(sql)
    assert len(updates) == 1
    u = updates[0]
    assert re.search(r"SET\s+brand\s*=\s*'Remibrutinib'", u)
    assert re.search(r"created_by\s*=\s*'gold_standard_eval'", u)
    assert set(re.findall(r"\('([^']+)',\s*'([^']+)'\)", u)) == _null_pairs()
    rb_pattern = re.search(r"experiment_name\s*~\s*'([^']+)'", u)
    assert rb_pattern
    for cohort in (*PATIENT_COHORTS, HCP_ADOPTION_COHORT):
        for brand in BRANDS:
            assert re.match(rb_pattern.group(1), goldstd_experiment_name(cohort, brand))
