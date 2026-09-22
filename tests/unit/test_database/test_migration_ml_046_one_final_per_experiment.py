"""Migration ml/046 content lock — one FINAL result row per experiment (#2206, owner fix).

WHY THIS MIGRATION EXISTS (measured 2026-09-22 on the live, self-contained prod
Supabase): ``ab_experiment_results`` had no unique key beyond its primary key, so
the "final results already computed" pre-check in ``compute_experiment_results``
and the writer's plain INSERT left a redelivery race — two late-ack deliveries
of the same FINAL task could both pass the check and both insert, and the
fidelity roll-up would then score whichever row was newest. The partial unique
index makes the claim atomic at the database. It is PARTIAL because interim and
post_hoc rows are legitimate history (``compute_experiment_results`` defaults to
interim and has no scheduled producer; an operator may recompute it more than
once per experiment).

These are text-level pins on the migration file; the faithful proof is the
BEGIN/ROLLBACK rehearsal recorded in the PR (index lands, a duplicate final
raises 23505, two interims are allowed, the original 5 indexes survive the
rollback).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATION = REPO_ROOT / "database" / "ml" / "046_ab_results_one_final_per_experiment.sql"
ROLLBACK = REPO_ROOT / "database" / "ml" / "rollback_046.sql"
INDEX = "uq_ab_results_one_final_per_experiment"
LEDGER_KEY = "ml/046_ab_results_one_final_per_experiment.sql"


def _content(path: Path = MIGRATION) -> str:
    return path.read_text()


def _stripped(text: str | None = None) -> str:
    """Mirror run_migrations.sh detection: strip ``--`` line comments first."""
    return re.sub(r"--.*$", "", text if text is not None else _content(), flags=re.MULTILINE)


def _find(pattern: str, text: str) -> int:
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    assert match is not None, f"pattern not found: {pattern!r}"
    return match.start()


# ---------------------------------------------------------------------------
# File + runner-wrappability
# ---------------------------------------------------------------------------


def test_migration_file_exists():
    assert MIGRATION.exists(), f"missing migration: {MIGRATION}"


def test_numbered_046_because_045_belongs_to_the_hpo_lane():
    """PR #2223 owns ``ml/045_persist_hpo_study_rpc.sql``; both PRs merge, so this
    file must not collide with it, and the file must say why it skipped 045."""
    assert not list((REPO_ROOT / "database" / "ml").glob("045_ab_results*"))
    assert "045" in _content(), "the migration must EXPLAIN (in a comment) why it is 046"


def test_stays_single_transaction_wrappable():
    """run_migrations.sh wraps the file in ``--single-transaction``; the dedupe
    and the index must land atomically or not at all (a dirty pre-state would
    otherwise fail the CREATE UNIQUE INDEX after the dedupe committed)."""
    stripped = _stripped()
    assert not re.search(r"ALTER\s+TYPE\s.*ADD\s+VALUE", stripped, re.IGNORECASE | re.DOTALL)
    assert not re.search(r"\bCONCURRENT" + r"LY\b", stripped, re.IGNORECASE)
    assert not re.search(r"^\s*(COMMIT|BEGIN)\s*;", stripped, re.IGNORECASE | re.MULTILINE)


def test_negative_detectors_are_live():
    """Positive control for the absence assertions above."""
    assert re.search(r"^\s*COMMIT\s*;", _stripped("x;\nCOMMIT;"), re.IGNORECASE | re.MULTILINE)
    assert re.search(r"\bCONCURRENT" + r"LY\b", _stripped("CREATE INDEX CONCURRENTLY x"))


# ---------------------------------------------------------------------------
# The index: partial, idempotent, on the right key
# ---------------------------------------------------------------------------


def test_unique_index_is_partial_on_final_only():
    """A table-wide key on (experiment_id, analysis_type) would forbid the
    legitimate repeated interim / post_hoc rows; the predicate scopes the
    singleton rule to ``final`` alone."""
    stripped = _stripped()
    start = _find(
        rf"CREATE\s+UNIQUE\s+INDEX\s+IF\s+NOT\s+EXISTS\s+{INDEX}\s+"
        r"ON\s+public\.ab_experiment_results\s*\(\s*experiment_id\s*\)",
        stripped,
    )
    statement = stripped[start : stripped.find(";", start)]
    assert re.search(r"WHERE\s+analysis_type\s*=\s*'final'", statement, re.IGNORECASE), (
        "the unique index must carry the analysis_type = 'final' predicate"
    )


def test_predicate_literal_matches_the_python_enum():
    """The Python writer decides "is this a FINAL row" by ``AnalysisType.FINAL``;
    the SQL predicate must be the same literal or the writer's 23505 handling
    and the database's rule disagree."""
    from src.services.results_analysis import AnalysisType

    assert f"'{AnalysisType.FINAL.value}'" == "'final'"
    assert re.search(
        rf"WHERE\s+analysis_type\s*=\s*'{AnalysisType.FINAL.value}'", _stripped(), re.IGNORECASE
    )


def test_writer_table_name_matches():
    from src.repositories.ab_results import ABResultsRepository

    assert ABResultsRepository.table_name == "ab_experiment_results"
    assert f"public.{ABResultsRepository.table_name}" in _stripped()


def test_history_rationale_is_documented():
    """The 119 precedent: a partial unique index must SAY why it is partial."""
    content = _content()
    assert re.search(r"interim", content, re.IGNORECASE)
    assert re.search(r"post_hoc", content, re.IGNORECASE)
    assert re.search(r"history", content, re.IGNORECASE)


# ---------------------------------------------------------------------------
# The guarded dedupe: repoint BEFORE delete, delete BEFORE the index
# ---------------------------------------------------------------------------


def test_dedupe_repoints_fidelity_comparisons_before_deleting_losers():
    """``ab_fidelity_comparisons.results_id`` is ``ON DELETE SET NULL``: deleting
    a duplicate final row would silently null a comparison's audit pointer. The
    migration must move those pointers to the surviving row FIRST."""
    stripped = _stripped()
    repoint_at = _find(
        r"UPDATE\s+public\.ab_fidelity_comparisons\s+\w*\s*SET\s+results_id\s*=", stripped
    )
    delete_at = _find(r"DELETE\s+FROM\s+public\.ab_experiment_results", stripped)
    index_at = _find(rf"CREATE\s+UNIQUE\s+INDEX\s+IF\s+NOT\s+EXISTS\s+{INDEX}", stripped)
    assert repoint_at < delete_at < index_at


def test_dedupe_keeps_the_earliest_computed_row():
    stripped = _stripped()
    _find(r"DISTINCT\s+ON\s*\(\s*experiment_id\s*\)", stripped)
    _find(r"ORDER\s+BY\s+experiment_id\s*,\s*computed_at\s+ASC", stripped)


def test_dedupe_is_scoped_to_final_rows():
    """Interim/post_hoc rows must never be deduped — they are the history the
    partial index exists to protect."""
    stripped = _stripped()
    delete_at = _find(r"DELETE\s+FROM\s+public\.ab_experiment_results", stripped)
    delete_stmt = stripped[delete_at : stripped.find(";", delete_at)]
    # The losers set is built from final rows only; the DELETE joins on it.
    assert re.search(r"losers", delete_stmt, re.IGNORECASE)
    losers_at = _find(r"losers\s+AS\s*\(", stripped)
    losers_def = stripped[losers_at : stripped.find(")", losers_at)]
    assert re.search(r"analysis_type\s*=\s*'final'", losers_def, re.IGNORECASE)


def test_dedupe_is_skipped_once_the_index_exists():
    """Second apply (deploy re-scour) must be a no-op: with the index in place
    there cannot be duplicates, and the dedupe must not even run."""
    _find(rf"to_regclass\(\s*'public\.{INDEX}'\s*\)\s+IS\s+NOT\s+NULL", _stripped())


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


def test_rollback_file_drops_the_index_and_the_ledger_row():
    assert ROLLBACK.exists(), f"missing rollback: {ROLLBACK}"
    stripped = _stripped(_content(ROLLBACK))
    _find(rf"DROP\s+INDEX\s+IF\s+EXISTS\s+public\.{INDEX}", stripped)
    _find(
        rf"DELETE\s+FROM\s+public\.schema_migrations\s+WHERE\s+filename\s*=\s*'{LEDGER_KEY}'",
        stripped,
    )


def test_rollback_says_the_dedupe_is_not_reversible():
    """Deleted duplicate finals are gone; the rollback must say so rather than
    imply a full reverse."""
    assert re.search(r"dedup|duplicate", _content(ROLLBACK), re.IGNORECASE)
    assert re.search(r"not\s+revers|cannot\s+be\s+re|irreversib", _content(ROLLBACK), re.IGNORECASE)


def test_rollback_is_not_a_forward_migration():
    """run_migrations.sh skips ``rollback_*.sql``; the name must keep that prefix."""
    assert ROLLBACK.name.startswith("rollback_")
