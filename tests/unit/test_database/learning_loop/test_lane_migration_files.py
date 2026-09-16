"""Static checks on the lane's migration files (no database; runs in CI).

``scripts/run_migrations.sh`` wraps a file and its ledger row in ``--single-transaction`` unless
the file matches its un-wrap detector (``ALTER TYPE … ADD VALUE``, ``CONCURRENTLY`` or a bare
``COMMIT;``). ml/039 must take the un-wrapped branch (a new enum value cannot be used in the
transaction that adds it) and ml/040 / ml/041 the wrapped one, so neither can silently lose
atomicity. The real-runner replay is in ``test_migration_runner.py`` (opt-in).
"""

from __future__ import annotations

import ast
import re

import pytest

from tests.integration.test_migrations_no_inner_txn import _scan_for_bare_txn
from tests.unit.test_database.learning_loop import _pg

ML = _pg.REPO_ROOT / "database"
RUNNER = _pg.REPO_ROOT / "scripts" / "run_migrations.sh"


def test_runner_detector_is_the_one_the_fixture_mirrors():
    script = RUNNER.read_text()
    assert re.search(
        r"grep -qiE \\\s*\n\s*"
        + re.escape(
            '"ALTER[[:space:]]+TYPE[[:space:]].*ADD[[:space:]]+VALUE|CONCURRENTLY|'
            '^[[:space:]]*COMMIT[[:space:]]*;"'
        )
        + r" \\\s*\n\s*"
        + re.escape('<<< "$(sed \'s/--.*$//\' "$migration_file")"; then'),
        script,
    )


@pytest.mark.parametrize(
    "key, unwrapped",
    [
        ("ml/039_tool_category_cohort.sql", True),
        ("ml/040_tool_registry_startup_sync.sql", False),
        ("ml/041_composer_learning_loop_recording.sql", False),
        ("ml/043_composer_refusal_reason_codes.sql", False),
    ],
)
def test_runner_branch(key, unwrapped):
    path = ML / key
    if not path.exists():
        pytest.fail(f"{key} is missing")
    assert _pg.runner_unwraps(path.read_text()) is unwrapped


# ml/043 is deliberately NOT in _pg.LANE_MIGRATIONS: that tuple defines the shared fixture's
# "prod before this lane" base, which is 039-041's (D4, 2026-09-12).
_REASON_CODES_MIGRATION = "ml/043_composer_refusal_reason_codes.sql"


@pytest.mark.parametrize("key", [*_pg.LANE_MIGRATIONS, _REASON_CODES_MIGRATION])
def test_no_script_level_transaction_control(key):
    path = ML / key
    if not path.exists():
        pytest.fail(f"{key} is missing")
    assert _scan_for_bare_txn(path) == []


@pytest.mark.parametrize(
    "key",
    [
        "ml/040_tool_registry_startup_sync.sql",
        "ml/041_composer_learning_loop_recording.sql",
        _REASON_CODES_MIGRATION,
    ],
)
def test_wrapped_files_never_mention_the_unwrap_triggers(key):
    # Comments are stripped by the runner, but a mention in a comment invites a later edit that
    # moves it into code; keep the words out of the wrapped files entirely.
    text = (ML / key).read_text()
    assert not re.search(r"ADD\s+VALUE|CONCURRENTLY", text, re.IGNORECASE)


def test_039_is_one_idempotent_statement():
    text = (ML / "ml/039_tool_category_cohort.sql").read_text()
    code = [
        line for line in (re.sub(r"--.*$", "", raw).strip() for raw in text.splitlines()) if line
    ]
    assert code == ["ALTER TYPE tool_category ADD VALUE IF NOT EXISTS 'COHORT';"]


@pytest.mark.parametrize("name", ["rollback_040.sql", "rollback_041.sql", "rollback_043.sql"])
def test_rollbacks_are_never_auto_applied_and_hold_no_transaction_control(name):
    path = ML / "ml" / name
    assert path.exists()
    assert not [k for k in _pg.runner_migration_keys() if k.endswith(name)]
    # Applied by hand with psql --single-transaction (runbook); its own BEGIN/COMMIT would end it.
    assert _scan_for_bare_txn(path) == []


def _select_expressions(sql: str, table: str) -> tuple:
    """The INSERT column list and the SELECT expressions feeding it, split at top-level commas."""
    # Strip line comments before any scan: a "-- 043" marker would stick to a column name, and an
    # apostrophe in a comment would flip the quote state below.
    head = re.sub(r"--[^\n]*", "", sql.split(f"INSERT INTO {table} (", 1)[1])
    columns = [c.strip() for c in head.split(")", 1)[0].replace("\n", " ").split(",")]
    body = head.split("SELECT", 1)[1].split("FROM jsonb_array_elements(p_steps)", 1)[0]
    parts, depth, quoted, current = [], 0, False, []
    for ch in body:
        if ch == "'":
            quoted = not quoted
        elif not quoted and ch == "(":
            depth += 1
        elif not quoted and ch == ")":
            depth -= 1
        if ch == "," and depth == 0 and not quoted:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    parts.append("".join(current).strip())
    return columns, [re.sub(r"--[^\n]*", "", p).strip() for p in parts]


def test_043_still_writes_no_text_into_error_message():
    """D1′: ml/041's NULL in the error_message slot is the database's second guard. 043 keeps it."""
    path = ML / _REASON_CODES_MIGRATION
    if not path.exists():
        pytest.fail(f"{_REASON_CODES_MIGRATION} is missing")
    columns, expressions = _select_expressions(path.read_text(), "composition_steps")
    assert len(columns) == len(expressions), (columns, expressions)
    assert expressions[columns.index("error_message")] == "NULL"
    assert "reason_code" in columns and "reason_details" in columns


_REASON_CODES_PY = _pg.REPO_ROOT / "src/agents/tool_composer/reason_codes.py"


def test_every_reason_code_passes_the_043_format_guard():
    """The SQL guard is a format, not a member list; every Python member must satisfy it.

    All three copies are pinned: the composition_steps CHECK, the tool_performance CHECK, and
    composer_record_steps' CASE. A CASE looser than the CHECK would raise on a bad code and lose
    the whole step batch instead of storing NULL.

    Read by AST, not imported: importing the tool_composer package costs ~564 MB here.
    """
    path = ML / _REASON_CODES_MIGRATION
    if not path.exists():
        pytest.fail(f"{_REASON_CODES_MIGRATION} is missing")
    guards = re.findall(r"reason_code'?\)? ~ '([^']+)'", path.read_text())
    assert len(guards) == 3, guards
    assert len(set(guards)) == 1, guards
    enum = next(
        n
        for n in ast.walk(ast.parse(_REASON_CODES_PY.read_text()))
        if isinstance(n, ast.ClassDef) and n.name == "ReasonCode"
    )
    values = [
        n.value.value
        for n in enum.body
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
    ]
    assert len(values) >= 30
    assert [v for v in values if not re.fullmatch(guards[0], v)] == []


def test_043_reducer_keeps_exactly_the_python_detail_keys():
    """composer_structure_reason_details' key rule is Python's _DETAIL_KEY (a fullmatch), anchored."""
    path = ML / _REASON_CODES_MIGRATION
    if not path.exists():
        pytest.fail(f"{_REASON_CODES_MIGRATION} is missing")
    sql_keys = re.findall(r"e\.key ~ '([^']+)'", path.read_text())
    assert len(sql_keys) == 1, sql_keys
    python_key = next(
        n.value.args[0].value
        for n in ast.walk(ast.parse(_REASON_CODES_PY.read_text()))
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_DETAIL_KEY" for t in n.targets)
    )
    assert sql_keys[0] == f"^{python_key}$"


def _function_text(sql: str, name: str) -> str:
    """From ``CREATE OR REPLACE FUNCTION <name>(`` through the line starting ``$fn$;``."""
    start = sql.index(f"CREATE OR REPLACE FUNCTION {name}(")
    end = sql.index("\n$fn$;", start) + len("\n$fn$;")
    return sql[start:end]


def _split_top_level(text: str) -> list:
    """Split at commas outside parentheses and quotes."""
    parts, depth, quoted, current = [], 0, False, []
    for ch in text:
        if ch == "'":
            quoted = not quoted
        elif not quoted and ch == "(":
            depth += 1
        elif not quoted and ch == ")":
            depth -= 1
        if ch == "," and depth == 0 and not quoted:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return [" ".join(p.split()) for p in parts]


def _reliability_function(migration: str) -> tuple:
    """ml/043's get_tool_reliability without comments, and its output columns mapped to the
    outer SELECT's expressions (whitespace-normalized)."""
    path = ML / migration
    if not path.exists():
        pytest.fail(f"{migration} is missing")
    fn = re.sub(r"--[^\n]*", "", _function_text(path.read_text(), "get_tool_reliability"))
    columns = [
        part.split()[0]
        for part in _split_top_level(fn.split("RETURNS TABLE (", 1)[1].split("\n)\n", 1)[0])
    ]
    # The last SELECT before the registry join is the outer one; the CTEs' come first.
    select = fn.split("RETURN QUERY", 1)[1].split("FROM tool_registry tr", 1)[0]
    expressions = _split_top_level(select.rsplit("SELECT", 1)[1])
    assert len(columns) == len(expressions), (columns, expressions)
    return fn, dict(zip(columns, expressions, strict=True))


def _filter_body(column: str, expression: str) -> str:
    match = re.fullmatch(r"count\(\*\) FILTER \(WHERE (.*)\)", expression)
    assert match, (column, expression)
    return match.group(1)


def _parenthesized(text: str, opening: int) -> str:
    """The text inside the parenthesis that opens at ``text[opening]``."""
    depth = 0
    for index in range(opening, len(text)):
        depth += {"(": 1, ")": -1}.get(text[index], 0)
        if depth == 0:
            return text[opening + 1 : index]
    raise AssertionError("unbalanced parentheses")


def test_043_coded_refusal_count_is_the_refusal_count_narrowed_to_coded_rows():
    """n_refused_coded is the denominator the admin page shows beside the most common code.

    It must count the same refusals as n_refused, restricted to rows that carry a code: a
    looser filter would overstate how representative the most common code is.
    """
    _, expressions = _reliability_function(_REASON_CODES_MIGRATION)
    refused = _filter_body("n_refused", expressions["n_refused"])
    assert refused == "p.counted AND p.outcome_class IN ('refused', 'input_rejected')"
    coded = _filter_body("n_refused_coded", expressions["n_refused_coded"])
    assert coded == refused + " AND p.reason_code IS NOT NULL"


def test_043_most_common_refusal_code_and_its_count_come_from_one_ranking():
    """mode() returns only the value, so a 1-1-1 tie read like a 3-of-3 majority (#2021).

    The code and its count must come from the same ranked row, over exactly the refusals
    n_refused_coded counts: a looser WHERE would overstate the count. Ties resolve to the
    highest count, then the code in "C" order, so the answer does not depend on the locale.
    """
    fn, expressions = _reliability_function(_REASON_CODES_MIGRATION)
    assert list(expressions)[-1] == "n_most_common_refusal_reason"
    assert expressions["most_common_refusal_reason"] == "rc.reason_code::text"
    assert expressions["n_most_common_refusal_reason"] == "rc.n"

    opening = fn.index("refusal_codes AS (") + len("refusal_codes AS ")
    cte = " ".join(_parenthesized(fn, opening).split())
    where = re.search(r" WHERE (.*) GROUP BY p\.tool_id, p\.reason_code$", cte)
    assert where, cte
    assert where.group(1) == _filter_body("n_refused_coded", expressions["n_refused_coded"])
    # Per tool: without the partition, rn = 1 is one global row, so a single tool gets a code and
    # every other tool shows NULL beside n_refused_coded > 0.
    assert "row_number() OVER (PARTITION BY p.tool_id ORDER BY" in cte, cte
    assert 'ORDER BY count(*) DESC, p.reason_code COLLATE "C"' in cte, cte
    assert "LEFT JOIN refusal_codes rc ON rc.tool_id = tr.tool_id AND rc.rn = 1" in " ".join(
        fn.split()
    )


@pytest.mark.parametrize("name", ["composer_record_steps", "get_tool_reliability"])
def test_rollback_043_restores_041_verbatim(name):
    path = ML / "ml" / "rollback_043.sql"
    if not path.exists():
        pytest.fail("rollback_043.sql is missing")
    m041 = (ML / "ml/041_composer_learning_loop_recording.sql").read_text()
    rollback = path.read_text()
    assert _function_text(rollback, name) == _function_text(m041, name)
    for statement in (
        r"CREATE VIEW v_tool_reliability AS\n[^\n]*\n",
        r"COMMENT ON VIEW v_tool_reliability IS\n[^\n]*\n",
    ):
        assert re.search(statement, m041).group(0) in rollback
