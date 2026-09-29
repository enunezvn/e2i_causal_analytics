"""Migration 164 (#2325): repair the model_selector episodic rows the pre-fix hook wrote.

Every prod ``model_selection_completed`` row (188/188, read-only 2026-09-29) holds NULL
``algorithm_name`` / ``selection_score`` and the description
``Model Selection: unknown (unknown). Score: 0.00. Reason: N/A``, which the chat RAG returns
verbatim. Each row's own ``raw_content.selection_rationale`` still carries the rationale text
``Selected <algorithm> (score: <x.xxx>)`` and the primary reason, so 164 rebuilds the fields
and the description from the row itself, compare-and-set.

The pure tests run in CI. The rehearsal is real Postgres, opt-in (``E2I_DB_INTEGRATION=1``,
docker): a clone of prod's schema as it is now (``base_clone_db``, before the pending
migrations), seeded with rows in prod's shape whose rationale is produced by the REAL
``generate_rationale`` node, then 164 exactly as the runner applies it, then its rollback.
Nothing here writes to ``supabase-db``.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict

import pytest

from tests.unit.test_database.learning_loop import _pg

# The session restores prod's schema into a throwaway container; the default 30 s cap killed a
# clone drop mid-teardown on this file's first run.
pytestmark = pytest.mark.timeout(300)

KEY = "164_backfill_model_selector_episodic_selection.sql"
M164 = _pg.REPO_ROOT / "database" / "migrations" / KEY
R164 = _pg.REPO_ROOT / "database" / "migrations" / f"rollback_{KEY}"
BROKEN = "Model Selection: unknown (unknown). Score: 0.00. Reason: N/A"


def _code(path) -> str:
    return "\n".join(re.sub(r"--.*$", "", line) for line in path.read_text().splitlines())


def _rationale(name: str, score: float, primary_reason: str | None = None) -> Dict[str, Any]:
    """The selection_rationale block the agent stores, built by the real node."""
    from src.agents.ml_foundation.model_selector.nodes.algorithm_registry import (
        ALGORITHM_REGISTRY,
    )
    from src.agents.ml_foundation.model_selector.nodes.rationale_generator import (
        generate_rationale,
    )

    primary = {**ALGORITHM_REGISTRY[name], "name": name, "selection_score": score}
    out = asyncio.run(
        generate_rationale(
            {
                "primary_candidate": primary,
                "alternative_candidates": [],
                "technical_constraints": [],
                "problem_type": "binary_classification",
            }
        )
    )
    assert "error" not in out, out
    if primary_reason is not None:
        out["primary_reason"] = primary_reason
    return out


def _sql_pattern() -> str:
    (pattern,) = re.findall(r"'(\^Selected [^']+)'", _code(M164))
    return pattern


# --------------------------------------------------------------------------
# Pure (CI)
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_164_is_wrapped_and_its_number_is_unique():
    assert _pg.runner_unwraps(M164.read_text()) is False
    assert not re.search(r"^\s*(BEGIN|COMMIT|ROLLBACK)\s*;", _code(M164).upper(), re.M)
    migrations = _pg.REPO_ROOT / "database" / "migrations"
    assert [p.name for p in migrations.glob("164_*.sql")] == [KEY]
    assert R164.name.startswith("rollback_")
    assert f"filename = '{KEY}'" in _code(R164)


@pytest.mark.unit
def test_the_pattern_parses_every_algorithm_the_registry_can_select():
    """164's regex (POSIX, but the same syntax in Python) must read what the node writes."""
    from src.agents.ml_foundation.model_selector.nodes.algorithm_registry import (
        ALGORITHM_REGISTRY,
    )

    pattern = re.compile(_sql_pattern())
    for name in ALGORITHM_REGISTRY:
        text = _rationale(name, 0.7580625)["selection_rationale"]
        m = pattern.match(text)
        assert m is not None, (name, text[:60])
        assert m.groups() == (name, "0.758")


# --------------------------------------------------------------------------
# Real Postgres (opt-in)
# --------------------------------------------------------------------------

_ROWS = {
    # prod's three shapes (176 LogisticRegression, 7 XGBoost, 5 Ridge on 2026-09-29)
    "a0000000-0000-0000-0000-000000000001": ("LogisticRegression", 0.7580625, None),
    "a0000000-0000-0000-0000-000000000002": ("XGBoost", 0.8125, None),
    "a0000000-0000-0000-0000-000000000003": ("Ridge", 0.7381, ""),  # empty reason
}
FIXED = "b0000000-0000-0000-0000-000000000001"  # written by the fixed hook
OTHER_AGENT = "b0000000-0000-0000-0000-000000000002"
NO_MATCH = "b0000000-0000-0000-0000-000000000003"  # rationale text in another format
HAS_NAME = "b0000000-0000-0000-0000-000000000004"  # broken text, but a name is recorded


def _raw(name: str, score: float, reason: str | None) -> Dict[str, Any]:
    """raw_content exactly as the pre-fix hook wrote it (keys from a prod row)."""
    return {
        "experiment_id": "exp_kisq_al_20260923184034_e9f834",
        "algorithm_name": None,
        "algorithm_family": None,
        "algorithm_class": None,
        "selection_score": None,
        "selection_rationale": _rationale(name, score, reason),
        "interpretability_score": None,
        "scalability_score": None,
        "expected_performance": {},
        "alternative_candidates": [],
        "benchmark_results": {},
    }


def _lit(value: Any) -> str:
    return "'" + json.dumps(value).replace("'", "''") + "'::jsonb"


def _seed(conn) -> Dict[str, Any]:
    raws = {mid: _raw(*spec) for mid, spec in _ROWS.items()}
    no_match = _raw("Ridge", 0.7, None)
    no_match["selection_rationale"]["selection_rationale"] = "Chose Ridge because"
    has_name = {**_raw("Ridge", 0.7, None), "algorithm_name": "Ridge"}
    fixed = {**_raw("XGBoost", 0.81, None), "algorithm_name": "XGBoost", "selection_score": 0.81}
    values = [
        f"('{mid}', 'model_selector', 'model_selection_completed', '{BROKEN}', {_lit(raw)})"
        for mid, raw in raws.items()
    ] + [
        f"('{FIXED}', 'model_selector', 'model_selection_completed', "
        f"'Model Selection: XGBoost (gradient_boosting). Score: 0.810. Reason: r', {_lit(fixed)})",
        f"('{OTHER_AGENT}', 'model_trainer', 'model_training_completed', '{BROKEN}', "
        f"{_lit(raws['a0000000-0000-0000-0000-000000000001'])})",
        f"('{NO_MATCH}', 'model_selector', 'model_selection_completed', '{BROKEN}', "
        f"{_lit(no_match)})",
        f"('{HAS_NAME}', 'model_selector', 'model_selection_completed', '{BROKEN}', "
        f"{_lit(has_name)})",
    ]
    conn.execute(
        "INSERT INTO episodic_memories (memory_id, agent_name, event_type, description, "
        "raw_content) VALUES " + ",\n".join(values) + ";"
        "UPDATE episodic_memories SET embedding = array_fill(0.01::real, ARRAY[1536])::vector;",
        user="postgres",
    )
    return raws


def _state(conn) -> Dict[str, tuple]:
    rows = conn.rows(
        "select memory_id, description, raw_content::text, md5(embedding::text) "
        "from episodic_memories order by memory_id"
    )
    out = {}
    for r in rows:
        mid, desc, raw, emb = r.split("|", 3)
        out[mid] = (desc, json.loads(raw), emb)
    return out


def _apply(conn) -> str:
    return _pg.apply_migration(conn, M164, record=KEY)


def _rollback(conn) -> None:
    proc = conn.pg.run_script(conn.db, R164.read_bytes(), single_transaction=True, user="postgres")
    assert proc.returncode == 0, proc.stderr.decode()


@pytest.mark.realdb_upgrade(KEY)
def test_164_rebuilds_each_row_from_its_own_rationale(base_clone_db):
    conn = base_clone_db("m164_apply")
    raws = _seed(conn)
    before = _state(conn)

    assert _apply(conn) == "wrapped"
    after = _state(conn)

    lr, xgb, ridge = (after[mid] for mid in _ROWS)
    reason_lr = raws["a0000000-0000-0000-0000-000000000001"]["selection_rationale"][
        "primary_reason"
    ]
    assert (
        lr[0]
        == f"Model Selection: LogisticRegression (family not recorded). Score: 0.758. Reason: {reason_lr}"
    )
    assert lr[1]["algorithm_name"] == "LogisticRegression"
    assert lr[1]["selection_score"] == 0.758
    assert lr[1]["primary_reason"] == reason_lr
    assert lr[1]["algorithm_family"] is None  # not in the row, not invented
    assert lr[1]["selection_backfill"]["selection_score_decimals"] == 3
    assert xgb[1]["algorithm_name"] == "XGBoost" and xgb[1]["selection_score"] == 0.812
    # an empty stored reason reads as absent
    assert ridge[0].endswith("Score: 0.738. Reason: not recorded")
    assert ridge[1]["primary_reason"] is None
    for mid in _ROWS:  # everything else in the row is kept; the embedding is untouched
        kept = {
            k: v
            for k, v in after[mid][1].items()
            if k
            not in {"algorithm_name", "selection_score", "primary_reason", "selection_backfill"}
        }
        assert kept == {
            k: v
            for k, v in before[mid][1].items()
            if k not in {"algorithm_name", "selection_score"}
        }
        assert after[mid][2] == before[mid][2]
    for untouched in (FIXED, OTHER_AGENT, NO_MATCH, HAS_NAME):
        assert after[untouched] == before[untouched], untouched
    assert conn.rows(f"select count(*) from schema_migrations where filename = '{KEY}'") == ["1"]


@pytest.mark.realdb_upgrade(KEY)
def test_164_second_application_changes_nothing_and_the_rollback_restores(base_clone_db):
    conn = base_clone_db("m164_rollback")
    _seed(conn)
    before = _state(conn)
    _apply(conn)
    once = _state(conn)
    _pg.apply_migration(conn, M164)  # a second application
    assert _state(conn) == once

    _rollback(conn)
    assert _state(conn) == before
    assert conn.rows(f"select count(*) from schema_migrations where filename = '{KEY}'") == ["0"]
    _rollback(conn)  # a second rollback matches nothing
    assert _state(conn) == before
