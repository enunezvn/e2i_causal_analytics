"""Every ``review_type`` the code WRITES must be a member of the ``expert_review_type``
Postgres enum as the migrations define it (CREATE TYPE in 010 plus every
``ALTER TYPE ... ADD VALUE`` under database/migrations).

Why (2026-09-23, Lane B real runs): ``scripts/author_cohort_dag.py --review`` wrote
``review_type='initial_dag'`` and the structural-prior loader keys on the same value,
but the enum in prod had only dag_approval / methodology_review / quarterly_audit /
ad_hoc_validation, so both paid authoring runs failed at the insert with Postgres
22P02 and no review row exists. The scaffold's tests ran against an in-memory repo
and never met the enum. The gate had hit the SAME class earlier and was corrected to
'dag_approval' (expert_review_gate.py, C1/R6-F2). This pin makes the literal-vs-enum
drift a unit failure, not a paid-run failure.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SQL_010 = REPO / "database" / "ml" / "010_causal_validation_tables.sql"
MIGRATIONS = REPO / "database" / "migrations"

# Written review types: keyword argument / dict key / module constant forms.
_LITERAL = re.compile(r"""(?:\breview_type\s*[=:]\s*|\bREVIEW_TYPE\s*=\s*)["']([a-z_]+)["']""")


def _enum_values() -> set[str]:
    mod = importlib.import_module("scripts.validate_vocabulary_enum_sync")
    files = [
        SQL_010,
        *sorted(p for p in MIGRATIONS.glob("*.sql") if not p.name.startswith("rollback_")),
    ]
    return set(mod.extract_enum_from_sql(files, "expert_review_type"))


def _written_literals() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for root in (REPO / "src", REPO / "scripts"):
        for py in root.rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            for m in _LITERAL.finditer(text):
                found.setdefault(m.group(1), []).append(str(py.relative_to(REPO)))
    return found


def test_enum_baseline_from_010_is_intact():
    assert {
        "dag_approval",
        "methodology_review",
        "quarterly_audit",
        "ad_hoc_validation",
    } <= _enum_values()


def test_the_scan_sees_the_known_writers():
    """Teeth: the regex must see the gate's and the authoring script's writes; a scan
    that finds nothing would pass the membership test vacuously."""
    lit = _written_literals()
    assert any(p.endswith("expert_review_gate.py") for p in lit.get("dag_approval", []))
    assert any(p.endswith("author_cohort_dag.py") for p in lit.get("initial_dag", []))
    assert any(p.endswith("structural_prior_loader.py") for p in lit.get("initial_dag", []))


def test_every_written_review_type_is_an_enum_member():
    enum = _enum_values()
    lit = _written_literals()
    missing = {v: paths for v, paths in lit.items() if v not in enum}
    assert not missing, f"review_type literals with no enum member (add a migration): {missing}"
