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

The scan is AST-based (codex r1 on PR #2246): a raw-text regex missed the dict-literal
writer (``{"review_type": "quarterly_audit"}`` in the repository's renewal path) and
matched a module docstring, so a teeth assertion could have survived the removal of
the real write. Four writer forms are covered: keyword argument ``review_type=``,
dict literal key ``"review_type"``, subscript assignment ``x["review_type"] = ``, and
the module constant ``REVIEW_TYPE = ``. Docstrings and comments are never matched.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SQL_010 = REPO / "database" / "ml" / "010_causal_validation_tables.sql"
MIGRATIONS = REPO / "database" / "migrations"


def _enum_values() -> set[str]:
    mod = importlib.import_module("scripts.validate_vocabulary_enum_sync")
    files = [
        SQL_010,
        *sorted(p for p in MIGRATIONS.glob("*.sql") if not p.name.startswith("rollback_")),
    ]
    return set(mod.extract_enum_from_sql(files, "expert_review_type"))


def _str(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _written_in(py: Path) -> list[tuple[str, int, str]]:
    """(value, line, form) for every review_type the module writes."""
    out: list[tuple[str, int, str]] = []
    tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "review_type" and (v := _str(kw.value)) is not None:
                    out.append((v, kw.value.lineno, "kwarg"))
        elif isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values, strict=True):
                if k is not None and _str(k) == "review_type" and (s := _str(v)) is not None:
                    out.append((s, v.lineno, "dict"))
        elif isinstance(node, ast.Assign):
            value = _str(node.value)
            if value is None:
                continue
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "REVIEW_TYPE":
                    out.append((value, node.lineno, "constant"))
                elif isinstance(t, ast.Subscript) and _str(t.slice) == "review_type":
                    out.append((value, node.lineno, "subscript"))
    return out


def _written_literals() -> dict[str, list[tuple[str, int, str]]]:
    found: dict[str, list[tuple[str, int, str]]] = {}
    for root in (REPO / "src", REPO / "scripts"):
        for py in sorted(root.rglob("*.py")):
            for value, line, form in _written_in(py):
                found.setdefault(value, []).append((str(py.relative_to(REPO)), line, form))
    return found


def test_enum_baseline_from_010_is_intact():
    assert {
        "dag_approval",
        "methodology_review",
        "quarterly_audit",
        "ad_hoc_validation",
    } <= _enum_values()


def test_the_scan_sees_every_known_writer_form():
    """Teeth: each writer form must be seen at its real write site; a scan that finds
    nothing (or only docstrings) would pass the membership test vacuously."""
    lit = _written_literals()
    forms = {(p, f) for entries in lit.values() for p, _, f in entries}
    assert ("src/causal_engine/expert_review_gate.py", "kwarg") in forms
    assert ("scripts/author_cohort_dag.py", "kwarg") in forms
    assert ("src/data/kg/structural_prior_loader.py", "constant") in forms
    assert ("src/repositories/expert_review.py", "dict") in forms
    # the author script's docstring mentions the value at its top; the WRITE is a call
    author = [e for e in lit["initial_dag"] if e[0] == "scripts/author_cohort_dag.py"]
    assert author and all(line > 100 for _, line, _ in author), author


def test_docstrings_and_comments_are_not_writes(tmp_path):
    src = '"""review_type=\'bogus\' in a docstring"""\n# review_type = "bogus"\nx = 1\n'
    f = tmp_path / "m.py"
    f.write_text(src)
    assert _written_in(f) == []


def test_subscript_and_dict_forms_are_seen(tmp_path):
    f = tmp_path / "m.py"
    f.write_text(
        'row = {"review_type": "a"}\nrow["review_type"] = "b"\nfn(review_type="c")\nREVIEW_TYPE = "d"\n'
    )
    assert sorted(v for v, _, _ in _written_in(f)) == ["a", "b", "c", "d"]


def test_every_written_review_type_is_an_enum_member():
    enum = _enum_values()
    missing = {v: e for v, e in _written_literals().items() if v not in enum}
    assert not missing, f"review_type written with no enum member (add a migration): {missing}"
