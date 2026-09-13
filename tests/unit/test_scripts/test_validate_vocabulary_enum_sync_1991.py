"""#1991 debt 4: the discovery gate vocabulary is under the enum-sync guard, and the guard
fails when any of its three sources drifts (mutation test)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
YAML = REPO / "config" / "domain_vocabulary.yaml"
SQL = REPO / "database" / "ml" / "026_causal_discovery_tables.sql"
SQL_010 = REPO / "database" / "ml" / "010_causal_validation_tables.sql"
SQL_138 = REPO / "database" / "migrations" / "138_refutation_test_type_negative_control.sql"


def _script():
    return importlib.import_module("scripts.validate_vocabulary_enum_sync")


def test_yaml_has_discovery_section_with_augment():
    vocab = yaml.safe_load(YAML.read_text())
    assert vocab["discovery_gate_decisions"]["values"] == ["accept", "review", "reject", "augment"]
    assert "negative_control_outcome" in vocab["refutation_test_types"]["values"]


def test_sql_extractor_reads_schema_qualified_type():
    values = _script().extract_enum_from_sql(SQL, "ml.gate_decision")
    assert values == ["accept", "review", "reject", "augment"]


def test_extractor_unions_create_and_alter_across_files():
    """refutation_test_type's full value set spans two files: the original
    CREATE TYPE in 010, and negative_control_outcome added by migration 138's
    ALTER TYPE ... ADD VALUE (non-transactional migrations can't share a
    CREATE TYPE statement, so it lives in its own file)."""
    values = _script().extract_enum_from_sql([SQL_010, SQL_138], "refutation_test_type")
    assert values == [
        "placebo_treatment",
        "random_common_cause",
        "data_subset",
        "bootstrap",
        "sensitivity_e_value",
        "negative_control_outcome",
    ]


def test_python_enum_matches_yaml():
    from src.causal_engine.discovery.base import DiscoveryGateDecision

    vocab = yaml.safe_load(YAML.read_text())
    assert [m.value for m in DiscoveryGateDecision] == vocab["discovery_gate_decisions"]["values"]


def test_guard_fails_on_drift(tmp_path):
    s = _script()
    drifted = tmp_path / "vocab.yaml"
    vocab = yaml.safe_load(YAML.read_text())
    vocab["discovery_gate_decisions"]["values"].remove("augment")
    drifted.write_text(yaml.safe_dump(vocab))
    assert s.validate_enum_sync(vocab_path=drifted) is False

    # The overall guard is already red on the repo today (pre-existing,
    # out-of-scope agent_name_type_v3 drift -- see test_guard_is_green_on_the_repo),
    # so "returns False" alone has no teeth for this mutation. Assert the
    # SPECIFIC checks this drift should break went red.
    results = {r.name: r for r in s.run_enum_checks(vocab_path=drifted)}
    assert results["ml.gate_decision"].ok is False
    assert results["python:DiscoveryGateDecision"].ok is False


def test_guard_fails_when_python_side_has_extra_value(tmp_path):
    """Mutation on the Python-enum side: give the YAML a fifth value the Python
    enum doesn't have, so the Python-vs-YAML comparison mismatches."""
    s = _script()
    drifted = tmp_path / "vocab_bogus.yaml"
    vocab = yaml.safe_load(YAML.read_text())
    vocab["discovery_gate_decisions"]["values"].append("bogus")
    drifted.write_text(yaml.safe_dump(vocab))
    assert s.validate_enum_sync(vocab_path=drifted) is False

    results = {r.name: r for r in s.run_enum_checks(vocab_path=drifted)}
    assert results["ml.gate_decision"].ok is False
    assert results["python:DiscoveryGateDecision"].ok is False


def test_guard_red_when_alter_type_migration_not_in_binding(monkeypatch):
    """If refutation_test_type were bound to ONLY 010 (missing 138's ALTER
    TYPE ... ADD VALUE), the extractor would only see five values while the
    YAML now has six (negative_control_outcome) -- the check must go red."""
    s = _script()

    def _truncate(entry):
        # entries are (name, sql_file(s), vocab_section, vocab_key[, label]);
        # only swap the sql_file(s) element, keep everything else (including
        # an optional label) as-is.
        if entry[0] == "refutation_test_type":
            return (entry[0], SQL_010) + tuple(entry[2:])
        return entry

    monkeypatch.setattr(s, "ENUM_CHECKS", [_truncate(entry) for entry in s.ENUM_CHECKS])

    results = {r.name: r for r in s.run_enum_checks()}
    assert results["refutation_test_type"].ok is False
    assert "negative_control_outcome" in " ".join(results["refutation_test_type"].errors)


def test_guard_red_when_python_enum_loses_a_member(monkeypatch):
    """Independent-source proof (codex r1 MEDIUM): mutate ONLY the Python
    enum -- the real YAML and the real SQL file are untouched -- and confirm
    the Python-vs-YAML check catches it while the SQL-vs-YAML check for the
    SAME vocab section stays green. A guard that only compared YAML-vs-YAML
    (or otherwise conflated the three sources) would not distinguish this
    from test_guard_fails_on_drift, where the YAML itself is mutated."""
    import enum

    import src.causal_engine.discovery.base as discovery_base

    # Missing "augment" relative to the real enum -- the script's lazy
    # `from src.causal_engine.discovery.base import DiscoveryGateDecision`
    # re-reads this attribute off the (already-imported, cached) module each
    # time run_enum_checks() executes, so patching the module attribute here
    # is visible to it.
    truncated_enum = enum.Enum(
        "DiscoveryGateDecision",
        {"ACCEPT": "accept", "REVIEW": "review", "REJECT": "reject"},
    )
    monkeypatch.setattr(discovery_base, "DiscoveryGateDecision", truncated_enum)

    results = {r.name: r for r in _script().run_enum_checks()}
    assert results["python:DiscoveryGateDecision"].ok is False
    assert "augment" in " ".join(results["python:DiscoveryGateDecision"].errors)
    assert results["ml.gate_decision"].ok is True


def test_guard_red_when_sql_file_loses_a_label(tmp_path, monkeypatch):
    """Independent-source proof (codex r1 MEDIUM): mutate ONLY a (tmp copy
    of the) SQL file -- the real YAML and the real Python enum are untouched
    -- and confirm the SQL-vs-YAML check catches it while the Python-vs-YAML
    check for the SAME vocab section stays green."""
    s = _script()
    original = SQL.read_text()
    mutated = original.replace(
        "        'reject',   -- Low confidence, use manual DAG\n"
        "        'augment'   -- Supplement manual DAG with high-confidence edges\n",
        "        'reject'   -- Low confidence, use manual DAG\n",
    )
    assert mutated != original, "expected 'augment' fragment not found in 026 -- SQL file changed"

    tmp_sql = tmp_path / "026_causal_discovery_tables_missing_augment.sql"
    tmp_sql.write_text(mutated)

    def _rebind(entry):
        if entry[0] == "ml.gate_decision":
            return (entry[0], tmp_sql) + tuple(entry[2:])
        return entry

    monkeypatch.setattr(s, "ENUM_CHECKS", [_rebind(entry) for entry in s.ENUM_CHECKS])

    results = {r.name: r for r in s.run_enum_checks()}
    assert results["ml.gate_decision"].ok is False
    assert "augment" in " ".join(results["ml.gate_decision"].errors)
    assert results["python:DiscoveryGateDecision"].ok is True


def test_extractor_strips_comment_lines(tmp_path):
    """A commented-out example statement must not be mistaken for a real one."""
    sql = tmp_path / "commented.sql"
    sql.write_text(
        "-- ALTER TYPE test_enum ADD VALUE 'ghost';\n"
        "CREATE TYPE test_enum AS ENUM (\n"
        "    'real'\n"
        ");\n"
    )
    values = _script().extract_enum_from_sql(sql, "test_enum")
    assert values == ["real"]
    assert "ghost" not in values


def test_extractor_alter_type_accepts_schema_prefix(tmp_path):
    """An ALTER TYPE written with an explicit schema prefix (e.g. `public.`)
    still matches when `enum_name` itself is queried unqualified."""
    sql = tmp_path / "schema_prefixed.sql"
    sql.write_text("ALTER TYPE public.widget_status ADD VALUE 'v';\n")
    values = _script().extract_enum_from_sql(sql, "widget_status")
    assert values == ["v"]


def test_extractor_alter_type_without_if_not_exists(tmp_path):
    """IF NOT EXISTS is optional in Postgres syntax and migration 138 happens
    to use it, but the extractor's (?:IF NOT EXISTS )? group must also match
    an ALTER TYPE statement that omits it."""
    sql = tmp_path / "no_if_not_exists.sql"
    sql.write_text("ALTER TYPE test_enum ADD VALUE 'v';\n")
    values = _script().extract_enum_from_sql(sql, "test_enum")
    assert values == ["v"]


def test_extractor_accepts_str_path():
    """A bare str path is ONE path, not a sequence of characters to iterate."""
    values = _script().extract_enum_from_sql(str(SQL), "ml.gate_decision")
    assert values == ["accept", "review", "reject", "augment"]


def test_python_side_checks_degrade_gracefully_on_import_error(monkeypatch):
    """If src.causal_engine can't be imported, the two Python-side checks
    show up as explicit failed CheckResults, not a crashing traceback."""
    s = _script()
    monkeypatch.setitem(sys.modules, "src.causal_engine.refutation_runner", None)
    results = {r.name: r for r in s.run_enum_checks()}
    assert results["python:DiscoveryGateDecision"].ok is False
    assert results["python:GateDecision"].ok is False
    assert "import failed" in results["python:DiscoveryGateDecision"].errors[0]
    assert "import failed" in results["python:GateDecision"].errors[0]


def test_ml_gate_decision_uses_a_readable_label_in_the_report(capsys):
    """The printed report shows the CURRENT type name, not the one ml/036
    renamed away from, even though CheckResult.name (used above) stays
    "ml.gate_decision" for the SQL regex binding. This is the GREEN (✅) line;
    see test_ml_gate_decision_label_appears_on_a_red_mismatch_line for the
    failure-path (❌ MISMATCH:) line."""
    _script().validate_enum_sync()
    captured = capsys.readouterr()
    assert "ml.gate_decision (→ public.discovery_gate_decision after ml/036)" in captured.out


def test_ml_gate_decision_label_appears_on_a_red_mismatch_line(tmp_path, monkeypatch, capsys):
    """The failure-path MISMATCH line must also use the readable label, not
    the raw binding name -- uses the same tmp-026-without-augment technique
    as test_guard_red_when_sql_file_loses_a_label."""
    s = _script()
    original = SQL.read_text()
    mutated = original.replace(
        "        'reject',   -- Low confidence, use manual DAG\n"
        "        'augment'   -- Supplement manual DAG with high-confidence edges\n",
        "        'reject'   -- Low confidence, use manual DAG\n",
    )
    assert mutated != original, "expected 'augment' fragment not found in 026 -- SQL file changed"

    tmp_sql = tmp_path / "026_missing_augment_for_label_test.sql"
    tmp_sql.write_text(mutated)

    def _rebind(entry):
        if entry[0] == "ml.gate_decision":
            return (entry[0], tmp_sql) + tuple(entry[2:])
        return entry

    monkeypatch.setattr(s, "ENUM_CHECKS", [_rebind(entry) for entry in s.ENUM_CHECKS])

    assert s.validate_enum_sync() is False
    captured = capsys.readouterr()
    assert (
        "❌ MISMATCH: ml.gate_decision (→ public.discovery_gate_decision after ml/036)"
        in captured.out
    )


def test_lane_checks_are_green_on_the_repo():
    """The checks this lane (#1991 debt 4) is responsible for are green on
    the real repo, independent of the pre-existing agent_name_type_v3 drift
    tracked separately (see test_guard_is_green_on_the_repo)."""
    results = {r.name: r for r in _script().run_enum_checks()}
    for name in (
        "ml.gate_decision",
        "gate_decision",
        "refutation_test_type",
        "python:DiscoveryGateDecision",
        "python:GateDecision",
    ):
        assert results[name].ok, f"{name}: {results[name].errors}"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "pre-existing agent_name_type_v3 drift (029 SQL vs agents section); "
        "follow-up issue -- remove this marker when fixed"
    ),
)
def test_guard_is_green_on_the_repo():
    assert _script().validate_enum_sync() is True
