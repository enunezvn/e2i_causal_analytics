"""#1991 debt 4: the discovery gate vocabulary is under the enum-sync guard, and the guard
fails when any of its three sources drifts (mutation test)."""

from __future__ import annotations

import importlib
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


def test_guard_fails_on_drift(tmp_path, monkeypatch):
    s = _script()
    drifted = tmp_path / "vocab.yaml"
    vocab = yaml.safe_load(YAML.read_text())
    vocab["discovery_gate_decisions"]["values"].remove("augment")
    drifted.write_text(yaml.safe_dump(vocab))
    monkeypatch.setattr(s, "VOCAB_PATH", drifted, raising=False)
    assert s.validate_enum_sync(vocab_path=drifted) is False

    # The overall guard is already red on the repo today (pre-existing,
    # out-of-scope agent_name_type_v3 drift -- see test_guard_is_green_on_the_repo),
    # so "returns False" alone has no teeth for this mutation. Assert the
    # SPECIFIC checks this drift should break went red.
    results = {r.name: r for r in s.run_enum_checks(vocab_path=drifted)}
    assert results["ml.gate_decision"].ok is False
    assert results["python:DiscoveryGateDecision"].ok is False


def test_guard_fails_when_python_side_has_extra_value(tmp_path, monkeypatch):
    """Mutation on the Python-enum side: give the YAML a fifth value the Python
    enum doesn't have, so the Python-vs-YAML comparison mismatches."""
    s = _script()
    drifted = tmp_path / "vocab_bogus.yaml"
    vocab = yaml.safe_load(YAML.read_text())
    vocab["discovery_gate_decisions"]["values"].append("bogus")
    drifted.write_text(yaml.safe_dump(vocab))
    monkeypatch.setattr(s, "VOCAB_PATH", drifted, raising=False)
    assert s.validate_enum_sync(vocab_path=drifted) is False

    results = {r.name: r for r in s.run_enum_checks(vocab_path=drifted)}
    assert results["ml.gate_decision"].ok is False
    assert results["python:DiscoveryGateDecision"].ok is False


def test_guard_red_when_alter_type_migration_not_in_binding(monkeypatch):
    """If refutation_test_type were bound to ONLY 010 (missing 138's ALTER
    TYPE ... ADD VALUE), the extractor would only see five values while the
    YAML now has six (negative_control_outcome) -- the check must go red."""
    s = _script()
    truncated_checks = [
        (name, SQL_010, section, key)
        if name == "refutation_test_type"
        else (name, sql_file, section, key)
        for name, sql_file, section, key in s.ENUM_CHECKS
    ]
    monkeypatch.setattr(s, "ENUM_CHECKS", truncated_checks)

    results = {r.name: r for r in s.run_enum_checks()}
    assert results["refutation_test_type"].ok is False
    assert "negative_control_outcome" in " ".join(results["refutation_test_type"].errors)


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
