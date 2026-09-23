#!/usr/bin/env python3
"""
Validate that database ENUMs match domain_vocabulary.yaml definitions.

This script ensures that all database ENUM types are synchronized with the
consolidated vocabulary file to prevent schema/vocabulary mismatches.

Usage:
    python -m scripts.validate_vocabulary_enum_sync

    NOTE: run it as a module (`-m`), or via pytest. Invoking it as a bare
    script (`python scripts/validate_vocabulary_enum_sync.py`) from inside a
    git worktree does NOT put the worktree root on sys.path (only the
    script's own directory is added), so `import src....` falls through to
    the venv's editable install, which resolves to wherever `pip install -e .`
    was originally run -- typically a different checkout than the worktree
    you're standing in. `-m` and pytest both add the current directory to
    sys.path and do not have this problem.

Exit Codes:
    0 - All ENUMs match vocabulary definitions
    1 - One or more ENUMs have mismatches
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import yaml

# Module-level so tests can monkeypatch the vocabulary source (e.g. to point
# at a drifted copy for a mutation test) without touching the filesystem.
PROJECT_ROOT = Path(__file__).parent.parent
VOCAB_PATH = PROJECT_ROOT / "config" / "domain_vocabulary.yaml"

# (enum_name, sql_file(s), vocab_section, vocab_key[, label]). sql_file(s) is
# a single Path for an enum whose full value set lives in one CREATE TYPE
# statement, or a list of Paths when later migrations extended it via
# `ALTER TYPE ... ADD VALUE` in a separate file (see extract_enum_from_sql).
# The optional 5th element is a display-only label for the printed report,
# for an enum whose real SQL/vocab-binding name has since been superseded
# (see the ml.gate_decision entry below); when omitted, enum_name is used.
# Module-level (like VOCAB_PATH) so tests can monkeypatch a single entry's
# binding without touching the filesystem.
EnumCheckEntry = Union[
    Tuple[str, Union[Path, Sequence[Path]], str, str],
    Tuple[str, Union[Path, Sequence[Path]], str, str, str],
]
ENUM_CHECKS: List[EnumCheckEntry] = [
    # Core schema ENUMs
    (
        "brand_type",
        PROJECT_ROOT / "database" / "core" / "e2i_ml_complete_v3_schema.sql",
        "brands",
        "values",
    ),
    (
        "region_type",
        PROJECT_ROOT / "database" / "core" / "e2i_ml_complete_v3_schema.sql",
        "regions",
        "values",
    ),
    (
        "agent_tier_type_v2",
        PROJECT_ROOT / "database" / "core" / "029_update_agent_enums_v4.sql",
        "agent_tiers",
        "values",
    ),
    (
        "agent_name_type_v3",
        PROJECT_ROOT / "database" / "core" / "029_update_agent_enums_v4.sql",
        "agents",
        "names",
    ),
    # Causal validation ENUMs
    (
        "refutation_test_type",
        # #2007/migration 138 added negative_control_outcome via ALTER TYPE
        # ... ADD VALUE in a separate file (non-transactional migrations
        # can't share a CREATE TYPE statement) -- both files are needed to
        # see the full value set.
        [
            PROJECT_ROOT / "database" / "ml" / "010_causal_validation_tables.sql",
            PROJECT_ROOT
            / "database"
            / "migrations"
            / "138_refutation_test_type_negative_control.sql",
        ],
        "refutation_test_types",
        "values",
    ),
    (
        "validation_status",
        PROJECT_ROOT / "database" / "ml" / "010_causal_validation_tables.sql",
        "validation_statuses",
        "values",
    ),
    (
        "gate_decision",
        PROJECT_ROOT / "database" / "ml" / "010_causal_validation_tables.sql",
        "gate_decisions",
        "values",
    ),
    (
        "expert_review_type",
        # Migration 152 added initial_dag (Lane B structural-author review) via
        # ALTER TYPE ... ADD VALUE in its own file, the 138 pattern.
        [
            PROJECT_ROOT / "database" / "ml" / "010_causal_validation_tables.sql",
            PROJECT_ROOT / "database" / "migrations" / "152_expert_review_type_initial_dag.sql",
        ],
        "expert_review_types",
        "values",
    ),
    # Causal discovery ENUMs (#1991 debt 4)
    (
        # Bind by the ORIGINAL type name: ml/036 renamed it to
        # public.discovery_gate_decision (same OID), but the SQL text here
        # still reads "CREATE TYPE ml.gate_decision". The 5th element is a
        # display-only label so the printed report shows the current name.
        "ml.gate_decision",
        PROJECT_ROOT / "database" / "ml" / "026_causal_discovery_tables.sql",
        "discovery_gate_decisions",
        "values",
        "ml.gate_decision (→ public.discovery_gate_decision after ml/036)",
    ),
]


class CheckResult(NamedTuple):
    """Outcome of one enum-vs-vocabulary comparison (SQL-backed or Python-enum-backed)."""

    name: str
    ok: bool
    value_count: int
    errors: List[str]
    label: str = ""  # display-only; falls back to `name` when empty (see ENUM_CHECKS)


def load_vocabulary(path: Optional[Path] = None) -> Dict:
    """Load consolidated domain vocabulary."""
    vocab_path = path or VOCAB_PATH

    if not vocab_path.exists():
        print(f"❌ ERROR: Vocabulary file not found: {vocab_path}")
        sys.exit(1)

    with open(vocab_path) as f:
        return yaml.safe_load(f)


def extract_enum_from_sql(sql_path: Union[Path, Sequence[Path]], enum_name: str) -> List[str]:
    """
    Extract ENUM values for `enum_name` from one or more SQL files.

    Unions two sources across every listed file:
      - the `CREATE TYPE <enum_name> AS ENUM (...)` values, from whichever
        listed file defines them; and
      - every `ALTER TYPE <enum_name> ADD VALUE [IF NOT EXISTS] '<value>'`
        found in any listed file (a later migration extending the enum;
        `ALTER TYPE ... ADD VALUE` is non-transactional and so is always its
        own file -- see database/migrations/138_*.sql for an example).

    Order is preserved: CREATE-statement order first, then ALTER order by
    file order/position, de-duplicated. Values are lowercased.

    Args:
        sql_path: Path to a SQL file, or a sequence of Paths.
        enum_name: Name of the ENUM type (may be schema-qualified, e.g.
            "ml.gate_decision").

    Returns:
        List of ENUM values (lowercase), in the order described above.
    """
    # A bare str is ONE path, not a sequence of characters to iterate; coerce
    # every element to Path so `.exists()` etc. work regardless of whether the
    # caller passed str, Path, or a mix of both in a list.
    paths = [Path(sql_path)] if isinstance(sql_path, (str, Path)) else [Path(p) for p in sql_path]

    values: List[str] = []
    seen = set()

    def _add(value: str) -> None:
        lowered = value.lower()
        if lowered not in seen:
            seen.add(lowered)
            values.append(lowered)

    # enum_name may be schema-qualified (e.g. "ml.gate_decision"); re.escape
    # keeps the literal "." from matching any character.
    escaped_name = re.escape(enum_name)
    create_pattern = rf"CREATE\s+TYPE\s+{escaped_name}\s+AS\s+ENUM\s*\((.*?)\);"
    # Tolerate an ALTER TYPE written with an explicit schema prefix even when
    # `enum_name` itself was passed unqualified (e.g. querying "widget_status"
    # should still match "ALTER TYPE public.widget_status ADD VALUE ...").
    alter_pattern = (
        rf"ALTER\s+TYPE\s+(?:[a-z_]+\.)?{escaped_name}\s+ADD\s+VALUE\s+"
        rf"(?:IF\s+NOT\s+EXISTS\s+)?'([^']+)'"
    )

    for path in paths:
        if not path.exists():
            continue

        with open(path) as f:
            content = f.read()

        # Drop whole-line SQL comments before matching, so a commented-out
        # example ALTER/CREATE TYPE statement (e.g. in explanatory prose)
        # can't be mistaken for a real one. A value-line trailing comment
        # (e.g. "'accept',   -- High confidence") is untouched since that
        # line doesn't START with "--".
        content = "\n".join(
            line for line in content.splitlines() if not line.lstrip().startswith("--")
        )

        create_match = re.search(create_pattern, content, re.DOTALL | re.IGNORECASE)
        if create_match:
            for value in re.findall(r"'([^']+)'", create_match.group(1)):
                _add(value)

        for value in re.findall(alter_pattern, content, re.IGNORECASE):
            _add(value)

    return values


def run_enum_checks(vocab_path: Optional[Path] = None) -> List[CheckResult]:
    """
    Compute per-enum comparison results without printing.

    Covers every entry in ENUM_CHECKS (SQL-vs-vocabulary), plus two
    Python-enum-vs-vocabulary checks: DiscoveryGateDecision (discovery gate)
    and GateDecision (refutation gate). Pure/testable -- validate_enum_sync
    does the printing on top of this.

    Args:
        vocab_path: Optional override for the vocabulary YAML.

    Returns:
        One CheckResult per enum, in ENUM_CHECKS order followed by the two
        Python-enum checks ("python:DiscoveryGateDecision", "python:GateDecision").
    """
    vocab = load_vocabulary(vocab_path)
    project_root = PROJECT_ROOT

    results: List[CheckResult] = []

    for entry in ENUM_CHECKS:
        enum_name, sql_file, vocab_section, vocab_key = entry[:4]
        label = entry[4] if len(entry) > 4 else enum_name
        sql_paths = [sql_file] if isinstance(sql_file, (str, Path)) else list(sql_file)

        # Extract ENUM values from SQL
        db_values = extract_enum_from_sql(sql_file, enum_name)

        if not db_values:
            files_display = ", ".join(str(p) for p in sql_paths)
            errors = [
                f"❌ ENUM NOT FOUND: {label}",
                f"   SQL File(s): {files_display}",
                "   Could not extract ENUM definition",
                "",
            ]
            results.append(CheckResult(enum_name, False, 0, errors, label))
            continue

        # Get vocabulary values
        vocab_data = vocab.get(vocab_section)

        if not vocab_data:
            errors = [
                f"❌ VOCAB SECTION NOT FOUND: {vocab_section}",
                f"   ENUM: {label}",
                "",
            ]
            results.append(CheckResult(enum_name, False, 0, errors, label))
            continue

        # Extract values based on vocab structure
        if vocab_key == "values":
            # Simple list of values
            if isinstance(vocab_data, dict) and "values" in vocab_data:
                vocab_values = [v.lower() for v in vocab_data["values"]]
            elif isinstance(vocab_data, list):
                vocab_values = [v.lower() for v in vocab_data]
            else:
                errors = [
                    f"❌ UNEXPECTED VOCAB STRUCTURE: {vocab_section}",
                    f"   ENUM: {label}",
                    "",
                ]
                results.append(CheckResult(enum_name, False, 0, errors, label))
                continue

        elif vocab_key == "names":
            # Agent names - extract from nested tier structure
            # agents: { tier_0_ml_foundation: [...], tier_1_coordination: [...], ... }
            if isinstance(vocab_data, dict):
                vocab_values = []
                for key, value in vocab_data.items():
                    # Skip metadata fields like 'description'
                    if key in ("description", "metadata"):
                        continue
                    # Extract agent names from tier lists
                    if isinstance(value, list):
                        vocab_values.extend([name.lower() for name in value])
            else:
                errors = [
                    f"❌ UNEXPECTED VOCAB STRUCTURE: {vocab_section}",
                    f"   ENUM: {label}",
                    "",
                ]
                results.append(CheckResult(enum_name, False, 0, errors, label))
                continue

        else:
            errors = [
                f"❌ UNKNOWN VOCAB KEY: {vocab_key}",
                f"   ENUM: {label}",
                "",
            ]
            results.append(CheckResult(enum_name, False, 0, errors, label))
            continue

        # Compare sets
        db_set = set(db_values)
        vocab_set = set(vocab_values)

        if db_set != vocab_set:
            # relative_to() raises if a bound path lies outside project_root
            # (e.g. a test rebinding a check to a tmp_path file) -- fall back
            # to the absolute path rather than crashing the whole guard over
            # a display-string nicety.
            def _display_path(p: Path) -> str:
                try:
                    return str(p.relative_to(project_root))
                except ValueError:
                    return str(p)

            files_display = ", ".join(_display_path(p) for p in sql_paths)
            errors = [
                f"❌ MISMATCH: {label}",
                f"   SQL File(s): {files_display}",
                f"   Vocab Section: {vocab_section}",
            ]

            missing_in_vocab = db_set - vocab_set
            if missing_in_vocab:
                errors.append(f"   Missing in Vocab: {sorted(missing_in_vocab)}")

            missing_in_db = vocab_set - db_set
            if missing_in_db:
                errors.append(f"   Missing in DB: {sorted(missing_in_db)}")

            errors.append("")
            results.append(CheckResult(enum_name, False, 0, errors, label))
        else:
            results.append(CheckResult(enum_name, True, len(db_values), [], label))

    # Python-enum-side checks (#1991 debt 4): the SQL/YAML checks above prove
    # the database and the vocabulary agree; this additionally proves the
    # in-process Python enums used by the discovery gate and the refutation
    # gate agree with the vocabulary. Imported lazily and only here -- any
    # submodule under src.causal_engine pulls in the whole package's
    # __init__ (expert_review_gate -> repositories -> causal_validation ->
    # refutation_runner -> mlops -> shap/great_expectations/mlflow), so this
    # costs ~15s regardless of which enum is imported first; there is no
    # "lighter" submodule that avoids it. Keeping the import out of module
    # scope keeps `import scripts.validate_vocabulary_enum_sync` itself fast.
    try:
        from src.causal_engine.discovery.base import DiscoveryGateDecision
        from src.causal_engine.refutation_runner import GateDecision
    except ImportError as exc:
        # Report both Python-side checks as explicit failures rather than
        # letting the whole guard crash with a traceback -- e.g. a stripped-
        # down environment missing a heavy optional dependency should still
        # get a readable "this check failed" report.
        import_error = [f"❌ import failed: {exc}"]
        results.append(CheckResult("python:DiscoveryGateDecision", False, 0, import_error))
        results.append(CheckResult("python:GateDecision", False, 0, import_error))
        return results

    python_enum_checks = [
        ("python:DiscoveryGateDecision", DiscoveryGateDecision, "discovery_gate_decisions"),
        ("python:GateDecision", GateDecision, "gate_decisions"),
    ]

    for enum_label, enum_cls, vocab_section in python_enum_checks:
        python_values = [member.value.lower() for member in enum_cls]

        vocab_data = vocab.get(vocab_section)
        if not vocab_data or not isinstance(vocab_data, dict) or "values" not in vocab_data:
            errors = [
                f"❌ VOCAB SECTION NOT FOUND: {vocab_section}",
                f"   Python Enum: {enum_label}",
                "",
            ]
            results.append(CheckResult(enum_label, False, 0, errors))
            continue

        vocab_values = [v.lower() for v in vocab_data["values"]]

        python_set = set(python_values)
        vocab_set = set(vocab_values)

        if python_set != vocab_set:
            errors = [
                f"❌ MISMATCH: {enum_label}",
                f"   Vocab Section: {vocab_section}",
            ]

            missing_in_vocab = python_set - vocab_set
            if missing_in_vocab:
                errors.append(f"   Missing in Vocab: {sorted(missing_in_vocab)}")

            missing_in_python = vocab_set - python_set
            if missing_in_python:
                errors.append(f"   Missing in Python Enum: {sorted(missing_in_python)}")

            errors.append("")
            results.append(CheckResult(enum_label, False, 0, errors))
        else:
            results.append(CheckResult(enum_label, True, len(python_values), []))

    return results


def validate_enum_sync(vocab_path: Optional[Path] = None) -> bool:
    """
    Validate all database ENUMs (and the discovery/refutation Python enums)
    match vocabulary definitions, printing a human-readable report.

    Args:
        vocab_path: Optional override for the vocabulary YAML (tests use this
            to point at a drifted copy without touching the real file).

    Returns:
        True if all checks pass, False otherwise.
    """
    print("=" * 80)
    print("Database ENUM Validation")
    print("=" * 80)
    print()

    results = run_enum_checks(vocab_path)

    all_errors: List[str] = []
    passed_checks = 0

    for result in results:
        if result.ok:
            display_name = result.label or result.name
            print(f"✅ {display_name:<64} ({result.value_count} values)")
            passed_checks += 1
        else:
            all_errors.extend(result.errors)

    total_checks = len(results)

    print()
    print("=" * 80)

    if all_errors:
        print(f"❌ ENUM Validation FAILED: {passed_checks}/{total_checks} checks passed")
        print("=" * 80)
        print()
        for error in all_errors:
            print(error)
        return False
    else:
        print(f"✅ All ENUMs match vocabulary definitions ({passed_checks}/{total_checks})")
        print("=" * 80)
        return True


def main():
    """Main entry point."""
    success = validate_enum_sync()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
