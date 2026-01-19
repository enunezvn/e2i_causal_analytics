#!/usr/bin/env python3
"""
Validate that database ENUMs match domain_vocabulary.yaml definitions.

This script ensures that all database ENUM types are synchronized with the
consolidated vocabulary file to prevent schema/vocabulary mismatches.

Usage:
    python scripts/validate_vocabulary_enum_sync.py

Exit Codes:
    0 - All ENUMs match vocabulary definitions
    1 - One or more ENUMs have mismatches
"""
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def load_vocabulary() -> Dict:
    """Load consolidated domain vocabulary."""
    vocab_path = Path(__file__).parent.parent / "config" / "domain_vocabulary.yaml"

    if not vocab_path.exists():
        print(f"❌ ERROR: Vocabulary file not found: {vocab_path}")
        sys.exit(1)

    with open(vocab_path) as f:
        return yaml.safe_load(f)


def extract_enum_from_sql(sql_path: Path, enum_name: str) -> List[str]:
    """
    Extract ENUM values from SQL CREATE TYPE statement.

    Args:
        sql_path: Path to SQL file
        enum_name: Name of the ENUM type

    Returns:
        List of ENUM values (lowercase)
    """
    if not sql_path.exists():
        return []

    with open(sql_path) as f:
        content = f.read()

    # Match: CREATE TYPE enum_name AS ENUM ('value1', 'value2', ...);
    pattern = rf"CREATE\s+TYPE\s+{enum_name}\s+AS\s+ENUM\s*\((.*?)\);"
    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)

    if match:
        values_str = match.group(1)
        # Extract quoted values
        values = re.findall(r"'([^']+)'", values_str)
        return [v.lower() for v in values]

    return []


def validate_enum_sync() -> bool:
    """
    Validate all database ENUMs match vocabulary definitions.

    Returns:
        True if all ENUMs match, False otherwise
    """
    print("=" * 80)
    print("Database ENUM Validation")
    print("=" * 80)
    print()

    vocab = load_vocabulary()
    project_root = Path(__file__).parent.parent

    # Define ENUM mappings: (enum_name, sql_file, vocab_section, vocab_key)
    enum_checks: List[Tuple[str, Path, str, str]] = [
        # Core schema ENUMs
        (
            "brand_type",
            project_root / "database" / "core" / "e2i_ml_complete_v3_schema.sql",
            "brands",
            "values",
        ),
        (
            "region_type",
            project_root / "database" / "core" / "e2i_ml_complete_v3_schema.sql",
            "regions",
            "values",
        ),
        (
            "agent_tier_type_v2",
            project_root / "database" / "core" / "029_update_agent_enums_v4.sql",
            "agent_tiers",
            "values",
        ),
        (
            "agent_name_type_v3",
            project_root / "database" / "core" / "029_update_agent_enums_v4.sql",
            "agents",
            "names",
        ),

        # Causal validation ENUMs
        (
            "refutation_test_type",
            project_root / "database" / "ml" / "010_causal_validation_tables.sql",
            "refutation_test_types",
            "values",
        ),
        (
            "validation_status",
            project_root / "database" / "ml" / "010_causal_validation_tables.sql",
            "validation_statuses",
            "values",
        ),
        (
            "gate_decision",
            project_root / "database" / "ml" / "010_causal_validation_tables.sql",
            "gate_decisions",
            "values",
        ),
        (
            "expert_review_type",
            project_root / "database" / "ml" / "010_causal_validation_tables.sql",
            "expert_review_types",
            "values",
        ),
    ]

    errors = []
    total_checks = 0
    passed_checks = 0

    for enum_name, sql_file, vocab_section, vocab_key in enum_checks:
        total_checks += 1

        # Extract ENUM values from SQL
        db_values = extract_enum_from_sql(sql_file, enum_name)

        if not db_values:
            errors.append(f"❌ ENUM NOT FOUND: {enum_name}")
            errors.append(f"   SQL File: {sql_file}")
            errors.append(f"   Could not extract ENUM definition")
            errors.append("")
            continue

        # Get vocabulary values
        vocab_data = vocab.get(vocab_section)

        if not vocab_data:
            errors.append(f"❌ VOCAB SECTION NOT FOUND: {vocab_section}")
            errors.append(f"   ENUM: {enum_name}")
            errors.append("")
            continue

        # Extract values based on vocab structure
        if vocab_key == "values":
            # Simple list of values
            if isinstance(vocab_data, dict) and "values" in vocab_data:
                vocab_values = [v.lower() for v in vocab_data["values"]]
            elif isinstance(vocab_data, list):
                vocab_values = [v.lower() for v in vocab_data]
            else:
                errors.append(f"❌ UNEXPECTED VOCAB STRUCTURE: {vocab_section}")
                errors.append(f"   ENUM: {enum_name}")
                errors.append("")
                continue

        elif vocab_key == "names":
            # Agent names - extract from nested tier structure
            # agents: { tier_0_ml_foundation: [...], tier_1_coordination: [...], ... }
            if isinstance(vocab_data, dict):
                vocab_values = []
                for key, value in vocab_data.items():
                    # Skip metadata fields like 'description'
                    if key in ('description', 'metadata'):
                        continue
                    # Extract agent names from tier lists
                    if isinstance(value, list):
                        vocab_values.extend([name.lower() for name in value])
            else:
                errors.append(f"❌ UNEXPECTED VOCAB STRUCTURE: {vocab_section}")
                errors.append(f"   ENUM: {enum_name}")
                errors.append("")
                continue

        else:
            errors.append(f"❌ UNKNOWN VOCAB KEY: {vocab_key}")
            errors.append(f"   ENUM: {enum_name}")
            errors.append("")
            continue

        # Compare sets
        db_set = set(db_values)
        vocab_set = set(vocab_values)

        if db_set != vocab_set:
            errors.append(f"❌ MISMATCH: {enum_name}")
            errors.append(f"   SQL File: {sql_file.relative_to(project_root)}")
            errors.append(f"   Vocab Section: {vocab_section}")

            missing_in_vocab = db_set - vocab_set
            if missing_in_vocab:
                errors.append(f"   Missing in Vocab: {sorted(missing_in_vocab)}")

            missing_in_db = vocab_set - db_set
            if missing_in_db:
                errors.append(f"   Missing in DB: {sorted(missing_in_db)}")

            errors.append("")
        else:
            print(f"✅ {enum_name:<25} ({len(db_values)} values)")
            passed_checks += 1

    print()
    print("=" * 80)

    if errors:
        print(f"❌ ENUM Validation FAILED: {passed_checks}/{total_checks} checks passed")
        print("=" * 80)
        print()
        for error in errors:
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
