"""Migration 119 content lock — validation_status semantics pin (#1352).

THE RULING (owner decision, 2026-07-30): STRONG semantics + dual evidence.
``causal_paths.validation_status = 'validated'`` is pinned to mean
"RefutationSuite evidence exists and passed", enforced in the schema. The
2,729 currently-'validated' synthetic paths KEEP their status with consistent
content-addressed synthetic refutation evidence seeded behind them; real paths
enter as 'pending' and only the RefutationNode promotes them (separate lane).

HARD INVARIANT: ``src/api/routes/causal.py`` (get_causal_value_chains, the
Home dashboard) filters ``.eq("validation_status", "validated")``. There must
NEVER be a moment — mid-migration or after — where validated rows lack
evidence or the dashboard's row count blinks. These text-level assertions lock
the two mechanisms that guarantee that:

1. run_migrations.sh wraps the file in ``--single-transaction`` (so seed +
   gate + constraint land atomically or not at all) — which only holds if the
   file avoids the runner's un-wrappable patterns;
2. within the file, evidence seeding MUST precede the enforcement DDL, with an
   abort-gate between them.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATION = REPO_ROOT / "database" / "migrations" / "119_validation_status_semantics.sql"


def _content() -> str:
    return MIGRATION.read_text()


def _stripped() -> str:
    """Mirror run_migrations.sh detection: strip -- line comments first."""
    return re.sub(r"--.*$", "", _content(), flags=re.MULTILINE)


def test_migration_file_exists():
    assert MIGRATION.exists(), f"missing migration: {MIGRATION}"


def test_stays_single_transaction_wrappable():
    """The atomicity design rides run_migrations.sh's --single-transaction
    wrapper. Any of these patterns (checked on comment-stripped content, as the
    runner does) would un-wrap the file and break the no-blink invariant."""
    stripped = _stripped()
    assert not re.search(r"ALTER\s+TYPE\s.*ADD\s+VALUE", stripped, re.IGNORECASE | re.DOTALL), (
        "ALTER TYPE ... ADD VALUE is non-transactional — would un-wrap the migration"
    )
    assert not re.search(r"\bCONCURRENT" + r"LY\b", stripped, re.IGNORECASE), (
        "the runner un-wraps on this keyword anywhere in the body"
    )
    assert not re.search(r"^\s*COMMIT\s*;", stripped, re.IGNORECASE | re.MULTILINE), (
        "a self-managed COMMIT ends the wrapper transaction early"
    )


def test_seed_backfill_precedes_enforcement_ddl():
    """Order is the point: seed evidence for existing validated rows FIRST,
    then the abort-gate, then constraint + trigger. Reversed order would leave
    a window where 'validated' rows violate the pinned semantics."""
    content = _content()
    seed_at = content.find("seed_synthetic_refutation_evidence")
    backfill_at = content.find("-- 3) BACKFILL")
    gate_at = content.find("-- 4) ATOMICITY GATE")
    constraint_at = content.find("ADD CONSTRAINT causal_paths_validation_status_domain_chk")
    trigger_at = content.find("CREATE TRIGGER trg_causal_paths_validated_evidence")
    assert -1 not in {seed_at, backfill_at, gate_at, constraint_at, trigger_at}, (
        "expected sections missing (seed fn / backfill / gate / constraint / trigger)"
    )
    assert seed_at < backfill_at < gate_at < constraint_at, (
        "evidence seeding and the abort-gate must precede the domain constraint"
    )
    assert gate_at < trigger_at, "the abort-gate must precede trigger installation"


def test_atomicity_gate_aborts_on_unbacked_validated_rows():
    content = _content()
    gate = content[content.find("-- 4) ATOMICITY GATE") :]
    assert "RAISE EXCEPTION" in gate
    assert "validation_status = 'validated'" in gate
    assert "NOT EXISTS" in gate


def test_estimate_id_namespace_matches_python_helper():
    """The SQL literal and the Python constant must be the same string — the
    cross-language uuid5 pin (see test_causal_validation_estimate_id.py)."""
    from src.repositories.causal_validation import CAUSAL_PATH_ESTIMATE_NAMESPACE

    assert f"'{CAUSAL_PATH_ESTIMATE_NAMESPACE}'" in _content()


def test_uuid_functions_schema_qualified():
    """uuid-ossp lives in the ``extensions`` schema on this Supabase; the
    trigger fires under arbitrary caller search_paths (PostgREST roles), so
    unqualified calls could break at runtime."""
    content = _content()
    assert "extensions.uuid_generate_v5" in content
    assert "extensions.uuid_ns_url()" in content


def test_new_paths_default_pending_not_null():
    content = _content()
    assert re.search(
        r"ALTER\s+COLUMN\s+validation_status\s+SET\s+DEFAULT\s+'pending'", content, re.IGNORECASE
    )
    assert re.search(
        r"ALTER\s+COLUMN\s+validation_status\s+SET\s+NOT\s+NULL", content, re.IGNORECASE
    )


def test_domain_check_covers_all_code_referenced_values():
    """Values referenced by live code: 'validated' (DGP + causal.py gate),
    'pending' (new default), 'needs_review'/'pending' (src/ml/data_generator),
    'overturned' (memory consolidator skip-rule), 'refuted' (the demotion
    value the RefutationNode lane needs)."""
    content = _content()
    m = re.search(
        r"CHECK\s*\(\s*validation_status\s+IN\s*\(([^)]*)\)", content, re.IGNORECASE | re.DOTALL
    )
    assert m, "domain CHECK constraint missing"
    values = set(re.findall(r"'([a-z_]+)'", m.group(1)))
    assert values == {"pending", "validated", "needs_review", "overturned", "refuted"}


def test_trigger_autoseeds_synthetic_and_rejects_real():
    content = _content()
    fn_at = content.find("enforce_validated_requires_refutation_evidence")
    assert fn_at != -1
    fn_body = content[fn_at:]
    assert "NEW.is_synthetic" in fn_body
    assert "seed_synthetic_refutation_evidence" in fn_body
    assert "RAISE EXCEPTION" in fn_body
    assert "check_violation" in fn_body
    # 'pending' promotion path is the RefutationNode's job — the hint must say so.
    assert "RefutationNode" in fn_body


def test_seeded_evidence_is_content_addressed_and_labeled_synthetic():
    """No invented random numbers: every metric must derive from the path row
    (md5 content hash -> deterministic unit fraction), and every seeded row
    must carry explicit synthetic provenance."""
    content = _content()
    assert "md5(" in content
    assert re.search(r"\brandom\s*\(", _stripped()) is None, (
        "seeded evidence must be deterministic — no random()"
    )
    assert "'is_synthetic', true" in content
    assert "dgp_backfill_migration_119" in content
    assert "'content_hash'" in content
    # analysis_context labels the rows as NOT real RefutationSuite output.
    assert "analysis_context" in content


def test_seeded_metrics_consistent_with_pass_thresholds():
    """The seeded pseudo-metrics must be CONSISTENT with status='passed' under
    RefutationRunner.PASS_THRESHOLDS (placebo p>0.05, deltas <20%, e-value>=2)
    — evidence that contradicts its own verdict would be plausible-wrong."""
    content = _content()
    assert "'passed'" in content
    assert "'proceed'" in content
    for test_type in (
        "placebo_treatment",
        "random_common_cause",
        "data_subset",
        "bootstrap",
        "sensitivity_e_value",
    ):
        assert test_type in content, f"missing seeded test_type {test_type}"


def test_column_semantics_documented():
    content = _content()
    assert re.search(
        r"COMMENT\s+ON\s+COLUMN\s+(public\.)?causal_paths\.validation_status",
        content,
        re.IGNORECASE,
    )
