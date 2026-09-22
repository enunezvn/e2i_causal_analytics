"""Migration 149 carries one column per field of the csu_escalation_causal
contract -- no more, no fewer (spec 2026-09-22 §3C.2). The synthetic backing
generator is the SSOT for the contract: a feature added to MART_SAFE_FEATURES
or a new extra column shows up here as a missing SQL column. The contract is
Lane A's (migration 148) with the treatment renamed, and the builder script
that materialises the backing derives the treatment through the mart
converter's own contrast selector.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.build_csu_escalation_synthetic_cohort import (  # noqa: E402
    GROUND_TRUTH_NAME,
    PARQUET_NAME,
    SUMMARY_NAME,
    build,
    write,
)
from src.data.manifests import MART_SAFE_FEATURES  # noqa: E402
from src.ml.synthetic.generators.csu_escalation_causal import (  # noqa: E402
    CONTRACT_COLUMNS,
    OUTCOMES,
    PLANTED_CONFOUNDERS,
    TREATMENT,
    generate_csu_escalation_cohort,
)

pytestmark = pytest.mark.unit

MIGRATION = _REPO_ROOT / "database/migrations/149_csu_escalation_causal.sql"
ROLLBACK = _REPO_ROOT / "database/migrations/rollback_149_csu_escalation_causal.sql"
TABLE = "csu_escalation_causal"
_SERVER_COLUMNS = {"created_at", "updated_at"}
_KNOWN_TYPES = {
    "TEXT",
    "VARCHAR",
    "INTEGER",
    "SMALLINT",
    "NUMERIC",
    "DATE",
    "BOOLEAN",
    "TIMESTAMPTZ",
}
_COLUMN_RE = re.compile(
    r"^\s*([a-z_][a-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*(?:\s+PRECISION)?)\b",
    re.IGNORECASE,
)
_TABLE_CONSTRAINT_PREFIXES = ("CONSTRAINT", "PRIMARY KEY", "UNIQUE", "CHECK", "FOREIGN KEY")
_N_SMALL = 400  # enough rows for both arms and every categorical level


def _sql_columns(sql: str) -> dict[str, str]:
    body = sql.split("CREATE TABLE IF NOT EXISTS", 1)[1]
    body = body[body.index("(") + 1 :]
    depth, end = 1, 0
    for i, ch in enumerate(body):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0:
            end = i
            break
    columns: dict[str, str] = {}
    for line in body[:end].splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        if stripped.upper().startswith(_TABLE_CONSTRAINT_PREFIXES):
            continue
        m = _COLUMN_RE.match(line)
        if not m:
            raise AssertionError(f"unparsed declaration line {line!r}")
        col_type = m.group(2).upper()
        if col_type not in _KNOWN_TYPES:
            raise AssertionError(f"unrecognised column type {col_type!r} on line {line!r}")
        columns[m.group(1)] = col_type
    return columns


# ---------------------------------------------------------------------------
# migration 149 <-> contract
# ---------------------------------------------------------------------------


def test_migration_149_columns_equal_the_contract():
    columns = _sql_columns(MIGRATION.read_text())
    assert set(columns) - _SERVER_COLUMNS == set(CONTRACT_COLUMNS), {
        "sql_only": sorted(set(columns) - _SERVER_COLUMNS - set(CONTRACT_COLUMNS)),
        "contract_only": sorted(set(CONTRACT_COLUMNS) - set(columns)),
    }
    # And the contract is Lane A's shape: the 64 baseline features + 4 outcomes.
    assert set(MART_SAFE_FEATURES) <= set(CONTRACT_COLUMNS)
    assert set(OUTCOMES) <= set(CONTRACT_COLUMNS)
    assert TREATMENT == "treatment_remibrutinib"
    assert "treatment_dupixent" not in columns


def test_migration_149_types_and_constraints():
    sql = MIGRATION.read_text()
    columns = _sql_columns(sql)
    assert f"CREATE TABLE IF NOT EXISTS public.{TABLE}" in sql
    assert re.search(r"^\s*patient_id\s+VARCHAR\(40\) PRIMARY KEY,", sql, re.MULTILINE)
    assert columns["is_synthetic"] == "BOOLEAN"
    assert re.search(r"^\s*is_synthetic\s+BOOLEAN NOT NULL DEFAULT false,", sql, re.MULTILINE)
    for col in (TREATMENT, *OUTCOMES):
        assert columns[col] == "SMALLINT", col
        assert f"CHECK ({col} IN (0, 1))" in sql, col
    for col in (
        "gdr_cd",
        "payer_category",
        "payer_product",
        "payer_bus",
        "charlson_risk_band",
        "elixhauser_risk_band",
        "geographic_region",
        "index_biologic_brand",
    ):
        assert columns[col] == "TEXT", col
    assert f"idx_{TABLE}_treatment" in sql and f"({TREATMENT}, is_synthetic)" in sql
    # The runner wraps each file in its own transaction: no BEGIN/COMMIT statements.
    assert not re.search(r"^\s*(BEGIN|COMMIT)\b", sql, re.MULTILINE | re.IGNORECASE)


def test_rollback_149_drops_only_this_table():
    sql = ROLLBACK.read_text()
    assert f"DROP TABLE IF EXISTS public.{TABLE};" in sql
    assert sql.count("DROP TABLE") == 1


# ---------------------------------------------------------------------------
# generator: the planted cohort's shape and truth
# ---------------------------------------------------------------------------


def test_generator_frame_matches_the_contract_and_is_deterministic():
    a, truths_a = generate_csu_escalation_cohort(n=_N_SMALL, seed=1)
    b, truths_b = generate_csu_escalation_cohort(n=_N_SMALL, seed=1)
    assert list(a.columns) == list(CONTRACT_COLUMNS)
    assert a.equals(b)
    assert truths_a == truths_b
    c, _ = generate_csu_escalation_cohort(n=_N_SMALL, seed=2)
    assert not a[TREATMENT].equals(c[TREATMENT])  # the seed is live
    assert bool(a["is_synthetic"].all())
    assert a["patient_id"].is_unique
    assert set(a[TREATMENT].unique()) == {0, 1}
    for outcome in OUTCOMES:
        assert set(a[outcome].unique()) <= {0, 1}
    # RHAPSIDO iff treated; competitors are XOLAIR / DUPIXENT.
    assert (a.loc[a[TREATMENT] == 1, "index_biologic_brand"] == "RHAPSIDO").all()
    assert set(a.loc[a[TREATMENT] == 0, "index_biologic_brand"].unique()) <= {"XOLAIR", "DUPIXENT"}
    assert a["treatment_start_date"].ge(a["index_date"]).all()
    assert a["geographic_region"].isna().any()  # exercises the __missing__ dummy


def test_generator_truth_is_recoverable_and_confounded():
    _, truths = generate_csu_escalation_cohort()
    assert set(truths) == set(OUTCOMES)
    primary = truths[OUTCOMES[0]]
    assert primary.treatment_variable == TREATMENT
    assert primary.confounders == list(PLANTED_CONFOUNDERS)
    assert 0.15 <= primary.true_ate <= 0.50  # the INDEX design band
    # The planted backdoor is REAL: the naive contrast is outside the tolerance
    # around the truth, so an unadjusted run cannot pass the recovery check.
    assert abs(primary.naive_diff - primary.true_ate) > primary.tolerance
    assert primary.is_estimate_valid(primary.true_ate + primary.tolerance / 2)
    assert not primary.is_estimate_valid(primary.naive_diff)
    cate = primary.cate_by_segment
    assert cate["high_severity"] > cate["medium_severity"] > cate["low_severity"] > 0
    assert truths["discontinued_180d"].true_ate < 0
    assert truths["biologic_switch_180d_flag"].true_ate == 0.0


def test_generator_refuses_a_tiny_cohort():
    with pytest.raises(ValueError):
        generate_csu_escalation_cohort(n=50)


# ---------------------------------------------------------------------------
# builder script: parquet + sidecar, treatment cross-checked through the mart
# converter's contrast selector
# ---------------------------------------------------------------------------


def test_builder_cross_checks_the_treatment_and_writes_the_sidecar(tmp_path):
    frame, summary = build(n=_N_SMALL, seed=3)
    assert summary["n"] == _N_SMALL and summary["columns"] == len(CONTRACT_COLUMNS)
    assert summary["arm_split"]["0"] + summary["arm_split"]["1"] == _N_SMALL
    assert dict(summary["attrition"])["in_contrast"] == _N_SMALL
    assert set(summary["ground_truth"]) == set(OUTCOMES)
    paths = write(frame, summary, tmp_path)
    assert paths["parquet"].name == PARQUET_NAME
    back = pd.read_parquet(paths["parquet"])
    assert list(back.columns) == list(CONTRACT_COLUMNS)
    assert len(back) == _N_SMALL and bool(back["is_synthetic"].all())
    truth = json.loads((tmp_path / GROUND_TRUTH_NAME).read_text())
    assert truth[OUTCOMES[0]]["treatment_variable"] == TREATMENT
    assert (tmp_path / SUMMARY_NAME).exists()


def test_builder_fails_loud_when_the_brand_and_the_arm_disagree(monkeypatch):
    """The cross-check has teeth: a generator whose brand label contradicts its
    arm must not produce a backing whose treatment silently follows one of them."""
    import scripts.build_csu_escalation_synthetic_cohort as builder

    real = builder.generate_csu_escalation_cohort

    def _corrupt(**kw):
        frame, truths = real(**kw)
        frame = frame.copy()
        first_treated = frame.index[frame[TREATMENT] == 1][0]
        frame.loc[first_treated, "index_biologic_brand"] = "XOLAIR"
        return frame, truths

    monkeypatch.setattr(builder, "generate_csu_escalation_cohort", _corrupt)
    with pytest.raises(AssertionError, match="brand-derived treatment differs"):
        builder.build(n=_N_SMALL, seed=3)
