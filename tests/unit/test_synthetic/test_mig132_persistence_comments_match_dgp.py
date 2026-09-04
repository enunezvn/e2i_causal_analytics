"""Drift-lock: the DB column comments on persistent_180d / discontinued_180d must
describe the data the generator actually ships.

Migration 064 (M2, 2026-06-09) documented both columns as "Filtered
treatment_initiated=1". That is the RWD semantic, and only for discontinued_180d
(convert_optum_rwd.py writes it solely for initiators of its discontinuation
cohort and never writes persistent_180d at all; its persistence target is
persistent_at_180d), but NOT the synthetic DGP's:
``generate_discontinuation_outcomes`` has no treatment_initiated input and draws
an outcome for every unit as a function of ``treatment_arm``. Measured on prod
2026-09-04 (all rows synthetic): 17,186 / 17,186 treatment_initiated=0 rows carry
persistent_180d, 0 complement violations. The stale comment already produced a
wrong user-facing definition once (PR #1893, caught in review), so the
EFFECTIVE comment — the last COMMENT ON COLUMN in migration order, which is what
the DB holds — is pinned here. Pure file parse, no DB (mirrors
test_mig130_registry_presence.py).
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np

from src.ml.synthetic.generators.cohort_outcomes import generate_discontinuation_outcomes

MIGRATIONS = Path(__file__).resolve().parents[3] / "database" / "migrations"
COLUMNS = ("persistent_180d", "discontinued_180d")

_COMMENT_RE = re.compile(
    r"COMMENT\s+ON\s+COLUMN\s+(?:public\.)?patient_journeys\.(\w+)\s+IS\s+'((?:[^']|'')*)'",
    re.IGNORECASE | re.DOTALL,
)
# An ALTER TABLE patient_journeys statement (up to its ';'); DROP COLUMN clauses
# inside it clear the column comment, so the parser must model them too.
_ALTER_RE = re.compile(
    r"ALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?:ONLY\s+)?(?:public\.)?patient_journeys\b([^;]*);",
    re.IGNORECASE | re.DOTALL,
)
_DROP_RE = re.compile(r"DROP\s+COLUMN\s+(?:IF\s+EXISTS\s+)?(\w+)", re.IGNORECASE)


def _migration_order(path: Path) -> tuple[int, str]:
    m = re.match(r"(\d+)_", path.name)
    return (int(m.group(1)) if m else 10**9, path.name)


def _effective_comments(migrations: Path = MIGRATIONS) -> dict[str, tuple[str, str]]:
    """column -> (migration filename, comment text) for the LAST comment applied,
    replaying COMMENT ON COLUMN and DROP COLUMN in migration (then statement) order."""
    effective: dict[str, tuple[str, str]] = {}
    for path in sorted(migrations.glob("*.sql"), key=_migration_order):
        sql = path.read_text()
        events: list[tuple[int, str, str, str]] = [
            (m.start(), "comment", m.group(1), m.group(2)) for m in _COMMENT_RE.finditer(sql)
        ]
        for alter in _ALTER_RE.finditer(sql):
            events.extend(
                (alter.start(), "drop", d.group(1), "") for d in _DROP_RE.finditer(alter.group(1))
            )
        for _, kind, col, text in sorted(events):
            if col not in COLUMNS:
                continue
            if kind == "drop":
                effective.pop(col, None)
            else:
                effective[col] = (path.name, " ".join(text.replace("''", "'").split()))
    return effective


def test_effective_comments_no_longer_claim_an_initiator_filter():
    effective = _effective_comments()
    for col in COLUMNS:
        assert col in effective, f"no COMMENT ON COLUMN for {col} in any migration"
        source, text = effective[col]
        lowered = text.lower()
        assert "filtered treatment_initiated=1" not in lowered, (
            f"{col}: effective comment ({source}) still claims an initiator filter "
            "the synthetic generator does not apply"
        )
        assert "regardless of treatment_initiated" in lowered, (
            f"{col}: effective comment ({source}) must state the synthetic population"
        )
        assert "treatment_arm" in text, f"{col}: effective comment must name the DGP driver"


def test_effective_comments_state_the_exact_complement():
    effective = _effective_comments()
    assert "1 - discontinued_180d" in effective["persistent_180d"][1]
    assert "1 - persistent_180d" in effective["discontinued_180d"][1]


def test_effective_comments_describe_the_rwd_path_per_column():
    """The two columns differ on the RWD path: convert_optum_rwd.py writes
    discontinued_180d (initiators of its discontinuation cohort only) but never
    persistent_180d — it emits persistent_at_180d instead — so a comment that
    claims RWD rows carry persistent_180d is wrong (codex iter-1, PR #1894)."""
    converter = (MIGRATIONS.parents[1] / "scripts" / "convert_optum_rwd.py").read_text()
    assert '"discontinued_180d"' in converter
    assert '"persistent_at_180d"' in converter
    # The converter writes journey dicts with quoted string keys; only that form
    # is a write path (a docstring merely mentioning the name is not).
    assert '"persistent_180d"' not in converter, (
        "convert_optum_rwd.py now writes persistent_180d; revisit the comment"
    )
    assert re.search(
        r"_target_discontinued_180d\(patid, init_date\).{0,200}init_date is not None",
        converter,
        re.DOTALL,
    ), "discontinued_180d is no longer guarded on an initiation date; revisit the comment"

    effective = _effective_comments()
    assert "persistent_at_180d" in effective["persistent_180d"][1], (
        "persistent_180d comment must say the RWD converter writes persistent_at_180d, "
        "not this column"
    )
    assert "initiators" in effective["discontinued_180d"][1].lower()


def test_effective_comment_is_cleared_by_a_later_drop_column(tmp_path):
    """A later DROP COLUMN wipes the DB comment; the parser must not keep
    reporting the older text as effective (codex iter-3, PR #1894). A comment
    re-applied after a recreate becomes effective again."""
    (tmp_path / "001_add.sql").write_text(
        "COMMENT ON COLUMN patient_journeys.persistent_180d IS 'old text';\n"
        "COMMENT ON COLUMN patient_journeys.discontinued_180d IS 'kept';\n"
    )
    (tmp_path / "002_drop.sql").write_text(
        "ALTER TABLE public.patient_journeys\n    DROP COLUMN IF EXISTS persistent_180d;\n"
    )
    effective = _effective_comments(tmp_path)
    assert "persistent_180d" not in effective
    assert effective["discontinued_180d"] == ("001_add.sql", "kept")

    (tmp_path / "003_recomment.sql").write_text(
        "COMMENT ON COLUMN patient_journeys.persistent_180d IS 'new text';\n"
    )
    assert _effective_comments(tmp_path)["persistent_180d"] == ("003_recomment.sql", "new text")


def test_generator_draws_for_every_row_without_an_initiator_input():
    """The premise the corrected comment rests on: no initiator input, no gaps."""
    params = inspect.signature(generate_discontinuation_outcomes).parameters
    assert "treatment_initiated" not in params

    n, rng = 500, np.random.default_rng(11)
    severity = np.clip(rng.normal(5.0, 2.0, n), 0, 10)
    out = generate_discontinuation_outcomes(
        rng=rng,
        treatment_arm=rng.integers(0, 2, n),
        disease_severity=severity,
        academic_hcp=(rng.random(n) < 0.30).astype(int),
        geographic_region=rng.choice(["midwest", "northeast", "south", "west"], n),
        insurance_type=rng.choice(["commercial", "medicare", "medicaid"], n),
        age_at_diagnosis=rng.integers(18, 85, n),
        comorbidity_burden=rng.poisson(1.3, n).clip(0, 5),
        prior_therapy_lines=rng.integers(0, 4, n),
        segment=np.where(
            severity > 7,
            "high_severity",
            np.where(severity > 4, "medium_severity", "low_severity"),
        ),
        brand_cate_scale=1.0,
    )
    for col in COLUMNS:
        assert len(out[col]) == n, col
        assert set(np.unique(out[col])) <= {0, 1}, col
    assert np.array_equal(out["persistent_180d"], 1 - out["discontinued_180d"])
