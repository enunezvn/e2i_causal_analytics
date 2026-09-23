"""Migration 155: the 12 goldstd registry rows record their calibration method (#2248).

Hermetic — reads the migration FILES, never a database (the BEGIN/apply/ROLLBACK
rehearsal on the PR proves the live effect). Owner decision 2026-09-23, option (a): a
retrain uses the calibration method of the model it retrains; the method of record is
``ml_model_registry.hyperparameters.calibration_method``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.mlops.gold_standard_eval.cohort_spec import BRANDS, PATIENT_COHORTS
from src.services.cohort_contract import RECORDED_CALIBRATION_METHODS, contract_from_registry_row

_DIR = Path(__file__).resolve().parents[3] / "database" / "migrations"
_MIGRATION = _DIR / "155_registry_goldstd_calibration_method.sql"
_ROLLBACK = _DIR / "rollback_155_registry_goldstd_calibration_method.sql"

_GOLDSTD = {
    f"{cohort}_{brand.lower()}_goldstd_lr_v1" for cohort in PATIENT_COHORTS for brand in BRANDS
} | {f"hcp_adoption_{brand.lower()}_goldstd_lr_v1" for brand in BRANDS}


def _body(path: Path) -> str:
    assert path.exists(), f"missing {path}"
    return "\n".join(
        line for line in path.read_text().splitlines() if not line.lstrip().startswith("--")
    )


_UPDATE_RE = re.compile(
    r"UPDATE\s+ml_model_registry\s+SET\s+(?P<set>.*?)\s+WHERE\s+(?P<where>.*?);", re.S
)


def _pins(body: str) -> list:
    """(model_name, id, trained_at) of every UPDATE; each must carry all three."""
    out = []
    for m in _UPDATE_RE.finditer(body):
        where = m.group("where")
        name = re.search(r"model_name\s*=\s*'([^']+)'", where)
        rid = re.search(r"\bid\s*=\s*'([0-9a-f-]{36})'", where)
        ts = re.search(r"trained_at\s*=\s*'([^']+)'", where)
        assert name and rid and ts, m.group(0)
        out.append((name.group(1), rid.group(1), ts.group(1)))
    return out


@pytest.mark.unit
def test_exactly_the_twelve_goldstd_rows_each_pinned_to_its_audited_registration():
    """codex r1 HIGH: names + the algorithm string are not proof of WHICH artifact;
    the (id, trained_at) pair is the audited registration (every upsert restamps
    trained_at)."""
    body = _body(_MIGRATION)
    pins = _pins(body)
    assert len(pins) == 12 == len(re.findall(r"\bUPDATE\b", body))
    assert {name for name, _, _ in pins} == _GOLDSTD
    assert len({rid for _, rid, _ in pins}) == 12
    assert not re.search(r"\b(ALTER|CREATE|DROP|DELETE|INSERT)\b", body, re.I)


@pytest.mark.unit
def test_the_value_is_the_sigmoid_the_goldstd_trainer_fits_and_the_reader_accepts():
    body = _body(_MIGRATION)
    literals = set(re.findall(r"\|\|\s*'(\{[^']*\})'::jsonb", body))
    assert len(literals) == 1, literals  # the same value on all 12 rows
    value = json.loads(literals.pop())
    assert value == {"calibration_method": "sigmoid"}
    assert value["calibration_method"] in RECORDED_CALIBRATION_METHODS
    # what the contract reader makes of a row carrying it
    assert contract_from_registry_row({"hyperparameters": value}) == {
        "calibration_method": "sigmoid"
    }


@pytest.mark.unit
def test_the_goldstd_trainer_really_fits_sigmoid():
    """The source the migration encodes: train_cohort_model's calibrator."""
    import numpy as np
    import pandas as pd

    from src.mlops.gold_standard_eval.cohort_deployer import (
        calibration_method_of,
        train_cohort_model,
    )

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60)})
    y = pd.Series((X["a"] + rng.normal(size=60) > 0).astype(int))
    assert calibration_method_of(train_cohort_model(None, X, y)) == "sigmoid"


@pytest.mark.unit
def test_it_merges_never_overwrites_and_is_scoped_to_real_calibrated_rows():
    body = _body(_MIGRATION)
    assert re.search(r"COALESCE\(hyperparameters,\s*'\{\}'::jsonb\)\s*\|\|", body)
    assert re.search(
        r"NOT\s*\(COALESCE\(hyperparameters,\s*'\{\}'::jsonb\)\s*\?\s*'calibration_method'\)", body
    )
    for m in _UPDATE_RE.finditer(body):
        assert re.search(r"COALESCE\(hyperparameters,\s*'\{\}'::jsonb\)\s*\|\|", m.group("set"))
        assert re.search(r"\?\s*'calibration_method'", m.group("where"))
        assert re.search(r"is_synthetic\s*=\s*false", m.group("where"))


@pytest.mark.unit
def test_the_rollback_removes_only_what_155_wrote():
    body = _body(_ROLLBACK)
    assert sorted(_pins(body)) == sorted(_pins(_body(_MIGRATION)))
    assert re.search(r"hyperparameters\s*-\s*'calibration_method'", body)
    assert re.search(r"hyperparameters\s*->>\s*'calibration_method'\s*=\s*'sigmoid'", body)
