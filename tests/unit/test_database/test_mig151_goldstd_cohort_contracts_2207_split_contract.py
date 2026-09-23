"""Migration 151: the goldstd registry rows get their cohort contract — only what is provable.

Hermetic — reads the migration FILE, never a database (the BEGIN/apply/ROLLBACK rehearsal
in the PR proves the live shape; these tests prove the file is SHAPED to be safe).

Owner decision (2026-09-23, decision 3): fill ONLY what is provable from the training
code; the rest stays NULL and is listed in the migration comment. Provable
(``src/mlops/gold_standard_eval/cohort_spec.py`` + ``feature_builder.py``):

* 9 patient rows ``<cohort>_<brand>_goldstd_lr_v1``: trained on ``patient_journeys``
  with ``brand=<brand>`` AND ``is_synthetic=True``, label ``_PATIENT_LABELS[cohort]``,
  covariates ``_PATIENT_COVARIATES[cohort]``. The live table carries ``data_split`` and
  ``training_samples`` == train+validation per brand EXACTLY (7085/7201/7145), so the
  table + brand identity is arithmetic, not inference. The contract MUST carry
  ``columns``: ``days_to_treatment`` is NULL on 100 % of non-initiated rows and
  ``discontinued_180d == 1 - persistent_180d`` — a whole-table retrain would learn its
  own label.
* 3 HCP rows: the goldstd frame is ``hcp_brand_adoption`` JOIN ``hcp_profiles``, the
  table is not in ``ML_TABLES``, and the scope_definer rewrites any target containing
  "adopt" to ``will_adopt`` — so ``cohort_data_source`` stays NULL; only the label
  column ``adopted`` is provable (harmless: the sweep enqueues only when BOTH are set).
* 2 csu rows: trained on an in-process generated dataset -> both NULL.

Every UPDATE is a compare-and-set on NULL (never overwrites a healed contract) and is
scoped to ``is_synthetic = false``; the JSON literals are byte-identical to
``encode_data_source`` so the sweep's decode/encode round trip is the identity.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.mlops.gold_standard_eval.cohort_spec import (
    _PATIENT_COVARIATES,
    _PATIENT_LABELS,
    BRANDS,
    PATIENT_COHORTS,
)
from src.repositories.ml_data_loader import ML_TABLES
from src.services.cohort_contract import decode_data_source, encode_data_source

_REPO = Path(__file__).resolve().parents[3]
_MIGRATION = _REPO / "database" / "migrations" / "151_registry_cohort_contracts_goldstd.sql"

_PATIENT_MODELS = {
    f"{cohort}_{brand.lower()}_goldstd_lr_v1": (cohort, brand)
    for cohort in PATIENT_COHORTS
    for brand in BRANDS
}
_HCP_MODELS = {f"hcp_adoption_{brand.lower()}_goldstd_lr_v1" for brand in BRANDS}
_CSU_MODELS = {
    "csu_treatment_initiation_lr_balanced_v1",
    "csu_treatment_initiation_lr_full_v1",
}

_UPDATE_RE = re.compile(
    r"UPDATE\s+ml_model_registry\s+SET\s+(?P<set>.*?)\s+WHERE\s+(?P<where>.*?);", re.S
)
_SET_RE = re.compile(r"(?P<col>cohort_\w+)\s*=\s*'(?P<val>(?:[^']|'')*)'")


def _sql() -> str:
    assert _MIGRATION.exists(), f"missing {_MIGRATION}"
    return _MIGRATION.read_text()


def _statements() -> list[dict]:
    body = "\n".join(line for line in _sql().splitlines() if not line.lstrip().startswith("--"))
    out = []
    for m in _UPDATE_RE.finditer(body):
        sets = {
            s.group("col"): s.group("val").replace("''", "'")
            for s in _SET_RE.finditer(m.group("set"))
        }
        name = re.search(r"model_name\s*=\s*'([^']+)'", m.group("where"))
        assert name, m.group(0)
        out.append(
            {"model": name.group(1), "set": sets, "where": m.group("where"), "raw": m.group(0)}
        )
    return out


def test_migration_file_exists_and_is_idempotent_in_shape() -> None:
    sql = _sql()
    assert "BEGIN" not in sql.upper().replace("-- ", "") or "NO BEGIN/COMMIT" in sql.upper()
    assert "COMMIT;" not in sql
    stmts = _statements()
    assert len(stmts) == 12, [s["model"] for s in stmts]


def test_every_update_is_compare_and_set_on_null_and_real_scoped() -> None:
    for s in _statements():
        where = " ".join(s["where"].split())
        assert "is_synthetic = false" in where, s["raw"]
        for col in s["set"]:
            assert f"{col} IS NULL" in where, s["raw"]
        assert s["set"], s["raw"]


def test_patient_rows_get_the_exact_goldstd_contract() -> None:
    by_model = {s["model"]: s for s in _statements()}
    for model, (cohort, brand) in _PATIENT_MODELS.items():
        assert model in by_model, f"{model} has no UPDATE"
        s = by_model[model]
        assert set(s["set"]) == {
            "cohort_data_source",
            "cohort_target_outcome",
            "cohort_feature_manifest_source",
        }, s["raw"]
        # Owner decision 2026-09-23 (measured disproof): the manifest of the DGP that
        # seeded patient_journeys (src/data/manifests/synthetic_csu_feature_manifest.py)
        # is the provable third column — without it the Layer-3 adversarial check
        # flags the DESIGNED drivers (disease_severity z=40σ, age_at_diagnosis z=41σ on
        # initiation) and routes to LLM remediation; with it "Declared-safe immunity"
        # exempts the 5 declared pre-index covariates and the QC gate passes.
        assert s["set"]["cohort_feature_manifest_source"] == "synthetic_csu"
        label = _PATIENT_LABELS[cohort]
        assert s["set"]["cohort_target_outcome"] == label
        literal = s["set"]["cohort_data_source"]
        decoded = decode_data_source(literal)
        assert isinstance(decoded, dict), literal
        # Canonical: exactly json.dumps(sort_keys=True); the round trip is the identity.
        assert encode_data_source(decoded) == literal
        assert literal == json.dumps(decoded, sort_keys=True)
        assert decoded["type"] == "table"
        assert decoded["table"] in ML_TABLES
        assert decoded["table"] == "patient_journeys"
        assert decoded["filters"] == {"brand": brand, "is_synthetic": True}
        assert brand in BRANDS
        assert decoded["columns"] == list(_PATIENT_COVARIATES[cohort]) + [label]
        assert set(decoded) == {"type", "table", "filters", "columns"}


def test_hcp_rows_get_only_the_label_and_csu_rows_nothing() -> None:
    by_model = {s["model"]: s for s in _statements()}
    for model in _HCP_MODELS:
        assert model in by_model, f"{model} has no UPDATE"
        assert by_model[model]["set"] == {"cohort_target_outcome": "adopted"}, by_model[model][
            "raw"
        ]
    for model in _CSU_MODELS:
        assert model not in by_model, f"csu row {model} must stay NULL"
    # No cohort_data_source SET ever names an HCP or csu model.
    for s in _statements():
        if "cohort_data_source" in s["set"]:
            assert s["model"] in _PATIENT_MODELS, s["raw"]
    assert set(by_model) == set(_PATIENT_MODELS) | _HCP_MODELS


def test_manifest_source_is_a_registered_manifest_and_only_on_patient_rows() -> None:
    from src.data.manifests.resolution import known_manifest_sources, resolve_manifest_source

    for s in _statements():
        manifest = s["set"].get("cohort_feature_manifest_source")
        if s["model"] in _PATIENT_MODELS:
            assert manifest in known_manifest_sources(), (s["model"], manifest)
            # The sweep's contract reaches tier_0/pipeline.py as
            # resolve_manifest_source(data_source_dict, manifest): must resolve, no M1/M2.
            contract = decode_data_source(s["set"]["cohort_data_source"])
            assert resolve_manifest_source(contract, manifest) == manifest
        else:
            assert manifest is None, s["raw"]  # HCP rows: frame is a JOIN, no data_source


@pytest.mark.parametrize("model", sorted(_PATIENT_MODELS))
def test_contract_columns_exclude_post_outcome_leaks(model: str) -> None:
    from src.mlops.gold_standard_eval.feature_builder import LEAKAGE_DENYLIST

    s = {x["model"]: x for x in _statements()}[model]
    decoded = decode_data_source(s["set"]["cohort_data_source"])
    label = s["set"]["cohort_target_outcome"]
    leaks = [c for c in decoded["columns"] if c in LEAKAGE_DENYLIST and c != label]
    assert leaks == [], leaks
