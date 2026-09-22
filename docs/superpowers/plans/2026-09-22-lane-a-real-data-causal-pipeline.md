# Lane A — Real-data causal pipeline (Dupixent vs Xolair → persistence) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the causal_impact agent estimable on REAL Optum claims data: export a causal cohort that KEEPS the treatment (`index_biologic_brand` → `treatment_dupixent`), land it in a new table via migration 148 + an idempotent loader, register it as dataset `optum_biologic_persistence` with discovery OFF by default, and record the estimate as a cert.

**Architecture:** Three seams, each already existing: (1) `scripts/convert_optum_mart.py` gains a fourth cohort `persistence_causal` whose selector computes four outcomes and the binary treatment, and whose record builder is allowed to carry six extra columns (the prediction cohorts are untouched — a test proves their frames still drop the treatment); (2) `database/migrations/148_*.sql` + `scripts/load_optum_causal_cohort.py` (parquet → PostgREST upsert on `patient_id`, dry-run default, arm-split verification after the write); (3) `src/api/routes/causal/datasets.py` registry entries so `_load_agent_estimation_frame` reads the table like any single-table dataset (64 baseline covariates, 7 of them one-hot categoricals), plus a per-dataset `auto_discover` default resolved at the submit endpoint and the discovery leaderboard.

**Tech Stack:** Python 3.12, pandas/pyarrow (export), supabase-py (PostgREST upsert/select), FastAPI + pydantic v2 (`model_fields_set`), pytest `-n 0`, PostgreSQL 15 (self-hosted Supabase on the droplet).

**Spec:** `docs/superpowers/specs/2026-09-22-real-data-causal-estimation-design.md` §3A (steps 1–5), §5, §6, §7.

---

## Facts this plan rests on (measured 2026-09-22, in this worktree unless noted)

| Fact | Evidence |
|---|---|
| PostgREST returns the full `limit` — `limit(20000)` on `patient_journeys` returned 20,000 rows | async client probe, this session |
| The raw drop carries `biologic_switch_180d_flag` (int32), `treatment_response` (string), `max_consecutive_biologic_coverage_days` (int32), `last_coverage_end` (date32) | pyarrow schema read of `data/rwd/Optum_Parquet/Optum_enriched.parquet` |
| The shipped persistence mart frame is 15,209 × 74; the 7 text baseline columns are `gdr_cd`, `payer_category`, `payer_product`, `payer_bus`, `charlson_risk_band`, `elixhauser_risk_band`, `geographic_region` (8.1 % NULL); no other NULLs | `data/rwd/mart/persistence/e2i_ml_v3_patient_journeys.parquet` |
| `MART_SAFE_FEATURES` has 64 names; importing `src.data.manifests` costs 1.4 s / 120 MB (the registry module already costs 17 s) | `/usr/bin/time` this session |
| Registry-consistency tests that iterate every dataset: `tests/unit/test_services/test_clinical_context/test_brand_map.py::test_every_curated_treatment_is_a_treatment_the_platform_actually_offers` (reverse direction only — a NEW offered treatment cannot break it) | read |
| `AGENT_FORCEABLE_ESTIMATORS` = `('CausalForestDML', 'LinearDML', 'dml_learner', 'drlearner', 'ols', 'propensity_score_weighting')`; hard cap `_AGENT_HARD_TIMEOUT_S = 900`, refutation budget 720 s | `src/causal_engine/estimator_registry.py`, `src/api/routes/causal/_common.py` |
| The deploy applies `database/migrations/*.sql` via `scripts/run_migrations.sh` and `database/**` is a deploy trigger; the real-DB deploy gate derives PENDING from prod's ledger | `.github/workflows/deploy.yml:98`, memory `lane_2167_contract146_apply_and_move_20260920` |
| The API prefix is `/api/causal`; a cert mints a token with `SUPABASE_URL/auth/v1/token?grant_type=password` (`E2I_ADMIN_PASSWORD`) and polls `GET /causal/agent-analyze/{id}` | `docs/demos/results/2026-09-09_expert_review_loop/run_discovery.py` |
| `worker_light` mounts no `data/rwd` volume; the loader therefore runs on the HOST venv through the PostgREST client (the precedent `scripts/load_hcp_brand_adoption.py` does the same) | `docker/docker-compose.yml` |

**Sequencing consequence (owner GO already recorded in spec §7):** the production table cannot be loaded before migration 148 exists in prod, and prod's API cannot serve the new dataset before the PR is merged + deployed. So the lane ships in two halves: pre-merge evidence (export run, migration rehearsal in `BEGIN…ROLLBACK`, loader dry-run, an in-process agent pre-flight on the exported frame) → PR → owner merge → the deploy applies 148 → owner-GO `--execute` load → the live API cert, committed as a follow-up PR. Task 12 is that second half.

**Not in scope (spec §2 + §3A):** the frontend dataset selector (`frontend/src/pages/CausalAnalysis.tsx:119-121` is a hard-coded three-entry list; Lane A's consumer is the API), the negative-control registry for this source (SKIPPED is the honest verdict until the omitted-confounder experiment is run here), Lane D's discovery pre-flight.

---

## File structure

| Path | Responsibility |
|---|---|
| Modify `scripts/convert_optum_mart.py` | fourth cohort `persistence_causal`: selector (four outcomes + treatment), `extra_cols` on the record builder, registry entries, `is_synthetic=False`, output name, CLI choice, data dictionary |
| Create `tests/unit/test_scripts/test_convert_optum_mart_causal.py` | selector truth table, extra-cols contract, end-to-end convert, prediction cohorts still drop the treatment, `all` excludes the causal cohort |
| Create `database/migrations/148_optum_biologic_persistence_causal.sql` | the table (one column per exported field) + indexes + comments |
| Create `database/migrations/rollback_148_optum_biologic_persistence_causal.sql` | `DROP TABLE IF EXISTS` (the runner skips `rollback_*`) |
| Create `tests/unit/test_scripts/test_optum_causal_cohort_contract.py` | SQL column set == export record key set (the "one column per exported field" contract) |
| Create `scripts/load_optum_causal_cohort.py` | parquet → validated frame → JSON-safe records → batched upsert on `patient_id`; `--dry-run` default; live arm-split verification |
| Create `tests/unit/test_scripts/test_load_optum_causal_cohort.py` | frame validation fail-loud cases, record serialisation, arm split, upsert batching, verification verdict |
| Create `tests/integration/test_optum_causal_cohort_realdb.py` | real-DB arm-split probe (gated `E2I_DB_INTEGRATION=1`, skipped when the parquet is absent) |
| Modify `src/api/routes/causal/datasets.py` | `optum_biologic_persistence` in `_CAUSAL_DATASET_SPECS`, `_CAUSAL_PHYSICAL_TABLE`, `_CAUSAL_NUMERIC_COLUMNS`, `_CAUSAL_CATEGORICAL_COLUMNS`, `_CAUSAL_BRAND_COLUMN`; `_CAUSAL_DISCOVERY_DEFAULT_OFF` + `_default_auto_discover` |
| Modify `src/repositories/provenance.py` | the new table joins `PROVENANCE_TAGGED_TABLES` (it carries `is_synthetic`) |
| Modify `src/api/routes/causal/agent.py` | resolve `auto_discover` per dataset when the caller did not set it; say so in the pending response |
| Modify `src/api/routes/causal/discovery.py` | the leaderboard's hard-coded `auto_discover=True` becomes the per-dataset default |
| Create `tests/unit/test_api/test_causal_optum_dataset_registry.py` | registry pins, manifest consistency, loader coercion + one-hot through a fake client, role gate, `auto_discover` resolution at the submit endpoint |
| Create `docs/demos/results/2026-09-22_optum_biologic_persistence_cert/` | `export_summary.md`, `migration_rehearsal.md`, `preflight_agent.py` + `preflight.md` (pre-merge); `run_cert.py`, `caveats.py`, `cert.md`, `raw_*.json` (post-deploy) |

---

## Conventions for every task

- Work in the lane worktree: `cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/real-data-causal`. Branch `claude/real-data-causal-estimation`. Verify with `git branch --show-current` before every commit.
- Interpreter: `PY=/home/enunez/Projects/e2i_causal_analytics/.venv/bin/python` (the worktree has no `.venv`). Always run tests as `$PY -m pytest -n 0 …` from the worktree dir so `src` resolves to the WORKTREE (a file run by path imports `src` from main via the editable `.pth`).
- Every pytest invocation carries `-n 0` and `-p no:cacheprovider`.
- Data lives only in the main checkout (`data/` is gitignored): `DATA=/home/enunez/Projects/e2i_causal_analytics/data`.
- Do NOT run mypy on this box (CI is the arbiter). Before each commit run ruff whole-tree exactly as CI does: `$PY -m ruff check --no-cache src/ tests/ && $PY -m ruff format --check src/ tests/   # CI lints src/ and tests/ only (backend-tests.yml:120-123); lint changed scripts/ files individually`.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

---

### Task 1: The causal cohort selector (four outcomes + the binary treatment)

**Files:**
- Modify: `scripts/convert_optum_mart.py` (constants after line 62; new function after `select_persistence_cohort`, i.e. after line 199)
- Test: `tests/unit/test_scripts/test_convert_optum_mart_causal.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_scripts/test_convert_optum_mart_causal.py`:

```python
"""Lane A (spec 2026-09-22 §3A.1): the CAUSAL cohort export of the Optum mart.

The prediction cohorts drop ``index_biologic_brand`` on purpose (the manifest
declares it post-index ``mart_treatment``). For causal estimation that column
IS the treatment, so ``persistence_causal`` is a SEPARATE cohort that keeps it
and adds the days-supply-robust primary outcome ``persistent_at_180d_g28``
(``docs/demos/results/2026-09-22_persistence_definition_disproof/``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.convert_optum_mart import (  # noqa: E402
    CAUSAL_ARMS,
    PERSIST_GRACE_DAYS,
    SWITCH_FLAG,
    TARGET_PERSISTENT_G28,
    TREATMENT_COL,
    select_persistence_causal_cohort,
)


def _initiators_two_arms() -> pd.DataFrame:
    ts = pd.Timestamp("2020-01-01")
    day = pd.Timedelta(days=1)
    base = {
        "claim_record_count": 10,
        "last_observed_date": ts + 400 * day,
        "terminal_gap_days": 0,
    }
    return pd.DataFrame(
        [
            # p4: XOLAIR, covered through day 220, gap 30 -> persist 1 / g28 1 / disc 0
            {
                "patid": 4,
                "index_biologic_brand": "XOLAIR",
                "treatment_start_date": ts,
                "last_coverage_end": ts + 220 * day,
                "max_internal_gap_days": 30,
                SWITCH_FLAG: 0,
                **base,
            },
            # p5: DUPIXENT, coverage ends day 100 + 120d gap -> persist 0 / g28 0 / disc 1
            {
                "patid": 5,
                "index_biologic_brand": "DUPIXENT",
                "treatment_start_date": ts,
                "last_coverage_end": ts + 100 * day,
                "max_internal_gap_days": 120,
                SWITCH_FLAG: 1,
                **base,
            },
            # p6: XOLAIR, covered to 220 but a >60d gap -> persist 0 / g28 0 / disc 0
            {
                "patid": 6,
                "index_biologic_brand": "XOLAIR",
                "treatment_start_date": ts,
                "last_coverage_end": ts + 220 * day,
                "max_internal_gap_days": 90,
                SWITCH_FLAG: 0,
                **base,
            },
            # p7: DUPIXENT, coverage ends day 160 (a 14-day-supply last fill), no gap
            #     -> shipped persist 0 (160 < 180) / g28 1 (160 >= 152) / disc 0
            {
                "patid": 7,
                "index_biologic_brand": "DUPIXENT",
                "treatment_start_date": ts,
                "last_coverage_end": ts + 160 * day,
                "max_internal_gap_days": 10,
                SWITCH_FLAG: None,  # NULL flag reads as 0, never drops the row
                **base,
            },
            # p8: a brand outside the observed contrast (a future drop) -> excluded,
            #     never coded 0 (= XOLAIR) silently
            {
                "patid": 8,
                "index_biologic_brand": "RHAPSIDO",
                "treatment_start_date": ts,
                "last_coverage_end": ts + 220 * day,
                "max_internal_gap_days": 0,
                SWITCH_FLAG: 0,
                **base,
            },
        ]
    )


def test_causal_selector_truth_table_and_treatment_coding():
    cohort, attrition = select_persistence_causal_cohort(
        _initiators_two_arms(), window_days=180, min_claim_count=2
    )
    steps = dict(attrition)
    assert steps["two_arm_contrast"] == 4  # p8 excluded
    assert set(cohort["patid"]) == {4, 5, 6, 7}
    by = cohort.set_index("patid")
    assert by[TREATMENT_COL].to_dict() == {4: 0, 5: 1, 6: 0, 7: 1}
    assert by["persistent_at_180d"].to_dict() == {4: 1, 5: 0, 6: 0, 7: 0}
    assert by[TARGET_PERSISTENT_G28].to_dict() == {4: 1, 5: 0, 6: 0, 7: 1}
    assert by["discontinued_180d"].to_dict() == {4: 0, 5: 1, 6: 0, 7: 0}
    assert by[SWITCH_FLAG].to_dict() == {4: 0, 5: 1, 6: 0, 7: 0}
    for col in (TREATMENT_COL, "persistent_at_180d", TARGET_PERSISTENT_G28, "discontinued_180d", SWITCH_FLAG):
        assert cohort[col].dtype.kind in "iu", col
    assert steps["target_positives"] == 2  # g28 positives: p4, p7
    assert steps["arm_dupixent"] == 2


def test_causal_selector_grace_is_28_days_and_arms_are_the_observed_pair():
    assert PERSIST_GRACE_DAYS == 28
    assert CAUSAL_ARMS == ("XOLAIR", "DUPIXENT")
    assert TREATMENT_COL == "treatment_dupixent"
    assert TARGET_PERSISTENT_G28 == "persistent_at_180d_g28"


def test_causal_selector_fails_loud_without_the_switch_flag():
    df = _initiators_two_arms().drop(columns=[SWITCH_FLAG])
    with pytest.raises(KeyError, match=SWITCH_FLAG):
        select_persistence_causal_cohort(df, window_days=180, min_claim_count=2)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest -n 0 -p no:cacheprovider tests/unit/test_scripts/test_convert_optum_mart_causal.py -q`
Expected: `ImportError: cannot import name 'CAUSAL_ARMS'` (collection error).

- [ ] **Step 3: Add the constants and the selector**

In `scripts/convert_optum_mart.py`, directly after `PERSIST_GAP_DAYS = 60` (line 62), add:

```python
# --- Lane A (spec 2026-09-22 §3A.1): the CAUSAL cohort ------------------------
# The persistence outcome is days-supply sensitive per brand (14-d Dupixent fills
# vs 28-45-d Xolair fills): the shipped ``persistent_at_180d`` gap of -17.6 pp
# collapses to 3.5 pp with a 14-day grace and INVERTS from 28 d on. The primary
# causal outcome is therefore covered-through-day-(180-28) AND no internal gap
# > 60 d — brand-invariant across 28-60 d grace. Owner decision, spec §7;
# evidence docs/demos/results/2026-09-22_persistence_definition_disproof/.
TARGET_PERSISTENT_G28 = "persistent_at_180d_g28"
PERSIST_GRACE_DAYS = 28
# Binary treatment: 1 = DUPIXENT, 0 = XOLAIR — the only contrast the drop
# observes (remibrutinib is absent; FDA approval 2025-09-30 is the drop's last
# observed day). Any other brand is EXCLUDED with an attrition step rather than
# silently coded as the reference arm.
TREATMENT_COL = "treatment_dupixent"
CAUSAL_ARMS = ("XOLAIR", "DUPIXENT")
# Precomputed switch flag carried from the raw drop (int32, both arms observed).
SWITCH_FLAG = "biologic_switch_180d_flag"
```

Directly after `select_persistence_cohort` (after line 199), add:

```python
def select_persistence_causal_cohort(
    df: pd.DataFrame, *, window_days: int = 180, min_claim_count: int = 2
) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Treatment-anchored CAUSAL cohort: Dupixent vs Xolair initiators with four
    outcomes and the binary treatment (Lane A, spec 2026-09-22 §3A.1).

    Same denominator as the prediction persistence cohort (``_initiator_eligible``)
    restricted to the observed two-arm contrast. Emits:

    - ``treatment_dupixent``     1 = DUPIXENT, 0 = XOLAIR
    - ``persistent_at_180d_g28`` PRIMARY: covered through day window-28 AND no
                                 internal gap > PERSIST_GAP_DAYS (brand-invariant)
    - ``discontinued_180d``      secondary, as shipped (brand-robust)
    - ``biologic_switch_180d_flag`` as shipped in the raw drop (NULL -> 0)
    - ``persistent_at_180d``     the shipped definition, reported only alongside
                                 the days-supply sweep (a measurement artefact)
    """
    df, attrition = _initiator_eligible(
        df, window_days=window_days, min_claim_count=min_claim_count
    )
    if SWITCH_FLAG not in df.columns:
        raise KeyError(
            f"{SWITCH_FLAG} is not in the input frame; the causal cohort carries it as "
            "an outcome and will not fabricate zeros for a missing column"
        )
    in_contrast = df["index_biologic_brand"].isin(CAUSAL_ARMS)
    df = df.loc[in_contrast].copy()
    attrition.append(("two_arm_contrast", len(df)))

    ts = pd.to_datetime(df["treatment_start_date"])
    lce = pd.to_datetime(df["last_coverage_end"])
    cov_to_end = (lce - ts).dt.days
    gap = df["max_internal_gap_days"].fillna(0)
    term = df["terminal_gap_days"].fillna(0)

    df[TREATMENT_COL] = df["index_biologic_brand"].eq("DUPIXENT").astype("int64")
    df[TARGET_PERSISTENT] = ((cov_to_end >= window_days) & (gap <= PERSIST_GAP_DAYS)).astype(
        "int64"
    )
    df[TARGET_PERSISTENT_G28] = (
        (cov_to_end >= window_days - PERSIST_GRACE_DAYS) & (gap <= PERSIST_GAP_DAYS)
    ).astype("int64")
    df[TARGET_DISCONTINUED] = (
        (cov_to_end < window_days) & ((gap >= DISCONT_GAP_DAYS) | (term >= DISCONT_GAP_DAYS))
    ).astype("int64")
    df[SWITCH_FLAG] = df[SWITCH_FLAG].fillna(0).astype("int64")

    attrition.append(("target_positives", int(df[TARGET_PERSISTENT_G28].sum())))
    attrition.append(("arm_dupixent", int(df[TREATMENT_COL].sum())))
    return df, attrition
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest -n 0 -p no:cacheprovider tests/unit/test_scripts/test_convert_optum_mart_causal.py -q`
Expected: `3 passed`.

- [ ] **Step 5: Run the existing converter suites to prove nothing moved**

Run: `$PY -m pytest -n 0 -p no:cacheprovider tests/unit/test_scripts/test_convert_optum_mart.py tests/unit/test_scripts/test_convert_optum_mart_multicohort.py -q`
Expected: all passed (same count as before the edit; record the number).

- [ ] **Step 6: Commit**

```bash
git branch --show-current   # claude/real-data-causal-estimation
git add scripts/convert_optum_mart.py tests/unit/test_scripts/test_convert_optum_mart_causal.py
git commit -m "feat(optum-mart): causal cohort selector — two-arm contrast, treatment_dupixent, persistent_at_180d_g28 primary

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Export the causal cohort (records carry the treatment; prediction cohorts do not)

> **Amended after the Task 1–2 quality review (commit `ec85fb7fa`), the landed code differs from the blocks below in these ways — the code is the truth:**
> - the shipped outcome formulas live in two module-level predicates `_persistent(cov_to_end, gap, *, window_days, grace_days=0)` and `_discontinued(cov_to_end, gap, term, *, window_days)` used by all three selectors (byte-identical prediction outputs proven across nine file hashes);
> - `CAUSAL_EXTRA_COLS` + `_CAUSAL_DICTIONARY_TYPE` became ONE ordered mapping `CAUSAL_EXTRA_COLUMNS: dict[str, tuple[dictionary_type, kind]]` with kinds `date` / `text` / `flag`; `CAUSAL_EXTRA_COLS = tuple(CAUSAL_EXTRA_COLUMNS)` is kept for the later tasks; a `flag` outside {0, 1} (NaN, None, 0.5) raises `ValueError` naming the column — never truncated;
> - the causal selector appends one `excluded_arm:<BRAND>` attrition step per brand outside the pair before `two_arm_contrast`;
> - `convert()` computes `arms` from `cohort_df["index_biologic_brand"].value_counts()`;
> - extra tests: boundary rows (cov_to_end 151/152, gap 60/61), flag validation, the allow-list guard parameterised over `extra_cols`, and a registry key-set test.

**Files:**
- Modify: `scripts/convert_optum_mart.py` — `build_journey_records` (line 202), `_data_dictionary_entries` (line 259), the cohort registry (lines 282–318), `convert` (line 376), `main` (line 427)
- Test: `tests/unit/test_scripts/test_convert_optum_mart_causal.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_scripts/test_convert_optum_mart_causal.py`:

```python
from scripts.convert_optum_mart import (  # noqa: E402
    CAUSAL_COHORT,
    CAUSAL_EXTRA_COLS,
    CAUSAL_RECORDS_NAME,
    COHORT_TARGETS,
    PREDICTION_COHORTS,
    TARGET_PERSISTENT,
    build_journey_records,
    convert,
)
from scripts.convert_optum_mart import main as convert_main  # noqa: E402


def _entity_mart_rows_two_arms() -> list[dict]:
    """A tiny entity-stacked mart with both arms + an untreated patient + an HCP row."""
    idx = pd.Timestamp("2020-01-01")
    day = pd.Timedelta(days=1)
    safe = {"age_at_index": 50.0, "charlson_score": 2, "cci_hiv": 0, "payer_category": "commercial"}
    return [
        {
            "entity_type": "patient",
            "patid": 1,
            "index_biologic_brand": "no_treatment",
            "treatment_start_date": pd.NaT,
            "index_date": idx,
            "claim_record_count": 10,
            "elig_start_date": idx - 300 * day,
            "zipcode_5": "10001",
            "last_observed_date": idx + 400 * day,
            "last_coverage_end": pd.NaT,
            "max_internal_gap_days": 0,
            "terminal_gap_days": 0,
            SWITCH_FLAG: 0,
            **safe,
        },
        {
            "entity_type": "patient",
            "patid": 2,
            "index_biologic_brand": "XOLAIR",
            "treatment_start_date": idx + 10 * day,
            "index_date": idx,
            "claim_record_count": 8,
            "elig_start_date": idx - 365 * day,
            "zipcode_5": "90001",
            "last_observed_date": idx + 410 * day,
            "last_coverage_end": idx + 110 * day,
            "max_internal_gap_days": 120,
            "terminal_gap_days": 0,
            SWITCH_FLAG: 1,
            **safe,
        },
        {
            "entity_type": "patient",
            "patid": 3,
            "index_biologic_brand": "DUPIXENT",
            "treatment_start_date": idx + 20 * day,
            "index_date": idx,
            "claim_record_count": 12,
            "elig_start_date": idx - 200 * day,
            "zipcode_5": "60601",
            "last_observed_date": idx + 420 * day,
            "last_coverage_end": idx + 240 * day,
            "max_internal_gap_days": 30,
            "terminal_gap_days": 0,
            SWITCH_FLAG: 0,
            **safe,
        },
        {
            "entity_type": "optum_hcp",
            "patid": 999,
            "index_biologic_brand": None,
            "treatment_start_date": pd.NaT,
            "index_date": pd.NaT,
            "claim_record_count": None,
            "elig_start_date": pd.NaT,
            "zipcode_5": None,
            "last_observed_date": pd.NaT,
            "last_coverage_end": pd.NaT,
            "max_internal_gap_days": None,
            "terminal_gap_days": None,
            SWITCH_FLAG: None,
            "age_at_index": None,
            "charlson_score": None,
            "cci_hiv": None,
            "payer_category": None,
        },
    ]


def test_build_journey_records_extra_cols_are_emitted_and_default_is_unchanged():
    tstart = pd.Timestamp("2020-03-01")
    cohort = pd.DataFrame(
        [
            {
                "patid": 77,
                "index_date": pd.Timestamp("2020-01-01"),
                "treatment_start_date": tstart,
                "elig_start_date": pd.Timestamp("2019-09-01"),
                "zipcode_5": "10001",
                "age_at_index": 50.0,
                "charlson_score": 2,
                "cci_hiv": 0,
                "index_biologic_brand": "DUPIXENT",
                TREATMENT_COL: 1,
                TARGET_PERSISTENT_G28: 1,
                "discontinued_180d": 0,
                SWITCH_FLAG: 0,
                "persistent_at_180d": 0,
            }
        ]
    )
    rec = build_journey_records(
        cohort,
        target=TARGET_PERSISTENT_G28,
        anchor_col="treatment_start_date",
        extra_cols=CAUSAL_EXTRA_COLS,
    )[0]
    assert rec["index_biologic_brand"] == "DUPIXENT"
    assert rec[TREATMENT_COL] == 1 and isinstance(rec[TREATMENT_COL], int)
    assert rec["treatment_start_date"] == tstart
    assert rec["persistent_at_180d"] == 0 and rec["discontinued_180d"] == 0 and rec[SWITCH_FLAG] == 0
    assert rec[TARGET_PERSISTENT_G28] == 1
    # default call (prediction cohorts) still drops every one of them
    plain = build_journey_records(cohort, target=TARGET_PERSISTENT_G28, anchor_col="treatment_start_date")[0]
    for col in CAUSAL_EXTRA_COLS:
        assert col not in plain, f"{col} leaked into a prediction record"


def test_causal_registry_entries():
    assert COHORT_TARGETS[CAUSAL_COHORT] == TARGET_PERSISTENT_G28
    assert CAUSAL_COHORT == "persistence_causal"
    assert CAUSAL_RECORDS_NAME == "e2i_causal_v1_biologic_persistence"
    assert CAUSAL_EXTRA_COLS == (
        "index_biologic_brand",
        TREATMENT_COL,
        "treatment_start_date",
        "discontinued_180d",
        SWITCH_FLAG,
        "persistent_at_180d",
    )
    assert CAUSAL_COHORT not in PREDICTION_COHORTS
    assert PREDICTION_COHORTS == ("initiation", "discontinuation", "persistence")


def test_convert_persistence_causal_end_to_end(tmp_path):
    mart = tmp_path / "mart.parquet"
    pd.DataFrame(_entity_mart_rows_two_arms()).to_parquet(mart)
    out = tmp_path / "causal"
    summary = convert(
        input_path=str(mart), output_dir=str(out), cohort=CAUSAL_COHORT, window_days=180, min_claim_count=2
    )
    assert summary["cohort"] == CAUSAL_COHORT
    assert summary["patients"] == 2
    assert summary["positives"] == 1  # p3 g28-persistent
    assert summary["arms"] == {"XOLAIR": 1, "DUPIXENT": 1}
    frame = pd.read_parquet(out / f"{CAUSAL_RECORDS_NAME}.parquet")
    assert not (out / "e2i_ml_v3_patient_journeys.parquet").exists()
    for col in (*CAUSAL_EXTRA_COLS, TARGET_PERSISTENT_G28, "is_synthetic", "payer_category"):
        assert col in frame.columns, col
    assert frame["is_synthetic"].dtype == bool and not frame["is_synthetic"].any()
    by = frame.set_index("patient_id")
    assert by.loc["PAT_2", TREATMENT_COL] == 0 and by.loc["PAT_3", TREATMENT_COL] == 1
    assert by.loc["PAT_2", "discontinued_180d"] == 1 and by.loc["PAT_3", TARGET_PERSISTENT_G28] == 1
    # journeys anchor at the first biologic fill
    assert pd.Timestamp(by.loc["PAT_3", "index_date"]) == pd.Timestamp("2020-01-21")
    attrition = pd.read_csv(out / "attrition_report.csv")
    assert "two_arm_contrast" in set(attrition["step"])
    dictionary = pd.read_csv(out / "data_dictionary.csv")
    assert set(CAUSAL_EXTRA_COLS) <= set(dictionary["feature"])
    assert set(dictionary.loc[dictionary["feature"] == TREATMENT_COL, "type"]) == {"treatment"}
    assert set(dictionary.loc[dictionary["feature"] == TARGET_PERSISTENT_G28, "type"]) == {"target"}


def test_prediction_cohorts_still_drop_the_treatment(tmp_path):
    """Spec §3A.1: the causal frame carries the treatment; the prediction frame does not."""
    mart = tmp_path / "mart.parquet"
    pd.DataFrame(_entity_mart_rows_two_arms()).to_parquet(mart)
    out = tmp_path / "persistence"
    convert(input_path=str(mart), output_dir=str(out), cohort="persistence", window_days=180, min_claim_count=2)
    frame = pd.read_parquet(out / "e2i_ml_v3_patient_journeys.parquet")
    # TARGET_PERSISTENT ("persistent_at_180d") is a member of CAUSAL_EXTRA_COLS
    # AND this PREDICTION cohort's own supervised target — it belongs in this
    # frame for a reason unrelated to Lane A, so it is excluded from the
    # leak-check and asserted present instead.
    for col in {*CAUSAL_EXTRA_COLS, TARGET_PERSISTENT_G28, "is_synthetic"} - {TARGET_PERSISTENT}:
        assert col not in frame.columns, f"{col} leaked into the prediction persistence frame"
    assert "persistent_at_180d" in frame.columns


def test_main_all_builds_prediction_cohorts_only(tmp_path):
    mart = tmp_path / "mart.parquet"
    pd.DataFrame(_entity_mart_rows_two_arms()).to_parquet(mart)
    base = tmp_path / "marts"
    assert convert_main(["--cohort", "all", "--input", str(mart), "--output", str(base)]) == 0
    assert {p.name for p in base.iterdir()} == set(PREDICTION_COHORTS)
    assert convert_main(["--cohort", CAUSAL_COHORT, "--input", str(mart), "--output", str(base / CAUSAL_COHORT)]) == 0
    assert (base / CAUSAL_COHORT / f"{CAUSAL_RECORDS_NAME}.parquet").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest -n 0 -p no:cacheprovider tests/unit/test_scripts/test_convert_optum_mart_causal.py -q`
Expected: `ImportError: cannot import name 'CAUSAL_COHORT'`.

- [ ] **Step 3: Extend `build_journey_records`**

Replace the signature and the tail of the per-row loop in `scripts/convert_optum_mart.py` (currently lines 202–256). The new function body:

```python
def build_journey_records(
    df: pd.DataFrame,
    *,
    target: str = TARGET,
    anchor_col: str = "index_date",
    extra_cols: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Map cohort rows to canonical journey record-dicts.

    Emits ONLY the pre-index allow-list (raw + derived geographic_region /
    enrollment_duration_days) + ids + the cohort ``target``. Leakage columns
    present in the input are NOT carried through (positive enumeration).

    ``anchor_col`` is the temporal index for the journey: ``index_date`` for the
    initiation cohort, ``treatment_start_date`` for the treatment-anchored
    discontinuation/persistence cohorts (the 64 baseline features are measured at
    the dx index, which is <= treatment-start, so they remain pre-index there).

    ``extra_cols`` (Lane A) is the CAUSAL cohort's positive enumeration of the
    columns the prediction cohorts must never carry: the treatment, its brand
    label, the treatment start and the secondary outcomes. Dates are emitted as
    Timestamps, text as-is, everything else as a plain int (0/1 flags). The
    default (empty) keeps every prediction cohort byte-identical.
    """
    raw_features = [c for c in MART_SAFE_FEATURES if c not in _DERIVED and c in df.columns]
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        index_date = pd.to_datetime(row[anchor_col])
        rec: dict[str, Any] = {
            "patient_journey_id": f"PJ_{row['patid']}",
            "patient_id": f"PAT_{row['patid']}",
            "patient_hash": patient_hash(row["patid"]),
            "index_date": index_date,
            "journey_start_date": index_date,
            # Journey metadata the data_preparer QC/GE contract expects (the
            # standard converter emits these on every cohort). ``ge_validator``
            # routes a patient frame carrying ``discontinuation_flag`` (and no
            # ``event_type``) to the no-event ``ml_patients`` GE suite; the QC
            # checker requires ``journey_status``. Neither is in
            # MART_SAFE_FEATURES, so the optum_mart manifest excludes them from
            # the model feature set (initiation does not model discontinuation).
            "journey_status": "active",
            "discontinuation_flag": 0,
            target: int(row[target]),
        }
        for col in raw_features:
            rec[col] = row[col]
        elig_raw = row.get("elig_start_date")
        elig = pd.to_datetime(elig_raw) if elig_raw is not None else pd.NaT
        rec["enrollment_duration_days"] = int((index_date - elig).days) if pd.notna(elig) else None
        zip5 = row.get("zipcode_5")
        rec["geographic_region"] = map_zipcode_to_region(zip5) if isinstance(zip5, str) else None
        # Transparent data-quality score: the populated fraction of THIS
        # patient's emitted model-input features (raw allow-list + the 2
        # derived columns). Honest completeness metadata — every existing
        # converter emits a ``data_quality_score`` and the harness cohort gate
        # filters on it. It is NOT a model feature: the manifest excludes it
        # (not in MART_SAFE_FEATURES) and the runner's exclude-set drops it.
        model_inputs = [rec[c] for c in raw_features]
        model_inputs.append(rec["enrollment_duration_days"])
        model_inputs.append(rec["geographic_region"])
        present = sum(1 for v in model_inputs if pd.notna(v))
        rec["data_quality_score"] = round(present / len(model_inputs), 4) if model_inputs else 0.0
        for col in extra_cols:
            value = row[col]
            if col.endswith("_date"):
                rec[col] = pd.to_datetime(value)
            elif isinstance(value, str):
                rec[col] = value
            else:
                rec[col] = int(value)
        records.append(rec)
    return records
```

- [ ] **Step 4: Extend the data dictionary**

Replace `_data_dictionary_entries` (lines 259–279) with:

```python
_CAUSAL_DICTIONARY_TYPE = {
    "index_biologic_brand": "treatment",
    TREATMENT_COL: "treatment",
    "treatment_start_date": "anchor",
    TARGET_DISCONTINUED: "outcome",
    SWITCH_FLAG: "outcome",
    TARGET_PERSISTENT: "outcome",
}


def _data_dictionary_entries(
    target: str = TARGET, *, extra_cols: tuple[str, ...] = ()
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for name in sorted(MART_SAFE_FEATURES) + [target]:
        contract = optum_mart_contract_for(name)
        ref = contract.knowable_at.reference if contract else "derived"
        entries.append(
            {
                "feature": name,
                "type": "target" if name == target else "feature",
                "source_table": "optum_mart.patient",
                "lookback_window": ref,
                "null_rate": "",
                "notes": (
                    "supervised label (post-index)"
                    if name == target
                    else "pre-index admissible; data_quality_band is upstream-opaque, NOT used as a gate"
                ),
            }
        )
    for name in extra_cols:
        entries.append(
            {
                "feature": name,
                "type": _CAUSAL_DICTIONARY_TYPE[name],
                "source_table": "optum_mart.patient",
                "lookback_window": "post-index (mart_treatment)",
                "null_rate": "",
                "notes": (
                    "CAUSAL export only — the treatment / its anchor / a secondary outcome; "
                    "never a prediction feature (manifest forbids it as such)"
                ),
            }
        )
    return entries
```

- [ ] **Step 5: Register the cohort**

Replace the registry block (currently lines 282–318, from `# --- Cohort registry` through `_OUTCOME_COLS = (...)`) with:

```python
# --- Cohort registry (initiation + the treatment-anchored disc/persistence
# prediction cohorts + the Lane A causal cohort) ---
CAUSAL_COHORT = "persistence_causal"
PREDICTION_COHORTS = ("initiation", "discontinuation", "persistence")
# The causal export's positive enumeration of non-feature columns (order = the
# parquet / table column order after the allow-list features).
CAUSAL_EXTRA_COLS: tuple[str, ...] = (
    "index_biologic_brand",
    TREATMENT_COL,
    "treatment_start_date",
    TARGET_DISCONTINUED,
    SWITCH_FLAG,
    TARGET_PERSISTENT,
)
CAUSAL_RECORDS_NAME = "e2i_causal_v1_biologic_persistence"
COHORT_TARGETS: dict[str, str] = {
    "initiation": TARGET,
    "discontinuation": TARGET_DISCONTINUED,
    "persistence": TARGET_PERSISTENT,
    CAUSAL_COHORT: TARGET_PERSISTENT_G28,
}
_SELECTOR_BY_COHORT = {
    "initiation": select_initiation_cohort,
    "discontinuation": select_discontinuation_cohort,
    "persistence": select_persistence_cohort,
    CAUSAL_COHORT: select_persistence_causal_cohort,
}
# Journey temporal anchor: dx-index for initiation; first biologic fill for the
# treatment-anchored cohorts (the 64 baseline features are knowable at dx-index,
# which is <= treatment-start, so they remain pre-index in that frame).
_ANCHOR_BY_COHORT = {
    "initiation": "index_date",
    "discontinuation": "treatment_start_date",
    "persistence": "treatment_start_date",
    CAUSAL_COHORT: "treatment_start_date",
}
_SPLIT_CONFIG_BY_COHORT = {
    "initiation": ("optum_mart_initiation_v1", "optum_mart_initiation"),
    "discontinuation": ("optum_mart_discontinuation_v1", "optum_mart_discontinuation"),
    "persistence": ("optum_mart_persistence_v1", "optum_mart_persistence"),
    CAUSAL_COHORT: ("optum_mart_persistence_causal_v1", "optum_mart_persistence_causal"),
}
_OUTPUT_BY_COHORT = {
    "initiation": DEFAULT_OUTPUT,
    "discontinuation": "data/rwd/mart/discontinuation",
    "persistence": "data/rwd/mart/persistence",
    CAUSAL_COHORT: "data/rwd/mart/persistence_causal",
}
_TREATMENT_ANCHORED = ("discontinuation", "persistence", CAUSAL_COHORT)
_EXTRA_COLS_BY_COHORT: dict[str, tuple[str, ...]] = {CAUSAL_COHORT: CAUSAL_EXTRA_COLS}
_RECORDS_NAME_BY_COHORT: dict[str, str] = {CAUSAL_COHORT: CAUSAL_RECORDS_NAME}
# Coverage/gap columns the treatment-anchored cohorts need beyond the allow-list
# (the switch flag is projected for every treatment-anchored read; only the
# causal selector consumes it).
_OUTCOME_COLS = (
    "last_observed_date",
    "last_coverage_end",
    "max_internal_gap_days",
    "terminal_gap_days",
    SWITCH_FLAG,
)
```

- [ ] **Step 6: Wire `convert` and `main`**

In `convert` (line 376 on), replace from `records = build_journey_records(...)` through `write_data_dictionary(...)` and the `summary` dict with:

```python
    extra_cols = _EXTRA_COLS_BY_COHORT.get(cohort, ())
    records = build_journey_records(
        cohort_df, target=target, anchor_col=anchor, extra_cols=extra_cols
    )
    if cohort == CAUSAL_COHORT:
        # Real rows, tagged so the real-mode provenance filter keeps them and a
        # synthetic-gold plant can never masquerade as claims data.
        for rec in records:
            rec["is_synthetic"] = False
    split = apply_chronological_split(records, date_key="journey_start_date", id_key="patient_id")

    out = Path(output_dir)
    write_records(
        out, _RECORDS_NAME_BY_COHORT.get(cohort, "e2i_ml_v3_patient_journeys"), records, fmt="parquet"
    )
    cfg_id, cfg_name = _SPLIT_CONFIG_BY_COHORT[cohort]
    registry = build_split_registry(
        split_config_id=cfg_id,
        config_name=cfg_name,
        config_version="v1",
        split_dates=split["split_dates"],
    )
    write_records(out, "e2i_ml_v3_split_registry", registry, fmt="json")
    write_attrition_report(out, attrition)
    write_data_dictionary(out, _data_dictionary_entries(target, extra_cols=extra_cols))

    positives = int(sum(r[target] for r in records))
    summary: dict[str, Any] = {
        "cohort": cohort,
        "patients": len(records),
        "positives": positives,
        "prevalence": round(positives / len(records), 4) if records else 0.0,
        "splits": split["counts"],
        "output_dir": str(out),
    }
    if cohort == CAUSAL_COHORT:
        arms: dict[str, int] = {}
        for rec in records:
            arms[rec["index_biologic_brand"]] = arms.get(rec["index_biologic_brand"], 0) + 1
        summary["arms"] = arms
    logger.info("Conversion summary: %s", summary)
    return summary
```

In `main`, change the `--cohort` argument and the cohort list:

```python
    parser.add_argument(
        "--cohort",
        default="initiation",
        choices=(*PREDICTION_COHORTS, CAUSAL_COHORT, "all"),
        help=(
            "Which cohort to build. 'all' builds the three PREDICTION cohorts; the "
            f"causal cohort '{CAUSAL_COHORT}' (keeps the treatment) is always explicit."
        ),
    )
```
and
```python
    cohorts = list(PREDICTION_COHORTS) if args.cohort == "all" else [args.cohort]
```

Also update the module docstring's usage block (lines 20–30) with one extra line:
```
    python scripts/convert_optum_mart.py --cohort persistence_causal   # Lane A causal export (keeps the treatment)
```

- [ ] **Step 7: Run the new and the existing converter suites**

Run: `$PY -m pytest -n 0 -p no:cacheprovider tests/unit/test_scripts/test_convert_optum_mart_causal.py tests/unit/test_scripts/test_convert_optum_mart.py tests/unit/test_scripts/test_convert_optum_mart_multicohort.py tests/unit/test_scripts/test_run_optum_tier0_mart_cohort.py tests/unit/test_scripts/test_run_optum_tier0_manifest_wiring.py -q`
Expected: all passed. `test_build_journey_records_emits_only_cataloged_columns` (multicohort) must still pass — it calls the default (no `extra_cols`).

- [ ] **Step 8: Ruff, then commit**

```bash
$PY -m ruff check --no-cache scripts/convert_optum_mart.py tests/unit/test_scripts/test_convert_optum_mart_causal.py && $PY -m ruff format --check scripts/convert_optum_mart.py tests/unit/test_scripts/test_convert_optum_mart_causal.py
git add scripts/convert_optum_mart.py tests/unit/test_scripts/test_convert_optum_mart_causal.py
git commit -m "feat(optum-mart): persistence_causal cohort export keeps the treatment; prediction cohorts unchanged

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Migration 148 + the "one column per exported field" contract test

> **Amended after review (commit `c9be0e9a5`):** the contract test's SQL parser captures ANY type word and raises `AssertionError` for a type outside `_KNOWN_TYPES` instead of silently skipping the column (a `DOUBLE PRECISION` / `BIGINT` / `JSONB` column could otherwise pass the "no more" half of the contract unnoticed); one extra test feeds such a line to `_sql_columns`.

**Files:**
- Create: `database/migrations/148_optum_biologic_persistence_causal.sql`
- Create: `database/migrations/rollback_148_optum_biologic_persistence_causal.sql`
- Test: `tests/unit/test_scripts/test_optum_causal_cohort_contract.py`

- [ ] **Step 1: Write the failing contract test**

Create `tests/unit/test_scripts/test_optum_causal_cohort_contract.py`:

```python
"""Migration 148 carries one column per field the causal export emits — no more,
no fewer (spec 2026-09-22 §3A.2). The export is the SSOT: a feature added to
MART_SAFE_FEATURES or a new extra column shows up here as a missing SQL column.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.convert_optum_mart import (  # noqa: E402
    CAUSAL_EXTRA_COLS,
    SWITCH_FLAG,
    TARGET_PERSISTENT_G28,
    TREATMENT_COL,
    build_journey_records,
)
from src.data.manifests import MART_SAFE_FEATURES  # noqa: E402

MIGRATION = _REPO_ROOT / "database/migrations/148_optum_biologic_persistence_causal.sql"
ROLLBACK = _REPO_ROOT / "database/migrations/rollback_148_optum_biologic_persistence_causal.sql"
TABLE = "optum_biologic_persistence_causal"
# Server-defaulted bookkeeping the loader never writes.
_SERVER_COLUMNS = {"created_at", "updated_at"}
_COLUMN_RE = re.compile(
    r"^\s*([a-z_][a-z0-9_]*)\s+(TEXT|VARCHAR|INTEGER|SMALLINT|NUMERIC|DATE|BOOLEAN|TIMESTAMPTZ)\b",
    re.IGNORECASE,
)


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
        m = _COLUMN_RE.match(line)
        if m and m.group(1).upper() != "CONSTRAINT":
            columns[m.group(1)] = m.group(2).upper()
    return columns


def _export_record_keys() -> set[str]:
    row = {
        "patid": 1,
        "index_date": pd.Timestamp("2020-01-01"),
        "treatment_start_date": pd.Timestamp("2020-02-01"),
        "elig_start_date": pd.Timestamp("2019-01-01"),
        "zipcode_5": "10001",
        "index_biologic_brand": "XOLAIR",
        TREATMENT_COL: 0,
        TARGET_PERSISTENT_G28: 1,
        "discontinued_180d": 0,
        SWITCH_FLAG: 0,
        "persistent_at_180d": 1,
    }
    # every raw allow-list feature present so raw_features is the full list
    for name in MART_SAFE_FEATURES:
        row.setdefault(name, 0)
    rec = build_journey_records(
        pd.DataFrame([row]),
        target=TARGET_PERSISTENT_G28,
        anchor_col="treatment_start_date",
        extra_cols=CAUSAL_EXTRA_COLS,
    )[0]
    return set(rec) | {"is_synthetic", "data_split"}


def test_migration_148_columns_equal_the_export_fields():
    columns = _sql_columns(MIGRATION.read_text())
    exported = _export_record_keys()
    assert set(columns) - _SERVER_COLUMNS == exported, {
        "sql_only": sorted(set(columns) - _SERVER_COLUMNS - exported),
        "export_only": sorted(exported - set(columns)),
    }


def test_migration_148_types_and_constraints():
    sql = MIGRATION.read_text()
    columns = _sql_columns(sql)
    assert f"CREATE TABLE IF NOT EXISTS public.{TABLE}" in sql
    # the header comment also says "patient_id" — match the column line, not the prose
    assert re.search(r"^\s*patient_id\s+VARCHAR\(40\) PRIMARY KEY,", sql, re.MULTILINE)
    assert columns["is_synthetic"] == "BOOLEAN"
    assert re.search(r"^\s*is_synthetic\s+BOOLEAN NOT NULL DEFAULT false,", sql, re.MULTILINE)
    for col in (TREATMENT_COL, TARGET_PERSISTENT_G28, "discontinued_180d", SWITCH_FLAG, "persistent_at_180d"):
        assert columns[col] == "SMALLINT", col
        assert f"CHECK ({col} IN (0, 1))" in sql, col
    for col in ("gdr_cd", "payer_category", "payer_product", "payer_bus", "charlson_risk_band", "elixhauser_risk_band", "geographic_region", "index_biologic_brand"):
        assert columns[col] == "TEXT", col
    for col in ("index_date", "journey_start_date", "treatment_start_date"):
        assert columns[col] == "DATE", col
    assert f"idx_{TABLE}_treatment" in sql and f"idx_{TABLE}_outcomes" in sql
    statements = "\n".join(l for l in sql.splitlines() if not l.strip().startswith("--"))
    assert "BEGIN;" not in statements.upper() and "COMMIT;" not in statements.upper()  # the runner wraps each file


def test_rollback_148_drops_only_the_new_table():
    sql = ROLLBACK.read_text()
    assert f"DROP TABLE IF EXISTS public.{TABLE};" in sql
    assert sql.count("DROP") == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest -n 0 -p no:cacheprovider tests/unit/test_scripts/test_optum_causal_cohort_contract.py -q`
Expected: `FileNotFoundError` on the migration.

- [ ] **Step 3: Write the migration**

Create `database/migrations/148_optum_biologic_persistence_causal.sql`:

```sql
-- Migration 148: optum_biologic_persistence_causal — the REAL-claims causal cohort
--
-- WHY: no causal DAG was ever built and no causal effect ever estimated on real
-- claims data (docs/demos/results/2026-09-22_discovery_real_claims_disproof/).
-- The Optum mart converter drops index_biologic_brand on purpose — the manifest
-- declares it a post-index mart_treatment column that would leak a PREDICTION
-- target. For causal estimation that column IS the treatment. This table holds
-- the separate causal export (scripts/convert_optum_mart.py --cohort
-- persistence_causal): Dupixent vs Xolair initiators (the only contrast the drop
-- observes), the 64 pre-index baseline features (MART_SAFE_FEATURES, measured
-- at the diagnosis index which precedes treatment start), the binary treatment
-- and four outcomes. The prediction tables are untouched.
--
-- Outcome choice (owner decision, spec 2026-09-22 §7): persistent_at_180d_g28 is
-- PRIMARY — covered through day 152 (a 28-day grace) AND no internal gap > 60 d.
-- The shipped persistent_at_180d is days-supply sensitive per brand (14-day
-- Dupixent fills vs 28-45-day Xolair fills; the -17.6 pp raw gap collapses with a
-- 14-day grace and inverts from 28 d) and is stored only so it can be reported
-- ALONGSIDE that sweep, never as an effect.
--
-- One column per exported field; tests/unit/test_scripts/
-- test_optum_causal_cohort_contract.py pins that equality against the export.
-- Grain = patient (patient_id PK; the loader upserts on it — idempotent).
-- Additive + idempotent (IF NOT EXISTS). is_synthetic defaults false: every
-- row here is real claims data and the real-mode provenance filter keeps it.
--
-- NOTE: no BEGIN/COMMIT here -- the migration runner wraps each file.

CREATE TABLE IF NOT EXISTS public.optum_biologic_persistence_causal (
    -- identity + journey metadata (the converter's record shape)
    patient_id                      VARCHAR(40) PRIMARY KEY,
    patient_journey_id              VARCHAR(40) NOT NULL,
    patient_hash                    VARCHAR(20) NOT NULL,
    index_date                      DATE NOT NULL,
    journey_start_date              DATE NOT NULL,
    journey_status                  TEXT NOT NULL DEFAULT 'active',
    discontinuation_flag            SMALLINT NOT NULL DEFAULT 0,
    data_quality_score              NUMERIC,
    data_split                      TEXT,
    -- treatment (Lane A): the contrast the drop observes
    index_biologic_brand            TEXT NOT NULL,
    treatment_dupixent              SMALLINT NOT NULL CHECK (treatment_dupixent IN (0, 1)),
    treatment_start_date            DATE NOT NULL,
    -- outcomes
    persistent_at_180d_g28          SMALLINT NOT NULL CHECK (persistent_at_180d_g28 IN (0, 1)),
    discontinued_180d               SMALLINT NOT NULL CHECK (discontinued_180d IN (0, 1)),
    biologic_switch_180d_flag       SMALLINT NOT NULL CHECK (biologic_switch_180d_flag IN (0, 1)),
    persistent_at_180d              SMALLINT NOT NULL CHECK (persistent_at_180d IN (0, 1)),
    -- 64 pre-index baseline features (MART_SAFE_FEATURES): demographics + payer
    age_at_index                    NUMERIC,
    gdr_cd                          TEXT,
    payer_category                  TEXT,
    payer_product                   TEXT,
    payer_bus                       TEXT,
    health_exchange_flag            INTEGER,
    lis_dual_flag                   INTEGER,
    enrollment_duration_days        INTEGER,
    geographic_region               TEXT,
    -- comorbidity summaries
    charlson_score                  INTEGER,
    charlson_risk_band              TEXT,
    elixhauser_van_walraven_score   INTEGER,
    elixhauser_risk_band            TEXT,
    comorbidity_diag_distinct_count INTEGER,
    comorbidity_diag_claim_count    INTEGER,
    high_comorbidity_burden_flag    INTEGER,
    -- Charlson components
    cci_mi                          INTEGER,
    cci_chf                         INTEGER,
    cci_pvd                         INTEGER,
    cci_cerebrovascular             INTEGER,
    cci_dementia                    INTEGER,
    cci_chronic_pulmonary           INTEGER,
    cci_rheumatic                   INTEGER,
    cci_peptic_ulcer                INTEGER,
    cci_mild_liver                  INTEGER,
    cci_diabetes_no_complication    INTEGER,
    cci_diabetes_complication       INTEGER,
    cci_paraplegia                  INTEGER,
    cci_renal                       INTEGER,
    cci_malignancy                  INTEGER,
    cci_severe_liver                INTEGER,
    cci_metastatic_cancer           INTEGER,
    cci_hiv                         INTEGER,
    -- Elixhauser components
    elx_chf                         INTEGER,
    elx_cardiac_arrhythmia          INTEGER,
    elx_valvular_disease            INTEGER,
    elx_pulmonary_circulation       INTEGER,
    elx_pvd                         INTEGER,
    elx_hypertension_uncomplicated  INTEGER,
    elx_hypertension_complicated    INTEGER,
    elx_paralysis                   INTEGER,
    elx_other_neurological          INTEGER,
    elx_chronic_pulmonary           INTEGER,
    elx_diabetes_uncomplicated      INTEGER,
    elx_diabetes_complicated        INTEGER,
    elx_hypothyroidism              INTEGER,
    elx_renal_failure               INTEGER,
    elx_liver_disease               INTEGER,
    elx_peptic_ulcer                INTEGER,
    elx_aids_hiv                    INTEGER,
    elx_lymphoma                    INTEGER,
    elx_metastatic_cancer           INTEGER,
    elx_solid_tumor_no_metastasis   INTEGER,
    elx_rheumatoid_collagen         INTEGER,
    elx_coagulopathy                INTEGER,
    elx_obesity                     INTEGER,
    elx_weight_loss                 INTEGER,
    elx_fluid_electrolyte           INTEGER,
    elx_blood_loss_anemia           INTEGER,
    elx_deficiency_anemia           INTEGER,
    elx_alcohol_abuse               INTEGER,
    elx_drug_abuse                  INTEGER,
    elx_psychoses                   INTEGER,
    elx_depression                  INTEGER,
    -- provenance + bookkeeping
    is_synthetic                    BOOLEAN NOT NULL DEFAULT false,
    created_at                      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- The causal loader filters is_synthetic and reads the treatment + one outcome
-- per run; the cert's arm-split probe counts by treatment.
CREATE INDEX IF NOT EXISTS idx_optum_biologic_persistence_causal_treatment
    ON public.optum_biologic_persistence_causal (treatment_dupixent, is_synthetic);
CREATE INDEX IF NOT EXISTS idx_optum_biologic_persistence_causal_outcomes
    ON public.optum_biologic_persistence_causal (persistent_at_180d_g28, discontinued_180d, biologic_switch_180d_flag);

COMMENT ON TABLE public.optum_biologic_persistence_causal IS
    'Lane A (2026-09-22): REAL Optum claims causal cohort — Dupixent vs Xolair CSU '
    'escalation-therapy initiators with the 64 pre-index baseline features, the binary '
    'treatment and four outcomes. Written by scripts/load_optum_causal_cohort.py from '
    'the persistence_causal export; read by the causal_impact agent as dataset '
    'optum_biologic_persistence. is_synthetic is false on every row.';
COMMENT ON COLUMN public.optum_biologic_persistence_causal.treatment_dupixent IS
    '1 = DUPIXENT, 0 = XOLAIR (index_biologic_brand). The only contrast the drop observes.';
COMMENT ON COLUMN public.optum_biologic_persistence_causal.persistent_at_180d_g28 IS
    'PRIMARY outcome: covered through day 152 (28-day grace) AND no internal gap > 60 d. '
    'Brand-invariant across a 28-60 d grace (persistence definition disproof 2026-09-22).';
COMMENT ON COLUMN public.optum_biologic_persistence_causal.persistent_at_180d IS
    'The shipped prediction definition (covered through day 180 AND gap <= 60). Days-supply '
    'sensitive per brand (14-d Dupixent vs 28-45-d Xolair fills): report ONLY alongside the '
    'grace sweep, never as an effect.';
COMMENT ON COLUMN public.optum_biologic_persistence_causal.discontinued_180d IS
    'Secondary outcome, as shipped: not covered through day 180 AND a >= 90 d internal or '
    'terminal gap. Brand-robust across the gap sweep.';
```

Create `database/migrations/rollback_148_optum_biologic_persistence_causal.sql`:

```sql
-- Rollback for migration 148 (NOT auto-applied: the runner skips rollback_* files).
-- Drops the Lane A causal cohort table. The rows are reproducible from the
-- persistence_causal export (scripts/convert_optum_mart.py + load_optum_causal_cohort.py).
DROP TABLE IF EXISTS public.optum_biologic_persistence_causal;
```

- [ ] **Step 4: Run the contract test**

Run: `$PY -m pytest -n 0 -p no:cacheprovider tests/unit/test_scripts/test_optum_causal_cohort_contract.py -q`
Expected: `3 passed`. If `test_migration_148_columns_equal_the_export_fields` fails, the assertion message lists `sql_only` / `export_only` — fix the SQL, never the test.

- [ ] **Step 5: Run the migration-runner meta tests**

Run: `$PY -m pytest -n 0 -p no:cacheprovider tests/unit/test_docker/ -q -k "migration or deploy_trigger"`
Expected: all passed (they parse `run_migrations.sh` and the deploy trigger; a new file in `database/migrations/` is in scope).

- [ ] **Step 6: Commit**

```bash
git add database/migrations/148_optum_biologic_persistence_causal.sql database/migrations/rollback_148_optum_biologic_persistence_causal.sql tests/unit/test_scripts/test_optum_causal_cohort_contract.py
git commit -m "feat(db): migration 148 optum_biologic_persistence_causal — one column per exported field (contract-tested)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: The idempotent loader (`--dry-run` default, arm-split verification)

> **Amended after review (commit `6332934d6`):** `--dry-run` is an explicit flag (mutually exclusive with `--execute`; neither = dry run) so the spec's literal command works; `arm_split` / `fetch_live_split` / `verify` also carry `"treatment": {"0": n, "1": n}` keyed on `treatment_dupixent` (the column the causal run reads); the fake client writes only on `.execute()` and returns `count=None` unless `count="exact"` was requested (both regressions mutation-proven caught). The review also established from postgrest 2.27.0's `execute()` that a PostgREST 4xx raises `APIError`, so a partial write aborts with a traceback and never prints VERIFIED.

**Files:**
- Create: `scripts/load_optum_causal_cohort.py`
- Test: `tests/unit/test_scripts/test_load_optum_causal_cohort.py`
- Create: `tests/integration/test_optum_causal_cohort_realdb.py`

- [ ] **Step 1: Write the failing unit tests**

Create `tests/unit/test_scripts/test_load_optum_causal_cohort.py`:

```python
"""scripts/load_optum_causal_cohort.py — parquet -> optum_biologic_persistence_causal.

No DB here: the client is a recording fake. The real-DB arm-split probe lives in
tests/integration/test_optum_causal_cohort_realdb.py (E2I_DB_INTEGRATION=1).
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.load_optum_causal_cohort import (  # noqa: E402
    BATCH_SIZE,
    ON_CONFLICT,
    OUTCOME_COLUMNS,
    TABLE,
    arm_split,
    fetch_live_split,
    load_frame,
    main,
    to_records,
    upsert,
    verify,
)


def _frame(n_x: int = 3, n_d: int = 2) -> pd.DataFrame:
    rows = []
    for i in range(n_x + n_d):
        dup = int(i >= n_x)
        rows.append(
            {
                "patient_journey_id": f"PJ_{i}",
                "patient_id": f"PAT_{i}",
                "patient_hash": f"{i:020x}",
                "index_date": pd.Timestamp("2020-01-01"),
                "journey_start_date": pd.Timestamp("2020-01-01"),
                "journey_status": "active",
                "discontinuation_flag": 0,
                "data_quality_score": 0.98,
                "data_split": "train",
                "index_biologic_brand": "DUPIXENT" if dup else "XOLAIR",
                "treatment_dupixent": dup,
                "treatment_start_date": pd.Timestamp("2020-01-01"),
                "persistent_at_180d_g28": 1 if i % 2 == 0 else 0,
                "discontinued_180d": 0,
                "biologic_switch_180d_flag": 0,
                "persistent_at_180d": 1 if i % 3 == 0 else 0,
                "age_at_index": 50.0 + i,
                "geographic_region": None if i == 0 else "south",
                "charlson_score": np.int64(i),
                "is_synthetic": False,
            }
        )
    return pd.DataFrame(rows)


def _write(tmp_path: Path, df: pd.DataFrame) -> Path:
    p = tmp_path / "cohort.parquet"
    df.to_parquet(p)
    return p


class _FakeQuery:
    def __init__(self, table: "_FakeTable", op: str):
        self._t, self._op, self._filters, self._count = table, op, [], None

    def select(self, cols, count=None):
        self._count = count
        return self

    def eq(self, col, val):
        self._filters.append((col, val))
        return self

    def execute(self):
        rows = [r for r in self._t.rows.values() if all(r.get(c) == v for c, v in self._filters)]
        return type("R", (), {"data": rows, "count": len(rows)})()


class _FakeTable:
    def __init__(self, missing: bool = False):
        self.rows: dict = {}
        self.upserts: list = []
        self.missing = missing

    def select(self, cols, count=None):
        if self.missing:
            raise RuntimeError('relation "optum_biologic_persistence_causal" does not exist')
        return _FakeQuery(self, "select").select(cols, count=count)

    def upsert(self, batch, on_conflict=None):
        self.upserts.append((list(batch), on_conflict))
        for rec in batch:
            self.rows[rec["patient_id"]] = rec
        return _FakeQuery(self, "upsert")


class _FakeClient:
    def __init__(self, missing: bool = False):
        self.t = _FakeTable(missing=missing)
        self.tables: list = []

    def table(self, name):
        self.tables.append(name)
        return self.t


def test_constants():
    assert TABLE == "optum_biologic_persistence_causal"
    assert ON_CONFLICT == "patient_id"
    assert BATCH_SIZE == 500
    assert OUTCOME_COLUMNS == (
        "persistent_at_180d_g28",
        "discontinued_180d",
        "biologic_switch_180d_flag",
        "persistent_at_180d",
    )


def test_load_frame_accepts_the_export(tmp_path):
    df = load_frame(_write(tmp_path, _frame()))
    assert len(df) == 5


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda d: d.assign(is_synthetic=True), "is_synthetic"),
        (lambda d: pd.concat([d, d.iloc[[0]]]), "patient_id"),
        (lambda d: d.drop(columns=["persistent_at_180d_g28"]), "persistent_at_180d_g28"),
        (lambda d: d.assign(treatment_dupixent=1 - d["treatment_dupixent"]), "index_biologic_brand"),
        (lambda d: d.assign(index_biologic_brand="RHAPSIDO"), "index_biologic_brand"),
        (lambda d: d.assign(persistent_at_180d_g28=2), "persistent_at_180d_g28"),
    ],
)
def test_load_frame_fails_loud(tmp_path, mutate, match):
    with pytest.raises(ValueError, match=match):
        load_frame(_write(tmp_path, mutate(_frame())))


def test_arm_split_counts_and_rates():
    split = arm_split(_frame(n_x=3, n_d=2))
    assert split["n"] == 5
    assert split["arms"] == {"XOLAIR": 3, "DUPIXENT": 2}
    assert split["outcome_positives"]["persistent_at_180d_g28"] == {"XOLAIR": 2, "DUPIXENT": 1}
    assert set(split["outcome_positives"]) == set(OUTCOME_COLUMNS)


def test_to_records_is_json_safe_and_deterministic():
    r1, r2 = to_records(_frame()), to_records(_frame())
    assert r1 == r2
    rec = r1[0]
    assert rec["index_date"] == "2020-01-01" and date.fromisoformat(rec["treatment_start_date"])
    assert rec["geographic_region"] is None  # NaN/None -> null, never the string 'nan'
    assert isinstance(rec["charlson_score"], int) and not isinstance(rec["charlson_score"], np.generic)
    assert isinstance(rec["treatment_dupixent"], int)
    assert rec["is_synthetic"] is False
    assert isinstance(rec["age_at_index"], float)


def test_upsert_batches_on_patient_id():
    client = _FakeClient()
    n = upsert(client, to_records(_frame(n_x=600, n_d=100)), batch_size=500)
    assert n == 700
    assert [len(b) for b, _ in client.t.upserts] == [500, 200]
    assert {oc for _, oc in client.t.upserts} == {ON_CONFLICT}
    assert client.tables and set(client.tables) == {TABLE}
    # idempotent: a second run rewrites the same keys, no growth
    upsert(client, to_records(_frame(n_x=600, n_d=100)), batch_size=500)
    assert len(client.t.rows) == 700


def test_fetch_live_split_counts_by_arm_and_outcome():
    client = _FakeClient()
    upsert(client, to_records(_frame(n_x=3, n_d=2)))
    live = fetch_live_split(client)
    assert live["n"] == 5 and live["arms"] == {"XOLAIR": 3, "DUPIXENT": 2}
    assert live["outcome_positives"]["persistent_at_180d_g28"] == {"XOLAIR": 2, "DUPIXENT": 1}


def test_fetch_live_split_reports_a_missing_table_as_none():
    assert fetch_live_split(_FakeClient(missing=True)) is None


def test_verify_verdicts():
    a = arm_split(_frame(n_x=3, n_d=2))
    assert verify(a, a) == []
    b = arm_split(_frame(n_x=3, n_d=1))
    problems = verify(a, b)
    assert any("DUPIXENT" in p for p in problems) and any("n" in p for p in problems)


def test_main_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    path = _write(tmp_path, _frame())
    client = _FakeClient()
    import scripts.load_optum_causal_cohort as mod

    monkeypatch.setattr(mod, "_client", lambda: client)
    assert main(["--input", str(path)]) == 0
    assert client.t.upserts == []
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "XOLAIR" in out


def test_main_execute_loads_then_verifies(tmp_path, monkeypatch, capsys):
    path = _write(tmp_path, _frame())
    client = _FakeClient()
    import scripts.load_optum_causal_cohort as mod

    monkeypatch.setattr(mod, "_client", lambda: client)
    assert main(["--input", str(path), "--execute"]) == 0
    assert len(client.t.rows) == 5
    assert "VERIFIED" in capsys.readouterr().out


def test_main_execute_returns_nonzero_when_live_split_disagrees(tmp_path, monkeypatch, capsys):
    path = _write(tmp_path, _frame())
    client = _FakeClient()
    import scripts.load_optum_causal_cohort as mod

    monkeypatch.setattr(mod, "_client", lambda: client)
    # a stale extra row the parquet does not carry
    client.t.rows["PAT_stale"] = {"patient_id": "PAT_stale", "index_biologic_brand": "XOLAIR", "treatment_dupixent": 0, **{c: 0 for c in OUTCOME_COLUMNS}}
    assert main(["--input", str(path), "--execute"]) == 1
    assert "MISMATCH" in capsys.readouterr().out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest -n 0 -p no:cacheprovider tests/unit/test_scripts/test_load_optum_causal_cohort.py -q`
Expected: `ModuleNotFoundError: No module named 'scripts.load_optum_causal_cohort'`.

- [ ] **Step 3: Write the loader**

Create `scripts/load_optum_causal_cohort.py`:

```python
#!/usr/bin/env python3
"""Idempotent loader for ``optum_biologic_persistence_causal`` (migration 148).

Lane A (spec docs/superpowers/specs/2026-09-22-real-data-causal-estimation-design.md
§3A.2): reads the causal export written by
``scripts/convert_optum_mart.py --cohort persistence_causal`` and upserts it on
``patient_id`` through the PostgREST client (the same path
``scripts/load_hcp_brand_adoption.py`` uses — the worker containers mount no
``data/rwd`` volume, so this runs on the host venv).

Fail-loud validation BEFORE any write: required columns present, every row
``is_synthetic == False``, ``patient_id`` unique, the treatment coding agrees with
``index_biologic_brand`` and only the observed two-arm contrast is present, every
outcome is 0/1. After ``--execute`` the live table's arm split and per-outcome
positives are re-read and compared with the parquet; a disagreement is printed as
MISMATCH and the exit code is 1 — the load is never reported as verified on the
strength of the write call alone.

USAGE
-----
    # DEFAULT: dry run. Validates the parquet, prints the arm split it WOULD
    # write and (if reachable) the live table's current split. Writes nothing.
    python -m scripts.load_optum_causal_cohort --input data/rwd/mart/persistence_causal/e2i_causal_v1_biologic_persistence.parquet

    # WRITE PATH — owner-GO step (spec §7 records the GO for the production load).
    python -m scripts.load_optum_causal_cohort --input <parquet> --execute
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TABLE = "optum_biologic_persistence_causal"
ON_CONFLICT = "patient_id"
BATCH_SIZE = 500
DEFAULT_INPUT = "data/rwd/mart/persistence_causal/e2i_causal_v1_biologic_persistence.parquet"
TREATMENT = "treatment_dupixent"
BRAND = "index_biologic_brand"
ARMS = ("XOLAIR", "DUPIXENT")
OUTCOME_COLUMNS = (
    "persistent_at_180d_g28",
    "discontinued_180d",
    "biologic_switch_180d_flag",
    "persistent_at_180d",
)
REQUIRED_COLUMNS = (
    "patient_id",
    "patient_journey_id",
    "patient_hash",
    "index_date",
    "journey_start_date",
    BRAND,
    TREATMENT,
    "treatment_start_date",
    *OUTCOME_COLUMNS,
    "is_synthetic",
)


# ---------------------------------------------------------------------------
# Validation (pure)
# ---------------------------------------------------------------------------


def load_frame(path: Path | str) -> pd.DataFrame:
    """Read the export and refuse anything that is not the causal cohort contract."""
    df = pd.read_parquet(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"export is missing required column(s) {missing}")
    if df["is_synthetic"].astype(bool).any():
        n = int(df["is_synthetic"].astype(bool).sum())
        raise ValueError(f"{n} row(s) carry is_synthetic=True; the causal cohort is real claims data only")
    dups = df["patient_id"].duplicated()
    if dups.any():
        raise ValueError(f"patient_id is not unique: {int(dups.sum())} duplicate(s)")
    off_contrast = ~df[BRAND].isin(ARMS)
    if off_contrast.any():
        raise ValueError(
            f"{int(off_contrast.sum())} row(s) have an index_biologic_brand outside {ARMS}: "
            f"{sorted(df.loc[off_contrast, BRAND].astype(str).unique())}"
        )
    expected = df[BRAND].eq("DUPIXENT").astype(int)
    if not df[TREATMENT].astype(int).eq(expected).all():
        raise ValueError(f"{TREATMENT} disagrees with index_biologic_brand on some rows")
    for col in OUTCOME_COLUMNS:
        values = set(pd.unique(df[col].dropna()))
        if not values <= {0, 1}:
            raise ValueError(f"{col} is not 0/1: found {sorted(values)}")
        if df[col].isna().any():
            raise ValueError(f"{col} has NULLs; the export fills the switch flag and derives the rest")
    return df


def arm_split(df: pd.DataFrame) -> Dict[str, Any]:
    """Counts by arm and per-outcome positives by arm — the verification unit."""
    arms = {arm: int((df[BRAND] == arm).sum()) for arm in ARMS}
    positives = {
        col: {arm: int(df.loc[df[BRAND] == arm, col].astype(int).sum()) for arm in ARMS}
        for col in OUTCOME_COLUMNS
    }
    return {"n": int(len(df)), "arms": arms, "outcome_positives": positives}


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.date().isoformat() if not pd.isna(value) else None
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, float):
        return None if math.isnan(value) else value
    return value


def to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """DataFrame -> JSON-safe upsert records (dates -> 'YYYY-MM-DD', NaN -> null,
    numpy scalars -> python). Deterministic for a given frame."""
    out: List[Dict[str, Any]] = []
    for rec in df.to_dict(orient="records"):
        out.append({k: _json_safe(v) for k, v in rec.items()})
    return out


# ---------------------------------------------------------------------------
# DB (only reachable with a client)
# ---------------------------------------------------------------------------


def _client() -> Any:
    from src.memory.services.factories import get_supabase_client

    return get_supabase_client()


def upsert(client: Any, records: List[Dict[str, Any]], *, batch_size: int = BATCH_SIZE) -> int:
    """Batched idempotent upsert on ``patient_id``. Returns rows written."""
    written = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        client.table(TABLE).upsert(batch, on_conflict=ON_CONFLICT).execute()
        written += len(batch)
        logger.info("  upserted %d/%d rows", written, len(records))
    return written


def fetch_live_split(client: Any) -> Optional[Dict[str, Any]]:
    """The live table's arm split via exact counts. None when the table is unreachable
    (e.g. migration 148 not yet applied) — the caller reports that, never a zero."""
    try:
        arms: Dict[str, int] = {}
        positives: Dict[str, Dict[str, int]] = {col: {} for col in OUTCOME_COLUMNS}
        for arm in ARMS:
            arms[arm] = int(
                client.table(TABLE).select("patient_id", count="exact").eq(BRAND, arm).execute().count
            )
            for col in OUTCOME_COLUMNS:
                positives[col][arm] = int(
                    client.table(TABLE)
                    .select("patient_id", count="exact")
                    .eq(BRAND, arm)
                    .eq(col, 1)
                    .execute()
                    .count
                )
        total = int(client.table(TABLE).select("patient_id", count="exact").execute().count)
        return {"n": total, "arms": arms, "outcome_positives": positives}
    except Exception as e:  # noqa: BLE001 — a missing relation / store hiccup is reported, not hidden
        logger.warning("Could not read live %s: %s", TABLE, e)
        return None


def verify(expected: Dict[str, Any], live: Dict[str, Any]) -> List[str]:
    """Every disagreement between the parquet split and the live split, as text."""
    problems: List[str] = []
    if expected["n"] != live["n"]:
        problems.append(f"n: parquet {expected['n']} vs live {live['n']}")
    for arm in ARMS:
        if expected["arms"][arm] != live["arms"][arm]:
            problems.append(f"arm {arm}: parquet {expected['arms'][arm]} vs live {live['arms'][arm]}")
        for col in OUTCOME_COLUMNS:
            e, l = expected["outcome_positives"][col][arm], live["outcome_positives"][col][arm]
            if e != l:
                problems.append(f"{col} positives in {arm}: parquet {e} vs live {l}")
    return problems


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _print_split(label: str, split: Dict[str, Any]) -> None:
    print(f"{label}: n={split['n']} arms={split['arms']}")
    for col in OUTCOME_COLUMNS:
        pos = split["outcome_positives"][col]
        rates = {
            arm: (round(pos[arm] / split["arms"][arm], 4) if split["arms"][arm] else None) for arm in ARMS
        }
        print(f"  {col}: positives={pos} rate={rates}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="causal export parquet")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="WRITE PATH: upsert the rows into the live table. Omit (default) for a dry run.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args(argv)
    dry_run = not args.execute

    print("=" * 70)
    print(f"{TABLE} loader ({'DRY RUN' if dry_run else 'EXECUTE'})  input={args.input}")
    print("=" * 70)
    df = load_frame(args.input)
    expected = arm_split(df)
    _print_split("WOULD WRITE" if dry_run else "WRITING", expected)

    client = None
    try:
        client = _client()
    except Exception as e:  # noqa: BLE001 — no client => dry-run still useful
        logger.warning("No Supabase client (%s).", e)

    live_before = fetch_live_split(client) if client is not None else None
    if live_before is None:
        print("LIVE TABLE: unreachable (migration 148 not applied, or no client)")
    else:
        _print_split("LIVE BEFORE", live_before)

    if dry_run:
        print("DRY RUN complete. No rows written. Re-run with --execute to write.")
        return 0
    if client is None:
        print("ERROR: cannot --execute without a Supabase client.")
        return 1

    n = upsert(client, to_records(df), batch_size=args.batch_size)
    print(f"EXECUTE: upserted {n} rows into {TABLE} (idempotent on {ON_CONFLICT}).")
    live_after = fetch_live_split(client)
    if live_after is None:
        print("MISMATCH: could not re-read the live table after the write.")
        return 1
    _print_split("LIVE AFTER", live_after)
    problems = verify(expected, live_after)
    if problems:
        print("MISMATCH between the parquet and the live table:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("VERIFIED: live arm split and per-outcome positives equal the parquet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the unit tests**

Run: `$PY -m pytest -n 0 -p no:cacheprovider tests/unit/test_scripts/test_load_optum_causal_cohort.py -q`
Expected: `17 passed` (the fail-loud case is parametrised six ways).

- [ ] **Step 5: Write the real-DB probe**

Create `tests/integration/test_optum_causal_cohort_realdb.py`:

```python
"""Lane A real-DB probe (spec §3A gates): the loaded table's arm split and
per-outcome positives equal the causal export parquet.

Gate: ``E2I_DB_INTEGRATION=1``; skipped when the parquet is absent (data/ is
gitignored — it exists only on the droplet's main checkout) or when migration 148
has not been applied (the loader's live read returns None).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_DB_INTEGRATION") != "1",
    reason="E2I_DB_INTEGRATION!=1; set to 1 to run against the real Supabase DB.",
)

_PARQUET = Path(
    os.getenv(
        "E2I_OPTUM_CAUSAL_PARQUET",
        "/home/enunez/Projects/e2i_causal_analytics/data/rwd/mart/persistence_causal/"
        "e2i_causal_v1_biologic_persistence.parquet",
    )
)


def test_live_arm_split_equals_the_export():
    if not _PARQUET.exists():
        pytest.skip(f"causal export not present at {_PARQUET}")
    from scripts.load_optum_causal_cohort import _client, arm_split, fetch_live_split, load_frame, verify

    expected = arm_split(load_frame(_PARQUET))
    live = fetch_live_split(_client())
    if live is None:
        pytest.skip("optum_biologic_persistence_causal unreachable (migration 148 not applied?)")
    assert live["n"] > 0, "live table is empty — the owner-GO load has not run"
    assert verify(expected, live) == []
```

Run: `E2I_DB_INTEGRATION=1 $PY -m pytest -n 0 -p no:cacheprovider tests/integration/test_optum_causal_cohort_realdb.py -q -rs`
Expected NOW (before the export/migration exist): `1 skipped` with the parquet-absent reason. It becomes a real assertion in Task 12.

- [ ] **Step 6: Ruff, then commit**

```bash
$PY -m ruff check --no-cache scripts/load_optum_causal_cohort.py tests/unit/test_scripts/test_load_optum_causal_cohort.py tests/integration/test_optum_causal_cohort_realdb.py && $PY -m ruff format --check scripts/load_optum_causal_cohort.py tests/unit/test_scripts/test_load_optum_causal_cohort.py tests/integration/test_optum_causal_cohort_realdb.py
git add scripts/load_optum_causal_cohort.py tests/unit/test_scripts/test_load_optum_causal_cohort.py tests/integration/test_optum_causal_cohort_realdb.py
git commit -m "feat(scripts): load_optum_causal_cohort — dry-run default, upsert on patient_id, live arm-split verification

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Register `optum_biologic_persistence` in the causal dataset registry

> **Amended after review (commits `fc2f68e8f`, `b6096abf0`, `d47a16525`):** the closed-list pin `tests/unit/test_repositories/test_has_provenance_family_894.py` gained the new table (27 → 28); `datasets.py`'s manifest import sits where ruff's isort wants it; **two behaviour fixes in `loaders.py`**: (1) `_load_agent_estimation_frame` refuses a constant treatment with a 400 right after the frame is built (a one-brand scope on this dataset would otherwise reach DoWhy, which returns a finite estimate on a constant treatment while `refutation.py`'s `nunique()==2` check silently switches to the continuous path), and the registry comment describes that guard; (2) `_one_hot_categoricals` emits a `<col>=__missing__` dummy when a categorical has NULLs (8.1 % of the real cohort's `geographic_region`) instead of collapsing NULL into the drop_first reference level — so the real run resolves 77 covariates, not 76; synthetic datasets have no NULL categoricals and are unchanged. Known, not fixed here: the discovery leaderboard's `_DISCOVERY_ROW_CAP` (5,000) would subsample this 15,209-row cohort if a `causal_paths` row ever names `treatment_dupixent` (none does today).

**Files:**
- Modify: `src/api/routes/causal/datasets.py` (`_CAUSAL_DATASET_SPECS` line 60 block end ~line 217; `_CAUSAL_NUMERIC_COLUMNS` line 413; `_CAUSAL_BRAND_COLUMN` line 472; `_CAUSAL_PHYSICAL_TABLE` line 571; `_CAUSAL_CATEGORICAL_COLUMNS` line 580)
- Modify: `src/repositories/provenance.py` (`PROVENANCE_TAGGED_TABLES`, line ~45)
- Test: `tests/unit/test_api/test_causal_optum_dataset_registry.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_api/test_causal_optum_dataset_registry.py`:

```python
"""Lane A: the REAL Optum causal dataset ``optum_biologic_persistence`` in the
registry, the loader's coercion/one-hot on its shape, the covariate role gate,
and the per-dataset ``auto_discover`` default (spec 2026-09-22 §3A.3-4).

Fake-client seams as in test_causal_agent_analyze_negative_control_2007.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pandas as pd
import pytest
from fastapi import BackgroundTasks, HTTPException

from src.api.routes.causal import agent as causal_routes
from src.api.routes.causal.datasets import (
    _ALL_CLINICAL_COVARIATES,
    _CAUSAL_BRAND_COLUMN,
    _CAUSAL_CATEGORICAL_COLUMNS,
    _CAUSAL_DATASET_SPECS,
    _CAUSAL_DISCOVERY_DEFAULT_OFF,
    _CAUSAL_NEGATIVE_CONTROL_OUTCOMES,
    _CAUSAL_NUMERIC_COLUMNS,
    _CAUSAL_PHYSICAL_TABLE,
    _JOIN_DATASETS,
    _brand_scoped_covariates,
    _default_auto_discover,
    _negative_control_outcome,
)
from src.api.routes.causal.loaders import _load_agent_estimation_frame
from src.api.schemas.causal import AgentCausalAnalysisRequest
from src.data.manifests import MART_SAFE_FEATURES
from src.repositories.provenance import PROVENANCE_TAGGED_TABLES

pytestmark = pytest.mark.unit

DATASET = "optum_biologic_persistence"
TABLE = "optum_biologic_persistence_causal"
TREATMENT = "treatment_dupixent"
OUTCOMES = ("persistent_at_180d_g28", "discontinued_180d", "biologic_switch_180d_flag", "persistent_at_180d")
CATEGORICALS = {
    "gdr_cd",
    "payer_category",
    "payer_product",
    "payer_bus",
    "charlson_risk_band",
    "elixhauser_risk_band",
    "geographic_region",
}
_CLIENT_FACTORY = "src.memory.services.factories.get_async_supabase_client"


# ---------------------------------------------------------------------------
# registry pins
# ---------------------------------------------------------------------------


def test_spec_treatment_outcomes_and_covariates():
    spec = _CAUSAL_DATASET_SPECS[DATASET]
    assert spec["treatment"] == [TREATMENT]
    assert spec["outcome"] == list(OUTCOMES)  # primary first
    assert spec["covariate"] == list(MART_SAFE_FEATURES)
    assert len(spec["covariate"]) == 64
    assert "randomized_treatment" not in spec  # observational: keeps the unmeasured-confounding gate
    assert "baseline_covariate" not in spec


def test_physical_table_brand_column_and_single_table_path():
    assert _CAUSAL_PHYSICAL_TABLE[DATASET] == TABLE
    assert _CAUSAL_BRAND_COLUMN[DATASET] == "index_biologic_brand"
    assert DATASET not in _JOIN_DATASETS
    assert TABLE in PROVENANCE_TAGGED_TABLES


def test_every_baseline_feature_has_exactly_one_coercion_role():
    numeric = _CAUSAL_NUMERIC_COLUMNS[DATASET]
    categorical = _CAUSAL_CATEGORICAL_COLUMNS[DATASET]
    assert categorical == CATEGORICALS
    assert not (numeric & categorical)
    assert set(MART_SAFE_FEATURES) == (numeric | categorical) - {TREATMENT, *OUTCOMES}
    assert {TREATMENT, *OUTCOMES} <= numeric


def test_no_negative_control_is_declared_until_measured_on_this_source():
    assert DATASET not in _CAUSAL_NEGATIVE_CONTROL_OUTCOMES
    assert _negative_control_outcome(DATASET, TREATMENT, OUTCOMES[0]) is None


def test_brand_scoping_passes_every_baseline_feature_through():
    # none of the 64 names is a synthetic clinical biomarker, so brand=None keeps all
    assert not (set(MART_SAFE_FEATURES) & _ALL_CLINICAL_COVARIATES)
    assert _brand_scoped_covariates(list(MART_SAFE_FEATURES), None) == list(MART_SAFE_FEATURES)


# ---------------------------------------------------------------------------
# auto_discover default
# ---------------------------------------------------------------------------


def test_discovery_is_off_by_default_only_for_the_real_dataset():
    assert _CAUSAL_DISCOVERY_DEFAULT_OFF == frozenset({DATASET})
    assert _default_auto_discover(DATASET) is False
    assert _default_auto_discover("patient_journeys") is True
    assert _default_auto_discover(None) is True


# ---------------------------------------------------------------------------
# loader: coercion + one-hot on this dataset's shape (fake client)
# ---------------------------------------------------------------------------


class _FakeQuery:
    def __init__(self, rows, selected):
        self._rows, self._selected = rows, selected

    def select(self, cols, *_a, **_k):
        self._selected.append(cols)
        return self

    def eq(self, *_a, **_k):
        return self

    def limit(self, *_a, **_k):
        return self

    async def execute(self):
        return type("R", (), {"data": self._rows})()


class _FakeClient:
    def __init__(self, rows):
        self._rows, self.selected, self.tables = rows, [], []

    def table(self, name, *_a, **_k):
        self.tables.append(name)
        return _FakeQuery(self._rows, self.selected)


def _rows():
    return [
        {TREATMENT: 0, "persistent_at_180d_g28": 1, "age_at_index": 61, "payer_category": "commercial", "gdr_cd": "F", "geographic_region": None},
        {TREATMENT: 1, "persistent_at_180d_g28": 0, "age_at_index": 34, "payer_category": "medicare", "gdr_cd": "M", "geographic_region": "south"},
        {TREATMENT: 1, "persistent_at_180d_g28": 1, "age_at_index": 45, "payer_category": "medicare_lis_dual", "gdr_cd": "F", "geographic_region": "west"},
    ]


@pytest.mark.asyncio
async def test_loader_reads_the_physical_table_and_one_hots_the_categoricals(monkeypatch):
    client = _FakeClient(_rows())
    monkeypatch.setattr(_CLIENT_FACTORY, AsyncMock(return_value=client))
    frame, cols = await _load_agent_estimation_frame(
        dataset=DATASET,
        treatment_var=TREATMENT,
        outcome_var="persistent_at_180d_g28",
        covariates=["age_at_index", "payer_category", "gdr_cd", "geographic_region"],
        limit=10,
    )
    assert client.tables == [TABLE]
    assert frame[TREATMENT].tolist() == [0.0, 1.0, 1.0]
    assert frame["age_at_index"].dtype.kind == "f"
    assert "payer_category" not in frame.columns and "payer_category=medicare" in frame.columns
    assert "gdr_cd=M" in cols and "geographic_region=west" in cols
    assert set(cols) == {TREATMENT, "persistent_at_180d_g28", "age_at_index", "payer_category=medicare", "payer_category=medicare_lis_dual", "gdr_cd=M", "geographic_region=west"}


@pytest.mark.asyncio
async def test_loader_rejects_an_outcome_in_the_covariate_slot(monkeypatch):
    monkeypatch.setattr(_CLIENT_FACTORY, AsyncMock(return_value=_FakeClient(_rows())))
    with pytest.raises(HTTPException) as exc:
        await _load_agent_estimation_frame(
            dataset=DATASET,
            treatment_var=TREATMENT,
            outcome_var="persistent_at_180d_g28",
            covariates=["discontinued_180d"],
            limit=10,
        )
    assert exc.value.status_code == 400 and "discontinued_180d" in exc.value.detail


# ---------------------------------------------------------------------------
# submit endpoint: auto_discover resolution
# ---------------------------------------------------------------------------


class _MemStore:
    def __init__(self) -> None:
        self._d: dict = {}

    async def get(self, key):
        return self._d.get(key)

    async def set(self, key, value):
        self._d[key] = value


def _stub_submit(monkeypatch):
    df = pd.DataFrame({TREATMENT: [0.0, 1.0, 1.0], "persistent_at_180d_g28": [1.0, 0.0, 1.0], "age_at_index": [61.0, 34.0, 45.0]})
    monkeypatch.setattr(
        causal_routes,
        "_load_agent_estimation_frame",
        AsyncMock(return_value=(df, [TREATMENT, "persistent_at_180d_g28", "age_at_index"])),
    )
    monkeypatch.setattr(causal_routes, "_agent_analysis_store", _MemStore())


async def _submit(monkeypatch, **request_kwargs):
    _stub_submit(monkeypatch)
    tasks = BackgroundTasks()
    req = AgentCausalAnalysisRequest(**request_kwargs)
    resp = await causal_routes.run_causal_agent_analysis(req, tasks, user={"role": "analyst"})
    task_request = tasks.tasks[0].args[1]
    return resp, task_request


@pytest.mark.asyncio
async def test_submit_defaults_discovery_off_for_the_real_dataset(monkeypatch):
    resp, task_request = await _submit(
        monkeypatch, treatment_var=TREATMENT, outcome_var="persistent_at_180d_g28", dataset=DATASET, covariates=["age_at_index"]
    )
    assert task_request.auto_discover is False
    assert any("discovery" in w.lower() and DATASET in w for w in resp.warnings)


@pytest.mark.asyncio
async def test_submit_honors_an_explicit_opt_in(monkeypatch):
    _, task_request = await _submit(
        monkeypatch, treatment_var=TREATMENT, outcome_var="persistent_at_180d_g28", dataset=DATASET, covariates=["age_at_index"], auto_discover=True
    )
    assert task_request.auto_discover is True


@pytest.mark.asyncio
async def test_submit_keeps_the_schema_default_for_synthetic_datasets(monkeypatch):
    _, task_request = await _submit(
        monkeypatch, treatment_var="treatment_arm", outcome_var="persistent_180d", dataset="patient_journeys", covariates=["disease_severity"]
    )
    assert task_request.auto_discover is True


def test_discovery_leaderboard_uses_the_per_dataset_default():
    # A source pin (the leaderboard job is not unit-runnable): the hard-coded
    # ``auto_discover=True`` must be gone from discovery.py.
    import inspect

    from src.api.routes.causal import discovery

    src = inspect.getsource(discovery)
    assert "auto_discover=True" not in src
    assert "auto_discover=_default_auto_discover(dataset)" in src
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest -n 0 -p no:cacheprovider tests/unit/test_api/test_causal_optum_dataset_registry.py -q`
Expected: `ImportError: cannot import name '_CAUSAL_DISCOVERY_DEFAULT_OFF'`.

- [ ] **Step 3: Add the registry entries to `datasets.py`**

At the top imports (after `from src.repositories.provenance import apply_provenance_filter`, line 24), add:

```python
from src.data.manifests import MART_SAFE_FEATURES
```

Inside `_CAUSAL_DATASET_SPECS`, after the `"nba_triggers": {...}` entry (before the closing `}` at ~line 217), add:

```python
    # Lane A (spec 2026-09-22 §3A.3): the REAL Optum claims causal cohort —
    # Dupixent vs Xolair CSU escalation-therapy initiators (n = 15,209; XOLAIR
    # 11,009 / DUPIXENT 4,200), migration 148, loaded by
    # scripts/load_optum_causal_cohort.py from the persistence_causal export.
    # Observational (no randomized_treatment): the unmeasured-confounding gate,
    # the E-value and the refutation suite run unchanged.
    #   * treatment_dupixent: 1 = DUPIXENT, 0 = XOLAIR — the only contrast the
    #     drop observes (remibrutinib absent).
    #   * outcomes, PRIMARY first: persistent_at_180d_g28 (covered through day
    #     152 AND no gap > 60 d; brand-invariant across a 28-60 d grace),
    #     discontinued_180d (brand-robust secondary), biologic_switch_180d_flag,
    #     and the SHIPPED persistent_at_180d — days-supply sensitive per brand
    #     (14-d Dupixent vs 28-45-d Xolair fills; the -17.6 pp raw gap is a
    #     measurement artefact), reported only alongside the grace sweep
    #     (docs/demos/results/2026-09-22_persistence_definition_disproof/).
    #   * covariates: the 64 pre-index baseline features the mart manifest
    #     allow-lists (measured at the diagnosis index, which precedes treatment
    #     start). Seven are text and one-hot (see _CAUSAL_CATEGORICAL_COLUMNS).
    # No negative control is declared: the omitted-confounder experiment the
    # _CAUSAL_NEGATIVE_CONTROL_OUTCOMES note mandates per data source has not
    # been run here, so the runner emits SKIPPED no_negative_control_declared.
    # Structure discovery is OFF by default (_CAUSAL_DISCOVERY_DEFAULT_OFF):
    # measured singular on this frame (rank 45/59) and ~230 s per PC fit; the
    # curated common-cause DAG is the run shape until Lane D lands.
    "optum_biologic_persistence": {
        "treatment": ["treatment_dupixent"],
        "outcome": [
            "persistent_at_180d_g28",
            "discontinued_180d",
            "biologic_switch_180d_flag",
            "persistent_at_180d",
        ],
        "covariate": list(MART_SAFE_FEATURES),
    },
```

Directly after `_DEFAULT_CAUSAL_DATASET = "patient_journeys"` add:

```python
# Lane A: datasets whose API default is auto_discover=False. Guided discovery
# was MEASURED to fail on the real claims frame (singular correlation matrix,
# rank 45/59; ~230 s per PC fit on 43 covariates —
# docs/demos/results/2026-09-22_discovery_real_claims_disproof/). Until Lane D's
# pre-flight lands, the default run uses the curated common-cause DAG. A caller
# that sets auto_discover=True explicitly is honored (PR #2203 then reports
# "could not run: singular…" instead of an empty DAG). The request schema's
# field default stays True (changing it would alter the generated api.ts).
_CAUSAL_DISCOVERY_DEFAULT_OFF: frozenset = frozenset({"optum_biologic_persistence"})


def _default_auto_discover(dataset: Optional[str]) -> bool:
    """The ``auto_discover`` value a request gets when the caller did not set it."""
    return (dataset or _DEFAULT_CAUSAL_DATASET) not in _CAUSAL_DISCOVERY_DEFAULT_OFF
```

Before `_CAUSAL_NUMERIC_COLUMNS` (line ~413) add the categorical set, then reference it in both maps:

```python
# Lane A: the seven TEXT baseline columns of the Optum mart (payer / geography /
# gender / the two comorbidity risk bands), one-hot encoded by the loader.
_OPTUM_BASELINE_CATEGORICALS: frozenset = frozenset(
    {
        "gdr_cd",
        "payer_category",
        "payer_product",
        "payer_bus",
        "charlson_risk_band",
        "elixhauser_risk_band",
        "geographic_region",
    }
)
```

Inside `_CAUSAL_NUMERIC_COLUMNS`, after the `"nba_triggers": {...}` entry add:

```python
    # Lane A: the treatment, the four outcomes and every NON-text baseline feature
    # float-coerce (ints / 0-1 flags / the age). test_causal_optum_dataset_registry
    # locks numeric ∪ categorical == MART_SAFE_FEATURES so a manifest change
    # cannot silently null-coerce a text column.
    "optum_biologic_persistence": {
        "treatment_dupixent",
        "persistent_at_180d_g28",
        "discontinued_180d",
        "biologic_switch_180d_flag",
        "persistent_at_180d",
        *(c for c in MART_SAFE_FEATURES if c not in _OPTUM_BASELINE_CATEGORICALS),
    },
```

In `_CAUSAL_BRAND_COLUMN` add:

```python
    # Lane A: the brand filter IS the treatment label. Scoping to one brand makes
    # the treatment constant and the run fails loudly at estimation — the
    # dropdown offers it because the table has it; the analyst's all-brands
    # default is the causal contrast.
    "optum_biologic_persistence": "index_biologic_brand",
```

Change `_CAUSAL_PHYSICAL_TABLE` to:

```python
_CAUSAL_PHYSICAL_TABLE: Dict[str, str] = {
    "nba_triggers": "triggers",
    # Lane A: migration 148.
    "optum_biologic_persistence": "optum_biologic_persistence_causal",
}
```

In `_CAUSAL_CATEGORICAL_COLUMNS` add:

```python
    "optum_biologic_persistence": set(_OPTUM_BASELINE_CATEGORICALS),
```

In `src/repositories/provenance.py`, inside `PROVENANCE_TAGGED_TABLES` after `"discovered_dags",` add:

```python
        # migrations/148 (Lane A, 2026-09-22)
        "optum_biologic_persistence_causal",
```

- [ ] **Step 4: Run the registry tests (the submit-endpoint ones still fail — that is Task 6)**

Run: `$PY -m pytest -n 0 -p no:cacheprovider tests/unit/test_api/test_causal_optum_dataset_registry.py -q`
Expected: `10 passed, 2 failed` — `test_submit_defaults_discovery_off_for_the_real_dataset` and `test_discovery_leaderboard_uses_the_per_dataset_default` are red until Task 6 (the explicit opt-in and the synthetic-default tests already pass because the schema default flows through unchanged).

- [ ] **Step 5: Run every registry-consuming suite**

Run: `$PY -m pytest -n 0 -p no:cacheprovider tests/unit/test_api/test_causal_triggers_dataset.py tests/unit/test_api/test_causal_geo_encoding.py tests/unit/test_api/test_causal_covariate_roles.py tests/unit/test_api/test_causal_hcp_adoption.py tests/unit/test_api/test_brand_scoped_covariates.py tests/unit/test_api/test_causal_agent_analyze_negative_control_2007.py tests/unit/test_services/test_clinical_context/test_brand_map.py tests/unit/test_kpi/test_causal_datasets_copay.py tests/unit/test_synthetic/test_arm_confounder_contract.py tests/unit/test_api/test_routes/test_segments.py tests/unit/test_repositories/test_has_provenance_family_894.py -q`
Expected: all passed.

- [ ] **Step 6: Commit (registry only; the two red submit tests are committed with Task 6)**

```bash
git add src/api/routes/causal/datasets.py src/repositories/provenance.py
git commit -m "feat(causal): register optum_biologic_persistence — real Optum cohort, 64 baseline covariates, discovery off by default

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Resolve `auto_discover` per dataset at the submit endpoint and the leaderboard

**Files:**
- Modify: `src/api/routes/causal/agent.py` (imports line 42–47; submit body after the `spec is None` 404 at ~line 105)
- Modify: `src/api/routes/causal/discovery.py` (imports line 48–57; line 512)
- Test: `tests/unit/test_api/test_causal_optum_dataset_registry.py` (already written)

- [ ] **Step 1: Confirm the two tests are red**

Run: `$PY -m pytest -n 0 -p no:cacheprovider tests/unit/test_api/test_causal_optum_dataset_registry.py -q -k "submit_defaults or leaderboard"`
Expected: `2 failed`.

- [ ] **Step 2: Resolve the default in `agent.py`**

Extend the import block:

```python
from .datasets import (
    _CAUSAL_DATASET_SPECS,
    _brand_scoped_covariates,
    _default_auto_discover,
    _is_randomized_treatment,
    _negative_control_outcome,
)
```

Directly after the `spec is None` 404 block in `run_causal_agent_analysis` (before the `default_covariates = (...)` statement), add:

```python
    # Lane A: the schema default (auto_discover=True) is right for the synthetic
    # gold standard and MEASURED wrong for the real claims frame (singular
    # correlation matrix; ~230 s per PC fit). When the caller did not set the
    # field, the dataset decides; an explicit value is always honored.
    discovery_note: List[str] = []
    if "auto_discover" not in request.model_fields_set:
        resolved = _default_auto_discover(request.dataset)
        request = request.model_copy(update={"auto_discover": resolved})
        if not resolved:
            discovery_note.append(
                f"Structure discovery is OFF by default for dataset '{request.dataset}' "
                "(measured to fail on this frame; Lane D pending) — the curated "
                "common-cause DAG is used. Pass auto_discover=true to attempt it."
            )
```

In the `pending = AgentCausalAnalysisResponse(...)` construction change the `warnings` line to:

```python
        warnings=[
            "Analysis submitted; poll GET /causal/agent-analyze/{id} for the result.",
            *discovery_note,
        ],
```

- [ ] **Step 3: Resolve the default in `discovery.py`**

Add `_default_auto_discover,` to the `from .datasets import (...)` block (alphabetical, after `_DEFAULT_CAUSAL_DATASET,`), and change line 512 from `auto_discover=True,` to:

```python
                    auto_discover=_default_auto_discover(dataset),
```

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest -n 0 -p no:cacheprovider tests/unit/test_api/test_causal_optum_dataset_registry.py tests/unit/test_api/test_causal_agent_analyze_negative_control_2007.py tests/unit/test_api/test_causal_randomized_flag.py tests/unit/test_api/test_routes/test_causal.py tests/unit/test_api/test_routes/test_causal_heavy_compute_bound.py -q`
Expected: all passed (`test_causal_randomized_flag.py` — if absent, drop it from the command; the 2007 file names it as the seam precedent).

- [ ] **Step 5: Prove the OpenAPI contract did not move**

Run: `$PY -c "import sys; sys.path.insert(0,'.'); from src.api.main import app; import json, hashlib; s=json.dumps(app.openapi()['components']['schemas']['AgentCausalAnalysisRequest'], sort_keys=True); print(hashlib.sha256(s.encode()).hexdigest())"` twice: once on this branch and once with `git stash`-free comparison by running the same command in the MAIN checkout (`cd /home/enunez/Projects/e2i_causal_analytics && …`). Expected: identical hashes (the request schema is untouched; CI's `Verify OpenAPI Types` byte-diffs `api.ts`).

- [ ] **Step 6: Ruff whole-tree (as CI), then commit**

```bash
$PY -m ruff check --no-cache src/ tests/ && $PY -m ruff format --check src/ tests/   # CI lints src/ and tests/ only (backend-tests.yml:120-123); lint changed scripts/ files individually
git add src/api/routes/causal/agent.py src/api/routes/causal/discovery.py tests/unit/test_api/test_causal_optum_dataset_registry.py
git commit -m "feat(causal): auto_discover defaults per dataset — off for optum_biologic_persistence, explicit opt-in honored

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Run the real export (droplet, main checkout's data) and record it

**Files:**
- Create: `docs/demos/results/2026-09-22_optum_biologic_persistence_cert/export_summary.md`
- Writes (gitignored): `/home/enunez/Projects/e2i_causal_analytics/data/rwd/mart/persistence_causal/`

- [ ] **Step 1: Check the box and any peer work first**

```bash
awk '/MemAvailable/ {print $2/1024 " MiB available"}' /proc/meminfo     # need >= 2048
pgrep -fa "convert_optum_mart|load_optum_causal|agent-analyze" || echo "no peer runs"
ls -d /home/enunez/Projects/e2i_causal_analytics/docs/demos/results/2026-09-22* 
```

- [ ] **Step 2: Run the export from the worktree with `-m` (worktree `src`), against main's data**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/real-data-causal
DATA=/home/enunez/Projects/e2i_causal_analytics/data
$PY -c "import src, sys; assert '.worktrees/real-data-causal' in src.__file__, src.__file__; print('ok', src.__file__)"
/usr/bin/time -f "wall %e s  maxrss %M KB" $PY -m scripts.convert_optum_mart \
  --cohort persistence_causal \
  --input  $DATA/rwd/Optum_Parquet/Optum_enriched.parquet \
  --output $DATA/rwd/mart/persistence_causal --verbose 2>&1 | tee /tmp/claude-1000/-home-enunez-Projects-e2i-causal-analytics/*/scratchpad/lane_a_export.log
```

Expected summary (from the disproof's faithful replication): `patients: 15209`, `arms: {'XOLAIR': 11009, 'DUPIXENT': 4200}`. Attrition steps: `patient_panel 814587`, `initiators 24429`, …, `coverage_end_observable 15209`, `two_arm_contrast 15209`.

- [ ] **Step 3: Verify the outcome rates against the disproof table**

```bash
$PY - <<'EOF'
import pandas as pd
df = pd.read_parquet("/home/enunez/Projects/e2i_causal_analytics/data/rwd/mart/persistence_causal/e2i_causal_v1_biologic_persistence.parquet")
print(df.shape, df["is_synthetic"].unique(), df["patient_id"].is_unique)
print(df.groupby("index_biologic_brand")[["persistent_at_180d_g28","discontinued_180d","biologic_switch_180d_flag","persistent_at_180d"]].mean().round(3))
print(df.groupby("index_biologic_brand").size())
EOF
```

Expected (from `docs/demos/results/2026-09-22_persistence_definition_disproof/README.md`): shape `(15209, 81)`; g28 XOLAIR 0.734 / DUPIXENT 0.744; discontinued 0.106 / 0.122; shipped persistence 0.523 / 0.347; switch ≈ 0.010 / 0.012. Any other number is a bug in Task 1 — stop and fix before continuing.

- [ ] **Step 4: Record the evidence**

Create `docs/demos/results/2026-09-22_optum_biologic_persistence_cert/export_summary.md` with: the exact command, wall time and max RSS, the summary dict, the attrition table (copy `attrition_report.csv` in as a Markdown table), the grouped rate table from Step 3, and the sentence "Matches the persistence-definition disproof's replication (n = 15,209; XOLAIR 11,009 / DUPIXENT 4,200; g28 0.734 / 0.744)". Also copy `attrition_report.csv` and `data_dictionary.csv` into that directory.

- [ ] **Step 5: Commit**

```bash
git add docs/demos/results/2026-09-22_optum_biologic_persistence_cert/
git commit -m "docs(evidence): Lane A causal export run — n=15,209, arms 11,009/4,200, rates match the disproof

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Rehearse migration 148 on prod's DB (runner-shaped, `BEGIN … ROLLBACK`) and dry-run the loader

**Files:**
- Create: `docs/demos/results/2026-09-22_optum_biologic_persistence_cert/migration_rehearsal.md`

- [ ] **Step 1: Rehearse the bytes the runner runs (memory: a `\i` rehearsal is not the runner)**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/real-data-causal
F=database/migrations/148_optum_biologic_persistence_causal.sql
{ echo "BEGIN;"; cat "$F"; printf "INSERT INTO public.schema_migrations(filename) VALUES ('%s') ON CONFLICT DO NOTHING;\n" "$(basename "$F")"; \
  echo "SELECT count(*) AS cols FROM information_schema.columns WHERE table_name='optum_biologic_persistence_causal';"; \
  echo "SELECT filename FROM public.schema_migrations WHERE filename LIKE '148%';"; echo "ROLLBACK;"; } \
  | docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1
```

Expected: `CREATE TABLE`, two `CREATE INDEX`, the `COMMENT`s, `INSERT 0 1`, `cols = 83` (the 81 exported fields, `is_synthetic` among them, plus `created_at` and `updated_at`; the contract test's `len(_export_record_keys()) + 2`), the ledger row, `ROLLBACK`.

- [ ] **Step 2: Negative control — nothing persisted**

```bash
docker exec -i supabase-db psql -U postgres -d postgres -tA -c "SELECT to_regclass('public.optum_biologic_persistence_causal'), (SELECT count(*) FROM public.schema_migrations WHERE filename LIKE '148%');"
```
Expected: `|0` (NULL regclass, zero ledger rows).

- [ ] **Step 3: Loader dry-run against the real parquet (table absent)**

```bash
$PY -m scripts.load_optum_causal_cohort --input /home/enunez/Projects/e2i_causal_analytics/data/rwd/mart/persistence_causal/e2i_causal_v1_biologic_persistence.parquet
```
Expected: `WOULD WRITE: n=15209 arms={'XOLAIR': 11009, 'DUPIXENT': 4200}`, per-outcome rates as in Task 7, `LIVE TABLE: unreachable (migration 148 not applied, or no client)`, `DRY RUN complete. No rows written.`; exit 0.

- [ ] **Step 4: The deploy gate's faithful pre-deploy experiment (memory: run it in the worktree before the PR)**

```bash
awk '/MemAvailable/ {print $2/1024 " MiB"}' /proc/meminfo   # >= 2048 or skip and say so
env -u E2I_DB_SIMULATE_PENDING -u E2I_LIVE_LLM E2I_DB_INTEGRATION=1 $PY -m pytest -n 0 -p no:cacheprovider -q -rs tests/unit/test_database/learning_loop/ 2>&1 | tail -15
```
Expected: the banner names `pending ['148_optum_biologic_persistence_causal.sql']` and the suite passes (~3.5 min). Record the banner line verbatim.

- [ ] **Step 5: Record and commit**

Write `migration_rehearsal.md` with the four command outputs verbatim (rehearsal, negative control, loader dry-run, gate banner + pass count), then:

```bash
git add docs/demos/results/2026-09-22_optum_biologic_persistence_cert/migration_rehearsal.md
git commit -m "docs(evidence): migration 148 runner-shaped rehearsal, negative control, loader dry-run, deploy-gate banner

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: In-process agent pre-flight on the exported frame (the cheapest disproof of "the run completes")

**Why:** the live cert cannot run until the PR is deployed. The single assumption the cert rests on is that the agent completes on n = 15,209 × ~80 resolved covariates inside the 900 s hard cap. This task measures it BEFORE the PR through the real coercion + one-hot + `_run_agent_analysis_task` code, on the exported parquet (no DB needed). Suggestive, not decisive (host CPU vs the container); the live cert in Task 12 is decisive.

**Files:**
- Create: `docs/demos/results/2026-09-22_optum_biologic_persistence_cert/preflight_agent.py`
- Create: `docs/demos/results/2026-09-22_optum_biologic_persistence_cert/preflight.md`

- [ ] **Step 1: Write the pre-flight script**

```python
#!/usr/bin/env python3
"""Lane A pre-merge pre-flight: run the causal_impact agent IN-PROCESS on the
exported causal frame through the production coercion + one-hot helpers and the
route's own background task. Measures wall time and records the estimate so the
live cert (post-deploy) is not the first time the run shape is exercised.

Run from the lane worktree:
  cd .worktrees/real-data-causal && $PY docs/demos/results/2026-09-22_optum_biologic_persistence_cert/preflight_agent.py <outcome> [estimator]
"""
from __future__ import annotations

import asyncio
import json
import resource
import sys
import time
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path.cwd()))
import src  # noqa: E402

assert ".worktrees/real-data-causal" in src.__file__, src.__file__

from src.api.routes.causal import agent as causal_routes  # noqa: E402
from src.api.routes.causal.datasets import (  # noqa: E402
    _CAUSAL_CATEGORICAL_COLUMNS,
    _CAUSAL_DATASET_SPECS,
    _CAUSAL_NUMERIC_COLUMNS,
)
from src.api.routes.causal.loaders import _coerce_estimation_row, _one_hot_categoricals  # noqa: E402
from src.api.schemas.causal import AgentCausalAnalysisRequest  # noqa: E402

DATASET = "optum_biologic_persistence"
TREATMENT = "treatment_dupixent"
PARQUET = Path("/home/enunez/Projects/e2i_causal_analytics/data/rwd/mart/persistence_causal/e2i_causal_v1_biologic_persistence.parquet")
OUT = Path(__file__).resolve().parent


class _MemStore:
    def __init__(self) -> None:
        self.d: dict = {}

    async def get(self, key):
        return self.d.get(key)

    async def set(self, key, value):
        self.d[key] = value


def build_frame(outcome: str):
    spec = _CAUSAL_DATASET_SPECS[DATASET]
    covariates = list(spec["covariate"])
    select_cols = [TREATMENT, outcome, *covariates]
    raw = pd.read_parquet(PARQUET, columns=select_cols)
    raw = raw.astype(object).where(raw.notna(), None)
    records = []
    for row in raw.to_dict(orient="records"):
        rec = _coerce_estimation_row(
            row,
            select_cols=select_cols,
            treatment_var=TREATMENT,
            outcome_var=outcome,
            numeric_cols=_CAUSAL_NUMERIC_COLUMNS[DATASET],
            categorical_cols=frozenset(_CAUSAL_CATEGORICAL_COLUMNS[DATASET]),
        )
        if rec is not None:
            records.append(rec)
    frame = pd.DataFrame(records)
    cats = [c for c in select_cols if c in _CAUSAL_CATEGORICAL_COLUMNS[DATASET]]
    frame, dummies = _one_hot_categoricals(frame, cats)
    resolved = [c for c in select_cols if c not in (TREATMENT, outcome) and c not in cats] + dummies
    return frame, resolved


async def run(outcome: str, estimator: str | None) -> dict:
    store = _MemStore()
    causal_routes._agent_analysis_store = store
    frame, covariates = build_frame(outcome)
    req = AgentCausalAnalysisRequest(
        treatment_var=TREATMENT, outcome_var=outcome, dataset=DATASET, limit=20000, estimator=estimator
    )
    req = req.model_copy(update={"auto_discover": False})
    aid = str(uuid.uuid4())
    t0 = time.monotonic()
    await causal_routes._run_agent_analysis_task(aid, req, frame, covariates, "database")
    wall = time.monotonic() - t0
    result = store.d[aid]
    payload = result.model_dump(mode="json")
    payload["_preflight"] = {
        "wall_s": round(wall, 1),
        "n_rows": int(frame.shape[0]),
        "n_covariates_resolved": len(covariates),
        "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "estimator_forced": estimator,
    }
    return payload


if __name__ == "__main__":
    outcome = sys.argv[1]
    estimator = sys.argv[2] if len(sys.argv) > 2 else None
    payload = asyncio.run(run(outcome, estimator))
    path = OUT / f"preflight_{outcome}{'_' + estimator if estimator else ''}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    p = payload["_preflight"]
    print(
        f"{outcome}: status={payload['status']} wall={p['wall_s']}s rss={p['max_rss_mb']}MB "
        f"ate={payload.get('ate')} ci=[{payload.get('ate_ci_lower')}, {payload.get('ate_ci_upper')}] "
        f"p={payload.get('p_value')} estimator={payload.get('selected_estimator')} "
        f"dag_source={payload.get('dag_source')} refutation_passed={payload['refutation'].get('passed')}"
    )
```

(Field names from `AgentCausalAnalysisResponse` in `src/api/schemas/causal.py:712-760`: `ate`, `ate_ci_lower`, `ate_ci_upper`, `p_value`, `selected_estimator`, `dag_source`, `refutation.passed`.)

- [ ] **Step 2: Run the primary outcome with Auto estimator, watching memory**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/real-data-causal
awk '/MemAvailable/ {print $2/1024 " MiB"}' /proc/meminfo
docker inspect -f '{{.State.StartedAt}}' e2i_api        # record — the container state around the evidence
timeout 1000 $PY docs/demos/results/2026-09-22_optum_biologic_persistence_cert/preflight_agent.py persistent_at_180d_g28 2>&1 | tee -a preflight_run.log | tail -5
```
Expected: `status=completed` (or `needs_review`) within the 900 s budget. If wall > 900 s or the task reports a timeout: record it, then re-run with `LinearDML` as the forced estimator and note that the cert will force it too (`estimator="LinearDML"` in Task 12's requests).

- [ ] **Step 3: Run the two secondary outcomes with the same shape**

```bash
for o in discontinued_180d biologic_switch_180d_flag; do timeout 1000 $PY docs/demos/results/2026-09-22_optum_biologic_persistence_cert/preflight_agent.py $o [LinearDML] 2>&1 | tail -2; done
```

- [ ] **Step 4: Write `preflight.md`**

Table: outcome | n_rows | resolved covariates | estimator | wall s | max RSS | status | ATE | 95 % CI | refutation summary | gate decision | dag_source. Then: "Read: suggestive (host process), not the cert; the container run in Task 12 is decisive." Note the shipped `persistent_at_180d` was NOT pre-flighted as an effect (artefact); it is run in Task 12 only alongside the sweep.

- [ ] **Step 5: Commit (JSON payloads included; no `.log` — `.gitignore` drops `*.log`, so paste the tail into `preflight.md`)**

```bash
git add docs/demos/results/2026-09-22_optum_biologic_persistence_cert/
git commit -m "docs(evidence): in-process agent pre-flight on the real causal frame (timing + estimates, three outcomes)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

**Amendment (2026-09-22, after the first two pre-flights hit the 900 s cap — codex r1 finding 2):**
Step 2's contingency ("force LinearDML") did not help: Auto AND forced LinearDML both died at the
refutation node's DoWhy reconstruction (900.5 s / 900.4 s). The run shape was resolved by attribution,
not by subsampling (the full cohort is the spec) and not by widening the cap:

| piece (n = 15,209 × 77) | measured | mechanism (read in the dependency's source) | fix |
|---|---|---|---|
| `_build_dowhy_estimate` reconstruction 478 s | `identify_effect` 467 s, econml fit ~11 s (probe log timestamps) | dowhy 0.14 default identifier accepts the full common-cause set on its first candidate, then re-runs a minimal-set search from the smallest subset upward and burns its 100,000-iteration cap when the minimal set IS the full set (k=12: 0.65 s; k≥17: the cap; k=77: 467 s) | `identify_effect(..., optimize_backdoor=True)` — same adjustment set at k=8/12/17/77, byte-identical LinearDML ATE at k=8/12, 0.01 s. Red-first: `tests/unit/test_agents/test_causal_impact/test_refutation_identify_budget.py` |
| `LinearDMLWrapper.fit` 605 s | `LogisticRegressionCV(cv=3, max_iter=500)` on the UNSCALED design 441.4 s vs 5.8 s standardised (`timing_probe2_*.json`); econml fit 12.7 s; `_honest_ate_ci` 0.4 s | lbfgs grinding to its iteration cap on mixed-unit columns (a non-converged fit that also moves with the units) | `nuisance_config.propensity_model()` = `StandardScaler → LogisticRegressionCV` at the 9 wrapper sites. Red-first: `tests/unit/test_causal_engine/test_energy_score/test_propensity_scale_invariance.py`. Served-output check on the live synthetic pairs (`tournament_*.json/.txt`): 0 winner flips, ATEs identical, max |Δ energy| 0.02 |

Result: the same pre-flight (Auto, discovery off) completes in **266.9 s** (`preflight.md`). Two things the
pre-flight taught that the plan had wrong: (1) "runs WITHOUT `.env` so it writes nothing to prod" was FALSE —
`src/ml/data_loader.py` lazily `load_dotenv()`s and walks up to the main checkout's `.env`, so the first run
that reached the refutation node persisted 6 `causal_validations` + 1 `validation_outcomes` rows and one
MLflow run to prod (reported to the owner; the script now blanks the Supabase env before importing `src`);
(2) the script must run from the worktree ROOT — from the evidence dir it imports `src` from main (its own
assert caught it). Follow-ups NOT in this lane: the same default identifier in
`src/causal_engine/pipeline/executors/dowhy.py:258`; `OrthoForestWrapper` cannot fit any frame
(`max_depth=None` TypeError); `_load_agent_estimation_frame`'s `.limit()` has no ORDER BY (nba_triggers
serves a different 5,000-row subset per run).

**Amendment 2 (codex r2 → REVISE, resolved the same day):** (1) `optimize_backdoor` gated on a
non-empty adjustment set (the path search returns no set for `[]`; `value=None` silently). (2) The
real design is rank 61 of 77 and the round-1 run served a CI econml had declared invalid → the loader
prunes exactly collinear columns once at load (`_prune_exactly_collinear`; ATE 0.03353→0.03353, SE
0.00855→0.00858, warning gone) and LinearDML/DRLearner refuse a warned SERVED fit (a subsampled
tournament fit is unserved and keeps its point estimate — two outcomes' 5,000-row subsamples are rank
61/62 without any constant column). (3) The pre-flight had re-implemented the loader's path and
silently skipped the prune (k stayed 77; runs fell to CausalForest at 600 s) → the loader's post-fetch
path is now one function, `_resolve_agent_estimation_frame`, and the pre-flight calls it. Final runs:
257.6 s / 221.3 s / 185.3 s, LinearDML, k=61, refutation 4/5 proceed on all three (`preflight.md`).
(4) The 6 + 1 incident rows were deleted after the owner's explicit approval; the MLflow run was left.

**Amendment 3 (codex r3 → REVISE: 1 HIGH + 3 MED, resolved in `fabbec0c9`):** (1) the prune criterion
compared the residual with the RAW norm and dropped a genuinely varying large-offset column (`1e10 +
arange`, residual/raw 2.9e-9); it now compares with the CENTERED norm (translation- and unit-invariant),
treats a column as constant only at machine rounding (1e-12), skips only at `n < k+1`, and documents the
resolved order (numerics, then dummies) without reordering the design (forest fits subsample by column
index). The real frame drops the SAME 16 columns, so the Amendment 2 numbers stand. (2) Auto refits the
next ranked successful tournament candidate when the winner's SERVED full-frame refit is refused (a forced
estimator still fails closed). (3) The identifier test asserts the median-split treatment and, on a
LinearDML rebuild, the effect modifiers. (4) `load_frame` refuses an empty / one-arm export before any
write; the real-DB gate asserts both arms. Pre-existing and NOT chased here: the wide
`tests/unit/test_agents/test_causal_impact/` run aborts at `test_refutation.py::test_run_all_refutation_tests`
(pytest-timeout inside an econml GRF fit) — reproduced at the pre-session commit `4ba8372b8`; CI's sharded
`test_agents` lane is the arbiter.

**Amendment 4 (codex r4 → REVISE: 1 HIGH + 1 MED, both fixed):** (1) the prune's constant test and fixed
1e-8 tolerance were still not translation/scale-invariant (`1e14 + arange` dropped as constant; 1e-200
scale underflowed; 5e-9 relative noise dropped although econml's `lstsq(rcond=None)` rank check —
max(n,k)·eps ≈ 3.4e-12 — would keep it). Now: constant = exact equality; reference value subtracted and a
power-of-two rescale before centering; tolerance = max(n,k)·eps. Real frame: same 16 columns, same order;
dropped ratio max 2.6e-15, kept min 0.047 (`prune_tolerance_probe.json`). (2) The next-candidate fallback
now preserves the refused winner's tournament score in `all_results`/`energy_scores`, keeps the tournament
gap, and narrates the fallback in `selection_reason`; the review gate is judged on the served score.
The #1392 test that pinned fail-closed-with-a-candidate-left (broken by the r3 fallback, unnoticed because
the energy-score directory had not been re-run after `fabbec0c9`) now asserts the new contract plus a
fail-closed sibling when every refit fails.

**Amendment 5 (codex r5 → REVISE: 3 HIGH + 1 MED):** the "matches econml's lstsq" claim was withdrawn
(econml checks its own unscaled final-stage matrix with a global tolerance; a design it still rejects is
refused fail-closed, never served — follow-up: standardise the final-stage inputs engine-wide); the prune
scales by a power of two before subtracting the reference (float-max safe); the fallback narration records
each refusal alone and one closing served/none sentence; `tournament_energy_score` + `served_refit` are
explicit fields on every serialisation surface including the API candidate (api.ts regenerated).

**Amendment 6 (codex r6 → REVISE: 4 HIGH + 1 MED):** the prune's guarantee is "numerically collinear at
machine precision" (a one-ULP variant IS dropped; test pins it); every refusal is recorded including the
last one and the subsample sentence never claims a reported ATE/CI on failure; the hand-written frontend
`EstimatorCandidate` type and the comparison panel carry `tournament_energy_score` / `served_refit` and
rank by the tournament score; the MLflow DB logger applies the finite filter to `energy_details`.

**Amendment 7 (codex r7 → REVISE: 6 HIGH — 3 fixed, 3 pre-existing filed):** prune renamed
`_prune_numerically_collinear`; a raising served refit is a recorded refusal and the fallback continues;
"Tournament winner" only for the first refusal, "no refit attempted" wording for zero tournament successes.
Filed (pre-existing, outside the lane): duplicate estimator-type identity (since #1392), non-finite energy
scores rankable, `energy_score_gap` 0.0 with < 2 scores. §3A's two reviewer caveats are Task 12 data.
Codex rounds stop here: r7's findings were pre-existing contracts or wording, the signal that the loop
now generates its own work (memory 2026-09-18); residuals are reported to the owner in the PR.

**Amendment 8 (codex r8, final → REVISE: 2 HIGH + 2 LOW):** the non-finite rebuttal was wrong — the
fallback opened a path that could serve a NaN-scored success unreviewed; fixed (finite-score eligibility,
fail closed otherwise, teeth proven by planting). The prune probe JSON was stale against its script; the
script now compares the shipped prune with an explicit r3 baseline and the JSON is regenerated with the
commit recorded. LOWs (tournament-fit raise; wording) are pre-existing/wording.

---

### Task 10: Codex review rounds to ACCEPT

- [ ] **Step 1: Write the brief** to the scratchpad as `lane_a_codex_r1.md`. It MUST contain this paragraph verbatim:

> If a recommendation solves a labeling problem instead of a functional problem, flag it as HIGH finding. If a recommendation preserves code without investigating intent (PR history, linked issues, user-requested functionality), flag it as HIGH finding. If a recommendation deletes code without verifying intent, flag it as HIGH finding. Audit the question being asked, not just the answer given.

and: the spec path and §3A; the diff range `git diff main...HEAD -- scripts/ src/ database/ tests/`; the evidence dir; the sequencing decision (cert post-deploy); the explicit instruction **do not run mypy, the whole-tree pytest, or anything heavier than targeted `pytest -n 0` files** (memory-pressured box); the questions to audit: (a) can a prediction cohort ever receive `extra_cols`? (b) is the `two_arm_contrast` step the right response to an unknown brand vs. failing? (c) is `model_fields_set` the honest seam for "caller did not set it"? (d) does the SQL column set equal the export (the contract test's premise)? (e) any path where `is_synthetic` reaches the design matrix? (f) the loader's verification — can it print VERIFIED on a partial write?

- [ ] **Step 2: Run codex read-only from the worktree**

```bash
codex exec --sandbox read-only -C /home/enunez/Projects/e2i_causal_analytics/.worktrees/real-data-causal "$(cat <scratchpad>/lane_a_codex_r1.md)" < /dev/null > <scratchpad>/lane_a_codex_r1.out 2>&1
grep -n "workdir:" <scratchpad>/lane_a_codex_r1.out | head -1     # must be THIS worktree
```

- [ ] **Step 3: Triage every finding with reasoning (CLAUDE.md reason-before-rules)** — for each: what the code is trying to do, why it has this shape, harm now, what the owner asked; then fix (red-first) or rebut with evidence. Commit fixes as `fix(lane-a): <finding> (codex r<N>)`. Re-brief as r2, r3 … until the verdict is ACCEPT with zero HIGH/MED. Report findings by LOCATION (memory: the distribution says when to stop).

---

### Task 11: Push, PR, and ask the owner for the merge

- [ ] **Step 1: Final local gates (targeted, as CI runs them)**

```bash
$PY -m ruff check --no-cache src/ tests/ && $PY -m ruff format --check src/ tests/   # CI lints src/ and tests/ only (backend-tests.yml:120-123); lint changed scripts/ files individually
$PY -m pytest -n 0 -p no:cacheprovider -q tests/unit/test_scripts/test_convert_optum_mart_causal.py tests/unit/test_scripts/test_convert_optum_mart.py tests/unit/test_scripts/test_convert_optum_mart_multicohort.py tests/unit/test_scripts/test_optum_causal_cohort_contract.py tests/unit/test_scripts/test_load_optum_causal_cohort.py tests/unit/test_api/test_causal_optum_dataset_registry.py tests/unit/test_api/test_causal_triggers_dataset.py tests/unit/test_api/test_causal_geo_encoding.py tests/unit/test_api/test_causal_covariate_roles.py tests/unit/test_api/test_causal_agent_analyze_negative_control_2007.py tests/unit/test_repositories/test_has_provenance_family_894.py tests/unit/test_docker/
grep -rn "if False\|SIMULATED" $(git diff --name-only main...HEAD) || echo "no debris"
git log --oneline main..HEAD
```

- [ ] **Step 2: Push and open the PR (no `close #N` keywords; "Part of" wording)**

```bash
git push -u origin claude/real-data-causal-estimation
gh pr create --base main --title "Lane A: real-data causal pipeline — Optum Dupixent vs Xolair persistence cohort, migration 148, registry" --body-file <scratchpad>/lane_a_pr_body.md
```

PR body: goal (one paragraph), what ships (the six bullets of §3A), what does NOT (frontend selector, negative control, discovery), the sequencing (deploy applies 148 → owner-GO load → live cert follow-up), the evidence dir with the three pre-merge documents, the codex round count and verdict, the gates run. End with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

- [ ] **Step 3: Watch CI via the actions runs API with the full sha (PAT 403s on `gh pr checks`)**

```bash
SHA=$(git rev-parse HEAD); gh api "repos/enunezvn/e2i_causal_analytics/actions/runs?head_sha=$SHA" --jq '.workflow_runs[] | [.name,.status,.conclusion] | @tsv'
```
Read the `mypy-report` artifact and diff its error count against main's (the gate is a ceiling). Fix reds red-first; never merge.

- [ ] **Step 4: Report to the owner** — PR link, CI state, the pre-flight timing table, and the exact three post-merge steps (deploy applies 148; `--execute` load; live cert). Wait for the merge decision. Use `--merge`, never squash, when told to merge.

---

### Task 12: Post-merge — the deploy applies 148, the owner-GO load, the live API cert (follow-up PR)

Run ONLY after the owner has merged and the deploy run is green.

- [ ] **Step 1: Verify the deploy applied migration 148**

```bash
gh api "repos/enunezvn/e2i_causal_analytics/actions/runs?branch=main&event=push&per_page=3" --jq '.workflow_runs[] | [.name,.head_sha,.conclusion] | @tsv'
docker exec -i supabase-db psql -U postgres -d postgres -tA -c "SELECT filename, applied_at FROM public.schema_migrations WHERE filename LIKE '148%'; SELECT to_regclass('public.optum_biologic_persistence_causal');"
docker inspect -f '{{.State.StartedAt}}' e2i_api
```
Expected: one ledger row for `148_optum_biologic_persistence_causal.sql`; the regclass resolves; a fresh StartedAt.

- [ ] **Step 2: Rehearse the load inside a transaction (spec §5), then execute (owner GO, spec §7)**

The PostgREST upsert cannot be wrapped in a SQL transaction from the client, so the rehearsal is the loader's own dry-run against the now-existing table (`LIVE BEFORE: n=0`), then the write, whose verification re-reads the table:

```bash
cd /home/enunez/Projects/e2i_causal_analytics   # main checkout, on main, post-deploy
git branch --show-current && git status -sb | head -3      # main, clean
$PY -m scripts.load_optum_causal_cohort --input data/rwd/mart/persistence_causal/e2i_causal_v1_biologic_persistence.parquet
$PY -m scripts.load_optum_causal_cohort --input data/rwd/mart/persistence_causal/e2i_causal_v1_biologic_persistence.parquet --execute 2>&1 | tee <scratchpad>/lane_a_load.log
```
Expected: `EXECUTE: upserted 15209 rows`, `LIVE AFTER: n=15209 arms={'XOLAIR': 11009, 'DUPIXENT': 4200}`, `VERIFIED`, exit 0. Re-run `--execute` once more: same counts (idempotence measured, not assumed).

- [ ] **Step 3: The real-DB probe**

```bash
E2I_DB_INTEGRATION=1 $PY -m pytest -n 0 -p no:cacheprovider tests/integration/test_optum_causal_cohort_realdb.py -q -rs
```
Expected: `1 passed`.

- [ ] **Step 4: Write and run the live cert script**

Create `docs/demos/results/2026-09-22_optum_biologic_persistence_cert/run_cert.py` (mint token + call helpers copied from `docs/demos/results/2026-09-09_expert_review_loop/run_discovery.py` lines 13–27; `API = os.environ.get("E2I_API_BASE", "https://eznomics.site/api")`):

```python
OUTCOMES = ["persistent_at_180d_g28", "discontinued_180d", "biologic_switch_180d_flag", "persistent_at_180d"]

def run_one(token, outcome, estimator=None):
    body = {"treatment_var": "treatment_dupixent", "outcome_var": outcome,
            "dataset": "optum_biologic_persistence", "limit": 20000}
    if estimator:
        body["estimator"] = estimator
    # auto_discover deliberately OMITTED: the per-dataset default (off) is under test
    pending = call(token, "POST", "/causal/agent-analyze", body=body)
    aid = pending["analysis_id"]
    t0 = time.time()
    while True:
        time.sleep(15)
        job = call(token, "GET", f"/causal/agent-analyze/{aid}")
        if job["status"] in ("completed", "needs_review", "failed") or time.time() - t0 > 1000:
            break
    job["_cert"] = {"submitted_warnings": pending["warnings"], "wall_s": round(time.time() - t0, 1),
                    "container_started_at": subprocess.run(["docker", "inspect", "-f", "{{.State.StartedAt}}", "e2i_api"], capture_output=True, text=True).stdout.strip()}
    (OUT / f"raw_{outcome}.json").write_text(json.dumps(job, indent=2))
    return job
```

Run the four outcomes sequentially (one heavy-compute slot per worker; never in parallel), with the estimator forced only if Task 9 showed Auto exceeds the cap. Also write `caveats.py`: reads the raw drop and prints `treatment_response` counts by arm (expected 4,109 "controlled" on Xolair vs 30 on Dupixent — data, not used as an outcome) and the `max_consecutive_biologic_coverage_days` quartiles by arm (14 vs 28–45), saved as `caveats.json`.

- [ ] **Step 5: Write `cert.md`** — verdict word first, then the table (outcome | n_rows | estimator | ATE | 95 % CI | p | refutation tests passed/total | negative control = SKIPPED `no_negative_control_declared` | E-value / gate decision | dag_source | discovery_enabled=false | wall s), the pending-response discovery warning quoted verbatim, the two caveats as data (from `caveats.json`), the `persistent_at_180d` row explicitly labelled "days-supply artefact — read with the grace sweep, not as an effect", the container StartedAt, and the load + probe outputs. A null (CI containing 0) is a finding; report it as such.

- [ ] **Step 6: Follow-up PR**

```bash
git checkout -b claude/lane-a-live-cert main && git add docs/demos/results/2026-09-22_optum_biologic_persistence_cert/ && git commit -m "docs(evidence): Lane A live cert — optum_biologic_persistence loaded (15,209) and estimated via the API for four outcomes

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" && git push -u origin claude/lane-a-live-cert && gh pr create --base main --title "docs: Lane A live cert (post-deploy load + API estimates)" --body-file <scratchpad>/lane_a_cert_pr_body.md
```

Report to the owner: the cert verdict line, the PR link, and that Lane E is next (its own plan).

---

## Self-review (done while writing)

**Spec coverage (§3A):** step 1 export → Tasks 1–2 (+ the "causal frame carries the treatment, prediction frame does not" test); step 2 table + load → Tasks 3–4 (+ realdb probe), the GO'd production load → Task 12; step 3 registry → Task 5 (negative control deliberately absent, SKIPPED asserted); step 4 run shape → Task 6 (`auto_discover` default off, explicit opt-in honored; curated DAG is the graph_builder's existing manual path, untouched; randomized design false by omission of `randomized_treatment`; `limit` 20,000 covers the cohort — PostgREST cap disproved); step 5 evidence → Tasks 7–9 (pre-merge) and 12 (the API run + both caveats as data). §5 honesty: SKIPPED negative control (Task 5 test), `is_synthetic=false` on every row (Tasks 2, 4), discovery-on reports the runner's failure (unchanged PR #2203 path), prod writes rehearsed (Task 8) and owner-GO (Task 12). §6 gates: export shape, loader, registry consistency, real-DB arm split, API cert — all present.

**Placeholder scan:** no TBD/TODO; every code step shows the code; the only "check the field names" note (Task 9 Step 1) points at the exact schema location and the JSON dump carries all fields regardless.

**Type consistency:** `select_persistence_causal_cohort(df, *, window_days, min_claim_count)` matches the other selectors and `_SELECTOR_BY_COHORT`'s call `selector(df, window_days=…, min_claim_count=…)`; `build_journey_records(…, extra_cols=…)` is called with `CAUSAL_EXTRA_COLS` in Tasks 2, 3; `arm_split` / `fetch_live_split` / `verify` share the `{"n", "arms", "outcome_positives"}` shape across Tasks 4 and 12; `_default_auto_discover(Optional[str]) -> bool` is used identically in Tasks 5, 6, 9; the table name `optum_biologic_persistence_causal` and dataset key `optum_biologic_persistence` are the same strings in Tasks 3, 4, 5, 12.
