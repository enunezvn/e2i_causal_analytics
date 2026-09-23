"""One idempotent loader engine for the causal cohort tables, parameterised by
a :class:`CohortSpec` (Lane C-load, 2026-09-23).

Lane A shipped ``scripts/load_optum_causal_cohort.py`` for
``optum_biologic_persistence_causal`` (migration 148, REAL claims rows only).
Lane C's ``csu_escalation_causal`` (migration 149) carries the SAME 81-column
contract with the treatment renamed (``treatment_remibrutinib``, 1 = RHAPSIDO,
0 = the XOLAIR / DUPIXENT competitor pool) and is backed by SYNTHETIC rows
(every one ``is_synthetic = true``) until the post-launch refresh. Rather than a
second copy of the loader, the engine lives here and each table gets a thin
entrypoint (``scripts/load_optum_causal_cohort.py``,
``scripts/load_csu_escalation_cohort.py``) that binds its spec, so the SAME
discipline applies to both: fail-loud validation of the exact column contract
BEFORE any write, the per-table provenance rule, the arm/treatment agreement,
dry-run by default, and after ``--execute`` a LIVE re-read compared with the
parquet at the aggregate level (arm split, treatment counts, per-outcome
positives) AND patient-by-patient across every exported field.

The provenance rule is the one thing the two tables disagree on, and it is the
one that matters most: the Optum table refuses any ``is_synthetic = true`` row
(real claims data only) and the CSU table refuses any row that is NOT
``is_synthetic = true`` (a real-looking row in the synthetic backing would be
served by real mode as real). A NULL is refused by both.

Synthetic-backed specs carry an extra precondition on the WRITE path (PR #2228
owner decision 2): the dataset must be covered by the API's dataset-level
provenance guard (``datasets._CAUSAL_SYNTHETIC_BACKED``) on the tree the loader
runs from, and the planted-truth seam must be closed -- otherwise the deployed
showcase flag (``E2I_INCLUDE_SYNTHETIC=true``) would serve the planted rows as
if real the moment they land. The guard is checked before the first upsert and
a failure is exit 1 with nothing written.
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.convert_optum_mart import (  # noqa: E402
    CAUSAL_EXTRA_COLS,
    CSU_ESCALATION_COMPETITOR_ARMS,
    CSU_ESCALATION_TREATED_ARM,
    CSU_ESCALATION_TREATMENT_COL,
    TARGET_PERSISTENT_G28,
)
from scripts.convert_optum_mart import TREATMENT_COL as _OPTUM_TREATMENT_COL  # noqa: E402
from src.data.manifests import MART_SAFE_FEATURES  # noqa: E402

logger = logging.getLogger(__name__)

BATCH_SIZE = 500
ROW_PAGE_SIZE = 1000
PROVENANCE_COLUMN = "is_synthetic"
OUTCOME_COLUMNS = (
    TARGET_PERSISTENT_G28,
    "discontinued_180d",
    "biologic_switch_180d_flag",
    "persistent_at_180d",
)
# The journey-metadata keys the converter emits for every cohort (not part of the
# owner-approved feature allow-list, not an outcome, not the treatment).
JOURNEY_METADATA_COLUMNS = (
    "patient_journey_id",
    "patient_id",
    "patient_hash",
    "index_date",
    "journey_start_date",
    "journey_status",
    "discontinuation_flag",
)
_LIVE_ONLY_COLUMNS = ("created_at", "updated_at")


@dataclass(frozen=True)
class CohortSpec:
    """Everything that differs between the causal cohort tables."""

    dataset: str  # the registry name (src/api/routes/causal/datasets.py)
    table: str
    migration: int
    treatment: str
    treated_arms: tuple  # brand labels coded treatment=1
    control_arms: tuple  # brand labels coded treatment=0
    synthetic: bool  # the ONLY is_synthetic value a row may carry
    provenance_reason: str  # printed when the rule refuses a row
    default_input: str
    # The datasets.py frozenset the dataset must be in before the WRITE path
    # runs (None: no guard precondition -- the Optum table holds real rows).
    provenance_guard: Optional[str] = None
    # The commit that put that guard on main; the DEPLOYED image must descend
    # from it (None: no deployed attestation).
    deployed_guard_commit: Optional[str] = None
    brand: str = "index_biologic_brand"
    on_conflict: str = "patient_id"

    @property
    def arms(self) -> tuple:
        """Every brand label of the contrast, control pool first (Lane A's
        print order for the Optum table: XOLAIR then DUPIXENT)."""
        return (*self.control_arms, *self.treated_arms)

    @property
    def required_columns(self) -> tuple:
        """The EXACT causal export contract (81 columns), derived from the
        converter's own constants rather than hand-listed, so no loader can
        drift from what ``scripts/convert_optum_mart.py`` emits: 7
        journey-metadata keys + the primary outcome + the 64 owner-approved
        pre-index baseline features (``MART_SAFE_FEATURES``) +
        data_quality_score + the 6 ``CAUSAL_EXTRA_COLS`` with THIS table's
        treatment column + is_synthetic + data_split."""
        extras = tuple(
            self.treatment if c == _OPTUM_TREATMENT_COL else c for c in CAUSAL_EXTRA_COLS
        )
        return (
            *JOURNEY_METADATA_COLUMNS,
            TARGET_PERSISTENT_G28,
            *MART_SAFE_FEATURES,
            "data_quality_score",
            *extras,
            PROVENANCE_COLUMN,
            "data_split",
        )


OPTUM_SPEC = CohortSpec(
    dataset="optum_biologic_persistence",
    table="optum_biologic_persistence_causal",
    migration=148,
    treatment=_OPTUM_TREATMENT_COL,
    treated_arms=("DUPIXENT",),
    control_arms=("XOLAIR",),
    synthetic=False,
    provenance_reason="the causal cohort is real claims data only",
    default_input="data/rwd/mart/persistence_causal/e2i_causal_v1_biologic_persistence.parquet",
)

CSU_SPEC = CohortSpec(
    dataset="csu_escalation_causal",
    table="csu_escalation_causal",
    migration=149,
    treatment=CSU_ESCALATION_TREATMENT_COL,
    treated_arms=(CSU_ESCALATION_TREATED_ARM,),
    control_arms=tuple(CSU_ESCALATION_COMPETITOR_ARMS),
    synthetic=True,
    provenance_reason=(
        "csu_escalation_causal is the SYNTHETIC backing (spec 2026-09-22 §3C.2) -- "
        "every row must be is_synthetic=true so real mode never serves it"
    ),
    default_input="data/rwd/synthetic_CSU/csu_escalation_causal/csu_escalation_causal_synthetic.parquet",
    provenance_guard="_CAUSAL_SYNTHETIC_BACKED",
    # Merge commit of PR #2228 (dataset-level guard + PLANTED_TRUTH_RUN seam);
    # live-verified on the deployed instance the same day
    # (docs/demos/results/2026-09-22_lane_c_live_verify/cert.md).
    deployed_guard_commit="c0860bbf42296160396a8355ea5d225fc8daa1ad",
)


# ---------------------------------------------------------------------------
# Validation (pure)
# ---------------------------------------------------------------------------


def _is_null(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, float) and math.isnan(value)


def provenance_problem(values: Any, *, spec: CohortSpec) -> Optional[str]:
    """Why these ``is_synthetic`` values break the table's rule, or None.

    Every value must be EXACTLY a bool (``bool`` / ``numpy.bool_``) equal to
    ``spec.synthetic``: a NULL, an int 0/1 or the STRING "false" is refused
    (codex r1 HIGH -- ``astype(bool)`` reads "false" as True and would let a
    real-looking row into the synthetic backing). Shared by :func:`load_frame`
    (the frame) and :func:`upsert` (the write boundary)."""
    values = list(values)
    nulls = sum(1 for v in values if _is_null(v))
    if nulls:
        return f"{nulls} row(s) carry a NULL is_synthetic; {spec.provenance_reason}"
    non_bool = sorted({repr(v) for v in values if not isinstance(v, (bool, np.bool_))})
    if non_bool:
        return f"is_synthetic carries non-boolean value(s) {non_bool[:5]}; {spec.provenance_reason}"
    off_rule = sum(1 for v in values if bool(v) is not spec.synthetic)
    if off_rule:
        return (
            f"{off_rule} row(s) carry is_synthetic={not spec.synthetic}; {spec.provenance_reason}"
        )
    return None


def load_frame(path: Path | str, *, spec: CohortSpec) -> pd.DataFrame:
    """Read the export and refuse anything that is not EXACTLY the causal cohort
    contract -- missing OR unexpected extra columns, so a frame lacking baseline
    confounders can never silently upsert NULLs into them -- or that breaks the
    table's provenance rule, patient_id uniqueness, the treatment/brand coding,
    the two-arm contrast or the 0/1 outcomes."""
    required = spec.required_columns
    df = pd.read_parquet(path)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"export is missing required column(s) {missing}")
    extra = [c for c in df.columns if c not in required]
    if extra:
        raise ValueError(f"export has unexpected column(s) not in the causal contract: {extra}")
    problem = provenance_problem(df[PROVENANCE_COLUMN].tolist(), spec=spec)
    if problem is not None:
        raise ValueError(problem)
    dups = df["patient_id"].duplicated()
    if dups.any():
        raise ValueError(f"patient_id is not unique: {int(dups.sum())} duplicate(s)")
    off_contrast = ~df[spec.brand].isin(spec.arms)
    if off_contrast.any():
        raise ValueError(
            f"{int(off_contrast.sum())} row(s) have an {spec.brand} outside {spec.arms}: "
            f"{sorted(df.loc[off_contrast, spec.brand].astype(str).unique())}"
        )
    expected = df[spec.brand].isin(spec.treated_arms).astype(int)
    if not df[spec.treatment].astype(int).eq(expected).all():
        raise ValueError(f"{spec.treatment} disagrees with {spec.brand} on some rows")
    # A causal contrast needs BOTH arms (the API refuses a constant treatment with
    # a 400): an empty or one-arm export must be refused BEFORE any write -- never
    # loaded and then reported VERIFIED against an equally empty/one-arm table.
    if df.empty:
        raise ValueError("export is empty: no rows to load, no contrast to estimate")
    n_treated = int(expected.sum())
    n_control = int(len(df) - n_treated)
    if n_treated == 0:
        raise ValueError(
            f"export has no treated-arm rows ({' / '.join(spec.treated_arms)}; "
            f"{spec.treatment}=1); a causal contrast needs both arms"
        )
    if n_control == 0:
        raise ValueError(
            f"export has no control-arm rows ({' / '.join(spec.control_arms)}; "
            f"{spec.treatment}=0); a causal contrast needs both arms"
        )
    for col in OUTCOME_COLUMNS:
        values = set(pd.unique(df[col].dropna()))
        if not values <= {0, 1}:
            raise ValueError(f"{col} is not 0/1: found {sorted(values)}")
        if df[col].isna().any():
            raise ValueError(
                f"{col} has NULLs; the export fills the switch flag and derives the rest"
            )
    return df


def arm_split(df: pd.DataFrame, *, spec: CohortSpec) -> Dict[str, Any]:
    """Counts by brand label, by the treatment column the causal run reads, and
    per-outcome positives by brand label -- the verification unit."""
    arms = {arm: int((df[spec.brand] == arm).sum()) for arm in spec.arms}
    positives = {
        col: {arm: int(df.loc[df[spec.brand] == arm, col].astype(int).sum()) for arm in spec.arms}
        for col in OUTCOME_COLUMNS
    }
    treatment_int = df[spec.treatment].astype(int)
    treatment = {str(v): int((treatment_int == v).sum()) for v in (0, 1)}
    return {"n": int(len(df)), "arms": arms, "outcome_positives": positives, "treatment": treatment}


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


def load_env() -> None:
    """Load ``.env`` for the CLIENT path only (codex r1 HIGH: never at import,
    so a unit-test process that imports the loader is not handed production
    credentials). The checkout's own .env first; a lane WORKTREE
    (.worktrees/<lane>/) carries none, so fall back to the nearest .env up the
    tree from THIS file (the main checkout's -- the same file the API container
    is configured from). Neither call overrides a variable already in the
    environment, so a caller that blanks SUPABASE_* keeps it blank."""
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env")
    load_dotenv(find_dotenv())


def get_client() -> Any:
    load_env()
    from src.memory.services.factories import get_supabase_client

    return get_supabase_client()


def upsert(
    client: Any,
    records: List[Dict[str, Any]],
    *,
    spec: CohortSpec,
    batch_size: int = BATCH_SIZE,
    deployed_commit: Optional[str] = None,
) -> int:
    """Batched idempotent upsert on ``patient_id``. Returns rows written.

    The WRITE BOUNDARY (codex r1 HIGH): the table's provenance rule and the
    guard precondition are enforced HERE, before the first batch, not only in
    ``main`` -- a caller that builds records by hand cannot bypass them."""
    problem = provenance_problem([r.get(PROVENANCE_COLUMN) for r in records], spec=spec)
    if problem is not None:
        raise ValueError(f"refusing to upsert into {spec.table}: {problem}")
    problem = guard_problem(spec, deployed_commit=deployed_commit)
    if problem is not None:
        raise RuntimeError(f"refusing to upsert into {spec.table}: GUARD -- {problem}")
    written = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        client.table(spec.table).upsert(batch, on_conflict=spec.on_conflict).execute()
        written += len(batch)
        logger.info("  upserted %d/%d rows", written, len(records))
    return written


def fetch_live_split(client: Any, *, spec: CohortSpec) -> Optional[Dict[str, Any]]:
    """The live table's arm split via exact counts. None when the table is unreachable
    (e.g. the migration not yet applied) -- the caller reports that, never a zero."""
    try:
        arms: Dict[str, int] = {}
        positives: Dict[str, Dict[str, int]] = {col: {} for col in OUTCOME_COLUMNS}
        for arm in spec.arms:
            arms[arm] = int(
                client.table(spec.table)
                .select("patient_id", count="exact")
                .eq(spec.brand, arm)
                .execute()
                .count
            )
            for col in OUTCOME_COLUMNS:
                positives[col][arm] = int(
                    client.table(spec.table)
                    .select("patient_id", count="exact")
                    .eq(spec.brand, arm)
                    .eq(col, 1)
                    .execute()
                    .count
                )
        treatment: Dict[str, int] = {}
        for v in (0, 1):
            treatment[str(v)] = int(
                client.table(spec.table)
                .select("patient_id", count="exact")
                .eq(spec.treatment, v)
                .execute()
                .count
            )
        total = int(client.table(spec.table).select("patient_id", count="exact").execute().count)
        return {"n": total, "arms": arms, "outcome_positives": positives, "treatment": treatment}
    except Exception as e:  # noqa: BLE001 — a missing relation / store hiccup is reported, not hidden
        logger.warning("Could not read live %s: %s", spec.table, e)
        return None


def fetch_live_provenance_counts(client: Any, *, spec: CohortSpec) -> Optional[Dict[str, int]]:
    """Exact live counts by ``is_synthetic`` value -- the number the dataset-level
    guard is judged on. None when unreachable."""
    try:
        out: Dict[str, int] = {}
        for value in (True, False):
            out[str(value).lower()] = int(
                client.table(spec.table)
                .select("patient_id", count="exact")
                .eq(PROVENANCE_COLUMN, value)
                .execute()
                .count
            )
        return out
    except Exception as e:  # noqa: BLE001 — reported, not hidden
        logger.warning("Could not read live %s provenance counts: %s", spec.table, e)
        return None


def verify(expected: Dict[str, Any], live: Dict[str, Any], *, spec: CohortSpec) -> List[str]:
    """Every disagreement between the parquet split and the live split, as text."""
    problems: List[str] = []
    if expected["n"] != live["n"]:
        problems.append(f"n: parquet {expected['n']} vs live {live['n']}")
    for v in ("0", "1"):
        e, l = expected["treatment"][v], live["treatment"][v]
        if e != l:
            problems.append(f"treatment={v}: parquet {e} vs live {l}")
    for arm in spec.arms:
        if expected["arms"][arm] != live["arms"][arm]:
            problems.append(
                f"arm {arm}: parquet {expected['arms'][arm]} vs live {live['arms'][arm]}"
            )
        for col in OUTCOME_COLUMNS:
            e, l = expected["outcome_positives"][col][arm], live["outcome_positives"][col][arm]
            if e != l:
                problems.append(f"{col} positives in {arm}: parquet {e} vs live {l}")
    return problems


def fetch_live_rows(client: Any, *, spec: CohortSpec) -> Optional[List[Dict[str, Any]]]:
    """Page the whole live table back, ``ROW_PAGE_SIZE`` rows at a time, until a
    short page. None when the table is unreachable -- never a partial or empty
    result mistaken for the truth (mirrors ``fetch_live_split``'s honesty)."""
    try:
        rows: List[Dict[str, Any]] = []
        start = 0
        while True:
            page = (
                client.table(spec.table)
                .select("*")
                .order(spec.on_conflict)  # a stable page order (codex r1 MED)
                .range(start, start + ROW_PAGE_SIZE - 1)
                .execute()
                .data
            )
            rows.extend(page)
            if len(page) < ROW_PAGE_SIZE:
                break
            start += ROW_PAGE_SIZE
        return rows
    except Exception as e:  # noqa: BLE001 — a missing relation / store hiccup is reported, not hidden
        logger.warning("Could not read live %s rows: %s", spec.table, e)
        return None


def _same(a: Any, b: Any) -> bool:
    """Value equality across the JSON round-trip: PostgREST returns NUMERIC/INTEGER
    columns as JSON numbers, DATE as 'YYYY-MM-DD' strings, and BOOLEAN as bools --
    none of which necessarily share a Python type with the exported record."""
    if a is None and b is None:
        return True
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


def verify_rows(
    expected_records: List[Dict[str, Any]], live_rows: List[Dict[str, Any]]
) -> List[str]:
    """Row-level diff keyed on ``patient_id``: missing/extra ids (counts + first 5)
    and, for every exported field of every row present in both, a value mismatch
    (first 10 reported, plus a total). Aggregate margins (``verify``) can match
    while patient_ids or covariates are wrong; this is the check that cannot be
    fooled that way. Live-only bookkeeping columns (created_at/updated_at) are
    never compared -- the export never emits them."""
    problems: List[str] = []
    expected_by_id = {r["patient_id"]: r for r in expected_records}
    live_by_id = {
        r["patient_id"]: {k: v for k, v in r.items() if k not in _LIVE_ONLY_COLUMNS}
        for r in live_rows
    }
    expected_ids, live_ids = set(expected_by_id), set(live_by_id)

    missing_ids = sorted(expected_ids - live_ids)
    if missing_ids:
        problems.append(
            f"{len(missing_ids)} patient_id(s) in the export missing from the live table "
            f"(first 5): {missing_ids[:5]}"
        )
    extra_ids = sorted(live_ids - expected_ids)
    if extra_ids:
        problems.append(
            f"{len(extra_ids)} patient_id(s) in the live table not present in the export "
            f"(first 5): {extra_ids[:5]}"
        )

    mismatches: List[str] = []
    for pid in sorted(expected_ids & live_ids):
        exp_row, live_row = expected_by_id[pid], live_by_id[pid]
        for field, exp_val in exp_row.items():
            if not _same(exp_val, live_row.get(field)):
                mismatches.append(
                    f"{pid}.{field}: parquet {exp_val!r} vs live {live_row.get(field)!r}"
                )
    if mismatches:
        problems.append(f"{len(mismatches)} field value mismatch(es) (first 10 shown):")
        problems.extend(mismatches[:10])
    return problems


# ---------------------------------------------------------------------------
# Guard precondition (synthetic-backed tables only)
# ---------------------------------------------------------------------------


_COMMIT_RE = re.compile(r"([0-9a-f]{40})$")


def deployed_image_commit(container: str = "e2i_api") -> Optional[str]:
    """The commit the DEPLOYED API container was built from: the 40-hex tag of
    its image (``ghcr.io/enunezvn/e2i-api:<sha>``). None when docker or the
    container cannot be read -- the caller refuses, never assumes."""
    try:
        out = subprocess.run(
            ["docker", "inspect", container, "--format", "{{.Config.Image}}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    m = _COMMIT_RE.search(out.stdout.strip()) if out.returncode == 0 else None
    return m.group(1) if m else None


def is_ancestor(ancestor: str, descendant: str) -> Optional[bool]:
    """``git merge-base --is-ancestor`` on this checkout; None when git cannot
    answer (unknown commit, shallow clone)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(_PROJECT_ROOT), "merge-base", "--is-ancestor", ancestor, descendant],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode == 0:
        return True
    if out.returncode == 1:
        return False
    return None


def deployed_guard_problem(
    spec: CohortSpec, deployed_commit: Optional[str] = None
) -> Optional[str]:
    """Why the DEPLOYED instance would serve the rows as real, or None.

    codex r1 HIGH: the tree check says the loader's code carries the guard;
    the harm is on the instance that serves traffic. The deployed image's
    commit (read from the ``e2i_api`` container, or given as
    ``deployed_commit`` on a host without docker) must DESCEND from the commit
    that put the guard on main (``spec.deployed_guard_commit``); anything
    unreadable is refused."""
    if spec.deployed_guard_commit is None:
        return None
    commit = deployed_commit or deployed_image_commit()
    if not commit:
        return (
            "cannot read the deployed e2i_api image commit (docker unavailable?); "
            "pass --deployed-commit <sha> to attest it"
        )
    verdict = is_ancestor(spec.deployed_guard_commit, commit)
    if verdict is None:
        return (
            f"git cannot tell whether deployed {commit[:9]} descends from the guard commit "
            f"{spec.deployed_guard_commit[:9]} (fetch main first)"
        )
    if not verdict:
        return (
            f"deployed {commit[:9]} does NOT descend from the guard commit "
            f"{spec.deployed_guard_commit[:9]}: the deployed instance would serve the "
            "synthetic rows as real"
        )
    return None


def guard_problem(spec: CohortSpec, deployed_commit: Optional[str] = None) -> Optional[str]:
    """Why the load must NOT run, or None.

    For a synthetic-backed spec: (1) THIS tree's API dataset-level provenance
    guard must list the dataset (so every reader applies ``is_synthetic =
    false`` in real mode regardless of the deployment flag) and the
    planted-truth seam must be closed -- read from the same ``datasets``
    module the API serves, not a hard-coded assumption; (2) the DEPLOYED
    instance must carry that guard (:func:`deployed_guard_problem`)."""
    if spec.provenance_guard is None:
        return None
    from src.api.routes.causal import datasets as datasets_mod

    guarded = getattr(datasets_mod, spec.provenance_guard, frozenset())
    if spec.dataset not in guarded:
        return (
            f"dataset {spec.dataset!r} is not in datasets.{spec.provenance_guard} "
            f"({sorted(guarded)}): real mode would serve the synthetic rows"
        )
    if bool(getattr(datasets_mod, "PLANTED_TRUTH_RUN", False)):
        return "datasets.PLANTED_TRUTH_RUN is True in this process: the seam must be closed"
    return deployed_guard_problem(spec, deployed_commit=deployed_commit)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def print_split(label: str, split: Dict[str, Any], *, spec: CohortSpec) -> None:
    print(f"{label}: n={split['n']} arms={split['arms']} treatment={split['treatment']}")
    for col in OUTCOME_COLUMNS:
        pos = split["outcome_positives"][col]
        rates = {
            arm: (round(pos[arm] / split["arms"][arm], 4) if split["arms"][arm] else None)
            for arm in spec.arms
        }
        print(f"  {col}: positives={pos} rate={rates}")


def build_parser(spec: CohortSpec, doc: Optional[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", default=spec.default_input, help="causal export parquet")
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--execute",
        action="store_true",
        help="WRITE PATH: upsert the rows into the live table. Omit (default) for a dry run.",
    )
    run_mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicit dry run (the default when neither flag is given). "
        "Mutually exclusive with --execute.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--deployed-commit",
        default=None,
        help="the commit the deployed e2i_api image was built from, for a host where docker "
        "cannot be read (verified by ancestry against the guard commit either way)",
    )
    return parser


def run(
    spec: CohortSpec,
    argv: Optional[List[str]] = None,
    *,
    client_factory: Callable[[], Any] = get_client,
    fetch_live_rows_fn: Optional[Callable[[Any], Optional[List[Dict[str, Any]]]]] = None,
    doc: Optional[str] = None,
) -> int:
    """The dry-run -> execute -> LIVE BEFORE/AFTER -> row-verification discipline
    for ``spec``. ``client_factory`` / ``fetch_live_rows_fn`` are the seams the
    per-table entrypoints bind to their own module globals (so a test's
    monkeypatch on the entrypoint module is honoured)."""
    args = build_parser(spec, doc).parse_args(argv)
    dry_run = not args.execute
    load_env()  # the CLI path: .env before the API modules the guard check imports
    _fetch_rows = fetch_live_rows_fn or (lambda c: fetch_live_rows(c, spec=spec))

    print("=" * 70)
    print(f"{spec.table} loader ({'DRY RUN' if dry_run else 'EXECUTE'})  input={args.input}")
    print("=" * 70)
    df = load_frame(args.input, spec=spec)
    expected = arm_split(df, spec=spec)
    print_split("WOULD WRITE" if dry_run else "WRITING", expected, spec=spec)
    print(f"PROVENANCE: every row is_synthetic={str(spec.synthetic).lower()} (verified)")

    problem = guard_problem(spec, deployed_commit=args.deployed_commit)
    if spec.provenance_guard is not None:
        deployed = args.deployed_commit or deployed_image_commit()
        print(
            f"GUARD: datasets.{spec.provenance_guard} covers {spec.dataset!r}, the "
            f"planted-truth seam is closed, and deployed {str(deployed)[:9]} descends from "
            f"the guard commit {str(spec.deployed_guard_commit)[:9]}"
            if problem is None
            else f"GUARD: REFUSED -- {problem}"
        )

    client = None
    try:
        client = client_factory()
    except Exception as e:  # noqa: BLE001 — no client => dry-run still useful
        logger.warning("No Supabase client (%s).", e)

    live_before = fetch_live_split(client, spec=spec) if client is not None else None
    if live_before is None:
        print(f"LIVE TABLE: unreachable (migration {spec.migration} not applied, or no client)")
    else:
        print_split("LIVE BEFORE", live_before, spec=spec)
        provenance = fetch_live_provenance_counts(client, spec=spec)
        if provenance is not None:
            print(f"LIVE BEFORE provenance: is_synthetic counts {provenance}")

    if dry_run:
        print("DRY RUN complete. No rows written. Re-run with --execute to write.")
        return 0
    if problem is not None:
        print("ERROR: the dataset guard refused the load; nothing written.")
        return 1
    if client is None:
        print("ERROR: cannot --execute without a Supabase client.")
        return 1

    expected_records = to_records(df)
    n = upsert(
        client,
        expected_records,
        spec=spec,
        batch_size=args.batch_size,
        deployed_commit=args.deployed_commit,
    )
    print(f"EXECUTE: upserted {n} rows into {spec.table} (idempotent on {spec.on_conflict}).")

    live_after = fetch_live_split(client, spec=spec)
    if live_after is None:
        print("MISMATCH: could not re-read the live table after the write.")
        return 1
    print_split("LIVE AFTER", live_after, spec=spec)
    provenance_after = fetch_live_provenance_counts(client, spec=spec)
    if provenance_after is not None:
        print(f"LIVE AFTER provenance: is_synthetic counts {provenance_after}")
    problems = verify(expected, live_after, spec=spec)

    live_rows = _fetch_rows(client)
    if live_rows is None:
        print("MISMATCH: could not re-read the live table's rows for row-level verification.")
        return 1
    row_problems = verify_rows(expected_records, live_rows)
    print(
        f"ROW VERIFICATION: compared {len(expected_records)} exported rows against {len(live_rows)} live rows."
    )

    all_problems = problems + row_problems
    if all_problems:
        print("MISMATCH between the parquet and the live table:")
        for p in all_problems:
            print(f"  - {p}")
        return 1
    print(
        "VERIFIED: live arm split, treatment counts, per-outcome positives, and "
        "every exported field of every row equal the parquet."
    )
    return 0
