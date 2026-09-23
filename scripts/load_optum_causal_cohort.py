#!/usr/bin/env python3
"""Idempotent loader for ``optum_biologic_persistence_causal`` (migration 148).

Lane A (spec docs/superpowers/specs/2026-09-22-real-data-causal-estimation-design.md
§3A.2): reads the causal export written by
``scripts/convert_optum_mart.py --cohort persistence_causal`` and upserts it on
``patient_id`` through the PostgREST client (the same path
``scripts/load_hcp_brand_adoption.py`` uses — the worker containers mount no
``data/rwd`` volume, so this runs on the host venv).

The engine is ``scripts/causal_cohort_loader.py`` (Lane C-load generalised it
so ``csu_escalation_causal`` loads under the SAME discipline); this module binds
the Optum ``CohortSpec`` and keeps Lane A's public names. Two deliberate
differences from the pre-generalisation module: ``is_synthetic`` must now be an
exact boolean on every row (a NULL, an int or the string "false" is refused
instead of being truth-coerced -- the outcome for the Optum table is the same
refusal, the message names the cause), and the one-arm refusal names the arm
by role ("treated-arm rows (DUPIXENT ...)" / "control-arm rows (XOLAIR ...)").
The live row re-read is ordered by ``patient_id`` so its pages are stable.

Fail-loud validation BEFORE any write: the frame must carry EXACTLY the export's
81-column contract (7 journey-metadata keys + the primary outcome + the 64
``MART_SAFE_FEATURES`` baseline covariates + ``data_quality_score`` + the 6
``CAUSAL_EXTRA_COLS`` + ``is_synthetic`` + ``data_split`` -- no missing column, no
unexpected extra one, so a frame missing baseline confounders can never silently
upsert NULLs into them), every row ``is_synthetic == False``, ``patient_id``
unique, the treatment coding agrees with ``index_biologic_brand`` and only the
observed two-arm contrast is present, every outcome is 0/1. After ``--execute``
the live table's arm split, treatment-column counts, and per-outcome positives are
re-read and compared with the parquet, AND every exported field of every row is
re-read and compared patient-by-patient (``fetch_live_rows``/``verify_rows``) --
matching aggregate margins alone cannot hide wrong patient_ids or corrupted
covariates. A disagreement at either level is printed as MISMATCH and the exit
code is 1 — the load is never reported as verified on the strength of the write
call alone.

USAGE
-----
    # DEFAULT: dry run. Validates the parquet, prints the arm split it WOULD
    # write and (if reachable) the live table's current split. Writes nothing.
    # --dry-run is an explicit alias for this default (mutually exclusive with
    # --execute).
    python -m scripts.load_optum_causal_cohort --input data/rwd/mart/persistence_causal/e2i_causal_v1_biologic_persistence.parquet

    # WRITE PATH — owner-GO step (spec §7 records the GO for the production load).
    python -m scripts.load_optum_causal_cohort --input <parquet> --execute
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts import causal_cohort_loader as _engine  # noqa: E402
from scripts.causal_cohort_loader import (  # noqa: E402,F401 -- Lane A's public names
    BATCH_SIZE,
    JOURNEY_METADATA_COLUMNS,
    OUTCOME_COLUMNS,
    ROW_PAGE_SIZE,
    CohortSpec,
    to_records,
    verify_rows,
)
from scripts.causal_cohort_loader import (
    OPTUM_SPEC as SPEC,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TABLE = SPEC.table
ON_CONFLICT = SPEC.on_conflict
DEFAULT_INPUT = SPEC.default_input
TREATMENT = SPEC.treatment
BRAND = SPEC.brand
ARMS = SPEC.arms
REQUIRED_COLUMNS = SPEC.required_columns


def load_frame(path: Path | str) -> "Any":
    return _engine.load_frame(path, spec=SPEC)


def arm_split(df: Any) -> Dict[str, Any]:
    return _engine.arm_split(df, spec=SPEC)


def _client() -> Any:
    return _engine.get_client()


def upsert(client: Any, records: List[Dict[str, Any]], *, batch_size: int = BATCH_SIZE) -> int:
    return _engine.upsert(client, records, spec=SPEC, batch_size=batch_size)


def fetch_live_split(client: Any) -> Optional[Dict[str, Any]]:
    return _engine.fetch_live_split(client, spec=SPEC)


def verify(expected: Dict[str, Any], live: Dict[str, Any]) -> List[str]:
    return _engine.verify(expected, live, spec=SPEC)


def fetch_live_rows(client: Any) -> Optional[List[Dict[str, Any]]]:
    return _engine.fetch_live_rows(client, spec=SPEC)


def main(argv: Optional[List[str]] = None) -> int:
    # The seams are looked up on THIS module at call time so a monkeypatch of
    # ``_client`` / ``fetch_live_rows`` here is what the run uses.
    return _engine.run(
        SPEC,
        argv,
        client_factory=lambda: _client(),
        fetch_live_rows_fn=lambda c: fetch_live_rows(c),
        doc=__doc__,
    )


if __name__ == "__main__":
    sys.exit(main())
