#!/usr/bin/env python3
"""Idempotent loader for ``csu_escalation_causal`` (migration 149): the SYNTHETIC
backing of Lane C's remibrutinib-vs-competitor dataset (Lane C-load, 2026-09-23;
PR #2228 owner decision 2).

Reads the parquet written by ``scripts/build_csu_escalation_synthetic_cohort.py``
(fixed seed 20260922, n = 3,000, every row ``is_synthetic = true``, ground-truth
sidecar alongside) and upserts it on ``patient_id`` through the PostgREST client,
under the SAME discipline as Lane A's ``scripts/load_optum_causal_cohort.py`` --
the engine is ``scripts/causal_cohort_loader.py``, this module binds the CSU
``CohortSpec``:

* exact 81-column contract (Lane A's with ``treatment_remibrutinib``), refused
  on a missing OR an unexpected column;
* provenance: EVERY row must be ``is_synthetic = true`` -- a real-looking row
  (false or NULL) is refused, because real mode would serve it as real;
* ``treatment_remibrutinib`` == (``index_biologic_brand`` == RHAPSIDO), only
  the RHAPSIDO / XOLAIR / DUPIXENT labels, both arms present, 0/1 outcomes;
* GUARD precondition on the write path: the tree's
  ``datasets._CAUSAL_SYNTHETIC_BACKED`` must list the dataset, the
  planted-truth seam must be closed, AND the deployed ``e2i_api`` image must
  descend from the commit that put the guard on main (read from docker, or
  attested with ``--deployed-commit <sha>``) -- else the deployed showcase
  flag would serve the planted rows as real the moment they land; refused,
  exit 1, nothing written. The same checks run again inside ``upsert``
  itself, so no caller can write around them;
* after ``--execute``: LIVE AFTER arm split / treatment counts / per-outcome
  positives / provenance counts and a patient-by-patient field comparison
  against the parquet; any disagreement is MISMATCH, exit 1.

USAGE
-----
    # DEFAULT: dry run (validates the parquet, prints WOULD WRITE + LIVE BEFORE,
    # writes nothing). Run from the checkout that carries this script; a lane
    # worktree finds the main checkout's .env up the tree.
    python -m scripts.load_csu_escalation_cohort --input data/rwd/synthetic_CSU/csu_escalation_causal/csu_escalation_causal_synthetic.parquet

    # WRITE PATH -- owner-GO step (a production write; the owner runs it).
    python -m scripts.load_csu_escalation_cohort --input <parquet> --execute
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
from scripts.causal_cohort_loader import (  # noqa: E402,F401 -- the loader's public names
    BATCH_SIZE,
    JOURNEY_METADATA_COLUMNS,
    OUTCOME_COLUMNS,
    ROW_PAGE_SIZE,
    CohortSpec,
    to_records,
    verify_rows,
)
from scripts.causal_cohort_loader import (
    CSU_SPEC as SPEC,
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


def upsert(
    client: Any,
    records: List[Dict[str, Any]],
    *,
    batch_size: int = BATCH_SIZE,
    deployed_commit: Optional[str] = None,
) -> int:
    """The write boundary: refuses (raises) on any non-synthetic record or an
    unmet guard BEFORE the first batch -- see ``causal_cohort_loader.upsert``."""
    return _engine.upsert(
        client, records, spec=SPEC, batch_size=batch_size, deployed_commit=deployed_commit
    )


def fetch_live_split(client: Any) -> Optional[Dict[str, Any]]:
    return _engine.fetch_live_split(client, spec=SPEC)


def fetch_live_provenance_counts(client: Any) -> Optional[Dict[str, int]]:
    return _engine.fetch_live_provenance_counts(client, spec=SPEC)


def verify(expected: Dict[str, Any], live: Dict[str, Any]) -> List[str]:
    return _engine.verify(expected, live, spec=SPEC)


def fetch_live_rows(client: Any) -> Optional[List[Dict[str, Any]]]:
    return _engine.fetch_live_rows(client, spec=SPEC)


def guard_problem(deployed_commit: Optional[str] = None) -> Optional[str]:
    """None when THIS tree's dataset guard covers the dataset, the planted-truth
    seam is closed, and the DEPLOYED e2i_api image descends from the guard
    commit; otherwise the reason the load must not run."""
    return _engine.guard_problem(SPEC, deployed_commit=deployed_commit)


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
