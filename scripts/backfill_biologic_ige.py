#!/usr/bin/env python3
"""Backfill the Phase 2 anti-IgE axis (biologic_experienced + ige_level) on the
EXISTING synthetic patient_journeys rows — for Remibrutinib/CSU patients ONLY.

WHY THIS SCRIPT EXISTS
----------------------
Phase 2 makes the biologic-naive/experienced + baseline-IgE axis (the axis the copilot
chatbot used to FABRICATE) real. The generator (patient_generator.py) now draws these
columns for fresh loads, but the live droplet is NOT reseeded — a full reseed would
clobber the ad-hoc "boosted" persistence labels that are only ~63% reproducible (see
regenerate_cohort_outcomes.py). So, exactly like that script, this brings the EXISTING
rows into line in place: it populates biologic_experienced (~40% Bernoulli) and
ige_level (lognormal, ~150 IU/mL median) for the Remibrutinib rows and leaves every
other brand NULL (the columns were added NULL by migration 107 and the gating leaves
off-brand indication columns NULL).

The values are drawn from the SAME distributions the generator uses, but are NOT
required to bit-match the generator's shared-RNG draws (the live table already diverges
from a fresh generate on the boosted labels). Determinism here is PER PATIENT: the RNG
is seeded from a stable hash of patient_id, so the backfill is fully reproducible and
IDEMPOTENT — re-running writes the identical values.

PREREQUISITES
-------------
  * migration 107 applied (adds biologic_experienced + ige_level, gates off-brand cols).
  * the brand-aware causal/segment code deployed (so gated NULLs never reach EconML).

USAGE
-----
    # DEFAULT: dry-run. Reads the Remibrutinib rows, computes values, reports the
    # distribution, writes NOTHING.
    python scripts/backfill_biologic_ige.py

    # WRITE PATH -- UPDATE the ~8k Remibrutinib rows in the live table.
    python scripts/backfill_biologic_ige.py --execute
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("backfill_biologic_ige")

TABLE = "patient_journeys"
KEY = "patient_id"
BRAND = "Remibrutinib"
BATCH_LOG = 1000

# Distribution constants — MUST match patient_generator.py's Phase 2 draws so the
# in-place backfill is distributionally identical to a fresh generate.
BIOLOGIC_EXPERIENCED_PREVALENCE = 0.40
IGE_LOGNORMAL_MEAN = 5.0
IGE_LOGNORMAL_SIGMA = 0.8
IGE_CLIP = (2.0, 2000.0)


def _rng_for(patient_id: str) -> np.random.Generator:
    """Deterministic per-patient RNG (stable across runs → idempotent writes)."""
    seed = int.from_bytes(hashlib.sha256(patient_id.encode()).digest()[:8], "big")
    return np.random.default_rng(seed)


def _draw(patient_id: str) -> tuple[int, float]:
    rng = _rng_for(patient_id)
    bio = int(rng.random() < BIOLOGIC_EXPERIENCED_PREVALENCE)
    ige = round(
        float(np.clip(rng.lognormal(IGE_LOGNORMAL_MEAN, IGE_LOGNORMAL_SIGMA), *IGE_CLIP)), 1
    )
    return bio, ige


def fetch_remibrutinib_ids(client: Any) -> list[str]:
    """All Remibrutinib patient_ids (paginated; PostgREST caps at 1000/req)."""
    ids: list[str] = []
    page = 0
    while True:
        lo, hi = page * 1000, page * 1000 + 999
        res = (
            client.table(TABLE)
            .select(KEY)
            .eq("brand", BRAND)
            .order(KEY)  # stable order so range pagination never dupes/skips rows
            .range(lo, hi)
            .execute()
        )
        rows = res.data or []
        ids.extend(r[KEY] for r in rows)
        if len(rows) < 1000:
            break
        page += 1
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="WRITE PATH: UPDATE biologic_experienced/ige_level for Remibrutinib rows. "
        "Omit (default) for a read-only dry-run.",
    )
    args = parser.parse_args()
    dry_run = not args.execute

    logger.info("=" * 68)
    logger.info(
        "Phase 2 anti-IgE backfill  (%s)  brand=%s", "DRY RUN" if dry_run else "EXECUTE", BRAND
    )
    logger.info("=" * 68)

    try:
        from src.memory.services.factories import get_supabase_client

        client = get_supabase_client()
    except Exception as e:  # pragma: no cover - env dependent
        logger.error("No Supabase client (%s). Run from the prod box with env set.", e)
        return 1
    if client is None:
        logger.error("No Supabase client. Set SUPABASE_URL + a service key.")
        return 1

    ids = fetch_remibrutinib_ids(client)
    if not ids:
        logger.error("No %s rows found — nothing to backfill.", BRAND)
        return 1
    logger.info("Read %d %s patient rows.", len(ids), BRAND)

    values = {pid: _draw(pid) for pid in ids}
    bios = np.array([v[0] for v in values.values()])
    iges = np.array([v[1] for v in values.values()])
    logger.info(
        "Computed distribution: biologic_experienced mean=%.3f (target %.2f); "
        "ige_level median=%.1f IQR=[%.1f, %.1f] IU/mL",
        bios.mean(),
        BIOLOGIC_EXPERIENCED_PREVALENCE,
        float(np.median(iges)),
        float(np.percentile(iges, 25)),
        float(np.percentile(iges, 75)),
    )

    if dry_run:
        logger.info("DRY RUN complete. No rows written. Re-run with --execute to write.")
        return 0

    written = 0
    for pid, (bio, ige) in values.items():
        client.table(TABLE).update({"biologic_experienced": bio, "ige_level": ige}).eq(
            KEY, pid
        ).execute()
        written += 1
        if written % BATCH_LOG == 0:
            logger.info("  updated %d/%d rows", written, len(values))
    logger.info("EXECUTE complete. Updated %d %s rows.", written, BRAND)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
