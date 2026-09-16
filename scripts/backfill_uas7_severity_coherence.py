#!/usr/bin/env python3
"""In-place: make live Remibrutinib UAS7 follow disease_severity, like the generator now does.

WHY
---
The generator ties UAS7 to the latent severity that cuts the severity tier
(src/ml/synthetic/dgp/clinical_severity.py, 2026-09-16). The live droplet is FROZEN
(a full reseed is disaster recovery only), so existing rows keep the old independent
UAS7 unless they are brought into line here. The stored UAS7 of a row generated
BEFORE the change IS the generator's raw ``integers(16, 43)`` draw, i.e. exactly the
copula's noise term, so the new value is a deterministic recompute from two stored
columns — the same function the generator calls, not a re-draw.

After this, run ``scripts/backfill_brand_axis_persistence.py --brand Remibrutinib``:
the UAS7 >= 28 axis membership moves, and persistence labels must follow it.

SAFETY
------
* NOT idempotent by construction (the mapping is applied to its own output on a
  re-run), so it fails closed instead: only rows created before ``--created-before``
  (rows written by the old generator) are read, and it REFUSES when those rows already
  show the designed correlation. The TSV backup is the exact undo.
* Writes through psql in ONE transaction that asserts the row count and the resulting
  correlation before COMMIT; ``WHERE ... IS DISTINCT FROM`` skips unchanged rows so the
  ``updated_at`` trigger fires only on real changes.

USAGE
-----
    # 1. read-only report + backup + SQL file
    python -m scripts.backfill_uas7_severity_coherence --created-before 2026-09-16T12:00:00Z \\
        --in-csv remi.csv --sql-out apply.sql
    # 2. apply
    docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 < apply.sql

``--in-csv`` is ``\\copy (SELECT patient_id, disease_severity, urticaria_severity_uas7,
segment_assignment, created_at FROM patient_journeys WHERE brand::text='Remibrutinib'
AND is_synthetic) TO STDOUT WITH CSV HEADER``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.ml.synthetic.dgp.clinical_severity import (  # noqa: E402
    UAS7_SEVERITY_RHO,
    uas7_from_severity,
)

#: Rows already showing more correlation than this were written by the new generator
#: (or this script already ran): refuse. Old-generator rows measure ~0.01.
ALREADY_APPLIED_CORR = 0.15


def _corr(frame: pd.DataFrame, col: str) -> float:
    return float(
        np.corrcoef(frame[col].astype(float), frame["disease_severity"].astype(float))[0, 1]
    )


def plan(live: pd.DataFrame, created_before: pd.Timestamp) -> pd.DataFrame:
    """Rows to rewrite, with old/new UAS7. Raises when the selection is already applied."""
    rows = live[pd.to_datetime(live["created_at"], utc=True) < created_before].copy()
    rows = rows[rows["urticaria_severity_uas7"].notna()]
    if rows.empty:
        raise SystemExit("No Remibrutinib rows with UAS7 created before the cutoff.")
    before = _corr(rows, "urticaria_severity_uas7")
    if before > ALREADY_APPLIED_CORR:
        raise SystemExit(
            f"REFUSING: corr(UAS7, severity) is already {before:.3f} on the selected rows "
            f"(> {ALREADY_APPLIED_CORR}); they were written by the new generator or this "
            f"script already ran. Restore from the TSV backup first if a re-run is intended."
        )
    rows["uas7_old"] = rows["urticaria_severity_uas7"].astype(int)
    rows["uas7_new"] = uas7_from_severity(
        rows["uas7_old"].to_numpy(), rows["disease_severity"].astype(float).to_numpy()
    )
    return rows


def report(rows: pd.DataFrame) -> str:
    lines = [f"rows selected: {len(rows)}  rho={UAS7_SEVERITY_RHO}"]
    for col in ("uas7_old", "uas7_new"):
        by_tier = rows.groupby("segment_assignment")[col]
        lines.append(
            f"{col}: corr={_corr(rows, col):.3f}  P(>=28)={(rows[col] >= 28).mean():.3f}  "
            f"mean by tier={by_tier.mean().round(1).to_dict()}  "
            f"P(>=28) by tier={by_tier.apply(lambda s: round(float((s >= 28).mean()), 3)).to_dict()}"
        )
    changed = int((rows["uas7_old"] != rows["uas7_new"]).sum())
    flips = int(((rows["uas7_old"] >= 28) != (rows["uas7_new"] >= 28)).sum())
    lines.append(f"values changed: {changed}  uncontrolled-CSU axis flips: {flips}")
    return "\n".join(lines)


def to_sql(rows: pd.DataFrame) -> str:
    """One transaction: stage, update changed rows only, assert, commit."""
    changed = rows[rows["uas7_old"] != rows["uas7_new"]]
    values = ",\n".join(
        f"('{pid}', {int(o)}, {int(n)})"
        for pid, o, n in changed[["patient_id", "uas7_old", "uas7_new"]].itertuples(index=False)
    )
    n = len(changed)
    lo, hi = UAS7_SEVERITY_RHO - 0.1, UAS7_SEVERITY_RHO + 0.1
    ids = ", ".join(f"'{pid}'" for pid in rows["patient_id"])
    return f"""BEGIN;
CREATE TEMP TABLE _uas7_plan (patient_id text PRIMARY KEY, uas7_old int, uas7_new int) ON COMMIT DROP;
INSERT INTO _uas7_plan VALUES
{values};
DO $$
DECLARE updated int; c double precision;
BEGIN
  UPDATE patient_journeys pj SET urticaria_severity_uas7 = p.uas7_new
  FROM _uas7_plan p
  WHERE pj.patient_id = p.patient_id
    AND pj.brand::text = 'Remibrutinib'
    AND pj.urticaria_severity_uas7 = p.uas7_old
    AND pj.urticaria_severity_uas7 IS DISTINCT FROM p.uas7_new;
  GET DIAGNOSTICS updated = ROW_COUNT;
  IF updated <> {n} THEN
    RAISE EXCEPTION 'expected {n} rows updated, got % (a row changed since the export)', updated;
  END IF;
  SELECT corr(urticaria_severity_uas7, disease_severity) INTO c
  FROM patient_journeys WHERE patient_id IN ({ids});
  IF c < {lo:.2f} OR c > {hi:.2f} THEN
    RAISE EXCEPTION 'post-update corr % outside [{lo:.2f}, {hi:.2f}]', c;
  END IF;
  RAISE NOTICE 'uas7 backfill: % rows updated, corr now %', updated, c;
END $$;
COMMIT;
"""


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--created-before", required=True, help="ISO timestamp; only older rows are touched."
    )
    ap.add_argument(
        "--in-csv", required=True, help="Export of the live Remibrutinib rows (see USAGE)."
    )
    ap.add_argument("--backup-dir", default=str(_PROJECT_ROOT / "data" / "backups"))
    ap.add_argument("--sql-out", help="Write the transactional UPDATE here (nothing is applied).")
    args = ap.parse_args()

    live = pd.read_csv(args.in_csv)
    rows = plan(live, pd.Timestamp(args.created_before).tz_convert("UTC"))
    print(report(rows))

    backup = Path(args.backup_dir)
    backup.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S")
    path = backup / f"remibrutinib_uas7_backup_{stamp}.tsv"
    rows[["patient_id", "uas7_old"]].to_csv(path, sep="\t", index=False)
    print(f"backup: {path}")

    if args.sql_out:
        Path(args.sql_out).write_text(to_sql(rows))
        print(f"sql: {args.sql_out}  (apply with psql -v ON_ERROR_STOP=1)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
