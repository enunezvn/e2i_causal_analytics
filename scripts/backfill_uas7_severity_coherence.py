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

WHICH ROWS
----------
The rows created before ``--cutoff`` (default :data:`OLD_GENERATOR_CUTOFF`; an explicit
cutoff may only be LATER, e.g. the deployed api container's StartedAt). The copula did
not exist before 2026-09-16, so no older row can hold a mapped value. A correlation is
NOT used to decide which rows are old (a cohort mixing both generators passes any
correlation band); it only detects that the selected set was already rewritten.

Rows created AT OR AFTER the cutoff are ambiguous: a weekly append that ran before the
deploy wrote them with the OLD generator, one after it with the new. The script refuses
while any exist, unless the operator has checked them against the deploy time and passes
``--newer-rows-are-new-generator`` (otherwise raise ``--cutoff`` to the deploy time so
the old-generator ones are included).

ROLLOUT (all three, in order)
-----------------------------
1. This script, then apply its SQL (below).
2. ``scripts/backfill_brand_axis_persistence.py --brand Remibrutinib --execute``: the
   UAS7 >= 28 axis moves (1,106 live rows cross 28), so the persistence labels must be
   re-derived; without ``--execute`` that script is a dry run.
3. Gold-standard retrain: the weekly ``scripts/reseed_synthetic.sh`` cron runs
   ``scripts/retrain_goldstd.sh`` (staging models + metric trends); until it does, the
   Remibrutinib persistence model is a fit to the old labels.
4. Serving layer, when it should catch up: ``scripts/sync_goldstd_serving.py`` in its
   documented order (SHAP bundles, Feast marker clear + full materialize, bentoml
   restart, SHAP cache refresh last). ``retrain_goldstd.sh`` deliberately does not do
   this, so it is the same step every weekly append already leaves to an operator.

SAFETY
------
* One psql transaction. The plan is staged in a temp table with every computation
  input (old UAS7, severity). EVERY staged row, changed or not, is locked and must
  still exist and match (synthetic, Remibrutinib, older than the cutoff, same severity,
  same UAS7) before anything is updated; then the changed rows are compare-and-set.
  ``IS DISTINCT FROM`` skips unchanged values so the ``updated_at`` trigger fires only
  on real changes. It asserts the updated count and the resulting correlation over the
  whole staged cohort (NULL fails) before COMMIT.
* Inputs are validated before any SQL is written: patient ids match
  :data:`_PATIENT_ID`, severity is finite in [0, 10], UAS7 is an integer in 16..42, and
  the correlation is defined.
* The TSV backup is the exact undo.

USAGE
-----
    docker exec supabase-db psql -U postgres -d postgres -c "\\copy (SELECT patient_id,
      disease_severity, urticaria_severity_uas7, segment_assignment, created_at
      FROM patient_journeys WHERE brand::text='Remibrutinib' AND is_synthetic
      ORDER BY patient_id) TO STDOUT WITH CSV HEADER" > remi.csv
    python -m scripts.backfill_uas7_severity_coherence --in-csv remi.csv --sql-out apply.sql
    docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 < apply.sql
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.ml.synthetic.dgp.clinical_severity import (  # noqa: E402
    UAS7_MAX,
    UAS7_MIN,
    UAS7_SEVERITY_RHO,
    uas7_from_severity,
)

#: No row created before this instant can come from the copula generator (it was
#: written on this date and ships no earlier than its merge).
OLD_GENERATOR_CUTOFF = pd.Timestamp("2026-09-16T00:00:00Z")

#: Old-generator rows measure ~0.00; the rewritten set measures ~rho.
ALREADY_APPLIED_CORR = 0.15

_PATIENT_ID = re.compile(r"^[A-Za-z0-9_]+$")


def _corr(frame: pd.DataFrame, col: str) -> float:
    return float(
        np.corrcoef(frame[col].astype(float), frame["disease_severity"].astype(float))[0, 1]
    )


def _validate(rows: pd.DataFrame) -> None:
    bad_ids = rows.loc[~rows["patient_id"].astype(str).str.match(_PATIENT_ID), "patient_id"]
    if not bad_ids.empty:
        raise SystemExit(f"REFUSING: unexpected patient_id format, e.g. {bad_ids.iloc[0]!r}")
    sev = pd.to_numeric(rows["disease_severity"], errors="coerce")
    if sev.isna().any() or not sev.between(0.0, 10.0).all():
        raise SystemExit("REFUSING: disease_severity missing or outside [0, 10]")
    uas7 = pd.to_numeric(rows["urticaria_severity_uas7"], errors="coerce")
    if (uas7 % 1 != 0).any() or not uas7.between(UAS7_MIN, UAS7_MAX).all():
        raise SystemExit("REFUSING: urticaria_severity_uas7 not an integer in 16..42")


def plan(
    live: pd.DataFrame,
    cutoff: pd.Timestamp = OLD_GENERATOR_CUTOFF,
    *,
    newer_rows_are_new_generator: bool = False,
) -> pd.DataFrame:
    """Old-generator rows with old/new UAS7. Raises when they were already rewritten or
    when rows at/after the cutoff exist and have not been vouched for."""
    if cutoff < OLD_GENERATOR_CUTOFF:
        raise SystemExit(f"REFUSING: --cutoff may not precede {OLD_GENERATOR_CUTOFF}.")
    live = live[live["urticaria_severity_uas7"].notna()]
    created = pd.to_datetime(live["created_at"], utc=True)
    newer = created >= cutoff
    if newer.any() and not newer_rows_are_new_generator:
        raise SystemExit(
            f"REFUSING: {int(newer.sum())} Remibrutinib rows were created at/after the cutoff "
            f"{cutoff} (earliest {created[newer].min()}). A weekly append before the deploy "
            f"wrote them with the OLD generator. Raise --cutoff to the deployed api "
            f"container's StartedAt, or pass --newer-rows-are-new-generator once checked."
        )
    rows = live[~newer].copy()
    if rows.empty:
        raise SystemExit("No Remibrutinib rows with UAS7 created before the cutoff.")
    _validate(rows)
    before = _corr(rows, "urticaria_severity_uas7")
    if not np.isfinite(before):
        raise SystemExit("REFUSING: corr(UAS7, severity) is undefined (constant input).")
    if before > ALREADY_APPLIED_CORR:
        raise SystemExit(
            f"REFUSING: corr(UAS7, severity) is already {before:.3f} on the old-generator "
            f"rows (> {ALREADY_APPLIED_CORR}); this script already ran. Restore from the TSV "
            f"backup first if a re-run is intended."
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


def to_sql(rows: pd.DataFrame, cutoff: pd.Timestamp = OLD_GENERATOR_CUTOFF) -> str:
    """One transaction: stage the whole cohort, lock and verify EVERY staged row, then
    compare-and-set the changed ones, assert, commit."""
    _validate(rows)
    values = ",\n".join(
        f"('{pid}', {float(sev)!r}, {int(o)}, {int(n)})"
        for pid, sev, o, n in rows[
            ["patient_id", "disease_severity", "uas7_old", "uas7_new"]
        ].itertuples(index=False)
    )
    n_changed = int((rows["uas7_old"] != rows["uas7_new"]).sum())
    lo, hi = UAS7_SEVERITY_RHO - 0.1, UAS7_SEVERITY_RHO + 0.1
    return f"""BEGIN;
CREATE TEMP TABLE _uas7_plan (
  patient_id text PRIMARY KEY, severity numeric, uas7_old int, uas7_new int
) ON COMMIT DROP;
INSERT INTO _uas7_plan VALUES
{values};
DO $$
DECLARE matched int; updated int; c double precision;
BEGIN
  PERFORM 1 FROM patient_journeys pj JOIN _uas7_plan p USING (patient_id) FOR UPDATE OF pj;
  SELECT count(*) INTO matched
  FROM patient_journeys pj JOIN _uas7_plan p USING (patient_id)
  WHERE pj.brand::text = 'Remibrutinib'
    AND pj.is_synthetic
    AND pj.created_at < '{cutoff.isoformat()}'::timestamptz
    AND pj.disease_severity = p.severity
    AND pj.urticaria_severity_uas7 = p.uas7_old;
  IF matched <> {len(rows)} THEN
    RAISE EXCEPTION 'expected all {len(rows)} staged rows unchanged since the export, % match', matched;
  END IF;
  UPDATE patient_journeys pj SET urticaria_severity_uas7 = p.uas7_new
  FROM _uas7_plan p
  WHERE pj.patient_id = p.patient_id
    AND pj.brand::text = 'Remibrutinib'
    AND pj.is_synthetic
    AND pj.created_at < '{cutoff.isoformat()}'::timestamptz
    AND pj.disease_severity = p.severity
    AND pj.urticaria_severity_uas7 = p.uas7_old
    AND pj.urticaria_severity_uas7 IS DISTINCT FROM p.uas7_new;
  GET DIAGNOSTICS updated = ROW_COUNT;
  IF updated <> {n_changed} THEN
    RAISE EXCEPTION 'expected {n_changed} rows updated, got % (a row changed since the export)', updated;
  END IF;
  SELECT corr(pj.urticaria_severity_uas7, pj.disease_severity) INTO c
  FROM patient_journeys pj JOIN _uas7_plan p USING (patient_id);
  IF c IS NULL OR c < {lo:.2f} OR c > {hi:.2f} THEN
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
    ap.add_argument("--in-csv", required=True, help="Export of the live Remibrutinib rows.")
    ap.add_argument(
        "--cutoff",
        default=str(OLD_GENERATOR_CUTOFF),
        help="Rows created before this are old-generator rows (never earlier than the default).",
    )
    ap.add_argument(
        "--newer-rows-are-new-generator",
        action="store_true",
        help="Operator checked: every row at/after --cutoff was written by the new generator.",
    )
    ap.add_argument("--backup-dir", default=str(_PROJECT_ROOT / "data" / "backups"))
    ap.add_argument("--sql-out", help="Write the transactional UPDATE here (nothing is applied).")
    args = ap.parse_args()

    cutoff = pd.Timestamp(args.cutoff).tz_convert("UTC")
    rows = plan(
        pd.read_csv(args.in_csv),
        cutoff,
        newer_rows_are_new_generator=args.newer_rows_are_new_generator,
    )
    print(f"cutoff: {cutoff}")
    print(report(rows))

    backup = Path(args.backup_dir)
    backup.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S")
    path = backup / f"remibrutinib_uas7_backup_{stamp}.tsv"
    rows[["patient_id", "disease_severity", "uas7_old"]].to_csv(path, sep="\t", index=False)
    print(f"backup: {path}")

    if args.sql_out:
        Path(args.sql_out).write_text(to_sql(rows, cutoff))
        print(f"sql: {args.sql_out}  (apply with psql -v ON_ERROR_STOP=1)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
