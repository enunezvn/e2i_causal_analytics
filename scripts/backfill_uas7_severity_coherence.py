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
Exactly the rows created before :data:`OLD_GENERATOR_CUTOFF`. The copula did not exist
before that instant, so no older row can hold a mapped value. A correlation is NOT used
to decide which rows are old (a cohort mixing both generators passes any correlation
band); it only detects that the selected set was already rewritten.

Rows created AT OR AFTER the cutoff are ambiguous. The weekly append runs from the HOST
checkout (``scripts/reseed_synthetic.sh``), so which generator wrote a batch depends on
when that checkout was reset to a commit containing the copula, which no container
timestamp records. There is deliberately no way to widen the selection: a wrong guess
would remap new-generator rows a second time and still pass every correlation guard.
Run this after the deploy and BEFORE the next Monday 03:00 append, when no such rows
exist. If some do, the script refuses. ``--newer-rows-are-new-generator`` may then only
EXCLUDE them, after checking ``/home/enunez/logs/e2i-reseed.log`` against the deploy's
checkout reset. An old-generator batch excluded by mistake stays independent, which is
the pre-change state and not a corrupted one.

PERSISTENCE LABELS (same transaction)
-------------------------------------
The UAS7 >= 28 axis moves (1,106 live rows cross 28) and drives the planted persistence
differential, so the labels have to follow. They are NOT re-derived wholesale:
``scripts/backfill_brand_axis_persistence.py --execute`` rewrites 3,132 of the 8,863
live labels (measured 2026-09-16), because the live labels never came from its RNG
stream (34% disagree even at the current UAS7). Only the change the new UAS7 CAUSES is
applied: the old and the new UAS7 both go through that script's ``regenerate`` (same
seed, same cohort, so the two streams are paired draw for draw); where the two
regenerated labels differ the row takes the new one, everywhere else the live label
stays. Measured on the live export: 193 rows move, 139 labels change, adjusted
UAS7 >= 28 effect +0.1435 (full re-derivation +0.154, no label step +0.123), prevalence
0.5663, proxy AUC 0.7588 (full re-derivation 0.7585). ``regenerate`` is one stream over
the patient_id-sorted frame, so the pairing holds for the frame passed in; on the live
box the cutoff selection is the whole Remibrutinib cohort. Do not run
``backfill_brand_axis_persistence.py --brand Remibrutinib --execute`` after this.

ROLLOUT (in order)
------------------
1. This script, then apply its SQL (below). UAS7 and the labels move together.
2. Gold-standard retrain: the weekly ``scripts/reseed_synthetic.sh`` cron runs
   ``scripts/retrain_goldstd.sh`` (staging models + metric trends); until it does, the
   Remibrutinib persistence model is a fit to the old labels.
3. Serving layer, when it should catch up: ``scripts/sync_goldstd_serving.py`` in its
   documented order (SHAP bundles, Feast marker clear + full materialize, bentoml
   restart, SHAP cache refresh last). ``retrain_goldstd.sh`` deliberately does not do
   this, so it is the same step every weekly append already leaves to an operator.

SAFETY
------
* One psql transaction. The plan is staged in a temp table with every computation
  input (old UAS7, severity, old labels). EVERY staged row, changed or not, is locked and must
  still exist and match (synthetic, Remibrutinib, older than the cutoff, same severity,
  same UAS7, same labels) before anything is updated; then the changed rows are compare-and-set.
  ``IS DISTINCT FROM`` skips unchanged values so the ``updated_at`` trigger fires only
  on real changes. It asserts the staged label-change count, the updated count, that
  every staged row now carries its planned values, and the resulting correlation over
  the whole staged cohort (NULL fails) before COMMIT.
* Inputs are validated before any SQL is written: patient ids match
  :data:`_PATIENT_ID`, the persistence covariates are present, severity is finite in
  [0, 10], UAS7 is an integer in 16..42, the labels are complementary 0/1, and the
  correlation is defined.
* The TSV backup is the exact undo.

USAGE
-----
    docker exec supabase-db psql -U postgres -d postgres -c "\\copy (SELECT patient_id,
      brand, treatment_arm, disease_severity, academic_hcp, geographic_region,
      segment_assignment, insurance_type, age_at_diagnosis, comorbidity_burden,
      prior_therapy_lines, copay_support, psp_enrolled, persistent_180d,
      discontinued_180d, urticaria_severity_uas7, created_at FROM patient_journeys WHERE brand::text='Remibrutinib' AND is_synthetic
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

from scripts.backfill_brand_axis_persistence import (  # noqa: E402
    _AXES as _PERSISTENCE_AXES,
)
from scripts.backfill_brand_axis_persistence import (  # noqa: E402
    _BASE_COVARIATE_COLS,
    regenerate,
)
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


_REQUIRED_COLS = [*_BASE_COVARIATE_COLS, "urticaria_severity_uas7", "created_at"]


def _validate(rows: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_COLS if c not in rows.columns]
    if missing:
        raise SystemExit(f"REFUSING: export lacks the persistence covariates {missing}")
    bad_ids = rows.loc[~rows["patient_id"].astype(str).str.match(_PATIENT_ID), "patient_id"]
    if not bad_ids.empty:
        raise SystemExit(f"REFUSING: unexpected patient_id format, e.g. {bad_ids.iloc[0]!r}")
    sev = pd.to_numeric(rows["disease_severity"], errors="coerce")
    if sev.isna().any() or not sev.between(0.0, 10.0).all():
        raise SystemExit("REFUSING: disease_severity missing or outside [0, 10]")
    uas7 = pd.to_numeric(rows["urticaria_severity_uas7"], errors="coerce")
    if (uas7 % 1 != 0).any() or not uas7.between(UAS7_MIN, UAS7_MAX).all():
        raise SystemExit("REFUSING: urticaria_severity_uas7 not an integer in 16..42")
    persist = pd.to_numeric(rows["persistent_180d"], errors="coerce")
    disc = pd.to_numeric(rows["discontinued_180d"], errors="coerce")
    if not persist.isin([0, 1]).all() or not (persist + disc == 1).all():
        raise SystemExit(
            "REFUSING: persistent_180d / discontinued_180d are not complementary 0/1 labels"
        )


def _label_delta(rows: pd.DataFrame) -> None:
    """Old/new persistence labels: the live label, moved only where the UAS7 change moves
    the regenerated label (paired streams; see PERSISTENCE LABELS)."""
    cfg = _PERSISTENCE_AXES["Remibrutinib"]

    def regenerated(uas7_col: str) -> np.ndarray:
        frame = rows.assign(urticaria_severity_uas7=rows[uas7_col])
        out = regenerate(frame, cfg).set_index("patient_id")
        return out.loc[rows["patient_id"], "persistent_180d"].to_numpy(dtype=int)

    r0, r1 = regenerated("uas7_old"), regenerated("uas7_new")
    rows["persist_old"] = rows["persistent_180d"].astype(int)
    rows["disc_old"] = rows["discontinued_180d"].astype(int)
    rows["persist_new"] = np.where(r0 != r1, r1, rows["persist_old"].to_numpy())
    rows["disc_new"] = 1 - rows["persist_new"]
    rows["label_regen_moved"] = r0 != r1


def plan(live: pd.DataFrame, *, newer_rows_are_new_generator: bool = False) -> pd.DataFrame:
    """Old-generator rows with old/new UAS7. Raises when they were already rewritten or
    when rows at/after the cutoff exist and have not been vouched for."""
    cutoff = OLD_GENERATOR_CUTOFF
    live = live[live["urticaria_severity_uas7"].notna()]
    created = pd.to_datetime(live["created_at"], utc=True)
    newer = created >= cutoff
    if newer.any() and not newer_rows_are_new_generator:
        raise SystemExit(
            f"REFUSING: {int(newer.sum())} Remibrutinib rows were created at/after the cutoff "
            f"{cutoff} (earliest {created[newer].min()}); nothing records which generator "
            f"wrote them. Check the reseed log against the deploy's checkout reset, then pass "
            f"--newer-rows-are-new-generator to EXCLUDE them (they are never remapped)."
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
    _label_delta(rows)
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
    relabelled = int((rows["persist_old"] != rows["persist_new"]).sum())
    lines.append(
        f"persistence labels changed: {relabelled} "
        f"(regenerated label moved on {int(rows['label_regen_moved'].sum())} rows)  "
        f"prevalence {rows['persist_old'].mean():.4f} -> {rows['persist_new'].mean():.4f}"
    )
    return "\n".join(lines)


def to_sql(rows: pd.DataFrame) -> str:
    """One transaction: stage the whole cohort, lock and verify EVERY staged row, then
    compare-and-set the changed ones (UAS7 and labels), assert, commit."""
    _validate(rows)
    cutoff = OLD_GENERATOR_CUTOFF
    values = ",\n".join(
        f"('{pid}', {float(sev)!r}, {int(o)}, {int(n)}, {int(po)}, {int(do)}, {int(pn)}, {int(dn)})"
        for pid, sev, o, n, po, do, pn, dn in rows[
            [
                "patient_id",
                "disease_severity",
                "uas7_old",
                "uas7_new",
                "persist_old",
                "disc_old",
                "persist_new",
                "disc_new",
            ]
        ].itertuples(index=False)
    )
    label_changed = rows["persist_old"] != rows["persist_new"]
    n_changed = int(((rows["uas7_old"] != rows["uas7_new"]) | label_changed).sum())
    n_labels = int(label_changed.sum())
    lo, hi = UAS7_SEVERITY_RHO - 0.1, UAS7_SEVERITY_RHO + 0.1
    return f"""BEGIN;
CREATE TEMP TABLE _uas7_plan (
  patient_id text PRIMARY KEY, severity numeric, uas7_old int, uas7_new int,
  persist_old smallint, disc_old smallint, persist_new smallint, disc_new smallint
) ON COMMIT DROP;
INSERT INTO _uas7_plan VALUES
{values};
DO $$
DECLARE matched int; updated int; relabelled int; c double precision;
BEGIN
  PERFORM 1 FROM patient_journeys pj JOIN _uas7_plan p USING (patient_id) FOR UPDATE OF pj;
  SELECT count(*) INTO matched
  FROM patient_journeys pj JOIN _uas7_plan p USING (patient_id)
  WHERE pj.brand::text = 'Remibrutinib'
    AND pj.is_synthetic
    AND pj.created_at < '{cutoff.isoformat()}'::timestamptz
    AND pj.disease_severity = p.severity
    AND pj.urticaria_severity_uas7 = p.uas7_old
    AND pj.persistent_180d = p.persist_old
    AND pj.discontinued_180d = p.disc_old;
  IF matched <> {len(rows)} THEN
    RAISE EXCEPTION 'expected all {len(rows)} staged rows unchanged since the export, % match', matched;
  END IF;
  SELECT count(*) INTO relabelled FROM _uas7_plan WHERE persist_new <> persist_old;
  IF relabelled <> {n_labels} THEN
    RAISE EXCEPTION 'expected {n_labels} staged label changes, % staged', relabelled;
  END IF;
  UPDATE patient_journeys pj SET urticaria_severity_uas7 = p.uas7_new,
    persistent_180d = p.persist_new, discontinued_180d = p.disc_new
  FROM _uas7_plan p
  WHERE pj.patient_id = p.patient_id
    AND pj.brand::text = 'Remibrutinib'
    AND pj.is_synthetic
    AND pj.created_at < '{cutoff.isoformat()}'::timestamptz
    AND pj.disease_severity = p.severity
    AND pj.urticaria_severity_uas7 = p.uas7_old
    AND pj.persistent_180d = p.persist_old
    AND pj.discontinued_180d = p.disc_old
    AND (pj.urticaria_severity_uas7 IS DISTINCT FROM p.uas7_new
      OR pj.persistent_180d IS DISTINCT FROM p.persist_new);
  GET DIAGNOSTICS updated = ROW_COUNT;
  IF updated <> {n_changed} THEN
    RAISE EXCEPTION 'expected {n_changed} rows updated, got % (a row changed since the export)', updated;
  END IF;
  SELECT count(*) INTO matched
  FROM patient_journeys pj JOIN _uas7_plan p USING (patient_id)
  WHERE pj.urticaria_severity_uas7 IS DISTINCT FROM p.uas7_new
    OR pj.persistent_180d IS DISTINCT FROM p.persist_new
    OR pj.discontinued_180d IS DISTINCT FROM p.disc_new;
  IF matched <> 0 THEN
    RAISE EXCEPTION '% staged rows do not carry their planned values after the update', matched;
  END IF;
  SELECT corr(pj.urticaria_severity_uas7, pj.disease_severity) INTO c
  FROM patient_journeys pj JOIN _uas7_plan p USING (patient_id);
  IF c IS NULL OR c < {lo:.2f} OR c > {hi:.2f} THEN
    RAISE EXCEPTION 'post-update corr % outside [{lo:.2f}, {hi:.2f}]', c;
  END IF;
  RAISE NOTICE 'uas7 backfill: % rows updated ({n_labels} relabelled), corr now %', updated, c;
END $$;
COMMIT;
"""


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--in-csv", required=True, help="Export of the live Remibrutinib rows.")
    ap.add_argument(
        "--newer-rows-are-new-generator",
        action="store_true",
        help="Operator checked the reseed log: EXCLUDE rows created at/after the cutoff.",
    )
    ap.add_argument("--backup-dir", default=str(_PROJECT_ROOT / "data" / "backups"))
    ap.add_argument("--sql-out", help="Write the transactional UPDATE here (nothing is applied).")
    args = ap.parse_args()

    rows = plan(
        pd.read_csv(args.in_csv),
        newer_rows_are_new_generator=args.newer_rows_are_new_generator,
    )
    print(f"cutoff: {OLD_GENERATOR_CUTOFF}")
    print(report(rows))

    backup = Path(args.backup_dir)
    backup.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S")
    path = backup / f"remibrutinib_uas7_backup_{stamp}.tsv"
    rows[["patient_id", "disease_severity", "uas7_old", "persist_old", "disc_old"]].to_csv(
        path, sep="\t", index=False
    )
    print(f"backup: {path}")

    if args.sql_out:
        Path(args.sql_out).write_text(to_sql(rows))
        print(f"sql: {args.sql_out}  (apply with psql -v ON_ERROR_STOP=1)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
