"""Lane T2 Task 1 — READ-ONLY disproof of the T2 premise, through the TWIN'S OWN loader.

The premise to disprove (plan .claude/plans/2026-09-23-lane-t2-twin-adopted-repoint.md):

  "the twin's OWN loader path, collapsed to (hcp, brand) and joined to
   hcp_brand_adoption.adopted, yields >= COHORT_MIN_ROWS (500) usable rows per brand in the
   window the twin actually uses, AND the DR interval makes engagement/speaker/peer/PSP
   significant in 3/3 brands."

Why through the twin's loader and not the T1 harness: verify_adoption_channel_recovery.py
builds its frame from scripts.backfill_hcp_treatment_arm.fetch_channel_rollups (is_synthetic
filter, exact-count paging, metric_date present).  The twin reads through
src.digital_twin.effect.cohort_loader.load_cohort_frame (different column list, NO
is_synthetic filter, single .limit(20000) with no paging).  A PASS on the harness frame is
not a PASS on the frame /simulate will actually see — that is the whole point of Task 1.

Also folds in the three review ride-alongs (plan "Review findings"):
  A  finding 1 — business_metrics has no `adopted`: the in-place constant flip 400s.
  A  finding 2 — hcp_brand_adoption.adopted is NOT NULL: the `.not_.is_(outcome,"null")`
     availability gate becomes vacuous.
  C  finding 3 — econml's _oob_preds aliasing: nanmean(_oob_preds) must equal the public ate_.

READ-ONLY: every DB call is .select(). No Redis, no writes, no MLflow.
"""

from __future__ import annotations

import json
import logging
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

WORKTREE = "/home/enunez/Projects/e2i_causal_analytics/.worktrees/twin-adopted-repoint"
sys.path.insert(0, WORKTREE)
logging.disable(logging.INFO)

import src  # noqa: E402

assert src.__file__.startswith(WORKTREE), f"WRONG TREE: {src.__file__}"

from src.data.per_hcp_cohort_collapse import (  # noqa: E402
    CHANNEL_COLUMNS,
    collapse_per_hcp_brand,
)
from src.digital_twin.effect.cohort_loader import (  # noqa: E402
    COHORT_METRIC_TYPE,
    COHORT_TABLE,
    _FETCH_LIMIT,
    load_cohort_frame,
)
from src.digital_twin.effect.provider import (  # noqa: E402
    COHORT_CONFOUNDERS,
    COHORT_MIN_ROWS,
    COHORT_OUTCOME_COLUMN,
)
from src.repositories import get_supabase_client  # noqa: E402

BRANDS = ("Remibrutinib", "Fabhalta", "Kisqali")
# The four the acceptance gate must make significant, plus the null that must cover 0.
FOCUS = (
    "engagement_score",
    "speaker_program_count",
    "peer_influence_score",
    "patient_support_enrollment",
    "rep_training_score",  # the NULL channel
)
OUT = Path(WORKTREE) / "docs/demos/results/2026-09-23_t2_premise_probe"
Z = 1.959963984540054


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)


def say(*a: Any) -> None:
    print(*a, flush=True)


# ---------------------------------------------------------------------------------------
# Part A — schema facts behind review findings 1 and 2
# ---------------------------------------------------------------------------------------
def part_a(client: Any) -> Dict[str, Any]:
    say("\n===== PART A — schema facts (review findings 1 and 2) =====")
    out: Dict[str, Any] = {}

    # A1: does business_metrics carry `adopted` at all?
    try:
        client.table(COHORT_TABLE).select("adopted").limit(1).execute()
        out["A1_business_metrics_has_adopted"] = True
        say("A1  business_metrics.adopted        : SELECT SUCCEEDED -> column EXISTS")
    except Exception as e:
        out["A1_business_metrics_has_adopted"] = False
        out["A1_error"] = str(e)[:300]
        say(f"A1  business_metrics.adopted        : REFUSED -> {str(e)[:180]}")

    # A2: the EXACT query _treatment_column_usable / _outcome_measurable_in_real_mode would
    #     build after an in-place COHORT_OUTCOME_COLUMN flip.
    try:
        r = (
            client.table(COHORT_TABLE)
            .select("hcp_id", count="exact")
            .eq("metric_type", COHORT_METRIC_TYPE)
            .not_.is_("adopted", "null")
            .limit(1)
            .execute()
        )
        out["A2_flipped_gate_query_ok"] = True
        out["A2_count"] = int(r.count or 0)
        say(f"A2  flipped availability gate query : SUCCEEDED count={r.count}")
    except Exception as e:
        out["A2_flipped_gate_query_ok"] = False
        out["A2_error"] = str(e)[:300]
        say(f"A2  flipped availability gate query : REFUSED -> {str(e)[:180]}")

    # A3: is `adopted` ever NULL? (NOT NULL => the .not_.is_(...,"null") gate is vacuous)
    tot = (
        client.table("hcp_brand_adoption")
        .select("id", count="exact")
        .eq("is_synthetic", True)
        .limit(1)
        .execute()
    )
    nulls = (
        client.table("hcp_brand_adoption")
        .select("id", count="exact")
        .eq("is_synthetic", True)
        .is_("adopted", "null")
        .limit(1)
        .execute()
    )
    out["A3_adoption_rows"] = int(tot.count or 0)
    out["A3_adopted_null_rows"] = int(nulls.count or 0)
    say(
        f"A3  hcp_brand_adoption rows={tot.count} adopted IS NULL={nulls.count}"
        f"  -> gate {'VACUOUS' if not nulls.count else 'still discriminates'}"
    )

    # A3b: the same question on the column the gate discriminates on TODAY, for contrast.
    cur_nulls = (
        client.table(COHORT_TABLE)
        .select("metric_id", count="exact")
        .eq("metric_type", COHORT_METRIC_TYPE)
        .is_(COHORT_OUTCOME_COLUMN, "null")
        .limit(1)
        .execute()
    )
    cur_tot = (
        client.table(COHORT_TABLE)
        .select("metric_id", count="exact")
        .eq("metric_type", COHORT_METRIC_TYPE)
        .limit(1)
        .execute()
    )
    out["A3b_rollup_rows"] = int(cur_tot.count or 0)
    out["A3b_current_outcome_null_rows"] = int(cur_nulls.count or 0)
    say(
        f"A3b per_hcp_rollup rows={cur_tot.count} {COHORT_OUTCOME_COLUMN} IS NULL="
        f"{cur_nulls.count}  (what the gate excludes today)"
    )
    return out


# ---------------------------------------------------------------------------------------
# Part B — the twin loader path: does it reach COHORT_MIN_ROWS on `adopted`?
# ---------------------------------------------------------------------------------------
def fetch_adoption(client: Any) -> pd.DataFrame:
    """Paged, total-ordered read of hcp_brand_adoption (mirrors fetch_live_adoption)."""
    rows: List[dict] = []
    page, size = 0, 1000
    while True:
        resp = (
            client.table("hcp_brand_adoption")
            .select("hcp_id,brand,adopted,treatment_arm,consideration_date,data_split")
            .eq("is_synthetic", True)
            .order("hcp_id")
            .order("brand")
            .range(page * size, (page + 1) * size - 1)
            .execute()
        )
        data = resp.data or []
        rows.extend(data)
        if len(data) < size:
            break
        page += 1
    df = pd.DataFrame(rows)
    if not df.empty:
        df["adopted"] = pd.to_numeric(df["adopted"], errors="coerce")
    return df


async def part_b(client: Any, aclient: Any) -> Dict[str, Any]:
    say("\n===== PART B — twin loader path -> collapse -> join adopted =====")
    out: Dict[str, Any] = {"brands": {}}

    adoption = fetch_adoption(client)
    out["adoption_rows_read"] = int(len(adoption))
    dup = int(adoption.duplicated(["hcp_id", "brand"]).sum()) if not adoption.empty else 0
    out["adoption_dup_pairs"] = dup
    say(f"B0  hcp_brand_adoption read {len(adoption)} rows, duplicate (hcp,brand) pairs={dup}")

    from src.digital_twin.effect.cohort_causal_estimator import _usable_rows

    frames: Dict[str, pd.DataFrame] = {}
    for brand in BRANDS:
        rec: Dict[str, Any] = {}
        # What the twin's own loader returns, and what the true row count is.
        exact = (
            client.table(COHORT_TABLE)
            .select("metric_id", count="exact")
            .eq("metric_type", COHORT_METRIC_TYPE)
            .eq("brand", brand)
            .limit(1)
            .execute()
        )
        t0 = time.time()
        df = await load_cohort_frame(aclient, brand)
        rec["loader_seconds"] = round(time.time() - t0, 1)
        rec["rollup_rows_exact"] = int(exact.count or 0)
        rec["loader_rows_returned"] = int(len(df))
        rec["loader_truncated"] = bool(len(df) < int(exact.count or 0))
        say(
            f"B1  {brand:<14} load_cohort_frame -> {len(df)} rows "
            f"(exact={exact.count}, _FETCH_LIMIT={_FETCH_LIMIT})"
            f"{'  *** TRUNCATED ***' if rec['loader_truncated'] else ''}"
        )
        if df.empty:
            out["brands"][brand] = rec
            continue

        # The loader's select carries no `brand` (it is .eq-filtered) and no metric_date.
        df = df.copy()
        df["brand"] = brand
        rec["loader_has_metric_date"] = "metric_date" in df.columns
        rec["loader_columns"] = sorted(df.columns.tolist())

        collapsed = collapse_per_hcp_brand(df)
        rec["collapsed_pairs"] = int(len(collapsed))
        # specialty is NOT a collapse column on the twin's frame unless the embed landed it
        if "specialty" in df.columns and "specialty" not in collapsed.columns:
            first = df.groupby(["hcp_id", "brand"], sort=True)["specialty"].first()
            collapsed = collapsed.merge(first.rename("specialty"), on=["hcp_id", "brand"], how="left")
        joined = collapsed.merge(
            adoption[["hcp_id", "brand", "adopted", "treatment_arm"]],
            on=["hcp_id", "brand"],
            how="inner",
        )
        rec["joined_rows"] = int(len(joined))
        rec["join_loss_pairs"] = int(len(collapsed) - len(joined))
        say(
            f"B2  {brand:<14} collapsed pairs={len(collapsed)} -> joined to adopted="
            f"{len(joined)} (lost {len(collapsed) - len(joined)})"
        )

        # The estimator's OWN usable-row rule, on `adopted`.
        per_channel: Dict[str, int] = {}
        for col in CHANNEL_COLUMNS:
            try:
                work = _usable_rows(
                    joined,
                    col,
                    outcome_col="adopted",
                    region_col="region",
                    confounders=COHORT_CONFOUNDERS,
                    specialty_col="specialty",
                )
                per_channel[col] = int(len(work))
            except Exception as e:  # noqa: BLE001
                per_channel[col] = -1
                rec.setdefault("usable_errors", {})[col] = str(e)[:200]
        rec["usable_rows_by_channel"] = per_channel
        worst = min(v for v in per_channel.values())
        rec["min_usable_rows"] = worst
        rec["meets_cohort_min_rows"] = bool(worst >= COHORT_MIN_ROWS)
        say(
            f"B3  {brand:<14} usable rows on `adopted` min={worst} "
            f"(COHORT_MIN_ROWS={COHORT_MIN_ROWS}) -> "
            f"{'PASS' if worst >= COHORT_MIN_ROWS else 'FAIL'}"
        )
        frames[brand] = joined
        out["brands"][brand] = rec

    out["_frames"] = frames
    return out


# ---------------------------------------------------------------------------------------
# Part C — DR interval vs shipped interval + the _oob_preds aliasing pin (finding 3)
# ---------------------------------------------------------------------------------------
def fit_one(sub: pd.DataFrame, col: str, seed: int = 42) -> Dict[str, Any]:
    """Mirror estimate_cohort_effect's fit EXACTLY, then read BOTH intervals off one forest."""
    from econml.dml import CausalForestDML
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    from src.digital_twin.effect.cohort_causal_estimator import (
        _LOG_CONFOUNDERS,
        _effect_modifier_matrix,
        _usable_rows,
    )

    work = _usable_rows(
        sub, col, outcome_col="adopted", region_col="region",
        confounders=COHORT_CONFOUNDERS, specialty_col="specialty",
    )
    rec: Dict[str, Any] = {"channel": col, "n": int(len(work))}
    t_thr = float(work["t_raw"].median())
    t = (work["t_raw"] > t_thr).astype(int).to_numpy()
    if len(np.unique(t)) < 2:
        rec["error"] = "no median contrast"
        return rec
    y = work["y"].to_numpy(dtype=float)
    x = _effect_modifier_matrix(work)
    cols = []
    for c in COHORT_CONFOUNDERS:
        v = work[c].to_numpy(dtype=float)
        cols.append(np.log1p(np.clip(v, 0.0, None)) if c in _LOG_CONFOUNDERS else v)
    w = np.column_stack(cols)

    cf = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=50, min_samples_leaf=5, random_state=seed),
        model_t=RandomForestClassifier(n_estimators=50, min_samples_leaf=5, random_state=seed),
        discrete_treatment=True, n_estimators=200, subforest_size=4,
        min_samples_leaf=10, random_state=seed,
    )
    cf.fit(y, t, X=x, W=w)

    lo, hi = cf.ate_interval(x, alpha=0.05)
    rec["shipped_lo"] = float(np.ravel(lo)[0])
    rec["shipped_hi"] = float(np.ravel(hi)[0])
    rec["shipped_width"] = rec["shipped_hi"] - rec["shipped_lo"]
    rec["shipped_excludes_0"] = bool(rec["shipped_lo"] > 0 or rec["shipped_hi"] < 0)

    ate = float(np.ravel(cf.ate_)[0])
    se = float(np.ravel(cf.ate_stderr_)[0])
    rec["dr_ate"] = ate
    rec["dr_se"] = se
    rec["dr_lo"] = ate - Z * se
    rec["dr_hi"] = ate + Z * se
    rec["dr_width"] = 2 * Z * se
    rec["dr_excludes_0"] = bool(rec["dr_lo"] > 0 or rec["dr_hi"] < 0)
    rec["width_ratio_shipped_over_dr"] = (
        rec["shipped_width"] / rec["dr_width"] if rec["dr_width"] else float("nan")
    )

    # ---- finding 3: the _oob_preds aliasing pin -------------------------------------
    final = cf.rlearner_model_final_
    oob = getattr(final, "_oob_preds", None)
    rec["oob_present"] = oob is not None
    if oob is not None:
        rec["oob_nanmean"] = float(np.nanmean(oob))
        rec["oob_matches_ate_"] = bool(np.isclose(rec["oob_nanmean"], ate, rtol=1e-9, atol=1e-12))
        rec["oob_nan_frac"] = float(np.mean(np.isnan(oob)))
        # the subset path Task 2 needs: econml's own _ate_and_stderr under a mask
        try:
            mask = np.ones(len(work), dtype=bool)
            p, s = final._ate_and_stderr(oob, mask)
            rec["ate_and_stderr_callable"] = True
            rec["ate_and_stderr_matches"] = bool(
                np.isclose(float(np.ravel(p)[0]), ate, rtol=1e-9, atol=1e-12)
                and np.isclose(float(np.ravel(s)[0]), se, rtol=1e-9, atol=1e-12)
            )
        except Exception as e:  # noqa: BLE001
            rec["ate_and_stderr_callable"] = False
            rec["ate_and_stderr_error"] = str(e)[:200]
    return rec


def part_c(frames: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    say("\n===== PART C — DR interval vs shipped, + _oob_preds pin (finding 3) =====")
    rows: List[dict] = []
    for brand, frame in frames.items():
        for col in FOCUS:
            t0 = time.time()
            try:
                rec = fit_one(frame, col)
            except Exception as e:  # noqa: BLE001
                rec = {"channel": col, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            rec["brand"] = brand
            rec["seconds"] = round(time.time() - t0, 1)
            rec["peak_rss_gib"] = round(rss_gib(), 3)
            rows.append(rec)
            if "error" in rec:
                say(f"C   {brand:<14} {col:<26} ERROR {rec['error']}")
            else:
                say(
                    f"C   {brand:<14} {col:<26} n={rec['n']:<5} ate={rec['dr_ate']:+.4f} "
                    f"shipped=({rec['shipped_lo']:+.4f},{rec['shipped_hi']:+.4f}) sig="
                    f"{str(rec['shipped_excludes_0']):<5} | DR=({rec['dr_lo']:+.4f},"
                    f"{rec['dr_hi']:+.4f}) sig={str(rec['dr_excludes_0']):<5} "
                    f"ratio={rec['width_ratio_shipped_over_dr']:.2f} "
                    f"oob==ate_:{rec.get('oob_matches_ate_')} {rec['seconds']}s "
                    f"rss={rec['peak_rss_gib']:.2f}G"
                )
    return {"fits": rows}


async def main() -> int:
    from src.memory.services.factories import loop_scoped_async_supabase_client

    client = get_supabase_client()
    result: Dict[str, Any] = {}
    result["A"] = part_a(client)
    async with loop_scoped_async_supabase_client() as aclient:
        b = await part_b(client, aclient)
    frames = b.pop("_frames")
    result["B"] = b
    result["C"] = part_c(frames) if frames else {"fits": [], "skipped": "no joined frame"}

    # ---- verdict --------------------------------------------------------------------
    say("\n===== VERDICT =====")
    rows_ok = all(
        r.get("meets_cohort_min_rows") for r in result["B"]["brands"].values()
    ) and len(result["B"]["brands"]) == len(BRANDS)
    fits = pd.DataFrame(result["C"]["fits"])
    sig_ok = null_ok = False
    if not fits.empty and "dr_excludes_0" in fits.columns:
        planted = fits[fits["channel"] != "rep_training_score"]
        nulls = fits[fits["channel"] == "rep_training_score"]
        n_sig = int(planted["dr_excludes_0"].sum())
        sig_ok = n_sig == len(planted)
        null_ok = int((~nulls["dr_excludes_0"].fillna(False)).sum()) >= 2
        say(f"  DR significance on planted channels : {n_sig}/{len(planted)} -> {'PASS' if sig_ok else 'FAIL'}")
        say(f"  shipped-interval significance       : {int(planted['shipped_excludes_0'].sum())}/{len(planted)}")
        say(f"  null covers 0 in >= 2/3 brands      : {'PASS' if null_ok else 'FAIL'}")
        if "oob_matches_ate_" in fits.columns:
            pin = fits["oob_matches_ate_"].dropna()
            say(f"  finding 3 nanmean(_oob_preds)==ate_ : {int(pin.sum())}/{len(pin)} fits")
    say(f"  rows >= COHORT_MIN_ROWS in 3/3 brands: {'PASS' if rows_ok else 'FAIL'}")
    result["verdict"] = {
        "rows_premise_holds": bool(rows_ok),
        "dr_significance_premise_holds": bool(sig_ok),
        "null_covers_zero": bool(null_ok),
    }
    say(f"  PREMISE HOLDS: {bool(rows_ok and sig_ok and null_ok)}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe.json").write_text(json.dumps(result, indent=2, default=str))
    if not fits.empty:
        fits.to_csv(OUT / "fits.csv", index=False)
    say(f"\nwrote {OUT}/probe.json")
    return 0


if __name__ == "__main__":
    import asyncio

    raise SystemExit(asyncio.run(main()))
