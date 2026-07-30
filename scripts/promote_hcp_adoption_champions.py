"""Calibrate + promote the 3 hcp_adoption gold-standard models to champion (#1354).

Owner ruling on #1354 (2026-07-30, APPROVE + CALIBRATE): the three staging
models ``hcp_adoption_{fabhalta,kisqali,remibrutinib}_goldstd_lr_v1`` become
the per-brand champions the ``prediction_synthesizer`` resolver serves —
PROVIDED their held-out calibration is computed from real scored rows at
promotion time and recorded on the registry rows. This script does exactly
that, and nothing else:

1. Loads each model from its registry ``artifact_path`` (the pickled
   ``CalibratedClassifierCV`` the eval pipeline serialized).
2. Reproduces the OOS-union held-out window — ``data_split IN ('test',
   'holdout')`` on ``hcp_brand_adoption`` (lockstep with ``_OOS_EVAL_SPLITS``
   in ``run_persistence_eval`` / ``backfill_goldstd_holdout_metrics``) — and
   scores it with the LOADED artifact after re-fitting only the deterministic
   ``FeatureBuilder`` encoder state on train+validation.
3. FAITHFULNESS GUARD: the recomputed ``auc_roc``/``accuracy`` must reproduce
   the stored ``ml_performance_metrics`` (source='holdout') values within
   ``TOL`` — a divergent artifact is HELD, never promoted on stale numbers.
4. PATHOLOGY GATE (per-brand HOLD):
     * ``calibration_slope`` unfittable or outside [0.5, 2.0] — outside that
       band the model's logits are mis-scaled by more than 2x in either
       direction (the conventional moderate-calibration band, Van Calster et
       al.), so its outputs cannot be trusted as probabilities.
     * ``brier_score >= prevalence * (1 - prevalence)`` — that is the Brier of
       a constant base-rate forecast, so meeting it means the model has no
       probabilistic skill over "predict the prevalence for everyone".
5. WRITE PHASE (``--execute`` only; default is a dry run that prints the exact
   intended updates): a single PK-scoped UPDATE per row setting ``pr_auc``,
   ``brier_score``, ``calibration_slope``, ``stage='production'``,
   ``is_champion=true``, ``promoted_at=now()`` — then a read-back verify.

The calibration-in-the-large intercept (offset-logit MLE) is computed and
REPORTED alongside the slope, but not written: ``ml_model_registry`` has no
calibration-intercept column (checked 2026-07-30), and adding one is out of
this lane's scope (migrations 118/119 are taken).

Why this is a direct registry UPDATE and not ``transition_stage``
-----------------------------------------------------------------
``MLModelRegistryRepository.transition_stage`` (a) refuses
``training_provenance='synthetic_gold'`` -> production (#968) and (b) with
``archive_existing=True`` archives EVERY other production row in the registry
— which would demote the ~100 per-trigger production champions wholesale.
The #968 gate exists to stop a synthetic-gold model silently joining a REAL
serving ensemble; here the owner explicitly ruled promotion for this family
after calibration verification, the three ``hcp_adoption_{brand}`` prediction
targets contain no other model (so there is no ensemble to pollute and no
feature-shape collision — the reason ``register_cohort_model`` hard-refuses
production does not apply), and this script performs the calibration
verification the gate exists to force. The generic gate stays untouched for
every other model. Likewise the ``tr_single_champion`` DB trigger (demotes
same-``experiment_id`` champions) is a verified no-op here: each of the three
rows is the only row of its experiment.

Scope guard: only the three ``MODEL_ALLOWLIST`` names can ever be written;
each write is ``WHERE id = <that row's uuid>``.

Run from a checkout with a loaded ``.env`` (reads are safe anywhere;
``--execute`` is the dispatcher's batch-time step):

  .venv/bin/python scripts/promote_hcp_adoption_champions.py            # dry run
  .venv/bin/python scripts/promote_hcp_adoption_champions.py --brand kisqali
  .venv/bin/python scripts/promote_hcp_adoption_champions.py --execute  # write phase

Idempotent: re-running (with or without ``--execute``) recomputes the same
deterministic metrics and re-issues the same single-row updates; an
already-promoted row keeps its ORIGINAL ``promoted_at`` (the first promotion
instant is never re-stamped), so a rerun's payload is byte-identical — a
semantic no-op.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np

# Recomputed auc/accuracy must reproduce the stored holdout row within this
# (mirrors scripts/backfill_goldstd_holdout_metrics.py).
TOL = 1e-3

# Calibration-slope acceptance band — see module docstring for the rationale.
SLOPE_RANGE = (0.5, 2.0)

# Lockstep with run_persistence_eval._OOS_EVAL_SPLITS / the backfill script:
# the held-out window is everything the model never trained on.
OOS_EVAL_SPLITS = ("test", "holdout")
TRAIN_SPLITS = ("train", "validation")

# Promotion scope — EXACTLY the three models the #1354 ruling covers.
BRANDS = ("fabhalta", "kisqali", "remibrutinib")
MODEL_ALLOWLIST = tuple(f"hcp_adoption_{b}_goldstd_lr_v1" for b in BRANDS)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/unit/test_scripts/, no I/O)
# ---------------------------------------------------------------------------


def positive_class_scores(model: Any, x: Any) -> "np.ndarray":
    """Positive-class probabilities from ``predict_proba``, honoring ``classes_``."""
    proba = np.asarray(model.predict_proba(x), dtype=float)
    if proba.ndim != 2 or proba.shape[1] == 1:
        # Degenerate single-class model: the constant class value is the score
        # (mirrors backfill_goldstd_holdout_metrics._window_scores).
        classes = getattr(model, "classes_", [0])
        return np.full(proba.shape[0], float(classes[0]))
    classes = list(model.classes_)
    pos = classes.index(1) if 1 in classes else 0
    return proba[:, pos]


def calibration_intercept(
    y_true: "np.ndarray",
    y_score: "np.ndarray",
    *,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> float | None:
    """Calibration-in-the-large: intercept-only logistic MLE with offset logit(p).

    The standard companion to the Cox calibration slope (Van Calster framework):
    fit ``logit(P(y=1)) = a + logit(y_score)`` with the slope FIXED at 1; ``a``
    near 0 means the score level matches the outcome rate, ``a > 0`` means the
    model under-predicts. Solved by Newton iteration on the 1-D score equation
    (deterministic; no solver dependency). Returns ``None`` for single-class
    labels or any numerical failure — never a fabricated value.
    """
    y = np.asarray(y_true, dtype=float)
    if np.unique(y).size < 2:
        return None
    eps = 1e-6
    p = np.clip(np.asarray(y_score, dtype=float), eps, 1.0 - eps)
    offset = np.log(p / (1.0 - p))
    a = 0.0
    converged = False
    try:
        for _ in range(max_iter):
            mu = 1.0 / (1.0 + np.exp(-(a + offset)))
            hess = float(np.sum(mu * (1.0 - mu)))
            if not np.isfinite(hess) or hess <= 0.0:
                return None
            step = float(np.sum(y - mu)) / hess
            a += step
            if abs(step) < tol:
                converged = True
                break
        # A non-converged final iterate is NOT reported: better to omit the
        # intercept than to print a number the solver did not actually reach.
        return float(a) if (converged and np.isfinite(a)) else None
    except (FloatingPointError, OverflowError, ValueError):
        return None


def pathology_gate(metrics: dict, prevalence: float) -> tuple[bool, list[str]]:
    """Apply the #1354 calibration pathology gate; returns (ok, hold_reasons)."""
    reasons: list[str] = []
    slope = metrics.get("calibration_slope")
    if slope is None:
        reasons.append("calibration_slope unfittable on the held-out window")
    elif not (SLOPE_RANGE[0] <= float(slope) <= SLOPE_RANGE[1]):
        reasons.append(
            f"calibration_slope {float(slope):.4f} outside "
            f"[{SLOPE_RANGE[0]}, {SLOPE_RANGE[1]}] (logits mis-scaled >2x)"
        )
    brier = metrics.get("brier_score")
    baseline = float(prevalence) * (1.0 - float(prevalence))
    if brier is None:
        reasons.append("brier_score missing")
    elif float(brier) >= baseline:
        reasons.append(
            f"brier_score {float(brier):.4f} >= prevalence baseline {baseline:.4f} "
            "(no skill over a constant base-rate forecast)"
        )
    return (not reasons), reasons


def faithfulness_check(
    recomputed: dict | None, stored: dict | None, tol: float = TOL
) -> tuple[bool, float, float]:
    """Does the loaded artifact reproduce the stored holdout auc/accuracy?"""
    if not recomputed or not stored:
        return False, 9.0, 9.0
    d_auc = abs(float(recomputed.get("auc_roc", -9.0)) - float(stored.get("auc_roc", 9.0)))
    d_acc = abs(float(recomputed.get("accuracy", -9.0)) - float(stored.get("accuracy", 9.0)))
    return (d_auc <= tol and d_acc <= tol), d_auc, d_acc


def build_registry_update(metrics: dict, promoted_at_iso: str) -> dict:
    """The EXACT single-row write payload (registry columns are numeric(5,4))."""
    return {
        "pr_auc": round(float(metrics["pr_auc"]), 4),
        "brier_score": round(float(metrics["brier_score"]), 4),
        "calibration_slope": round(float(metrics["calibration_slope"]), 4),
        "stage": "production",
        "is_champion": True,
        "promoted_at": promoted_at_iso,
    }


def decide(
    stored: dict | None,
    computed: dict,
    prevalence: float,
    promoted_at_iso: str,
    tol: float = TOL,
) -> tuple[str, list[str], dict | None]:
    """(action, reasons, update): 'promote' with payload, or 'hold' with why."""
    ok, d_auc, d_acc = faithfulness_check(computed, stored, tol)
    if not ok:
        return (
            "hold",
            [
                f"unfaithful: auc_d={d_auc:.4f} acc_d={d_acc:.4f} — the loaded artifact "
                "does not reproduce the stored holdout headline (refusing to promote)"
            ],
            None,
        )
    gate_ok, reasons = pathology_gate(computed, prevalence)
    if not gate_ok:
        return "hold", reasons, None
    return "promote", [], build_registry_update(computed, promoted_at_iso)


# ---------------------------------------------------------------------------
# I/O plumbing (async supabase client; conventions mirror the backfill script)
# ---------------------------------------------------------------------------


async def _apply_update(client: Any, row_id: str, update: dict) -> None:
    """Single-row, PK-scoped registry UPDATE (the only write this script makes)."""
    await client.table("ml_model_registry").update(update).eq("id", row_id).execute()


async def _fetch_registry_row(client: Any, model_name: str) -> dict | None:
    res = await (
        client.table("ml_model_registry")
        .select("id,model_name,auc,stage,is_champion,artifact_path,experiment_id,promoted_at")
        .eq("model_name", model_name)
        .execute()
    )
    rows = res.data or []
    if len(rows) > 1:
        # (model_name, model_version) is unique but model_name alone is not —
        # a hypothetical v2.0 would make "the row to promote" ambiguous. Refuse
        # loudly rather than promote an arbitrary one.
        raise RuntimeError(
            f"{len(rows)} registry rows for {model_name!r}; refusing ambiguous promotion"
        )
    return rows[0] if rows else None


async def _stored_holdout(client: Any, model_id: str) -> dict | None:
    """Latest stored holdout metric per name, keyed by the registry row's OWN id.

    Uses the already-fetched registry ``id`` directly (no name re-resolution)
    and orders ``measured_at`` DESC (``id`` DESC as a stable same-timestamp
    tie-breaker) keeping the FIRST occurrence per metric name, so the
    faithfulness reference is deterministically the newest snapshot even if
    historical duplicates ever exist.
    """
    res = await (
        client.table("ml_performance_metrics")
        .select("metric_name,metric_value,measured_at")
        .eq("model_id", model_id)
        .eq("source", "holdout")
        .order("measured_at", desc=True)
        .order("id", desc=True)
        .execute()
    )
    out: dict[str, float] = {}
    for r in res.data or []:
        name = r["metric_name"]
        if name not in out:
            out[name] = float(r["metric_value"])
    return out or None


async def _score_artifact(client: Any, brand: str, artifact_path: str) -> dict:
    """Load the pickle, reproduce the OOS-union window, score it.

    Returns ``{"metrics", "intercept", "n", "prevalence"}``; raises on any
    structural failure (missing artifact, empty frames, feature mismatch) so
    the caller reports it as FAIL with the real reason.
    """
    import pickle

    from src.mlops.gold_standard_eval.cohort_spec import make_hcp_spec
    from src.mlops.gold_standard_eval.feature_builder import FeatureBuilder
    from src.mlops.gold_standard_eval.scorer import score

    with open(artifact_path, "rb") as fh:
        model = pickle.load(fh)  # noqa: S301 — our own registry artifact

    spec = make_hcp_spec(brand.capitalize())
    fb = FeatureBuilder(spec)
    frame = await fb.load_frame(client, splits=None)
    if frame.empty:
        raise RuntimeError("empty cohort frame from hcp_brand_adoption")
    train = frame.loc[frame["data_split"].isin(TRAIN_SPLITS)]
    if train.empty:
        raise RuntimeError("no train/validation rows to fit the encoder state")
    fb.build_from_frame(train)  # deterministic encoder state (medians/columns)
    oos = frame.loc[frame["data_split"].isin(OOS_EVAL_SPLITS)]
    if oos.empty:
        raise RuntimeError("no test/holdout rows — held-out window is empty")

    x = fb.transform(oos)
    model_cols = list(getattr(model, "feature_names_in_", []))
    if model_cols and model_cols != list(x.columns):
        if set(model_cols) == set(x.columns):
            x = x[model_cols]  # same features, different order — realign
        else:
            missing = set(model_cols) ^ set(x.columns)
            raise RuntimeError(f"artifact/encoder feature mismatch: {sorted(missing)[:6]}")

    y = oos[spec.label_column].astype(int).to_numpy()
    s = positive_class_scores(model, x)
    return {
        "metrics": score(y, s),
        "intercept": calibration_intercept(y, s),
        "n": int(len(y)),
        "prevalence": float(np.mean(y)),
    }


async def _verify_written(client: Any, row_id: str, update: dict) -> bool:
    """Read-back verify: the row now carries exactly the promoted values."""
    res = await (
        client.table("ml_model_registry")
        .select("stage,is_champion,pr_auc,brier_score,calibration_slope")
        .eq("id", row_id)
        .execute()
    )
    rows = res.data or []
    if not rows:
        return False
    row = rows[0]
    return (
        row.get("stage") == "production"
        and bool(row.get("is_champion"))
        and all(
            row.get(k) is not None and abs(float(row[k]) - update[k]) < 1e-6
            for k in ("pr_auc", "brier_score", "calibration_slope")
        )
    )


async def run(client: Any, *, execute: bool, only_brand: str | None) -> int:
    """Score + gate + (optionally) promote each brand; returns the exit code.

    Split from ``main`` so the whole orchestration is unit-testable against a
    fake client (dry-run-never-writes / single-PK-write / rerun-no-op).
    """
    brands = [b for b in BRANDS if only_brand is None or b == only_brand]
    mode = "EXECUTE" if execute else "DRY-RUN"
    print(f"hcp_adoption champion promotion (#1354): {len(brands)} brand(s); mode={mode}")

    promoted = held = failed = 0
    now_iso = datetime.now(timezone.utc).isoformat()
    for brand in brands:
        model_name = f"hcp_adoption_{brand}_goldstd_lr_v1"
        assert model_name in MODEL_ALLOWLIST  # scope guard — never anything else
        try:
            row = await _fetch_registry_row(client, model_name)
            if row is None:
                print(f"  FAIL {model_name}: no registry row")
                failed += 1
                continue
            stored = await _stored_holdout(client, row["id"])
            if not stored:
                print(f"  HOLD {model_name}: no stored holdout metrics to verify against")
                held += 1
                continue
            res = await _score_artifact(client, brand, row["artifact_path"])
            m, n, prev = res["metrics"], res["n"], res["prevalence"]
            icpt = res["intercept"]
            already = row.get("stage") == "production" and bool(row.get("is_champion"))
            # Idempotency: promoted_at records the FIRST promotion instant —
            # an already-promoted row keeps its original timestamp, so a rerun
            # re-issues a byte-identical payload (semantic no-op).
            prior_ts = row.get("promoted_at")
            promoted_at: str = str(prior_ts) if (already and prior_ts) else now_iso
            action, reasons, update = decide(stored, m, prev, promoted_at)
            # Null-safe rendering: an unfittable slope may be an ABSENT key
            # (scorer contract) or an explicit None — neither may crash the
            # evidence line, or a HOLD would surface as FAIL.
            slope = m.get("calibration_slope")
            slope_txt = "None" if slope is None else f"{float(slope):.6f}"
            evidence = (
                f"n={n} prevalence={prev:.4f} auc={m['auc_roc']:.6f} "
                f"pr_auc={m['pr_auc']:.6f} brier={m['brier_score']:.6f} "
                f"slope={slope_txt} "
                f"citl_intercept={icpt if icpt is None else round(icpt, 6)} (report-only)"
            )
            if action == "hold":
                print(f"  HOLD {model_name}: {'; '.join(reasons)}")
                print(f"       {evidence}")
                held += 1
                continue
            # decide() always returns a payload for "promote"; narrow for mypy
            # and fail loudly if that contract is ever broken.
            assert update is not None
            note = " [already promoted — idempotent rewrite]" if already else ""
            if not execute:
                print(f"  OK(dry) {model_name}:{note} would UPDATE ml_model_registry")
                print(f"       SET {update} WHERE id='{row['id']}'")
                print(f"       {evidence}")
                promoted += 1
                continue
            await _apply_update(client, row["id"], update)
            if not await _verify_written(client, row["id"], update):
                print(f"  FAIL {model_name}: write did not read back as promoted")
                failed += 1
                continue
            print(f"  DONE {model_name}:{note} promoted + metrics recorded")
            print(f"       {evidence}")
            promoted += 1
        except Exception as e:  # noqa: BLE001 — one brand must not abort the rest
            print(f"  FAIL {model_name}: {type(e).__name__}: {str(e)[:200]}")
            failed += 1
    print(f"summary: promoted={promoted} held={held} failed={failed} mode={mode}")
    return 0 if failed == 0 else 1


async def main(execute: bool, only_brand: str | None) -> int:
    from src.repositories.drift_monitoring import get_drift_monitoring_client

    client = await get_drift_monitoring_client()
    return await run(client, execute=execute, only_brand=only_brand)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--execute",
        action="store_true",
        help="perform the registry writes (default: dry-run printing the exact updates)",
    )
    ap.add_argument("--brand", default=None, choices=BRANDS, help="restrict to one brand")
    args = ap.parse_args()
    sys.exit(asyncio.run(main(args.execute, args.brand)))
