"""Sync the gold-standard SERVING layer after a patient/HCP cohort reseed.

THE RECURRENCE BUG THIS CLOSES
------------------------------
A reseed (``regenerate_cohort_outcomes.py`` / ``run_patient_cohorts`` /
``run_hcp_cohorts``) regenerates the cohort rows and re-registers the
``*_goldstd_lr_v1`` models, but it leaves THREE serving artifacts stale and
DECOUPLED from the registry, so the live Feature-Importance page silently keeps
showing the PRE-reseed covariate set:

1. SHAP serving bundles — ``data/ml_artifacts/shap_serving/<cohort>/<name>.bundle.pkl``
   (the BentoML service explains over THESE, not the registry).  Re-fit from the
   enriched live rows; the bentoml container must then be restarted to reload.
2. Feast online store — the materializer runs ``materialize-incremental`` keyed
   on ``event_timestamp``; reseeded rows carry HISTORICAL ``event_date``s outside
   the incremental window, so they are never re-materialized.  A FULL materialize
   (owned by the ``e2i_feast`` sidecar) is required.  AND (#1296) even a FULL
   materialize silently NO-OPS on a SAME-DAY re-reseed: Feast's Redis store skips
   any write whose ``event_timestamp`` is ``<=`` the stored ``_ts:<view>`` dedup
   marker, and both goldstd views derive ``event_timestamp`` from a day-granular
   ``event_date`` cast to midnight — so a second reseed on the same calendar day
   ties the marker and is DROPPED (materialize still exits 0, serving stays
   stale).  The ``_ts:<view>`` markers must therefore be CLEARED before the FULL
   materialize (``feature_repo/clear_goldstd_ts_markers.py``, run in the sidecar).
3. SHAP ``global_importance`` cache — ``ml_shap_analyses`` rows are served
   verbatim until ``refresh=true`` recomputes them.

Run it AFTER a reseed+deploy. The steps are STRICTLY ORDERED — the cache refresh
MUST come last (after the Feast materialize + bentoml restart), or it durably
caches stale serving output. So this script does NOT auto-refresh; it
re-materializes the bundles and prints the remaining operator steps, and the
``--refresh-only`` terminal phase does the cache refresh once the rest is done.

Usage (in order)::

    # 1) re-materialize the 12 bundles + print the next steps
    python -m scripts.sync_goldstd_serving

    # 2) Clear the Feast Redis _ts:<view> dedup markers FIRST (#1296), so a
    #    same-day re-reseed actually propagates. Feast drops writes whose
    #    event_timestamp is <= the stored marker and event_date is day-granular,
    #    so without this the FULL materialize below silently no-ops on a second
    #    reseed on the same calendar day. Dry-run first, then clear (sidecar):
    docker exec e2i_feast python /feast-src/clear_goldstd_ts_markers.py --dry-run
    docker exec e2i_feast python /feast-src/clear_goldstd_ts_markers.py

    # 3) Feast FULL materialize (e2i_feast sidecar — NOT incremental).
    #    NOTE: --views is a REPEATABLE flag, one view per flag — a
    #    space-separated list dies with "Got unexpected extra argument"
    #    AND still exits 0 through a pipe (live-caught 2026-07-04):
    docker exec e2i_feast feast --chdir /feast materialize \\
        2020-01-01T00:00:00 "$(date -u +%Y-%m-%dT%H:%M:%S)" \\
        --views goldstd_cohort_features --views goldstd_hcp_cohort_features

    # 4) reload the re-materialized bundles. The container is e2i_bentoml
    #    (plain compose) or e2i_bentoml_dev (dev overlay) — resolve with
    #    `docker ps --format '{{.Names}}' | grep bentoml` first:
    docker restart e2i_bentoml

    # 5) ONLY NOW repopulate the SHAP global-importance cache:
    python -m scripts.sync_goldstd_serving --refresh-only

The app image cannot ``import feast`` (#307), so the e2i_feast sidecar owns the
Feast materialize — hence step 2 is an operator command, not an in-process call.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import urllib.request
from typing import List, Tuple
from urllib.error import HTTPError

logger = logging.getLogger(__name__)

# The 4-cohort × 3-brand serving matrix the live Feature-Importance + Model
# Performance pages render.  Mirrors cohort_spec.PATIENT_COHORTS + the HCP cohort.
_PATIENT_COHORTS = ("initiation", "persistence", "discontinuation")
_COHORTS = (*_PATIENT_COHORTS, "hcp_adoption")
_BRANDS = ("Remibrutinib", "Fabhalta", "Kisqali")

# The 4 prognostic drivers the T9/T11 enrichment added to _BASE7. Their presence
# in the SERVED schema (keep_columns) proves the bentoml process reloaded the
# re-materialized 7-cov bundles; their presence in a refresh RESULT proves Feast
# actually served 7 features (a bare HTTP 200 can be a recompute over stale 3-cov
# serving — codex round-2). These are the machine-checkable freshness signals.
_NEW_PATIENT_DRIVERS = frozenset(
    {"insurance_type", "age_at_diagnosis", "comorbidity_burden", "prior_therapy_lines"}
)


async def _rematerialize_all_bundles() -> int:
    """Re-fit + write all 12 serving bundles from the enriched live cohort rows."""
    from scripts.rematerialize_goldstd_bundles import SPEC_REGISTRY, _amain

    return await _amain(sorted(SPEC_REGISTRY.keys()))


def _admin_token(api_base: str) -> str:
    """Mint an admin JWT from Supabase auth (mirrors the live verify scripts)."""
    su = os.environ["SUPABASE_URL"]
    anon = os.environ["SUPABASE_ANON_KEY"]
    pw = os.environ["E2I_ADMIN_PASSWORD"]
    body = json.dumps({"email": "admin@e2i.local", "password": pw}).encode()
    req = urllib.request.Request(
        f"{su}/auth/v1/token?grant_type=password",
        data=body,
        headers={"apikey": anon, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())["access_token"]


def _get_json(url: str, token: str, timeout: int = 180) -> Tuple[int, dict]:
    """GET a JSON endpoint; return ``(status, body)`` (body ``{}`` on error)."""
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode())
        except Exception:  # noqa: BLE001
            return exc.code, {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("GET %s failed: %s", url, exc)
        return 0, {}


def _serving_ready(api_base: str, token: str) -> bool:
    """Pre-flight guard: is the bentoml SERVING layer on the enriched bundles yet?

    Reads ``/explain/models`` and checks the patient cohorts' ``keep_columns``
    (sourced from the live bentoml model_info) carry the 4 new drivers. If they
    still show the base 3, the bundle re-materialize + bentoml container
    restart have NOT taken effect — refreshing now would recompute the
    cache over stale serving and DURABLY persist it (codex round-2 HIGH). Abort.
    """
    status, body = _get_json(f"{api_base}/explain/models", token, timeout=60)
    if status != 200:
        logger.error("serving-ready pre-check: /explain/models -> %s", status)
        return False
    by_type = {
        m.get("model_type"): (m.get("keep_columns") or []) for m in body.get("supported_models", [])
    }
    for cohort in _PATIENT_COHORTS:
        served = set(by_type.get(cohort, []))
        if not _NEW_PATIENT_DRIVERS.issubset(served):
            logger.error(
                "serving-ready pre-check FAILED: %s keep_columns=%s lacks the new "
                "drivers %s — run the bundle re-materialize + Feast FULL materialize "
                "+ the bentoml restart (`docker restart e2i_bentoml`; _dev suffix "
                "under the dev overlay) FIRST.",
                cohort,
                sorted(served),
                sorted(_NEW_PATIENT_DRIVERS - served),
            )
            return False
    return True


def _raw_covariates(features: List[dict]) -> set:
    """Collapse encoded SHAP feature names back to their raw covariate parents."""
    raw = set()
    for f in features:
        name = str(f.get("feature_name", "")).split("__")[0]
        for prefix in ("geographic_region", "specialty", "insurance_type"):
            if name.startswith(prefix):
                name = prefix
        raw.add(name)
    return raw


def _refresh_global_cache(api_base: str, token: str) -> List[Tuple[str, str, int, bool]]:
    """Force a fresh recompute of every cohort×brand global-importance row.

    Returns ``[(cohort, brand, http_status, enriched), ...]``. ``enriched`` is the
    SEMANTIC freshness signal: for a patient slot it is True only when the 200's
    feature payload actually carries the new drivers (a bare 200 can be a recompute
    over stale 3-cov serving — codex round-2). HCP slots are enriched-by-definition.
    """
    results: List[Tuple[str, str, int, bool]] = []
    for cohort in _COHORTS:
        for brand in _BRANDS:
            url = (
                f"{api_base}/explain/global?model_type={cohort}"
                f"&brand={brand}&sample_size=20&refresh=true"
            )
            status, body = _get_json(url, token)
            enriched = True
            if status == 200 and cohort in _PATIENT_COHORTS:
                raw = _raw_covariates(body.get("features", []))
                # ALL 4 drivers must be present — same subset contract as the
                # _serving_ready() pre-flight (a partially-enriched payload is
                # NOT acceptable: it would mean Feast served some drivers null).
                enriched = _NEW_PATIENT_DRIVERS.issubset(raw)
            elif status != 200:
                enriched = False
            results.append((cohort, brand, status, enriched))
            logger.info(
                "refresh %s/%s -> %s%s",
                cohort,
                brand,
                status,
                "" if enriched else "  [STALE: 200 but base-3 only]" if status == 200 else "",
            )
    return results


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # Load .env ONCE up front so EVERY path (incl. --refresh-only) has the
    # Supabase/admin env the bundle step and the cache refresh both need.
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-base",
        default=os.environ.get("E2I_API_BASE", "https://eznomics.site/api"),
        help="Live API base for the global-importance cache refresh.",
    )
    parser.add_argument(
        "--skip-bundles", action="store_true", help="Do NOT re-materialize bundles."
    )
    parser.add_argument(
        "--refresh-only",
        action="store_true",
        help=(
            "Skip bundles; ONLY refresh the live SHAP cache. Run this LAST — AFTER "
            "the Feast FULL materialize AND the bentoml restart have completed, "
            "else the recompute caches stale serving output durably."
        ),
    )
    args = parser.parse_args()

    # --- Refresh-only terminal phase (operator runs this AFTER Feast+restart) ---
    if args.refresh_only:
        token = _admin_token(args.api_base)
        # GUARD (codex round-2): do NOT recompute the cache over stale serving.
        # Verify the bentoml layer is actually on the enriched bundles first; a
        # doc-ordering is not enough for an op that durably persists.
        if not _serving_ready(args.api_base, token):
            logger.error(
                "ABORT: serving layer is not on the enriched 7-cov bundles. Run "
                "the bundle re-materialize + Feast FULL materialize + the bentoml "
                "restart (`docker restart e2i_bentoml`; _dev suffix under the dev "
                "overlay), THEN re-run --refresh-only."
            )
            return 2
        logger.info("Serving layer verified enriched. Refreshing the cache (12 slots)...")
        results = _refresh_global_cache(args.api_base, token)
        ok = sum(1 for _, _, s, _ in results if s == 200)
        fresh = sum(1 for _, _, _, e in results if e)
        logger.info(
            "Cache refresh: %d/%d -> 200, %d/%d enriched", ok, len(results), fresh, len(results)
        )
        # FAIL on either a non-200 OR a 200 that came back base-3 only (stale Feast).
        bad = [(c, b, s, e) for c, b, s, e in results if s != 200 or not e]
        if bad:
            logger.warning(
                "Slots not freshly enriched — confirm the Feast FULL materialize "
                "ran (online store serves the 4 drivers), then re-run: %s",
                bad,
            )
            return 1
        return 0

    # --- Default phase: re-materialize bundles, then PRINT the operator steps ---
    # The cache refresh is intentionally NOT auto-run here: it must follow the
    # Feast materialize + bentoml restart, or it durably caches stale output
    # (codex HIGH). The operator runs `--refresh-only` as the terminal step.
    rc = 0
    if not args.skip_bundles:
        logger.info("Re-materializing 12 serving bundles from live rows...")
        rc = asyncio.run(_rematerialize_all_bundles())
        if rc != 0:
            logger.error(
                "Bundle re-materialization FAILED (rc=%d); NOT printing the "
                "activation steps. Fix the bundles before touching serving.",
                rc,
            )
            return rc

    logger.info(
        "Bundles done. Now run, IN ORDER, then the terminal cache refresh:\n"
        "  1) Clear the Feast _ts:<view> dedup markers FIRST so a same-day\n"
        "     re-reseed propagates (#1296 — Feast drops writes whose event_ts is\n"
        "     <= the stored marker, and event_date is day-granular). Dry-run,\n"
        "     then clear, in the e2i_feast sidecar:\n"
        "       docker exec e2i_feast python /feast-src/clear_goldstd_ts_markers.py --dry-run\n"
        "       docker exec e2i_feast python /feast-src/clear_goldstd_ts_markers.py\n"
        "  2) Feast FULL materialize (e2i_feast sidecar — NOT incremental;\n"
        "     --views is REPEATABLE, one view per flag):\n"
        "       docker exec e2i_feast feast --chdir /feast materialize "
        '2020-01-01T00:00:00 "$(date -u +%%Y-%%m-%%dT%%H:%%M:%%S)" '
        "--views goldstd_cohort_features --views goldstd_hcp_cohort_features\n"
        "  3) Reload the re-materialized bundles (container is e2i_bentoml, or\n"
        "     e2i_bentoml_dev under the dev overlay — check docker ps):\n"
        "       docker restart e2i_bentoml\n"
        "  4) Repopulate the SHAP cache (ONLY after 1+2+3):\n"
        "       python -m scripts.sync_goldstd_serving --refresh-only"
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
