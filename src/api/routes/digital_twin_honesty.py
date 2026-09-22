"""
Digital Twin honest-surfacing helpers (#2206)
==============================================

Everything the ``/digital-twin`` routes derive from the stored model rows to state
the truth about a twin model: whether its fidelity was ever measured (a NULL is
``unvalidated``, never a pass), what its R² was scored against, whether brand is
a feature, and whether several brand rows are one shared recorded fit (a fit
fingerprint over the recorded config/frame/columns/metrics). Also the guard that
a pre-screen's experiment link names an existing, same-brand experiment.

Split out of ``digital_twin.py`` by the module-size ratchet; the route module
re-exports these names.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import HTTPException

from src.api.dependencies.auth import resolve_brand_for_read
from src.api.schemas.digital_twin import FidelityStatusEnum, R2ScoreBasisEnum

logger = logging.getLogger(__name__)


async def _verify_experiment_link(repo: Any, experiment_id: UUID, brand: str) -> None:
    """The experiment a pre-screen links to must exist and be this brand's (#2206).

    ``twin_simulations.experiment_design_id`` has no FK, so nothing else would
    catch a typo'd or foreign-brand id; the fidelity producer resolves the twin
    simulation through this link and would only ever skip.
    """
    res = await (
        repo.client.table("ml_experiments")
        .select("id,brand")
        .eq("id", str(experiment_id))
        .limit(1)
        .execute()
    )
    rows = res.data or []
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"experiment_design_id {experiment_id} does not name an existing experiment.",
        )
    exp_brand = rows[0].get("brand")
    if exp_brand and str(exp_brand) != str(brand):
        raise HTTPException(
            status_code=422,
            detail=(
                f"experiment_design_id {experiment_id} belongs to brand {exp_brand}, "
                f"not {brand}; a pre-screen links only to its own brand's experiment."
            ),
        )


# ---------------------------------------------------------------------------
# #2206 — honest model surfacing. Every statement below is DERIVED from the rows
# (fit fingerprints, feature columns, provenance, fidelity columns), never hardcoded.
# ---------------------------------------------------------------------------

# Wall-clock is not part of the fit: two runs of one deterministic training
# (same frame, seed, config, features) differ only here.
_FIT_FINGERPRINT_EXCLUDED_METRICS = frozenset({"training_duration_seconds"})


def _canonical(value: Any) -> Any:
    """JSONB round-trips ``1`` and ``1.0`` interchangeably; hash them the same.

    Integral floats become ints (lossless); ints are never widened to float, so
    values above 2**53 keep their identity.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        # Integral floats fold to int; other floats stay NUMERIC (json.dumps emits
        # repr), so 0.2 never collides with the string "0.2" (codex r9 #2).
        return int(value) if value.is_integer() else value
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return value


def _fit_fingerprint(row: Dict[str, Any]) -> str:
    """A content hash of the RECORDED fit: training_config (including the
    ``training_frame`` identity — source/seed/rows — when the trainer recorded it),
    feature/target columns, and every reported metric except wall-clock.

    Equal fingerprints = the same recorded fit. It does not hash the artifact
    itself; rows trained before ``training_frame`` was recorded are compared on
    config + columns + full-precision metrics (prod: three brand rows, one seed-0
    synthetic frame, identical R²/CV/importances to 16 digits).
    """
    pm = dict(row.get("performance_metrics") or {})
    for key in _FIT_FINGERPRINT_EXCLUDED_METRICS:
        pm.pop(key, None)
    payload = _canonical(
        {
            "training_config": row.get("training_config") or {},
            "feature_columns": list(row.get("feature_columns") or []),
            "target_columns": list(row.get("target_columns") or []),
            "performance_metrics": pm,
        }
    )
    canonical = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _model_fidelity_state(
    row: Dict[str, Any],
) -> "tuple[FidelityStatusEnum, Optional[float], int]":
    """(status, fidelity_score, fidelity_sample_count) from the model row's columns.

    Defined ahead of the enum classes below; the annotation is a forward reference.
    """
    from src.digital_twin.models.simulation_models import classify_fidelity

    score = row.get("fidelity_score")
    score_f = None if score is None else float(score)
    status, _warn, _reason = classify_fidelity(score_f)
    return FidelityStatusEnum(status.value), score_f, int(row.get("fidelity_sample_count") or 0)


def _stored_fidelity_fields(model_row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Fidelity fields for a STORED simulation, derived from its model row now (#2206).

    twin_simulations persists only the gate's verdict of its time — and that gate
    let a NULL score pass as ``fidelity_warning=False``. Reading the model row
    through the same rule the engine now uses keeps the stored history honest:
    an unvalidated model warns, a validated one does not.
    """
    from src.digital_twin.models.simulation_models import classify_fidelity

    score = (model_row or {}).get("fidelity_score")
    score_f = None if score is None else float(score)
    status, warning, reason = classify_fidelity(score_f)
    return {
        "fidelity_status": FidelityStatusEnum(status.value),
        "model_fidelity_score": score_f,
        "fidelity_warning": warning,
        "fidelity_warning_reason": reason,
    }


def _r2_score_basis(data_provenance: Optional[str]) -> "R2ScoreBasisEnum":
    if not data_provenance:
        return R2ScoreBasisEnum.UNKNOWN
    prov = str(data_provenance).lower()
    if prov.startswith("synthetic"):
        return R2ScoreBasisEnum.SYNTHETIC_TARGET
    if prov.startswith("rwd"):
        return R2ScoreBasisEnum.RWD_TARGET
    return R2ScoreBasisEnum.UNKNOWN


def _model_honesty_fields(
    row: Dict[str, Any],
    census: List[Dict[str, Any]],
    user: Dict[str, Any],
) -> Dict[str, Any]:
    """The #2206 fields for one model row, given the census of ALL active rows of
    its twin_type. ``shared_fit_model_count`` is the data-derived fact; the other
    brands' names are given only when the caller may read those brands (H11)."""
    tc = row.get("training_config") or {}
    provenance = tc.get("data_provenance", row.get("data_provenance"))
    fingerprint = _fit_fingerprint(row)
    # Distinct BRANDS sharing the fingerprint (codex r5 #2): nothing enforces one
    # active row per brand, and two active versions of one brand are one label.
    same_fit_brands = {
        str(r.get("brand"))
        for r in census
        if r.get("brand")
        and _fit_fingerprint(r) == fingerprint
        and str(r.get("twin_type", "")) == str(row.get("twin_type", ""))
    }
    own_brand = str(row.get("brand")) if row.get("brand") else None
    if own_brand:
        same_fit_brands.add(own_brand)
    others = sorted(b for b in same_fit_brands if b != own_brand)
    visible_others = [b for b in others if resolve_brand_for_read(user, b)[0]]
    status, score, n = _model_fidelity_state(row)
    features = [str(c) for c in (row.get("feature_columns") or [])]
    frame_meta = tc.get("training_frame") or {}
    return {
        # A content digest is the frame's identity; source/seed/path alone are not.
        "training_frame_recorded": bool(frame_meta.get("content_sha256")),
        "fidelity_status": status,
        "fidelity_score": score,
        "fidelity_sample_count": n,
        "data_provenance": provenance,
        "r2_score_basis": _r2_score_basis(provenance),
        "brand_is_feature": "brand" in {f.lower() for f in features},
        "training_fingerprint": fingerprint,
        "shared_fit_model_count": max(1, len(same_fit_brands)),
        "shared_fit_with": visible_others,
    }


# list_active_models defaults to 100 rows; a census that stopped there would
# silently drop fits. Ask for far more than any real registry and say if it is hit.
_CENSUS_LIMIT = 10_000


async def _active_model_census(repo: Any, twin_type_enum: Any) -> List[Dict[str, Any]]:
    """Every active model of the twin_type, across brands — the shared-fit census.

    Brand scoping is applied to the LISTING afterwards; the census must see all
    brands or a single-brand caller could never learn that their fit is shared.
    """
    rows = list(
        await repo.list_active_models(twin_type=twin_type_enum, brand=None, limit=_CENSUS_LIMIT)
        or []
    )
    if len(rows) >= _CENSUS_LIMIT:
        logger.warning(
            "Active twin-model census hit its limit (%d rows); shared-fit counts may omit rows",
            _CENSUS_LIMIT,
        )
    return rows
