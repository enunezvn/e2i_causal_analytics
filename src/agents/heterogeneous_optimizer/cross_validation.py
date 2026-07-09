"""Cross-library validation for the Heterogeneous Optimizer agent.

Compares the two independent estimators the graph already runs on every
/segment-analysis request:

* EconML ``CausalForestDML`` — per-segment CATE point estimates (cate_estimator)
* CausalML uplift model — per-segment mean uplift scores (uplift_analyzer)

Both estimate the same quantity (treatment-vs-control effect per segment) with
different algorithms, so genuine heterogeneity should reproduce across them in
effect DIRECTION and segment RANKING. Magnitudes are deliberately NOT compared:
a causal forest's doubly-robust CATE and an uplift forest's score live on
different estimator scales, so rank correlation + sign agreement are the
honest, scale-free invariants.

History: the B7.4 state channels (``library_agreement_score``,
``validation_passed``, ``cross_library_validation``,
``econml_causalml_agreement``) were scaffolded in the B4–B10 causal expansion
but never computed — the /segment-analysis "Library Validation" card rendered
the resulting ``null`` as a fabricated "0% / Failed". This module computes
them for real.
"""

import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

# Minimum paired segments for a meaningful rank comparison. Below this a
# Spearman correlation is essentially unconstrained, so we honestly report
# "not computed" rather than a noise verdict.
MIN_SEGMENTS_FOR_VALIDATION = 3

# Agreement score at/above which cross-library validation passes. The score is
# 0.5*sign_agreement + 0.5*max(0, spearman_rho): identical rankings with
# consistent directions score 1.0; consistent directions but uncorrelated
# rankings score ~0.5 (fails — the libraries don't reproduce the ordering the
# targeting recommendation depends on).
AGREEMENT_THRESHOLD = 0.7


def _finite(value: Any) -> Optional[float]:
    """Return ``value`` as a finite float, else None."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def compute_cross_library_validation(
    cate_by_segment: Optional[Mapping[str, Sequence[Mapping[str, Any]]]],
    uplift_by_segment: Optional[Mapping[str, Sequence[Mapping[str, Any]]]],
    uplift_model_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute EconML↔CausalML agreement over segments both libraries scored.

    Segments are paired on ``(dimension, segment_value)``. Returns a partial
    state update. When computable:

    * ``library_agreement_score`` / ``econml_causalml_agreement`` — 0..1
    * ``validation_passed`` — score >= AGREEMENT_THRESHOLD
    * ``cross_library_validation`` — method + components (rho, sign agreement,
      n compared, threshold, uplift model)

    When NOT computable (uplift missing, <MIN_SEGMENTS pairs, non-finite
    values), returns ONLY a ``cross_library_validation`` dict with
    ``computed: False`` and a reason — never a fabricated verdict. Pure
    computation; raises nothing on well-formed dict inputs.
    """
    pairs: List[tuple] = []
    for dim, cate_results in (cate_by_segment or {}).items():
        uplift_lookup = {
            str(r.get("segment_value")): _finite(r.get("mean_uplift_score"))
            for r in (uplift_by_segment or {}).get(dim, [])
        }
        for c in cate_results or []:
            cate_val = _finite(c.get("cate_estimate"))
            uplift_val = uplift_lookup.get(str(c.get("segment_value")))
            if cate_val is not None and uplift_val is not None:
                pairs.append((cate_val, uplift_val))

    if not uplift_by_segment:
        return {
            "cross_library_validation": {
                "computed": False,
                "reason": "no uplift results to compare against (uplift analysis absent)",
            }
        }
    if len(pairs) < MIN_SEGMENTS_FOR_VALIDATION:
        return {
            "cross_library_validation": {
                "computed": False,
                "reason": (
                    f"only {len(pairs)} segment(s) scored by both libraries "
                    f"(need >= {MIN_SEGMENTS_FOR_VALIDATION} for a meaningful comparison)"
                ),
            }
        }

    import warnings as _warnings

    import numpy as np
    from scipy import stats

    cate_arr = np.array([p[0] for p in pairs])
    uplift_arr = np.array([p[1] for p in pairs])

    sign_agreement = float(np.mean(np.sign(cate_arr) == np.sign(uplift_arr)))
    with _warnings.catch_warnings():
        # Constant input makes rho nan — handled explicitly below; the scipy
        # warning would only add log noise for an anticipated case.
        _warnings.simplefilter("ignore", stats.ConstantInputWarning)
        rho = float(stats.spearmanr(cate_arr, uplift_arr).statistic)

    if math.isnan(rho):
        # Constant scores in one library -> rank correlation undefined. Fall
        # back to direction agreement alone and say so.
        score = sign_agreement
        rho_out: Optional[float] = None
        method = "sign_agreement only (rank correlation undefined: constant scores)"
    else:
        score = 0.5 * sign_agreement + 0.5 * max(0.0, rho)
        rho_out = rho
        method = "0.5*sign_agreement + 0.5*max(0, spearman_rho)"

    score = float(min(max(score, 0.0), 1.0))
    passed = score >= AGREEMENT_THRESHOLD

    logger.info(
        "Cross-library validation: agreement=%.3f (rho=%s, sign=%.2f, n=%d) -> %s",
        score,
        f"{rho_out:.3f}" if rho_out is not None else "n/a",
        sign_agreement,
        len(pairs),
        "PASSED" if passed else "FAILED",
    )

    return {
        "cross_library_validation": {
            "computed": True,
            "method": method,
            "n_segments_compared": len(pairs),
            "spearman_rho": rho_out,
            "sign_agreement": sign_agreement,
            "threshold": AGREEMENT_THRESHOLD,
            "uplift_model": uplift_model_type,
        },
        "econml_causalml_agreement": score,
        "library_agreement_score": score,
        "validation_passed": passed,
    }


def serialize_validation_for_llm(state: Dict[str, Any]) -> str:
    """One-line summary of the cross-library validation for the narrative LM.

    Feeding the verdict into ``CATEInterpretationSignature`` keeps the
    Strategic Interpretation and the Library Validation card from silently
    contradicting each other (the original /segment-analysis review finding).
    """
    detail = state.get("cross_library_validation") or {}
    score = state.get("library_agreement_score")
    if score is None or not detail.get("computed"):
        reason = detail.get("reason")
        return f"not computed ({reason})" if reason else "not computed"

    passed = state.get("validation_passed")
    n = detail.get("n_segments_compared")
    rho = detail.get("spearman_rho")
    parts = [
        f"{'PASSED' if passed else 'FAILED'}",
        f"agreement {float(score):.0%} between the two independent estimation "
        f"methods (EconML causal forest vs CausalML uplift model) "
        f"across {n} segments",
    ]
    if rho is not None:
        parts.append(f"rank correlation {float(rho):.2f}")
    sign = detail.get("sign_agreement")
    if sign is not None:
        parts.append(f"direction agreement {float(sign):.0%}")
    return " — ".join([parts[0], ", ".join(parts[1:])])
