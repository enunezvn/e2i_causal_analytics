"""Per-HCP CATE -> resource_optimizer allocation_targets builder (Shard 08 T4).

Maps each HCP's Shard-03 heterogeneous treatment effect (cate_estimate) to a
NON-NEGATIVE response coefficient that VARIES across HCPs, attaches a positive
current_allocation, and sets a BINDING budget = current total so the solver must
REALLOCATE toward high-CATE HCPs rather than expand spend.

problem_formulator (resource_optimizer) rejects negative expected_response and
uses it as the maximize-outcome objective coefficient, so a higher coefficient
on a high-CATE HCP makes the optimal solution shift budget there.

STALE-PLAN DEVIATION (documented):
The plan's _load_cate_spend_frame assumed a ``hcp_adoption_<brand>.parquet`` with
a ``current_spend`` column joined onto the per-HCP CATE artifact. Verified against
the upstream producers — there is NO current_spend anywhere in the synthetic
pipeline: ``hcp_adoption_artifact.write_per_hcp_cate_artifact`` emits EXACTLY
[hcp_id, cate_estimate, is_synthetic], and ``load_synthetic_data.write_cohort_frames``
copies that artifact verbatim into ``cohort_frames/hcp_adoption__<brand>.parquet``.
So this builder derives a deterministic, is_synthetic-stamped EQUAL baseline
``current_allocation`` when the frame carries no ``current_spend`` (an equal start
is the strongest reallocation test — gate 8 / T6 use an equal start), and honours a
``current_spend`` column when a caller supplies one (the plan's unit fixture does).
Nothing is fabricated as real; every target carries is_synthetic=True.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Minimum positive coefficient so the lowest-CATE HCP is still > 0 (problem_formulator
# treats expected_response<0 as an error and 0 as "no response" — keep a small floor).
_RESPONSE_FLOOR = 0.10

# Synthetic equal-baseline current_allocation per HCP, used ONLY when the per-HCP CATE
# frame carries no current_spend column (the realized Shard-06 artifact never does).
# Equal start => the solver's reallocation is driven purely by expected_response (CATE).
_SYNTHETIC_BASELINE_SPEND = 10_000.0

# Per-HCP CATE artifact (per_hcp_cate_hcp_adoption_<brand>.parquet) written by Shard 06;
# the underlying per-unit tau originates from the Shard-03 DGP.
_CATE_DIR = Path("data/synthetic")


def targets_from_cate_frame(frame: pd.DataFrame) -> Tuple[List[Dict[str, Any]], float]:
    """Build (allocation_targets, budget) from a per-HCP CATE frame.

    Required column: ``hcp_id``, ``cate_estimate``. Optional: ``current_spend``
    (a positive per-HCP allocation; synthesized to an equal baseline when absent)
    and ``is_synthetic``.

    expected_response = RESPONSE_FLOOR + (cate - min_cate), so it is non-negative,
    strictly varies with CATE, and preserves CATE ordering. Budget = total current
    allocation (binding).
    """
    if frame is None or frame.empty:
        return [], 0.0
    f = frame.copy()
    min_cate = float(f["cate_estimate"].min())
    f["expected_response"] = _RESPONSE_FLOOR + (f["cate_estimate"].astype(float) - min_cate)

    has_spend = "current_spend" in f.columns
    targets: List[Dict[str, Any]] = []
    for _, row in f.iterrows():
        spend = float(row["current_spend"]) if has_spend else _SYNTHETIC_BASELINE_SPEND
        if spend <= 0:
            continue  # current_allocation must be positive (gate 8)
        targets.append(
            {
                "entity_id": str(row["hcp_id"]),
                "entity_type": "hcp",
                "current_allocation": spend,
                "expected_response": float(row["expected_response"]),
                "is_synthetic": bool(row.get("is_synthetic", True)),
            }
        )
    budget = float(sum(t["current_allocation"] for t in targets))
    return targets, budget


def _load_cate_spend_frame(brand: Optional[str]) -> Optional[pd.DataFrame]:
    """Load the per-HCP CATE artifact (Shard 06) for ``brand``.

    Fail-closed: an unresolved brand or a missing artifact returns None (no
    fabrication). The realized artifact has cols [hcp_id, cate_estimate,
    is_synthetic]; current_allocation is synthesized downstream (see module
    docstring), so no separate spend artifact is required.
    """
    if not brand:
        return None
    cate_path = _CATE_DIR / f"per_hcp_cate_hcp_adoption_{brand}.parquet"
    if not cate_path.exists():
        logger.info(
            "allocation_builder: missing per-HCP CATE artifact for brand=%r (%s) -> fail closed",
            brand,
            cate_path,
        )
        return None
    cate = pd.read_parquet(cate_path, columns=["hcp_id", "cate_estimate", "is_synthetic"])
    return cate if not cate.empty else None


def build_allocation_targets(*, brand: Optional[str]) -> Tuple[List[Dict[str, Any]], float]:
    """Public entry used by the dispatcher synthesizer. ([], 0.0) when unresolved."""
    frame = _load_cate_spend_frame(brand)
    if frame is None:
        return [], 0.0
    return targets_from_cate_frame(frame)


__all__ = [
    "build_allocation_targets",
    "targets_from_cate_frame",
]
