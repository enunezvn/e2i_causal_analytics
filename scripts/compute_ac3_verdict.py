"""Compute AC3 verdict from two precision reports.

AC3 rule (strengthened per plan-239-n200 §3):

    miprov2_wins iff
        mipro.overall.gated.precision_instrument >= boot.overall.gated.precision_instrument
        AND  ∀ cohort c ∈ {CSU_remibrutinib, PNH_fabhalta, BC_kisqali}:
            mipro[c].gated.precision_instrument >= boot[c].gated.precision_instrument

Input report schema (from ``scripts/measure_layer4_precision.py``):

    {
      "per_cohort": [
        {"cohort": "CSU_remibrutinib", "gate": "gated"|"ungated",
         "precision_instrument": float|None, ...},
        ...
      ],
      "overall": {
        "gated":   {"cohort": "OVERALL", "gate": "gated",   "precision_instrument": float|None, ...},
        "ungated": {"cohort": "OVERALL", "gate": "ungated", "precision_instrument": float|None, ...}
      }
    }

Output: AC3 verdict JSON committed at ``artifacts/dspy/ac3_verdict_n200.json``.

Usage::

    python scripts/compute_ac3_verdict.py BOOT_REPORT.json MIPRO_REPORT.json OUT_VERDICT.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

# Order matters for the verdict table: OVERALL first, then per-cohort.
COHORTS = ("OVERALL", "CSU_remibrutinib", "PNH_fabhalta", "BC_kisqali")


def _gated_precision_for_cohort(report: dict[str, Any], cohort: str) -> Optional[float]:
    """Pull precision_instrument from gated bucket for a given cohort label."""
    if cohort == "OVERALL":
        gated_overall = report.get("overall", {}).get("gated")
        if gated_overall is None:
            return None
        return gated_overall.get("precision_instrument")

    for entry in report.get("per_cohort", []):
        if entry.get("cohort") == cohort and entry.get("gate") == "gated":
            return entry.get("precision_instrument")
    return None


def main(boot_path: str, mipro_path: str, out_path: str) -> int:
    boot = json.loads(Path(boot_path).read_text())
    mipro = json.loads(Path(mipro_path).read_text())

    rows: list[dict[str, Any]] = []
    overall_ok: Optional[bool] = None
    cohort_ok: bool = True
    cohort_regressions: list[dict[str, Any]] = []

    for c in COHORTS:
        b = _gated_precision_for_cohort(boot, c)
        m = _gated_precision_for_cohort(mipro, c)

        if m is None or b is None:
            meets = None
        else:
            meets = m >= b

        rows.append(
            {
                "cohort": c,
                "bootstrap_gated_precision_instrument": b,
                "miprov2_gated_precision_instrument": m,
                "miprov2_meets_or_exceeds_bootstrap": meets,
            }
        )

        if c == "OVERALL":
            overall_ok = meets
        else:
            if m is not None and b is not None and m < b:
                cohort_ok = False
                cohort_regressions.append(
                    {"cohort": c, "bootstrap": b, "miprov2": m, "delta": m - b}
                )

    miprov2_wins = bool(overall_ok) and cohort_ok
    branch_decision = "miprov2_default" if miprov2_wins else "bootstrap_default"

    verdict = {
        "schema_version": 1,
        "ac3_rule": (
            "miprov2.overall.gated.precision_instrument >= "
            "bootstrap.overall.gated.precision_instrument "
            "AND for each cohort c in {CSU_remibrutinib, PNH_fabhalta, BC_kisqali}: "
            "miprov2[c].gated.precision_instrument >= bootstrap[c].gated.precision_instrument"
        ),
        "compile_seed": 42,
        "compile_n_examples": 240,
        "golden_set_n_entries": 91,
        "rows": rows,
        "overall_ok": overall_ok,
        "cohort_ok": cohort_ok,
        "cohort_regressions": cohort_regressions,
        "miprov2_wins": miprov2_wins,
        "branch_decision": branch_decision,
    }

    Path(out_path).write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    print(f"AC3 verdict written to {out_path}")
    print(f"miprov2_wins: {miprov2_wins}")
    print(f"branch_decision: {branch_decision}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
