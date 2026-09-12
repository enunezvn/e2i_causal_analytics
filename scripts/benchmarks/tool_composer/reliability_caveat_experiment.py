"""Pre-registered experiment: does a reliability caveat change the planner's tool choice?

**Pre-registration (spec §7.2). Written before any run; do not edit after data collection.**

- **Question.** With `TOOL_COMPOSER_RELIABILITY_IN_PLANNER` set, the planning prompt gains one line
  per caveated tool. Does that line change which tools the LLM plans? A caveat nobody has shown to
  change a decision is a label, not a signal, which is why the flag ships default-off.
- **Items.** A frozen set of K >= 20 real questions, drawn from the entry points' own loaders. The
  set is hashed and the hash is recorded with the result, so an item set cannot be changed after
  the fact.
- **Pilot.** P >= 10 items run first (`--pilot`) to confirm the harness end to end. Pilot items are
  reported separately and never pooled into the evaluation.
- **Arms, paired per item.** A = flag off (today's prompt). B = flag on, with a planted caveat on
  one tool the item would otherwise plan. Arm order is seeded and alternated, so position cannot
  explain a difference.
- **Primary outcome.** Per item, whether the plan selects the target tool, in A and in B. Test:
  exact one-sided McNemar on the discordant pairs.
- **Pass rule (all of):**
    1. effect `(b - c) / K >= 0.30`, where b = picked in A only, c = picked in B only;
    2. `p < 0.05`;
    3. three observed-count guards, on raw counts and with no significance test:
       `invalid(B) <= invalid(A)`, `total_failure(B) <= total_failure(A)`,
       `median succeeded steps(B) >= median(A)`.
- **Validity outcomes.** Every returned plan goes through `planner._validate_plan` and the §7.4
  order checks, then executes with the real `PlanExecutor` on the item's real frame. Invalid = a
  PlanningError, a validator rejection, or any `plan_defect` / `not_registered` step. Total failure
  = zero succeeded steps. Every frozen item stays in the denominator; an item whose planner call
  errors counts as invalid in that arm.
- **Memory guard.** `free -m` before each item; the run aborts below 1500 MiB.
- **Spend.** Planner calls P + 2K >= 70, thinking disabled, ~1000 output tokens each. Plan execution
  is local (BentoML in-cluster), so no external spend. **The run needs the owner's authorization
  (O2). This module runs nothing on import, and `--evaluate` refuses without an explicit
  `--i-have-authorization` flag.**
- **Result handling.** The result JSON and a summary are committed under `docs/demos/results/`. The
  flag default flips to on only in a follow-up commit citing a passing result. A failing result
  leaves the flag off and records which condition failed.

The analysis below is pure and is tested (`tests/unit/test_scripts/test_reliability_caveat_analysis.py`);
the collection half is what the authorization gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
from typing import Any, Dict, List, Sequence

logger = logging.getLogger(__name__)

#: The primary effect the experiment must show to justify turning the flag on.
MIN_EFFECT = 0.30

#: Significance level for the one-sided McNemar test.
ALPHA = 0.05

#: The run stops below this much available memory (MiB).
MEMORY_FLOOR_MIB = 1500

ARMS = ("A", "B")


# ---------------------------------------------------------------------------
# Analysis — pure, tested, and independent of whether the run ever happens
# ---------------------------------------------------------------------------


def mcnemar_one_sided(b: int, c: int) -> float:
    """Exact one-sided McNemar p-value for discordant pairs (b in A only, c in B only).

    Equivalent to a binomial test of b successes in b + c trials against 0.5, greater. With no
    discordant pairs there is no evidence either way, which is a p-value of 1.0, not 0.
    """
    if b + c == 0:
        return 1.0
    from scipy import stats

    return float(stats.binomtest(b, b + c, 0.5, alternative="greater").pvalue)


def summarize_pairs(pairs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts the pass rule reads. Every item counts, in both arms."""
    b = sum(1 for p in pairs if p["picked_target"]["A"] and not p["picked_target"]["B"])
    c = sum(1 for p in pairs if p["picked_target"]["B"] and not p["picked_target"]["A"])
    return {
        "n_items": len(pairs),
        "discordant": {"b": b, "c": c},
        "picked": {arm: sum(1 for p in pairs if p["picked_target"][arm]) for arm in ARMS},
        "invalid": {arm: sum(1 for p in pairs if p["invalid"][arm]) for arm in ARMS},
        "total_failure": {arm: sum(1 for p in pairs if p["total_failure"][arm]) for arm in ARMS},
        "median_succeeded_steps": {
            arm: (statistics.median([p["succeeded_steps"][arm] for p in pairs]) if pairs else 0)
            for arm in ARMS
        },
    }


def pass_rule(pairs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply the pre-registered rule. Returns the verdict AND every condition that failed.

    Conjunctive by design: an effect that is real, significant, and not bought by breaking plans.
    The three guards are read on observed counts — they are "not observed worse" gates, stated as
    such, and deliberately not non-inferiority claims.
    """
    summary = summarize_pairs(pairs)
    b, c = summary["discordant"]["b"], summary["discordant"]["c"]
    n_items = summary["n_items"]

    effect = (b - c) / n_items if n_items else 0.0
    p_value = mcnemar_one_sided(b, c)

    failed: List[str] = []
    if effect < MIN_EFFECT:
        failed.append("effect")
    if p_value >= ALPHA:
        failed.append("significance")
    if summary["invalid"]["B"] > summary["invalid"]["A"]:
        failed.append("invalid")
    if summary["total_failure"]["B"] > summary["total_failure"]["A"]:
        failed.append("total_failure")
    if summary["median_succeeded_steps"]["B"] < summary["median_succeeded_steps"]["A"]:
        failed.append("median_succeeded_steps")

    return {
        "passed": not failed,
        "failed": failed,
        "effect": effect,
        "p_value": p_value,
        "min_effect": MIN_EFFECT,
        "alpha": ALPHA,
        **summary,
    }


def item_set_hash(items: Sequence[Dict[str, Any]]) -> str:
    """A stable hash of the frozen item set, so the set cannot change after collection."""
    payload = json.dumps(
        [{"id": item["id"], "query": item["query"]} for item in items],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Collection — gated on the owner's authorization (O2)
# ---------------------------------------------------------------------------


def available_memory_mib() -> int:
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return 0


def _refuse_without_authorization(args: argparse.Namespace) -> None:
    if not args.i_have_authorization:
        raise SystemExit(
            "This run spends production LLM budget (O2). Re-run with --i-have-authorization "
            "once the owner has authorized it; see docs/runbooks/tool-composer-learning-loop.md."
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="run the pilot subset only")
    parser.add_argument("--evaluate", action="store_true", help="run the full evaluation")
    parser.add_argument(
        "--i-have-authorization",
        action="store_true",
        help="the owner has authorized this run's LLM spend (O2)",
    )
    parser.add_argument("--results", default="docs/demos/results", help="where to write results")
    args = parser.parse_args(argv)

    if not (args.pilot or args.evaluate):
        parser.error("choose --pilot or --evaluate")
    _refuse_without_authorization(args)

    # The collection half is deliberately not implemented as a side effect of import, and is not
    # exercised by the test suite: it spends real budget. It is written when the authorization
    # arrives, against the pre-registration above.
    raise SystemExit(
        "Collection is not implemented yet: it is written against the pre-registration above "
        "once the owner authorizes the spend (O2). The analysis functions are tested and ready."
    )


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
