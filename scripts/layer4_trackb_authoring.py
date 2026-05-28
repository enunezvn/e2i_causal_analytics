#!/usr/bin/env python3
"""Track-2B authoring-yield harness (Plan v4 Layer B / Phase 2, Tasks 7-8).

Deterministic, testable tooling around the Track-2B experiment:

* ``stratified_sample`` — pick ~N literature golden-set features spread across
  the three temporal regimes (pre-index demographics, the hard index-windowed
  set, post-index Layer-1 controls) while covering all six causal roles, so the
  yield experiment isn't biased toward an easy regime/role.
* ``regime_of`` / ``knowable_at`` — map a golden-set entry to its temporal
  regime from the ``knowable_at=`` token in ``derivation_pseudocode``.

The DAG-edge authoring itself is the experiment's *data* (authored blind to the
``ground_truth_role`` label, by a subagent) — it is deliberately NOT in this
module. ``build_edge_augmented_fixture`` (added under TDD for Task 8) assembles
the authored edges into the ``--decider structural`` fixture shape.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

# knowable_at tokens that mark each temporal regime.
_PRE = {"pre", "preindex"}
_INDEX = {"index_date", "index_date_minus_1", "date"}
# Everything else (index_date_plus_*, t_plus_*, event_date, ...) is post-index.

_ROLES = ("ancestor", "collider", "confounder", "descendant", "instrument", "mediator")
_REGIMES = ("index", "pre", "post")

_KNOWABLE_RE = re.compile(r"knowable_at=([A-Za-z0-9_]+)")


def knowable_at(entry: dict[str, Any]) -> str:
    """Extract the ``knowable_at`` token from an entry's derivation pseudocode."""
    m = _KNOWABLE_RE.search(entry.get("derivation_pseudocode", ""))
    return m.group(1) if m else ""


def regime_of(entry: dict[str, Any]) -> str:
    """Classify an entry into a temporal regime: ``pre`` | ``index`` | ``post``."""
    k = knowable_at(entry)
    if k in _PRE:
        return "pre"
    if k in _INDEX:
        return "index"
    return "post"


def stratified_sample(
    entries: list[dict[str, Any]], n: int = 10, seed: int = 0
) -> list[dict[str, Any]]:
    """Deterministic role- and regime-stratified sample of ``n`` entries.

    Guarantees (given a set that contains all six roles and all three regimes):
    every causal role appears at least once AND every temporal regime appears at
    least once, with the remainder filled in a seed-reproducible order.
    """
    rng = random.Random(seed)
    order = list(range(len(entries)))
    rng.shuffle(order)  # deterministic visiting order (tie-break)

    chosen: list[int] = []
    chosen_set: set[int] = set()

    def add(i: int) -> None:
        if i not in chosen_set and len(chosen) < n:
            chosen.append(i)
            chosen_set.add(i)

    # 1) one entry per role → role coverage.
    roles_have: set[str] = set()
    for i in order:
        if len(chosen) >= n:
            break
        r = entries[i].get("ground_truth_role")
        if r is not None and r not in roles_have:
            add(i)
            roles_have.add(r)

    # 2) ensure each temporal regime is represented.
    regimes_have = {regime_of(entries[i]) for i in chosen}
    for reg in _REGIMES:
        if reg in regimes_have or len(chosen) >= n:
            continue
        for i in order:
            if regime_of(entries[i]) == reg:
                add(i)
                regimes_have.add(reg)
                break

    # 3) fill the remainder deterministically.
    for i in order:
        if len(chosen) >= n:
            break
        add(i)

    return [entries[i] for i in chosen]


_STRUCTURAL_FROM_AUTHORED = ("feature_node", "treatment_node", "outcome_node", "edges")


def build_edge_augmented_fixture(
    golden_entries: list[dict[str, Any]],
    authored: dict[str, dict[str, Any]],
    *,
    fixture_kind: str = "literature_heldout_edges",
) -> dict[str, Any]:
    """Assemble a ``--decider structural`` fixture from golden entries + edges.

    For each ``feature_name`` in ``authored``, emit an entry that carries the
    **independent literature** ``ground_truth_role`` and ``cohort`` from the
    golden set (never from the blind authoring payload — that independence is
    what makes the precision test non-circular) and the
    ``feature_node``/``treatment_node``/``outcome_node``/``edges`` from the
    authored DAG. The prose ``rationale`` is deliberately NOT carried over.

    Raises ``KeyError`` if an authored ``feature_name`` is absent from the
    golden set (an authoring typo must fail loudly, not silently drop).
    """
    by_name = {e["feature_name"]: e for e in golden_entries}
    entries: list[dict[str, Any]] = []
    for feature_name, dag in authored.items():
        if feature_name not in by_name:
            raise KeyError(
                f"authored feature {feature_name!r} is not in the golden set "
                f"({len(by_name)} entries) — authoring typo or wrong fixture?"
            )
        golden = by_name[feature_name]
        entry: dict[str, Any] = {
            "feature_name": feature_name,
            "ground_truth_role": golden["ground_truth_role"],
            "cohort": golden.get("cohort") or golden.get("scenario") or "unknown",
        }
        for k in _STRUCTURAL_FROM_AUTHORED:
            entry[k] = dag[k]
        entries.append(entry)
    return {"fixture_kind": fixture_kind, "total_entries": len(entries), "entries": entries}


def _main() -> None:
    ap = argparse.ArgumentParser(description="Track-2B authoring-yield sampler.")
    ap.add_argument("--golden-set", default="tests/fixtures/causal_role_golden_set.json")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    entries = json.loads(Path(args.golden_set).read_text())["entries"]
    sample = stratified_sample(entries, n=args.n, seed=args.seed)
    for e in sample:
        print(
            f"{e['cohort']:<18} {regime_of(e):<6} {e['ground_truth_role']:<11} {e['feature_name']}"
        )


if __name__ == "__main__":
    _main()
