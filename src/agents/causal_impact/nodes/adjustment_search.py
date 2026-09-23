"""Bounded backdoor adjustment-set search (#2233).

Why this module exists
----------------------
``GraphBuilderNode`` ships, on every manual-DAG path (REJECT / REVIEW / AUGMENT),
a DAG in which every declared covariate is a common cause of treatment and
outcome. On such a DAG no set of size <= 3 d-separates T and Y, so the minimal-set
enumeration exhausts C(k, <= 3) criterion checks before its full-set fallback:
76,154 checks at k = 77, measured 242-316 s (Lane D evidence, item 5), and on the
live Optum persistence cert (2026-09-22) the AUGMENT-path search ran on the event
loop past gunicorn's ``--timeout 120`` — the arbiter aborted the worker (code 134)
and the analysis row was orphaned ``running``.

The search is therefore bounded twice, and the caller runs it OFF the loop
(``asyncio.to_thread``) so the worker heartbeat survives whatever the bound:

* ``max_candidates`` — a CANDIDATE CAP: above it the subset enumeration is skipped
  outright and only the full-candidate-set check runs. Deterministic (the same
  DAG gives the same answer on any box). On the agent path the shipped set is then
  unioned with every declared covariate by ``_apply_adjustment_guarantee``, which
  on that path is the full candidate set anyway, so nothing the estimate adjusts
  on changes.
* ``time_budget_s`` — a WALL-TIME budget on the enumeration: when it runs out the
  sets found so far ship (best-so-far, always admissible); with none found the
  full-candidate-set fallback runs (one check) exactly as the unbounded search
  would have after exhausting the enumeration.

A hit bound is never silent: ``AdjustmentSearchResult.warning()`` names it for the
response's warnings channel. Validity is never traded for speed — every returned
set passed ``satisfies_backdoor_criterion``; ``[[]]`` is returned only when no
admissible set exists at all (the pre-existing documented fallback).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, List, Optional, Set

import networkx as nx

from src.ml.causal_role_dgp.backdoor import satisfies_backdoor_criterion

# Wall-time budget for the enumeration. Sized from the measured per-check cost
# (~3.2 ms on the 15,209-row frame's 176-edge AUGMENT DAG: 242 s / 76,154) so a
# 61-candidate Lane-A frame (37,882 checks, ~120 s unbounded) degrades to its
# full-set fallback with a warning instead of stalling; the search runs off the
# loop, so this bounds the ANALYSIS latency, not the worker's liveness. Override
# per run with state ``adjustment_search_time_budget_s`` (None = unbounded).
ADJUSTMENT_SEARCH_TIME_BUDGET_S = 60.0
# Candidate cap: C(40, <= 3) = 10,701 checks (~35 s at the measured cost) is the
# largest enumeration worth paying for a minimal set the guarantee channel will
# union with every declared covariate anyway. Override with state
# ``adjustment_search_max_candidates`` (None = uncapped).
ADJUSTMENT_SEARCH_MAX_CANDIDATES = 40

MAX_SET_SIZE = 3
MAX_SETS = 3

Criterion = Callable[[nx.DiGraph, Set[str], str, str], bool]


@dataclass(frozen=True)
class AdjustmentSearchResult:
    adjustment_sets: List[List[str]]
    n_candidates: int
    n_checks: int
    elapsed_s: float
    bound_hit: Optional[str]  # None | "time_budget" | "candidate_cap"
    time_budget_s: Optional[float] = None
    max_candidates: Optional[int] = None

    def warning(self) -> Optional[str]:
        """One line for the response's warnings channel when a bound was hit."""
        if self.bound_hit == "candidate_cap":
            return (
                f"Backdoor adjustment-set search: {self.n_candidates} candidate covariates "
                f"exceed the cap of {self.max_candidates}; the minimal-set enumeration was "
                "skipped and the full admissible candidate set was used instead (a valid, "
                "non-minimal adjustment; declared covariates are unioned in regardless)."
            )
        if self.bound_hit == "time_budget":
            budget = f"{self.time_budget_s:g}" if self.time_budget_s is not None else "?"
            return (
                f"Backdoor adjustment-set search stopped by its {budget} s time budget after "
                f"{self.n_checks} criterion checks over {self.n_candidates} candidates "
                f"({self.elapsed_s:.1f} s); the sets found so far were used (a valid "
                "adjustment; declared covariates are unioned in regardless)."
            )
        return None


def _candidates(dag: nx.DiGraph, treatment: str, outcome: str) -> Set[str]:
    # Backdoor criterion (Pearl 2009, Def. 3.3.1): candidates are non-descendants
    # of the treatment. An isolated node lies on no path, so it can neither block
    # nor open one: no minimal backdoor set contains it and Z ∪ {isolated} is
    # admissible iff Z is — enumerating it only multiplies the search (declared
    # covariates among them still reach the adjustment set through the guarantee).
    descendants = nx.descendants(dag, treatment)
    nodes = (set(dag.nodes()) - {treatment, outcome}) - descendants
    return {n for n in nodes if dag.degree(n) > 0}


def find_adjustment_sets(
    dag: nx.DiGraph,
    treatment: str,
    outcome: str,
    *,
    time_budget_s: Optional[float] = ADJUSTMENT_SEARCH_TIME_BUDGET_S,
    max_candidates: Optional[int] = ADJUSTMENT_SEARCH_MAX_CANDIDATES,
    criterion: Criterion = satisfies_backdoor_criterion,
    clock: Callable[[], float] = time.monotonic,
) -> AdjustmentSearchResult:
    """Find backdoor adjustment sets, smallest first, under the two bounds.

    Returns up to ``MAX_SETS`` minimal sets of size <= ``MAX_SET_SIZE``; failing
    that the full candidate set when it is admissible; failing that ``[[]]`` (the
    documented no-adjustment fallback for a genuinely unblockable backdoor).
    """
    start = clock()

    def _done(sets: List[List[str]], n_checks: int, bound: Optional[str]) -> AdjustmentSearchResult:
        return AdjustmentSearchResult(
            adjustment_sets=sets,
            n_candidates=n_candidates,
            n_checks=n_checks,
            elapsed_s=max(0.0, clock() - start),
            bound_hit=bound,
            time_budget_s=time_budget_s,
            max_candidates=max_candidates,
        )

    # Guard: treatment/outcome must be present and distinct. A degenerate
    # treatment == outcome query has no meaningful backdoor adjustment and would
    # make nx.is_d_separator raise (non-disjoint x/y node sets).
    if treatment not in dag or outcome not in dag or treatment == outcome:
        n_candidates = 0
        return _done([[]], 0, None)

    candidates = _candidates(dag, treatment, outcome)
    n_candidates = len(candidates)
    ordered = sorted(candidates)
    n_checks = 0
    bound: Optional[str] = None
    found: List[List[str]] = []

    capped = max_candidates is not None and n_candidates > max_candidates
    if capped:
        bound = "candidate_cap"
    else:
        # Search by increasing set size; return the smallest valid sets found.
        stop = False
        for size in range(0, min(MAX_SET_SIZE, n_candidates) + 1):
            for combo in combinations(ordered, size):
                if time_budget_s is not None and clock() - start >= time_budget_s:
                    bound = "time_budget"
                    stop = True
                    break
                n_checks += 1
                if criterion(dag, set(combo), treatment, outcome):
                    found.append(list(combo))
                    if len(found) >= MAX_SETS:
                        return _done(found, n_checks, bound)
            if found or stop:
                break
        if found:
            return _done(found, n_checks, bound)

    # No MINIMAL admissible set of size <= 3 (found, or reachable within the
    # bounds). Before defaulting to no adjustment — which would silently leave
    # the estimate CONFOUNDED — try the FULL candidate set: with > 3 independent
    # confounders the only admissible set is all of them.
    if candidates:
        n_checks += 1
        if criterion(dag, set(candidates), treatment, outcome):
            return _done([ordered], n_checks, bound)

    # Genuinely no admissible set (e.g. an unblockable backdoor path): documented
    # fallback to no adjustment.
    return _done([[]], n_checks, bound)
