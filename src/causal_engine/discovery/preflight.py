"""Discovery frame pre-flight: make the DAG-learning frame full-rank and small.

Lane D of the real-data causal estimation program (spec
``docs/superpowers/specs/2026-09-22-real-data-causal-estimation-design.md``,
"Lane D — guided discovery on claims frames", item 1). Called from
``GraphBuilderNode._run_discovery`` before the guided tiers are built.

Why it exists (measured, ``docs/demos/results/2026-09-22_discovery_real_claims_disproof/``
and ``docs/demos/results/2026-09-22_lane_d_guided_discovery_claims/``): a real
claims frame carries whole comorbidity flag families twice (Charlson and
Elixhauser) plus their composite scores, so its correlation matrix is singular
(rank 63 of 79 on the resolved Optum persistence frame) and fisherz refuses it
outright; and PC's cost is driven by the number of conditional-independence
tests, so even a full-rank frame with 43 covariates costs ~230 s per fit
(~81 min under the production 20-resample bootstrap, against a 900 s agent
timeout). Three steps, in this order:

1. **Constant columns** are dropped (exact ``min == max`` — constancy is
   equality, not a tolerance).
2. **Exactly linearly dependent columns** are dropped by a greedy
   rank-preserving pass: a column is kept iff it raises the rank of the
   correlation matrix of what was kept before it. Implemented incrementally as
   Gram–Schmidt on centered columns (the correlation matrix's rank IS the rank
   of the centered design), so "earlier column wins" is the manifest order the
   caller passes. The redundancy tolerance is ``max(n, k) * eps`` — the same
   convention ``numpy.linalg.lstsq`` uses to call a design rank-deficient — so
   a genuine component at 5e-9 relative is kept; the criterion is
   translation- and scale-invariant (offset removed exactly, power-of-two
   rescale, residual compared with the CENTERED norm).
3. **Cap** at ``max_covariates`` by a pre-treatment screening rule that never
   looks at the treatment–outcome relation: rank the surviving covariates by
   absolute Pearson association with T and, separately, with Y; take the
   largest ``k`` such that the union of the two top-``k`` lists fits under the
   cap. Ties (at 1e-12) break by manifest order. ``protected`` covariates (the
   caller's anchored confounders — their prior-required edges need the node in
   the frame) are pruned first and exempt from the cap.

Every dropped or capped covariate is reported by name so the caller can keep
it in the adjustment guarantee: the estimate still conditions on it; only the
structure learner does not see it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

# Ties in the screening statistic closer than this are ties: two columns that
# differ only in floating-point summation noise must rank by manifest order,
# not by which sum happened to round up.
_ASSOCIATION_TIE_DECIMALS = 12


@dataclass
class PreflightResult:
    """What the pre-flight decided, by column name.

    ``kept`` is the DAG-learning covariate list in manifest order; every other
    list names covariates that stay in the adjustment guarantee but are not
    handed to the structure learner.
    """

    kept: List[str]
    constant: List[str]
    collinear: List[str]
    capped: List[str]
    protected: List[str]
    max_covariates: int
    n_offered: int
    n_rows: int
    n_rows_used: int
    screening: Dict[str, Any] = field(default_factory=dict)

    @property
    def removed(self) -> List[str]:
        """Every covariate the learner will not see, in manifest order."""
        gone = set(self.constant) | set(self.collinear) | set(self.capped)
        return [c for c in self._offered_order if c in gone]

    # Set by the builder; not part of the public payload.
    _offered_order: List[str] = field(default_factory=list, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kept": list(self.kept),
            "constant": list(self.constant),
            "collinear": list(self.collinear),
            "capped": list(self.capped),
            "protected": list(self.protected),
            "max_covariates": self.max_covariates,
            "n_offered": self.n_offered,
            "n_kept": len(self.kept),
            "n_rows": self.n_rows,
            "n_rows_used": self.n_rows_used,
            "screening": dict(self.screening),
        }


def _rank_tolerance(n_rows: int, n_cols: int) -> float:
    """``max(n, k) * eps``: what ``numpy.linalg.lstsq(rcond=None)`` treats as
    rank-deficient. Anything looser would drop information the estimators
    downstream would still have used."""
    return float(max(n_rows, n_cols) * np.finfo(float).eps)


def _prune_rank(
    X: np.ndarray,
    names: Sequence[str],
) -> tuple[List[str], List[str], List[str]]:
    """Return ``(kept, constant, collinear)`` for the columns of ``X`` (rows =
    observations, one column per name, no NaN), in the given order."""
    n = X.shape[0]
    tol = _rank_tolerance(n, len(names) + 1)
    basis: List[np.ndarray] = []
    kept: List[str] = []
    constant: List[str] = []
    collinear: List[str] = []
    for j, name in enumerate(names):
        x = X[:, j]
        if x.min() == x.max():
            constant.append(name)
            continue
        # Remove the offset exactly (Sterbenz) before any rounding-prone step,
        # then rescale by a power of two (exact) so neither a 1e14 offset nor
        # 1e-200 units can leak rounding into the residual comparison.
        x = x - x[0]
        x = x / (2.0 ** math.floor(math.log2(float(np.abs(x).max()))))
        centered = x - x.mean()
        c_norm = float(np.linalg.norm(centered))
        resid = centered.copy()
        for _ in range(2):  # re-orthogonalise once for numerical stability
            for q in basis:
                resid = resid - q * float(q @ resid)
        r_norm = float(np.linalg.norm(resid))
        if r_norm <= tol * c_norm:
            collinear.append(name)
            continue
        basis.append(resid / r_norm)
        kept.append(name)
    return kept, constant, collinear


def _abs_association(X: np.ndarray, target: np.ndarray) -> np.ndarray:
    """|Pearson r| of every column of ``X`` with ``target`` (0.0 where either
    side is constant), rounded so float noise cannot break a tie."""
    if target.min() == target.max():
        return np.zeros(X.shape[1])
    xc = X - X.mean(axis=0)
    tc = target - target.mean()
    denom = np.linalg.norm(xc, axis=0) * np.linalg.norm(tc)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(denom > 0, (xc.T @ tc) / denom, 0.0)
    return np.round(np.abs(r), _ASSOCIATION_TIE_DECIMALS)


def preflight_discovery_frame(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: Sequence[str],
    *,
    max_covariates: int = 20,
    protected: Iterable[str] = (),
) -> PreflightResult:
    """Decide which of ``covariates`` the structure learner sees.

    Args:
        data: the estimation frame (treatment, outcome and covariates as
            numeric columns). Rows with a NaN in any of these columns are
            excluded from the STATISTICS only (``n_rows_used`` reports how
            many remained); the frame itself is not imputed or modified.
        treatment, outcome: never touched — the pre-flight decides covariates
            only. Their own pathologies (a constant treatment) belong to the
            loader's guards.
        covariates: candidate columns in manifest order. Order is meaning:
            of two exactly dependent columns the earlier survives, and the
            screening rule breaks ties by it.
        max_covariates: cap on the DAG-learning covariate count (>= 1).
        protected: covariates that must survive (anchored confounders). They
            are pruned FIRST, so a duplicate elsewhere in the list is the one
            dropped, and they are exempt from the cap (still counted against
            it when filling the remainder).

    Returns:
        A :class:`PreflightResult`; ``kept`` is in manifest order.

    Raises:
        KeyError: a named column is absent from ``data``.
        ValueError: ``max_covariates < 1``, or no complete row remains.
    """
    if max_covariates < 1:
        raise ValueError(f"max_covariates must be >= 1, got {max_covariates}")
    covariates = list(covariates)
    missing = [c for c in (treatment, outcome, *covariates) if c not in data.columns]
    if missing:
        raise KeyError(f"pre-flight: column(s) not in the frame: {missing}")
    protected_list = [c for c in covariates if c in set(protected)]

    columns = [treatment, outcome, *covariates]
    numeric = data[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    complete = np.isfinite(numeric).all(axis=1)
    n_rows = int(numeric.shape[0])
    n_used = int(complete.sum())
    if n_used == 0:
        raise ValueError(
            "pre-flight: no complete row (treatment, outcome and every covariate "
            "non-missing) — cannot rank or screen the covariates"
        )
    numeric = numeric[complete]
    t_col = numeric[:, 0]
    y_col = numeric[:, 1]
    cov_index = {name: idx + 2 for idx, name in enumerate(covariates)}

    # Step 1 + 2: protected first (so they win duplicates), then the rest in
    # manifest order; results are reported back in manifest order.
    order = protected_list + [c for c in covariates if c not in set(protected_list)]
    X = numeric[:, [cov_index[c] for c in order]]
    kept_unordered, constant, collinear = _prune_rank(X, order)
    kept_set = set(kept_unordered)
    kept = [c for c in covariates if c in kept_set]
    constant = [c for c in covariates if c in set(constant)]
    collinear = [c for c in covariates if c in set(collinear)]
    protected_kept = [c for c in protected_list if c in kept_set]

    # Step 3: cap by pre-treatment screening on T and on Y separately.
    screening: Dict[str, Any] = {
        "rule": "union of top-k by |corr| with treatment and top-k by |corr| with outcome",
        "k": None,
        "top_by_treatment": [],
        "top_by_outcome": [],
    }
    capped: List[str] = []
    if len(kept) > max_covariates:
        candidates = [c for c in kept if c not in set(protected_kept)]
        room = max(0, max_covariates - len(protected_kept))
        Xc = numeric[:, [cov_index[c] for c in candidates]]
        assoc_t = _abs_association(Xc, t_col)
        assoc_y = _abs_association(Xc, y_col)
        position = {c: i for i, c in enumerate(candidates)}
        by_t = sorted(candidates, key=lambda c: (-assoc_t[position[c]], position[c]))
        by_y = sorted(candidates, key=lambda c: (-assoc_y[position[c]], position[c]))
        chosen: List[str] = []
        best_k = 0
        for k in range(1, len(candidates) + 1):
            union = list(dict.fromkeys(by_t[:k] + by_y[:k]))
            if len(union) > room:
                break
            chosen = union
            best_k = k
        chosen_set = set(chosen) | set(protected_kept)
        capped = [c for c in kept if c not in chosen_set]
        kept = [c for c in kept if c in chosen_set]
        screening.update(
            {
                "k": best_k,
                "top_by_treatment": by_t[:best_k],
                "top_by_outcome": by_y[:best_k],
                "association_with_treatment": {
                    c: float(assoc_t[position[c]]) for c in candidates
                },
                "association_with_outcome": {c: float(assoc_y[position[c]]) for c in candidates},
            }
        )

    return PreflightResult(
        kept=kept,
        constant=constant,
        collinear=collinear,
        capped=capped,
        protected=protected_kept,
        max_covariates=max_covariates,
        n_offered=len(covariates),
        n_rows=n_rows,
        n_rows_used=n_used,
        screening=screening,
        _offered_order=covariates,
    )


def preflight_summary(result: PreflightResult, *, max_names: Optional[int] = None) -> str:
    """One human-readable line for the API warnings channel: what was pruned
    and capped, by name, and the guarantee that none of it left the adjustment
    set. ``max_names`` truncates each list for display (None = every name)."""

    def _names(items: List[str]) -> str:
        if not items:
            return "none"
        shown = items if max_names is None else items[:max_names]
        extra = len(items) - len(shown)
        text = ", ".join(shown)
        return f"{text} (+{extra} more)" if extra > 0 else text

    return (
        f"Discovery pre-flight: {result.n_offered} covariates offered, "
        f"{len(result.kept)} handed to structure learning (cap {result.max_covariates}); "
        f"dropped {len(result.constant)} constant [{_names(result.constant)}], "
        f"{len(result.collinear)} exactly collinear [{_names(result.collinear)}], "
        f"{len(result.capped)} capped by pre-treatment screening [{_names(result.capped)}]. "
        "Every dropped or capped covariate stays in the adjustment set; only the "
        "structure learner does not see it."
    )
