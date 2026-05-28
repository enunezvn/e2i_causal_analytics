"""Layer 3 — Adversarial Leakage Discriminator.

Per-feature suspicion scoring via permutation-baseline-relative z-score.
The threshold for "suspicious" is DATA-DERIVED, not hardcoded — it adapts
to each cohort's null distribution automatically.

Why this replaces the hardcoded 0.65 / 0.80 thresholds:
- A feature with single-feature AUC 0.65 in a 200-patient low-prevalence
  cohort might be 2σ above the permutation null (legitimate weak signal).
- The same AUC 0.65 in a 5000-patient large cohort might be 8σ above the
  null (clear leakage).
- The hardcoded thresholds treated these the same. The z-score doesn't.

Disease-agnostic by construction: permutation tests work on any binary target.
No per-cohort tuning needed.

Reference: .claude/plans/adaptive_temporal_validity_redesign.md (Layer 3).
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def compute_adversarial_score(
    feature: np.ndarray | pd.Series,
    target: np.ndarray | pd.Series,
    *,
    n_permutations: int = 1000,
    seed: int = 42,
    z_threshold: float = 5.0,
) -> dict[str, Any]:
    """Score a feature's suspiciousness via permutation baseline.

    Args:
        feature: 1D array of feature values.
        target: 1D array of binary target values.
        n_permutations: Number of label shuffles to build the null distribution.
        seed: RNG seed for reproducibility.
        z_threshold: How many standard deviations above the null is "suspicious".
            Default 5σ — strict but not absolute. The threshold itself is
            documented and adjustable per use-case (governance), unlike the
            previous hardcoded AUC thresholds which had no statistical meaning.

    Returns:
        Dictionary with:
        - actual_auc: feature's effective AUC (max of auc, 1-auc) — the fold
          makes the test direction-agnostic, so a feature with raw AUC near 0
          (perfect anticorrelation with the target) is just as suspicious as
          one with raw AUC near 1.
        - null_mean: mean AUC under permuted labels (also folded)
        - null_std: std of folded permuted AUC distribution
        - z_score: (actual_auc - null_mean) / null_std
        - p_value: plus-one (Phipson & Smyth 2010) empirical upper-tail
          permutation p-value on the folded scale:
          ``(1 + #{folded_null >= folded_actual}) / (1 + n_permutations)``.
          Floored at ``1 / (1 + n_permutations)`` and therefore NEVER exactly
          0.0 — a finite permutation sample cannot prove impossibility, and a
          literal 0.0 would corrupt the downstream BH multiple-testing step
          (see ``benjamini_hochberg``). This is the unbiased fix for the old
          ``np.mean(null >= actual)`` form, which could return exactly 0.0.
        - suspicious: True if z_score > z_threshold
        - n_permutations: actual number of permutations completed
    """
    feature_arr = np.asarray(feature, dtype=float)
    target_arr = np.asarray(target, dtype=int)

    # Compute the actual single-feature AUC (effective, max of auc and 1-auc)
    try:
        raw_auc = float(roc_auc_score(target_arr, feature_arr))
    except ValueError:
        # Degenerate cases: only one class in target, all-NaN feature, etc.
        return {
            "actual_auc": float("nan"),
            "null_mean": float("nan"),
            "null_std": float("nan"),
            "z_score": float("nan"),
            "p_value": float("nan"),
            "suspicious": False,
            "n_permutations": 0,
        }
    actual_auc = max(raw_auc, 1 - raw_auc)

    # Build the null distribution by shuffling target labels
    rng = np.random.default_rng(seed)
    null_aucs: list[float] = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(target_arr)
        try:
            null_raw = float(roc_auc_score(shuffled, feature_arr))
            null_aucs.append(max(null_raw, 1 - null_raw))
        except ValueError:
            continue

    if not null_aucs:
        return {
            "actual_auc": actual_auc,
            "null_mean": float("nan"),
            "null_std": float("nan"),
            "z_score": float("nan"),
            "p_value": float("nan"),
            "suspicious": False,
            "n_permutations": 0,
        }

    null_arr = np.array(null_aucs)
    null_mean = float(np.mean(null_arr))
    null_std = float(np.std(null_arr))

    # Z-score against null distribution
    if null_std > 0:
        z_score = (actual_auc - null_mean) / null_std
    else:
        z_score = float("inf") if actual_auc > null_mean else 0.0

    # Plus-one (Phipson & Smyth 2010) empirical upper-tail p_value on the
    # folded scale. The fold of both ``actual_auc`` and the null distribution
    # to [0.5, 1.0] makes this an effectively two-sided test on the underlying
    # raw AUC: a raw AUC of 0.05 (anticorrelation) and 0.95 (correlation) both
    # fold to 0.95 and produce the same suspicious-leakage signal.
    #
    # The ``(1 + count) / (1 + n)`` form floors the p-value at 1/(1+n) so it is
    # NEVER exactly 0.0 — a finite permutation sample cannot prove
    # impossibility, and a literal 0.0 would corrupt the downstream BH
    # multiple-testing math (it would dominate every rank). This replaces the
    # old ``np.mean(null >= actual)``, which returned exactly 0.0 for a sharp
    # leak despite the docstring's (then-false) floor claim.
    n_perm_done = len(null_arr)
    p_value = float((1 + int(np.sum(null_arr >= actual_auc))) / (1 + n_perm_done))

    return {
        "actual_auc": actual_auc,
        "null_mean": null_mean,
        "null_std": null_std,
        "z_score": z_score,
        "p_value": p_value,
        "suspicious": z_score > z_threshold,
        "n_permutations": len(null_aucs),
    }


def compute_feature_ablation(
    X: pd.DataFrame,
    target: np.ndarray | pd.Series,
    *,
    model_factory: Any | None = None,
    n_permutations: int = 200,
    seed: int = 42,
    z_threshold: float = 5.0,
) -> dict[str, Any]:
    """Per-feature ablation: drop each feature, retrain, measure |delta_AUC|.

    A feature whose removal drops the model's AUC significantly is either a
    critical legitimate predictor OR a leak that the multi-feature model
    relies on. The output is descriptive — Layer 4's downstream judgment
    decides whether the dependency is legitimate or suspect.

    Args:
        X: Feature DataFrame (n_samples × n_features).
        target: 1D binary target array.
        model_factory: Callable returning a fresh sklearn-compatible classifier
            with predict_proba. Defaults to a small Logistic Regression for
            speed; production usage should pass the actual model class.
        n_permutations: Number of permutation rounds to build per-feature null
            of |delta_AUC| (smaller than discriminator since each round
            requires retraining).
        seed: RNG seed.
        z_threshold: Same as compute_adversarial_score.

    Returns:
        Dictionary with:
        - full_auc: AUC of model trained on ALL features
        - per_feature: list of dicts, one per feature, with:
            - feature: name
            - delta_auc: full_auc - auc_without_feature (positive = feature helps)
            - z_score: z-score of delta_auc against permutation null
            - p_value: plus-one upper-tail permutation p-value of delta_auc,
              floored at 1/(1+n_permutations) (never exactly 0.0); the input
              to BH multiple-testing across features (see ``benjamini_hochberg``)
            - suspicious: True if z_score > z_threshold
    """
    if model_factory is None:
        from sklearn.linear_model import LogisticRegression

        def model_factory():  # noqa: E306
            return LogisticRegression(max_iter=200, random_state=seed)

    X_arr = X.copy()
    y_arr = np.asarray(target, dtype=int)
    feature_names = list(X_arr.columns)

    # Train the full model
    full_model = model_factory()
    full_model.fit(X_arr.values, y_arr)
    full_auc = float(roc_auc_score(y_arr, full_model.predict_proba(X_arr.values)[:, 1]))

    per_feature_results: list[dict[str, Any]] = []

    for feat_name in feature_names:
        # Train without this feature
        X_minus = X_arr.drop(columns=[feat_name]).values
        try:
            ablated_model = model_factory()
            ablated_model.fit(X_minus, y_arr)
            ablated_auc = float(roc_auc_score(y_arr, ablated_model.predict_proba(X_minus)[:, 1]))
        except (ValueError, RuntimeError):
            ablated_auc = float("nan")
        delta_auc = full_auc - ablated_auc if not np.isnan(ablated_auc) else float("nan")

        # Permutation null for delta_auc: shuffle the FEATURE column, retrain,
        # measure delta. Smaller n_permutations because of training cost.
        #
        # Each feature gets an INDEPENDENT permutation stream keyed to its name
        # (not one shared sequential RNG). A shared RNG made each feature's null
        # depend on how many draws earlier features consumed — coupling the
        # per-feature p-values and making the multiple-testing inputs depend on
        # arbitrary column order. Keying by name (crc32) yields reproducible,
        # column-order-invariant, mutually-independent nulls — the correct
        # inputs for the BH step (see ``benjamini_hochberg``).
        feat_rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), zlib.crc32(feat_name.encode("utf-8"))])
        )
        null_deltas: list[float] = []
        for _ in range(n_permutations):
            shuffled_feat = feat_rng.permutation(X_arr[feat_name].values)
            X_perm = X_arr.copy()
            X_perm[feat_name] = shuffled_feat
            try:
                perm_model = model_factory()
                perm_model.fit(X_perm.values, y_arr)
                perm_auc = float(
                    roc_auc_score(y_arr, perm_model.predict_proba(X_perm.values)[:, 1])
                )
                null_deltas.append(full_auc - perm_auc)
            except (ValueError, RuntimeError):
                continue

        null_mean = float(np.mean(null_deltas)) if null_deltas else float("nan")
        null_std = float(np.std(null_deltas)) if null_deltas else float("nan")
        # Match compute_adversarial_score's null_std=0 semantics: a degenerate
        # null distribution (every permuted ablation produced the same delta)
        # means a deterministically known signal — return +inf (or 0) rather
        # than NaN so a consumer's ``suspicious = z_score > z_threshold``
        # test fires consistently across the two functions.
        if np.isnan(delta_auc) or np.isnan(null_std):
            z_score = float("nan")
        elif null_std > 0:
            z_score = (delta_auc - null_mean) / null_std
        else:
            # null_std == 0: every permuted delta_auc equals null_mean
            z_score = float("inf") if delta_auc > null_mean else 0.0

        # Plus-one upper-tail p-value of the |delta_AUC| permutation null,
        # matching compute_adversarial_score's one-sided semantics (a large
        # positive delta_auc = feature is important/leaky). Floored at
        # 1/(1+n), never exactly 0.0 — the valid input for the BH step.
        if np.isnan(delta_auc) or not null_deltas:
            p_value = float("nan")
        else:
            null_delta_arr = np.asarray(null_deltas)
            p_value = float((1 + int(np.sum(null_delta_arr >= delta_auc))) / (1 + len(null_deltas)))

        per_feature_results.append(
            {
                "feature": feat_name,
                "full_auc": full_auc,
                "ablated_auc": ablated_auc,
                "delta_auc": delta_auc,
                "null_mean": null_mean,
                "null_std": null_std,
                "z_score": z_score,
                "p_value": p_value,
                "suspicious": z_score > z_threshold if not np.isnan(z_score) else False,
            }
        )

    return {
        "full_auc": full_auc,
        "n_features": len(feature_names),
        "n_permutations": n_permutations,
        "per_feature": per_feature_results,
    }


def min_permutations_for_fdr(n_features: int, q: float) -> int:
    """Minimum permutations for a BH rejection to be *possible* at all.

    The smallest plus-one permutation p-value is ``1 / (1 + n)`` (zero null
    exceedances). For the most-significant feature (BH rank 1) to clear its
    threshold ``q / m`` we need ``1/(1+n) <= q/m``  i.e.  ``n >= m/q - 1``.
    The smallest integer satisfying that is ``ceil(m/q) - 1`` (== ``ceil(m/q -
    1)``), which we return.

    This is the *bare feasibility floor*: at exactly this budget only a rank-1
    feature with ZERO null exceedances can clear BH, so statistical power is
    near zero. A caller sizing a real budget should use substantially more (the
    Layer-4 plan suggests ~1000); this value exists to detect the
    structurally-always-empty misconfiguration, NOT to recommend a budget. The
    previous ``ceil(m/q)`` was one too high — it called the exact-boundary
    budget (where rank-1 rejection IS possible) structurally empty.

    Worked example: m=40 features at FDR q=0.05 needs ``ceil(40/0.05) - 1 =
    799`` permutations for rank-1 to be reachable (the plus-one floor ``1/800``
    exactly equals ``q/m``); the legacy ``n_permutations=200`` ablation default
    could therefore *never* flag anything, regardless of how clear the leak was.

    Args:
        n_features: number of features (hypotheses) the BH step ranks over.
        q: target false-discovery rate, in (0, 1).

    Returns:
        ceil(n_features / q) - 1 (the exact feasibility floor); 0 when
        n_features <= 0.
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"q (target FDR) must be in (0, 1); got {q}")
    if n_features <= 0:
        return 0
    return math.ceil(n_features / q) - 1


def benjamini_hochberg(
    p_values: Sequence[float] | np.ndarray,
    q: float = 0.05,
    *,
    n_permutations: int | None = None,
) -> np.ndarray:
    """Benjamini-Hochberg FDR-controlled rejection mask (the BH step-up rule).

    Returns a boolean array aligned with the INPUT order of ``p_values``:
    ``True`` = reject the null for that feature (= a confident leak signal at
    false-discovery-rate <= ``q``).

    The step-up rule: sort p ascending, find the largest rank ``k`` with
    ``p_(k) <= (k/m) * q``, and reject ALL ranks ``1..k`` (even ranks whose own
    p-value exceeds their individual threshold — this is what distinguishes BH
    from naive per-test thresholding and is what controls the FDR).

    Why BH and not the more conservative Benjamini-Yekutieli (BY): BY divides
    the thresholds by ``H_m = sum(1/i)`` (~3.5 at m=40). Combined with the
    plus-one permutation floor ``1/(1+n)``, BY's rank-1 threshold
    ``q/(m*H_m)`` is unreachable for realistic feature counts and permutation
    budgets (it would demand ~5x more permutations), yielding an always-empty
    set. BH is the appropriate control here; the leak hypotheses are not
    adversarially anti-correlated.

    Input contract: p-values must be valid probabilities in ``(0, 1]``. Plus-one
    permutation p-values are NEVER exactly 0.0 (see ``compute_adversarial_score``
    / ``compute_feature_ablation``); a 0.0 — or any value outside ``(0, 1]`` — is
    rejected with ValueError rather than silently sorted to the top of the BH
    ranking and rejected, which would defeat the plus-one contract and let a
    stale empirical-zero sidecar corrupt the confident set. Non-finite entries
    (``NaN`` — e.g. an ablation on a degenerate feature) are permitted and
    treated as non-significant (never rejected).

    Args:
        p_values: per-feature permutation p-values (use the plus-one estimator
            from ``compute_adversarial_score`` / ``compute_feature_ablation`` so
            none is exactly 0.0).
        q: target false-discovery rate, in (0, 1).
        n_permutations: if given, the permutation budget that PRODUCED these
            p-values. Two checks then fire: (1) the function refuses to run
            (ValueError) when the budget is too small for ANY rejection to be
            possible (``n_permutations < min_permutations_for_fdr(m, q)``),
            surfacing the always-empty-set misconfiguration instead of silently
            returning all-False; and (2) every finite p-value is validated to be
            achievable from that budget (``>= 1/(1 + n_permutations)``) — a
            sub-floor value cannot have come from the plus-one estimator at this
            budget and is rejected as invalid input.

    Returns:
        Boolean ndarray (input-aligned). All-False when nothing clears BH.
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"q (target FDR) must be in (0, 1); got {q}")

    p = np.asarray(list(p_values), dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)

    # Validate the p-values. ONLY NaN (e.g. a degenerate ablation that could
    # not compute a delta_auc) is tolerated and treated as non-significant;
    # every other entry must be a genuine probability in (0, 1]. A 0.0, an
    # out-of-range value, or a +/-inf cannot be a plus-one permutation p-value
    # and would otherwise enter the BH ranking — a -inf in particular sorts
    # FIRST and would be wrongly rejected as a confident leak. Fail loud.
    nan_mask = np.isnan(p)
    invalid = ~nan_mask & (~np.isfinite(p) | (p <= 0.0) | (p > 1.0))
    if np.any(invalid):
        raise ValueError(
            "p_values must be valid probabilities in (0, 1] (NaN tolerated as "
            f"non-significant); got {p[invalid].tolist()}. Plus-one permutation "
            "p-values are never exactly 0.0, infinite, or out of range — such a "
            "value would be sorted into the BH ranking and wrongly rejected."
        )

    if n_permutations is not None:
        required = min_permutations_for_fdr(m, q)
        if n_permutations < required:
            raise ValueError(
                f"n_permutations={n_permutations} is too small for BH at q={q} "
                f"over m={m} features: the plus-one floor 1/(1+{n_permutations})="
                f"{1.0 / (1 + n_permutations):.3g} exceeds the BH rank-1 threshold "
                f"q/m={q / m:.3g}, so the confident set is structurally empty. "
                f"Need n_permutations >= {required}."
            )
        # Every non-NaN p-value (now guaranteed finite and in (0, 1] by the
        # check above) must be achievable from this budget; a sub-floor value
        # cannot have come from the plus-one estimator at n_permutations (e.g. a
        # stale empirical-zero sidecar, a fixture, or a mismatched-n caller) and
        # would corrupt the confident set. Fail loud.
        floor = 1.0 / (1 + n_permutations)
        below_floor = ~nan_mask & (p < floor - 1e-12)
        if np.any(below_floor):
            raise ValueError(
                f"p_values {p[below_floor].tolist()} are below the plus-one "
                f"floor 1/(1+{n_permutations})={floor:.3g} and so cannot have "
                "been produced by the plus-one estimator at this permutation "
                "budget; pass the n_permutations that actually produced them."
            )

    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    thresholds = (np.arange(1, m + 1) / m) * q
    below = sorted_p <= thresholds

    mask_sorted = np.zeros(m, dtype=bool)
    if below.any():
        k = int(np.nonzero(below)[0].max())  # largest 0-based rank that passes
        mask_sorted[: k + 1] = True

    mask = np.empty(m, dtype=bool)
    mask[order] = mask_sorted  # map sorted-order decisions back to input order
    return mask


def fdr_permutation_budget(
    n_features: int,
    q: float,
    *,
    default: int,
    cap: int,
) -> tuple[int, bool]:
    """Feasibility-aware permutation budget for the BH/FDR confident set.

    A plus-one permutation p-value can only resolve below the BH rank-1
    threshold ``q / m`` when ``n_permutations >= min_permutations_for_fdr(m,
    q)`` (the floor scales as ``ceil(m / q)`` — quadratic in feature count for a
    fixed q). This helper sizes the budget for that feasibility AND signals when
    a cohort is too wide to afford it within ``cap``, so the caller can fall
    back to the static σ-band instead of silently producing an always-empty
    confident set.

    Returns ``(n_permutations, feasible)``:
      * ``feasible=True``  → run BH at ``n_permutations`` — raised to the
        feasibility floor so a rejection is *possible*, but never lowered below
        ``default`` (which preserves z-score quality for narrow cohorts whose
        floor is tiny).
      * ``feasible=False`` → the floor exceeds ``cap``; FDR is infeasible at
        this width. Returns ``default`` (the budget for the σ-band's z-scores)
        and the caller MUST use the static σ-band, not the confident set.

    Args:
        n_features: number of features (BH hypotheses) to be ranked.
        q: target false-discovery rate, in (0, 1).
        default: the configured baseline permutation count (e.g. the legacy
            ``DEFAULT_PERMUTATIONS``); the floor never lowers the budget below
            this. ``cap`` is expected to be >= ``default``.
        cap: the maximum affordable permutation budget. When the feasibility
            floor exceeds it, FDR is declared infeasible (σ-band fallback).

    Returns:
        ``(n_permutations, feasible)``.
    """
    floor = min_permutations_for_fdr(n_features, q)
    if floor > cap:
        return default, False
    return max(floor, default), True


def fdr_confident_set(
    p_values: Sequence[float] | np.ndarray,
    effect_sizes: Sequence[float] | np.ndarray,
    *,
    q: float,
    n_permutations: int,
    effect_floor: float,
) -> np.ndarray:
    """Confident-leak mask = BH-rejection ∩ ``|effect| > effect_floor``.

    The FDR-controlled, cohort-size-adaptive replacement for the static σ-band's
    auto-fire (severity=high) tier. A feature is a *confident* leak only when
    BOTH conditions hold (input-aligned boolean ``True``):

      * its plus-one permutation p-value clears Benjamini-Hochberg at FDR ``q``
        (``benjamini_hochberg`` — statistical confidence that adapts to the
        cohort's null automatically, unlike a fixed z-threshold), AND
      * its absolute effect size exceeds ``effect_floor`` (the issue-#194
        pharma-actionable bar). A BH-significant feature with a tiny effect is
        the "ambiguous interior" — routed to review, NOT auto-dropped.

    ``NaN`` effect sizes (a degenerate score/ablation that could not compute a
    delta) are never confident. The ``n_permutations`` that produced
    ``p_values`` is forwarded to ``benjamini_hochberg`` so its feasibility +
    plus-one-floor guards fire (an always-empty / sub-floor misconfiguration
    raises rather than silently returning all-False).

    Args:
        p_values: per-feature plus-one permutation p-values, input-aligned with
            ``effect_sizes``.
        effect_sizes: per-feature signed effect size (e.g. ``actual_auc -
            null_mean`` for the marginal path, or ``delta_auc`` for ablation).
        q: target false-discovery rate, in (0, 1).
        n_permutations: the permutation budget that produced ``p_values``.
        effect_floor: minimum ``|effect|`` for a BH-significant feature to count
            as a confident leak (else it is the ambiguous interior → review).

    Returns:
        Boolean ndarray (input-aligned): ``True`` = confident leak.
    """
    # Alignment check FIRST — before any dtype coercion — so a length mismatch
    # raises the intended error rather than a confusing conversion error.
    p_list = list(p_values)
    eff_list = list(effect_sizes)
    if len(p_list) != len(eff_list):
        raise ValueError(
            "p_values and effect_sizes must be the same length (input-aligned); "
            f"got {len(p_list)} p-values and {len(eff_list)} effect sizes"
        )
    bh = benjamini_hochberg(p_list, q, n_permutations=n_permutations)
    eff = np.asarray(eff_list, dtype=float)
    meaningful = np.isfinite(eff) & (np.abs(eff) > float(effect_floor))
    # numpy's ``&`` operator is typed to return ``Any``; cast back to the
    # declared ndarray so the Any does not leak out (mypy no-any-return).
    return cast("np.ndarray", np.asarray(bh, dtype=bool) & meaningful)
