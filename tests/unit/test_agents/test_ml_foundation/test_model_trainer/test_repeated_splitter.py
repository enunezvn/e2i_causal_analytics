"""W3-lite Day 4 (shard 17 W3 row Day 4) — RepeatedStratifiedSplitter unit tests.

Spec: shard 21 §A (RepeatedStratifiedSplitter design + Public API + Stratification target +
Seed derivation contract) and §G.1 (Determinism tests). The splitter is a deterministic,
k=10 protocol wrapper around ``StratifiedShuffleSplit`` with explicit train/val/test
materialization (15% val materialized via second-level stratified split, since sklearn's
shuffle splitter only emits 2-way splits).

Day-3 contract dependency: per-fold seed must satisfy the Day-3 ``resolve_fold_random_state``
zero-as-valid contract (locked by ``test_zero_is_a_valid_seed`` in test_fold_random_state.py).
The splitter's ``_derive_seed`` may emit zero, and the orchestrator threads that value into
state[``fold_random_state``] without reinterpreting it as "unset".

Form per Q-W3-4 RESOLVED 2026-05-01 (cycle 3 + cleanup):
``SeedSequence((fold_idx, seed_base)).generate_state(1)[0]`` — vary first, root last
(numpy parallel-RNG canonical idiom); no ``.spawn(1)[0]`` after 2026-05-01 user-decided
cleanup pass for cross-layer symmetry with outer Q-W4-3 helper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Public API import + FoldSpec dataclass shape
# ---------------------------------------------------------------------------


class TestPublicAPI:
    """Module-level import + class shape per shard 21 §A."""

    def test_module_importable(self) -> None:
        from src.agents.ml_foundation.model_trainer.splitting import (  # noqa: F401
            FoldSpec,
            RepeatedStratifiedSplitter,
        )

    def test_fold_spec_fields_present(self) -> None:
        from src.agents.ml_foundation.model_trainer.splitting import FoldSpec

        spec = FoldSpec(
            fold_idx=0,
            seed=12345,
            train_idx=np.array([0, 1, 2]),
            val_idx=np.array([3]),
            test_idx=np.array([4]),
        )
        assert spec.fold_idx == 0
        assert spec.seed == 12345
        assert spec.train_idx.tolist() == [0, 1, 2]
        assert spec.val_idx.tolist() == [3]
        assert spec.test_idx.tolist() == [4]

    def test_fold_spec_label_format(self) -> None:
        from src.agents.ml_foundation.model_trainer.splitting import FoldSpec

        spec = FoldSpec(
            fold_idx=7,
            seed=42,
            train_idx=np.array([]),
            val_idx=np.array([]),
            test_idx=np.array([]),
        )
        assert spec.fold_label == "fold_07"


# ---------------------------------------------------------------------------
# Helper to build a deterministic stratified dataset
# ---------------------------------------------------------------------------


def _make_synthetic_dataset(
    n: int = 200, n_features: int = 4, prevalence: float = 0.30, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Build a binary-classification dataset with known prevalence."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n, n_features)),
        columns=[f"x{i}" for i in range(n_features)],
    )
    n_positive = int(round(n * prevalence))
    y = np.zeros(n, dtype=int)
    positive_idx = rng.choice(n, size=n_positive, replace=False)
    y[positive_idx] = 1
    return X, pd.Series(y, name="y")


# ---------------------------------------------------------------------------
# G.1 — Determinism tests (shard 21 §G.1)
# ---------------------------------------------------------------------------


class TestSplitterDeterminism:
    """Per shard 21 §G.1 — splitter must be deterministic given (seed_base, fold_idx)."""

    def test_splitter_deterministic_across_runs(self) -> None:
        from src.agents.ml_foundation.model_trainer.splitting import (
            RepeatedStratifiedSplitter,
        )

        X, y = _make_synthetic_dataset()
        s1 = RepeatedStratifiedSplitter(seed_base=42)
        s2 = RepeatedStratifiedSplitter(seed_base=42)
        folds_1 = list(s1.split(X, y))
        folds_2 = list(s2.split(X, y))
        assert len(folds_1) == len(folds_2) == 10
        for f1, f2 in zip(folds_1, folds_2, strict=True):
            assert f1.fold_idx == f2.fold_idx
            assert f1.seed == f2.seed
            np.testing.assert_array_equal(f1.train_idx, f2.train_idx)
            np.testing.assert_array_equal(f1.val_idx, f2.val_idx)
            np.testing.assert_array_equal(f1.test_idx, f2.test_idx)

    def test_splitter_seed_base_changes_all_folds(self) -> None:
        from src.agents.ml_foundation.model_trainer.splitting import (
            RepeatedStratifiedSplitter,
        )

        X, y = _make_synthetic_dataset()
        folds_42 = list(RepeatedStratifiedSplitter(seed_base=42).split(X, y))
        folds_43 = list(RepeatedStratifiedSplitter(seed_base=43).split(X, y))
        # Bumping seed_base must change every fold's seed (not just fold 0).
        seeds_42 = {f.seed for f in folds_42}
        seeds_43 = {f.seed for f in folds_43}
        assert seeds_42.isdisjoint(seeds_43)

    def test_splitter_per_fold_seeds_unique(self) -> None:
        from src.agents.ml_foundation.model_trainer.splitting import (
            RepeatedStratifiedSplitter,
        )

        X, y = _make_synthetic_dataset()
        folds = list(RepeatedStratifiedSplitter(seed_base=42).split(X, y))
        seeds = [f.seed for f in folds]
        assert len(set(seeds)) == 10, f"Fold seeds collided: {seeds}"

    def test_splitter_yields_exactly_k_folds(self) -> None:
        from src.agents.ml_foundation.model_trainer.splitting import (
            RepeatedStratifiedSplitter,
        )

        X, y = _make_synthetic_dataset()
        folds = list(RepeatedStratifiedSplitter(k=10, seed_base=42).split(X, y))
        assert len(folds) == 10
        for i, f in enumerate(folds):
            assert f.fold_idx == i


# ---------------------------------------------------------------------------
# G.1 — Index disjointness + coverage + stratification
# ---------------------------------------------------------------------------


class TestSplitterIndexInvariants:
    """Per shard 21 §G.1 — train/val/test indices disjoint + stratification preserved."""

    def test_splitter_index_disjointness(self) -> None:
        from src.agents.ml_foundation.model_trainer.splitting import (
            RepeatedStratifiedSplitter,
        )

        X, y = _make_synthetic_dataset(n=200, prevalence=0.30)
        for spec in RepeatedStratifiedSplitter(seed_base=42).split(X, y):
            train_set = set(spec.train_idx.tolist())
            val_set = set(spec.val_idx.tolist())
            test_set = set(spec.test_idx.tolist())
            assert train_set.isdisjoint(val_set), f"fold {spec.fold_idx} train ∩ val nonempty"
            assert train_set.isdisjoint(test_set), f"fold {spec.fold_idx} train ∩ test nonempty"
            assert val_set.isdisjoint(test_set), f"fold {spec.fold_idx} val ∩ test nonempty"

    def test_splitter_index_total_coverage_of_each_fold(self) -> None:
        """Each fold's union of indices should cover N rows (shuffle_split with 70/15/15)."""
        from src.agents.ml_foundation.model_trainer.splitting import (
            RepeatedStratifiedSplitter,
        )

        X, y = _make_synthetic_dataset(n=200, prevalence=0.30)
        for spec in RepeatedStratifiedSplitter(seed_base=42).split(X, y):
            covered = (
                set(spec.train_idx.tolist())
                | set(spec.val_idx.tolist())
                | set(spec.test_idx.tolist())
            )
            assert covered == set(range(len(X))), (
                f"fold {spec.fold_idx} did not cover all N rows: missing "
                f"{set(range(len(X))) - covered}"
            )

    def test_splitter_stratification_preserved(self) -> None:
        """Per-fold test set prevalence within ±0.03 of population prevalence."""
        from src.agents.ml_foundation.model_trainer.splitting import (
            RepeatedStratifiedSplitter,
        )

        prevalence = 0.30
        X, y = _make_synthetic_dataset(n=400, prevalence=prevalence)
        for spec in RepeatedStratifiedSplitter(seed_base=42).split(X, y):
            test_prev = float(y.iloc[spec.test_idx].mean())
            assert abs(test_prev - prevalence) < 0.03, (
                f"fold {spec.fold_idx} test prevalence {test_prev:.3f} drifted "
                f"from population {prevalence:.3f}"
            )


# ---------------------------------------------------------------------------
# Seed derivation contract (Q-W3-4 RESOLVED form)
# ---------------------------------------------------------------------------


class TestSeedDerivationCanonicalForm:
    """_derive_seed must use SeedSequence((fold_idx, seed_base)).generate_state(1)[0]."""

    def test_derive_seed_matches_canonical_idiom(self) -> None:
        from src.agents.ml_foundation.model_trainer.splitting import (
            RepeatedStratifiedSplitter,
        )

        for fold_idx in range(10):
            expected = int(np.random.SeedSequence((fold_idx, 42)).generate_state(1)[0])
            assert RepeatedStratifiedSplitter._derive_seed(42, fold_idx) == expected, (
                f"fold {fold_idx} seed mismatch: expected canonical "
                f"SeedSequence((fold_idx, seed_base)).generate_state(1)[0]"
            )

    def test_derive_seed_argument_order_is_canonical_vary_first(self) -> None:
        """seed=(fold_idx, seed_base) NOT (seed_base, fold_idx).

        Codex cycle-3 verdict B + cycle-4 endorsement: numpy parallel-RNG
        canonical idiom is "vary IDs first, root seed last."
        """
        from src.agents.ml_foundation.model_trainer.splitting import (
            RepeatedStratifiedSplitter,
        )

        # If implementation reversed the order, fold_idx=0 + seed_base=42 would
        # equal fold_idx=42 + seed_base=0 (same tuple). Prove it doesn't.
        a = RepeatedStratifiedSplitter._derive_seed(42, 0)
        b = RepeatedStratifiedSplitter._derive_seed(0, 42)
        assert a != b, (
            "Argument order looks reversed: (42, 0) and (0, 42) produced the "
            "same seed, which would only happen if the canonical idiom was inverted."
        )

    def test_derive_seed_is_deterministic(self) -> None:
        from src.agents.ml_foundation.model_trainer.splitting import (
            RepeatedStratifiedSplitter,
        )

        for seed_base, fold_idx in [(42, 0), (42, 9), (0, 0), (0, 9)]:
            calls = [RepeatedStratifiedSplitter._derive_seed(seed_base, fold_idx) for _ in range(5)]
            assert len(set(calls)) == 1, f"Non-deterministic at ({seed_base}, {fold_idx})"

    def test_derive_seed_with_zero_seed_base(self) -> None:
        """Seed_base=0 must NOT short-circuit to fallback (Day-3 zero-as-valid contract).

        Day-3 ``resolve_fold_random_state`` treats 0 as a valid seed (locked by
        ``test_zero_is_a_valid_seed``). The splitter MUST yield deterministic, non-equal
        per-fold seeds when seed_base=0.
        """
        from src.agents.ml_foundation.model_trainer.splitting import (
            RepeatedStratifiedSplitter,
        )

        seeds = [RepeatedStratifiedSplitter._derive_seed(0, fold_idx) for fold_idx in range(10)]
        assert len(set(seeds)) == 10, f"seed_base=0 produced colliding fold seeds: {seeds}"

    def test_inner_seed_canonical_form(self) -> None:
        """Cycle-15 I-1 (codex): _derive_inner_seed uses canonical SeedSequence idiom.

        Replaces the original ad-hoc ``(fold_seed + 1) % 2**32`` arithmetic offset
        with a second SeedSequence call ``SeedSequence((fold_idx + 1000, seed_base))``
        — compositionally symmetric with the outer Q-W3-4 form.
        """
        from src.agents.ml_foundation.model_trainer.splitting import (
            RepeatedStratifiedSplitter,
        )

        for fold_idx in range(10):
            expected = int(np.random.SeedSequence((fold_idx + 1000, 42)).generate_state(1)[0])
            assert RepeatedStratifiedSplitter._derive_inner_seed(42, fold_idx) == expected

    def test_inner_seed_distinct_from_outer_seed(self) -> None:
        """Cycle-15 I-1 (codex): inner_seed must differ from outer_seed for every fold.

        The inner stratified split (val-vs-train materialization) needs its own
        deterministic seed distinct from the outer (test-vs-rest) seed so the val
        draw within a fold cannot spuriously correlate with the outer test draw.
        """
        from src.agents.ml_foundation.model_trainer.splitting import (
            RepeatedStratifiedSplitter,
        )

        for seed_base in (0, 42, 43, 12345, 2**31 - 1):
            outer = [RepeatedStratifiedSplitter._derive_seed(seed_base, i) for i in range(10)]
            inner = [RepeatedStratifiedSplitter._derive_inner_seed(seed_base, i) for i in range(10)]
            assert set(outer).isdisjoint(set(inner)), (
                f"outer/inner seed collision at seed_base={seed_base}: outer={outer}, inner={inner}"
            )

    def test_inner_seed_unique_across_folds(self) -> None:
        """Inner seeds must also be distinct across folds (mirrors outer guarantee)."""
        from src.agents.ml_foundation.model_trainer.splitting import (
            RepeatedStratifiedSplitter,
        )

        inner_seeds = [RepeatedStratifiedSplitter._derive_inner_seed(42, i) for i in range(10)]
        assert len(set(inner_seeds)) == 10, f"inner seeds collided across folds: {inner_seeds}"


# ---------------------------------------------------------------------------
# Configurability + invariants
# ---------------------------------------------------------------------------


class TestSplitterConfig:
    """Configurable k, train/val/test fractions, validation."""

    def test_default_fractions_are_70_15_15(self) -> None:
        from src.agents.ml_foundation.model_trainer.splitting import (
            RepeatedStratifiedSplitter,
        )

        s = RepeatedStratifiedSplitter()
        assert s.train_frac == 0.70
        assert s.val_frac == 0.15
        assert s.test_frac == 0.15

    def test_default_k_is_10(self) -> None:
        from src.agents.ml_foundation.model_trainer.splitting import (
            RepeatedStratifiedSplitter,
        )

        assert RepeatedStratifiedSplitter().k == 10

    def test_default_seed_base_is_42(self) -> None:
        from src.agents.ml_foundation.model_trainer.splitting import (
            RepeatedStratifiedSplitter,
        )

        assert RepeatedStratifiedSplitter().seed_base == 42

    def test_invalid_fraction_sum_raises(self) -> None:
        from src.agents.ml_foundation.model_trainer.splitting import (
            RepeatedStratifiedSplitter,
        )

        with pytest.raises(ValueError, match="must sum to 1.0"):
            RepeatedStratifiedSplitter(train_frac=0.7, val_frac=0.2, test_frac=0.2)

    def test_k_less_than_2_raises(self) -> None:
        from src.agents.ml_foundation.model_trainer.splitting import (
            RepeatedStratifiedSplitter,
        )

        with pytest.raises(ValueError, match="k must be >= 2"):
            RepeatedStratifiedSplitter(k=1)

    def test_unknown_strategy_raises(self) -> None:
        from src.agents.ml_foundation.model_trainer.splitting import (
            RepeatedStratifiedSplitter,
        )

        with pytest.raises(ValueError, match="strategy"):
            RepeatedStratifiedSplitter(strategy="invalid_strategy")

    def test_configurable_k(self) -> None:
        from src.agents.ml_foundation.model_trainer.splitting import (
            RepeatedStratifiedSplitter,
        )

        X, y = _make_synthetic_dataset()
        folds = list(RepeatedStratifiedSplitter(k=5).split(X, y))
        assert len(folds) == 5
