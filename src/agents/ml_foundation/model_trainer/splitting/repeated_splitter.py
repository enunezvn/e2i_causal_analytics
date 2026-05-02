"""RepeatedStratifiedSplitter — k=10 repeated train/val/test splits.

Phase 1 W3-lite Day 4 (shard 17 W3 row Day 4, shard 21 §A).

Wraps ``sklearn.model_selection.StratifiedShuffleSplit`` and materializes a
3-way (train/val/test) partition per fold. ``StratifiedShuffleSplit`` only
produces 2-way splits; the wrapper performs a second-level stratified split
on the train block to materialize the validation indices explicitly.

Per Q-W3-1 RESOLVED 2026-05-01 (cycle 2 codex), default
``strategy="shuffle_split"``: each fold is an independent stratified
resampling draw (rows reused across folds — bootstrap-like), with explicit
70/15/15 train/val/test partitions per draw. The estimand is "frozen-config
performance variability over k stratified resampling draws conditional on
the observed Phase 1 dataset" — NOT a calibrated generalization-error CI.

Per Q-W3-4 RESOLVED 2026-05-01 (cycle 3 + 2026-05-01 cleanup), per-fold
seeds derive via ``SeedSequence((fold_idx, seed_base)).generate_state(1)[0]``
("vary first, root last" per numpy parallel-RNG canonical idiom; no
``.spawn(1)[0]`` after 2026-05-01 user-decided cleanup pass for cross-layer
symmetry with outer Q-W4-3 helper). Treats 0 as a valid seed (Day-3 contract
locked by ``test_zero_is_a_valid_seed`` in ``test_fold_random_state.py``).

Cross-environment reproducibility caveat (NEP 19): exact replay requires
pinned NumPy and sklearn versions; SeedSequence stream compatibility holds
only under same numpy build. Reviewers / replay tooling MUST log
``numpy.__version__`` and ``sklearn.__version__`` alongside the per-fold seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

__all__ = ["FoldSpec", "RepeatedStratifiedSplitter"]


_VALID_STRATEGIES = ("shuffle_split", "kfold_repeat")


@dataclass(frozen=True)
class FoldSpec:
    """One fold's deterministic identifiers + indices into the raw dataset.

    ``train_idx`` / ``val_idx`` / ``test_idx`` are positional indices into the
    original ``X`` / ``y`` arrays (use ``X.iloc[spec.train_idx]`` for a
    ``DataFrame``). All three sets are pairwise-disjoint per fold; their
    union covers ``range(N)`` for ``strategy="shuffle_split"`` (since the
    splitter materializes all rows into one of the three partitions per
    draw).
    """

    fold_idx: int
    seed: int
    train_idx: np.ndarray = field(compare=False)
    val_idx: np.ndarray = field(compare=False)
    test_idx: np.ndarray = field(compare=False)

    @property
    def fold_label(self) -> str:
        return f"fold_{self.fold_idx:02d}"


class RepeatedStratifiedSplitter:
    """k=10 repeated stratified train/val/test splitter.

    See module docstring for design rationale + Q-W3-1/Q-W3-4 dispositions.
    """

    def __init__(
        self,
        k: int = 10,
        seed_base: int = 42,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        strategy: str = "shuffle_split",
    ) -> None:
        if k < 2:
            raise ValueError(f"k must be >= 2 (got k={k})")
        if not isclose(train_frac + val_frac + test_frac, 1.0, abs_tol=1e-9):
            raise ValueError(
                "train_frac + val_frac + test_frac must sum to 1.0 "
                f"(got {train_frac} + {val_frac} + {test_frac} "
                f"= {train_frac + val_frac + test_frac})"
            )
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy={strategy!r}; valid: {_VALID_STRATEGIES}"
            )
        self.k = k
        self.seed_base = seed_base
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.strategy = strategy

    @staticmethod
    def _derive_seed(seed_base: int, fold_idx: int) -> int:
        """Per-fold 32-bit seed derivation (Q-W3-4 RESOLVED canonical form).

        ``SeedSequence((fold_idx, seed_base)).generate_state(1)[0]``:
        argument order "vary first (fold_idx), root last (seed_base)" per
        numpy canonical idiom; ``.generate_state(1)[0]`` produces a 32-bit
        seed for sklearn's ``random_state=int`` consumption.

        Pigeonhole-collision probability across all 45 fold pairs at k=10 is
        ~1e-8 (acceptable at Phase 1 scale; documented as a known limit at
        the 32-bit sklearn boundary). May emit zero — consumers MUST honor
        Day-3 zero-as-valid contract (``resolve_fold_random_state``).
        """
        return int(np.random.SeedSequence((fold_idx, seed_base)).generate_state(1)[0])

    @staticmethod
    def _derive_inner_seed(seed_base: int, fold_idx: int) -> int:
        """Second-level (val/train) seed for the per-fold inner stratified split.

        The inner split materializes the 15% validation partition out of the
        70%+15% "train+val rest" block (since ``StratifiedShuffleSplit`` only
        emits 2-way splits). It needs its own deterministic seed distinct
        from the outer seed so that the val-vs-train draw within a fold
        cannot spuriously correlate with the outer test-vs-rest draw.

        Cycle-15 codex review (I-1) flagged the original arithmetic offset
        ``(fold_seed + 1) % 2**32`` as ad-hoc and unspecified in shard 21.
        Canonical fix: derive the inner seed via a second ``SeedSequence`` call
        with a disjoint entropy tuple ``(fold_idx + 1000, seed_base)``. Both
        the outer and inner derivations now use the same numpy SeedSequence
        idiom (compositionally symmetric with the outer Q-W3-4 form), and the
        ``+ 1000`` offset moves the inner seed into a different region of the
        SeedSequence entropy tree, guaranteeing inner ≠ outer for any
        (seed_base, fold_idx) pair within k < 1000.

        Locked by ``test_inner_seed_distinct_from_outer_seed`` and
        ``test_inner_seed_canonical_form`` in ``test_repeated_splitter.py``.
        """
        return int(
            np.random.SeedSequence((fold_idx + 1000, seed_base)).generate_state(1)[0]
        )

    def split(
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> Iterator[FoldSpec]:
        """Yield k FoldSpec objects, deterministic given (seed_base, fold_idx).

        Algorithm (strategy="shuffle_split"):
          1. Derive per-fold seed from (fold_idx, seed_base).
          2. Outer split: stratified shuffle into (test_frac vs rest) using
             the derived seed.
          3. Inner split: stratified shuffle of "rest" into (val vs train)
             using a deterministic offset of the same fold seed (so the
             second-level split is reproducible alongside the outer split).
          4. Emit FoldSpec(fold_idx, seed, train_idx, val_idx, test_idx).

        For ``strategy="kfold_repeat"`` (Phase 2 reservation), this method
        raises NotImplementedError until the W3-full nested-CV path lands.
        """
        if self.strategy == "kfold_repeat":
            raise NotImplementedError(
                "strategy='kfold_repeat' reserved for Phase 2 nested-CV; "
                "Phase 1 ships only 'shuffle_split'."
            )

        # Coerce to a length-N target for stratification while preserving X/y typing.
        n = len(X)
        y_arr = np.asarray(y)
        all_indices = np.arange(n, dtype=np.int64)

        # Re-normalize fractions: relative_val_frac = val_frac / (1 - test_frac)
        # so that the inner split allocates val_frac of the total (not val_frac of rest).
        # This makes the 70/15/15 layout exact regardless of upstream rounding.
        relative_val_frac = self.val_frac / (self.train_frac + self.val_frac)

        for fold_idx in range(self.k):
            fold_seed = self._derive_seed(self.seed_base, fold_idx)

            outer = StratifiedShuffleSplit(
                n_splits=1,
                test_size=self.test_frac,
                random_state=fold_seed,
            )
            ((rest_idx, test_idx),) = outer.split(all_indices, y_arr)

            # Inner split on the "rest" block: relative val fraction over (train + val).
            # Use a SECOND SeedSequence derivation (canonical form, cycle-15 I-1 fix —
            # see _derive_inner_seed docstring) so the inner draw is reproducible AND
            # disjoint from the outer draw via the entropy tree.
            inner_seed = self._derive_inner_seed(self.seed_base, fold_idx)
            inner = StratifiedShuffleSplit(
                n_splits=1,
                test_size=relative_val_frac,
                random_state=inner_seed,
            )
            rest_indices_local = np.arange(len(rest_idx))
            ((train_local, val_local),) = inner.split(rest_indices_local, y_arr[rest_idx])

            train_idx = rest_idx[train_local].astype(np.int64)
            val_idx = rest_idx[val_local].astype(np.int64)
            test_idx = test_idx.astype(np.int64)

            yield FoldSpec(
                fold_idx=fold_idx,
                seed=fold_seed,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
            )
