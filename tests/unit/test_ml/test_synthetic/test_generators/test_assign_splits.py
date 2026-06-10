"""Red-first tests for issue #864 — ``BaseGenerator._assign_splits`` must cut
chronological boundaries on ROW MASS, not unique-date count.

Under ``--anchor-to-now`` the date distribution is deliberately bimodal
(~60% of rows packed into the last ~30 unique days, ~40% spread over ~3
years). Unique-date quantile boundaries therefore map the dense recent
window into the holdout tail: measured on the 2026-06-10 FULL_SIZES
snapshot, holdout got 60.8% of rows and train 24.8% (designed 5% / 60%).

The fix cuts each boundary where the cumulative ROW share crosses the
ratio, keeping whole dates on one side so the chronological (leak-safe)
strategy is preserved under any date density.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter

import numpy as np
import pytest

from src.ml.synthetic.config import DGPType
from src.ml.synthetic.generators import GeneratorConfig, PatientGenerator

DESIGNED = {"train": 0.60, "validation": 0.20, "test": 0.15, "holdout": 0.05}


def _generator(**config_kwargs) -> PatientGenerator:
    """Concrete generator giving access to the shared base ``_assign_splits``."""
    config = GeneratorConfig(seed=42, n_records=10, dgp_type=DGPType.CONFOUNDED, **config_kwargs)
    return PatientGenerator(config)


def _shares(splits: list[str]) -> dict[str, float]:
    counts = Counter(splits)
    n = len(splits)
    return {k: v / n for k, v in counts.items()}


def _anchored_bimodal_dates(
    n: int = 10_000,
    recent_fraction: float = 0.6,
    recent_days: int = 30,
    span_days: int = 1065,
) -> list[str]:
    """Mimic ``_anchored_dates``: 60% of rows in the last 30 unique days,
    40% uniform across the ~3-year historical tail."""
    ref = dt.date(2026, 6, 10)
    rng = np.random.default_rng(42)
    n_recent = int(n * recent_fraction)
    recent = [
        (ref - dt.timedelta(days=int(d))).isoformat()
        for d in rng.integers(0, recent_days, n_recent)
    ]
    historical = [
        (ref - dt.timedelta(days=int(d))).isoformat()
        for d in rng.integers(recent_days, span_days, n - n_recent)
    ]
    return historical + recent


class TestAssignSplitsRowMass:
    def test_row_mass_shares_under_anchored_skew(self) -> None:
        """THE #864 case: bimodal anchored dates must still yield ~60/20/15/5
        ROW shares (the dense recent window holds ~2% of rows per unique day,
        so whole-date boundary rounding is bounded by ~0.02 per split)."""
        gen = _generator()
        dates = _anchored_bimodal_dates()
        shares = _shares(gen._assign_splits(dates))
        assert shares.get("train", 0.0) == pytest.approx(DESIGNED["train"], abs=0.05)
        assert shares.get("validation", 0.0) == pytest.approx(DESIGNED["validation"], abs=0.04)
        assert shares.get("test", 0.0) == pytest.approx(DESIGNED["test"], abs=0.04)
        assert shares.get("holdout", 0.0) == pytest.approx(DESIGNED["holdout"], abs=0.03)

    def test_uniform_dates_keep_designed_shares(self) -> None:
        """The legacy uniform-density case must keep working: ~60/20/15/5."""
        gen = _generator()
        rng = np.random.default_rng(7)
        start = dt.date(2022, 1, 1)
        dates = [
            (start + dt.timedelta(days=int(d))).isoformat() for d in rng.integers(0, 1095, 8000)
        ]
        shares = _shares(gen._assign_splits(dates))
        for split, designed in DESIGNED.items():
            assert shares.get(split, 0.0) == pytest.approx(designed, abs=0.03), split

    def test_chronological_order_preserved(self) -> None:
        """Whole dates stay on one side and split blocks are chronologically
        contiguous: every train date < every validation date < every test
        date < every holdout date (leak-safety of the temporal strategy)."""
        gen = _generator()
        dates = _anchored_bimodal_dates(n=5_000)
        splits = gen._assign_splits(dates)
        by_split: dict[str, list[str]] = {}
        for d, s in zip(dates, splits, strict=True):
            by_split.setdefault(s, []).append(d)
        order = ["train", "validation", "test", "holdout"]
        present = [s for s in order if s in by_split]
        for earlier, later in zip(present, present[1:], strict=False):
            assert max(by_split[earlier]) <= min(by_split[later]), (earlier, later)

    def test_only_boundary_dates_span_splits(self) -> None:
        """Non-boundary dates land whole on one side; at most 3 dates (the
        quota boundaries) may span splits, and any spanning date covers only
        ADJACENT splits (so the validator's strict temporal check passes)."""
        gen = _generator()
        dates = _anchored_bimodal_dates(n=3_000)
        date_to_splits: dict[str, set[str]] = {}
        for d, s in zip(dates, gen._assign_splits(dates), strict=True):
            date_to_splits.setdefault(d, set()).add(s)
        order = ["train", "validation", "test", "holdout"]
        spanning = {d: v for d, v in date_to_splits.items() if len(v) > 1}
        assert len(spanning) <= 3
        for d, v in spanning.items():
            idxs = sorted(order.index(s) for s in v)
            assert idxs == list(range(idxs[0], idxs[-1] + 1)), (d, v)

    def test_cap_point_mass_still_yields_designed_shares(self) -> None:
        """The anchor-cap artifact: derived tables collapse the future tail
        onto ONE reference date (measured 40.6% of treatment_events rows on
        2026-06-10). That boundary date must CHUNK across adjacent splits so
        the designed row shares survive — no whole-date scheme can place a
        40% point mass without destroying the ratios."""
        gen = _generator()
        rng = np.random.default_rng(11)
        ref = dt.date(2026, 6, 10)
        historical = [
            (ref - dt.timedelta(days=int(d))).isoformat() for d in rng.integers(1, 1065, 6_000)
        ]
        dates = historical + [ref.isoformat()] * 4_000  # 40% on the cap date
        shares = _shares(gen._assign_splits(dates))
        for split, designed in DESIGNED.items():
            assert shares.get(split, 0.0) == pytest.approx(designed, abs=0.02), split

    def test_single_unique_date_chunks_to_designed_shares(self) -> None:
        """Degenerate single-date input chunks to the designed ratios (the
        old unique-date boundary math put ALL rows in HOLDOUT: int(1*0.6)=0
        → i=0 never < 0). Chronology is meaningless within one date."""
        gen = _generator()
        splits = gen._assign_splits(["2026-06-01"] * 100)
        shares = _shares(splits)
        for split, designed in DESIGNED.items():
            assert shares.get(split, 0.0) == pytest.approx(designed, abs=0.01), split

    def test_empty_input_returns_empty(self) -> None:
        gen = _generator()
        assert gen._assign_splits([]) == []


class TestPatientGeneratorAnchoredSplits:
    def test_anchor_to_now_split_shares(self) -> None:
        """End-to-end through the generator: with ``anchor_to_now=True`` the
        emitted ``data_split`` row shares must approximate the designed
        ratios — the exact configuration that produced the scrambled
        (train 25% / holdout 61%) 2026-06-10 snapshot."""
        config = GeneratorConfig(
            seed=42, n_records=4_000, dgp_type=DGPType.CONFOUNDED, anchor_to_now=True
        )
        gen = PatientGenerator(config)
        df = gen.generate()
        dist = df["data_split"].value_counts(normalize=True)
        assert dist.get("train", 0.0) == pytest.approx(0.60, abs=0.07)
        assert dist.get("validation", 0.0) == pytest.approx(0.20, abs=0.05)
        assert dist.get("test", 0.0) == pytest.approx(0.15, abs=0.05)
        assert dist.get("holdout", 0.0) == pytest.approx(0.05, abs=0.04)
