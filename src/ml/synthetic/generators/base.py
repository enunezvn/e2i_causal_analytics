"""
Base Generator for Synthetic Data.

Provides common functionality for all entity generators.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Generic, Iterator, List, Optional, TypeVar

import numpy as np
import pandas as pd

from ..config import Brand, DGPType, SyntheticDataConfig

T = TypeVar("T", bound=pd.DataFrame)


@dataclass
class GeneratorConfig:
    """Configuration for a data generator."""

    seed: int = 42
    batch_size: int = 1000
    n_records: int = 1000
    brand: Optional[Brand] = None
    dgp_type: Optional[DGPType] = None
    start_date: date = field(default_factory=lambda: date(2022, 1, 1))
    end_date: date = field(default_factory=lambda: date(2024, 12, 31))
    # Rolling-window anchoring (Shard 04): when True, _random_dates remaps the
    # [start_date,end_date] span onto a window ending at anchor_reference (or
    # today). Defeats migration-044 NOW()-30d staleness; regenerated per run
    # because the reference is resolved at call time.
    anchor_to_now: bool = False
    anchor_reference: Optional[date] = None
    # Share of records biased into the last 30 days of the rolling window so
    # NOW()-30d windowed KPIs read non-zero.
    anchor_recent_fraction: float = 0.6
    verbose: bool = False
    # Namespacing prefix prepended to every generated entity id. Keeps a synthetic
    # validation dataset's ids DISJOINT from the existing dev baseline so the loader's
    # UPSERT cannot clobber pre-existing rows (and cleanup by is_synthetic is FK-safe).
    # Must keep ids within varchar(20): longest is patient_journey_id 'patient_000000'
    # (14) -> a <=3-char prefix is safe.
    id_prefix: str = ""


@dataclass
class GenerationResult:
    """Result of data generation."""

    df: pd.DataFrame
    entity_type: str
    n_records: int
    generation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if generation produced valid data."""
        return self.df is not None and len(self.df) == self.n_records


class BaseGenerator(ABC, Generic[T]):
    """
    Abstract base class for synthetic data generators.

    All entity generators inherit from this class and implement
    the abstract generate() method.
    """

    def __init__(self, config: Optional[GeneratorConfig] = None):
        """
        Initialize the generator.

        Args:
            config: Generator configuration. Uses defaults if not provided.
        """
        self.config = config or GeneratorConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._master_config = SyntheticDataConfig()

    @property
    @abstractmethod
    def entity_type(self) -> str:
        """Return the entity type being generated."""
        pass

    @abstractmethod
    def generate(self) -> pd.DataFrame:
        """
        Generate synthetic data.

        Returns:
            DataFrame containing generated records.
        """
        pass

    def generate_batched(self) -> Iterator[pd.DataFrame]:
        """
        Generate data in batches for memory efficiency.

        Yields:
            DataFrames of batch_size records each.
        """
        total_records = self.config.n_records
        batch_size = self.config.batch_size
        generated = 0

        while generated < total_records:
            remaining = total_records - generated
            current_batch = min(batch_size, remaining)

            # Create a new config with updated n_records
            batch_config = GeneratorConfig(
                seed=self.config.seed + generated,  # Vary seed per batch
                batch_size=current_batch,
                n_records=current_batch,
                brand=self.config.brand,
                dgp_type=self.config.dgp_type,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                anchor_to_now=self.config.anchor_to_now,
                anchor_reference=self.config.anchor_reference,
                anchor_recent_fraction=self.config.anchor_recent_fraction,
                verbose=self.config.verbose,
                id_prefix=self.config.id_prefix,
            )

            # Create new generator instance for batch
            batch_generator = self.__class__(batch_config)
            yield batch_generator.generate()

            generated += current_batch

    def generate_with_result(self) -> GenerationResult:
        """
        Generate data and return with metadata.

        Returns:
            GenerationResult with data and generation info.
        """
        import time

        start_time = time.time()
        df = self.generate()
        elapsed = time.time() - start_time

        return GenerationResult(
            df=df,
            entity_type=self.entity_type,
            n_records=len(df),
            generation_time=elapsed,
            metadata={
                "seed": self.config.seed,
                "brand": self.config.brand.value if self.config.brand else None,
                "dgp_type": self.config.dgp_type.value if self.config.dgp_type else None,
            },
        )

    def _generate_ids(self, prefix: str, n: int, width: int = 5) -> List[str]:
        """Generate sequential IDs with prefix."""
        return [f"{self.config.id_prefix}{prefix}_{i:0{width}d}" for i in range(n)]

    def _random_choice(
        self,
        options: List[Any],
        n: int,
        p: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Generate random choices from options."""
        return self._rng.choice(options, size=n, p=p)

    def _random_int(self, low: int, high: int, n: int) -> np.ndarray:
        """Generate random integers in range [low, high)."""
        return self._rng.integers(low, high, size=n)

    def _random_float(self, low: float, high: float, n: int) -> np.ndarray:
        """Generate random floats in range [low, high)."""
        return self._rng.uniform(low, high, n)

    def _random_normal(
        self,
        mean: float,
        std: float,
        n: int,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
    ) -> np.ndarray:
        """Generate random normal values with optional clipping."""
        values = self._rng.normal(mean, std, n)
        if clip_min is not None or clip_max is not None:
            values = np.clip(values, clip_min, clip_max)
        return values

    def _random_dates(
        self,
        n: int,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[str]:
        """Generate random dates as ISO strings.

        With config.anchor_to_now, the configured span is remapped onto a
        rolling window ending at anchor_reference (or today), with
        anchor_recent_fraction of rows biased into the last 30 days so
        NOW()-30d windowed KPIs read non-zero. Regenerated per run because the
        reference is resolved at call time. Default off = legacy [start,end] span.
        """
        start = start_date or self.config.start_date
        end = end_date or self.config.end_date

        if self.config.anchor_to_now:
            return self._anchored_dates(n)

        start_ord = start.toordinal()
        end_ord = end.toordinal()

        random_days = self._rng.integers(start_ord, end_ord + 1, size=n)
        return [date.fromordinal(d).isoformat() for d in random_days]

    def _anchored_dates(self, n: int) -> List[str]:
        """Rolling window ending at the run-time reference date.

        anchor_recent_fraction of rows land uniformly in (ref-30d, ref]; the rest
        spread across the historical tail [ref-span, ref-30d]. The reference is
        resolved here (anchor_reference or date.today()) so each run regenerates.
        """
        ref = self.config.anchor_reference or date.today()
        span_days = (self.config.end_date - self.config.start_date).days
        span_days = max(span_days, 90)  # floor so a tiny span still spreads
        ref_ord = ref.toordinal()
        recent_floor = ref_ord - 30

        frac = float(np.clip(self.config.anchor_recent_fraction, 0.0, 1.0))
        is_recent = self._rng.random(n) < frac
        n_recent = int(is_recent.sum())
        out = np.empty(n, dtype=np.int64)
        # Recent rows: uniformly in (NOW()-30d, NOW()]
        out[is_recent] = self._rng.integers(recent_floor, ref_ord + 1, size=n_recent)
        # Older rows: uniformly across the historical tail of the window
        n_old = n - n_recent
        out[~is_recent] = self._rng.integers(ref_ord - span_days, recent_floor, size=n_old)
        return [date.fromordinal(int(d)).isoformat() for d in out]

    def _shift_dates_to_window(self, dates: List[str]) -> List[str]:
        """Cap each ISO date/datetime string at the rolling-window reference (today)
        when anchoring is on, so dates DERIVED from anchored source dates
        (e.g. treatment_date = journey + offset) carry no future timestamps while
        KEEPING the recency the source already has — the future tail collapses onto
        the reference, the rest is left untouched (so the 60%-recent mixture from
        _anchored_dates survives; an affine rescale would flatten it). Handles
        'YYYY-MM-DD' and 'YYYY-MM-DD HH:MM:SS'. No-op when anchor_to_now is off or
        the list is empty. Reused by Shards 05/06/09."""
        if not self.config.anchor_to_now or not dates:
            return dates
        ref = self.config.anchor_reference or date.today()
        out: List[str] = []
        for d in dates:
            if date.fromisoformat(d[:10]) > ref:
                out.append(ref.isoformat() + d[10:])  # preserve any ' HH:MM:SS' tail
            else:
                out.append(d)
        return out

    def _anchor_cap_timestamp(self, ts: "pd.Timestamp") -> "pd.Timestamp":
        """Cap a single derived pandas Timestamp at the rolling-window reference
        (end of the reference day) when anchoring is on; no-op when off. Per-element
        so it composes with per-record generation (prediction/trigger records derive
        their timestamp from the anchored journey date + an offset that could land in
        the future). Preserves the source recency; only the future tail is collapsed."""
        if not self.config.anchor_to_now:
            return ts
        ref = self.config.anchor_reference or date.today()
        # Cap at the START of the reference day (midnight). Derived timestamps are
        # midnight-based (journey date + whole-day offset), so this keeps capped rows
        # at <= today-midnight <= NOW() for any consumer that filters `<= NOW()`,
        # rather than end-of-day which would read as a few hours in the future.
        return min(ts, pd.Timestamp(ref))

    def _assign_splits(
        self,
        dates: List[str],
        ratios: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        """
        Assign data splits based on dates (chronological).

        Args:
            dates: List of ISO date strings.
            ratios: Split ratios. Uses default 60/20/15/5 if not provided.

        Returns:
            List of split assignments.
        """
        ratios = ratios or {
            "train": 0.60,
            "validation": 0.20,
            "test": 0.15,
            "holdout": 0.05,
        }

        # Sort dates to get boundaries
        sorted_dates = sorted(set(dates))
        n_unique = len(sorted_dates)

        # Calculate cumulative boundaries
        cum_train = int(n_unique * ratios["train"])
        cum_val = int(n_unique * (ratios["train"] + ratios["validation"]))
        cum_test = int(n_unique * (ratios["train"] + ratios["validation"] + ratios["test"]))

        # Create date-to-split mapping
        date_to_split = {}
        for i, d in enumerate(sorted_dates):
            if i < cum_train:
                date_to_split[d] = "train"
            elif i < cum_val:
                date_to_split[d] = "validation"
            elif i < cum_test:
                date_to_split[d] = "test"
            else:
                date_to_split[d] = "holdout"

        return [date_to_split[d] for d in dates]

    def _log(self, message: str) -> None:
        """Log message if verbose mode enabled."""
        if self.config.verbose:
            print(f"[{self.entity_type}] {message}")
