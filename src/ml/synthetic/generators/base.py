"""
Base Generator for Synthetic Data.

Provides common functionality for all entity generators.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, Generic, Iterator, List, Optional, TypeVar

import numpy as np
import pandas as pd

from ..config import SyntheticDataConfig, Brand, DGPType


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
    verbose: bool = False


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
                verbose=self.config.verbose,
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
        return [f"{prefix}_{i:0{width}d}" for i in range(n)]

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
        """Generate random dates within range as ISO strings."""
        start = start_date or self.config.start_date
        end = end_date or self.config.end_date

        start_ord = start.toordinal()
        end_ord = end.toordinal()

        random_days = self._rng.integers(start_ord, end_ord + 1, size=n)
        return [date.fromordinal(d).isoformat() for d in random_days]

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
