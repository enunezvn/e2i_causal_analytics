"""Every synthetic dataset must be stamped is_synthetic=True before load."""

from scripts.load_synthetic_data import SMALL_SIZES, generate_datasets
from src.ml.synthetic.config import DGPType


def test_all_datasets_stamped_is_synthetic():
    datasets = generate_datasets(SMALL_SIZES, DGPType.CONFOUNDED, seed=42)
    assert datasets, "generate_datasets returned nothing"
    for table, df in datasets.items():
        assert "is_synthetic" in df.columns, f"{table} not stamped"
        assert df["is_synthetic"].notna().all(), f"{table} has NULL is_synthetic"
        assert bool(df["is_synthetic"].all()) is True, f"{table} has False/0 rows"
