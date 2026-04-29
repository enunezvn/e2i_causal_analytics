"""Tests for TriggerGenerator brand_id emission (6B-infra-6).

Covers the audit-driven gap surfaced by the 2026-04-28 pre-apply blast-radius
audit: migration 033 promotes ``triggers.brand_id`` to NOT NULL with no DEFAULT,
so every producer of trigger rows must emit ``brand_id``. ``TriggerGenerator``
is the canonical synthetic-data producer (consumed by
``scripts/load_synthetic_data.py``); without ``brand_id`` in its output, fresh-DB
re-bootstrap fails on the ``triggers`` insert.

Sourcing strategy is Option A from the plan: pass-through of the existing
``brand`` value, since the post-migration column semantic IS the brand
identifier.
"""

import pandas as pd
import pytest

from src.ml.synthetic.config import BRANDS, Brand
from src.ml.synthetic.generators import GeneratorConfig, TriggerGenerator


def _make_patient_df(brands: list[str], n_per_brand: int = 5) -> pd.DataFrame:
    """Build a minimal patient_df satisfying TriggerGenerator's `.get()` lookups."""
    rows = []
    for brand in brands:
        for i in range(n_per_brand):
            rows.append(
                {
                    "patient_id": f"pt_{brand}_{i:03d}",
                    "hcp_id": f"hcp_{i:03d}",
                    "engagement_score": 5.0 + (i % 5),
                    "treatment_initiated": i % 2,
                    "journey_start_date": "2023-01-01",
                    "brand": brand,
                }
            )
    return pd.DataFrame(rows)


def _make_hcp_df(n: int = 5) -> pd.DataFrame:
    """Build a minimal hcp_df (TriggerGenerator only checks `is not None`)."""
    return pd.DataFrame({"hcp_id": [f"hcp_{i:03d}" for i in range(n)]})


class TestTriggerGeneratorBrandId:
    """6B-infra-6: brand_id column must be emitted by TriggerGenerator."""

    def test_linked_mode_emits_brand_id_column(self):
        """Linked-mode path (patient_df + hcp_df): brand_id is in output columns."""
        patient_df = _make_patient_df(brands=BRANDS)
        hcp_df = _make_hcp_df()
        config = GeneratorConfig(seed=42, n_records=len(patient_df))
        gen = TriggerGenerator(config, patient_df=patient_df, hcp_df=hcp_df)

        df = gen.generate()

        assert "brand_id" in df.columns, "TriggerGenerator must emit brand_id"
        assert df["brand_id"].notna().all(), "brand_id must be populated for every row"

    def test_linked_mode_brand_id_matches_brand(self):
        """Linked-mode path: brand_id is the pass-through of the brand value."""
        patient_df = _make_patient_df(brands=BRANDS)
        hcp_df = _make_hcp_df()
        config = GeneratorConfig(seed=42, n_records=len(patient_df))
        gen = TriggerGenerator(config, patient_df=patient_df, hcp_df=hcp_df)

        df = gen.generate()

        assert (df["brand_id"] == df["brand"]).all(), (
            "Option A sourcing: brand_id must equal brand on every row"
        )

    def test_linked_mode_brand_id_values_are_brand_enum(self):
        """Linked-mode path: brand_id values are members of the Brand enum."""
        patient_df = _make_patient_df(brands=BRANDS)
        hcp_df = _make_hcp_df()
        config = GeneratorConfig(seed=42, n_records=len(patient_df))
        gen = TriggerGenerator(config, patient_df=patient_df, hcp_df=hcp_df)

        df = gen.generate()

        valid_values = {b.value for b in Brand}
        assert set(df["brand_id"].unique()).issubset(valid_values), (
            "brand_id values must be drawn from Brand enum"
        )

    def test_standalone_mode_emits_brand_id_column(self):
        """Standalone path (no patient_df / hcp_df): brand_id is in output columns."""
        config = GeneratorConfig(seed=42, n_records=50)
        gen = TriggerGenerator(config)

        df = gen.generate()

        assert "brand_id" in df.columns, "Standalone path must also emit brand_id"
        assert df["brand_id"].notna().all(), "brand_id must be populated for every row"
        assert (df["brand_id"] == df["brand"]).all(), (
            "Option A sourcing: brand_id must equal brand on every row"
        )

    @pytest.mark.parametrize("brand", list(Brand))
    def test_single_brand_propagates_to_brand_id(self, brand: Brand):
        """When config pins a single brand, brand_id mirrors it across all rows."""
        config = GeneratorConfig(seed=42, n_records=30, brand=brand)
        gen = TriggerGenerator(config)

        df = gen.generate()

        assert (df["brand_id"] == brand.value).all(), (
            f"All brand_id values must equal {brand.value} when config pins it"
        )
