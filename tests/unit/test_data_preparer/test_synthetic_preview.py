"""Unit tests for the Phase 3 synthetic-preview adapter.

Covers the adapter's own logic (scenario validation, size clamping, metadata
shape, the no-auto-mix invariant) with a patched generator, plus one real
``generate_scenario`` call to confirm the true integration.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from src.agents.ml_foundation.data_preparer.adapters.synthetic_preview import (
    PREVIEW_MAX_N,
    PREVIEW_MIN_N,
    build_synthetic_preview,
)

_REAL_SCENARIO = "scenario_a_diagnostic_ebc_idfs_5y_balanced"


def _fake_dataset(n_total: int) -> SimpleNamespace:
    """Minimal SyntheticDataset stand-in with the attrs the adapter reads."""
    n_train = int(n_total * 0.6)
    n_val = int(n_total * 0.2)
    n_test = n_total - n_train - n_val
    rng = np.random.default_rng(0)
    metadata = SimpleNamespace(
        scenario=SimpleNamespace(value=_REAL_SCENARIO),
        seed=42,
        n_total=n_total,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        realized_prevalence=0.31,
        target_prevalence=0.30,
        feature_names=("f0", "f1", "f2"),
        audit_fingerprint="deadbeef",
    )
    return SimpleNamespace(
        X_train=rng.normal(size=(n_train, 3)),
        y_train=rng.integers(0, 2, size=n_train),
        X_val=rng.normal(size=(n_val, 3)),
        y_val=rng.integers(0, 2, size=n_val),
        X_test=rng.normal(size=(n_test, 3)),
        y_test=rng.integers(0, 2, size=n_test),
        stratify=rng.integers(0, 2, size=n_total),
        metadata=metadata,
    )


class TestAdapterLogic:
    """Adapter behavior with the generator patched out (fast, deterministic)."""

    def test_metadata_shape_and_no_auto_mix(self, tmp_path):
        captured = {}

        def fake_generate(scenario_enum, *, seed, n_total):
            captured["scenario"] = scenario_enum.value
            captured["n_total"] = n_total
            return _fake_dataset(n_total)

        with patch("src.ml.synthetic_v2.api.generate_scenario", side_effect=fake_generate):
            meta = build_synthetic_preview(
                scenario=_REAL_SCENARIO,
                recommended_n=500,
                workflow_id="wf-123",
                output_root=tmp_path,
            )

        # Load-bearing invariant.
        assert meta["auto_mixed_into_training"] is False
        assert meta["scenario"] == _REAL_SCENARIO
        assert meta["requested_recommended_n"] == 500
        assert meta["preview_n_total"] == 500
        assert meta["preview_n_capped"] is False
        # Artifacts actually written to disk.
        out_dir = tmp_path / "synthetic_preview_wf-123"
        assert (out_dir / "preview_cohort.npz").exists()
        meta_on_disk = json.loads((out_dir / "preview_metadata.json").read_text())
        assert meta_on_disk == meta

    def test_recommendation_above_ceiling_is_capped(self, tmp_path):
        seen = {}

        def fake_generate(scenario_enum, *, seed, n_total):
            seen["n_total"] = n_total
            return _fake_dataset(n_total)

        with patch("src.ml.synthetic_v2.api.generate_scenario", side_effect=fake_generate):
            meta = build_synthetic_preview(
                scenario=_REAL_SCENARIO,
                recommended_n=10_000_000,
                workflow_id="wf-cap",
                output_root=tmp_path,
            )

        assert seen["n_total"] == PREVIEW_MAX_N
        assert meta["preview_n_total"] == PREVIEW_MAX_N
        assert meta["requested_recommended_n"] == 10_000_000
        assert meta["preview_n_capped"] is True

    def test_recommendation_below_floor_is_raised(self, tmp_path):
        seen = {}

        def fake_generate(scenario_enum, *, seed, n_total):
            seen["n_total"] = n_total
            return _fake_dataset(n_total)

        with patch("src.ml.synthetic_v2.api.generate_scenario", side_effect=fake_generate):
            meta = build_synthetic_preview(
                scenario=_REAL_SCENARIO,
                recommended_n=10,
                workflow_id="wf-floor",
                output_root=tmp_path,
            )

        assert seen["n_total"] == PREVIEW_MIN_N
        assert meta["preview_n_total"] == PREVIEW_MIN_N
        assert meta["preview_n_capped"] is True

    def test_unknown_scenario_raises_with_valid_list(self, tmp_path):
        with pytest.raises(ValueError, match="valid scenarios"):
            build_synthetic_preview(
                scenario="not_a_real_scenario",
                recommended_n=500,
                workflow_id="wf-bad",
                output_root=tmp_path,
            )


@pytest.mark.asyncio
async def test_real_generate_scenario_integration(tmp_path):
    """One real call through synthetic_v2 to confirm the true wiring."""
    meta = build_synthetic_preview(
        scenario=_REAL_SCENARIO,
        recommended_n=300,
        workflow_id="wf-real",
        output_root=tmp_path,
    )
    assert meta["preview_n_total"] == 300
    assert meta["auto_mixed_into_training"] is False
    assert meta["n_train"] + meta["n_val"] + meta["n_test"] == 300
    out_dir = tmp_path / "synthetic_preview_wf-real"
    arrays = np.load(out_dir / "preview_cohort.npz")
    assert arrays["X_train"].shape[0] == meta["n_train"]
    assert arrays["y_train"].shape[0] == meta["n_train"]
