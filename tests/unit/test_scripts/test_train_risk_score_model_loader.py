"""Unit tests for the real-Optum loader in ``scripts/train_risk_score_model.py``.

Codex pass-1 HIGH-1 + MEDIUM-1 (issue #171 PR #172):

  - HIGH-1: the loader must DROP every column in ``OPTUM_FORBIDDEN_AS_FEATURES``
    + ``OPTUM_TARGETS`` from the training feature matrix, even when those
    columns slip past the manifest gate at cohort-build time. In particular,
    ``treatment_initiated`` is an exact alias of ``initiated_biologic_180d``
    and must not appear in features.

  - MEDIUM-1: the loader must read ONLY the patient_journeys artifact, not
    glob every parquet/csv in the cohort dir. The converter emits multiple
    tables (treatment_events, hcp_profiles, split_registry); concatenating
    them produces NaN labels and contaminated feature selection.

These tests do NOT require real Optum data — they build a tiny synthetic
patient_journeys parquet inline and verify the loader's allow-list logic.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _load_loader_module():
    """Import scripts/train_risk_score_model.py for direct function access."""
    spec = importlib.util.spec_from_file_location(
        "train_risk_score_model",
        PROJECT_ROOT / "scripts" / "train_risk_score_model.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_risk_score_model"] = module
    spec.loader.exec_module(module)
    return module


def _make_synthetic_journeys() -> pd.DataFrame:
    """Build a tiny patient_journeys frame with a target + safe features +
    forbidden columns that the loader must drop.
    """
    n = 40
    rng = np.random.RandomState(0)
    return pd.DataFrame(
        {
            # Targets — must be dropped from features (HIGH-1).
            "initiated_biologic_180d": rng.randint(0, 2, n),
            "treatment_initiated": rng.randint(0, 2, n),  # alias of target
            "discontinuation_flag": rng.randint(0, 2, n),
            "discontinued_180d": rng.randint(0, 2, n),
            "persistent_at_180d": rng.randint(0, 2, n),
            # Manifest-safe features (pre-or-at-index per
            # src/data/manifests/optum_feature_manifest.py).
            "age_at_index": rng.randint(18, 80, n),
            "ed_visits_urticaria_angio": rng.poisson(1.0, n),
            "h1_1g_fill_count": rng.poisson(0.5, n),
            # Metadata / split tracking.
            "data_split": ["train"] * 30 + ["validation"] * 10,
            "patid": np.arange(n),
            "index_date": ["2024-01-01"] * n,
        }
    )


class TestLoaderAntiLeakage:
    def test_loader_drops_treatment_initiated_alias(self, tmp_path: Path) -> None:
        """HIGH-1 regression: ``treatment_initiated`` (alias of the target) must
        not appear in the training feature matrix.
        """
        df = _make_synthetic_journeys()
        df.to_parquet(tmp_path / "e2i_ml_v3_patient_journeys.parquet")
        loader = _load_loader_module()
        X_train, y_train, X_val, y_val = loader._load_real_optum_data(
            tmp_path, target="initiated_biologic_180d"
        )
        # Anti-leakage anchor.
        for forbidden in (
            "initiated_biologic_180d",
            "treatment_initiated",  # the load-bearing alias.
            "discontinuation_flag",
            "discontinued_180d",
            "persistent_at_180d",
        ):
            assert forbidden not in X_train.columns, (
                f"Loader leaked target/alias {forbidden!r} into training features."
            )
            assert forbidden not in X_val.columns, (
                f"Loader leaked target/alias {forbidden!r} into validation features."
            )
        # Sanity: y is still the right thing.
        assert y_train.shape == (30,)
        assert y_val.shape == (10,)
        # Safe features survived.
        assert "age_at_index" in X_train.columns
        assert "ed_visits_urticaria_angio" in X_train.columns

    def test_loader_only_reads_patient_journeys_file(self, tmp_path: Path) -> None:
        """MEDIUM-1 regression: stray ``treatment_events`` / ``hcp_profiles``
        parquet files in the same dir must NOT be merged into the feature frame.
        """
        df = _make_synthetic_journeys()
        df.to_parquet(tmp_path / "e2i_ml_v3_patient_journeys.parquet")
        # Stray treatment_events file with a totally different schema.
        stray = pd.DataFrame({"treatment_event_id": ["evt_001", "evt_002"], "drug": ["X", "Y"]})
        stray.to_parquet(tmp_path / "e2i_ml_v3_treatment_events.parquet")
        # Stray hcp_profiles file.
        hcp = pd.DataFrame({"hcp_id": ["H1", "H2", "H3"], "specialty": ["A", "D", "I"]})
        hcp.to_parquet(tmp_path / "e2i_ml_v3_hcp_profiles.parquet")

        loader = _load_loader_module()
        X_train, y_train, _, _ = loader._load_real_optum_data(
            tmp_path, target="initiated_biologic_180d"
        )
        # If the loader incorrectly globbed all parquets, X_train would have
        # rows = 40 + 2 + 3 = 45 (with NaN labels). The correct behavior is
        # rows == 30 (train split only).
        assert len(X_train) == 30
        assert "treatment_event_id" not in X_train.columns
        assert "specialty" not in X_train.columns
        # The y vector is dense int (no NaN).
        assert all(int(v) in (0, 1) for v in y_train)

    def test_loader_raises_when_journeys_file_missing(self, tmp_path: Path) -> None:
        """Loader fails loud if patient_journeys file is absent."""
        loader = _load_loader_module()
        with pytest.raises(FileNotFoundError, match="e2i_ml_v3_patient_journeys"):
            loader._load_real_optum_data(tmp_path, target="initiated_biologic_180d")

    def test_loader_raises_when_target_missing(self, tmp_path: Path) -> None:
        """Loader fails loud if the target column isn't in the frame."""
        df = _make_synthetic_journeys().drop(columns=["initiated_biologic_180d"])
        df.to_parquet(tmp_path / "e2i_ml_v3_patient_journeys.parquet")
        loader = _load_loader_module()
        with pytest.raises(KeyError, match="initiated_biologic_180d"):
            loader._load_real_optum_data(tmp_path, target="initiated_biologic_180d")

    def test_loader_csv_fallback_works(self, tmp_path: Path) -> None:
        """Loader supports CSV fallback when no parquet present."""
        df = _make_synthetic_journeys()
        df.to_csv(tmp_path / "e2i_ml_v3_patient_journeys.csv", index=False)
        loader = _load_loader_module()
        X_train, y_train, _, _ = loader._load_real_optum_data(
            tmp_path, target="initiated_biologic_180d"
        )
        assert len(X_train) == 30
        assert "treatment_initiated" not in X_train.columns
