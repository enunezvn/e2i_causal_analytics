"""Part 1 (#39) — re-materialization entrypoint for gold-standard serving bundles.

The gold-standard ``*_goldstd_lr_v1`` registry rows point at ``.pkl`` artifacts
that were written in (now-reaped) worktrees and are GONE — and even when present
those were the BARE estimator (no FeatureBuilder). This entrypoint re-materializes
a SERVING BUNDLE (model + fitted FeatureBuilder + encoded feature_columns) to a
NEW durable path under ``data/ml_artifacts/shap_serving/<cohort>/`` so the BentoML
service can serve a RAW 3-covariate request.

It must NOT mutate ``ml_model_registry`` (that is the gold-standard session's
domain). It only loads rows, fits, and writes a bundle file.

These tests use a FAKE async DB client returning a small synthetic-shaped frame
(no live DB), exercising the REAL FeatureBuilder + train_cohort_model + bundle
serializer end-to-end.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.rematerialize_goldstd_bundles import (
    SHAP_SERVING_ROOT,
    SPEC_REGISTRY,
    rematerialize_bundle,
)
from src.mlops.gold_standard_eval.cohort_spec import INITIATION


class _FakeQuery:
    """Minimal async supabase-py query stub returning a fixed frame's rows."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self._offset = 0

    def select(self, *_a, **_k):
        return self

    def eq(self, *_a, **_k):
        return self

    def in_(self, *_a, **_k):
        return self

    def lt(self, *_a, **_k):
        return self

    def order(self, *_a, **_k):
        return self

    def range(self, start, end):
        self._start, self._end = start, end
        return self

    async def execute(self):
        class _R:
            pass

        r = _R()
        # Single page then empty (cap-agnostic loader stops on empty page).
        if self._start == 0:
            r.data = self._rows
        else:
            r.data = []
        return r


class _FakeDB:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def table(self, _name: str):
        return _FakeQuery(self._rows)


def _synthetic_rows(n: int = 400) -> list[dict]:
    rng = np.random.default_rng(1)
    regions = rng.choice(["midwest", "northeast", "south", "west"], n)
    return [
        {
            "patient_id": f"PAT-{i}",
            "journey_start_date": "2025-01-01",
            "data_split": "train",
            "disease_severity": float(round(rng.normal(5, 1.5), 2)),
            "academic_hcp": int(rng.integers(0, 2)),
            "geographic_region": str(regions[i]),
            "treatment_initiated": int(rng.integers(0, 2)),
        }
        for i in range(n)
    ]


class TestSpecRegistry:
    def test_registry_covers_all_twelve_live_models(self) -> None:
        """Every live ``*_goldstd_lr_v1`` registry model_name is resolvable."""
        expected = {
            "csu_initiation_goldstd_lr_v1",
            "pnh_persistence_goldstd_lr_v1",
            "pnh_discontinuation_goldstd_lr_v1",
            "initiation_remibrutinib_goldstd_lr_v1",
            "initiation_fabhalta_goldstd_lr_v1",
            "initiation_kisqali_goldstd_lr_v1",
            "persistence_remibrutinib_goldstd_lr_v1",
            "persistence_fabhalta_goldstd_lr_v1",
            "persistence_kisqali_goldstd_lr_v1",
            "discontinuation_remibrutinib_goldstd_lr_v1",
            "discontinuation_fabhalta_goldstd_lr_v1",
            "discontinuation_kisqali_goldstd_lr_v1",
        }
        assert set(SPEC_REGISTRY.keys()) == expected

    def test_serving_root_is_under_data_ml_artifacts(self) -> None:
        assert SHAP_SERVING_ROOT.parts[-2:] == ("ml_artifacts", "shap_serving")


@pytest.mark.asyncio
class TestRematerializeBundle:
    async def test_writes_bundle_for_initiation(self, tmp_path: Path) -> None:
        db = _FakeDB(_synthetic_rows())
        result = await rematerialize_bundle(
            db,
            model_name="csu_initiation_goldstd_lr_v1",
            spec=INITIATION,
            out_root=tmp_path,
        )
        # Bundle written to <out_root>/<cohort>/<model_name>.bundle.pkl
        path = Path(result["bundle_path"])
        assert path.exists()
        assert path.name == "csu_initiation_goldstd_lr_v1.bundle.pkl"
        assert path.parent.name == "initiation"
        # AUC is a real finite number in [0,1].
        assert 0.0 <= result["auc"] <= 1.0
        assert result["feature_count"] == 9
        assert result["training_samples"] > 0

    async def test_bundle_round_trips_raw_to_proba(self, tmp_path: Path) -> None:
        db = _FakeDB(_synthetic_rows())
        result = await rematerialize_bundle(
            db,
            model_name="csu_initiation_goldstd_lr_v1",
            spec=INITIATION,
            out_root=tmp_path,
        )
        with open(result["bundle_path"], "rb") as fh:
            bundle = pickle.load(fh)
        raw = pd.DataFrame(
            [{"disease_severity": 5.61, "academic_hcp": 0, "geographic_region": "northeast"}]
        )
        encoded = bundle["preprocessor"].transform(raw)
        assert list(encoded.columns) == bundle["feature_columns"]
        proba = bundle["model"].predict_proba(encoded)[:, 1]
        assert np.isfinite(proba[0])

    async def test_does_not_touch_registry(self, tmp_path: Path) -> None:
        """The function takes a db for READS only; it must never call insert/upsert.

        We pass a db whose ``table`` returns a query stub WITHOUT insert/upsert/
        update; if the function tried to mutate the registry it would AttributeError.
        """
        db = _FakeDB(_synthetic_rows())
        # No insert/update/upsert on _FakeQuery → a write attempt raises.
        await rematerialize_bundle(
            db,
            model_name="csu_initiation_goldstd_lr_v1",
            spec=INITIATION,
            out_root=tmp_path,
        )  # must complete without AttributeError
