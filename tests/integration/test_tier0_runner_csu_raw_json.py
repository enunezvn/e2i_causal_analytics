"""Integration test: ``scripts/run_tier0_test.py`` ingests raw CSU JSON
end-to-end without ``unhashable type: 'list'``.

Pins the closure of issue #197. The CSU ``e2i_ml_v3_patient_journeys.json``
file produced by ``scripts/convert_csu_rwd.py`` carries list-typed metadata
columns (``comorbidities``, ``secondary_diagnosis_codes``,
``data_sources_matched``). Before #197 these would crash the
``data_transformer`` node at ``df[col].nunique()`` if they reached
``_identify_column_types``.

Defense layers (post-#197):
  1. ``data_loader._drop_unhashable_columns`` strips them at file ingest.
  2. ``data_transformer._column_has_unhashable_cells`` skips them in the
     type-detection step (defense-in-depth — protects callers who bypass
     the loader, e.g. preassembled DataFrames).
  3. The runner script's Step 4 / Step 5 feature discovery explicitly
     excludes them by name.

This test runs steps 1 + 2 of the runner against a small CSU JSON
fixture built deterministically from a slice of the real CSU file.
Steps 3-8 are out of scope (they take >100s wall-clock on real CSU and
their own integration tests cover them); the goal here is to pin the
list-column ingestion contract — not the full pipeline metrics.

Marked ``slow`` (uses real CSU JSON layout, runs the data_preparer
agent end-to-end) and skipped when the CSU file is unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.integration._asyncio_compat import run_sync

REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_csu_journeys_file() -> Path | None:
    """Locate the canonical CSU journeys file.

    Worktrees may not have the full ``data/rwd/csu/`` tree; the file
    typically lives at the parent repo's ``data/rwd/csu/``. Search both.
    """
    candidates = [
        REPO_ROOT / "data" / "rwd" / "csu" / "e2i_ml_v3_patient_journeys.json",
        # Fallback: parent worktree may be a sibling of this worktree.
        REPO_ROOT.parent.parent.parent / "data" / "rwd" / "csu" / "e2i_ml_v3_patient_journeys.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


@pytest.fixture(scope="module")
def csu_fixture_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a small CSU directory with a slice of the real journeys file.

    Keeps the full column surface (45 columns including ``comorbidities``
    and ``secondary_diagnosis_codes`` with list-typed cells) so the
    transformer's list-defense path is actually exercised end-to-end —
    not just the columns the existing ``test_csu_full_data_preparer_e2e``
    fixture whitelists.
    """
    source = _find_csu_journeys_file()
    if source is None:
        pytest.skip(
            "CSU journeys file not present; this test requires "
            "data/rwd/csu/e2i_ml_v3_patient_journeys.json"
        )

    records = json.loads(source.read_text())
    # 300 rows: enough to clear the data_preparer minimum-samples gates
    # (~150 per split after 60/20/20 chronological split) and keep the
    # test wall-clock under ~150s on a real CSU run.
    slice_ = records[:300]
    target_dir = tmp_path_factory.mktemp("csu_runner_fixture")
    (target_dir / "e2i_ml_v3_patient_journeys.json").write_text(json.dumps(slice_))
    return target_dir


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(300)
def test_transformer_survives_raw_csu_list_columns(csu_fixture_dir: Path) -> None:
    """The transformer node must NOT raise ``TypeError: unhashable type:
    'list'`` when fed the raw CSU JSON via ``_load_from_files``.

    This is the surgical closure assertion for issue #197 — runs the
    actual ``_load_from_files`` + ``transform_data`` pipeline on the
    real-shape CSU fixture (lists in ``comorbidities`` /
    ``secondary_diagnosis_codes`` survive ingestion until the
    loader-side guard drops them; if any list cell reaches the
    transformer, the new ``_column_has_unhashable_cells`` guard
    catches it without crashing).
    """

    from src.agents.ml_foundation.data_preparer.nodes.data_loader import (
        _load_from_files,
    )
    from src.agents.ml_foundation.data_preparer.nodes.data_transformer import (
        transform_data,
    )

    data_source = {"type": "file_dir", "path": str(csu_fixture_dir)}
    dataset = _load_from_files(
        data_source=data_source,
        entity_column=None,
        date_column="journey_start_date",
    )
    train_df = dataset["train"]
    val_df = dataset["val"]
    test_df = dataset["test"]

    assert len(train_df) > 0, "Loader produced empty train split"
    # Loader-side guard MUST have stripped the list-typed columns.
    assert "comorbidities" not in train_df.columns
    assert "secondary_diagnosis_codes" not in train_df.columns

    # Now call transform_data with default scope_spec — list cols are
    # already gone, so the transformer's type-detection step doesn't
    # see them at all. The test pins that the loader → transformer
    # contract works end-to-end without any pre-cleaning.
    state = {
        "experiment_id": "test_issue_197_runner_e2e",
        "train_df": train_df,
        "validation_df": val_df,
        "test_df": test_df,
        "scope_spec": {
            "target_column": "treatment_initiated",
            "scaling_method": "minmax",
            "imputation_strategy": "mean",
            "extract_datetime_features": False,
            "excluded_features": [],  # NOT pre-cleaning list cols
        },
    }
    result = run_sync(transform_data(state))
    assert result.get("error") is None, (
        f"transform_data crashed: {result.get('error')!r}\n"
        f"This means the issue #197 defense regressed; check "
        f"data_transformer._column_has_unhashable_cells "
        f"and data_loader._drop_unhashable_columns."
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(300)
def test_transformer_survives_unstripped_list_columns(csu_fixture_dir: Path) -> None:
    """Defense-in-depth: even if the loader's strip is bypassed (caller
    preassembles ``train_df`` with list-typed cells intact), the
    transformer's ``_column_has_unhashable_cells`` guard prevents the
    crash. This pins the surgical fix introduced for issue #197.
    """

    import pandas as pd

    from src.agents.ml_foundation.data_preparer.nodes.data_transformer import (
        transform_data,
    )

    # Load the raw JSON directly — bypass the loader → bypass the strip.
    journeys_path = csu_fixture_dir / "e2i_ml_v3_patient_journeys.json"
    records = json.loads(journeys_path.read_text())
    df = pd.DataFrame(records)

    # Confirm the fixture actually carries list-typed cells (otherwise
    # the test would tautologically pass without exercising the guard).
    assert "comorbidities" in df.columns
    assert "secondary_diagnosis_codes" in df.columns
    # Use a precomputed split column so transform_data has consistent splits.
    if "data_split" in df.columns:
        train_df = df[df["data_split"] == "train"].copy()
        val_df = df[df["data_split"].isin(["validation", "val"])].copy()
        test_df = df[df["data_split"] == "test"].copy()
    else:
        train_df = df.iloc[: int(len(df) * 0.6)].copy()
        val_df = df.iloc[int(len(df) * 0.6) : int(len(df) * 0.8)].copy()
        test_df = df.iloc[int(len(df) * 0.8) :].copy()

    state = {
        "experiment_id": "test_issue_197_bypass_loader",
        "train_df": train_df,
        "validation_df": val_df,
        "test_df": test_df,
        "scope_spec": {
            "target_column": "treatment_initiated",
            "scaling_method": "minmax",
            "imputation_strategy": "mean",
            "extract_datetime_features": False,
            # Crucially: NOT excluding the list-typed columns. The guard
            # must catch them silently.
            "excluded_features": [],
        },
    }
    result = run_sync(transform_data(state))
    assert result.get("error") is None, (
        f"transform_data crashed on raw list cols (loader bypassed): "
        f"{result.get('error')!r}\n"
        f"This means the issue #197 transformer-side guard regressed; "
        f"check data_transformer._column_has_unhashable_cells."
    )
    feature_columns = result.get("feature_columns") or []
    # Codex pass-1 MEDIUM-1 (2026-05-14): the transformer DROPS list-
    # typed columns from the frame, mirroring loader semantics so
    # downstream model_trainer preprocessor does not re-trip the same
    # nunique() crash. The columns must NOT be in feature_columns.
    assert "comorbidities" not in feature_columns
    assert "secondary_diagnosis_codes" not in feature_columns
    # And the drop is recorded in transformations_applied for auditability.
    transformations = result.get("transformations_applied") or []
    drop_entries = [t for t in transformations if t.get("type") == "drop_unhashable_columns"]
    assert len(drop_entries) == 1
    dropped_set = set(drop_entries[0].get("columns") or [])
    assert "comorbidities" in dropped_set
    assert "secondary_diagnosis_codes" in dropped_set

    # Codex pass-2 MEDIUM-2 (2026-05-14): the cleaned frames also thread
    # back into state under canonical ``train_df`` / ``validation_df`` /
    # ``test_df`` keys so downstream data_preparer nodes (feast_registrar,
    # baseline_computer, finalize_output) consume the cleaned schema.
    assert "train_df" in result, (
        "transform_data MUST surface cleaned train_df in state delta (Codex pass-2 MED-2)"
    )
    state_train_df = result["train_df"]
    assert "comorbidities" not in state_train_df.columns
    assert "secondary_diagnosis_codes" not in state_train_df.columns
    # Target preserved in state's train_df.
    assert "treatment_initiated" in state_train_df.columns
