"""Unit tests for data_preparer's file ingestion path.

Covers:
- FileIngestor dispatch on extension (parquet / json / csv).
- Directory vs explicit-mapping input shapes.
- Missing-file errors.
- Round-trip integrity (read what the converter writes).
- _load_from_files splitter behaviour: precomputed data_split column vs
  fallback to entity/temporal/random.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.agents.ml_foundation.data_preparer.ingestion import (
    CsvReader,
    FileIngestor,
    IngestionError,
    JsonReader,
    ParquetReader,
)
from src.agents.ml_foundation.data_preparer.nodes.data_loader import (
    _drop_unhashable_columns,
    _load_from_files,
    _split_from_column,
)


@pytest.fixture
def journey_records() -> list[dict]:
    return [
        {
            "patient_journey_id": f"PJ_{i:06d}",
            "patient_id": f"PAT_{i:06d}",
            "journey_start_date": f"2022-0{(i % 9) + 1}-15",
            "treatment_initiated": i % 2,
            "data_split": ["train", "train", "train", "validation", "test", "holdout"][i % 6],
        }
        for i in range(12)
    ]


@pytest.fixture
def hcp_records() -> list[dict]:
    return [{"hcp_id": f"HCP_{i:06d}", "specialty": "Allergy/Immunology"} for i in range(3)]


@pytest.fixture
def directory_with_all_formats(
    tmp_path: Path, journey_records: list[dict], hcp_records: list[dict]
) -> Path:
    """Canonical directory with parquet + json + csv files."""
    pd.DataFrame(journey_records).to_parquet(tmp_path / "e2i_ml_v3_patient_journeys.parquet")
    with open(tmp_path / "e2i_ml_v3_hcp_profiles.json", "w") as f:
        json.dump(hcp_records, f)
    pd.DataFrame([{"treatment_event_id": "TE_000000", "patient_id": "PAT_000000"}]).to_csv(
        tmp_path / "e2i_ml_v3_treatment_events.csv", index=False
    )
    return tmp_path


# --------------------------------------------------------------------------- #
# FileIngestor — unit-level tests                                             #
# --------------------------------------------------------------------------- #


class TestFileIngestorDispatch:
    def test_parquet_reader_extensions(self) -> None:
        assert ".parquet" in ParquetReader.extensions
        assert ".pq" in ParquetReader.extensions

    def test_json_reader_extensions(self) -> None:
        assert ".json" in JsonReader.extensions

    def test_csv_reader_extensions(self) -> None:
        assert ".csv" in CsvReader.extensions

    def test_ingest_file_parquet(self, tmp_path: Path) -> None:
        df_in = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        p = tmp_path / "f.parquet"
        df_in.to_parquet(p)

        df_out = FileIngestor().ingest_file(p)
        pd.testing.assert_frame_equal(df_out, df_in)

    def test_ingest_file_json(self, tmp_path: Path) -> None:
        p = tmp_path / "f.json"
        with open(p, "w") as f:
            json.dump([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}], f)

        df_out = FileIngestor().ingest_file(p)
        assert len(df_out) == 2
        assert list(df_out.columns) == ["a", "b"]

    def test_ingest_file_csv(self, tmp_path: Path) -> None:
        df_in = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        p = tmp_path / "f.csv"
        df_in.to_csv(p, index=False)

        df_out = FileIngestor().ingest_file(p)
        pd.testing.assert_frame_equal(df_out, df_in)

    def test_unknown_extension_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "f.xyz"
        p.write_text("not parseable")
        with pytest.raises(IngestionError, match="No reader registered"):
            FileIngestor().ingest_file(p)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(IngestionError, match="File not found"):
            FileIngestor().ingest_file(tmp_path / "nope.parquet")

    def test_json_top_level_object_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        with open(p, "w") as f:
            json.dump({"not": "a list"}, f)
        with pytest.raises(IngestionError, match="top-level list"):
            FileIngestor().ingest_file(p)


class TestFileIngestorDirectory:
    def test_reads_canonical_files(self, directory_with_all_formats: Path) -> None:
        frames = FileIngestor().ingest_directory(directory_with_all_formats)
        assert set(frames.keys()) == {
            "patient_journeys",
            "treatment_events",
            "hcp_profiles",
        }
        assert len(frames["patient_journeys"]) == 12
        assert len(frames["hcp_profiles"]) == 3
        assert len(frames["treatment_events"]) == 1

    def test_missing_patient_journeys_raises(self, tmp_path: Path) -> None:
        # Only hcp file present.
        with open(tmp_path / "e2i_ml_v3_hcp_profiles.json", "w") as f:
            json.dump([{"hcp_id": "HCP_000000"}], f)
        with pytest.raises(IngestionError, match="patient_journeys"):
            FileIngestor().ingest_directory(tmp_path)

    def test_optional_files_missing_ok(self, tmp_path: Path, journey_records: list[dict]) -> None:
        pd.DataFrame(journey_records).to_parquet(tmp_path / "e2i_ml_v3_patient_journeys.parquet")
        frames = FileIngestor().ingest_directory(tmp_path)
        assert "patient_journeys" in frames
        assert "hcp_profiles" not in frames
        assert "treatment_events" not in frames

    def test_not_a_directory(self, tmp_path: Path) -> None:
        p = tmp_path / "file.txt"
        p.write_text("x")
        with pytest.raises(IngestionError, match="Not a directory"):
            FileIngestor().ingest_directory(p)


class TestFileIngestorMapping:
    def test_explicit_mapping(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"a": [1]})
        p1 = tmp_path / "foo.parquet"
        p2 = tmp_path / "bar.json"
        df.to_parquet(p1)
        with open(p2, "w") as f:
            json.dump([{"b": 2}], f)

        frames = FileIngestor().ingest_mapping({"one": p1, "two": p2})
        assert set(frames.keys()) == {"one", "two"}
        assert len(frames["one"]) == 1
        assert len(frames["two"]) == 1

    def test_mapping_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(IngestionError, match="File not found"):
            FileIngestor().ingest_mapping({"x": tmp_path / "nope.parquet"})


# --------------------------------------------------------------------------- #
# _load_from_files — node-level tests                                         #
# --------------------------------------------------------------------------- #


class TestLoadFromFiles:
    def test_file_dir_with_precomputed_split(self, directory_with_all_formats: Path) -> None:
        result = _load_from_files(
            {"type": "file_dir", "path": str(directory_with_all_formats)},
            entity_column="patient_id",
            date_column="journey_start_date",
        )
        # journey_records fixture has splits in pattern [train, train, train, val, test, holdout] * 2
        assert len(result["train"]) == 6
        assert len(result["val"]) == 2
        assert len(result["test"]) == 2
        assert result["holdout"] is not None
        assert len(result["holdout"]) == 2

    def test_files_mapping_with_precomputed_split(
        self, tmp_path: Path, journey_records: list[dict]
    ) -> None:
        p = tmp_path / "journeys.parquet"
        pd.DataFrame(journey_records).to_parquet(p)

        result = _load_from_files(
            {"type": "files", "paths": {"patient_journeys": str(p)}},
            entity_column="patient_id",
            date_column="journey_start_date",
        )
        assert len(result["train"]) == 6
        assert len(result["val"]) == 2

    def test_fallback_to_splitter_without_data_split_column(self, tmp_path: Path) -> None:
        # No data_split column — must fall through to get_data_splitter().
        df = pd.DataFrame(
            {
                "patient_id": [f"PAT_{i:06d}" for i in range(20)],
                "journey_start_date": [f"2022-01-{i + 1:02d}" for i in range(20)],
                "treatment_initiated": [i % 2 for i in range(20)],
            }
        )
        p = tmp_path / "e2i_ml_v3_patient_journeys.parquet"
        df.to_parquet(p)

        result = _load_from_files(
            {"type": "file_dir", "path": str(tmp_path)},
            entity_column="patient_id",
            date_column="journey_start_date",
        )
        total = sum(len(result[k]) for k in ("train", "val", "test") if result[k] is not None)
        total += len(result["holdout"]) if result["holdout"] is not None else 0
        assert total == len(df)

    def test_random_fallback_produces_holdout_for_split_enforcer(self, tmp_path: Path) -> None:
        """Finding #4(B): when neither an entity nor a date column is present,
        the random fallback previously produced a 0-sample holdout (60/20/20),
        which the model_trainer split_enforcer hard-fails on. It must now
        request the 60/20/15/5 holdout-bearing contract so a non-empty holdout
        is produced and no row is dropped."""
        df = pd.DataFrame(
            {
                "patient_id": [f"PAT_{i:06d}" for i in range(40)],
                "treatment_initiated": [i % 2 for i in range(40)],
            }
        )
        p = tmp_path / "e2i_ml_v3_patient_journeys.parquet"
        df.to_parquet(p)

        result = _load_from_files(
            {"type": "file_dir", "path": str(tmp_path)},
            entity_column=None,  # force past the entity branch
            date_column="nonexistent_date",  # force past the temporal branch -> random
        )

        # Holdout must be populated so split_enforcer's empty-holdout check passes.
        assert result["holdout"] is not None
        assert len(result["holdout"]) > 0
        # No row is dropped across the four splits.
        total = (
            len(result["train"]) + len(result["val"]) + len(result["test"]) + len(result["holdout"])
        )
        assert total == len(df)
        # Test split honours the 15% (not 20%) contract expected by the enforcer.
        assert len(result["test"]) == int(len(df) * 0.15)

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(IngestionError, match="Unknown file data_source"):
            _load_from_files(
                {"type": "bogus"},
                entity_column=None,
                date_column="d",
            )

    def test_file_dir_missing_path_raises(self) -> None:
        with pytest.raises(IngestionError, match="path required"):
            _load_from_files(
                {"type": "file_dir"},
                entity_column=None,
                date_column="d",
            )

    def test_files_missing_paths_raises(self) -> None:
        with pytest.raises(IngestionError, match="paths required"):
            _load_from_files(
                {"type": "files"},
                entity_column=None,
                date_column="d",
            )


class TestSplitFromColumn:
    def test_partitions_correctly(self) -> None:
        df = pd.DataFrame(
            {
                "x": list(range(10)),
                "data_split": [
                    "train",
                    "train",
                    "train",
                    "train",
                    "train",
                    "train",
                    "validation",
                    "validation",
                    "test",
                    "holdout",
                ],
            }
        )
        result = _split_from_column(df)
        assert len(result["train"]) == 6
        assert len(result["val"]) == 2
        assert len(result["test"]) == 1
        assert result["holdout"] is not None
        assert len(result["holdout"]) == 1

    def test_val_alias_accepted(self) -> None:
        df = pd.DataFrame({"x": [1, 2], "data_split": ["val", "train"]})
        result = _split_from_column(df)
        assert len(result["val"]) == 1
        assert len(result["train"]) == 1

    def test_empty_holdout_returns_none(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3], "data_split": ["train"] * 3})
        result = _split_from_column(df)
        assert result["holdout"] is None


class TestDropUnhashableColumns:
    """Defensive filter for object-dtype cols with unhashable cells.

    Real CSU patient_journeys.json carries list-typed metadata
    (`comorbidities`, `secondary_diagnosis_codes`, `data_sources_matched`)
    that crash `nunique()` / `value_counts()` in leakage_detector,
    data_transformer, and baseline_computer. The filter removes these
    pre-split so downstream nodes never see them.
    """

    def test_drops_list_cells(self, caplog: pytest.LogCaptureFixture) -> None:
        df = pd.DataFrame(
            {
                "patient_id": ["p1", "p2", "p3"],
                "comorbidities": [["asthma"], [], ["allergic_rhinitis"]],
                "age": [30, 40, 50],
            }
        )
        with caplog.at_level("WARNING"):
            result = _drop_unhashable_columns(df)
        assert list(result.columns) == ["patient_id", "age"]
        assert "comorbidities" in caplog.text

    def test_drops_dict_cells(self) -> None:
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "metadata": [{"key": "val"}, {"key": "other"}],
                "score": [0.1, 0.2],
            }
        )
        result = _drop_unhashable_columns(df)
        assert list(result.columns) == ["id", "score"]

    def test_drops_set_cells(self) -> None:
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "tags": [{"a", "b"}, {"c"}],
            }
        )
        result = _drop_unhashable_columns(df)
        assert "tags" not in result.columns

    def test_drops_tuple_cells(self) -> None:
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "coord": [(1, 2), (3, 4)],
            }
        )
        result = _drop_unhashable_columns(df)
        assert "coord" not in result.columns

    def test_drops_ndarray_cells(self, caplog: pytest.LogCaptureFixture) -> None:
        # Iter-5 audit (2026-05-09): Optum-init e2e crashed at
        # baseline_computer.py:75 because the Optum converter writes Parquet
        # and pyarrow's ListArray decode produces ``numpy.ndarray`` cells
        # rather than Python ``list`` cells. PR #105's filter only included
        # ``list`` so the np.ndarray case slipped through. The two
        # representations carry the same logical "list-of-strings" payload
        # (``comorbidities``, ``secondary_diagnosis_codes``, etc.) and must
        # be dropped uniformly regardless of file format.
        df = pd.DataFrame(
            {
                "patient_id": ["p1", "p2", "p3"],
                "comorbidities": [
                    np.array(["asthma"], dtype=object),
                    np.array([], dtype=object),
                    np.array(["allergic_rhinitis", "atopic_dermatitis"], dtype=object),
                ],
                "age": [30, 40, 50],
            }
        )
        with caplog.at_level("WARNING"):
            result = _drop_unhashable_columns(df)
        assert list(result.columns) == ["patient_id", "age"]
        assert "comorbidities" in caplog.text

    def test_preserves_scalar_object_columns(self) -> None:
        df = pd.DataFrame(
            {
                "id": ["p1", "p2"],
                "diagnosis_code": ["L20.9", "L50.1"],
                "age_group": ["18-34", "65+"],
            }
        )
        result = _drop_unhashable_columns(df)
        assert list(result.columns) == ["id", "diagnosis_code", "age_group"]

    def test_preserves_numeric_columns(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]})
        result = _drop_unhashable_columns(df)
        assert list(result.columns) == ["a", "b"]

    def test_preserves_all_null_object_columns(self) -> None:
        # All-null object cols cannot be sampled to detect type — leave in
        # place. nunique() returns 0 for these (benign downstream).
        df = pd.DataFrame(
            {
                "id": ["p1", "p2", "p3"],
                "always_null": [None, None, None],
            }
        )
        result = _drop_unhashable_columns(df)
        assert "always_null" in result.columns

    def test_drops_list_when_first_value_is_null(self) -> None:
        # First non-null cell is what's sampled — None values don't shield
        # an unhashable column from being detected.
        df = pd.DataFrame(
            {
                "id": ["p1", "p2", "p3"],
                "list_with_nulls": [None, ["asthma"], None],
            }
        )
        result = _drop_unhashable_columns(df)
        assert "list_with_nulls" not in result.columns

    def test_drops_mixed_scalar_then_list(self) -> None:
        # codex review HIGH-B regression: a column whose FIRST non-null
        # cell is a scalar but LATER rows contain unhashable values must
        # still be detected and dropped. Sampling iloc[0] alone would
        # silently let this column through; the full-scan implementation
        # catches it.
        df = pd.DataFrame(
            {
                "id": ["p1", "p2", "p3"],
                "tricky": ["scalar_first", "still_scalar", ["asthma", "rhinitis"]],
            }
        )
        result = _drop_unhashable_columns(df)
        assert "tricky" not in result.columns

    def test_drops_mixed_scalar_then_dict(self) -> None:
        # Same as above but with dict in a later row.
        df = pd.DataFrame(
            {
                "id": ["p1", "p2", "p3"],
                "tricky": ["s1", "s2", {"key": "val"}],
            }
        )
        result = _drop_unhashable_columns(df)
        assert "tricky" not in result.columns

    def test_no_op_when_all_columns_hashable(self) -> None:
        df = pd.DataFrame(
            {
                "id": ["p1", "p2"],
                "age": [30, 40],
                "diagnosis": ["L20.9", "L50.1"],
            }
        )
        result = _drop_unhashable_columns(df)
        # Same column set; same shape.
        assert list(result.columns) == ["id", "age", "diagnosis"]
        assert len(result) == 2

    def test_load_from_files_integration_with_list_cols(self, tmp_path: Path) -> None:
        """End-to-end: _load_from_files cleans list cols before split."""
        records = [
            {
                "patient_id": f"PAT_{i:06d}",
                "journey_start_date": f"2022-0{(i % 9) + 1}-15",
                "comorbidities": [] if i % 2 == 0 else ["asthma"],
                "secondary_diagnosis_codes": [],
                "treatment_initiated": i % 2,
                "data_split": ["train", "train", "validation", "test", "holdout", "train"][i % 6],
            }
            for i in range(12)
        ]
        path = tmp_path / "e2i_ml_v3_patient_journeys.json"
        with open(path, "w") as f:
            json.dump(records, f)

        result = _load_from_files(
            {"type": "file_dir", "path": str(tmp_path)},
            entity_column="patient_id",
            date_column="journey_start_date",
        )

        # List cols dropped before splitter saw them — no nunique crash.
        for split in ("train", "val", "test", "holdout"):
            df = result[split]
            if df is None:
                continue
            assert "comorbidities" not in df.columns
            assert "secondary_diagnosis_codes" not in df.columns
            # Scalar cols preserved.
            assert "patient_id" in df.columns
            assert "treatment_initiated" in df.columns
