"""Unit tests for ``scripts/converter_schema_reconciliation``.

These tests lock the contract of the reconciliation logic itself — they
don't exercise the real CSU / Optum data files (that's an integration
concern). Each test uses minimal in-memory schemas so the suite runs
fast and stays decoupled from the data trees.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# Import under test — module lives in scripts/, not src/
from converter_schema_reconciliation import (  # noqa: E402
    CONCEPT_CATALOGUE,
    FieldMapping,
    ReconciliationReport,
    _build_synthetic_csu_schema,
    _build_synthetic_optum_schema,
    _dtype_family,
    main,
    reconcile,
    write_report,
)

# --------------------------------------------------------------------------- #
# _dtype_family — coarse-grained dtype equivalence                            #
# --------------------------------------------------------------------------- #


class TestDtypeFamily:
    @pytest.mark.parametrize(
        ("dtype", "expected"),
        [
            ("int64", "int"),
            ("Int64", "int"),
            ("int", "int"),
            ("INT", "int"),
            ("smallint", "int"),
            ("float64", "float"),
            ("Float32", "float"),
            ("double", "float"),
            ("bool", "bool"),
            ("boolean", "bool"),
            ("datetime64[ns]", "datetime"),
            ("timestamp[us]", "datetime"),
            ("date", "datetime"),
            ("string", "string"),
            ("object", "string"),
            ("utf8", "string"),
            ("large_string", "string"),
            ("list<item: string>", "list"),
            ("array", "list"),
            ("dict<...>", "dict"),
            ("struct<...>", "dict"),
            ("map<...>", "dict"),
            ("custom_unknown_type", "custom_unknown_type"),
        ],
    )
    def test_dtype_family_collapses(self, dtype: str, expected: str) -> None:
        assert _dtype_family(dtype) == expected


# --------------------------------------------------------------------------- #
# FieldMapping invariants                                                     #
# --------------------------------------------------------------------------- #


class TestFieldMapping:
    def test_dtype_match_when_same_family(self) -> None:
        m = FieldMapping(
            concept="age",
            csu_column="age_continuous",
            csu_dtype="float64",
            optum_column="age_at_index",
            optum_dtype="Float32",
            semantic_match=True,
        )
        assert m.dtype_match is True
        assert m.is_clean is True

    def test_dtype_mismatch_flagged(self) -> None:
        m = FieldMapping(
            concept="age",
            csu_column="age",
            csu_dtype="int64",
            optum_column="age",
            optum_dtype="float64",
            semantic_match=True,
        )
        assert m.dtype_match is False
        assert m.is_clean is False

    def test_semantic_mismatch_flagged_even_with_dtype_match(self) -> None:
        m = FieldMapping(
            concept="brand",
            csu_column="brand",
            csu_dtype="string",
            optum_column="brand",
            optum_dtype="string",
            semantic_match=False,
        )
        assert m.dtype_match is True
        assert m.is_clean is False

    def test_missing_column_one_side_not_clean(self) -> None:
        m = FieldMapping(
            concept="lookback_start_date",
            csu_column=None,
            csu_dtype=None,
            optum_column="lookback_start_date",
            optum_dtype="string",
            semantic_match=False,
        )
        assert m.dtype_match is False
        assert m.is_clean is False

    def test_dtype_match_returns_false_when_either_dtype_none(self) -> None:
        m = FieldMapping(
            concept="x",
            csu_column="x",
            csu_dtype=None,
            optum_column="x",
            optum_dtype="int",
            semantic_match=True,
        )
        assert m.dtype_match is False


# --------------------------------------------------------------------------- #
# reconcile() core logic                                                      #
# --------------------------------------------------------------------------- #


class TestReconcileCoreLogic:
    """The reconciliation engine on small, controlled fixtures."""

    def test_overlapping_fields_detected(self) -> None:
        """Concept present on both sides shows up in field_mappings as overlapping."""
        csu_schema = {"foo": "int", "bar": "string"}
        optum_schema = {"foo_optum": "int", "bar_optum": "string"}
        catalogue = [
            {
                "concept": "foo_concept",
                "csu_column": "foo",
                "optum_column": "foo_optum",
                "semantic_match": True,
                "notes": "",
            },
            {
                "concept": "bar_concept",
                "csu_column": "bar",
                "optum_column": "bar_optum",
                "semantic_match": True,
                "notes": "",
            },
        ]
        report = reconcile(csu_schema, optum_schema, catalogue=catalogue)
        assert report.overlapping_concepts_total == 2
        assert report.overlapping_concepts_clean == 2
        assert not report.has_mismatches

    def test_dtype_mismatch_is_flagged(self) -> None:
        """A dtype family mismatch should produce a non-clean overlap."""
        csu_schema = {"x": "int64"}
        optum_schema = {"x": "string"}
        catalogue = [
            {
                "concept": "x_concept",
                "csu_column": "x",
                "optum_column": "x",
                "semantic_match": True,
                "notes": "",
            },
        ]
        report = reconcile(csu_schema, optum_schema, catalogue=catalogue)
        assert report.overlapping_concepts_total == 1
        assert report.overlapping_concepts_clean == 0
        assert report.has_mismatches

    def test_semantic_mismatch_is_flagged(self) -> None:
        csu_schema = {"x": "string"}
        optum_schema = {"x": "string"}
        catalogue = [
            {
                "concept": "x_concept",
                "csu_column": "x",
                "optum_column": "x",
                "semantic_match": False,
                "notes": "different meanings",
            },
        ]
        report = reconcile(csu_schema, optum_schema, catalogue=catalogue)
        assert report.overlapping_concepts_total == 1
        assert report.overlapping_concepts_clean == 0
        assert report.has_mismatches

    def test_one_sided_concept_not_overlapping(self) -> None:
        csu_schema = {"a": "int"}
        optum_schema = {"b": "int"}
        catalogue = [
            {
                "concept": "csu_only",
                "csu_column": "a",
                "optum_column": None,
                "semantic_match": False,
                "notes": "",
            },
            {
                "concept": "optum_only",
                "csu_column": None,
                "optum_column": "b",
                "semantic_match": False,
                "notes": "",
            },
        ]
        report = reconcile(csu_schema, optum_schema, catalogue=catalogue)
        assert report.overlapping_concepts_total == 0
        assert report.overlapping_concepts_clean == 0
        # has_mismatches uses the "clean < total" rule; with total=0, no mismatch
        assert not report.has_mismatches

    def test_csu_only_columns_reported(self) -> None:
        csu_schema = {"shared": "int", "csu_extra": "string"}
        optum_schema = {"shared": "int"}
        catalogue = [
            {
                "concept": "shared_concept",
                "csu_column": "shared",
                "optum_column": "shared",
                "semantic_match": True,
                "notes": "",
            },
        ]
        report = reconcile(csu_schema, optum_schema, catalogue=catalogue)
        assert "csu_extra" in report.csu_only_columns
        assert "shared" not in report.csu_only_columns

    def test_optum_only_columns_reported(self) -> None:
        csu_schema = {"shared": "int"}
        optum_schema = {"shared": "int", "optum_extra": "float"}
        catalogue = [
            {
                "concept": "shared_concept",
                "csu_column": "shared",
                "optum_column": "shared",
                "semantic_match": True,
                "notes": "",
            },
        ]
        report = reconcile(csu_schema, optum_schema, catalogue=catalogue)
        assert "optum_extra" in report.optum_only_columns
        assert "shared" not in report.optum_only_columns


# --------------------------------------------------------------------------- #
# JSON roundtrip                                                              #
# --------------------------------------------------------------------------- #


class TestJsonRoundtrip:
    def test_report_writes_valid_json(self, tmp_path: Path) -> None:
        csu_schema = {"foo": "int"}
        optum_schema = {"foo": "int"}
        catalogue = [
            {
                "concept": "foo_concept",
                "csu_column": "foo",
                "optum_column": "foo",
                "semantic_match": True,
                "notes": "",
            },
        ]
        report = reconcile(csu_schema, optum_schema, catalogue=catalogue)
        out = tmp_path / "recon.json"
        write_report(report, out)
        assert out.exists()

        # Reload and verify structure
        with out.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["mode"] == "synthetic"
        assert loaded["csu_columns_total"] == 1
        assert loaded["optum_columns_total"] == 1
        assert loaded["overlapping_concepts_total"] == 1
        assert loaded["overlapping_concepts_clean"] == 1
        assert loaded["has_mismatches"] is False
        assert isinstance(loaded["field_mappings"], list)
        assert loaded["field_mappings"][0]["concept"] == "foo_concept"

    def test_roundtrip_preserves_all_concept_metadata(self, tmp_path: Path) -> None:
        csu_schema = {"x": "int"}
        optum_schema = {"x": "string"}  # mismatched dtype on purpose
        catalogue = [
            {
                "concept": "x_concept",
                "csu_column": "x",
                "optum_column": "x",
                "semantic_match": False,
                "notes": "intentional mismatch for test",
            },
        ]
        report = reconcile(csu_schema, optum_schema, catalogue=catalogue)
        out = tmp_path / "recon.json"
        write_report(report, out)
        with out.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        m = loaded["field_mappings"][0]
        assert m["csu_dtype"] == "int"
        assert m["optum_dtype"] == "string"
        assert m["semantic_match"] is False
        assert m["notes"] == "intentional mismatch for test"
        assert loaded["has_mismatches"] is True


# --------------------------------------------------------------------------- #
# Real catalogue / synthetic-schema integration                                #
# --------------------------------------------------------------------------- #


class TestRealCatalogue:
    """The actual CONCEPT_CATALOGUE applied to the documented synthetic schemas."""

    def test_catalogue_runs_against_synthetic_schemas(self) -> None:
        csu_schema = _build_synthetic_csu_schema()
        optum_schema = _build_synthetic_optum_schema()
        report = reconcile(csu_schema, optum_schema)

        # Sanity bounds — catalogue covers a meaningful subset
        assert report.overlapping_concepts_total > 0
        assert report.csu_columns_total > 0
        assert report.optum_columns_total > 0
        # Many concepts in the catalogue are intentionally semantic-mismatch
        # (treatment_initiated, brand, data_quality_score, ...). The audit
        # documents these as known-divergent. So the report SHOULD have
        # mismatches under the production catalogue.
        assert report.has_mismatches is True

    def test_catalogue_shape(self) -> None:
        """Every catalogue entry has the required keys."""
        required = {"concept", "csu_column", "optum_column", "semantic_match"}
        for entry in CONCEPT_CATALOGUE:
            assert required.issubset(entry.keys()), (
                f"Catalogue entry missing keys: {entry.get('concept')}"
            )

    def test_catalogue_concept_names_unique(self) -> None:
        names = [e["concept"] for e in CONCEPT_CATALOGUE]
        assert len(names) == len(set(names)), "Duplicate concept name in catalogue"

    def test_catalogue_includes_known_leaky_features(self) -> None:
        """The 5 known-leaky CSU features must be in the catalogue (CSU-only)."""
        leaky = {
            "engagement_score",
            "days_on_therapy",
            "hcp_visits",
            "medication_claim_count",
            "disease_severity",
        }
        catalogued_csu = {e["csu_column"] for e in CONCEPT_CATALOGUE if e["csu_column"]}
        missing = leaky - catalogued_csu
        assert not missing, f"Catalogue missing leaky features: {missing}"

    def test_known_leaky_features_have_no_optum_counterpart(self) -> None:
        """Leaky CSU features should be CSU-only — Optum doesn't emit them."""
        leaky = {
            "engagement_score",
            "days_on_therapy",
            "hcp_visits",
            "medication_claim_count",
            "disease_severity",
        }
        for entry in CONCEPT_CATALOGUE:
            if entry["csu_column"] in leaky:
                assert entry["optum_column"] is None, (
                    f"Leaky feature {entry['csu_column']} unexpectedly mapped "
                    f"to Optum column {entry['optum_column']}"
                )


# --------------------------------------------------------------------------- #
# CLI smoke test                                                              #
# --------------------------------------------------------------------------- #


class TestCli:
    def test_cli_synthetic_mode_writes_report(self, tmp_path: Path) -> None:
        out = tmp_path / "recon.json"
        rc = main(["--mode", "synthetic", "--output", str(out), "--no-fail-on-mismatch"])
        assert rc == 0
        assert out.exists()
        with out.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["mode"] == "synthetic"
        assert loaded["overlapping_concepts_total"] > 0

    def test_cli_returns_nonzero_on_mismatch_by_default(self, tmp_path: Path) -> None:
        """CI integration: exit 1 when there are mismatches and --no-fail-on-mismatch
        is NOT passed.

        The production catalogue intentionally contains semantic mismatches
        (brand, treatment_initiated, etc.) so the default-mode CLI run is
        expected to return 1.
        """
        out = tmp_path / "recon.json"
        rc = main(["--mode", "synthetic", "--output", str(out)])
        assert rc == 1, "Default CLI run should exit 1 when catalogue has mismatches"

    def test_cli_files_mode_returns_2_when_inputs_missing(self, tmp_path: Path) -> None:
        """--mode files with missing inputs returns 2 (file-not-found)."""
        rc = main(
            [
                "--mode",
                "files",
                "--csu-input",
                str(tmp_path / "missing_csu.json"),
                "--optum-input",
                str(tmp_path / "missing_optum.parquet"),
                "--output",
                str(tmp_path / "out.json"),
            ]
        )
        assert rc == 2


# --------------------------------------------------------------------------- #
# ReconciliationReport.to_dict                                                #
# --------------------------------------------------------------------------- #


class TestReportToDict:
    def test_to_dict_includes_has_mismatches(self) -> None:
        report = ReconciliationReport(
            mode="synthetic",
            csu_columns_total=1,
            optum_columns_total=1,
            overlapping_concepts_total=1,
            overlapping_concepts_clean=0,
        )
        d = report.to_dict()
        assert "has_mismatches" in d
        assert d["has_mismatches"] is True

    def test_to_dict_field_mappings_serializable(self) -> None:
        m = FieldMapping(
            concept="x",
            csu_column="x",
            csu_dtype="int",
            optum_column="x",
            optum_dtype="int",
            semantic_match=True,
        )
        report = ReconciliationReport(
            mode="synthetic",
            csu_columns_total=1,
            optum_columns_total=1,
            overlapping_concepts_total=1,
            overlapping_concepts_clean=1,
            field_mappings=[m],
        )
        d = report.to_dict()
        # Should be json-serialisable end-to-end
        json.dumps(d)
