"""Smoke tests for scripts/build_kg_cache.py.

Live KG calls are NOT exercised here — those tests gate on
``UMLS_UTS_API_KEY`` (skipped in CI). The CLI is exercised via its
public functions; HTTP clients are passed as ``None`` for the no-op
case and are not constructed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_cli_help_runs_without_error():
    """`python scripts/build_kg_cache.py --help` exits 0 with usage text."""
    result = subprocess.run(
        [sys.executable, "scripts/build_kg_cache.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "manifest-module" in result.stdout


def test_build_with_no_entity_features_writes_empty_cache(tmp_path: Path):
    """A manifest with zero entity-bearing features produces an empty
    cache file (no-op success).
    """
    from scripts.build_kg_cache import build_cache_for_manifest
    from src.data.feature_contract import FeatureContract, KnowableAt

    features = [
        FeatureContract(
            name="age",
            knowable_at=KnowableAt(reference="enrollment"),
            source="demo",
            derivation_inputs=("age",),
        )
    ]
    out = tmp_path / "kg_cache"
    cache_path = build_cache_for_manifest(
        features=features,
        target_entity_codes=[("RXNORM", "479158")],
        out_dir=out,
        umls_client=None,
        open_targets_client=None,
    )

    assert cache_path.exists()
    payload = json.loads(cache_path.read_text())
    assert payload == []

    # Companion summary
    summary_path = cache_path.with_suffix(".summary.md")
    assert summary_path.exists()
    summary = summary_path.read_text()
    assert "KG Cache Summary" in summary


def test_build_with_entity_feature_emits_record(tmp_path: Path):
    """A manifest with one entity-bearing feature produces one record."""
    from scripts.build_kg_cache import build_cache_for_manifest
    from src.data.feature_contract import FeatureContract, KnowableAt

    features = [
        FeatureContract(
            name="primary_diagnosis_code",
            knowable_at=KnowableAt(reference="enrollment"),
            source="demo",
            derivation_inputs=("diagcode",),
            kg_entity_codes=(("ICD10CM", "L50.9"), ("UMLS", "C0042109")),
        )
    ]
    out = tmp_path / "kg_cache"
    cache_path = build_cache_for_manifest(
        features=features,
        target_entity_codes=[],
        out_dir=out,
        umls_client=None,
        open_targets_client=None,
    )

    payload = json.loads(cache_path.read_text())
    assert len(payload) == 1
    assert payload[0]["feature_name"] == "primary_diagnosis_code"
    assert payload[0]["status"] in {"queried_no_edges", "ok"}


def test_parse_target_codes_handles_empty_string():
    """Empty --target-entity-codes parses to []."""
    from scripts.build_kg_cache import _parse_target_codes

    assert _parse_target_codes("") == []
    assert _parse_target_codes("  ") == []


def test_parse_target_codes_parses_multiple():
    from scripts.build_kg_cache import _parse_target_codes

    out = _parse_target_codes("RXNORM:479158,RXNORM:1011295")
    assert out == [("RXNORM", "479158"), ("RXNORM", "1011295")]


def test_parse_target_codes_rejects_malformed():
    """Missing colon → ValueError surfaced to caller."""
    import pytest

    from scripts.build_kg_cache import _parse_target_codes

    with pytest.raises(ValueError, match="expected SYSTEM:code"):
        _parse_target_codes("RXNORM_no_colon_479158")


def test_cache_filename_omits_cohort(tmp_path: Path):
    """Disease-agnostic invariant: only the two fingerprints in the path."""
    from scripts.build_kg_cache import build_cache_for_manifest
    from src.data.feature_contract import FeatureContract, KnowableAt

    features = [
        FeatureContract(
            name="age",
            knowable_at=KnowableAt(reference="enrollment"),
            source="demo",
            derivation_inputs=("age",),
        )
    ]
    out = tmp_path / "kg_cache"
    cache_path = build_cache_for_manifest(
        features=features,
        target_entity_codes=[],
        out_dir=out,
        umls_client=None,
        open_targets_client=None,
    )

    # Cache filename pattern: {manifest_fp}__{target_fp}.json
    assert cache_path.name.endswith(".json")
    assert "__" in cache_path.name
    assert "csu" not in cache_path.name
    assert "optum" not in cache_path.name
