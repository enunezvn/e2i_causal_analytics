"""Integration smoke tests for `scripts/run_phase1_multi_disease.py`."""

from __future__ import annotations

import asyncio
import importlib.util as _importlib_util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "scripts"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SPEC = _importlib_util.spec_from_file_location(
    "run_phase1_multi_disease", SCRIPTS_DIR / "run_phase1_multi_disease.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = _importlib_util.module_from_spec(_SPEC)
sys.modules["run_phase1_multi_disease"] = _RUNNER
_SPEC.loader.exec_module(_RUNNER)


class TestRunMultiDisease:
    def test_runs_all_three_scenarios(self) -> None:
        artifact = asyncio.run(
            _RUNNER.run_multi_disease(
                None,
                seed=42,
                n_total=500,
                run_rwd_validation=True,
            )
        )
        assert artifact["schema_version"] == "phase1_multi_disease.v1"
        codes = {s["short_code"] for s in artifact["scenarios"]}
        assert codes == {"A", "B", "C"}

    def test_filter_by_short_codes(self) -> None:
        artifact = asyncio.run(
            _RUNNER.run_multi_disease(
                ["A", "C"],
                seed=42,
                n_total=500,
                run_rwd_validation=False,
            )
        )
        codes = {s["short_code"] for s in artifact["scenarios"]}
        assert codes == {"A", "C"}

    def test_unknown_short_code_raises(self) -> None:
        with pytest.raises(SystemExit):
            asyncio.run(
                _RUNNER.run_multi_disease(
                    ["Z"],
                    seed=42,
                    n_total=500,
                    run_rwd_validation=False,
                )
            )

    def test_scenario_c_carries_rwd_block_when_enabled(self) -> None:
        artifact = asyncio.run(
            _RUNNER.run_multi_disease(
                ["C"],
                seed=42,
                n_total=500,
                run_rwd_validation=True,
            )
        )
        sc = artifact["scenarios"][0]
        assert "rwd_concurrent_validation" in sc
        rwd = sc["rwd_concurrent_validation"]
        assert rwd["enabled"] is True
        assert rwd["status"] == "scaffolded"

    def test_scenario_c_skips_rwd_when_disabled(self) -> None:
        artifact = asyncio.run(
            _RUNNER.run_multi_disease(
                ["C"],
                seed=42,
                n_total=500,
                run_rwd_validation=False,
            )
        )
        sc = artifact["scenarios"][0]
        assert "rwd_concurrent_validation" in sc
        assert sc["rwd_concurrent_validation"]["enabled"] is False

    def test_scenarios_ab_have_no_rwd_block(self) -> None:
        artifact = asyncio.run(
            _RUNNER.run_multi_disease(
                ["A", "B"],
                seed=42,
                n_total=500,
                run_rwd_validation=True,
            )
        )
        for sc in artifact["scenarios"]:
            assert "rwd_concurrent_validation" not in sc


class TestRenderMarkdownSummary:
    def test_renders_one_row_per_scenario(self) -> None:
        artifact = asyncio.run(
            _RUNNER.run_multi_disease(
                None,
                seed=42,
                n_total=500,
                run_rwd_validation=False,
            )
        )
        md = _RUNNER.render_markdown_summary(artifact)
        assert "Phase 1 multi-disease run" in md
        for code in ("A", "B", "C"):
            assert f"| {code} |" in md
        assert "phase1_multi_disease.v1" in md

    def test_markdown_round_trip_through_json_artifact(self) -> None:
        artifact = asyncio.run(
            _RUNNER.run_multi_disease(
                None,
                seed=42,
                n_total=500,
                run_rwd_validation=False,
            )
        )
        # Ensure json-serializable
        s = json.dumps(artifact)
        assert "schema_version" in s
