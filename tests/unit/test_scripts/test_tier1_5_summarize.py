"""Unit tests for the Tier 1-5 harness summarize / expected-fail gate (#616).

``scripts/run_tier1_5_test.summarize_results`` is the honest-signal step the
workflow calls: it renders a per-agent pass/fail table to ``$GITHUB_STEP_SUMMARY``
and enforces the expected-fail allow-list. These tests pin its exit-code contract
and table contents without running the (heavy, docker-dependent) harness.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TIER1_5_SCRIPT = PROJECT_ROOT / "scripts" / "run_tier1_5_test.py"


def _load_script_module(name: str, path: Path):
    """Load a script-as-module, registering it in ``sys.modules`` first."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


@pytest.fixture(scope="module")
def harness():
    mod = _load_script_module("run_tier1_5_test_summarize", TIER1_5_SCRIPT)
    assert mod is not None, "Failed to import run_tier1_5_test.py"
    return mod


def _results(rows: list[dict]) -> dict:
    passed = sum(1 for r in rows if r.get("success"))
    return {
        "summary": {
            "total_agents": len(rows),
            "passed": passed,
            "failed": len(rows) - passed,
        },
        "results": rows,
    }


def _agent(name: str, tier: int, success: bool, **extra) -> dict:
    row = {"agent_name": name, "tier": tier, "success": success}
    row.update(extra)
    return row


def _write(tmp_path: Path, data: dict) -> str:
    p = tmp_path / "results.json"
    p.write_text(json.dumps(data))
    return str(p)


def test_clean_baseline_exits_zero(harness, tmp_path):
    """13/13 pass + empty allow-list -> exit 0, table shows N/N."""
    rows = [_agent(f"a{i}", 1, True) for i in range(13)]
    results = _write(tmp_path, _results(rows))
    summary = tmp_path / "summary.md"
    code = harness.summarize_results(results, "", str(summary))
    assert code == 0
    text = summary.read_text()
    assert "13/13 agents passed" in text
    assert "No new (non-allow-listed) agent failures" in text


def test_new_failure_not_on_allow_list_exits_one(harness, tmp_path):
    rows = [_agent("orchestrator", 1, True), _agent("health_score", 3, False, error="boom")]
    results = _write(tmp_path, _results(rows))
    summary = tmp_path / "summary.md"
    code = harness.summarize_results(results, "", str(summary))
    assert code == 1
    text = summary.read_text()
    assert "NEW-FAIL" in text
    assert "health_score" in text


def test_known_failure_on_allow_list_exits_zero(harness, tmp_path):
    rows = [_agent("orchestrator", 1, True), _agent("health_score", 3, False, error="boom")]
    results = _write(tmp_path, _results(rows))
    summary = tmp_path / "summary.md"
    code = harness.summarize_results(results, "health_score", str(summary))
    assert code == 0
    text = summary.read_text()
    assert "KNOWN-FAIL" in text
    assert "No new (non-allow-listed) agent failures" in text


def test_mixed_known_and_new_failure_exits_one(harness, tmp_path):
    rows = [
        _agent("health_score", 3, False, error="known"),
        _agent("gap_analyzer", 2, False, error="new"),
    ]
    results = _write(tmp_path, _results(rows))
    summary = tmp_path / "summary.md"
    # health_score is allow-listed; gap_analyzer is NOT -> still hard fail.
    code = harness.summarize_results(results, "health_score", str(summary))
    assert code == 1
    text = summary.read_text()
    assert "gap_analyzer" in text


def test_missing_results_file_exits_one(harness, tmp_path):
    summary = tmp_path / "summary.md"
    code = harness.summarize_results(str(tmp_path / "nope.json"), "", str(summary))
    assert code == 1
    assert "not found" in summary.read_text()


def test_empty_results_exits_one(harness, tmp_path):
    """Fail-CLOSED on a present-but-empty results set (#616 hardening, codex L-2).

    The honest gate exists to fail-closed; a results JSON with zero agent rows
    means the harness ran no agent — it must HARD FAIL (exit 1), not slip through
    as a vacuous 0/0 'pass'."""
    results = _write(tmp_path, _results([]))
    summary = tmp_path / "summary.md"
    code = harness.summarize_results(results, "", str(summary))
    assert code == 1
    assert "no agent rows" in summary.read_text().lower()


def test_allow_list_is_case_insensitive(harness, tmp_path):
    """Allow-list matching must be case-insensitive (#616 hardening, codex M-1).

    Agent names are snake_case lowercase; a maintainer typing 'Orchestrator' in
    TIER1_5_EXPECTED_FAIL_AGENTS must still match the failing 'orchestrator'
    instead of mis-routing it to a (confusing) hard fail."""
    rows = [_agent("orchestrator", 1, False, error="known")]
    results = _write(tmp_path, _results(rows))
    summary = tmp_path / "summary.md"
    code = harness.summarize_results(results, "Orchestrator", str(summary))
    assert code == 0
    text = summary.read_text()
    assert "KNOWN-FAIL" in text


def test_mock_data_source_rendered_as_plumbing_only(harness, tmp_path):
    """A marked-mock agent (#616 fix#2) must render as a plumbing-only PASS in
    the table, not as a plain green that hides canned reasoning."""
    rows = [
        _agent(
            "tool_composer",
            1,
            True,
            data_source={"detected_source": "mock", "passed": True},
            quality_gate={"passed": True},
        )
    ]
    results = _write(tmp_path, _results(rows))
    summary = tmp_path / "summary.md"
    code = harness.summarize_results(results, "", str(summary))
    assert code == 0
    assert "mock (plumbing-only)" in summary.read_text()


def test_allow_list_parsing_trims_and_ignores_blanks(harness):
    assert harness._parse_expected_fail(" a , b ,, c ") == {"a", "b", "c"}
    assert harness._parse_expected_fail("") == set()
    assert harness._parse_expected_fail(None) == set()
    # Case-folded so matching is case-insensitive (#616 hardening, codex M-1).
    assert harness._parse_expected_fail("Orchestrator, TOOL_COMPOSER") == {
        "orchestrator",
        "tool_composer",
    }


def test_step_summary_none_falls_back_to_stdout(harness, tmp_path, capsys):
    rows = [_agent("a", 1, True)]
    results = _write(tmp_path, _results(rows))
    code = harness.summarize_results(results, "", None)
    assert code == 0
    out = capsys.readouterr().out
    assert "Tier 1-5 Agent Harness" in out


# ---------------------------------------------------------------------------
# Per-agent timeout resolution (CLI --timeout is a FLOOR, not a cap)
# ---------------------------------------------------------------------------
#
# Regression: with `--timeout 180`, experiment_monitor's per-agent config value
# (20s) silently CAPPED the run BELOW the explicit CLI flag and the agent timed
# out on a LOKY-serialized box. An explicit CLI timeout must act as a floor;
# per-agent timeouts LONGER than the CLI value are preserved.


def test_explicit_cli_timeout_floors_short_agent_config(harness):
    assert harness.resolve_agent_timeout({"timeout": 20.0}, 180.0) == 180.0


def test_longer_agent_config_timeout_is_preserved(harness):
    assert harness.resolve_agent_timeout({"timeout": 300.0}, 180.0) == 300.0


def test_no_cli_timeout_uses_agent_config(harness):
    assert harness.resolve_agent_timeout({"timeout": 20.0}, None) == 20.0


def test_no_agent_config_uses_cli_timeout(harness):
    assert harness.resolve_agent_timeout({}, 180.0) == 180.0


def test_neither_uses_default(harness):
    assert harness.resolve_agent_timeout({}, None) == harness.DEFAULT_AGENT_TIMEOUT_SECONDS
