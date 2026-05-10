"""Tests for ``scripts/check_lifecycle_state.py`` — Plan v4 Gate N2 scanner.

Covers acceptance criteria:

3. Scanner detects YAML config without ``lifecycle_state``.
4. Scanner detects Python module gate-relevant constant without
   ``LIFECYCLE_STATE_*``.
5. Scanner accepts a config with valid ``lifecycle_state``.
6. Lifecycle-change doc check: scanner detects a change to ``lifecycle_state``
   without corresponding doc and fails.
7. CI workflow YAML is valid (yaml-load — actionlint not assumed available).

Tests use the scanner's helper functions directly with synthesized fixture
content so they do not depend on the live repo state.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

# Import the scanner module by file path because ``scripts/`` is not a
# package (no ``__init__.py``) and is not on ``sys.path`` by default.
_SCANNER_PATH = Path(__file__).resolve().parents[3] / "scripts" / "check_lifecycle_state.py"
_spec = importlib.util.spec_from_file_location("_check_lifecycle_state", _SCANNER_PATH)
assert _spec is not None and _spec.loader is not None
_scanner_mod = importlib.util.module_from_spec(_spec)
sys.modules["_check_lifecycle_state"] = _scanner_mod
_spec.loader.exec_module(_scanner_mod)

scan_python_modules = _scanner_mod.scan_python_modules
scan_yaml_configs = _scanner_mod.scan_yaml_configs
scan_lifecycle_changes = _scanner_mod.scan_lifecycle_changes
ScanFinding = _scanner_mod.ScanFinding
GATE_RELEVANT_PYTHON_MODULES: dict[str, frozenset[str]] = _scanner_mod.GATE_RELEVANT_PYTHON_MODULES


@pytest.fixture
def fake_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a minimal fake repo with the directory shape the scanner expects.

    Patches ``GATE_RELEVANT_PYTHON_MODULES`` + ``YAML_CONFIG_DENYLIST`` to
    sentinels so each test can control discovery.
    """
    (tmp_path / "config").mkdir()
    (tmp_path / "src" / "test_module").mkdir(parents=True)
    (tmp_path / "docs" / "calibration").mkdir(parents=True)
    monkeypatch.setattr(_scanner_mod, "GATE_RELEVANT_PYTHON_MODULES", {})
    monkeypatch.setattr(_scanner_mod, "YAML_CONFIG_DENYLIST", frozenset())
    return tmp_path


# ---------------------------------------------------------------------------
# Acceptance #4: Python module missing LIFECYCLE_STATE_* fails the scanner.
# ---------------------------------------------------------------------------


class TestScanPythonModules:
    def test_missing_constant_emits_error(
        self, fake_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rel = "src/test_module/gate_a.py"
        (fake_repo / rel).write_text("X = 1\n", encoding="utf-8")
        monkeypatch.setattr(
            _scanner_mod,
            "GATE_RELEVANT_PYTHON_MODULES",
            {rel: frozenset({"LIFECYCLE_STATE_GATE_A"})},
        )
        findings = scan_python_modules(fake_repo)
        assert len(findings) == 1
        f = findings[0]
        assert f.severity == "error"
        assert f.code == "missing_lifecycle_constant"
        assert f.path == rel
        assert "LIFECYCLE_STATE_GATE_A" in f.message

    def test_attribute_rhs_recognized(
        self, fake_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rel = "src/test_module/gate_b.py"
        (fake_repo / rel).write_text(
            "from src.lifecycle import GateLifecycleState\n"
            "LIFECYCLE_STATE_B = GateLifecycleState.ADVISORY\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            _scanner_mod,
            "GATE_RELEVANT_PYTHON_MODULES",
            {rel: frozenset({"LIFECYCLE_STATE_B"})},
        )
        findings = scan_python_modules(fake_repo)
        assert findings == []

    def test_string_literal_rhs_recognized(
        self, fake_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rel = "src/test_module/gate_c.py"
        (fake_repo / rel).write_text('LIFECYCLE_STATE_C = "advisory"\n', encoding="utf-8")
        monkeypatch.setattr(
            _scanner_mod,
            "GATE_RELEVANT_PYTHON_MODULES",
            {rel: frozenset({"LIFECYCLE_STATE_C"})},
        )
        findings = scan_python_modules(fake_repo)
        assert findings == []

    def test_annotated_rhs_recognized(
        self, fake_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rel = "src/test_module/gate_d.py"
        (fake_repo / rel).write_text(
            "from src.lifecycle import GateLifecycleState\n"
            "LIFECYCLE_STATE_D: GateLifecycleState = GateLifecycleState.DEVELOPMENT\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            _scanner_mod,
            "GATE_RELEVANT_PYTHON_MODULES",
            {rel: frozenset({"LIFECYCLE_STATE_D"})},
        )
        findings = scan_python_modules(fake_repo)
        assert findings == []

    def test_unrecognized_rhs_emits_error(
        self, fake_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rel = "src/test_module/gate_e.py"
        (fake_repo / rel).write_text(
            "def _get_state():\n    return 'advisory'\nLIFECYCLE_STATE_E = _get_state()\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            _scanner_mod,
            "GATE_RELEVANT_PYTHON_MODULES",
            {rel: frozenset({"LIFECYCLE_STATE_E"})},
        )
        findings = scan_python_modules(fake_repo)
        assert len(findings) == 1
        f = findings[0]
        assert f.code == "unrecognized_lifecycle_rhs"
        assert "LIFECYCLE_STATE_E" in f.message

    def test_invalid_string_literal_unrecognized(
        self, fake_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rel = "src/test_module/gate_f.py"
        (fake_repo / rel).write_text('LIFECYCLE_STATE_F = "not_a_real_state"\n', encoding="utf-8")
        monkeypatch.setattr(
            _scanner_mod,
            "GATE_RELEVANT_PYTHON_MODULES",
            {rel: frozenset({"LIFECYCLE_STATE_F"})},
        )
        findings = scan_python_modules(fake_repo)
        assert len(findings) == 1
        assert findings[0].code == "unrecognized_lifecycle_rhs"

    def test_invalid_attribute_value_unrecognized(
        self, fake_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rel = "src/test_module/gate_g.py"
        (fake_repo / rel).write_text(
            "from src.lifecycle import GateLifecycleState\n"
            "LIFECYCLE_STATE_G = GateLifecycleState.NOT_REAL\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            _scanner_mod,
            "GATE_RELEVANT_PYTHON_MODULES",
            {rel: frozenset({"LIFECYCLE_STATE_G"})},
        )
        findings = scan_python_modules(fake_repo)
        # AnnAssign / Assign with unknown attribute returns None ->
        # unrecognized_lifecycle_rhs.
        assert len(findings) == 1
        assert findings[0].code == "unrecognized_lifecycle_rhs"

    def test_missing_module_emits_error(
        self, fake_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            _scanner_mod,
            "GATE_RELEVANT_PYTHON_MODULES",
            {"src/test_module/never_existed.py": frozenset({"LIFECYCLE_STATE_X"})},
        )
        findings = scan_python_modules(fake_repo)
        assert len(findings) == 1
        assert findings[0].code == "missing_python_module"

    # ----------------------------------------------------------------------
    # N2 finding H2 + L2: scanner must detect class-level constants too.
    # ----------------------------------------------------------------------

    def test_class_level_constant_detected(
        self, fake_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rel = "src/test_module/gate_class.py"
        (fake_repo / rel).write_text(
            "from src.lifecycle import GateLifecycleState\n"
            "class EvaluatorConfig:\n"
            "    LIFECYCLE_STATE_T22 = GateLifecycleState.ADVISORY\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            _scanner_mod,
            "GATE_RELEVANT_PYTHON_MODULES",
            {rel: frozenset({"LIFECYCLE_STATE_T22"})},
        )
        findings = scan_python_modules(fake_repo)
        assert findings == [], (
            "class-level LIFECYCLE_STATE_* constants must be detected; "
            f"got findings={[f.to_dict() for f in findings]}"
        )

    def test_class_level_invalid_rhs_flagged(
        self, fake_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rel = "src/test_module/gate_class_invalid.py"
        (fake_repo / rel).write_text(
            "class Cfg:\n"
            '    LIFECYCLE_STATE_BAD = "not_a_real_state"\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(
            _scanner_mod,
            "GATE_RELEVANT_PYTHON_MODULES",
            {rel: frozenset({"LIFECYCLE_STATE_BAD"})},
        )
        findings = scan_python_modules(fake_repo)
        assert len(findings) == 1
        assert findings[0].code == "unrecognized_lifecycle_rhs"

    def test_function_local_constant_not_detected(
        self, fake_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A LIFECYCLE_STATE_* assigned inside a function body is NOT a
        stable declaration — it must not be picked up; the gate should
        report it as missing.
        """
        rel = "src/test_module/gate_func.py"
        (fake_repo / rel).write_text(
            "from src.lifecycle import GateLifecycleState\n"
            "def _make_cfg():\n"
            "    LIFECYCLE_STATE_INNER = GateLifecycleState.ADVISORY\n"
            "    return LIFECYCLE_STATE_INNER\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            _scanner_mod,
            "GATE_RELEVANT_PYTHON_MODULES",
            {rel: frozenset({"LIFECYCLE_STATE_INNER"})},
        )
        findings = scan_python_modules(fake_repo)
        assert len(findings) == 1
        assert findings[0].code == "missing_lifecycle_constant"


# ---------------------------------------------------------------------------
# Acceptance #3 + #5: gate-shaped YAML without lifecycle_state fails;
# YAML with valid lifecycle_state passes.
# ---------------------------------------------------------------------------


class TestScanYamlConfigs:
    def test_gate_shaped_yaml_without_lifecycle_state_fails(
        self, fake_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (fake_repo / "config" / "fake_gate.yaml").write_text(
            "threshold: 0.7\nbuffer: 0.05\n", encoding="utf-8"
        )
        findings = scan_yaml_configs(fake_repo)
        assert len(findings) == 1
        f = findings[0]
        assert f.severity == "error"
        assert f.code == "missing_lifecycle_state_key"
        assert f.path == "config/fake_gate.yaml"

    def test_gate_shaped_yaml_with_valid_lifecycle_state_passes(
        self, fake_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (fake_repo / "config" / "fake_gate.yaml").write_text(
            "lifecycle_state: advisory\nthreshold: 0.7\nbuffer: 0.05\n",
            encoding="utf-8",
        )
        findings = scan_yaml_configs(fake_repo)
        assert findings == []

    def test_yaml_with_invalid_lifecycle_state_value_fails(
        self, fake_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (fake_repo / "config" / "fake_gate.yaml").write_text(
            "lifecycle_state: shadow\nthreshold: 0.7\n", encoding="utf-8"
        )
        findings = scan_yaml_configs(fake_repo)
        assert len(findings) == 1
        f = findings[0]
        assert f.severity == "error"
        assert f.code == "invalid_lifecycle_state_value"
        assert f.details == {"value": "shadow"}

    def test_non_gate_yaml_without_lifecycle_state_skipped(
        self, fake_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No threshold/cutoff/buffer keywords -> skipped.
        (fake_repo / "config" / "vocabulary.yaml").write_text(
            "brands:\n  - foo\n  - bar\n", encoding="utf-8"
        )
        findings = scan_yaml_configs(fake_repo)
        assert findings == []

    def test_denylisted_yaml_skipped(
        self, fake_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (fake_repo / "config" / "denied.yaml").write_text("threshold: 0.7\n", encoding="utf-8")
        monkeypatch.setattr(
            _scanner_mod,
            "YAML_CONFIG_DENYLIST",
            frozenset({"config/denied.yaml"}),
        )
        findings = scan_yaml_configs(fake_repo)
        assert findings == []

    def test_yaml_parse_error_emits_error(
        self, fake_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (fake_repo / "config" / "broken.yaml").write_text(
            "threshold: 0.7\n  invalid: indentation\n - garbled: yaml: syntax\n",
            encoding="utf-8",
        )
        findings = scan_yaml_configs(fake_repo)
        # One yaml_parse_error finding.
        assert any(f.code == "yaml_parse_error" for f in findings)

    # ----------------------------------------------------------------------
    # N2 finding H3: nested config dirs (config/env/prod.yaml) must be
    # scanned, not silently skipped.
    # ----------------------------------------------------------------------

    def test_nested_config_dir_scanned(
        self, fake_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        nested = fake_repo / "config" / "env"
        nested.mkdir(parents=True)
        (nested / "prod.yaml").write_text("threshold: 0.7\nbuffer: 0.05\n", encoding="utf-8")
        findings = scan_yaml_configs(fake_repo)
        assert len(findings) == 1
        assert findings[0].path == "config/env/prod.yaml"
        assert findings[0].code == "missing_lifecycle_state_key"

    def test_nested_config_dir_with_lifecycle_state_passes(
        self, fake_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        nested = fake_repo / "config" / "overlays" / "staging"
        nested.mkdir(parents=True)
        (nested / "limits.yml").write_text(
            "lifecycle_state: advisory\nthreshold: 0.7\n", encoding="utf-8"
        )
        findings = scan_yaml_configs(fake_repo)
        assert findings == []

    def test_nested_config_denylist_honored(
        self, fake_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        nested = fake_repo / "config" / "env"
        nested.mkdir(parents=True)
        (nested / "denied.yaml").write_text("threshold: 0.7\n", encoding="utf-8")
        monkeypatch.setattr(
            _scanner_mod,
            "YAML_CONFIG_DENYLIST",
            frozenset({"config/env/denied.yaml"}),
        )
        findings = scan_yaml_configs(fake_repo)
        assert findings == []


# ---------------------------------------------------------------------------
# Acceptance #6: lifecycle-state changes without a signed doc fail.
# ---------------------------------------------------------------------------


class TestExtractLifecycleChanges:
    """Direct test of the diff-line parser without a real git diff."""

    def _extract(self, diff_text: str) -> list[dict[str, Any]]:
        return _scanner_mod._extract_lifecycle_changes(diff_text)

    def test_python_attribute_change_detected(self) -> None:
        diff = (
            "diff --git a/src/x.py b/src/x.py\n"
            "--- a/src/x.py\n"
            "+++ b/src/x.py\n"
            "@@ -1 +1 @@\n"
            "-LIFECYCLE_STATE_T22 = GateLifecycleState.ADVISORY\n"
            "+LIFECYCLE_STATE_T22 = GateLifecycleState.CALIBRATING\n"
        )
        changes = self._extract(diff)
        assert changes == [
            {
                "slug": "t22",
                "from_state": "advisory",
                "to_state": "calibrating",
                "source_path": "src/x.py",
            }
        ]

    def test_python_string_literal_change_detected(self) -> None:
        diff = (
            "+++ b/src/y.py\n"
            "@@ -1 +1 @@\n"
            '-LIFECYCLE_STATE_T22 = "advisory"\n'
            '+LIFECYCLE_STATE_T22 = "enforced"\n'
        )
        changes = self._extract(diff)
        assert len(changes) == 1
        assert changes[0]["from_state"] == "advisory"
        assert changes[0]["to_state"] == "enforced"

    def test_python_annotated_form_change_detected(self) -> None:
        diff = (
            "+++ b/src/z.py\n"
            "@@ -1 +1 @@\n"
            "-LIFECYCLE_STATE_T22: GateLifecycleState = GateLifecycleState.ADVISORY\n"
            "+LIFECYCLE_STATE_T22: GateLifecycleState = GateLifecycleState.CALIBRATING\n"
        )
        changes = self._extract(diff)
        assert len(changes) == 1
        assert changes[0]["slug"] == "t22"

    def test_yaml_change_detected(self) -> None:
        diff = (
            "+++ b/config/fake_gate.yaml\n"
            "@@ -1 +1 @@\n"
            "-lifecycle_state: advisory\n"
            "+lifecycle_state: calibrating\n"
        )
        changes = self._extract(diff)
        assert changes == [
            {
                "slug": "fake_gate",
                "from_state": "advisory",
                "to_state": "calibrating",
                "source_path": "config/fake_gate.yaml",
            }
        ]

    def test_pure_addition_emits_no_change(self) -> None:
        # New constant landing — handled by the missing-constant check, not
        # the change check. Diff has + line but no matching - line.
        diff = "+++ b/src/x.py\n@@ -0,0 +1 @@\n+LIFECYCLE_STATE_T22 = GateLifecycleState.ADVISORY\n"
        changes = self._extract(diff)
        assert changes == []

    def test_identity_edit_emits_no_change(self) -> None:
        # Cosmetic edit (e.g., trailing whitespace) keeps from == to.
        diff = (
            "+++ b/src/x.py\n"
            "@@ -1 +1 @@\n"
            "-LIFECYCLE_STATE_T22 = GateLifecycleState.ADVISORY\n"
            "+LIFECYCLE_STATE_T22 = GateLifecycleState.ADVISORY\n"
        )
        changes = self._extract(diff)
        assert changes == []


class TestScanLifecycleChangesIntegration:
    """End-to-end: monkeypatch ``subprocess.check_output`` so the test does
    not need a real git repo."""

    @pytest.fixture
    def fake_repo_with_change_machinery(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> Path:
        (tmp_path / "docs" / "calibration").mkdir(parents=True)
        return tmp_path

    def _patch_subprocess(
        self,
        monkeypatch: pytest.MonkeyPatch,
        diff_text: str,
    ) -> None:
        def fake_check_output(*args: Any, **kwargs: Any) -> str:
            return diff_text

        monkeypatch.setattr(_scanner_mod.subprocess, "check_output", fake_check_output)

    def test_change_without_doc_emits_error(
        self,
        fake_repo_with_change_machinery: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        diff = (
            "+++ b/src/x.py\n"
            "@@ -1 +1 @@\n"
            "-LIFECYCLE_STATE_T22 = GateLifecycleState.ADVISORY\n"
            "+LIFECYCLE_STATE_T22 = GateLifecycleState.CALIBRATING\n"
        )
        self._patch_subprocess(monkeypatch, diff)
        findings = scan_lifecycle_changes(fake_repo_with_change_machinery, "origin/main")
        assert any(f.code == "missing_lifecycle_change_doc" for f in findings)

    def test_change_with_doc_passes(
        self,
        fake_repo_with_change_machinery: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        diff = (
            "+++ b/src/x.py\n"
            "@@ -1 +1 @@\n"
            "-LIFECYCLE_STATE_T22 = GateLifecycleState.ADVISORY\n"
            "+LIFECYCLE_STATE_T22 = GateLifecycleState.CALIBRATING\n"
        )
        self._patch_subprocess(monkeypatch, diff)
        # Plant a matching doc.
        (
            fake_repo_with_change_machinery
            / "docs"
            / "calibration"
            / "t22_lifecycle_change_advisory_to_calibrating_20260615.md"
        ).write_text("transition record\n", encoding="utf-8")
        findings = scan_lifecycle_changes(fake_repo_with_change_machinery, "origin/main")
        assert findings == []

    def test_enforced_doc_missing_required_fields_fails(
        self,
        fake_repo_with_change_machinery: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        diff = (
            "+++ b/src/x.py\n"
            "@@ -1 +1 @@\n"
            "-LIFECYCLE_STATE_T22 = GateLifecycleState.CALIBRATING\n"
            "+LIFECYCLE_STATE_T22 = GateLifecycleState.ENFORCED\n"
        )
        self._patch_subprocess(monkeypatch, diff)
        # Doc body lacks the 4 required fields.
        (
            fake_repo_with_change_machinery
            / "docs"
            / "calibration"
            / "t22_lifecycle_change_calibrating_to_enforced_20260615.md"
        ).write_text("just some prose\n", encoding="utf-8")
        findings = scan_lifecycle_changes(fake_repo_with_change_machinery, "origin/main")
        codes = {f.code for f in findings}
        assert "enforced_doc_missing_fields" in codes

    def test_enforced_doc_with_all_fields_passes(
        self,
        fake_repo_with_change_machinery: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        diff = (
            "+++ b/src/x.py\n"
            "@@ -1 +1 @@\n"
            "-LIFECYCLE_STATE_T22 = GateLifecycleState.CALIBRATING\n"
            "+LIFECYCLE_STATE_T22 = GateLifecycleState.ENFORCED\n"
        )
        self._patch_subprocess(monkeypatch, diff)
        (
            fake_repo_with_change_machinery
            / "docs"
            / "calibration"
            / "t22_lifecycle_change_calibrating_to_enforced_20260615.md"
        ).write_text(
            "start_date: 2026-06-15\n"
            "end_date: 2026-09-15\n"
            "drift_summary: |\n  No drift observed; ready to enforce.\n"
            "signing_reviewer: Erik Nunez\n",
            encoding="utf-8",
        )
        findings = scan_lifecycle_changes(fake_repo_with_change_machinery, "origin/main")
        assert findings == []


# ---------------------------------------------------------------------------
# Acceptance #7: CI workflow YAML loads cleanly.
# ---------------------------------------------------------------------------


def test_lifecycle_state_guard_workflow_yaml_is_valid() -> None:
    workflow_path = (
        Path(__file__).resolve().parents[3] / ".github" / "workflows" / "lifecycle_state_guard.yml"
    )
    assert workflow_path.is_file(), workflow_path
    with workflow_path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    assert isinstance(doc, dict)
    assert "jobs" in doc
    job = doc["jobs"]["lifecycle-state-guard"]
    assert job["runs-on"] == "ubuntu-latest"
    step_names = [s.get("name", "") for s in job["steps"] if isinstance(s, dict)]
    # Sanity: the two scanner steps are present.
    assert any("baseline" in n for n in step_names)
    assert any("change-detection" in n for n in step_names)


def test_lifecycle_state_guard_propagates_scanner_exit_code() -> None:
    """N2 finding H1: scanner steps MUST propagate the script exit code.

    `python ... > file.json` lets bash see the redirect succeed (file write
    rarely fails) even when the scanner returns 1, so the job prints findings
    then passes green. Both scanner steps must capture ``${PIPESTATUS[0]}``
    via ``tee`` and ``exit`` with that value.
    """
    workflow_path = (
        Path(__file__).resolve().parents[3] / ".github" / "workflows" / "lifecycle_state_guard.yml"
    )
    with workflow_path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    job = doc["jobs"]["lifecycle-state-guard"]
    scanner_steps = [
        s
        for s in job["steps"]
        if isinstance(s, dict) and "lifecycle-state scanner" in s.get("name", "")
    ]
    assert len(scanner_steps) == 2, (
        f"expected 2 scanner steps; got {[s.get('name') for s in scanner_steps]}"
    )
    for step in scanner_steps:
        run_block = step["run"]
        assert "tee" in run_block, f"step {step['name']!r} must pipe via `tee`, got: {run_block}"
        assert "PIPESTATUS[0]" in run_block, (
            f"step {step['name']!r} must capture ${{PIPESTATUS[0]}}, got: {run_block}"
        )
        assert 'exit "$EXIT"' in run_block or "exit $EXIT" in run_block, (
            f"step {step['name']!r} must propagate exit code, got: {run_block}"
        )


def test_lifecycle_state_guard_uses_event_before_for_push() -> None:
    """N2 finding M3: for push events, the workflow must use
    ``github.event.before`` (the SHA of the previous tip of the pushed
    branch) rather than HEAD~1, which only catches the last commit of a
    multi-commit push.
    """
    workflow_path = (
        Path(__file__).resolve().parents[3] / ".github" / "workflows" / "lifecycle_state_guard.yml"
    )
    with workflow_path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    job = doc["jobs"]["lifecycle-state-guard"]
    base_ref_step = next(
        (s for s in job["steps"] if isinstance(s, dict) and s.get("id") == "base-ref"),
        None,
    )
    assert base_ref_step is not None, "no `base-ref` step in workflow"
    env_block = base_ref_step.get("env", {})
    # The env var should reference github.event.before so the workflow
    # author can see the dependency at-a-glance.
    assert "EVENT_BEFORE" in env_block, base_ref_step
    assert "github.event.before" in env_block["EVENT_BEFORE"], env_block
    run_block = base_ref_step["run"]
    assert "EVENT_BEFORE" in run_block, run_block


# ---------------------------------------------------------------------------
# Repo-level smoke test: scanner passes against the actual repo.
# ---------------------------------------------------------------------------


def test_scanner_passes_against_real_repo() -> None:
    """After the bootstrap commits, the scanner should pass on the real repo
    with no findings.  Re-runs ``scan_python_modules`` + ``scan_yaml_configs``
    against the actual repo root.
    """
    repo_root = Path(__file__).resolve().parents[3]
    findings = scan_python_modules(repo_root)
    findings.extend(scan_yaml_configs(repo_root))
    errors = [f for f in findings if f.severity == "error"]
    assert errors == [], "scanner emitted unexpected errors against real repo: " + "; ".join(
        f"{f.path}: {f.code}" for f in errors
    )
