"""Pins for the two live discrepancies PR #1966 left behind (#1975, #1979).

Both are the *same shape* as the bug their issue was filed to close, which is
why they are pinned rather than just corrected:

- #1979 repointed three docstrings away from a non-existent protocol document
  and onto `docs/reports/causal-validation-pipeline-review-20260605.md`, which
  also does not exist -- the file lives under `docs/Archive/`. A citation fix
  that lands on another broken path is the original bug again.
- #1975 corrected the `causal_validation.gate_decisions` numbers to the live
  constants but did not touch `agents.causal_impact.validation`, which carried
  its own E-value cutoff. On 2026-09-10 the cutoff was removed outright: the
  sensitivity test is a benchmarked reading, not a gate, so there is no number
  for the YAML to mirror. The pin is now the *absence* of the cutoff on both
  sides -- a re-added `e_value_threshold` / `e_value_min` is the bug again.

These assert against the live constants and the real filesystem, so they fail
if either drifts again.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
AGENT_CONFIG = REPO_ROOT / "config" / "agent_config.yaml"

# The three modules #1979 repointed.
REPOINTED_MODULES = [
    REPO_ROOT / "src" / "causal_engine" / "refutation_runner.py",
    REPO_ROOT / "src" / "causal_engine" / "validation_outcome.py",
    REPO_ROOT / "src" / "causal_engine" / "validation_outcome_store.py",
]

# A docs/ path cited inside a module docstring.
DOC_REF = re.compile(r"(docs/[A-Za-z0-9_./-]+\.(?:md|html))")


def _module_docstring(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.find('"""')
    assert start != -1, f"{path.name} has no module docstring"
    end = text.find('"""', start + 3)
    return text[start + 3 : end]


class TestCitedDocsExist:
    """#1979: every docs/ path cited in these docstrings must resolve."""

    @pytest.mark.parametrize("module", REPOINTED_MODULES, ids=lambda p: p.name)
    def test_every_cited_doc_is_tracked_in_git(self, module):
        """Existing on THIS disk is not enough -- it must exist for everyone.

        #1966 repointed these docstrings at
        `docs/reports/causal-validation-pipeline-review-20260605.md`. The file
        is really at `docs/Archive/...`, but `docs/Archive/` is **gitignored**
        (.gitignore:125), so that document is not in the repository at all: it
        resolves only on a machine that happens to have a stale local copy.
        Checking `is_file()` alone would pass on such a machine and fail in CI
        and in a fresh clone -- so the pin is git-trackedness, not local
        presence.
        """
        cited = DOC_REF.findall(_module_docstring(module))
        assert cited, f"{module.name} cites no docs/ path -- the references block vanished"
        untracked = [
            ref
            for ref in cited
            if subprocess.run(
                ["git", "ls-files", "--error-unmatch", ref],
                cwd=REPO_ROOT,
                capture_output=True,
            ).returncode
            != 0
        ]
        assert not untracked, (
            f"{module.name} cites {untracked}, which are not tracked in git. A reader "
            f"on any other checkout cannot open them -- the same dead-citation bug "
            f"#1979 was filed to close."
        )

    def test_the_protocol_document_is_still_gone(self):
        """Positive control: the ORIGINAL bad reference must stay absent.

        Without this the test above could pass by the references block being
        deleted wholesale rather than corrected.
        """
        assert not (REPO_ROOT / "docs" / "E2I_Causal_Validation_Protocol.html").is_file()
        for module in REPOINTED_MODULES:
            assert "E2I_Causal_Validation_Protocol" not in module.read_text(encoding="utf-8")


class TestConfigMirrorsLiveConstants:
    """#1975: documentation-only YAML must not contradict the live constants."""

    @staticmethod
    def _cfg():
        with AGENT_CONFIG.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    @staticmethod
    def _validation_block(cfg):
        return cfg["agents"]["causal_impact"]["validation"]

    @staticmethod
    def _runner_constant(name):
        """Read a class-level constant literal straight out of the source.

        Deliberately NOT `from src.causal_engine.refutation_runner import
        RefutationRunner`: that import pulls scipy/dowhy, which makes the test
        unrunnable wherever those are absent and expensive on a memory-capped
        box. Both sides of this comparison are declared literals, so reading the
        literal is the faithful check -- and it keeps the test hermetic.
        """
        src = (REPO_ROOT / "src" / "causal_engine" / "refutation_runner.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "RefutationRunner":
                for stmt in node.body:
                    targets = (
                        stmt.targets
                        if isinstance(stmt, ast.Assign)
                        else [stmt.target]
                        if isinstance(stmt, ast.AnnAssign)
                        else []
                    )
                    for t in targets:
                        if isinstance(t, ast.Name) and t.id == name:
                            return ast.literal_eval(stmt.value)
        raise AssertionError(f"RefutationRunner.{name} not found in source")

    def test_the_e_value_cutoff_is_gone_from_runner_and_yaml(self):
        """2026-09-10 (spec sensitivity-gate-calibration §4.5): the sensitivity test
        is a benchmarked reading with no cutoff. Neither the runner nor the YAML may
        carry an ``e_value_threshold`` / ``e_value_min`` again."""
        pass_thresholds = self._runner_constant("PASS_THRESHOLDS")
        assert "e_value_min" not in pass_thresholds
        default_config = self._runner_constant("DEFAULT_CONFIG")
        assert default_config["sensitivity_e_value"]["critical"] is False
        assert "e_value_threshold" not in default_config["sensitivity_e_value"]
        block = self._validation_block(self._cfg())
        assert "e_value_threshold" not in block
        assert block["sensitivity_rule"] == "measured_confounding_benchmark"

    def test_gate_decision_numbers_still_mirror_the_runner(self):
        """Guards the part #1966 DID fix, so it cannot silently regress."""
        live = self._runner_constant("GATE_THRESHOLDS")
        gate = self._cfg()["causal_validation"]["gate_decisions"]
        assert gate["proceed_threshold"] == live["proceed"]
        assert gate["review_threshold"] == live["review"]

    def test_documentation_only_block_says_so(self):
        """The block is unconsumed; a reader must not take it for live config.

        `grep -rn "e_value_threshold" src/` finds nothing since 2026-09-10 -- the
        cutoff is gone and the block names the rule instead.
        `min_refutation_pass_rate`'s only src/ hits are in src/ml/synthetic/, an
        unrelated validator with its own Python default -- a name collision, not a
        consumer of this key.
        """
        text = AGENT_CONFIG.read_text(encoding="utf-8").splitlines()
        idx = next(i for i, line in enumerate(text) if line.strip().startswith("sensitivity_rule:"))
        preceding = "\n".join(text[max(0, idx - 12) : idx]).lower()
        assert "documentation-only" in preceding or "not consumed" in preceding, (
            "the causal_impact.validation block carries live-looking numbers but is "
            "read by nothing -- it must say so, as the gate_decisions block now does"
        )


class TestOurOwnCitationsResolve:
    """The citations this branch itself writes must resolve (codex, #1979 arc).

    Both started as line numbers and were stale on arrival: each was written
    against origin/main and then shifted by a sibling edit in the SAME commit --
    the YAML cited refutation_runner.py:442 (really 441, after a docstring line
    was removed) and the causal_validation docstring cited the SQL at :396
    (really 403, after this branch added lines above it). Writing a dead citation
    inside the change that exists to kill dead citations is exactly the trap, so
    they are symbol anchors now and this test keeps them honest.
    """

    def test_the_yaml_anchor_names_a_real_constant(self):
        cfg = AGENT_CONFIG.read_text(encoding="utf-8")
        assert 'RefutationRunner.DEFAULT_CONFIG["sensitivity_e_value"]' in cfg
        assert "src/causal_engine/evalue.py" in cfg
        runner = (REPO_ROOT / "src" / "causal_engine" / "refutation_runner.py").read_text(
            encoding="utf-8"
        )
        assert "DEFAULT_CONFIG" in runner and '"sensitivity_e_value"' in runner
        assert '"e_value_threshold"' not in runner, (
            "the cutoff was removed on 2026-09-10; do not re-add it"
        )
        assert (REPO_ROOT / "src" / "causal_engine" / "evalue.py").is_file()

    def test_the_migration_anchor_names_a_real_sql_function(self):
        """#1971 retired the Python ``can_use_estimate`` (and its docstring), so
        the symbol anchor now lives in migration 133, which drops the SQL one.
        The anchor must still resolve to the function ml/010 defines."""
        migration = (
            REPO_ROOT / "database" / "migrations" / "133_retire_can_use_estimate.sql"
        ).read_text(encoding="utf-8")
        assert "CREATE OR REPLACE FUNCTION can_use_estimate" in migration, (
            "the migration should cite the baseline definition by name, not by line"
        )
        sql = (REPO_ROOT / "database" / "ml" / "010_causal_validation_tables.sql").read_text(
            encoding="utf-8"
        )
        assert "CREATE OR REPLACE FUNCTION can_use_estimate" in sql

    def test_no_bare_line_number_citations_were_reintroduced(self):
        """Line-number citations drift; these two already did, in one commit."""
        import re as _re

        for path in (
            AGENT_CONFIG,
            REPO_ROOT / "src" / "repositories" / "causal_validation.py",
        ):
            text = path.read_text(encoding="utf-8")
            hits = _re.findall(r"(?:\.py|\.sql):\d+", text)
            assert not hits, f"{path.name} reintroduced line citations: {hits}"
