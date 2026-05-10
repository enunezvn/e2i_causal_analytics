"""Unit tests for ``scripts/check_g2_prior_runs.py``.

HIGH-5 (iter-3): the durable prior-run check uses a GitHub API run
search keyed by the S_prespec SHA appearing in each prior run's tag
commit message. Pin the filter logic via inline runs / commit-messages
fixtures (no live GitHub).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import check_g2_prior_runs as P  # noqa: E402

S_PRESPEC = "7f616f6f7f616f6f7f616f6f7f616f6f7f616f6f"
S_PRESPEC_SHORT = "7f616f6f"


def _run(
    db_id: int,
    head_sha: str,
    *,
    name: str = "G2 refs/tags/tier1b-b2-experiment-1",
    status: str = "completed",
    conclusion: str = "success",
) -> Dict[str, Any]:
    return {
        "databaseId": db_id,
        "headSha": head_sha,
        "name": name,
        "displayTitle": name,
        "status": status,
        "conclusion": conclusion,
        "headBranch": "main",
        "createdAt": "2026-05-10T00:00:00Z",
    }


class TestFilterPriorRuns:
    """The filter returns prior runs whose tag commit message
    references S_prespec AND that are NOT the current run."""

    def test_no_prior_runs_returns_empty(self) -> None:
        prior = P.filter_prior_runs_for_prespec(
            [],
            s_prespec=S_PRESPEC,
            s_prespec_short=S_PRESPEC_SHORT,
            commit_messages={},
        )
        assert prior == []

    def test_single_prior_run_with_matching_message_is_returned(self) -> None:
        runs: List[Dict[str, Any]] = [
            _run(101, "abc123"),
        ]
        prior = P.filter_prior_runs_for_prespec(
            runs,
            s_prespec=S_PRESPEC,
            s_prespec_short=S_PRESPEC_SHORT,
            commit_messages={"abc123": f"Tag commit referencing S_prespec={S_PRESPEC_SHORT}"},
            current_run_id="999",
        )
        assert len(prior) == 1
        assert prior[0]["databaseId"] == 101

    def test_message_with_full_sha_is_matched(self) -> None:
        runs: List[Dict[str, Any]] = [
            _run(102, "def456"),
        ]
        prior = P.filter_prior_runs_for_prespec(
            runs,
            s_prespec=S_PRESPEC,
            s_prespec_short=S_PRESPEC_SHORT,
            commit_messages={"def456": f"References full SHA {S_PRESPEC} explicitly"},
            current_run_id="999",
        )
        assert len(prior) == 1

    def test_message_without_s_prespec_token_does_not_match(self) -> None:
        runs: List[Dict[str, Any]] = [
            _run(103, "xyz789"),
        ]
        prior = P.filter_prior_runs_for_prespec(
            runs,
            s_prespec=S_PRESPEC,
            s_prespec_short=S_PRESPEC_SHORT,
            commit_messages={"xyz789": "Unrelated commit message — no S_prespec ref"},
        )
        assert prior == []

    def test_current_run_excluded_by_run_id(self) -> None:
        runs: List[Dict[str, Any]] = [
            _run(101, "abc123"),  # the current run, even though it matches
            _run(202, "def456"),  # a different prior run
        ]
        commits = {
            "abc123": f"S_prespec={S_PRESPEC_SHORT}",
            "def456": f"Also references {S_PRESPEC_SHORT}",
        }
        prior = P.filter_prior_runs_for_prespec(
            runs,
            s_prespec=S_PRESPEC,
            s_prespec_short=S_PRESPEC_SHORT,
            commit_messages=commits,
            current_run_id="101",
        )
        assert len(prior) == 1
        assert prior[0]["databaseId"] == 202

    def test_current_run_excluded_by_head_sha(self) -> None:
        """When the API returns a self-record before the current step
        knows its own ID, the headSha filter still excludes it."""
        runs: List[Dict[str, Any]] = [
            _run(101, "abc123"),
            _run(202, "abc123"),  # SAME headSha → both self-or-mate
        ]
        commits = {"abc123": f"S_prespec={S_PRESPEC_SHORT}"}
        prior = P.filter_prior_runs_for_prespec(
            runs,
            s_prespec=S_PRESPEC,
            s_prespec_short=S_PRESPEC_SHORT,
            commit_messages=commits,
            current_head_sha="abc123",
        )
        # All runs share the head_sha → all excluded by current_head_sha filter.
        assert prior == []

    def test_multiple_matching_prior_runs_are_all_returned(self) -> None:
        runs: List[Dict[str, Any]] = [
            _run(202, "child1"),
            _run(303, "child2"),
            _run(404, "child3"),
        ]
        commits = {
            "child1": f"first attempt, S_prespec={S_PRESPEC_SHORT}",
            "child2": f"second attempt, S_prespec={S_PRESPEC_SHORT}",
            "child3": f"third attempt, S_prespec={S_PRESPEC_SHORT}",
        }
        prior = P.filter_prior_runs_for_prespec(
            runs,
            s_prespec=S_PRESPEC,
            s_prespec_short=S_PRESPEC_SHORT,
            commit_messages=commits,
            current_run_id="999",
        )
        assert len(prior) == 3

    def test_missing_commit_message_treated_as_skip(self) -> None:
        """When the API doesn't return a commit message for a headSha,
        we conservatively skip that record (it MAY or MAY NOT be a
        sibling). Caller should re-query."""
        runs: List[Dict[str, Any]] = [
            _run(202, "unfetched"),
        ]
        prior = P.filter_prior_runs_for_prespec(
            runs,
            s_prespec=S_PRESPEC,
            s_prespec_short=S_PRESPEC_SHORT,
            commit_messages={},  # no message for "unfetched"
        )
        assert prior == []


class TestMain:
    """End-to-end main() with --runs-json + --commit-messages-json."""

    def test_main_returns_zero_when_no_prior_runs(self, capsys: pytest.CaptureFixture) -> None:
        rc = P.main(
            [
                "--s-prespec",
                S_PRESPEC,
                "--s-prespec-short",
                S_PRESPEC_SHORT,
                "--runs-json",
                "[]",
                "--commit-messages-json",
                "{}",
            ]
        )
        assert rc == 0
        captured = capsys.readouterr()
        assert "no prior runs" in captured.out

    def test_main_returns_one_when_prior_run_found(self, capsys: pytest.CaptureFixture) -> None:
        runs = [_run(101, "abc")]
        commits = {"abc": f"refs S_prespec {S_PRESPEC_SHORT}"}
        rc = P.main(
            [
                "--s-prespec",
                S_PRESPEC,
                "--s-prespec-short",
                S_PRESPEC_SHORT,
                "--runs-json",
                json.dumps(runs),
                "--commit-messages-json",
                json.dumps(commits),
                "--current-run-id",
                "999",
            ]
        )
        assert rc == 1
        captured = capsys.readouterr()
        assert "prior workflow run(s) detected" in captured.err
        assert "101" in captured.err

    def test_main_returns_zero_when_only_current_run_present(self) -> None:
        """The current run's own record in the API is ignored."""
        runs = [_run(101, "abc")]
        commits = {"abc": f"refs S_prespec {S_PRESPEC_SHORT}"}
        rc = P.main(
            [
                "--s-prespec",
                S_PRESPEC,
                "--s-prespec-short",
                S_PRESPEC_SHORT,
                "--runs-json",
                json.dumps(runs),
                "--commit-messages-json",
                json.dumps(commits),
                "--current-run-id",
                "101",
            ]
        )
        assert rc == 0

    def test_main_rejects_invalid_runs_json(self) -> None:
        rc = P.main(
            [
                "--s-prespec",
                S_PRESPEC,
                "--s-prespec-short",
                S_PRESPEC_SHORT,
                "--runs-json",
                "not valid json",
                "--commit-messages-json",
                "{}",
            ]
        )
        assert rc == 2

    def test_main_rejects_runs_json_that_is_not_a_list(self) -> None:
        rc = P.main(
            [
                "--s-prespec",
                S_PRESPEC,
                "--s-prespec-short",
                S_PRESPEC_SHORT,
                "--runs-json",
                '{"not": "a list"}',
                "--commit-messages-json",
                "{}",
            ]
        )
        assert rc == 2
