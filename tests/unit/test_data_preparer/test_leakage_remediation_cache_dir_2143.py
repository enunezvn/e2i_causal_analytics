"""#2143: the leakage-remediation cache must not require a writable cwd.

``_get_cache_dir`` used ``Path(".cache") / "leakage_remediation"`` and
``mkdir``-ed it. In the ``e2i_api`` container the rootfs is read-only and the
cwd is ``/app``, so the mkdir raised, the node's outer ``except`` turned that
into ``leakage_remediation_status="error"``, and remediation never ran
(measured 2026-09-15 during the #2120 Tier-0 attempt). The cache only saves
repeat LLM calls, so an unwritable location must never cost the remediation.

The read-only cwd here is a ``chmod 0o555`` directory, so ``mkdir`` raises
``PermissionError`` rather than the container's ``EROFS``; both are
``OSError`` on the same line.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.agents.ml_foundation.data_preparer.nodes import leakage_remediation as lr


@pytest.fixture
def read_only_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ro = tmp_path / "ro_cwd"
    ro.mkdir()
    ro.chmod(0o555)
    if os.access(ro, os.W_OK):  # root ignores mode bits; the premise would not hold
        ro.chmod(0o755)
        pytest.skip("cannot make a read-only directory as this user")
    monkeypatch.chdir(ro)
    yield ro
    ro.chmod(0o755)


@pytest.fixture
def isolated_tempdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    monkeypatch.delenv("E2I_CACHE_DIR", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    return tmp


def _leaky_state() -> dict:
    """A CRITICAL logical-dependency leak the node drops without an LLM."""
    rng = np.random.default_rng(2143)
    n = 400
    target = rng.integers(0, 2, n)
    train_df = pd.DataFrame(
        {
            "target": target,
            "leak": target,
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "f3": rng.normal(size=n),
        }
    )
    return {
        "train_df": train_df,
        "validation_df": None,
        "test_df": None,
        "holdout_df": None,
        "scope_spec": {"prediction_target": "target", "problem_type": "binary_classification"},
        "leakage_severity": "critical",
        "leaked_features": ["leak"],
        "leakage_findings": [
            {"feature": "leak", "check_name": "logical_dependency", "severity": "CRITICAL"}
        ],
        "leakage_remediation_attempts": 0,
        "experiment_id": "exp-2143",
    }


class TestLeakageRemediationCacheDir:
    async def test_node_remediates_with_a_read_only_cwd(self, read_only_cwd, isolated_tempdir):
        result = await lr.review_and_remediate_leakage(_leaky_state())

        assert result["leakage_remediation_status"] == "applied", result.get(
            "leakage_remediation_reasoning"
        )
        assert "leak" in result["leakage_dropped_features"]
        assert not (read_only_cwd / ".cache").exists()
        cached = list((isolated_tempdir / "e2i_cache" / "leakage_remediation").glob("*.json"))
        assert len(cached) == 1, cached

    def test_writable_cwd_keeps_the_existing_cache_location(self, tmp_path, monkeypatch):
        monkeypatch.delenv("E2I_CACHE_DIR", raising=False)
        monkeypatch.chdir(tmp_path)

        assert lr._get_cache_dir() == tmp_path / ".cache" / "leakage_remediation"

    def test_configured_cache_root_wins(self, tmp_path, monkeypatch, read_only_cwd):
        root = tmp_path / "configured"
        monkeypatch.setenv("E2I_CACHE_DIR", str(root))

        assert lr._get_cache_dir() == root / "leakage_remediation"

    def test_unwritable_configured_root_falls_back(
        self, tmp_path, monkeypatch, read_only_cwd, isolated_tempdir
    ):
        monkeypatch.setenv("E2I_CACHE_DIR", str(read_only_cwd / "nope"))

        assert lr._get_cache_dir() == isolated_tempdir / "e2i_cache" / "leakage_remediation"

    def test_existing_but_unwritable_cache_dir_falls_back(
        self, tmp_path, monkeypatch, isolated_tempdir
    ):
        """``mkdir(exist_ok=True)`` succeeds on a directory that already exists
        on a read-only filesystem, so existence is not writability."""
        cwd = tmp_path / "cwd"
        existing = cwd / ".cache" / "leakage_remediation"
        existing.mkdir(parents=True)
        existing.chmod(0o555)
        try:
            if os.access(existing, os.W_OK):
                pytest.skip("cannot make a read-only directory as this user")
            monkeypatch.chdir(cwd)
            assert lr._get_cache_dir() == isolated_tempdir / "e2i_cache" / "leakage_remediation"
        finally:
            existing.chmod(0o755)

    def test_no_writable_location_runs_uncached(
        self, tmp_path, monkeypatch, read_only_cwd, isolated_tempdir
    ):
        isolated_tempdir.chmod(0o555)
        try:
            assert lr._get_cache_dir() is None
            assert lr._load_cached_analysis("k") is None
            lr._save_cached_analysis("k", {"viable": True})  # must not raise
        finally:
            isolated_tempdir.chmod(0o755)
