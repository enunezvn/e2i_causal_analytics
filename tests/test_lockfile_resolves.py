"""Forcing-function test: requirements-dev.txt must resolve cleanly under pip.

This test is the regression guard for the class of bug that bit
PR #305 (dill/multiprocess), PR #308 (feast/protobuf — and the latent
numpy/pyarrow/tenacity conflicts that PR #308 did NOT clear), and
issue #307 (layered-conflict diagnosis): a transitive constraint of a
pinned package conflicts with another pin in the lockfile.

Why a test rather than a workflow step: tests run on every
``paths-ignore``-style mismatch a workflow may have. The Tier 1-5 harness
workflow's ``paths:`` filter excludes ``requirements-dev.txt``, so changes
to the lockfile silently bypass the harness install step. Codifying the
resolver invariant as a test means it is exercised on the same CI shard
that gates every backend change.

Implementation note: ``pip install --dry-run`` performs the full
resolution graph (downloading metadata, not packages) without installing.
Exit code is the resolver verdict.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REQS_DEV = REPO_ROOT / "requirements-dev.txt"

# Resolver dry-run can take 30-90s against cold pypi.org. Override the
# global 30s pytest timeout.
RESOLVER_TIMEOUT_S = 300


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(RESOLVER_TIMEOUT_S)
def test_requirements_dev_resolves() -> None:
    """pip's strict resolver must satisfy every pin in requirements-dev.txt.

    Failure shape: one of the pinned packages declares a ``Requires-Dist``
    constraint that contradicts another pin (or contradicts what pip would
    otherwise pick to satisfy a downstream pin). pip exits non-zero with a
    "ResolutionImpossible" error.

    Falsifiability: revert any lockfile-conflicting change (e.g.
    re-add ``feast[postgres]==0.44.0`` while ``numpy==2.3.5`` is pinned)
    and this test will trip.
    """
    assert REQS_DEV.is_file(), f"missing requirements file: {REQS_DEV}"
    assert shutil.which("pip") is not None or sys.executable, "no python/pip available"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--no-cache-dir",
            "--quiet",
            "-r",
            str(REQS_DEV),
        ],
        capture_output=True,
        text=True,
        timeout=RESOLVER_TIMEOUT_S,
    )

    if result.returncode != 0:
        # pip's conflict report is at the end of stderr; clip to last ~2KB
        # to keep the failure surface readable.
        tail = result.stderr[-2000:] if result.stderr else "(no stderr)"
        pytest.fail(
            "pip strict resolver could not satisfy requirements-dev.txt. "
            "A pinned package's transitive constraint conflicts with "
            "another pin (or with what pip would otherwise resolve).\n"
            f"exit code: {result.returncode}\n\n"
            "stderr (last 2 KB):\n"
            f"{tail}"
        )
