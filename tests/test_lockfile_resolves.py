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

import subprocess
import venv
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REQS_DEV = REPO_ROOT / "requirements-dev.txt"

# Resolver dry-run against a clean venv can take 60-180s the first time
# (cold metadata fetch). Override the global 30s pytest timeout.
RESOLVER_TIMEOUT_S = 600


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(RESOLVER_TIMEOUT_S)
def test_requirements_dev_resolves(tmp_path: Path) -> None:
    """pip's strict resolver must satisfy every pin in requirements-dev.txt.

    Hermeticity: builds a fresh tempdir venv per run via ``venv.EnvBuilder``
    and passes ``--no-cache-dir --isolated`` to pip. This avoids false
    positives or negatives caused by packages installed in the host venv
    or by user-level pip configuration. To falsify: revert this PR's
    feast removal from requirements-dev.txt — the test then trips with
    ``ResolutionImpossible: feast and numpy==2.3.5``.

    Failure shape: one of the pinned packages declares a ``Requires-Dist``
    constraint that contradicts another pin (or contradicts what pip would
    otherwise pick to satisfy a downstream pin). pip exits non-zero with a
    "ResolutionImpossible" error in stderr.

    Coverage gap: this test only checks ``requirements-dev.txt`` in
    isolation. The Tier 1-5 harness install path is
    ``pip install -r requirements.txt && pip install -r requirements-dev.txt``
    (two invocations, not one). A future expansion could mirror that
    sequential install — but the second invocation silently reconciles
    cross-file pin drift (e.g. bentoml in requirements.txt vs
    requirements-dev.txt), so single-file resolution remains the
    primary guard against transitive-constraint regressions.
    """
    assert REQS_DEV.is_file(), f"missing requirements file: {REQS_DEV}"

    # Step 1: build a fresh venv (without pip upgrade to save time;
    # the bundled pip is recent enough for --dry-run).
    venv_root = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True, clear=True).create(str(venv_root))
    venv_python = venv_root / "bin" / "python"
    assert venv_python.is_file(), f"venv build failed: {venv_python} missing"

    # Step 2: run pip dry-run inside the fresh venv. ``--isolated``
    # ignores user-level pip config; ``--no-cache-dir`` skips the wheel
    # cache so the resolver always re-fetches metadata.
    result = subprocess.run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--no-cache-dir",
            "--isolated",
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
