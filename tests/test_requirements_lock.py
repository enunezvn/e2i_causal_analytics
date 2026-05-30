"""Forcing-function tests for #534: the hash-pinned ``requirements.lock``.

Background
----------
``requirements.txt`` is a near-complete ``pip freeze`` snapshot: every line is
an exact ``==`` pin EXCEPT (a) the three ``mlflow`` lines, deliberately held as
a CVE-anchored range ``>=3.11.0,<3.12.0`` (see ``test_mlflow_upgrade_pin.py``),
and (b) two local patched packages installed from ``./patches/`` paths. Because
PyPI versions are immutable, the ``==`` lines are already reproducible; the real
rebuild-drift surface is the mlflow range, and there is no supply-chain
integrity (no artifact hashes). #534 (follow-up to #528 / #530 / #491) closes
both gaps with a compiled, hash-pinned ``requirements.lock`` consumed by
``docker/Dockerfile``.

Design
------
``requirements.lock`` is generated with::

    uv pip compile requirements.txt --generate-hashes \\
        --no-emit-package copilotkit --no-emit-package ag-ui-langgraph \\
        --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \\
        -o requirements.lock

``--no-emit-package`` drops the two local patch packages from the lock (a bare
local path cannot be hashed, and ``pip install --require-hashes`` is
all-or-nothing) while KEEPING their transitive deps, hashed. The Dockerfile then
installs the hash-locked closure with ``--require-hashes`` and the two patches
``--no-deps`` (their deps are already present from the lock).

These are FAST static guards (no network, no venv). The companion
``test_requirements_lock_installs_under_require_hashes`` (slow + integration)
proves the lock actually installs under ``pip --require-hashes``.
"""

from __future__ import annotations

import re
import subprocess
import venv
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

REPO_ROOT = Path(__file__).resolve().parents[1]
REQS_TXT = REPO_ROOT / "requirements.txt"
REQS_LOCK = REPO_ROOT / "requirements.lock"
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"

# Local patched packages installed from ./patches/ paths; excluded from the
# hash-locked file (installed separately --no-deps in the Dockerfile).
PATCH_PACKAGES = {canonicalize_name("copilotkit"), canonicalize_name("ag-ui-langgraph")}

# The mlflow range held in requirements.txt (kept in lockstep with
# test_mlflow_upgrade_pin.py::MLFLOW_REQUIRED_SPEC). The lock must pin each
# mlflow package to a single concrete version WITHIN this window.
MLFLOW_SPEC = SpecifierSet(">=3.11.0,<3.12.0")
MLFLOW_PACKAGES = {"mlflow", "mlflow-skinny", "mlflow-tracing"}

_REQ_RE = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<ver>[^\s;\\]+)")
_HASH_RE = re.compile(r"^--hash=")


def _parse_lock(text: str) -> tuple[dict[str, dict], list[str]]:
    """Parse a uv / pip-compile ``--generate-hashes`` lock.

    Returns ``(entries, anomalies)`` where ``entries`` maps canonical package
    name -> ``{"version": str, "hashes": list[str]}`` and ``anomalies`` is the
    list of non-comment, non-blank, non-hash lines that are NOT a
    ``name==version`` requirement (e.g. a bare local path, an editable, or a
    VCS / URL install) — any of which would break ``pip install
    --require-hashes``.
    """
    entries: dict[str, dict] = {}
    anomalies: list[str] = []
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if _HASH_RE.match(line):
            if current is not None:
                entries[current]["hashes"].append(line.split("=", 1)[1].rstrip(" \\"))
            continue
        m = _REQ_RE.match(line)
        if m:
            name = canonicalize_name(m.group("name"))
            entries[name] = {"version": m.group("ver"), "hashes": []}
            current = name
            continue
        # Anything else at requirement position is an anomaly.
        anomalies.append(line)
        current = None
    return entries, anomalies


def _parse_requirements_pins(text: str) -> dict[str, str]:
    """Return ``{canonical_name: version}`` for every exact ``==`` pin in a
    requirements file, skipping comments, blanks, ranges, and ``./path``
    installs."""
    pins: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()  # strip inline comments
        if not line or line.startswith("./") or line.startswith("-"):
            continue
        if "==" not in line:
            continue
        m = _REQ_RE.match(line)
        if m:
            pins[canonicalize_name(m.group("name"))] = m.group("ver")
    return pins


def test_lock_file_exists() -> None:
    assert REQS_LOCK.is_file(), (
        f"missing {REQS_LOCK.name}; generate it with `uv pip compile "
        f"requirements.txt --generate-hashes ...` (see module docstring)."
    )


def test_every_lock_requirement_is_hashed_and_no_bare_paths() -> None:
    entries, anomalies = _parse_lock(REQS_LOCK.read_text())
    assert entries, "lock parsed to zero packages; the lock format may have changed"
    assert not anomalies, (
        "requirements.lock contains non-hashable requirement lines (bare path, "
        "editable, or URL) that would break `pip install --require-hashes`:\n"
        + "\n".join(f"  {a}" for a in anomalies)
    )
    unhashed = sorted(n for n, e in entries.items() if not e["hashes"])
    assert not unhashed, (
        "every requirement in requirements.lock must carry at least one "
        f"--hash (else --require-hashes rejects the install). Unhashed: {unhashed}"
    )


def test_lock_excludes_local_patch_packages() -> None:
    entries, _ = _parse_lock(REQS_LOCK.read_text())
    present = PATCH_PACKAGES & set(entries)
    assert not present, (
        f"local patch packages {sorted(present)} must be EXCLUDED from "
        f"requirements.lock (they install --no-deps from ./patches/ in the "
        f"Dockerfile). Regenerate with --no-emit-package for each."
    )


def test_lock_pins_mlflow_concretely_within_range() -> None:
    entries, _ = _parse_lock(REQS_LOCK.read_text())
    for pkg in MLFLOW_PACKAGES:
        cname = canonicalize_name(pkg)
        assert cname in entries, f"{pkg} missing from requirements.lock"
        ver = entries[cname]["version"]
        assert ver in MLFLOW_SPEC, (
            f"{pkg}=={ver} in requirements.lock is outside the CVE-anchored "
            f"window {MLFLOW_SPEC}; the lock must pin within the source range "
            f"(lockstep with test_mlflow_upgrade_pin.py)."
        )


def test_lock_covers_every_requirements_txt_pin() -> None:
    pins = _parse_requirements_pins(REQS_TXT.read_text())
    entries, _ = _parse_lock(REQS_LOCK.read_text())
    missing = [name for name in pins if name not in entries]
    drifted = [
        f"{name}: requirements.txt=={ver} vs lock=={entries[name]['version']}"
        for name, ver in pins.items()
        if name in entries and entries[name]["version"] != ver
    ]
    assert not missing, (
        "requirements.lock is missing exact pins present in requirements.txt "
        f"(regenerate the lock): {sorted(missing)}"
    )
    assert not drifted, (
        "requirements.lock version-drifted from requirements.txt's exact pins "
        "(the lock must preserve the tested versions):\n" + "\n".join(drifted)
    )


def test_dockerfile_installs_from_hash_locked_file() -> None:
    text = DOCKERFILE.read_text()
    # The two image-building stages (dependencies + development) must each
    # install the hash-locked closure, not the loose requirements.txt.
    assert "requirements.lock" in text, "Dockerfile must reference requirements.lock"
    assert text.count("--require-hashes") >= 2, (
        "both the `dependencies` and `development` stages must install with "
        "`pip install --require-hashes -r requirements.lock`"
    )
    # No stage may still install the main closure from requirements.txt.
    assert not re.search(r"pip install[^\n]*-r\s+requirements\.txt", text), (
        "Dockerfile still installs from requirements.txt; the hash-locked "
        "requirements.lock must be the dependency source."
    )


def test_dockerfile_installs_patches_no_deps() -> None:
    text = DOCKERFILE.read_text()
    # Both patch packages install --no-deps (their deps are already in the lock).
    assert text.count("--no-deps") >= 2, (
        "the two local patches must install with --no-deps in both stages"
    )
    for patch in ("./patches/ag-ui-langgraph", "./patches/copilotkit"):
        assert patch in text, f"Dockerfile must install the local patch {patch}"


def test_dockerignore_keeps_requirements_lock_in_context() -> None:
    text = DOCKERIGNORE.read_text()
    assert "!requirements.lock" in text, (
        "requirements.lock must be explicitly re-included in .dockerignore "
        "(sibling to `!requirements.txt`) so COPY can see it in the build context."
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(900)
def test_requirements_lock_installs_under_require_hashes(tmp_path: Path) -> None:
    """``pip install --require-hashes --dry-run`` must accept requirements.lock.

    Proves the lock is hash-complete + internally consistent + resolvable in a
    clean venv — the structural guarantee ``--require-hashes`` enforces (every
    requirement pinned ``==`` and carrying a hash, the closure complete). The
    dry-run resolves and validates hash PRESENCE for the full graph; the
    per-artifact hash VERIFICATION happens at real image-build time, which the
    deploy pipeline runs as a monitored rebuild.

    To falsify: drop a ``--hash`` line from any lock entry — pip then exits
    non-zero with "In --require-hashes mode, all requirements must have their
    versions pinned with ==".

    Same hermeticity contract as ``tests/test_lockfile_resolves.py``: a fresh
    tempdir venv per run, ``--no-cache-dir --isolated``.
    """
    assert REQS_LOCK.is_file(), f"missing requirements file: {REQS_LOCK}"

    venv_root = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True, clear=True).create(str(venv_root))
    venv_python = venv_root / "bin" / "python"
    assert venv_python.is_file(), f"venv build failed: {venv_python} missing"

    result = subprocess.run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--no-cache-dir",
            "--isolated",
            "--require-hashes",
            "-r",
            str(REQS_LOCK),
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )

    if result.returncode != 0:
        tail = result.stderr[-2500:] if result.stderr else "(no stderr)"
        pytest.fail(
            "pip --require-hashes could not accept requirements.lock. Either a "
            "requirement lacks a hash, a transitive dep is missing from the "
            "closure, or two pins conflict.\n"
            f"exit code: {result.returncode}\n\n"
            "stderr (last 2.5 KB):\n"
            f"{tail}"
        )
