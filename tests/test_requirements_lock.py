"""Forcing-function tests for #534: the hash-pinned ``requirements.lock``.

Background
----------
``requirements.txt`` is a near-complete ``pip freeze`` snapshot: every line is
an exact ``==`` pin EXCEPT (a) the three ``mlflow`` lines, deliberately held as
a CVE-anchored range ``>=3.15.1,<3.16.0`` (see ``test_mlflow_upgrade_pin.py``),
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
import tomllib
import venv
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

REPO_ROOT = Path(__file__).resolve().parents[1]
REQS_TXT = REPO_ROOT / "requirements.txt"
REQS_LOCK = REPO_ROOT / "requirements.lock"
PYPROJECT = REPO_ROOT / "pyproject.toml"
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"

# Local patched packages installed from ./patches/ paths; excluded from the
# hash-locked file (installed separately --no-deps in the Dockerfile).
PATCH_PACKAGES = {canonicalize_name("copilotkit"), canonicalize_name("ag-ui-langgraph")}

# The mlflow range held in requirements.txt (kept in lockstep with
# test_mlflow_upgrade_pin.py::MLFLOW_REQUIRED_SPEC). The lock must pin each
# mlflow package to a single concrete version WITHIN this window.
MLFLOW_SPEC = SpecifierSet(">=3.15.1,<3.16.0")
MLFLOW_PACKAGES = {"mlflow", "mlflow-skinny", "mlflow-tracing"}

_REQ_RE = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<ver>[^\s;\\]+)")
_HASH_RE = re.compile(r"^--hash=")

# The local patch packages live here and declare their runtime deps in a Poetry
# pyproject.toml. The Dockerfile installs them with --no-deps, so a dep a patch
# adds without regenerating the lock would NOT fail the build — it would fail
# silently at import time. These guards (test_lock_covers_patch_package_dependencies)
# assert every dep a patch actually installs is present in the lock at a
# satisfying version. INSTALLED_PATCH_EXTRAS records which Poetry extras the
# Dockerfile requests: ag-ui-langgraph is installed as `[fastapi]`, copilotkit
# with NO extra (its `crewai` optional dep is therefore not installed and not
# expected in the lock).
PATCH_DIRS = {
    canonicalize_name("ag-ui-langgraph"): REPO_ROOT / "patches" / "ag-ui-langgraph",
    canonicalize_name("copilotkit"): REPO_ROOT / "patches" / "copilotkit",
}
INSTALLED_PATCH_EXTRAS = {
    canonicalize_name("ag-ui-langgraph"): {"fastapi"},
    canonicalize_name("copilotkit"): set(),
}


def _caret_to_specifier(version: str) -> str:
    """Convert a Poetry caret constraint body (e.g. ``0.1.10``) to a PEP 440
    range. Caret allows changes that do not modify the left-most non-zero
    component: ``^1.2.3`` -> ``>=1.2.3,<2.0.0``; ``^0.1.10`` -> ``>=0.1.10,<0.2.0``;
    ``^0.0.3`` -> ``>=0.0.3,<0.0.4``."""
    parts = [int(x) for x in version.split(".")]
    parts += [0] * (3 - len(parts))
    major, minor, patch = parts[0], parts[1], parts[2]
    if major > 0:
        upper = f"{major + 1}.0.0"
    elif minor > 0:
        upper = f"0.{minor + 1}.0"
    else:
        upper = f"0.0.{patch + 1}"
    return f">={version},<{upper}"


def _poetry_constraint_to_specifierset(constraint: str) -> SpecifierSet:
    """Translate the subset of Poetry version syntax the patches actually use
    (caret ``^``, and PEP 440 operators ``>= <= == != > <``, comma-joined) into a
    ``packaging`` SpecifierSet. Raises on any unhandled form so the guard fails
    loudly rather than silently passing an unparseable constraint."""
    pieces = []
    for part in (p.strip() for p in constraint.split(",")):
        if part.startswith("^"):
            pieces.append(_caret_to_specifier(part[1:]))
        elif part.startswith((">=", "<=", "==", "!=", "~=", ">", "<")):
            pieces.append(part)
        else:
            raise ValueError(f"unhandled Poetry version constraint: {constraint!r}")
    return SpecifierSet(",".join(pieces))


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
        # uv's ``--emit-index-url`` writes ``--index-url`` / ``--extra-index-url``
        # directives into the lock. They are REQUIRED so ``pip install
        # --require-hashes`` can find the pytorch CPU wheel (``torch==2.9.1+cpu``
        # lives only on download.pytorch.org/whl/cpu, not PyPI). These are index
        # directives, not requirements, and do not break ``--require-hashes`` —
        # skip them (an editable / bare path / VCS line is still an anomaly).
        if line.startswith(("--index-url", "--extra-index-url")):
            current = None
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


def test_lock_covers_patch_package_dependencies() -> None:
    """Every dep the two ./patches/ packages actually install must be present in
    requirements.lock at a satisfying version.

    The Dockerfile installs the patches with ``--no-deps`` (their deps are meant
    to be satisfied by the hash-locked closure), so a dep a patch adds WITHOUT a
    lock regeneration would not fail the image build — it would surface only as a
    runtime ImportError. This guard closes that gap (the exact transitive-drift
    class #534 targets) for the patch packages specifically.

    To falsify: add a new dependency (e.g. ``respx = ">=0.20"``) to
    ``patches/copilotkit/pyproject.toml`` without regenerating the lock — this
    test then reports ``respx`` ABSENT from requirements.lock.

    Skipped per dep: ``python`` itself, local patch-to-patch deps (copilotkit ->
    ag-ui-langgraph, installed separately), path deps, and optional deps whose
    Poetry extra the Dockerfile does not request (copilotkit's ``crewai``).
    """
    entries, _ = _parse_lock(REQS_LOCK.read_text())
    problems: list[str] = []

    for pkg_cname, patch_dir in PATCH_DIRS.items():
        pyproject = patch_dir / "pyproject.toml"
        assert pyproject.is_file(), f"missing patch pyproject: {pyproject}"
        poetry = tomllib.loads(pyproject.read_text())["tool"]["poetry"]
        deps = poetry.get("dependencies", {})
        extras = poetry.get("extras", {})

        # Canonical dep names reachable through the extras the Dockerfile requests.
        installed_extra_deps: set[str] = set()
        for extra in INSTALLED_PATCH_EXTRAS[pkg_cname]:
            for dep_name in extras.get(extra, []):
                installed_extra_deps.add(canonicalize_name(dep_name))

        for name, spec in deps.items():
            if name == "python":
                continue
            cname = canonicalize_name(name)
            if cname in PATCH_PACKAGES:
                continue  # local patch-to-patch dep, installed separately --no-deps

            if isinstance(spec, dict):
                if spec.get("path"):
                    continue  # local path dependency, not a PyPI artifact to lock
                optional = bool(spec.get("optional", False))
                version = spec.get("version")
            else:
                optional = False
                version = spec

            if optional and cname not in installed_extra_deps:
                continue  # optional dep whose extra the Dockerfile does not install

            if cname not in entries:
                problems.append(
                    f"{pkg_cname}: installs '{name}' ({version or 'any'}) but it is "
                    f"ABSENT from requirements.lock (regenerate the lock)"
                )
                continue
            if version:
                specset = _poetry_constraint_to_specifierset(str(version))
                locked = entries[cname]["version"]
                if locked not in specset:
                    problems.append(
                        f"{pkg_cname}: requires '{name}' {version} but the lock pins "
                        f"{locked} (outside {specset})"
                    )

    assert not problems, (
        "patch package dependencies drifted from requirements.lock — the "
        "--no-deps patch install would fail at import time, not build time:\n"
        + "\n".join(f"  {p}" for p in problems)
    )


def _dockerfile_flat() -> str:
    """Dockerfile text with shell line-continuations collapsed, so a
    ``pip install \\`` + newline + ``-r requirements.txt`` cannot evade a
    single-line ``[^\\n]*`` regex (otherwise the guard would pass on a
    multi-line regression)."""
    return re.sub(r"\\\s*\n\s*", " ", DOCKERFILE.read_text())


def test_dockerfile_installs_from_hash_locked_file() -> None:
    text = _dockerfile_flat()
    # The two image-building stages (dependencies + development) must each
    # install the hash-locked closure, not the loose requirements.txt.
    assert "requirements.lock" in text, "Dockerfile must reference requirements.lock"
    assert text.count("--require-hashes") >= 2, (
        "both the `dependencies` and `development` stages must install with "
        "`pip install --require-hashes -r requirements.lock`"
    )
    # No stage may still install the main closure from requirements.txt
    # (matched on the continuation-collapsed text, so a multi-line RUN cannot hide it).
    assert not re.search(r"pip install[^\n]*-r\s+requirements\.txt", text), (
        "Dockerfile still installs from requirements.txt; the hash-locked "
        "requirements.lock must be the dependency source."
    )


def test_dockerfile_installs_patches_no_deps() -> None:
    text = _dockerfile_flat()
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


BENTOML_REQS = REPO_ROOT / "docker" / "bentoml" / "requirements-bentoml.txt"


def _declared_dowhy_requirements() -> list[tuple[str, Requirement]]:
    """Every dowhy version spec the repo declares as a RESOLVER INPUT.

    Sources:

    * ``pyproject.toml [project.dependencies]`` — what uv/pip resolve for the
      dev + tier1-5 harness environments.
    * ``docker/bentoml/requirements-bentoml.txt`` — installed directly by
      ``docker/bentoml/Dockerfile`` into the BentoML serving image (a causal
      Bento service is wired in ``docker/bentoml/docker-compose.yaml``).

    The third dowhy-declaring surface — the generated causal Bento package
    list in ``scripts/deploy_model.py`` — is code, guarded by
    ``tests/unit/test_scripts/test_deploy_model_service_packages.py``.
    """
    found: list[tuple[str, Requirement]] = []
    for dep in tomllib.loads(PYPROJECT.read_text())["project"]["dependencies"]:
        req = Requirement(dep)
        if canonicalize_name(req.name) == "dowhy":
            found.append(("pyproject.toml", req))
    for raw in BENTOML_REQS.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith(("-", "./")):
            continue
        req = Requirement(line)
        if canonicalize_name(req.name) == "dowhy":
            found.append(("docker/bentoml/requirements-bentoml.txt", req))
    return found


def test_pyproject_dowhy_floor_is_networkx35_compatible() -> None:
    """Every declared dowhy floor must exclude releases broken by networkx >= 3.5 (#869).

    dowhy < 0.13 calls ``nx.algorithms.d_separated``, which networkx renamed to
    ``is_d_separator`` (3.3) and removed (3.5). The repo's networkx floors are
    ``>=3.0`` (or absent) — modern resolvers pick 3.6+ — so a spec that still
    admits dowhy < 0.13 lets a resolver pair them: every
    CausalModel.identify_effect and refuter call then raises AttributeError and
    the causal_impact refutation node fail-closes (refutation_tests_total=0).
    That exact pairing (dowhy==0.12 + networkx==3.6.1) is what uv resolved for
    python < 3.13 from the pre-#869 ``dowhy>=0.11.0`` floor: dowhy 0.13/0.14
    cap scipy at 1.15.3 on python < 3.13, so with scipy free to float to 1.16+
    the resolver backtracked to the nx-incompatible dowhy 0.12 instead.
    dowhy >= 0.13 imports ``is_d_separator`` with a ``d_separated`` fallback
    and works against every networkx the project allows.

    To falsify: lower the dowhy floor in pyproject.toml or
    docker/bentoml/requirements-bentoml.txt below 0.13 — this test reports the
    spec admits an nx-incompatible dowhy.

    Runtime companion (proves the installed pairing actually works, no mocks):
    tests/unit/test_causal_engine/test_dowhy_networkx_compat.py.
    """
    dowhy_reqs = _declared_dowhy_requirements()
    sources = {source for source, _ in dowhy_reqs}
    assert "pyproject.toml" in sources, "pyproject.toml no longer declares a dowhy dependency"
    assert "docker/bentoml/requirements-bentoml.txt" in sources, (
        "requirements-bentoml.txt no longer declares a dowhy dependency"
    )

    nx_incompatible = [
        f"{source}: {req}" for source, req in dowhy_reqs if req.specifier.contains("0.12")
    ]
    assert not nx_incompatible, (
        "these dependency sources admit dowhy releases that call the removed "
        "nx.algorithms.d_separated (broken under networkx >= 3.5, #869); "
        "raise the floor to >=0.13:\n" + "\n".join(nx_incompatible)
    )
    # Sanity: every spec must still admit the deployed pin (requirements.txt).
    too_tight = [
        f"{source}: {req}" for source, req in dowhy_reqs if not req.specifier.contains("0.14")
    ]
    assert not too_tight, (
        "these dowhy specs no longer admit the deployed dowhy==0.14 pin:\n" + "\n".join(too_tight)
    )
