"""Forcing-function tests for #362 mlflow upgrade.

Pins two post-upgrade invariants:

1. ``requirements.txt`` pins ``mlflow`` / ``mlflow-skinny`` / ``mlflow-tracing``
   to ``>=3.10.0,<3.11.0`` (the lowest line that resolves the CVE-2026-2614
   arbitrary-file-read advisory). Pre-upgrade these were ``==3.7.0``.

2. ``.github/workflows/security.yml`` carries an ``--ignore-vuln`` allowlist
   that scopes precisely to the post-upgrade residual: the 3 mlflow advisories
   still unfixed at 3.10.x and the 4 mistune transitive advisories. Asserted
   with ``extra_exact`` set semantics so any future drift — adding back a
   resolved entry, or removing one we still need — fails the test.

These tests are RED on the pre-upgrade tree (mlflow==3.7.0 + 13 mlflow ignores)
and GREEN once the bump + ignore-list cleanup land. The pin lives at the repo
top level (matches ``tests/test_bentoml_requirements_exact_pin.py``).
"""

from __future__ import annotations

import re
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_REQS = REPO_ROOT / "requirements.txt"
DEV_REQS = REPO_ROOT / "requirements-dev.txt"
PYPROJECT = REPO_ROOT / "pyproject.toml"
SECURITY_YML = REPO_ROOT / ".github" / "workflows" / "security.yml"
MLFLOW_DOCKERFILE = REPO_ROOT / "docker" / "mlflow" / "Dockerfile"
DOCKER_COMPOSE = REPO_ROOT / "docker" / "docker-compose.yml"
DOCKER_COMPOSE_SECURE = REPO_ROOT / "docker" / "docker-compose.secure.yml"
ARCHITECTURE_MD = REPO_ROOT / "docs" / "ARCHITECTURE.md"

# Target docker image tag for the mlflow service. Pinned in lockstep with
# requirements.txt mlflow line (#362). 3.10.1 is the latest 3.10.x patch.
MLFLOW_DOCKER_IMAGE_TAG = "v3.10.1"

# Each mlflow package in requirements.txt must satisfy this specifier post-#362.
MLFLOW_REQUIRED_SPEC = SpecifierSet(">=3.10.0,<3.11.0")

# Whitelisted MLFLOW-RELATED advisory IDs that must remain ignored AFTER the
# upgrade. Three remain because their upstream fix is only available in
# 3.11.0rc0 (we refuse rc lines) or has no upstream fix yet.
EXPECTED_MLFLOW_IGNORES: frozenset[str] = frozenset(
    {
        # PYSEC-2026-94 alias — auth bypass on artifact-download endpoint;
        # fix in 3.11.0rc0 only (refuse rc).
        "GHSA-46r5-x6jq-v8g6",
        # PYSEC-2026-93 alias — Stored XSS via YAML MLmodel parsing;
        # fix in 3.11.0rc0 only (refuse rc).
        "GHSA-fh64-r2vc-xvhr",
        # CVE-2026-0545 alias — FastAPI /ajax-api/3.0/jobs/* unauth bypass
        # when basic-auth+job execution enabled; no upstream fix.
        # Mitigated — MLFLOW_SERVER_ENABLE_JOB_EXECUTION is unset.
        "GHSA-7qhf-v65m-g5f3",
    }
)

# Whitelisted MISTUNE advisory IDs (unchanged from pre-upgrade — mistune is a
# transitive dep that mlflow 3.10.x does NOT bump).
EXPECTED_MISTUNE_IGNORES: frozenset[str] = frozenset(
    {
        "GHSA-8mp2-v27r-99xp",  # CVE-2026-33079 — LINK_TITLE_RE ReDoS
        "GHSA-8g87-j6q8-g93x",  # CVE-2026-44708 — math-plugin XSS
        "GHSA-58cw-g322-p94v",  # CVE-2026-44896 — render_figure() injection
        "GHSA-v87v-83h2-53w7",  # CVE-2026-44897 — HTMLRenderer.heading() id=
    }
)

# Exact set we expect to find when grepping mlflow|mistune lines in security.yml.
EXPECTED_TOTAL_IGNORES: frozenset[str] = EXPECTED_MLFLOW_IGNORES | EXPECTED_MISTUNE_IGNORES

_PKG_LINE_RE = re.compile(
    r"^\s*(mlflow|mlflow-skinny|mlflow-tracing)\s*([<>=!~][^#\n\r]*)\s*$",
    re.MULTILINE,
)

_IGNORE_LINE_RE = re.compile(
    r"--ignore-vuln\s+(GHSA-[a-zA-Z0-9-]+|PYSEC-[0-9-]+)\b[^`]*`([^`]*)`",
)


def _read_mlflow_constraints(path: Path = ROOT_REQS) -> dict[str, str]:
    text = path.read_text()
    matches = _PKG_LINE_RE.findall(text)
    assert matches, f"no mlflow requirement lines found in {path.name}"
    return {name: spec.strip() for name, spec in matches}


def test_mlflow_packages_pinned_to_required_spec() -> None:
    """All three mlflow packages must satisfy ``>=3.10.0,<3.11.0``.

    Pre-upgrade these were ``==3.7.0``, which is why this test fails RED on
    main. ``mlflow==3.7.0`` is not in ``>=3.10.0,<3.11.0`` per PEP 440.
    """
    constraints = _read_mlflow_constraints()
    for pkg in ("mlflow", "mlflow-skinny", "mlflow-tracing"):
        assert pkg in constraints, (
            f"{pkg} missing from requirements.txt — every mlflow package must "
            f"be pinned in lockstep to avoid skew between client + skinny + tracing."
        )
        spec = SpecifierSet(constraints[pkg])
        # Sample two versions on either side of the bound to assert the spec
        # has the right SHAPE without depending on a single version probe.
        assert Version("3.7.0") not in spec, (
            f"{pkg} spec {constraints[pkg]!r} still admits the pre-upgrade "
            f"vulnerable 3.7.0 — #362 requires bumping past CVE-2026-2614."
        )
        assert Version("3.10.0") in spec and Version("3.10.1") in spec, (
            f"{pkg} spec {constraints[pkg]!r} does not include the target 3.10.x line."
        )
        # The cap also matters — refusing 3.11.x release candidates per #362.
        assert Version("3.11.0") not in spec, (
            f"{pkg} spec {constraints[pkg]!r} allows 3.11.x — #362 explicitly "
            f"stays on the 3.10.x line for now (3.11.x is rc-tagged)."
        )


def test_mlflow_packages_use_lockstep_specifier() -> None:
    """The three mlflow packages must share the IDENTICAL specifier string.

    Skew (e.g. mlflow>=3.10 + mlflow-skinny==3.7.0) corrupts the resolver and
    causes silent runtime mismatches. Locked-in lockstep is the safest pattern.
    """
    constraints = _read_mlflow_constraints()
    specs = {constraints[k] for k in ("mlflow", "mlflow-skinny", "mlflow-tracing")}
    assert len(specs) == 1, (
        f"mlflow / mlflow-skinny / mlflow-tracing specifiers diverged: "
        f"{constraints!r}; they must move in lockstep."
    )


def test_mlflow_dev_requirements_match_root_pin() -> None:
    """``requirements-dev.txt`` mlflow pins must match ``requirements.txt``.

    Same forcing-function shape as ``test_bentoml_requirements_exact_pin.py``:
    range/exact pins in one file but not the other create silent version
    skew on dev-vs-prod rebuilds. Pre-#362 both files agreed on ``==3.7.0``;
    this test ensures #362 bumps BOTH together.
    """
    root = _read_mlflow_constraints(ROOT_REQS)
    dev = _read_mlflow_constraints(DEV_REQS)
    for pkg in ("mlflow", "mlflow-skinny", "mlflow-tracing"):
        assert root.get(pkg) == dev.get(pkg), (
            f"{pkg} pin drift between requirements.txt and requirements-dev.txt: "
            f"root={root.get(pkg)!r} vs dev={dev.get(pkg)!r}; both must agree."
        )


_PYPROJECT_MLFLOW_RE = re.compile(r'"mlflow\s*([<>=!~][^"]*?)"', re.MULTILINE)


def test_pyproject_mlflow_dependency_in_required_spec() -> None:
    """``pyproject.toml [project.dependencies]`` mlflow line must also satisfy
    ``>=3.10.0,<3.11.0`` — otherwise ``pip install .`` ignores requirements.txt
    and may resolve an unaudited older mlflow.

    Codex iter-2 H3 finding (pre-fix the line was ``mlflow>=2.16.0`` which
    silently allowed installs outside the audited range).
    """
    text = PYPROJECT.read_text()
    matches = _PYPROJECT_MLFLOW_RE.findall(text)
    assert matches, (
        "no mlflow entry in pyproject.toml [project.dependencies]; "
        "the previous bound (mlflow>=2.16.0) must have been removed accidentally."
    )
    for raw_spec in matches:
        spec = SpecifierSet(raw_spec.strip())
        assert Version("3.7.0") not in spec, (
            f"pyproject.toml mlflow spec {raw_spec!r} still admits the pre-#362 "
            f"3.7.0 — bump to >=3.10.0,<3.11.0 in lockstep with requirements.txt."
        )
        assert Version("2.16.0") not in spec, (
            f"pyproject.toml mlflow spec {raw_spec!r} still admits mlflow 2.x "
            f"(the pre-#362 floor was 2.16.0); bump to >=3.10.0,<3.11.0."
        )
        assert Version("3.10.1") in spec, (
            f"pyproject.toml mlflow spec {raw_spec!r} excludes the target "
            f"3.10.1; align with requirements.txt."
        )


def test_mlflow_dockerfile_pins_match_required_spec() -> None:
    """``docker/mlflow/Dockerfile`` must install an mlflow within
    ``>=3.10.0,<3.11.0`` (lockstep with requirements.txt).

    Codex iter-2 H1 finding (pre-fix the Dockerfile pinned ``mlflow==2.9.2``,
    so the MLflow server container could run an unaudited older version
    while the Python clients had been bumped).
    """
    text = MLFLOW_DOCKERFILE.read_text()
    # The Dockerfile installs mlflow via a quoted pip arg; accept both quoted
    # and unquoted forms.
    line_re = re.compile(
        r"['\"]?\s*mlflow\s*([<>=!~][^\s'\"]+(?:\s*,\s*[<>=!~][^\s'\"]+)*)\s*['\"]?",
        re.MULTILINE,
    )
    matches = line_re.findall(text)
    assert matches, (
        "no mlflow pip-install line found in docker/mlflow/Dockerfile; "
        "the install RUN block may have been edited out."
    )
    found_valid = False
    for raw_spec in matches:
        try:
            spec = SpecifierSet(raw_spec.strip())
        except Exception:  # noqa: BLE001 — packaging.specifiers raises subclasses
            continue
        if Version("3.10.1") in spec and Version("3.7.0") not in spec:
            found_valid = True
            break
    assert found_valid, (
        f"docker/mlflow/Dockerfile does not install mlflow within "
        f"the required >=3.10.0,<3.11.0 range. Found specs: {matches!r}."
    )


def _grep_compose_mlflow_image(path: Path) -> list[str]:
    """Return all `ghcr.io/mlflow/mlflow:vX.Y.Z` tags declared in a compose
    YAML file."""
    text = path.read_text()
    return re.findall(r"ghcr\.io/mlflow/mlflow:(v[\d.]+(?:rc\d+)?)", text)


def test_docker_compose_mlflow_image_tag_locked_to_required_spec() -> None:
    """Both ``docker-compose.yml`` and ``docker-compose.secure.yml`` must
    pin the mlflow service image to the lockstep tag (``v3.10.1``).

    Codex iter-2 H2 finding (pre-fix both compose files used
    ``v3.1.0`` — an unrelated older 3.1.x line that's outside the audited
    >=3.10.0,<3.11.0 spec window).
    """
    for compose_path in (DOCKER_COMPOSE, DOCKER_COMPOSE_SECURE):
        tags = _grep_compose_mlflow_image(compose_path)
        assert tags, (
            f"no ghcr.io/mlflow/mlflow:vX.Y.Z image tag found in "
            f"{compose_path.name}; the mlflow service may have been removed."
        )
        for tag in tags:
            assert tag == MLFLOW_DOCKER_IMAGE_TAG, (
                f"{compose_path.name} pins mlflow image {tag!r} but expected "
                f"{MLFLOW_DOCKER_IMAGE_TAG!r} (lockstep with requirements.txt #362)."
            )


def test_architecture_doc_mlflow_image_tag_matches() -> None:
    """``docs/ARCHITECTURE.md`` mlflow image tag must agree with the compose
    files. Stale doc tags mislead reviewers about what's deployed.
    """
    text = ARCHITECTURE_MD.read_text()
    tags = re.findall(r"ghcr\.io/mlflow/mlflow:(v[\d.]+(?:rc\d+)?)", text)
    assert tags, (
        "no mlflow image tag found in docs/ARCHITECTURE.md; the MLOps "
        "container table may have been restructured."
    )
    for tag in tags:
        assert tag == MLFLOW_DOCKER_IMAGE_TAG, (
            f"docs/ARCHITECTURE.md references stale mlflow image {tag!r}; "
            f"expected {MLFLOW_DOCKER_IMAGE_TAG!r} (matches compose files)."
        )


_PKG_LEAD_RE = re.compile(r"^\s*#\s*(mlflow|mistune)\s+\d", re.IGNORECASE)


def _collect_mlflow_mistune_ignores() -> set[str]:
    """Grep security.yml for ignore-vuln rows whose rationale starts with
    ``mlflow X.Y.Z`` or ``mistune X.Y.Z`` (i.e. the IGNORED package is
    mlflow or mistune, not just a contextual mention). Returns the bare
    GHSA/PYSEC IDs.

    This filter intentionally excludes joblib + pyarrow + similar entries
    whose rationale references "MLflow artifacts" or "MLflow-tracked models"
    as context but whose underlying vulnerable package is something else.
    """
    text = SECURITY_YML.read_text()
    out: set[str] = set()
    for vuln_id, rationale in _IGNORE_LINE_RE.findall(text):
        if _PKG_LEAD_RE.match(rationale):
            out.add(vuln_id)
    return out


def test_security_yml_mlflow_ignores_match_post_upgrade_set_exactly() -> None:
    """The mlflow/mistune ``--ignore-vuln`` set must equal the expected
    post-upgrade residual EXACTLY — no extras, no missing.

    Pre-upgrade this set is the union of EXPECTED_TOTAL_IGNORES plus 10 mlflow
    GHSAs that get resolved by the 3.10.x bump (xch3/gq3w/q2r8/fhff/g6pg/
    vhcx/r23q/65h7/42h5/rvhj). Asserting exact-set semantics with
    EXPECTED_TOTAL_IGNORES is the forcing function that ensures those 10 are
    removed in the same PR as the bump.
    """
    actual = _collect_mlflow_mistune_ignores()
    extras = actual - EXPECTED_TOTAL_IGNORES
    missing = EXPECTED_TOTAL_IGNORES - actual
    assert not extras and not missing, (
        f"security.yml mlflow/mistune ignore set has drifted from #362 "
        f"post-upgrade expectation.\n"
        f"  extras (REMOVE — resolved by mlflow 3.10.x upgrade or stale): "
        f"{sorted(extras)}\n"
        f"  missing (ADD — required to keep CI green): {sorted(missing)}"
    )


def test_mlflow_connector_search_runs_signature_pagination_compat() -> None:
    """Drift-guard: confirm our connector's ``search_runs`` call site uses
    only positional/keyword args that are stable across mlflow 3.7 → 3.10.x.

    The MLflow 3.x line introduced an optional ``page_token`` to
    ``MlflowClient.search_runs``. Our connector uses the top-level
    ``mlflow.search_runs`` (returns DataFrame) and intentionally caps results
    at ``max_results=100`` via the existing kwargs. If a future bump
    silently changes argument names (e.g. ``experiment_ids`` →
    ``experiment_id``), this test catches the drift at CI time.
    """
    import inspect

    import mlflow

    sig = inspect.signature(mlflow.search_runs)
    required_kwargs = {"experiment_ids", "filter_string", "order_by", "max_results"}
    actual = set(sig.parameters)
    missing = required_kwargs - actual
    assert not missing, (
        f"mlflow.search_runs no longer exposes {missing} as kwargs; "
        f"update src/mlops/mlflow_connector.py:search_runs accordingly."
    )

    # Also pin MlflowClient.search_runs (used directly by integration tests).
    from mlflow.tracking import MlflowClient

    client_sig = inspect.signature(MlflowClient.search_runs)
    client_required = {"experiment_ids", "filter_string", "max_results"}
    client_actual = set(client_sig.parameters)
    client_missing = client_required - client_actual
    assert not client_missing, (
        f"MlflowClient.search_runs no longer exposes {client_missing} as "
        f"kwargs; update tests/integration/test_mlflow_repeated_runs_real.py."
    )
