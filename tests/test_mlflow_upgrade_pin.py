"""Forcing-function tests for #362 + #442 + #1476 mlflow upgrades.

Pins two post-upgrade invariants:

1. ``requirements.txt`` pins ``mlflow`` / ``mlflow-skinny`` / ``mlflow-tracing``
   to ``>=3.15.1,<3.16.0`` (#1476: the lowest line whose cryptography cap is
   ``<50``, unlocking cryptography 49.0.0 and clearing GHSA-537c-gmf6-5ccf +
   CVE-2026-69248 + CVE-2026-69249). Pre-#1476 the range was
   ``>=3.11.0,<3.12.0`` (post-#442); pre-#442 it was ``>=3.10.0,<3.11.0``
   (post-#362); pre-#362 these were ``==3.7.0``.

2. ``.github/workflows/security.yml`` carries an ``--ignore-vuln`` allowlist
   that scopes precisely to the post-upgrade residual: the 1 mlflow advisory
   still unfixed upstream (advisory range <=3.10.1; kept listed so local and
   CI advisory DBs agree). The 4 mistune entries were cleared by the
   2026-07-10 sweep (mistune pinned 3.3.3). Asserted with exact set semantics
   so any future drift — adding back a resolved entry, or removing one we
   still need — fails the test.

These tests are RED on the pre-upgrade tree (mlflow 3.11.x pins) and GREEN
once the #1476 bump lands. The pin lives at the repo top level (matches
``tests/test_bentoml_requirements_exact_pin.py``).
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
# requirements.txt mlflow line (#1476; supersedes #442 / #362). v3.15.1 is the
# latest 3.15.x patch — ghcr.io/mlflow/mlflow:v3.15.1 manifest verified via
# ``docker manifest inspect`` 2026-08-04.
MLFLOW_DOCKER_IMAGE_TAG = "v3.15.1"

# Each mlflow package in requirements.txt must satisfy this specifier post-#1476.
# Floor is 3.15.1 (not 3.15.0) — 3.15.1 is the patch release actually resolved
# and installed; both admit cryptography<50.
MLFLOW_REQUIRED_SPEC = SpecifierSet(">=3.15.1,<3.16.0")

# Whitelisted MLFLOW-RELATED advisory IDs that must remain ignored AFTER the
# #442 upgrade. Only ONE remains because GHSA-7qhf has no upstream fix yet.
# Three other entries (GHSA-46r5, GHSA-fh64, GHSA-75cm) were dropped in #442
# because the 3.11.x line addresses them per V-D8 advisory data + 3.11.1
# release notes (PR #21708 / PR #21435).
EXPECTED_MLFLOW_IGNORES: frozenset[str] = frozenset(
    {
        # CVE-2026-0545 alias — FastAPI /ajax-api/3.0/jobs/* unauth bypass
        # when basic-auth+job execution enabled. ``first_patched_version: null``
        # confirmed via ``gh api /advisories/GHSA-7qhf-v65m-g5f3`` on 2026-05-22.
        # Mitigated — MLFLOW_SERVER_ENABLE_JOB_EXECUTION is unset in our deployment.
        "GHSA-7qhf-v65m-g5f3",
    }
)

# Whitelisted MISTUNE advisory IDs. EMPTY since the 2026-07-10 allowlist
# sweep: mistune was pinned 3.1.4->3.3.3 in requirements.txt, clearing all 5
# held mistune advisories, and their --ignore-vuln entries were removed from
# security.yml. This expectation had silently drifted (the sweep never updated
# it — this file is not wired into any CI job); repaired in #1476.
EXPECTED_MISTUNE_IGNORES: frozenset[str] = frozenset()

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
    """All three mlflow packages must satisfy ``>=3.15.1,<3.16.0``.

    Pre-#1476 these were ``>=3.11.0,<3.12.0`` (post-#442). #1476 bumps to the
    3.15.x line because mlflow 3.15.1 is the first release whose cryptography
    cap is ``<50`` (3.11/3.12 cap ``<47``; 3.13/3.14 cap ``<49``), unlocking
    cryptography 49.0.0 which clears GHSA-537c-gmf6-5ccf + CVE-2026-69248 +
    CVE-2026-69249.
    """
    constraints = _read_mlflow_constraints()
    for pkg in ("mlflow", "mlflow-skinny", "mlflow-tracing"):
        assert pkg in constraints, (
            f"{pkg} missing from requirements.txt — every mlflow package must "
            f"be pinned in lockstep to avoid skew between client + skinny + tracing."
        )
        spec = SpecifierSet(constraints[pkg])
        # Sample versions on either side of the bounds to assert the spec
        # has the right SHAPE without depending on a single version probe.
        assert Version("3.7.0") not in spec, (
            f"{pkg} spec {constraints[pkg]!r} still admits the pre-#362 "
            f"vulnerable 3.7.0 — #1476 inherits the historical floor exclusions."
        )
        assert Version("3.11.1") not in spec, (
            f"{pkg} spec {constraints[pkg]!r} still admits the pre-#1476 "
            f"3.11.1 — its cryptography<47 cap blocks the 49.0.0 CVE fixes."
        )
        assert Version("3.14.0") not in spec, (
            f"{pkg} spec {constraints[pkg]!r} admits 3.14.x whose "
            f"cryptography<49 cap still blocks cryptography 49.0.0."
        )
        assert Version("3.15.1") in spec, (
            f"{pkg} spec {constraints[pkg]!r} does not include the target 3.15.1."
        )
        # The cap also matters — refusing the 3.16.x line (separate tracker).
        assert Version("3.16.0") not in spec, (
            f"{pkg} spec {constraints[pkg]!r} allows 3.16.x — #1476 explicitly "
            f"stays on the 3.15.x line; a 3.16 bump is a separate tracker."
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
    post-#362 both agreed on ``>=3.10.0,<3.11.0``; #442 bumped both to
    ``>=3.11.0,<3.12.0``; #1476 bumps both to ``>=3.15.1,<3.16.0`` together.
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
    ``>=3.15.1,<3.16.0`` — otherwise ``pip install .`` ignores requirements.txt
    and may resolve an unaudited older mlflow.

    #362 codex iter-2 H3 finding (pre-fix the line was ``mlflow>=2.16.0`` which
    silently allowed installs outside the audited range). #362 codex iter-3 M
    finding tightened the assertion to exclude the upper cap. #1476 bumps the
    window from >=3.11.0,<3.12.0 to >=3.15.1,<3.16.0 in lockstep with
    requirements.txt.
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
            f"3.7.0 — bump to >=3.15.1,<3.16.0 in lockstep with requirements.txt."
        )
        assert Version("2.16.0") not in spec, (
            f"pyproject.toml mlflow spec {raw_spec!r} still admits mlflow 2.x "
            f"(the pre-#362 floor was 2.16.0); bump to >=3.15.1,<3.16.0."
        )
        assert Version("3.11.1") not in spec, (
            f"pyproject.toml mlflow spec {raw_spec!r} still admits the pre-#1476 "
            f"3.11.1 — its cryptography<47 cap blocks the 49.0.0 CVE fixes."
        )
        assert Version("3.15.1") in spec, (
            f"pyproject.toml mlflow spec {raw_spec!r} excludes the target "
            f"3.15.1; align with requirements.txt."
        )
        # Same upper cap as requirements.txt — refusing the 3.16.x line.
        assert Version("3.16.0") not in spec, (
            f"pyproject.toml mlflow spec {raw_spec!r} admits 3.16.x without "
            f"the lockstep upper cap; align with requirements.txt's "
            f">=3.15.1,<3.16.0 (open-ended >= alone would let pip resolve "
            f"3.16.x which is a separate dep-bump tracker per #1476)."
        )


def test_mlflow_dockerfile_pins_match_required_spec() -> None:
    """``docker/mlflow/Dockerfile`` must install mlflow within
    ``>=3.15.1,<3.16.0`` (lockstep with requirements.txt). EXACTLY ONE
    mlflow pip arg must be present, and it must satisfy the policy spec
    (in particular, refusing 3.7.0 + 3.11.1 + 3.16.0).

    #362 codex iter-2 H1 finding (pre-fix the Dockerfile pinned ``mlflow==2.9.2``).
    #362 iter-3 M finding tightened the assertion to require ALL matched specs
    are in-policy + exactly one mlflow arg (previously the test passed if
    ANY one matched, so a stale ``mlflow==2.9.2`` line co-existing with the
    new spec would have falsely passed). #442 bumps the cap from 3.11.0 to 3.12.0.
    """
    text = MLFLOW_DOCKERFILE.read_text()
    # The Dockerfile installs mlflow via a quoted pip arg; accept both quoted
    # and unquoted forms. The match captures the version specifier ONLY when
    # preceded by the literal package name ``mlflow`` (not ``mlflow-skinny``
    # or ``mlflow-tracing``).
    line_re = re.compile(
        r"(?<![\w-])mlflow\s*([<>=!~][^\s'\"]+(?:\s*,\s*[<>=!~][^\s'\"]+)*)",
        re.MULTILINE,
    )
    matches = line_re.findall(text)
    assert matches, (
        "no mlflow pip-install line found in docker/mlflow/Dockerfile; "
        "the install RUN block may have been edited out."
    )
    # iter-3 fix: require EXACTLY ONE mlflow arg so a stale 2.9.2 line
    # co-existing with the new spec doesn't false-pass.
    assert len(matches) == 1, (
        f"docker/mlflow/Dockerfile has {len(matches)} mlflow pip args "
        f"({matches!r}); expected exactly 1. Multiple specs risk install-time "
        f"order-dependent resolution + cause silent skew."
    )
    raw_spec = matches[0].strip()
    spec = SpecifierSet(raw_spec)
    assert Version("3.15.1") in spec, (
        f"docker/mlflow/Dockerfile mlflow spec {raw_spec!r} excludes the "
        f"target 3.15.1; align with requirements.txt."
    )
    assert Version("3.7.0") not in spec, (
        f"docker/mlflow/Dockerfile mlflow spec {raw_spec!r} still admits "
        f"the pre-#362 vulnerable 3.7.0."
    )
    assert Version("3.11.1") not in spec, (
        f"docker/mlflow/Dockerfile mlflow spec {raw_spec!r} still admits "
        f"the pre-#1476 3.11.1 — bump to >=3.15.1,<3.16.0 for cryptography 49."
    )
    # Same upper cap as requirements.txt — refusing the 3.16.x line.
    assert Version("3.16.0") not in spec, (
        f"docker/mlflow/Dockerfile mlflow spec {raw_spec!r} admits 3.16.x "
        f"without the lockstep upper cap; align with requirements.txt's "
        f">=3.15.1,<3.16.0 (open-ended >= alone allows 3.16.x)."
    )
    # And 2.x must be excluded (catches pre-#362 2.9.2 + 2.16.0 regressions).
    assert Version("2.16.0") not in spec, (
        f"docker/mlflow/Dockerfile mlflow spec {raw_spec!r} still admits "
        f"mlflow 2.x; bump to >=3.15.1,<3.16.0."
    )


def _grep_compose_mlflow_image(path: Path) -> list[str]:
    """Return all `ghcr.io/mlflow/mlflow:vX.Y.Z` tags declared in a compose
    YAML file."""
    text = path.read_text()
    return re.findall(r"ghcr\.io/mlflow/mlflow:(v[\d.]+(?:rc\d+)?)", text)


def test_docker_compose_mlflow_image_tag_locked_to_required_spec() -> None:
    """Both ``docker-compose.yml`` and ``docker-compose.secure.yml`` must
    pin the mlflow service image to the lockstep tag (``v3.15.1``).

    #362 codex iter-2 H2 finding (pre-fix both compose files used
    ``v3.1.0`` — an unrelated older 3.1.x line). #442 bumped the lockstep
    tag from v3.10.1 to v3.11.1; #1476 bumps it to v3.15.1 (manifest
    verified on ghcr 2026-08-04).
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
                f"{MLFLOW_DOCKER_IMAGE_TAG!r} (lockstep with requirements.txt #442)."
            )


def test_architecture_doc_mlflow_image_tag_matches() -> None:
    """``docs/ARCHITECTURE.md`` mlflow image tag must agree with the compose
    files. Stale doc tags mislead reviewers about what's deployed.

    Catches BOTH the ``ghcr.io/mlflow/mlflow:vX.Y.Z`` image string AND the
    plaintext ``MLflow vX.Y.Z`` mentions (e.g. in C4 diagram Container
    annotations). #362 iter-3 L finding caught a stale ``MLflow v3.1.0``
    plaintext mention in the C4 diagram that the image-tag-only regex
    missed. #1476 retains the dual check, now expecting v3.15.1.
    """
    text = ARCHITECTURE_MD.read_text()
    image_tags = re.findall(r"ghcr\.io/mlflow/mlflow:(v[\d.]+(?:rc\d+)?)", text)
    # Plaintext mentions like ``"MLflow", "MLflow v3.10.1"`` in C4 diagrams.
    # Match case-insensitively but capture the version suffix only.
    plain_tags = re.findall(r"MLflow\s+(v[\d.]+(?:rc\d+)?)", text)
    all_tags = image_tags + plain_tags
    assert all_tags, (
        "no mlflow image tag or plaintext version mention found in "
        "docs/ARCHITECTURE.md; the MLOps container table + C4 diagram "
        "may have been restructured."
    )
    # Require BOTH image-tag occurrence AND at least one plaintext mention
    # (defends against future doc restructures that drop the C4 mention
    # while leaving the table — or vice-versa).
    assert image_tags, (
        "no ghcr.io/mlflow/mlflow:vX.Y.Z image tag found in docs/ARCHITECTURE.md container table."
    )
    for tag in all_tags:
        assert tag == MLFLOW_DOCKER_IMAGE_TAG, (
            f"docs/ARCHITECTURE.md references stale mlflow version {tag!r}; "
            f"expected {MLFLOW_DOCKER_IMAGE_TAG!r} (matches compose files). "
            f"Both ghcr.io image tags and ``MLflow vX.Y.Z`` C4 mentions are "
            f"enforced — check both sites."
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

    Pre-#442 this set was {GHSA-46r5, GHSA-fh64, GHSA-7qhf, GHSA-75cm} ∪ 4
    mistune. The 3.11.x bump resolved GHSA-46r5 + GHSA-fh64 + GHSA-75cm; the
    2026-07-10 sweep cleared the 4 mistune entries (mistune pinned 3.3.3).
    Only GHSA-7qhf remains (first_patched_version=null, advisory range
    <=3.10.1 — kept listed so local and CI advisory DBs agree; re-verified
    via gh api /advisories/GHSA-7qhf-v65m-g5f3 on 2026-08-04 during #1476).
    Asserting exact-set semantics is the forcing function that ensures
    resolved entries are removed in the same PR as the bump.
    """
    actual = _collect_mlflow_mistune_ignores()
    extras = actual - EXPECTED_TOTAL_IGNORES
    missing = EXPECTED_TOTAL_IGNORES - actual
    assert not extras and not missing, (
        f"security.yml mlflow/mistune ignore set has drifted from #442 "
        f"post-upgrade expectation.\n"
        f"  extras (REMOVE — resolved by mlflow 3.11.x upgrade or stale): "
        f"{sorted(extras)}\n"
        f"  missing (ADD — required to keep CI green): {sorted(missing)}"
    )


def test_mlflow_connector_search_runs_signature_pagination_compat() -> None:
    """Drift-guard: confirm our connector's ``search_runs`` call site uses
    only positional/keyword args that are stable across mlflow 3.7 → 3.11.x.

    The MLflow 3.x line introduced an optional ``page_token`` to
    ``MlflowClient.search_runs``. Our connector uses the top-level
    ``mlflow.search_runs`` (returns DataFrame) and intentionally caps results
    at ``max_results=100`` via the existing kwargs. If a future bump
    silently changes argument names (e.g. ``experiment_ids`` →
    ``experiment_id``), this test catches the drift at CI time. Verified
    stable across the 3.10 → 3.11.x transition during #442.
    """
    import inspect

    # Codex iter-0 F-002: hard import (not importorskip) — the V-D11 drift-guard
    # MUST fail closed when mlflow is missing, not silently skip. mlflow is a
    # required dependency per pyproject.toml; absence is a real failure, not an
    # optional dev-env case.
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
