"""Every credential `src/` reads must reach the containers (#1622).

`x-common-env` in `docker/docker-compose.yml` is an explicit **whitelist**: a
variable absent from it is simply not present in the container, no matter what the
droplet's `.env` says. That makes the failure silent and one-sided — the key exists
on the host, every host-side script and the whole local test suite keep working, and
only the containerised code degrades.

Measured on 2026-08-14: `UMLS_UTS_API_KEY` and `OPENFDA_API_KEY` were both set and
valid in `.env`, and both were ABSENT inside `e2i_api`. Neither had ever appeared in
the compose file (`git log -S` empty for both) — they were never added when the
clients that read them landed. `CitationResolver` degraded to `umls=None` (synonym
expansion off: 1 candidate term instead of 2) and openFDA ran unauthenticated (no
rate-limit headers versus `x-ratelimit-limit: 240`).

## Two rules, because neither alone is sufficient

**Rule A — code-derived (runs everywhere, including CI).** Scan `src/` for
credential-named env reads and require each to be forwarded, aliased, or exempt.
Catches a key added to a client before it is added to compose.

**Rule B — `.env`-derived (droplet/local only).** A credential that `src/` reads
AND the host has AND the container lacks. This encodes the #1622 failure directly
rather than inferring it from a naming convention, and would have caught both keys
by construction. `.env` is gitignored, so this rule SKIPS where the file is absent;
it is a droplet guard, not a CI gate. Rule A is the CI gate.

The "src/ reads it" half of that intersection is not decoration — see the test's own
docstring: `.env` here also holds credentials for tooling the application never
touches, and requiring all of them to be forwarded flags 13 non-defects.

## Known bound (stated rather than left implicit)

The detector matches *literal* env reads — `os.environ.get("X")`, `os.getenv(...)`,
`os.environ[...]` — with a credential-shaped name. It cannot see an indirect read
such as `KEY = "X_TOKEN"; os.getenv(KEY)`, nor a name outside the credential
pattern. Verified 2026-08-14: every indirect read in `src/` is a feature flag or
tuning knob (`_ENV_POLICY`, `_CITATION_BUDGET_ENV`, …), none is a credential, and
`src/` contains no Pydantic `BaseSettings`. This raises the floor; it does not prove
completeness.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPOSE = _REPO_ROOT / "docker" / "docker-compose.yml"
_SRC = _REPO_ROOT / "src"
_DOTENV = _REPO_ROOT / ".env"

#: A credential-shaped variable name. Deliberately broader than `*_API_KEY`: the
#: original guard missed `SMTP_PASSWORD`, `SUPABASE_JWT_SECRET` and
#: `SUPABASE_SERVICE_ROLE_KEY` purely because of their suffixes.
_CREDENTIAL_NAME = (
    r"(?:[A-Z0-9_]*(?:API_KEY|_TOKEN|_SECRET|_PASSWORD|_PAT|_CREDENTIALS)|[A-Z0-9_]+_KEY)"
)

_ENV_READ = re.compile(
    rf"""os\.(?:environ\.get|getenv|environ)\s*[(\[]\s*["']({_CREDENTIAL_NAME})["']"""
)

#: Credentials deliberately NOT forwarded. An entry is a claim that the container is
#: *supposed* to run without it — so each carries the reason.
_EXEMPT: dict[str, str] = {
    # Opik is SELF-HOSTED here (x-common-env pins OPIK_URL=http://opik-backend:8080)
    # and the deployment is intentionally stopped. OPIK_API_KEY authenticates Opik
    # *Cloud*, which this box never calls. OPIK_ENABLED — the switch that actually
    # gates the tracer — IS forwarded.
    "OPIK_API_KEY": "self-hosted Opik; cloud key unused and the deployment is stopped",
    # The email alert channel is off: alert_routing.py gates it on
    # ALERT_EMAIL_ENABLED (default "false"), and the whole SMTP_* family
    # (HOST/PORT/USERNAME/PASSWORD) is unforwarded together. Forwarding only the
    # password would be incoherent.
    "SMTP_PASSWORD": "email alert channel disabled; the entire SMTP_* family is unforwarded",
    # Not consumed by the current verification path. api/dependencies/auth.py
    # verifies via Supabase get_user() using SUPABASE_URL + SUPABASE_ANON_KEY (both
    # forwarded); this secret is only needed if local HS256 verification is added.
    # The module's own startup message says so explicitly.
    "SUPABASE_JWT_SECRET": "unused by the get_user() auth path; only for local HS256 verification",
}

#: Credentials read under one name but satisfied by a DIFFERENT forwarded variable
#: through an in-code fallback. Not the same as exempt: the container does need the
#: value, it just arrives under another name. The alias target is verified below.
_ALIASED: dict[str, str] = {
    # supabase_client.py:37 —
    #   os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "") or os.environ.get("SUPABASE_SERVICE_KEY", "")
    "SUPABASE_SERVICE_ROLE_KEY": "SUPABASE_SERVICE_KEY",
}


def _credentials_read_by_src() -> set[str]:
    found: set[str] = set()
    for path in _SRC.rglob("*.py"):
        found.update(_ENV_READ.findall(path.read_text(encoding="utf-8", errors="ignore")))
    return found


def _forwarded_by_compose() -> set[str]:
    """Variable names assigned anywhere in the compose file's env mappings."""
    return set(re.findall(r"^\s{2,}([A-Z0-9_]+):\s", _COMPOSE.read_text(), flags=re.MULTILINE))


def _credentials_set_in_dotenv() -> dict[str, str]:
    """Credential-shaped vars with a NON-EMPTY value in `.env`."""
    values: dict[str, str] = {}
    name_only = re.compile(rf"^({_CREDENTIAL_NAME})=(.*)$")
    for raw in _DOTENV.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = name_only.match(line)
        if match and match.group(2).strip():
            values[match.group(1)] = match.group(2).strip()
    return values


# ------------------------------------------------------------------ Rule A (CI gate)


def test_every_credential_src_reads_is_forwarded_aliased_or_exempt() -> None:
    read = _credentials_read_by_src()
    assert read, "found no credential env reads in src/ — the detector regex is broken"

    forwarded = _forwarded_by_compose()
    missing = sorted(
        k for k in read if k not in forwarded and k not in _EXEMPT and k not in _ALIASED
    )

    assert not missing, (
        f"{missing} are read from the environment by src/ but are absent from "
        "docker/docker-compose.yml. x-common-env is a WHITELIST, so these are empty "
        "inside every container even when set in .env — the code degrades silently "
        "while host-side scripts keep working. Add them to x-common-env, or record "
        "them in _EXEMPT (with the reason they are meant to be unset) or _ALIASED "
        "(with the forwarded variable that satisfies them)."
    )


# ------------------------------------------------- Rule B (droplet/local, skipped in CI)


@pytest.mark.skipif(not _DOTENV.is_file(), reason="no .env here (gitignored); CI runs Rule A")
def test_credentials_src_reads_and_the_host_has_actually_reach_the_container() -> None:
    """The #1622 shape stated exactly: code reads it ∧ host has it ∧ container lacks it.

    The intersection is load-bearing. `.env` on this droplet is a kitchen-sink file
    that also holds credentials for tooling the application never touches —
    measured: DIGITALOCEAN_TOKEN, GRAFANA_ADMIN_PASSWORD, FLOWER_PASSWORD,
    MLFLOW_AUTH_PASSWORD, SUPABASE_POSTGRES_PASSWORD, the OPIK_* stack passwords and
    others are set in `.env`, referenced NOWHERE under `docker/` (neither as a key
    nor as `${VAR}`), and read by NO module in `src/`. Requiring every credential in
    `.env` to be forwarded would flag all 13 as defects and drown the real signal.

    Intersecting with "src/ actually reads it" keeps exactly the failures that
    matter, and would still have caught UMLS_UTS_API_KEY and OPENFDA_API_KEY on the
    day they were added: both were read by `src/`, both had values on the host, and
    neither reached the container.
    """
    forwarded = _forwarded_by_compose()
    in_env = _credentials_set_in_dotenv()
    assert in_env, ".env parsed but yielded no credential entries — parser is broken"

    read = _credentials_read_by_src()
    dropped = sorted(
        k
        for k in read & set(in_env)
        if k not in forwarded and k not in _EXEMPT and k not in _ALIASED
    )

    assert not dropped, (
        f"{dropped} are read by src/ AND have real values in .env, but are not "
        "forwarded by docker/docker-compose.yml — so they are EMPTY in every "
        "container while every host-side script keeps working. This is exactly the "
        "#1622 failure. Add them to x-common-env, or record why they are "
        "deliberately host-only in _EXEMPT."
    )


# ------------------------------------------------------- the bookkeeping stays honest


def test_exemptions_are_real_reads_and_carry_a_reason() -> None:
    """An exemption for a credential nobody reads is stale config posing as intent."""
    read = _credentials_read_by_src()
    for key, reason in _EXEMPT.items():
        assert key in read, (
            f"_EXEMPT lists {key}, but no src/ module reads it any more — drop the "
            "exemption rather than carrying a dead justification."
        )
        assert reason.strip(), f"_EXEMPT[{key}] must state WHY it is not forwarded"


def test_alias_targets_are_actually_forwarded() -> None:
    """An alias is only valid if the variable it defers to really is forwarded."""
    read = _credentials_read_by_src()
    forwarded = _forwarded_by_compose()
    for alias, target in _ALIASED.items():
        assert alias in read, f"_ALIASED lists {alias}, but no src/ module reads it"
        assert target in forwarded, (
            f"_ALIASED maps {alias} -> {target}, but {target} is NOT forwarded by "
            "docker/docker-compose.yml — so neither name reaches the container and "
            "the alias is hiding a real gap."
        )


def test_the_two_keys_this_issue_was_filed_about_are_forwarded() -> None:
    """Explicit regression pins for #1622, independent of the detector regex."""
    forwarded = _forwarded_by_compose()
    for key in ("UMLS_UTS_API_KEY", "OPENFDA_API_KEY"):
        assert key in forwarded, (
            f"{key} is not forwarded by docker/docker-compose.yml — this is the exact "
            "regression #1622 fixed."
        )
