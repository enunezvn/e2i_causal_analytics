"""Every runtime API key `src/` reads must be forwarded to the containers (#1622).

`x-common-env` in `docker/docker-compose.yml` is an explicit **whitelist**: a
variable absent from it is simply not present in the container, no matter what the
droplet's `.env` says. That makes the failure silent and one-sided — the key exists
on the host, every host-side script works, and only the containerised code degrades.

Measured on 2026-08-14: `UMLS_UTS_API_KEY` and `OPENFDA_API_KEY` were both set and
valid in `.env`, and both were ABSENT inside `e2i_api`. Neither had ever appeared in
the compose file (`git log -S` empty for both) — they were never added when the
clients that read them landed. Consequences were real but quiet:

* `UMLSClient()` raised `UMLSAuthError`, so `CitationResolver` degraded to
  `umls=None` and synonym expansion was off — `_candidate_terms('urticaria',
  'C0041834')` yields 2 terms with UMLS and 1 without, so genuine supporting
  citations could score as unverified.
* openFDA ran unauthenticated: no rate-limit headers at all, versus
  `x-ratelimit-limit: 240` with the key.

The compose file itself already names this failure class ("the same never-forwarded
gap as OPIK_ENABLED/OPENAI_API_KEY"), which is why this guard is derived from the
CODE rather than from a hand-maintained list — a curated list goes stale exactly the
way the thing it guards does.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPOSE = _REPO_ROOT / "docker" / "docker-compose.yml"
_SRC = _REPO_ROOT / "src"

# Reads of the form os.environ.get("X_API_KEY") / os.getenv(...) / os.environ["..."].
_ENV_READ = re.compile(
    r"""os\.(?:environ\.get|getenv|environ)\s*[(\[]\s*["']([A-Z0-9_]*API_KEY)["']"""
)

#: Keys deliberately NOT forwarded, each with the reason it is exempt. An entry here
#: is a claim that the container is *supposed* to run without it.
_EXEMPT: dict[str, str] = {
    # Opik is SELF-HOSTED on this droplet (x-common-env pins
    # OPIK_URL=http://opik-backend:8080), and the deployment is intentionally
    # stopped. OPIK_API_KEY authenticates Opik *Cloud*, which this box never calls.
    # OPIK_ENABLED — the switch that actually gates the tracer — IS forwarded.
    "OPIK_API_KEY": "self-hosted Opik; cloud key unused and the deployment is stopped",
}


def _api_keys_read_by_src() -> set[str]:
    found: set[str] = set()
    for path in _SRC.rglob("*.py"):
        found.update(_ENV_READ.findall(path.read_text(encoding="utf-8", errors="ignore")))
    return found


def _keys_forwarded_by_compose() -> set[str]:
    """Variable names assigned anywhere in the compose file's env mappings."""
    text = _COMPOSE.read_text()
    return set(re.findall(r"^\s{2,}([A-Z0-9_]+):\s", text, flags=re.MULTILINE))


def test_every_api_key_src_reads_is_forwarded_or_explicitly_exempt() -> None:
    """A key `src/` reads must reach the container, or be a documented exemption."""
    read = _api_keys_read_by_src()
    assert read, "found no *_API_KEY env reads in src/ — the detector regex is broken"

    forwarded = _keys_forwarded_by_compose()
    missing = {k for k in read if k not in forwarded and k not in _EXEMPT}

    assert not missing, (
        f"{sorted(missing)} are read from the environment by src/ but are absent from "
        f"docker/docker-compose.yml. x-common-env is a WHITELIST, so these are simply "
        f"empty inside every container even when set in .env — the code degrades "
        f"silently while host-side scripts keep working. Add them to x-common-env, or "
        f"add them to _EXEMPT here with the reason they are meant to be unset."
    )


def test_the_two_keys_this_issue_was_filed_about_are_forwarded() -> None:
    """Explicit regression pins for #1622, independent of the detector regex."""
    forwarded = _keys_forwarded_by_compose()
    for key in ("UMLS_UTS_API_KEY", "OPENFDA_API_KEY"):
        assert key in forwarded, (
            f"{key} is not forwarded by docker/docker-compose.yml — this is the exact "
            "regression #1622 fixed."
        )


def test_exemptions_are_real_reads_and_carry_a_reason() -> None:
    """An exemption for a key nobody reads is stale config pretending to be intent."""
    read = _api_keys_read_by_src()
    for key, reason in _EXEMPT.items():
        assert key in read, (
            f"_EXEMPT lists {key}, but no src/ module reads it any more — drop the "
            "exemption rather than carrying a dead justification."
        )
        assert reason.strip(), f"_EXEMPT[{key}] must state WHY it is not forwarded"
