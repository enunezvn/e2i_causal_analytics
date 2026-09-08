"""Operator switches must reach the containers, and must not change meaning on the way (#1931).

The sibling module ``test_compose_env_credential_forwarding.py`` guards the same
whitelist for *credentials*. This one guards **operator switches** — the knobs
`.env.example` and `DEPLOYMENT.md` tell an operator they may change — because the
credential detector is keyed on credential-shaped names and cannot see them.

## The failure this encodes

`x-common-env` is a whitelist. `ADAPTIVE_CRITERIA` is documented in two places as
*the* rollback lever for the v3 adaptive success-criteria engine
(`.env.example`, `criteria_validator.py:18`), and it had never been added to that
whitelist. Measured 2026-09-07, before the fix::

    $ docker exec e2i_api printenv ADAPTIVE_CRITERIA
    $ echo $?
    1

So an operator reaching for the documented rollback during an incident edits the
droplet `.env`, gets a **silent no-op**, and believes the platform rolled back.
Seven more switches were in the same state.

## Why the obvious fix would have been worse

`${VAR:-}` is *not* "leave it unset". Docker sets the variable to the **empty
string**. Measured on the same container, on a var compose already forwards that
way::

    $ docker exec e2i_api printenv CHATBOT_RAG_REWRITE_COT
    $ echo $?
    0

Empty-but-present. That is harmless for a reader that guards on
``raw.strip() == ""``, and a **behaviour flip** for one that does not::

    os.getenv("ADAPTIVE_CRITERIA", "true").strip().lower() in _TRUTHY
    #   unset -> "true" -> True        ''  -> "" -> False

i.e. adding ``ADAPTIVE_CRITERIA: ${ADAPTIVE_CRITERIA:-}`` would have silently
**executed** the platform-wide rollback the line exists to make *possible*.

So the invariant this module pins is not "is it forwarded" alone. It is:

    forwarded  AND  host-unset behaviour is unchanged by the forwarding.

## Keeping the model honest

The per-reader semantics below are a *reimplementation*, not the real function —
importing the real ones would drag dspy and the ML stack into an integration run.
A reimplementation can drift, so each entry also carries a verbatim slice of its
reader's source, asserted to still be present. Edit the reader and this module
fails until the model is re-synced. Same device as the Haiku pricing pin in
``tests/unit/test_data/test_evaluator_telemetry.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pytest
import yaml

pytestmark = [pytest.mark.integration]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPOSE = _REPO_ROOT / "docker" / "docker-compose.yml"
_ENV_EXAMPLE = _REPO_ROOT / ".env.example"
_DEPLOYMENT_MD = _REPO_ROOT / "DEPLOYMENT.md"

#: The truthy-string set the flag readers share (``_TRUTHY`` in both modules).
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _truthy_flag(default: str) -> Callable[[Optional[str]], object]:
    """A reader shaped ``os.getenv(VAR, default).strip().lower() in _TRUTHY``."""

    def read(raw: Optional[str]) -> object:
        return (default if raw is None else raw).strip().lower() in _TRUTHY

    return read


def _empty_guarded(default: object) -> Callable[[Optional[str]], object]:
    """A reader shaped ``if raw is None or raw.strip() == "": return default``."""

    def read(raw: Optional[str]) -> object:
        if raw is None or raw.strip() == "":
            return default
        return raw

    return read


@dataclass(frozen=True)
class Switch:
    """One operator switch, with enough of its reader to check the forwarding."""

    #: Module that reads it, relative to the repo root.
    reader: str
    #: A verbatim slice of that module. Pins the model below against drift.
    reader_literal: str
    #: Reimplementation of the reader's env handling.
    read: Callable[[Optional[str]], object]
    #: Why this switch is an operator's to change.
    why: str


#: Every switch #1931 found unforwarded (read by `src/`, documented as operator-
#: settable, absent from `x-common-env` until that issue) -- plus every operator
#: switch added since, so a new knob cannot ship host-side only.
_SWITCHES: dict[str, Switch] = {
    "ADAPTIVE_CRITERIA": Switch(
        reader="src/agents/ml_foundation/scope_definer/nodes/criteria_validator.py",
        reader_literal='os.getenv("ADAPTIVE_CRITERIA", "true").strip().lower() in _TRUTHY',
        read=_truthy_flag("true"),
        why="THE documented rollback switch for the v3 adaptive success-criteria engine",
    ),
    "ADAPTIVE_VALIDITY_EVALUATOR_ENABLED": Switch(
        reader="src/data/causal_role_evaluator.py",
        reader_literal='raw = os.environ.get(ENABLE_ENV_VAR, "").strip().lower()',
        read=_truthy_flag(""),
        why="operator opt-in for the Layer-4 Haiku audit evaluator (spends real budget)",
    ),
    "ADAPTIVE_VALIDITY_EVALUATOR_MODEL": Switch(
        reader="src/data/causal_role_evaluator.py",
        reader_literal='raw = os.environ.get(MODEL_ENV_VAR, "").strip()',
        read=_empty_guarded(""),
        why="model pin for the Layer-4 evaluator; falls back to DEFAULT_EVALUATOR_MODEL",
    ),
    "SEGMENT_ANALYSIS_BUDGET_SECONDS": Switch(
        reader="src/api/routes/segments.py",
        reader_literal='if raw is None or raw.strip() == "":\n        return SEGMENT_ANALYSIS_BUDGET_SECONDS_DEFAULT',
        read=_empty_guarded(900.0),
        why="run budget for one segment graph fit; retuning it must not need a code change",
    ),
    "AGENT_COMPUTE_EXECUTOR_WORKERS": Switch(
        reader="src/api/dependencies/compute.py",
        reader_literal='raw = os.environ.get("AGENT_COMPUTE_EXECUTOR_WORKERS")',
        read=_empty_guarded("in-code default"),
        why="agent-graph pool size; a box-headroom knob",
    ),
    "HEAVY_COMPUTE_MAX_CONCURRENCY": Switch(
        reader="src/api/dependencies/compute.py",
        reader_literal='raw = os.environ.get("HEAVY_COMPUTE_MAX_CONCURRENCY")',
        read=_empty_guarded("in-code default"),
        why="heavy-op admission budget sized against the 5G cgroup",
    ),
    "HEAVY_COMPUTE_EXECUTOR_WORKERS": Switch(
        reader="src/api/dependencies/compute.py",
        reader_literal='raw = os.environ.get("HEAVY_COMPUTE_EXECUTOR_WORKERS")',
        read=_empty_guarded("in-code default"),
        why="heavy executor pool size; paired with the concurrency budget above",
    ),
    "HEAVY_OFFLOAD_ENABLED": Switch(
        reader="src/api/dependencies/compute.py",
        reader_literal='os.environ.get("HEAVY_OFFLOAD_ENABLED", "false").strip().lower() in _TRUTHY',
        read=_truthy_flag("false"),
        why="P2 heavy-offload feature flag, DARK by default",
    ),
    # #1971: added with the switch itself. Reader: unset -> False; "" -> False
    # (in _FALSY); truthy set -> True. So `${VAR:-}` cannot flip behaviour.
    "CAUSAL_IMPACT_REQUIRE_DAG_APPROVAL": Switch(
        reader="src/agents/causal_impact/nodes/refutation.py",
        reader_literal="raw = os.environ.get(_ENV_REQUIRE_DAG_APPROVAL)",
        read=_empty_guarded(False),
        why="expert-review enforcement on the causal_impact path (halt a REVIEW band without approval)",
    ),
}


def _common_env() -> dict[str, object]:
    compose = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))
    common = compose.get("x-common-env")
    assert isinstance(common, dict) and common, (
        "x-common-env is missing or not a mapping in docker/docker-compose.yml — "
        "this module's whole premise (it is the forwarding whitelist) is broken."
    )
    return common


_INTERPOLATION = re.compile(r"^\$\{([A-Z0-9_]+)(?::-(.*))?\}$", re.DOTALL)


def _value_in_container_when_host_unset(raw_value: object) -> Optional[str]:
    """What the container actually sees, given the host `.env` does not set it.

    ``${VAR:-default}`` -> ``default`` (docker sets it, empty string included —
    see the module docstring's ``printenv`` measurement). A literal -> itself.
    ``${VAR}`` with no default -> the empty string, same rule.
    """
    if raw_value is None:
        return None
    text = str(raw_value)
    match = _INTERPOLATION.match(text.strip())
    if match:
        return match.group(2) if match.group(2) is not None else ""
    return text


# --------------------------------------------------------------- the forwarding


@pytest.mark.parametrize("name", sorted(_SWITCHES))
def test_operator_switch_is_forwarded_into_the_containers(name: str) -> None:
    """A switch an operator is told to set must not be a no-op inside Docker."""
    common = _common_env()
    switch = _SWITCHES[name]

    assert name in common, (
        f"{name} is read by {switch.reader} and documented as operator-settable "
        f"({switch.why}), but is absent from x-common-env in "
        "docker/docker-compose.yml. That anchor is a WHITELIST, so the host .env "
        "value never reaches api/workers/scheduler: the operator's change is a "
        "SILENT no-op in the containers while it keeps working host-side. This is "
        "the #1931 failure."
    )


@pytest.mark.parametrize("name", sorted(_SWITCHES))
def test_forwarding_does_not_change_the_host_unset_behaviour(name: str) -> None:
    """Forwarding must add reachability, never flip a default.

    ``${VAR:-}`` sets the variable to the EMPTY STRING in the container, which is
    only equivalent to unset for a reader that guards on it. For
    ``ADAPTIVE_CRITERIA`` it is not: an empty default would have silently
    executed the platform-wide rollback the forwarding exists to enable.
    """
    common = _common_env()
    switch = _SWITCHES[name]
    if name not in common:  # covered by the test above; don't double-report
        pytest.skip(f"{name} not forwarded yet")

    container_value = _value_in_container_when_host_unset(common[name])
    assert switch.read(container_value) == switch.read(None), (
        f"{name} is forwarded as {common[name]!r}, so with the host .env silent "
        f"the container sees {container_value!r} instead of the variable being "
        f"unset — and {switch.reader} does NOT read those the same way:\n"
        f"    unset -> {switch.read(None)!r}\n"
        f"    {container_value!r} -> {switch.read(container_value)!r}\n"
        "Forwarding must make the switch reachable, not change what happens when "
        "nobody touches it. Mirror the in-code default in the compose default "
        "(e.g. ${VAR:-true}) instead of using the empty one."
    )


@pytest.mark.parametrize("name", sorted(_SWITCHES))
def test_the_modelled_reader_still_matches_its_source(name: str) -> None:
    """The reimplementation above is only trustworthy while the source agrees."""
    switch = _SWITCHES[name]
    source = (_REPO_ROOT / switch.reader).read_text(encoding="utf-8")
    assert switch.reader_literal in source, (
        f"{switch.reader} no longer contains:\n    {switch.reader_literal}\n"
        f"so this module's model of how {name} is read is stale, and the "
        "empty-default equivalence check above is no longer evidence of anything. "
        "Re-read the function and update the Switch entry."
    )


def test_adaptive_criteria_carries_the_in_code_default_explicitly() -> None:
    """The one switch that may not use the empty default — pinned by name.

    Kept separate from the generic check so the reason survives as prose: this is
    the var whose obvious fix was the dangerous one, and a future edit to
    ``${ADAPTIVE_CRITERIA:-}`` must fail with that sentence attached rather than
    with a generic equivalence error.
    """
    assert _common_env().get("ADAPTIVE_CRITERIA") == "${ADAPTIVE_CRITERIA:-true}", (
        "ADAPTIVE_CRITERIA must be forwarded with the in-code default 'true' "
        "spelled out. criteria_validator.py reads it as "
        "os.getenv('ADAPTIVE_CRITERIA', 'true'), so an empty compose default "
        "puts '' in the container, '' is not in _TRUTHY, and the v3 adaptive "
        "engine turns OFF platform-wide — the compose file would perform the "
        "rollback instead of enabling it."
    )


# ------------------------------------------------------------------ the docs agree


@pytest.mark.parametrize("doc", [_ENV_EXAMPLE, _DEPLOYMENT_MD])
def test_docs_do_not_still_call_a_forwarded_switch_inert(doc: Path) -> None:
    """`.env.example` and DEPLOYMENT.md carry a 'NOT forwarded' list (#1921).

    It was accurate when written. Once a var is forwarded, leaving it listed
    there is worse than the original gap: it tells an operator the working
    switch does nothing.
    """
    common = _common_env()
    text = doc.read_text(encoding="utf-8")

    inert_claim = re.compile(
        r"^.*(?:NOT forwarded|not forwarded|NOT in the compose whitelist|is inert).*$",
        re.MULTILINE,
    )
    offenders = sorted(
        {
            name
            for name in _SWITCHES
            if name in common
            for line in inert_claim.findall(text)
            if name in line
        }
    )

    assert not offenders, (
        f"{doc.name} still describes {offenders} as not forwarded / inert, but "
        "docker/docker-compose.yml now forwards them. Stale documentation of a "
        "fixed gap sends the operator away from a switch that works."
    )
