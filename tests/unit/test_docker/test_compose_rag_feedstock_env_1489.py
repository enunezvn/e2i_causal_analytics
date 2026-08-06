"""The #1486 RAG-prompt GEPA leg's env knobs must reach the containers (#1489 d2).

``src/tasks/dspy_optimization_tasks`` reads ``DSPY_RAG_RECORDS_PATH`` and
``DSPY_RAG_MAX_METRIC_CALLS`` from ``os.environ``. Neither was in compose's
``x-common-env``, so on the deployed stack the host ``.env`` values never
reached the worker that runs the beat and the in-code defaults governed
unconditionally — the leg could not be enabled from the outside at all. This is
the same silent-no-op class as ``OPIK_ENABLED`` / ``OPENAI_API_KEY`` /
``LLM_MODEL``, each of which carries the same warning in the compose file.

``DSPY_RAG_RECORDS_PATH`` also has a trap the other flags do not: it names a
*filesystem path*, and it is resolved inside the container. Every data volume in
this compose is a **named docker volume**, not a host bind mount, so a host path
(the natural thing for an operator to write, since the replay that produces the
file runs on the host) resolves to "records file not found" and the leg skips
forever while looking configured. The compose comment has to say so — hence the
comment-content assertions here, which are unusual but deliberate: the
documentation IS the mitigation for a trap the schema cannot express.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMPOSE_PATH = _REPO_ROOT / "docker" / "docker-compose.yml"

# The beat that runs the leg is ``dspy-prompt-optimization-daily``, routed to
# the ``analytics`` queue (src/workers/celery_app.py). Only ``worker_medium``
# serves that queue (--queues=analytics,reports,aggregations), so it is the one
# service that MUST have the env to execute the leg. The rest are the other
# services that merge the anchor, asserted together so a future split of
# x-common-env cannot quietly drop the executor.
#
# ``scheduler`` is deliberately absent: #528-A documents that celery beat does
# NOT inherit <<: *common-env, and it only EMITS the task — the worker executes
# it. Pinned by test_scheduler_is_excluded_by_design below so this omission
# stays an informed decision rather than an oversight.
_ANALYTICS_QUEUE_EXECUTOR = "worker_medium"
_SERVICES_NEEDING_ENV = ("api", "worker_light", _ANALYTICS_QUEUE_EXECUTOR, "worker_heavy")

_RECORDS_PATH_VAR = "DSPY_RAG_RECORDS_PATH"
_MAX_METRIC_CALLS_VAR = "DSPY_RAG_MAX_METRIC_CALLS"


def _load_compose() -> dict[str, Any]:
    return cast(dict[str, Any], yaml.safe_load(_COMPOSE_PATH.read_text()))


def _service_env(compose: dict[str, Any], service: str) -> dict[str, Any]:
    body = (compose.get("services", {}) or {}).get(service) or {}
    env = body.get("environment") or {}
    assert isinstance(env, dict), (
        f"{service}.environment must be a mapping for the *common-env merge to apply; got {type(env)}"
    )
    return cast(dict[str, Any], env)


@pytest.mark.parametrize("var", [_RECORDS_PATH_VAR, _MAX_METRIC_CALLS_VAR])
def test_leg_env_is_declared_in_the_shared_anchor(var: str) -> None:
    """Both knobs live in x-common-env, not on one service."""
    compose = _load_compose()
    common = compose.get("x-common-env") or {}
    assert var in common, (
        f"{var} missing from x-common-env. src/tasks/dspy_optimization_tasks reads it "
        f"from os.environ; without a passthrough entry the host .env value never "
        f"reaches the containers and the in-code default governs (the OPIK_ENABLED lesson)."
    )


@pytest.mark.parametrize("var", [_RECORDS_PATH_VAR, _MAX_METRIC_CALLS_VAR])
@pytest.mark.parametrize("service", _SERVICES_NEEDING_ENV)
def test_leg_env_reaches_every_service_that_merges_the_anchor(service: str, var: str) -> None:
    """The anchor merge must actually land the key on the beat's worker."""
    compose = _load_compose()
    env = _service_env(compose, service)
    assert var in env, (
        f"{var} does not reach {service}. Expected via the <<: *common-env merge; "
        f"got keys: {sorted(env)[:12]}..."
    )


@pytest.mark.parametrize("var", [_RECORDS_PATH_VAR, _MAX_METRIC_CALLS_VAR])
def test_leg_env_is_a_passthrough_not_a_hardcoded_value(var: str) -> None:
    """Both must interpolate the host .env, and neither may bake a default in.

    The in-code defaults are the SSOT (``_RAG_DEFAULT_MAX_METRIC_CALLS`` = 40,
    and "unset means skip" for the path). A compose-side default would be a
    second, silently-diverging source of truth for a budget that costs real
    judge calls.
    """
    common = _load_compose().get("x-common-env") or {}
    value = str(common.get(var, ""))
    assert value.startswith("${") and value.endswith("}"), (
        f"{var} must be a ${{...}} passthrough of the host .env; got {value!r}"
    )
    inner = value[2:-1]
    assert inner.split(":-")[0] == var, f"{var} must interpolate its own name; got {value!r}"
    default = inner.split(":-", 1)[1] if ":-" in inner else ""
    assert default == "", (
        f"{var} must default to empty so the in-code default governs; compose bakes {default!r}. "
        f"An empty value is what _rag_max_metric_calls()/the path check already treat as unset."
    )


def test_the_analytics_queue_executor_is_the_one_that_needs_it() -> None:
    """Pins WHY worker_medium is the load-bearing service in the list above.

    If the beat is ever rerouted to another queue, this fails and forces the
    service list to be revisited instead of silently guarding the wrong worker.
    """
    celery_app = (_REPO_ROOT / "src" / "workers" / "celery_app.py").read_text()
    entry = celery_app.index('"dspy-prompt-optimization-daily"')
    block = celery_app[entry : entry + 400]
    assert '"queue": "analytics"' in block, (
        "the DSPy optimization beat no longer routes to the analytics queue; "
        f"_SERVICES_NEEDING_ENV may now guard the wrong worker. Block:\n{block[:300]}"
    )
    compose = _load_compose()
    body = (compose.get("services", {}) or {}).get(_ANALYTICS_QUEUE_EXECUTOR) or {}
    assert "analytics" in str(body.get("command", "")), (
        f"{_ANALYTICS_QUEUE_EXECUTOR} no longer serves the analytics queue"
    )


def test_scheduler_is_excluded_by_design_not_by_oversight() -> None:
    """The scheduler emits the beat but never executes it (#528-A).

    Asserting the exclusion keeps it a decision: if the scheduler ever starts
    merging the anchor, this fails and the service list gets re-derived.
    """
    text = _COMPOSE_PATH.read_text()
    scheduler_at = text.index("\n  scheduler:")
    next_service = text.index("\n  frontend:", scheduler_at)
    # Comments only: the #528-A note inside this very block QUOTES the merge key
    # in order to say it is absent, so a raw substring search matches the
    # documentation of the invariant and never the invariant itself.
    code_lines = [
        line
        for line in text[scheduler_at:next_service].split("\n")
        if not line.strip().startswith("#")
    ]
    assert "<<: *common-env" not in "\n".join(code_lines), (
        "scheduler now merges *common-env; it would inherit the RAG leg env and "
        "the comment in _SERVICES_NEEDING_ENV explaining its absence is stale."
    )


def test_records_path_entry_warns_that_the_path_is_container_side() -> None:
    """The host-path trap must be documented where an operator will read it.

    Not a style check: every volume in this file is a named docker volume, so a
    host path silently becomes a permanent skip. Nothing in the YAML schema can
    express that, so the comment is the only place the mitigation can live.
    """
    text = _COMPOSE_PATH.read_text()
    anchor = text.index("x-common-env:")
    entry = text.index(f"  {_RECORDS_PATH_VAR}:", anchor)
    # The contiguous comment block immediately above the entry.
    preceding = text[anchor:entry].rsplit("\n\n", 1)[-1].lower()
    assert "container" in preceding, (
        f"the {_RECORDS_PATH_VAR} compose comment must say the path is resolved "
        f"inside the container. Got:\n{preceding}"
    )
    assert re.search(r"named (docker )?volume|not a (host )?bind mount|bind mount", preceding), (
        f"the {_RECORDS_PATH_VAR} compose comment must explain WHY a host path "
        f"fails (named volumes, no host bind mount). Got:\n{preceding}"
    )


def test_no_host_bind_mount_exists_to_make_a_host_path_work() -> None:
    """Pins the premise the comment above rests on.

    If someone later adds a host bind mount for a data root, this test fails and
    forces the comment to be revisited rather than left stale and wrong.
    """
    compose = _load_compose()
    for service in _SERVICES_NEEDING_ENV:
        body = (compose.get("services", {}) or {}).get(service) or {}
        for mount in body.get("volumes") or []:
            if not isinstance(mount, str):
                continue
            source = mount.split(":", 1)[0]
            assert not source.startswith(("/", ".", "~")), (
                f"{service} bind-mounts host path {source!r}; the "
                f"{_RECORDS_PATH_VAR} container-path comment may now be stale."
            )
