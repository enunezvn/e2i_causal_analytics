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


def _declared_leg_env_vars() -> list[str]:
    """Every env name the RAG leg reads, DERIVED from the code that reads it.

    Not a hardcoded list on purpose. This exact defect is what #1489 deferral 2
    is: a flag the code reads that compose never forwards, so the host .env is
    silently inert and the in-code default governs. Writing the names out by
    hand reproduces the bug the moment someone adds the next flag — as happened
    in this very lane, where the two knobs the deferral named were wired and the
    two NEW ones the DB source introduced were not (caught by codex, not by the
    first version of this test).

    Deriving from the modules turns a one-off fix into a guard for the class.

    Known limits, stated so nobody reads this as broader than it is: it sees
    only names bound to an UPPERCASE module constant and prefixed ``DSPY_RAG_``.
    Env names read as string literals are invisible to it —
    ``dspy_optimization_tasks`` has three (``DSPY_MIN_SIGNALS``,
    ``DSPY_LEARN_FOCUS_AGENTS``, ``DSPY_LEARN_WINDOW_HOURS``), and all three are
    ALSO absent from x-common-env. That is the same silent-no-op class, but it
    belongs to the feedback-learner legs rather than the RAG leg #1489 deferral
    2 names, and forwarding ``DSPY_MIN_SIGNALS`` would change when the nightly
    optimization triggers — a behavioral change that needs its own decision, not
    a drive-by. Reported, deliberately not fixed here.
    """
    from src.tasks import dspy_optimization_tasks as leg
    from src.tasks import rag_example_sources as sources

    names: set[str] = set()
    for module in (sources, leg):
        for attr in dir(module):
            if not attr.isupper():
                continue
            value = getattr(module, attr)
            if isinstance(value, str) and value.startswith("DSPY_RAG_"):
                names.add(value)
    return sorted(names)


def _load_compose() -> dict[str, Any]:
    return cast(dict[str, Any], yaml.safe_load(_COMPOSE_PATH.read_text()))


def _service_env(compose: dict[str, Any], service: str) -> dict[str, Any]:
    body = (compose.get("services", {}) or {}).get(service) or {}
    env = body.get("environment") or {}
    assert isinstance(env, dict), (
        f"{service}.environment must be a mapping for the *common-env merge to apply; got {type(env)}"
    )
    return cast(dict[str, Any], env)


def test_every_env_the_leg_reads_is_declared_in_the_shared_anchor() -> None:
    """EVERY DSPY_RAG_* knob the code reads, not just the two #1489 named.

    Asserted as one set rather than parametrized so the failure message lists
    what is missing, which is the actionable part.
    """
    common = _load_compose().get("x-common-env") or {}
    declared = _declared_leg_env_vars()
    missing = [name for name in declared if name not in common]
    assert not missing, (
        f"{missing} read by the RAG leg but missing from x-common-env. Without a "
        f"passthrough entry the host .env value never reaches the containers and "
        f"the in-code default governs (the OPIK_ENABLED lesson) — which is exactly "
        f"the defect #1489 deferral 2 exists to fix. Declared knobs: {declared}"
    )


def test_the_derivation_actually_finds_the_known_knobs() -> None:
    """Guards the guard: a derivation that found nothing would pass vacuously."""
    declared = _declared_leg_env_vars()
    assert _RECORDS_PATH_VAR in declared
    assert _MAX_METRIC_CALLS_VAR in declared
    assert len(declared) >= 4, f"expected the DB-source knobs too; got {declared}"


@pytest.mark.parametrize("service", _SERVICES_NEEDING_ENV)
def test_leg_env_reaches_every_service_that_merges_the_anchor(service: str) -> None:
    """The anchor merge must actually land every knob on the beat's worker."""
    compose = _load_compose()
    env = _service_env(compose, service)
    missing = [name for name in _declared_leg_env_vars() if name not in env]
    assert not missing, (
        f"{missing} do not reach {service}. Expected via the <<: *common-env merge; "
        f"got keys: {sorted(env)[:12]}..."
    )


@pytest.mark.parametrize("var", _declared_leg_env_vars())
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


def test_the_suggested_records_directory_is_actually_mounted_read_write() -> None:
    """The comment tells an operator where to put the file; that must be true.

    An earlier version said /app/optimized_modules was worker_medium's ONE
    writable shared location. It has seven. The advice was still right — that
    volume already holds this leg's artifacts and .trigger_state.json — but the
    reason given was false, and a comment a test asserts on has to be accurate
    or the test launders a wrong claim into an invariant.
    """
    compose = _load_compose()
    body = (compose.get("services", {}) or {}).get(_ANALYTICS_QUEUE_EXECUTOR) or {}
    writable = [
        m for m in (body.get("volumes") or []) if isinstance(m, str) and not m.endswith(":ro")
    ]
    targets = [m.split(":", 1)[1] for m in writable if ":" in m]
    assert "/app/optimized_modules" in targets, (
        f"the compose comment points operators at /app/optimized_modules, but "
        f"{_ANALYTICS_QUEUE_EXECUTOR} does not mount it read-write. Writable: {targets}"
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
