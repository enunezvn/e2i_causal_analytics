"""#528-A — production-target deploy flip invariants (hermetic, no docker/DB).

The prod deploy historically ran the **dev overlay in prod**
(``docker compose -f docker-compose.yml -f docker-compose.dev.yml``), which pinned
api/workers/scheduler to ``target: development`` + ``uvicorn --reload`` +
``ENVIRONMENT=development`` — the #527 dev-in-prod incident. #528-A flips the app
services to the real prod target while keeping the **frontend** on dev/Vite (its
prod nginx stage was deleted in ``aa58e71c``; restoring it is #528-B).

The flip is delivered as:
  * a slim ``docker/docker-compose.frontend-dev.yml`` overlay that reproduces ONLY
    the frontend dev override (no api/worker/scheduler blocks) — so base prod target
    stands for the app services;
  * ``deploy.yml`` consumes that slim overlay instead of the full dev overlay,
    rewrites the rebuild-detector exclusion list (``src/``+``config/`` are now baked
    into the prod image → must force ``--build``; ``feature_repo/`` + ``frontend/*``
    stay bind-mounted → recreate only), adds ``feast`` + ``feast-materializer`` to
    the recreate set, and performs an ORDERED rollout (fresh store before the
    prod-target API serves).

These checks parse the compose YAML + the deploy workflow statically, so they run in
CI without docker or a database. The full ``docker compose config`` render and the
on-droplet rollout are exercised by the operational GATE, not here.
"""

from __future__ import annotations

from pathlib import Path

import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_COMPOSE = REPO_ROOT / "docker" / "docker-compose.yml"
FRONTEND_DEV_OVERLAY = REPO_ROOT / "docker" / "docker-compose.frontend-dev.yml"
DEPLOY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy.yml"

# App services that must run the prod target after the flip (frontend intentionally
# excluded — it stays dev/Vite until #528-B restores its prod image).
PROD_APP_SERVICES = ("api", "worker_light", "worker_medium", "scheduler")


class _ComposeLoader(yaml.SafeLoader):
    """SafeLoader that tolerates compose's local ``!override`` / ``!reset`` tags."""


def _passthrough(loader: yaml.Loader, tag_suffix: str, node: yaml.Node):  # noqa: ANN401
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


# Only local (``!foo``) tags — leaves standard yaml type resolution untouched.
_ComposeLoader.add_multi_constructor("!", _passthrough)


def _load(path: Path) -> dict:
    with open(path) as fh:
        return yaml.load(fh, Loader=_ComposeLoader) or {}


def _exclusion_tokens(script: str) -> list[str]:
    """The `|`-separated alternatives inside the rebuild-detector `grep -vE` anchor."""
    import re

    m = re.search(r"grep -vE '\^\(([^']+)\)'", script)
    assert m, "could not locate the anchored grep -vE exclusion alternation"
    return m.group(1).split("|")


def _services(doc: dict) -> dict:
    return doc.get("services", {}) or {}


def _deploy_text() -> str:
    return DEPLOY_WORKFLOW.read_text()


def _deploy_script() -> str:
    """The SSH deploy script body (the heredoc that runs on the droplet)."""
    text = _deploy_text()
    # The risky recreate logic lives after the compose command is defined.
    assert "COMPOSE_CMD=" in text, "deploy.yml lost its COMPOSE_CMD definition"
    return text


# --------------------------------------------------------------------------- #
# The slim frontend-only overlay
# --------------------------------------------------------------------------- #
def test_frontend_dev_overlay_exists_and_is_frontend_only():
    """The prod-deploy overlay must override ONLY the frontend.

    If it carried api/worker/scheduler blocks it would re-pin them to dev — the very
    thing the flip removes.
    """
    assert FRONTEND_DEV_OVERLAY.exists(), (
        f"missing slim overlay {FRONTEND_DEV_OVERLAY} — #528-A consumes it instead of "
        "the full docker-compose.dev.yml"
    )
    svcs = set(_services(_load(FRONTEND_DEV_OVERLAY)).keys())
    assert svcs == {"frontend"}, (
        f"frontend-dev overlay must override ONLY 'frontend'; found {sorted(svcs)}. "
        "Any app-service block here would re-pin it to the dev target."
    )


def test_frontend_dev_overlay_reproduces_full_dev_override():
    """The slim overlay must carry the FULL frontend dev override.

    A partial copy (e.g. missing the ``ports !override`` or the Vite ``command``)
    would leave the frontend half-prod/half-dev and break the deploy, since the base
    frontend ``target: production`` stage does not exist (deleted in aa58e71c).
    """
    fe = _services(_load(FRONTEND_DEV_OVERLAY))["frontend"]
    assert (fe.get("build") or {}).get("target") == "development", (
        "frontend overlay must set build.target=development (base prod stage is absent)"
    )
    assert fe.get("container_name") == "e2i_frontend_dev"
    ports = [str(p) for p in (fe.get("ports") or [])]
    assert any("3002:5173" in p for p in ports), f"frontend must map 3002:5173 (Vite); got {ports}"
    vols = [str(v) for v in (fe.get("volumes") or [])]
    for needed in ("frontend/src", "frontend/public", "frontend/index.html", "node_modules"):
        assert any(needed in v for v in vols), (
            f"frontend overlay missing bind mount {needed!r}; got {vols}"
        )
    env = fe.get("environment") or []
    env_text = " ".join(env if isinstance(env, list) else [f"{k}={v}" for k, v in env.items()])
    for key in ("NODE_ENV", "VITE_API_URL", "VITE_WS_URL", "VITE_COPILOT_ENABLED"):
        assert key in env_text, f"frontend overlay missing env {key!r}; got {env_text!r}"
    assert "npm run dev" in str(fe.get("command", "")), (
        "frontend overlay must run the Vite dev server"
    )
    hc = fe.get("healthcheck") or {}
    assert "5173" in str(hc.get("test", "")), "frontend healthcheck must probe the Vite port 5173"


# --------------------------------------------------------------------------- #
# Base app services keep the prod target (regression guard)
# --------------------------------------------------------------------------- #
def test_scheduler_runs_production_environment_after_flip():
    """The scheduler must run ENVIRONMENT=production once the dev overlay is dropped.

    Unlike api/workers (which inherit ``<<: *common-env`` → ENVIRONMENT=production),
    the base scheduler has a bespoke, minimal env block that does NOT set ENVIRONMENT,
    and pre-flip the full dev overlay supplied ``ENVIRONMENT=development``. Without an
    explicit prod value the flip leaves ``celery beat`` scheduling tasks with
    ENVIRONMENT unset while the workers that execute them run production — a half-flip.
    Local dev still overrides this to development via docker-compose.dev.yml.
    """
    sched = _services(_load(BASE_COMPOSE)).get("scheduler", {})
    env = sched.get("environment") or {}
    if isinstance(env, list):
        env = dict(e.split("=", 1) for e in env if "=" in e)
    assert env.get("ENVIRONMENT") == "production", (
        "base scheduler must set ENVIRONMENT=production (it does not inherit *common-env); "
        "else the prod-target flip runs celery beat with ENVIRONMENT unset"
    )


def test_base_app_services_are_prod_target():
    """The base compose must define the app services at target=production.

    With the dev overlay dropped from the deploy, the base definition is what runs in
    prod — so a regression that reintroduced a dev target in base would silently
    recreate the #527 incident.
    """
    base = _services(_load(BASE_COMPOSE))
    for svc in PROD_APP_SERVICES:
        assert svc in base, f"base compose missing service {svc!r}"
        target = (base[svc].get("build") or {}).get("target")
        assert target == "production", f"base {svc} must be build.target=production, got {target!r}"


# --------------------------------------------------------------------------- #
# deploy.yml consumes the slim overlay, not the full dev overlay
# --------------------------------------------------------------------------- #
def test_deploy_consumes_frontend_dev_overlay_not_full_dev():
    """The deploy command must -f the slim frontend overlay, never the full dev one.

    Using docker-compose.dev.yml here is exactly the dev-in-prod regression (#527).
    """
    script = _deploy_script()
    assert "docker-compose.frontend-dev.yml" in script, (
        "deploy must consume the slim docker-compose.frontend-dev.yml overlay"
    )
    # The forward compose command must not -f the full dev overlay. (The rollback
    # path MAY fall back to it when rolling back to a pre-flip commit; see the
    # overlay-existence test — so we scope this to the forward COMPOSE_CMD region.)
    forward, _, _rb = script.partition('if [ "$HEALTHY" = false ]')
    assert "-f docker/docker-compose.dev.yml" not in forward, (
        "forward deploy must not -f docker-compose.dev.yml (that is the #527 dev-in-prod path)"
    )


def test_deploy_overlay_is_chosen_by_file_existence_for_safe_rollback():
    """Rollback to a pre-flip commit must not break on a missing overlay file.

    Pre-flip commits have no docker-compose.frontend-dev.yml. The deploy must pick the
    overlay by existence (frontend-dev if present, else the full dev overlay), so a
    rollback to such a commit cleanly reverts the flip instead of failing on a missing
    ``-f`` target.
    """
    script = _deploy_script()
    assert "docker-compose.dev.yml" in script, (
        "deploy must still reference docker-compose.dev.yml as the pre-flip rollback fallback"
    )
    # Existence-gated selection (a `[ -f ... frontend-dev.yml ]` test choosing the overlay).
    assert "frontend-dev.yml" in script and "-f " in script
    assert "[ -f docker/docker-compose.frontend-dev.yml ]" in script, (
        "deploy must select the overlay by file existence so pre-flip rollback falls back to dev.yml"
    )


# --------------------------------------------------------------------------- #
# Rebuild-detector exclusion list reflects the new bake/bind split
# --------------------------------------------------------------------------- #
def test_rebuild_exclusion_drops_baked_src_and_config():
    """src/ and config/ are baked into the prod image now → a change must force --build.

    Post-flip the app services have NO bind mounts; the prod Dockerfile COPYs src/ +
    config/. Leaving them in the bind-mount exclusion would skip the rebuild and
    silently serve stale baked code.
    """
    script = _deploy_script()
    assert "grep -vE" in script, "deploy must keep the bind-mount exclusion (grep -vE)"
    # Match the `|`-separated alternation tokens exactly — so 'src/' is not mistaken
    # for the substring inside the legitimate 'frontend/src/' token.
    tokens = _exclusion_tokens(script)
    assert "src/" not in tokens, (
        "exclusion must drop 'src/' (baked into prod image → force rebuild)"
    )
    assert "config/" not in tokens, (
        "exclusion must drop 'config/' (baked into prod image → force rebuild)"
    )


def test_rebuild_exclusion_keeps_still_bindmounted_paths():
    """feature_repo/ + frontend/* stay bind-mounted → recreate, not rebuild."""
    tokens = _exclusion_tokens(_deploy_script())
    assert "feature_repo/" in tokens, "feature_repo/ still bind-mounted to feast + materializer"
    assert "frontend/src/" in tokens, "frontend stays dev/Vite → frontend/src/ still bind-mounted"


# --------------------------------------------------------------------------- #
# feast + feast-materializer join the recreate set, ordered before the API flip
# --------------------------------------------------------------------------- #
def test_recreate_set_includes_feast_and_materializer():
    """The durable populate (feast + materializer) must land in the recreate set.

    Both are in base compose (added in #556 PR2) but the deploy never recreated a
    feast-image service. Without them the flip ships a prod-target API over a store
    nothing keeps fresh.
    """
    script = _deploy_script()
    assert "feast-materializer" in script, (
        "deploy must recreate feast-materializer (durable populate)"
    )
    # 'feast' as a standalone recreate target (word-boundary; not just 'feast-materializer').
    import re

    assert re.search(r"(?<![\w-])feast(?![\w-])", script), "deploy must recreate the feast sidecar"


def test_ordered_rollout_materializes_before_prod_api_serves():
    """Fresh store BEFORE the prod-target API serves (round-3 HIGH-B stale-serve race).

    The rollout must, IN ORDER: recreate feast + feast-materializer, poll the
    materializer heartbeat FILE and compare it to this deploy's start (a fresh
    materialize THIS deploy — not the Docker 5m health status), and only THEN recreate
    api/workers/scheduler. Asserts the concrete shell constructs in order so a rewrite
    that drops the gate cannot pass on incidental comment text.
    """
    # Scope to the forward path (the API-health rollback also names these services).
    forward, _, _rb = _deploy_script().partition('if [ "$HEALTHY" = false ]')
    feast_idx = forward.find("--force-recreate feast feast-materializer")
    probe_idx = forward.find("docker exec e2i_feast_materializer cat /tmp/materializer_heartbeat")
    fresh_cmp_idx = forward.find('-ge "$GATE_START"')
    app_idx = forward.find("api frontend worker_light worker_medium scheduler")
    assert feast_idx != -1, "deploy must recreate feast + feast-materializer first"
    assert probe_idx != -1, (
        "deploy must poll the materializer heartbeat file (not Docker health status)"
    )
    assert fresh_cmp_idx != -1, (
        "deploy must compare the heartbeat to this deploy's start (fresh THIS deploy)"
    )
    assert app_idx != -1, "deploy must recreate the prod-target app services"
    assert feast_idx < probe_idx < app_idx, (
        "ordered rollout must gate on a fresh materialize BETWEEN the feast recreate and the app flip"
    )
    assert fresh_cmp_idx < app_idx, "the freshness comparison must gate the app flip"


def test_materialize_gate_failure_is_fail_loud_and_rolls_feast_back():
    """A failed freshness gate must NOT flip the API and MUST roll feast back.

    On gate failure the app services are untouched (still pre-deploy), but `feast`
    was just recreated at NEW_SHA and the still-serving old API depends on it
    (FEAST_URL=http://feast:6566). So the failure branch must roll feast +
    feast-materializer back to PREV_SHA, then exit non-zero — never silently leave a
    broken/new Feast under the old API.
    """
    script = _deploy_script()
    marker = 'if [ "$MAT_FRESH" = false ]; then'
    assert marker in script, "missing the materialize-gate failure branch"
    branch = script.split(marker, 1)[1].split("# Store is fresh", 1)[0]
    assert 'git checkout "$PREV_SHA"' in branch, "gate failure must check out PREV_SHA"
    assert "--force-recreate feast feast-materializer" in branch, (
        "gate failure must recreate feast + materializer at PREV_SHA (restore the Feast the old API uses)"
    )
    assert "exit 1" in branch, "gate failure must fail the deploy loudly"
    # Order matters: the feast rollback must run BEFORE exit 1 — else `exit 1` would
    # be dead-code-before-rollback, leaving the new/broken Feast under the old API.
    checkout_idx = branch.index('git checkout "$PREV_SHA"')
    up_idx = branch.index("--force-recreate feast feast-materializer", checkout_idx)
    exit_idx = branch.index("exit 1", up_idx)
    assert checkout_idx < up_idx < exit_idx, (
        "gate failure must checkout PREV_SHA, then recreate feast, THEN exit 1 (rollback before exit)"
    )


# --------------------------------------------------------------------------- #
# Trigger set tracks the consumed compose files
# --------------------------------------------------------------------------- #
def test_deploy_trigger_includes_frontend_dev_overlay():
    """A change to the consumed slim overlay must TRIGGER the deploy.

    on.push.paths gates whether the workflow fires at all; the slim overlay is now a
    live deploy input, so it must be listed or a frontend-dev change silently never
    deploys.
    """
    doc = yaml.safe_load(_deploy_text())
    on = doc.get("on")
    if on is None:  # PyYAML (YAML 1.1) parses bare ``on:`` as boolean True
        on = doc.get(True)
    paths = (on or {}).get("push", {}).get("paths", []) or []
    assert "docker/docker-compose.frontend-dev.yml" in paths, (
        f"deploy on.push.paths must include the consumed slim overlay (parsed: {paths})"
    )
