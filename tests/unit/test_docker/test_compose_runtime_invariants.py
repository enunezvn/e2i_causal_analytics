"""Regression guards for the 2026-05-26/27 production outage class.

These tests encode, as permanent CI invariants, the root causes diagnosed during
the prod API/worker outage so they cannot silently regress:

* **R-A / R-C — stale-venv outage:** the dev compose override must NOT mask
  ``/app/.venv`` with a bare anonymous volume. That mask let a months-old baked
  venv outlive advancing code (the original ``ModuleNotFoundError`` outage).
* **R-B — non-root tmpfs perms:** every app-service ``/app/tmp`` and ``/tmp``
  tmpfs must be writable by the non-root ``e2i`` user (uid/gid 1000). ``mode=1770``
  on a root-owned tmpfs (the reverted ``496b2e43`` regression) makes matplotlib's
  cache dir (imported transitively via ``shap``) unwritable and crashes the API
  at boot / workers at shap-task exec.
* **R-B — matplotlib cache home:** the Dockerfile must point ``MPLCONFIGDIR`` at
  the writable tmpfs and force a headless backend, in *both* the
  base-inherited (development) path and the production stage (which does not
  derive ``FROM base``).
* **R-A — fd exhaustion:** the api service runs uvicorn ``--reload`` (dev==prod
  share this compose); it needs a raised ``nofile`` ulimit.
* **R-D — deploy band-aid:** the deploy workflow must not rely on
  ``--renew-anon-volumes`` (a band-aid for the now-removed ``.venv`` mask) and
  must keep its hang/rollback hardening.

The faithful *runtime* proof (matplotlib actually writing its cache as e2i) is the
phased container smoke in the rollout; these tests guard the declared config that
the smoke depends on, and run with no Docker daemon required.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# tests/unit/test_docker/<this file>  ->  parents[3] == repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_COMPOSE = REPO_ROOT / "docker" / "docker-compose.yml"
DEV_COMPOSE = REPO_ROOT / "docker" / "docker-compose.dev.yml"
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"
DEPLOY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy.yml"

# App services built from our Dockerfile (USER e2i, uid 1000). Infra services
# (redis, falkordb, frontend nginx, supabase, ...) run other images/users and are
# intentionally out of scope.
APP_SERVICES = {"api", "worker_light", "worker_medium", "worker_heavy", "scheduler"}


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


def _services(doc: dict) -> dict:
    return doc.get("services") or {}


def _tmpfs_entries(svc: dict) -> list[str]:
    raw = svc.get("tmpfs")
    if raw is None:
        return []
    return [raw] if isinstance(raw, str) else list(raw)


def _parse_tmpfs(entry: str) -> tuple[str, dict[str, str]]:
    """``/app/tmp:size=512M,mode=1770`` -> ``("/app/tmp", {"size": "512M", ...})``."""
    path, _, opt_str = entry.partition(":")
    opts: dict[str, str] = {}
    for kv in opt_str.split(","):
        kv = kv.strip()
        if not kv:
            continue
        key, _, val = kv.partition("=")
        opts[key.strip()] = val.strip()
    return path, opts


def _all_tmpfs_for_path(target_path: str):
    """Yield (file_label, service, opts) for every app-service tmpfs at target_path."""
    for label, compose in (("base", BASE_COMPOSE), ("dev", DEV_COMPOSE)):
        for name, svc in _services(_load(compose)).items():
            if name not in APP_SERVICES:
                continue
            for entry in _tmpfs_entries(svc):
                path, opts = _parse_tmpfs(entry)
                if path == target_path:
                    yield label, name, opts


# --------------------------------------------------------------------------- #
# R-A / R-C — no stale-venv anonymous-volume mask
# --------------------------------------------------------------------------- #
def test_dev_compose_has_no_venv_anon_mask():
    """The /app/.venv anon-volume mask is the stale-venv outage class — forbid it."""
    offenders = []
    for name, svc in _services(_load(DEV_COMPOSE)).items():
        for vol in svc.get("volumes") or []:
            if isinstance(vol, str) and vol.strip() == "/app/.venv":
                offenders.append(name)
            elif (
                isinstance(vol, dict)
                and vol.get("target") == "/app/.venv"
                and not vol.get("source")
            ):
                offenders.append(name)
    assert not offenders, (
        "dev compose still masks /app/.venv with a bare anonymous volume "
        f"(the stale-venv outage class) in: {sorted(set(offenders))}"
    )


# --------------------------------------------------------------------------- #
# R-B — tmpfs writable by the non-root e2i user
# --------------------------------------------------------------------------- #
def test_every_app_service_has_writable_app_tmp_tmpfs():
    """Every app service needs an e2i-owned (uid/gid 1000), private (0700) /app/tmp.

    MPLCONFIGDIR=/app/tmp is baked into the image ENV for all services and the
    rootfs is read_only:true, so matplotlib's import-time cache write (reached
    transitively via ``import shap``) requires a writable /app/tmp tmpfs on EVERY
    app service — including the scheduler, which autodiscovers shap-importing task
    packages (src.mlops / src.agents) at beat startup.
    """
    base = _services(_load(BASE_COMPOSE))
    dev = _services(_load(DEV_COMPOSE))
    for svc in sorted(APP_SERVICES):
        opts_seen = []
        for source in (base, dev):
            for entry in _tmpfs_entries(source.get(svc) or {}):
                path, opts = _parse_tmpfs(entry)
                if path == "/app/tmp":
                    opts_seen.append(opts)
        assert opts_seen, (
            f"{svc} declares no /app/tmp tmpfs, but MPLCONFIGDIR=/app/tmp + a "
            "read_only rootfs require one (matplotlib cache write would fail)"
        )
        for opts in opts_seen:
            assert opts.get("mode") != "1770", (
                f"{svc} /app/tmp uses the reverted root-owned mode=1770"
            )
            assert opts.get("uid") == "1000", f"{svc} /app/tmp missing uid=1000 ({opts})"
            assert opts.get("gid") == "1000", f"{svc} /app/tmp missing gid=1000 ({opts})"
            assert opts.get("mode") == "0700", (
                f"{svc} /app/tmp must be mode=0700, got mode={opts.get('mode')!r}"
            )


def test_shared_tmp_tmpfs_is_sticky_world_writable():
    """/tmp must be sticky world-writable (1777) so e2i can write celerybeat-schedule."""
    seen = list(_all_tmpfs_for_path("/tmp"))
    assert len(seen) >= 2, f"expected >=2 /tmp tmpfs declarations, parsed {len(seen)}"
    for label, name, opts in seen:
        assert opts.get("mode") != "1770", (
            f"{label}:{name} /tmp uses the reverted root-owned mode=1770 "
            "(non-root e2i cannot write /tmp/celerybeat-schedule)"
        )
        assert opts.get("mode") == "1777", (
            f"{label}:{name} /tmp must be sticky mode=1777, got mode={opts.get('mode')!r}"
        )


# --------------------------------------------------------------------------- #
# R-B — matplotlib cache home baked into the image (both stages)
# --------------------------------------------------------------------------- #
def test_dockerfile_sets_matplotlib_cache_env():
    """MPLCONFIGDIR/MPLBACKEND must be set for development (via base) AND production."""
    text = DOCKERFILE.read_text()
    # base stage = everything before the dependencies stage; development FROM base
    # inherits it. production is `FROM python:...` and does NOT inherit base ENV.
    base_region = text.split("AS dependencies", 1)[0]
    prod_split = text.split("AS production", 1)
    assert len(prod_split) == 2, "Dockerfile has no `AS production` stage"
    prod_region = prod_split[1]

    assert "MPLCONFIGDIR=/app/tmp" in base_region, (
        "base stage (inherited by development) must set MPLCONFIGDIR=/app/tmp"
    )
    assert "MPLBACKEND=Agg" in base_region, "base stage must set MPLBACKEND=Agg"
    assert "MPLCONFIGDIR=/app/tmp" in prod_region, (
        "production stage must set MPLCONFIGDIR=/app/tmp (it does not inherit base ENV)"
    )
    assert "MPLBACKEND=Agg" in prod_region, "production stage must set MPLBACKEND=Agg"


# --------------------------------------------------------------------------- #
# R-A — fd exhaustion under uvicorn --reload
# --------------------------------------------------------------------------- #
def test_api_has_nofile_ulimit():
    """api runs uvicorn --reload; it needs a raised nofile ulimit to avoid fd exhaustion."""
    api = _services(_load(DEV_COMPOSE)).get("api") or {}
    nofile = (api.get("ulimits") or {}).get("nofile")
    assert nofile is not None, "api must set ulimits.nofile (fd exhaustion under --reload)"
    if isinstance(nofile, dict):
        assert int(nofile.get("soft", 0)) >= 65536, f"nofile soft too low: {nofile}"
        assert int(nofile.get("hard", 0)) >= 65536, f"nofile hard too low: {nofile}"
    else:
        assert int(nofile) >= 65536, f"nofile too low: {nofile}"


# --------------------------------------------------------------------------- #
# R-D — deploy workflow correctness
# --------------------------------------------------------------------------- #
def test_forward_deploy_drops_renew_anon_volumes():
    """The forward deploy must not use --renew-anon-volumes (the .venv mask is gone).

    The rollback path MAY keep it: rollback can target a pre-hotfix commit that
    still carries the mask, where renewing the anon volume from the rebuilt image
    is the faithful behavior. So the prohibition is scoped to the forward path.
    """
    text = DEPLOY_WORKFLOW.read_text()
    marker = 'if [ "$HEALTHY" = false ]'
    assert marker in text, "deploy.yml rollback guard not found — structure changed"
    forward, _, _rollback = text.partition(marker)
    assert "--renew-anon-volumes" not in forward, (
        "forward deploy must not use --renew-anon-volumes once the /app/.venv mask is removed"
    )


def test_deploy_conditional_build_is_retry_safe_and_covers_non_bindmounted_inputs():
    """Rebuild on a no-delta retry, and on ANY non-bind-mounted image input.

    The detector rebuilds unless EVERY changed file is a dev-overlay bind mount, so
    any baked image input (Dockerfile, requirements, pyproject, patches/, frontend
    build config like vite.config.ts/tailwind.config.js, package manifests, ...)
    forces a rebuild without enumerating each one — closing the false-negative class
    where a triggered deploy silently recreates the frontend from a stale image.
    """
    text = DEPLOY_WORKFLOW.read_text()
    # A no-delta re-run (PREV==NEW, e.g. a retry after a partial/interrupted build)
    # must force a rebuild, else a stale local image (lacking MPLCONFIGDIR) is reused.
    assert '[ "$PREV_SHA" = "$NEW_SHA" ]' in text, (
        "conditional build must treat a no-delta run (PREV==NEW) as needing a rebuild"
    )
    # Bind-mount EXCLUSION (rebuild unless all changes are bind-mounted), not an input
    # enumeration — so a new baked image input cannot silently bypass the rebuild.
    assert "grep -vE" in text, (
        "conditional build must use a bind-mount exclusion (grep -vE capture) so any "
        "non-bind-mounted image input forces a rebuild"
    )
    assert "grep -qv" not in text, (
        "avoid grep -qv here: the droplet's grep is ugrep, whose -qv exit status is "
        "unreliable and would risk a false-negative that skips a needed rebuild"
    )
    for bind in ("src/", "frontend/src/"):
        assert bind in text, f"bind-mount exclusion must list the dev-overlay mount {bind!r}"


def test_deploy_trigger_includes_patch_dependencies():
    """A patch-only push must TRIGGER the workflow, not just be matched post-trigger.

    patches/ is COPYed into the image and requirements.txt pip-installs ./patches/*,
    so patches/** is a real build input. The in-script rebuild detector covers it, but
    that only runs after on.push.paths triggers the workflow — so the trigger must list
    it too, or a patch-only change silently never deploys.
    """
    doc = yaml.safe_load(DEPLOY_WORKFLOW.read_text())
    # PyYAML (YAML 1.1) parses a bare ``on:`` mapping key as the boolean True.
    on = doc.get("on")
    if on is None:
        on = doc.get(True)
    paths = (on or {}).get("push", {}).get("paths", []) or []
    assert "patches/**" in paths, (
        f"deploy on.push.paths must include 'patches/**' (parsed paths: {paths})"
    )


def test_deploy_trigger_includes_compose_files():
    """A compose-only change must TRIGGER the workflow, not silently never deploy.

    The deploy script runs ``docker compose -f docker/docker-compose.yml
    -f docker/docker-compose.dev.yml ... up`` — so those two files are live deploy
    inputs (tmpfs perms, volumes, env, ulimits). They were absent from on.push.paths,
    so the very compose hardening shipped in PR #527 would not have auto-deployed; the
    in-script rebuild detector recreates from compose, but only after the trigger fires.

    ``docker-compose.secure.yml`` is intentionally NOT required here: the deploy
    workflow does not consume it (no ``-f docker-compose.secure.yml`` in the script),
    so a change to it must not trigger this deploy. Asserting the two consumed files
    by literal name (not a ``docker-compose*.yml`` glob) keeps the trigger set aligned
    with what the script actually ``-f``-mounts.
    """
    doc = yaml.safe_load(DEPLOY_WORKFLOW.read_text())
    # PyYAML (YAML 1.1) parses a bare ``on:`` mapping key as the boolean True.
    on = doc.get("on")
    if on is None:
        on = doc.get(True)
    paths = (on or {}).get("push", {}).get("paths", []) or []
    for required in ("docker/docker-compose.yml", "docker/docker-compose.dev.yml"):
        assert required in paths, (
            f"deploy on.push.paths must include {required!r} — the deploy script "
            f"consumes it via -f, so a change to it must trigger a deploy "
            f"(parsed paths: {paths})"
        )
    assert "docker/docker-compose.secure.yml" not in paths, (
        "docker-compose.secure.yml must NOT be in the deploy trigger: the deploy "
        "workflow does not consume it (no -f docker-compose.secure.yml), so including "
        "it would fire spurious deploys"
    )


def test_deploy_workflow_keeps_safety_hardening():
    """Guard: the deploy.yml rewrite must not drop the existing hang/rollback hardening."""
    text = DEPLOY_WORKFLOW.read_text()
    assert "command_timeout" in text, "deploy must keep command_timeout (appleboy hang guard)"
    assert "--max-time" in text, "deploy must keep a bounded health curl (--max-time)"
    assert "--connect-timeout" in text, "deploy must keep a bounded health curl (--connect-timeout)"
