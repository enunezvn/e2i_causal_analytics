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

import fnmatch
import re
from pathlib import Path

import yaml

# tests/unit/test_docker/<this file>  ->  parents[3] == repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_COMPOSE = REPO_ROOT / "docker" / "docker-compose.yml"
DEV_COMPOSE = REPO_ROOT / "docker" / "docker-compose.dev.yml"
SECURE_COMPOSE = REPO_ROOT / "docker" / "docker-compose.secure.yml"
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
# N3 (#705) — OpenMP runtime for the digital-twin effect estimator
# --------------------------------------------------------------------------- #
def test_production_stage_installs_libgomp1_for_twin_estimator():
    """The production image MUST apt-install ``libgomp1`` (the OpenMP runtime).

    The digital-twin ``/simulate`` flagship runs the real effect estimator, which
    imports causalml's ``UpliftRandomForest`` -> ``lightgbm``; ``lib_lightgbm.so``
    dlopens the SYSTEM ``libgomp.so.1``. The ``python:3.12-slim-bookworm`` base
    does not ship it, and the libgomp copies bundled inside the
    torch/sklearn/xgboost wheels are not on the loader path. Without ``libgomp1``
    on the production stage, every real ``/simulate`` call fails effect estimation
    ("Effect estimation failed: ... libgomp.so.1: cannot open shared object
    file"), is gated to HTTP 422, and the twin flagship is down end-to-end.

    This regression is INVISIBLE to the host-run test suite: dev/CI hosts carry
    the ``libgomp1`` OS package, so ``import lightgbm`` succeeds and every test
    stays green — only a faithful in-container run on the slim image exposes it
    (the 2026-06-06 N3 incident; the same host-vs-container blind spot that hid
    H4). Guard the declared dependency so the one-line apt entry cannot be dropped
    unnoticed in a future apt-list refactor.

    Production does NOT derive ``FROM base`` (it is ``FROM python:...``), so the
    package must appear in the **production-stage** apt list specifically; a
    base-only install never reaches the shipped image (the same reason the
    matplotlib ENV is asserted twice above).
    """
    text = DOCKERFILE.read_text()
    prod_split = text.split("AS production", 1)
    assert len(prod_split) == 2, "Dockerfile has no `AS production` stage"
    prod_region = prod_split[1]

    # Match the EXACT apt package entry (``libgomp1`` followed by whitespace, a
    # line-continuation ``\``, or end-of-line), not a comment that merely
    # references ``libgomp.so.1`` (different token — has a dot) nor a prefix
    # variant such as ``libgomp1-dev`` (a build dep that does not belong on the
    # runtime stage).
    has_pkg = any(
        re.match(r"libgomp1(\s|\\|$)", line.strip())
        for line in prod_region.splitlines()
        if not line.strip().startswith("#")
    )
    assert has_pkg, (
        "production stage must apt-install libgomp1 (OpenMP runtime for "
        "lightgbm/causalml in the twin effect estimator); without it every real "
        "/simulate 422s on 'libgomp.so.1: cannot open shared object file' (#705 N3)"
    )


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
# #565 — worker_light OOM-churn: full-app prefork children must fit the cgroup
# --------------------------------------------------------------------------- #
# Diagnosed 2026-05-30 on the live droplet. ``celery_app.autodiscover_tasks([...])``
# finalizes the app in the worker MASTER *before* the prefork pool forks, so every
# child carries the fully-imported heavy task graph (shap/torch/xgboost/econml/
# dowhy/statsmodels via src.mlops/src.causal/src.agents/...). The kernel OOM log
# showed ``constraint=CONSTRAINT_MEMCG`` on the worker_light cgroup killing a
# ``celery`` child at ~400 MB anon-rss / 1.7 GB VM. worker_light ran
# ``--concurrency=4`` under a 1 GB limit (256 MB/child budget) → the 4-child
# working set blew the cgroup → exit 137 → ``restart: unless-stopped`` loop. That
# also failed the ``inspect ping`` healthcheck, which stranded the scheduler during
# deploy (``compose up`` exit 1, no rollback). The ``inspect ping`` client itself is
# light (measured ~51 MB) and is NOT the memory driver — so "lighten the
# healthcheck" would not have fixed it. A faithful c=2 boot probe of the same image
# peaked at 905 MB and idled at 607 MB (== worker_medium), confirming c=2 fits.
WORKER_TIERS = ("worker_light", "worker_medium", "worker_heavy")

# Measured per-child resident set of a full-app-loaded prefork worker (~400 MB
# anon-rss in the OOM log). A tier's cgroup limit must budget at least this much
# for EVERY concurrent child, or an all-children spike trips the OOM-killer.
MIN_MB_PER_FULL_APP_CHILD = 400

# Measured c=2 boot peak of the worker_light image (full-app import across the
# pool boot — faithful throwaway probe of the deployed image). The cgroup limit
# must clear this WITH margin, or the OOM-killer fires during the all-children
# boot/recreate spike (the deploy-time failure mode), even when the per-child
# budget alone is satisfied (e.g. c=2 @ 800MB passes 400/child but < 905MB peak).
WORKER_LIGHT_BOOT_PEAK_MB = 905

# Faithful c=2 probe split of the 607 MB idle total: ~300 MB master/shared + ~2x150 MB
# children. Used to budget --max-memory-per-child so graceful recycle beats the OOM.
WORKER_MASTER_OVERHEAD_MB = 300
WORKER_LIGHT_NORMAL_CHILD_MB = 350  # threshold floor: above normal per-child ~180-250MB


def _mem_to_mb(value: object) -> float:
    """``"1G"`` / ``"512M"`` / ``1073741824`` -> megabytes."""
    s = str(value).strip().upper()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([KMG]?)B?", s)
    assert m, f"unparseable memory value {value!r}"
    num = float(m.group(1))
    return {"K": num / 1024, "M": num, "G": num * 1024, "": num / (1024 * 1024)}[m.group(2)]


def _worker_concurrency(svc: dict) -> int:
    cmd = svc.get("command", "")
    if isinstance(cmd, list):
        cmd = " ".join(str(c) for c in cmd)
    m = re.search(r"--concurrency[= ](\d+)", cmd)
    assert m, f"could not find --concurrency in command: {cmd!r}"
    return int(m.group(1))


def _limit_mb(svc: dict) -> float:
    limits = ((svc.get("deploy") or {}).get("resources") or {}).get("limits") or {}
    mem = limits.get("memory")
    assert mem is not None, "worker must declare deploy.resources.limits.memory"
    return _mem_to_mb(mem)


def test_full_app_worker_memory_budget_covers_every_prefork_child():
    """#565: each full-app worker tier must budget >= the measured per-child RSS
    (~400 MB) for EVERY concurrent prefork child, else an all-children memory spike
    trips the cgroup OOM-killer — which worker_light @ c=4/1G (256 MB/child) did."""
    services = _services(_load(BASE_COMPOSE))
    offenders = []
    for tier in WORKER_TIERS:
        svc = services.get(tier)
        assert svc, f"{tier} must exist in the base compose"
        conc = _worker_concurrency(svc)
        budget_per_child = _limit_mb(svc) / conc
        if budget_per_child < MIN_MB_PER_FULL_APP_CHILD:
            offenders.append(
                f"{tier}: {_limit_mb(svc):.0f}MB limit / concurrency {conc} = "
                f"{budget_per_child:.0f}MB per child < {MIN_MB_PER_FULL_APP_CHILD}MB floor"
            )
    assert not offenders, (
        "full-app worker tier(s) under-budget the per-child resident set "
        "(#565 OOM-churn class): " + "; ".join(offenders)
    )


def test_worker_light_concurrency_fits_its_cgroup():
    """#565: worker_light's prefork concurrency must be <= 2 so its full-app children
    fit the cgroup on this memory-starved droplet (was 4 → OOM-churn). worker_medium
    runs the same image at c=2 stable at ~608 MB; a faithful c=2 boot probe peaked
    at 905 MB. If the box grows, raise the limit in lockstep (see the per-child
    budget guard) before raising concurrency."""
    svc = _services(_load(BASE_COMPOSE)).get("worker_light") or {}
    conc = _worker_concurrency(svc)
    assert conc <= 2, f"worker_light --concurrency must be <= 2 to fit its cgroup; got {conc}"


def test_worker_light_memory_limit_clears_measured_boot_peak():
    """#565: worker_light's cgroup limit must exceed the measured 905 MB c=2 boot
    peak WITH headroom (set to 1.5G). The per-child budget guard alone is not
    enough: c=2 @ 800MB-1G passes 400/child but still OOMs on the boot/recreate
    spike. This pins the absolute floor to the measured peak so a future shrink
    below it is caught."""
    svc = _services(_load(BASE_COMPOSE)).get("worker_light") or {}
    limit = _limit_mb(svc)
    floor = WORKER_LIGHT_BOOT_PEAK_MB * 1.25  # 905MB peak + 25% headroom ~= 1131MB
    assert limit >= floor, (
        f"worker_light memory limit {limit:.0f}MB must clear the measured 905MB "
        f"boot peak + headroom (>= {floor:.0f}MB); got {limit:.0f}MB"
    )


def test_worker_light_recycles_children_before_cgroup_oom():
    """#565 defense-in-depth: --max-memory-per-child must be set, BELOW the cgroup
    limit (so a child that grows is recycled gracefully after its task instead of
    SIGKILLed mid-task by the OOM-killer, which loses the task + forces a requeue)
    and ABOVE normal full-app RSS (so it does not thrash-recycle). --max-tasks-
    per-child recycles on count, not memory, so it cannot bound a pathological
    task's RSS."""
    svc = _services(_load(BASE_COMPOSE)).get("worker_light") or {}
    cmd = svc.get("command", "")
    if isinstance(cmd, list):
        cmd = " ".join(str(c) for c in cmd)
    m = re.search(r"--max-memory-per-child[= ](\d+)", cmd)
    assert m, f"worker_light must set --max-memory-per-child (KB); got {cmd!r}"
    per_child_mb = int(m.group(1)) / 1024
    conc = _worker_concurrency(svc)
    limit_mb = _limit_mb(svc)
    # codex F6-R1: worst case is ALL `conc` children at the threshold + master/shared
    # overhead must still fit the cgroup, so a graceful recycle always precedes the
    # OOM-killer (a raw "< cgroup limit" check ignores the sibling+master aggregate).
    worst_case = conc * per_child_mb + WORKER_MASTER_OVERHEAD_MB
    assert worst_case <= limit_mb, (
        f"--max-memory-per-child {per_child_mb:.0f}MB x concurrency {conc} + "
        f"{WORKER_MASTER_OVERHEAD_MB}MB master overhead = {worst_case:.0f}MB exceeds the "
        f"{limit_mb:.0f}MB cgroup limit — a child could OOM before recycling"
    )
    assert per_child_mb >= WORKER_LIGHT_NORMAL_CHILD_MB, (
        f"--max-memory-per-child {per_child_mb:.0f}MB must exceed the normal per-child "
        f"working set (~{WORKER_LIGHT_NORMAL_CHILD_MB}MB) to avoid thrash-recycling"
    )


def test_scheduler_does_not_block_on_worker_light_health():
    """#565: celery-beat (scheduler) only publishes scheduled tasks to the redis
    broker — it does NOT require a healthy worker_light. A ``service_healthy`` edge
    let a flapping worker_light strand the scheduler ('Created') and red the deploy
    (``compose up`` exit 1, no rollback path). The edge must be exactly
    ``service_started`` (or dropped): ``service_completed_successfully`` or other
    states would block beat indefinitely, so a loose 'not service_healthy' check is
    not enough."""
    scheduler = _services(_load(BASE_COMPOSE)).get("scheduler") or {}
    dep = scheduler.get("depends_on") or {}
    if "worker_light" not in dep:
        return  # no edge at all -> beat cannot be stranded by worker_light
    wl = dep["worker_light"]
    cond = wl.get("condition") if isinstance(wl, dict) else None
    assert cond == "service_started", (
        "scheduler->worker_light must be condition: service_started (or the edge "
        "dropped) — service_healthy stranded the scheduler on a worker OOM-flap, and "
        "service_completed_successfully / other states would block beat forever. "
        f"got condition={cond!r}"
    )


# --------------------------------------------------------------------------- #
# R-D — deploy workflow correctness
# --------------------------------------------------------------------------- #
def test_forward_deploy_drops_renew_anon_volumes():
    """The forward deploy must not use --renew-anon-volumes (the .venv mask is gone).

    The rollback path MAY keep it: rollback can target a pre-hotfix commit that
    still carries the mask, where renewing the anon volume from the rebuilt image
    is the faithful behavior. Since #563 the rollback lives in a single
    rollback_to_prev() helper (defined near the top of the script), so the
    prohibition is scoped to the FORWARD service-recreate ``up``s — the ones gated by
    ``$BUILD_FLAG`` / ``$APP_BUILD_FLAG`` — not the helper.
    """
    text = DEPLOY_WORKFLOW.read_text()
    assert "rollback_to_prev()" in text, "the #563 rollback helper must exist"
    # Coalesce shell line-continuations so a forward `up` split across physical lines is
    # ONE logical command — else a --renew-anon-volumes on a continuation line (after the
    # build-flag line) would slip past a per-physical-line scan.
    logical = re.sub(r"\\\n\s*", " ", text)
    forward_ups = [
        ln
        for ln in logical.splitlines()
        if "up -d" in ln and ("$BUILD_FLAG" in ln or "$APP_BUILD_FLAG" in ln)
    ]
    assert forward_ups, (
        "forward recreate `up`s ($BUILD_FLAG/$APP_BUILD_FLAG) not found — structure changed"
    )
    for ln in forward_ups:
        assert "--renew-anon-volumes" not in ln, (
            "forward deploy must not use --renew-anon-volumes once the /app/.venv mask "
            "is removed: " + ln.strip()
        )
    # The rollback path MAY keep it (pre-flip .venv-mask targets) — assert it survives.
    assert "--renew-anon-volumes" in text, (
        "rollback must still renew anon volumes for a pre-flip rollback target"
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
    # Post-#528-A the prod-target app services have NO bind mounts; src/ + config/ are
    # baked into the image (a change to either must force a rebuild → NOT excluded).
    # The paths that remain bind-mounted (recreate, not rebuild) are feature_repo/
    # (feast + materializer) and frontend/* (the frontend stays dev/Vite via the slim
    # overlay until #528-B). Detailed token-level assertions live in
    # tests/unit/test_docker/test_prod_target_flip.py.
    for bind in ("frontend/src/", "feature_repo/"):
        assert bind in text, f"bind-mount exclusion must list the still-bind-mounted path {bind!r}"


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


# =============================================================================
# #556 - Feast materializer service invariants
# =============================================================================
# The Redis online store is only kept fresh by the in-sidecar materializer
# (the app/worker image cannot import feast, #307, so the Celery materialize
# beats are no-ops). These guards pin the materializer's declared config so it
# cannot silently regress. Runtime proof is the deploy-time materialize smoke.

DOCKERFILE_FEAST = REPO_ROOT / "docker" / "Dockerfile.feast"
POPULATE_HELPER = REPO_ROOT / "docker" / "feast" / "_populate_feast.sh"
MATERIALIZER_ENTRYPOINT = REPO_ROOT / "docker" / "feast" / "materializer-entrypoint.sh"
MATERIALIZE_CONFIG = REPO_ROOT / "config" / "feast_materialization.yaml"


def _materializer() -> dict:
    svc = _services(_load(BASE_COMPOSE)).get("feast-materializer")
    assert svc, "feast-materializer service must exist in the base compose (#556)"
    return svc


def test_materializer_uses_same_image_as_feast_serving():
    """Built from Dockerfile.feast — the ONLY image that can import feast. A second
    independently-versioned feast image would be a second 0.43.0 drift surface."""
    svc = _materializer()
    build = svc.get("build") or {}
    assert build.get("dockerfile") == "Dockerfile.feast", (
        f"feast-materializer must build from Dockerfile.feast (got {build!r})"
    )


def test_materializer_overrides_entrypoint_to_materialize_loop():
    svc = _materializer()
    assert svc.get("entrypoint") == ["/materializer-entrypoint.sh"], (
        f"feast-materializer must override the entrypoint to the materialize loop "
        f"(got {svc.get('entrypoint')!r})"
    )


def test_materializer_shares_registry_and_mounts_config():
    """Shares the feast_registry volume (reads what `feast` applied; materialize-only)
    and mounts the cadence config read-only (#556: config/ is not on the app image)."""
    svc = _materializer()
    vols = svc.get("volumes") or []
    joined = "\n".join(vols)
    assert "feast_registry:/feast/data" in joined, "must share the feast_registry volume"
    assert "../feature_repo:/feast-src:ro" in joined, "must mount the feature repo source ro"
    assert "feast_materialization.yaml:/feast/feast_materialization.yaml:ro" in joined, (
        "must mount config/feast_materialization.yaml read-only for the cadence"
    )


def test_materializer_waits_for_feast_apply():
    """Materialize-only depends on `feast` having applied the registry first."""
    svc = _materializer()
    dep = svc.get("depends_on") or {}
    assert "feast" in dep, "feast-materializer must depend_on the feast service"
    cond = dep["feast"].get("condition") if isinstance(dep["feast"], dict) else None
    assert cond == "service_healthy", (
        f"feast-materializer must wait for feast service_healthy (got {cond!r})"
    )


def test_materializer_has_healthcheck_and_no_serving_port():
    """A hung loop must be visible (heartbeat healthcheck); the materializer is
    not a server, so it must not publish a port."""
    svc = _materializer()
    assert svc.get("healthcheck"), "feast-materializer must declare a healthcheck (hung-loop guard)"
    assert not svc.get("ports"), "feast-materializer is not a server; it must not publish ports"


def test_dockerfile_feast_ships_materializer_scripts():
    text = DOCKERFILE_FEAST.read_text()
    assert "materializer-entrypoint.sh" in text, (
        "Dockerfile.feast must COPY the materializer entrypoint"
    )
    assert "_populate_feast.sh" in text, "Dockerfile.feast must COPY the shared populate helper"
    assert POPULATE_HELPER.exists() and MATERIALIZER_ENTRYPOINT.exists(), (
        "the populate helper + materializer entrypoint scripts must exist"
    )


def test_materialization_config_enumerates_all_online_views():
    """#556: every ONLINE-served feature view must be in the materialization config,
    else the materializer leaves the omitted views to decay (exactly the failure
    that left the 1-day-TTL views stale). market_dynamics is the only allowed
    omission from the online set (it is online=False)."""
    cfg = yaml.safe_load(MATERIALIZE_CONFIG.read_text()) or {}
    configured = set((cfg.get("feature_views") or {}).keys())
    online_served = {
        "hcp_conversion_features",
        "hcp_engagement_features",
        "patient_journey_features",
        "patient_adherence_features",
        "trigger_effectiveness_features",
        "trigger_response_features",
        "hcp_profile_features",
        "territory_performance_features",
    }
    missing = online_served - configured
    assert not missing, f"materialization config is missing online-served views: {sorted(missing)}"
    # market_dynamics is present but disabled (online=False)
    md = (cfg.get("feature_views") or {}).get("market_dynamics_features") or {}
    assert md.get("enabled") is False, (
        "market_dynamics_features must be enabled=False in the config (#556)"
    )


# =============================================================================
# Priority 1 OOM bounding - api gunicorn worker count
# =============================================================================
# Each inline-heavy API request (digital-twin simulate, SHAP explain) peaks at
# ~1.3 GiB. With the api cgroup capped at 5G, 4 gunicorn workers each running one
# heavy request concurrently peaked at ~5.2 GiB and OOM-killed e2i_api. Halving
# the api worker count to 2 (combined with the per-worker heavy-compute slot in
# src/api/dependencies/compute.py) keeps the worst-case concurrent heavy peak
# (2 workers x 1 in-flight heavy op ~= 2.6 GiB) safely under the 5G cap. These
# guards pin the worker count at 2 across both compose files and the baked
# Dockerfile CMD so the OOM regression cannot silently return.


def _api_command_str(compose_path: Path) -> str:
    api = _services(_load(compose_path)).get("api")
    assert api, f"api service must exist in {compose_path.name}"
    cmd = api.get("command", "")
    if isinstance(cmd, list):
        cmd = " ".join(str(c) for c in cmd)
    return cmd


def test_api_command_uses_two_workers_base_compose():
    cmd = _api_command_str(BASE_COMPOSE)
    assert re.search(r"--workers\s+2\b", cmd), (
        f"api gunicorn command must use --workers 2 (OOM bound); got {cmd!r}"
    )
    assert not re.search(r"--workers\s+4\b", cmd), (
        "api gunicorn command must NOT use --workers 4 (OOM regression class)"
    )


def test_api_command_uses_two_workers_secure_compose():
    cmd = _api_command_str(SECURE_COMPOSE)
    assert re.search(r"--workers\s+2\b", cmd), (
        f"secure api gunicorn command must use --workers 2; got {cmd!r}"
    )
    assert not re.search(r"--workers\s+4\b", cmd), (
        "secure api gunicorn command must NOT use --workers 4"
    )


def test_api_env_workers_is_two_base_compose():
    """The redundant WORKERS env var (mapping-style in docker-compose.yml) must
    advertise 2, matching the gunicorn --workers flag."""
    api = _services(_load(BASE_COMPOSE)).get("api") or {}
    env = api.get("environment") or {}
    assert isinstance(env, dict), "docker-compose.yml api environment is mapping-style"
    assert str(env.get("WORKERS")) == "2", (
        f"WORKERS env must be 2 to match --workers 2; got {env.get('WORKERS')!r}"
    )


def test_api_env_workers_is_two_secure_compose():
    """The redundant WORKERS env var (list-style in docker-compose.secure.yml)
    must advertise 2."""
    api = _services(_load(SECURE_COMPOSE)).get("api") or {}
    env = api.get("environment") or []
    # secure.yml uses list-style env entries (``- WORKERS=4``).
    entries = env if isinstance(env, list) else [f"{k}={v}" for k, v in env.items()]
    worker_entries = [e for e in entries if str(e).startswith("WORKERS=")]
    assert worker_entries, f"secure api must declare a WORKERS env entry; got {entries!r}"
    for e in worker_entries:
        assert e == "WORKERS=2", f"secure api WORKERS env must be 2; got {e!r}"


def test_dockerfile_cmd_uses_two_workers():
    """The baked production CMD JSON array must request exactly 2 workers, not 4."""
    text = DOCKERFILE.read_text()
    assert '"--workers", "2"' in text, 'Dockerfile CMD must use ["--workers", "2"] (OOM bound)'
    assert '"--workers", "4"' not in text, (
        'Dockerfile CMD must NOT use ["--workers", "4"] (OOM regression class)'
    )


# ---------------------------------------------------------------------------
# Priority 3 (OOM): gunicorn --config wiring (preload + gc.freeze, dark by default)
# ---------------------------------------------------------------------------

GUNICORN_CONF_FLAG = "--config /app/config/gunicorn.conf.py"


def test_api_command_uses_gunicorn_config_base_compose() -> None:
    """API command must pass --config /app/config/gunicorn.conf.py (base compose)."""
    command = _api_command_str(BASE_COMPOSE)
    assert GUNICORN_CONF_FLAG in command, (
        f"Expected '{GUNICORN_CONF_FLAG}' in API command, got: {command!r}"
    )


def test_api_command_uses_gunicorn_config_secure_compose() -> None:
    """Secure compose API command must pass --config /app/config/gunicorn.conf.py."""
    command = _api_command_str(SECURE_COMPOSE)
    assert GUNICORN_CONF_FLAG in command, (
        f"Expected '{GUNICORN_CONF_FLAG}' in secure API command, got: {command!r}"
    )


def test_dockerfile_cmd_uses_gunicorn_config() -> None:
    """Dockerfile baked CMD must pass the gunicorn --config path."""
    content = DOCKERFILE.read_text()
    assert '"--config", "/app/config/gunicorn.conf.py"' in content, (
        'Expected \'"--config", "/app/config/gunicorn.conf.py"\' in Dockerfile CMD'
    )


def test_api_command_preserves_workers_2_with_config_base() -> None:
    """--workers 2 must remain (config does not set workers; CLI owns it)."""
    command = _api_command_str(BASE_COMPOSE)
    assert re.search(r"--workers\s+2\b", command), (
        f"Expected '--workers 2' preserved alongside --config, got: {command!r}"
    )


def test_api_command_preserves_workers_2_with_config_secure() -> None:
    command = _api_command_str(SECURE_COMPOSE)
    assert re.search(r"--workers\s+2\b", command), (
        f"Expected '--workers 2' preserved in secure command, got: {command!r}"
    )


def test_dockerfile_cmd_preserves_workers_2_with_config() -> None:
    content = DOCKERFILE.read_text()
    assert '"--workers", "2"' in content, (
        'Expected \'"--workers", "2"\' preserved in Dockerfile CMD'
    )


def test_dockerfile_copies_config_for_gunicorn_conf() -> None:
    """Production stage must COPY config/ so the gunicorn conf is baked in."""
    text = DOCKERFILE.read_text()
    prod_split = text.split("AS production", 1)
    assert len(prod_split) == 2, "Dockerfile has no `AS production` stage"
    prod_region = prod_split[1]
    assert "COPY config/" in prod_region, (
        "production stage must COPY config/ so config/gunicorn.conf.py lands in "
        "the read-only image (the --config path points into it)"
    )


def test_gunicorn_conf_file_committed() -> None:
    """The gunicorn config the --config flag points at must exist in the repo."""
    conf = REPO_ROOT / "config" / "gunicorn.conf.py"
    assert conf.is_file(), f"missing {conf}"


# P2 worker_heavy right-sizing bounds for the 16GB box (was 32G/16cpus). A single
# heavy SHAP/twin task peaks ~1.3 GiB; cap the declared limit so a future accidental
# replicas>0 cannot reserve a box-sinking allocation. Concurrency stays low so each
# full-app prefork child clears the shared #565 per-child floor (MIN_MB_PER_FULL_APP_CHILD).
P2_WORKER_HEAVY_MAX_MEMORY_MB = 4096
P2_WORKER_HEAVY_MAX_CONCURRENCY = 2


def _worker_heavy() -> dict:
    """The base-compose worker_heavy service (the P2 offload target)."""
    svc = _services(_load(BASE_COMPOSE)).get("worker_heavy")
    assert svc, "worker_heavy service must exist in the base compose (P2 offload target)"
    return svc


def test_worker_heavy_stays_replicas_zero():
    """worker_heavy must remain replicas: 0 — the P2 offload is dark and the box
    has no headroom to run a heavy worker yet. Enabling it is a deliberate scale-up
    decision, not a silent default."""
    svc = _worker_heavy()
    replicas = (svc.get("deploy") or {}).get("replicas")
    assert replicas == 0, f"worker_heavy must stay replicas: 0 (P2 dark default); got {replicas!r}"


def test_worker_heavy_is_right_sized_for_the_16gb_box():
    """worker_heavy must be right-sized for the 16GB box (was 32G/16cpus). A single
    heavy SHAP/twin task peaks ~1.3 GiB; the limit must be modest so a future
    accidental replicas>0 cannot reserve a box-sinking 32G."""
    svc = _worker_heavy()
    limit_mb = _limit_mb(svc)
    assert limit_mb <= P2_WORKER_HEAVY_MAX_MEMORY_MB, (
        f"worker_heavy memory limit {limit_mb:.0f}MB must be <= "
        f"{P2_WORKER_HEAVY_MAX_MEMORY_MB}MB for the 16GB box (was 32G)"
    )
    # Per-child budget guard (shared with the #565 floor) must still hold.
    conc = _worker_concurrency(svc)
    assert conc <= P2_WORKER_HEAVY_MAX_CONCURRENCY, (
        f"worker_heavy --concurrency must be <= {P2_WORKER_HEAVY_MAX_CONCURRENCY} "
        f"on the 16GB box; got {conc}"
    )
    assert limit_mb / conc >= MIN_MB_PER_FULL_APP_CHILD, (
        f"worker_heavy {limit_mb:.0f}MB / concurrency {conc} = {limit_mb / conc:.0f}MB "
        f"per child < {MIN_MB_PER_FULL_APP_CHILD}MB floor"
    )


def test_worker_heavy_still_consumes_the_heavy_queues():
    """Right-sizing must not drop the queues the offload tasks route to
    (shap/twins) — else enqueued tasks would never be consumed."""
    svc = _worker_heavy()
    cmd = svc.get("command", "")
    if isinstance(cmd, list):
        cmd = " ".join(str(c) for c in cmd)
    m = re.search(r"--queues[= ](\S+)", cmd)
    assert m, f"worker_heavy must declare --queues; got {cmd!r}"
    queues = set(m.group(1).split(","))
    assert {"shap", "twins"} <= queues, (
        f"worker_heavy must consume the shap + twins queues (P2 offload targets); got {queues}"
    )


def test_heavy_offload_flag_not_baked_on_in_any_compose():
    """HEAVY_OFFLOAD_ENABLED must NOT be set truthy in any compose env — the
    offload ships DARK; enabling it is an explicit ops action, not a baked default."""
    truthy = {"1", "true", "yes", "on"}
    offenders = []
    for label, compose in (
        ("base", BASE_COMPOSE),
        ("dev", DEV_COMPOSE),
        ("secure", SECURE_COMPOSE),
    ):
        for name, svc in _services(_load(compose)).items():
            env = svc.get("environment")
            if env is None:
                continue
            if isinstance(env, dict):
                val = env.get("HEAVY_OFFLOAD_ENABLED")
            else:  # list-style "KEY=VALUE"
                val = None
                for entry in env:
                    if str(entry).startswith("HEAVY_OFFLOAD_ENABLED="):
                        val = str(entry).split("=", 1)[1]
            if val is not None and str(val).strip().lower() in truthy:
                offenders.append(f"{label}:{name}={val}")
    assert not offenders, (
        "HEAVY_OFFLOAD_ENABLED is baked truthy (P2 must ship dark) in: " + ", ".join(offenders)
    )


# =============================================================================
# 2026-06-23 deploy-failure class — the BASE prod compose bentoml service must
# carry a gold-standard serving bundle source (not just the dev overlay).
# =============================================================================
# INCIDENT (deploy run 28055627598, batch T6 #1092 + T5 #1091 + T8 #1090): T5's
# change-gated bentoml force-recreate ran under the BASE prod compose — pick_overlay()
# in deploy.yml returns "" once the frontend Dockerfile has an `AS production` stage
# (#528-B), so NO overlay is layered. But the base bentoml service mounts neither the
# shap_serving cohort bundles NOR the gold_standard_eval package. The recreated
# container booted in DEGRADED mode (available_models=[], model_loaded=false), failed
# the deploy's POST /model_info readiness gate ("bentoml has no cohort bundles loaded
# at /model_info"), and the ensuing rollback ran the OOM-prone frontend React build
# (the #563 double-fault), failing the deploy. The 12 cohort bundles had only ever
# been served by the dev-overlay container e2i_bentoml_dev (docker-compose.dev.yml),
# which binds them; deploys never recreated bentoml until T5 added that step. So the
# base service that the deploy actually recreates MUST be self-sufficient for serving.
#
# These guards pin the declared mounts; the faithful runtime proof is the deploy's
# own /model_info readiness gate against a real recreated container.

# Substring form is robust to the compose `../`-relative source prefix and the
# trailing `:ro` access mode (`../data/...:/home/bentoml/...:ro`).
BENTOML_BUNDLE_MOUNT = "data/ml_artifacts/shap_serving:/home/bentoml/data/ml_artifacts/shap_serving"
BENTOML_GOLDSTD_MOUNT = "src/mlops/gold_standard_eval:/home/bentoml/src/mlops/gold_standard_eval"


def _bentoml_volumes(compose_path: Path) -> list[str]:
    svc = _services(_load(compose_path)).get("bentoml") or {}
    return [str(v) for v in (svc.get("volumes") or [])]


def test_base_bentoml_binds_shap_serving_bundles():
    """The deploy force-recreates bentoml under the BASE prod compose (no overlay,
    #528-B), so the base bentoml service must bind the shap_serving bundle dir — else
    a recreated container FS-discovers ZERO cohort bundles and boots degraded (the
    2026-06-23 deploy-failure root cause). The serving FS fallback
    (_discover_goldstd_bundles_from_fs in scripts/bentoml/e2i_serving_service.py) walks
    ``data/ml_artifacts/shap_serving`` relative to the /home/bentoml workdir, so the
    mount target must land exactly there."""
    vols = "\n".join(_bentoml_volumes(BASE_COMPOSE))
    assert BENTOML_BUNDLE_MOUNT in vols, (
        "base bentoml must bind the shap_serving bundle dir at its in-container "
        "target (data/ml_artifacts/shap_serving under the bentoml workdir) so a "
        "deploy-recreated prod container can serve the 12 cohort bundles (it boots "
        f"degraded with 0 models otherwise — the deploy /model_info gate then fails). "
        f"volumes=\n{vols}"
    )


def test_base_bentoml_binds_gold_standard_eval_for_unpickling():
    """The cohort bundles pickle a FeatureBuilder (src.mlops.gold_standard_eval.
    feature_builder; its only dep cohort_spec also lives in that package). The prod
    bentoml image does NOT bake gold_standard_eval, so pickle.load raises
    ModuleNotFoundError and EVERY bundle is silently skipped (0 models) unless the
    package is mounted at the matching import path. Bind it read-only."""
    vols = "\n".join(_bentoml_volumes(BASE_COMPOSE))
    assert BENTOML_GOLDSTD_MOUNT in vols, (
        "base bentoml must bind src/mlops/gold_standard_eval at its in-container "
        "import path (under the bentoml workdir) so pickle.load can import "
        "FeatureBuilder+cohort_spec to deserialize the bundles (unpickle "
        f"ModuleNotFoundError -> 0 models otherwise). volumes=\n{vols}"
    )


def test_base_bentoml_serving_binds_are_read_only():
    """Both serving binds are READ-ONLY: bentoml serves from them, never writes. A
    writable bind would let the serving container mutate the repo's gold_standard_eval
    source or the materialized bundles (which sync_goldstd_serving owns)."""
    vols = _bentoml_volumes(BASE_COMPOSE)
    for needle in (BENTOML_BUNDLE_MOUNT, BENTOML_GOLDSTD_MOUNT):
        entry = next((v for v in vols if needle in v), None)
        assert entry is not None, f"missing bentoml serving bind for {needle}; volumes={vols}"
        assert entry.rstrip().endswith(":ro"), (
            f"bentoml serving bind must be read-only (:ro); got {entry!r}"
        )


# =============================================================================
# #1479 mlflow v3.15 security middleware - cross-container Host header
# =============================================================================
def test_mlflow_server_allows_compose_dns_host_header():
    """mlflow >=3.15 ships security middleware that 403s any Host header outside
    localhost/private-IPs ("Invalid Host header - possible DNS rebinding attack
    detected"). Every app container reaches the tracking server as
    http://mlflow:5000 (x-common-env MLFLOW_TRACKING_URI), so the server command
    must explicitly allow the compose DNS name — measured live 2026-08-05 right
    after the #1479 recreate: without --allowed-hosts every cross-container
    tracking call failed with 403 while the localhost healthcheck stayed green.
    Setting the flag REPLACES the built-in default (localhost + private IPs),
    so localhost/127.0.0.1 must be re-listed for the healthcheck + host-side UI.

    Matching semantics (mlflow/server/security_utils.py, is_allowed_host_header):
    EXACT string equality unless the entry contains '*', then fnmatch. The Host
    header carries the port ('mlflow:5000'), so a bare 'mlflow' entry never
    matches — measured live: the first fix attempt with bare names still 403'd.
    This test replicates the real matcher against the real Host headers."""
    doc = _load(REPO_ROOT / "docker" / "docker-compose.yml")
    command = _services(doc)["mlflow"]["command"]
    assert "--allowed-hosts" in command, (
        "mlflow server command must set --allowed-hosts (v3.15 middleware 403s "
        "the compose-DNS Host header 'mlflow:5000' otherwise)"
    )
    match = re.search(r"--allowed-hosts\s+(\S+)", command)
    assert match, f"--allowed-hosts must carry an inline value: {command!r}"
    hosts = match.group(1).split(",")

    def _allowed(host_header: str) -> bool:
        return any(
            fnmatch.fnmatch(host_header, entry) if "*" in entry else host_header == entry
            for entry in hosts
        )

    # The three real Host headers this server receives:
    assert _allowed("mlflow:5000"), (
        f"app containers send Host 'mlflow:5000' (x-common-env tracking URI) — "
        f"not matched by --allowed-hosts {hosts}"
    )
    assert _allowed("localhost:5000"), (
        f"container healthcheck sends Host 'localhost:5000' — not matched by {hosts}"
    )
    assert _allowed("127.0.0.1:5000"), (
        f"host-side UI sends Host '127.0.0.1:5000' — not matched by {hosts}"
    )
