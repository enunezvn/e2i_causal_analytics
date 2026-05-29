"""#528-B — frontend production (nginx) image invariants (hermetic, no docker/DB).

#528-A flipped the app services to the prod target but kept the frontend on dev/Vite
because its production nginx stage had been deleted in aa58e71c (it had no consumer
then). #528-B restores that nginx image so the frontend serves a real production
build, and flips the frontend off the slim dev overlay.

The nginx production stage runs as the non-root ``e2i`` user (uid 1000) under a
``read_only`` rootfs, so its writable scratch dirs (``/var/cache/nginx``, ``/var/run``)
come from tmpfs — and those tmpfs MUST be owned by uid 1000, exactly the #527 lesson.
Verified out-of-band: with ``mode=0770`` and no ``uid=`` the container dies with
``mkdir() "/var/cache/nginx/client_temp" failed (13: Permission denied)``; adding
``uid=1000,gid=1000`` lets nginx start and serve ``/health``.

These checks parse the Dockerfile + compose statically (CI-runnable). The full image
build is validated by the frontend build-push CI job on a clean runner.
"""

from __future__ import annotations

from pathlib import Path

import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_COMPOSE = REPO_ROOT / "docker" / "docker-compose.yml"
FRONTEND_DOCKERFILE = REPO_ROOT / "docker" / "frontend" / "Dockerfile"
NGINX_CONF = REPO_ROOT / "docker" / "frontend" / "nginx.conf"
DEPLOY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy.yml"


class _ComposeLoader(yaml.SafeLoader):
    """SafeLoader tolerating compose's local ``!override`` / ``!reset`` tags."""


def _passthrough(loader: yaml.Loader, tag_suffix: str, node: yaml.Node):  # noqa: ANN401
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


_ComposeLoader.add_multi_constructor("!", _passthrough)


def _frontend_service() -> dict:
    with open(BASE_COMPOSE) as fh:
        doc = yaml.load(fh, Loader=_ComposeLoader) or {}
    return (doc.get("services") or {}).get("frontend") or {}


def _parse_tmpfs(entry: str) -> tuple[str, dict[str, str]]:
    path, _, opt_str = entry.partition(":")
    opts: dict[str, str] = {}
    for kv in opt_str.split(","):
        kv = kv.strip()
        if not kv:
            continue
        key, _, val = kv.partition("=")
        opts[key.strip()] = val.strip()
    return path, opts


# --------------------------------------------------------------------------- #
# Dockerfile: the revived builder + production stages
# --------------------------------------------------------------------------- #
def test_frontend_dockerfile_has_builder_and_production_stages():
    text = FRONTEND_DOCKERFILE.read_text()
    assert "AS builder" in text, "frontend Dockerfile must have the production build stage"
    assert "AS production" in text, (
        "frontend Dockerfile must have the nginx production stage (#528-B)"
    )


def test_frontend_production_stage_serves_nginx_as_nonroot():
    text = FRONTEND_DOCKERFILE.read_text()
    prod = text.split("AS production", 1)[1]
    assert (
        "nginx" in prod.split("AS production", 1)[0].lower()
        or "nginx" in text.split("AS production")[0][-80:]
    ), "production stage must be based on an nginx image"
    assert "FROM nginx" in text, "production stage must FROM an nginx base image"
    assert "COPY docker/frontend/nginx.conf" in text, (
        "production stage must install the custom nginx.conf"
    )
    assert "COPY --from=builder" in text, (
        "production stage must copy the built static assets from builder"
    )
    assert "USER e2i" in prod, "nginx must run as the non-root e2i user (uid 1000)"
    assert "EXPOSE 80" in prod, "production stage must expose port 80"
    assert "/health" in prod, "production stage must provide a /health endpoint"


def test_frontend_nginx_conf_exists():
    assert NGINX_CONF.exists(), "docker/frontend/nginx.conf must be restored (#528-B)"
    assert "upstream api_backend" in NGINX_CONF.read_text(), "nginx.conf must reverse-proxy the api"


# --------------------------------------------------------------------------- #
# Compose: the frontend tmpfs must be writable by the non-root nginx (uid 1000)
# --------------------------------------------------------------------------- #
def test_frontend_tmpfs_writable_by_nonroot_nginx():
    """read_only rootfs + non-root nginx ⇒ the scratch tmpfs must be uid 1000.

    Without uid=1000 the tmpfs is root-owned mode 0770 and nginx (uid 1000) cannot
    create /var/cache/nginx/client_temp — the container emerg-exits. This is the
    #527 non-root-tmpfs lesson applied to the revived frontend image.
    """
    fe = _frontend_service()
    assert fe.get("read_only") is True, "frontend prod image runs read_only (regression guard)"
    tmpfs = fe.get("tmpfs") or []
    tmpfs = [tmpfs] if isinstance(tmpfs, str) else list(tmpfs)
    seen = {}
    for entry in tmpfs:
        path, opts = _parse_tmpfs(entry)
        seen[path] = opts
    for required in ("/var/cache/nginx", "/var/run"):
        assert required in seen, f"frontend must mount a writable tmpfs at {required}"
        opts = seen[required]
        assert opts.get("uid") == "1000", (
            f"{required} tmpfs must set uid=1000 so non-root nginx can write it "
            f"(else: mkdir client_temp Permission denied); got {opts}"
        )
        assert opts.get("gid") == "1000", f"{required} tmpfs must set gid=1000; got {opts}"


# --------------------------------------------------------------------------- #
# GHCR pull (B3): build-push job, image refs, droplet pull with local fallback
# --------------------------------------------------------------------------- #
def test_frontend_build_push_job_exists_and_deploy_needs_it():
    """A frontend image must be built+pushed to GHCR and gate the deploy.

    Only e2i-api was pushed before; without a frontend push job the droplet has no
    frontend image to pull. The deploy job must `needs` it so the pull can't race a
    missing image.
    """
    doc = yaml.safe_load(DEPLOY_WORKFLOW.read_text())
    jobs = doc.get("jobs", {}) or {}
    assert "build-and-push-frontend" in jobs, (
        "deploy.yml must define a frontend build-push job (#528-B)"
    )
    fe = jobs["build-and-push-frontend"]
    steps_text = yaml.safe_dump(fe)
    assert "docker/frontend/Dockerfile" in steps_text, (
        "frontend job must build the frontend Dockerfile"
    )
    assert "production" in steps_text, "frontend job must build the production target"
    needs = jobs.get("deploy", {}).get("needs", [])
    needs = [needs] if isinstance(needs, str) else list(needs)
    assert "build-and-push-frontend" in needs, "deploy must depend on the frontend build-push job"


def test_compose_services_have_ghcr_image_refs():
    """app + frontend services carry a GHCR image: ref so the droplet can pull them.

    Without an image: ref `docker compose pull` has nothing to fetch and the droplet
    falls back to building locally — defeating the GHCR-pull goal.
    """
    with open(BASE_COMPOSE) as fh:
        doc = yaml.load(fh, Loader=_ComposeLoader) or {}
    svcs = doc.get("services") or {}
    for name, img_substr in (
        ("api", "e2i-api"),
        ("worker_light", "e2i-api"),
        ("worker_medium", "e2i-api"),
        ("scheduler", "e2i-api"),
        ("frontend", "e2i-frontend"),
    ):
        image = str(svcs.get(name, {}).get("image", ""))
        assert "ghcr.io" in image and img_substr in image, (
            f"{name} must have a GHCR image: ref containing {img_substr!r}; got {image!r}"
        )
        # Tag must be parameterized by the deployed commit (not a frozen tag).
        assert "IMAGE_TAG" in image, (
            f"{name} image tag must be ${{IMAGE_TAG}}-parameterized; got {image!r}"
        )


def test_droplet_pulls_from_ghcr_with_local_build_fallback():
    """The droplet must log in + pull app images, but fall back to a local build.

    A registry hiccup (or a baked-input change) must never wedge the deploy — the pull
    is best-effort and the local build is the fallback (image: + build: coexist).
    """
    script = DEPLOY_WORKFLOW.read_text()
    assert "docker login" in script, "droplet must authenticate to GHCR before pulling"
    assert "$COMPOSE_CMD pull" in script, "droplet must pull the app images from GHCR"
    assert "APP_BUILD_FLAG" in script, "app-services rollout must use a pull-aware build flag"
    assert "--no-build" in script, (
        "a successful pull must run the app services without a local build"
    )
    # The fallback path must still build locally.
    assert script.count("--build") >= 1, "a failed pull must fall back to a local build"
