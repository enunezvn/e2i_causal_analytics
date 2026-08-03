"""#1448 — the deployed image must carry the ``pyproject.toml`` project-root marker.

``src/utils/project_root.py`` resolves the project root by walking UP from its own
location to a ``pyproject.toml`` marker. Three Tier-0 agents call it at *module
import* time (``model_trainer`` via
``nodes/detect_class_imbalance._DEFAULT_CONFIG_PATH``; ``model_selector`` and
``model_deployer`` transitively through the trainer's package), so a missing marker
under ``/app`` makes all three raise ``ProjectRootNotFoundError`` during
``create_agent_registry`` — the prod PARTIAL-registry incident (18/21 agents).

The ``dependencies`` and ``development`` stages already ``COPY … pyproject.toml ./``;
the ``production`` stage (the one prod actually runs — ``docker-compose.yml`` pins
``target: production`` for api/worker/scheduler) starts from a fresh
``python:3.12-slim-bookworm`` and copied only ``src/``, ``config/`` and ``scripts/``.

These checks are hermetic: they parse ``docker/Dockerfile`` statically, materialise
the paths a stage bakes into the image under ``tmp_path``, and run the REAL
``find_project_root`` over that tree. No docker daemon, no build. A real image build
is still required to confirm the deployed artifact (see the issue/PR notes).
"""

from __future__ import annotations

import posixpath
import re
import shlex
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest
import yaml  # type: ignore[import-untyped]

from src.agents.factory import REQUIRE_FULL_REGISTRY_ENV
from src.utils.project_root import ProjectRootNotFoundError, find_project_root

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"
BASE_COMPOSE = REPO_ROOT / "docker" / "docker-compose.yml"

# Marker set that ``find_project_root`` uses by default, and therefore the set the
# image must satisfy for the Tier-0 agents to import.
MARKER = "pyproject.toml"

# Path of the resolver module inside the image, relative to the stage WORKDIR.
RESOLVER_RELPATH = "src/utils/project_root.py"


# ---------------------------------------------------------------------------
# Minimal Dockerfile parser (only FROM / WORKDIR / COPY are interpreted)
# ---------------------------------------------------------------------------


def _logical_lines(text: str) -> List[str]:
    """Join backslash continuations, drop blank + comment lines."""
    out: List[str] = []
    buf = ""
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line.endswith("\\"):
            buf += line[:-1].strip() + " "
            continue
        buf += stripped
        out.append(buf.strip())
        buf = ""
    if buf:
        out.append(buf.strip())
    return out


class Stage:
    """One ``FROM … AS <name>`` block."""

    def __init__(self, name: str, parent: str) -> None:
        self.name = name
        self.parent = parent
        self.instructions: List[str] = []

    @property
    def workdir_explicit(self) -> Optional[str]:
        for ins in self.instructions:
            if ins.upper().startswith("WORKDIR "):
                return ins.split(None, 1)[1].strip()
        return None

    def copies(self) -> List[Tuple[List[str], List[str], str]]:
        """Return ``(flags, sources, dest)`` for every COPY in this stage."""
        out: List[Tuple[List[str], List[str], str]] = []
        for ins in self.instructions:
            if not ins.upper().startswith("COPY "):
                continue
            parts = shlex.split(ins)[1:]
            flags = [p for p in parts if p.startswith("--")]
            args = [p for p in parts if not p.startswith("--")]
            if len(args) < 2:
                continue
            out.append((flags, args[:-1], args[-1]))
        return out


def _parse_stages() -> Dict[str, Stage]:
    stages: Dict[str, Stage] = {}
    current: Optional[Stage] = None
    for line in _logical_lines(DOCKERFILE.read_text()):
        if line.upper().startswith("FROM "):
            body = line.split(None, 1)[1]
            m = re.search(r"\s+AS\s+(\S+)\s*$", body, re.IGNORECASE)
            parent = body.split()[0]
            name = m.group(1) if m else f"_anon{len(stages)}"
            current = Stage(name=name, parent=parent)
            stages[name] = current
        elif current is not None:
            current.instructions.append(line)
    return stages


def _workdir(stages: Dict[str, Stage], name: str) -> str:
    """Resolve a stage's WORKDIR, inheriting from its local parent stage."""
    seen = set()
    cursor: Optional[str] = name
    while cursor and cursor in stages and cursor not in seen:
        seen.add(cursor)
        explicit = stages[cursor].workdir_explicit
        if explicit:
            return explicit
        cursor = stages[cursor].parent
    return "/"


def _stages_shipping_src(stages: Dict[str, Stage]) -> List[str]:
    """Stages that bake the application source tree into the image."""
    shipping = []
    for name, stage in stages.items():
        for _flags, srcs, _dest in stage.copies():
            if any(s.rstrip("/") == "src" for s in srcs):
                shipping.append(name)
                break
    return shipping


def _materialize(stages: Dict[str, Stage], name: str, root: Path) -> Path:
    """Create, under ``root``, the paths the stage's COPYs bake into the image.

    Returns the image WORKDIR as a real directory under ``root``.
    """
    workdir = _workdir(stages, name)
    stage = stages[name]

    def _abs(p: str) -> Path:
        joined = p if p.startswith("/") else posixpath.join(workdir, p)
        return root / posixpath.normpath(joined).lstrip("/")

    for _flags, srcs, dest in stage.copies():
        dest_is_dir = dest.endswith("/") or dest in (".", "..")
        for src in srcs:
            src_is_dir = src.endswith("/")
            target = _abs(dest) / Path(src.rstrip("/")).name if dest_is_dir else _abs(dest)
            if src_is_dir:
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()

    image_workdir = _abs(".")
    image_workdir.mkdir(parents=True, exist_ok=True)
    return image_workdir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def stages() -> Dict[str, Stage]:
    return _parse_stages()


def test_parser_sees_the_expected_stages(stages):
    """Guard the parser itself: if the Dockerfile is restructured, fail here (a
    clear parser error) rather than vacuously passing the marker assertions."""
    assert {"base", "dependencies", "development", "production"} <= set(stages)
    assert _workdir(stages, "production") == "/app"
    assert _workdir(stages, "development") == "/app"


def test_every_stage_shipping_src_also_ships_the_project_root_marker(stages):
    """#1448: a stage that bakes ``src/`` MUST also provide ``pyproject.toml`` at
    its WORKDIR, or every module-level ``find_project_root()`` call in that source
    tree raises at import time."""
    shipping = _stages_shipping_src(stages)
    assert shipping, "expected at least one stage to COPY src/ into the image"

    missing = []
    for name in shipping:
        workdir = _workdir(stages, name)
        has_marker = False
        for _flags, srcs, dest in stages[name].copies():
            dest_is_dir = dest.endswith("/") or dest in (".", "..")
            for src in srcs:
                baked = posixpath.join(dest, Path(src).name) if dest_is_dir else dest
                baked_abs = posixpath.normpath(
                    baked if baked.startswith("/") else posixpath.join(workdir, baked)
                )
                if baked_abs == posixpath.join(workdir, MARKER):
                    has_marker = True
        if not has_marker:
            missing.append(name)

    assert not missing, (
        f"stage(s) {missing} COPY src/ into the image but never place {MARKER!r} at "
        f"the WORKDIR — find_project_root() will raise for every module that calls it "
        f"at import time (model_selector / model_trainer / model_deployer)."
    )


@pytest.mark.parametrize("stage_name", ["production", "development"])
def test_find_project_root_resolves_against_the_real_image_layout(
    stage_name, stages, tmp_path, monkeypatch
):
    """Run the REAL resolver over a tree materialised from the stage's COPYs.

    This is the behavioural assertion behind the static one above: it fails with the
    exact prod error string when the marker is absent.
    """
    monkeypatch.delenv("E2I_CONFIG_DIR", raising=False)

    image_root = _materialize(stages, stage_name, tmp_path / stage_name)
    resolver = image_root / RESOLVER_RELPATH
    resolver.parent.mkdir(parents=True, exist_ok=True)
    resolver.touch()

    try:
        resolved = find_project_root(start=resolver)
    except ProjectRootNotFoundError as exc:  # pragma: no cover - failure path
        pytest.fail(
            f"{stage_name} image layout cannot resolve a project root: {exc}. "
            f"The image must ship {MARKER!r} at the stage WORKDIR."
        )

    assert resolved == image_root, (
        f"expected the {stage_name} WORKDIR ({image_root}) to be the project root, got {resolved}"
    )
    # config/ is what the resolved root is actually used for (observability.yaml,
    # imbalance_strategy.yaml) — assert the stage ships it so the resolution is useful.
    assert (resolved / "config").is_dir()


# ---------------------------------------------------------------------------
# #1448 — the registry-completeness gate must be reachable from the host .env
# ---------------------------------------------------------------------------


class _ComposeLoader(yaml.SafeLoader):
    """SafeLoader tolerating compose's local ``!override`` / ``!reset`` tags."""


def _passthrough(loader: yaml.Loader, tag_suffix: str, node: yaml.Node):  # noqa: ANN401
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


_ComposeLoader.add_multi_constructor("!", _passthrough)


def test_registry_gate_env_is_forwarded_to_the_containers():
    """``x-common-env`` is an explicit whitelist: a variable absent from it NEVER
    reaches api/worker/scheduler, so the in-code default governs regardless of the
    host ``.env`` (the OPIK_ENABLED / OPENAI_API_KEY lesson documented in that
    block). An un-forwarded gate flag would be an inert no-op in prod."""
    with open(BASE_COMPOSE) as fh:
        doc = yaml.load(fh, Loader=_ComposeLoader) or {}

    common_env = doc.get("x-common-env") or {}
    assert REQUIRE_FULL_REGISTRY_ENV in common_env, (
        f"{REQUIRE_FULL_REGISTRY_ENV} must be forwarded via x-common-env or arming "
        "the agent-registry completeness gate from the host .env has no effect."
    )
    # Default must keep the gate DISARMED so a partial registry degrades (and now
    # alerts at ERROR) instead of taking the API down on an unrelated agent fault.
    assert "false" in str(common_env[REQUIRE_FULL_REGISTRY_ENV]).lower()
