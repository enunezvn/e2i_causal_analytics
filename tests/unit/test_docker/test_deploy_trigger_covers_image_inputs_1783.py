"""#1783 — every repo path baked into the production image must be a deploy trigger.

ROOT CAUSE AS FOUND (verified against this repo, not inferred): ``docker/Dockerfile``'s
production stage ``COPY``s five repo paths into the image, and ``deploy.yml``'s
``on.push.paths`` covered only three of them:

    line 188  COPY pyproject.toml ./              -> 'pyproject.toml'      COVERED
    line 191  COPY src/ ./src/                    -> 'src/**'              COVERED
    line 192  COPY config/ ./config/              -> 'config/**'           COVERED
    line 202  COPY data/kg_cache/ ./data/kg_cache/ -> (nothing)            UNCOVERED
    line 208  COPY scripts/ ./scripts/            -> 'scripts/bentoml/**'  PARTIAL
                                                     'scripts/deploy/**'

``scripts/`` is the gap #1783 reports; ``data/kg_cache/`` is a second instance the issue
does not mention (3 files tracked in git, explicitly un-ignored in ``.dockerignore`` via
``!data/kg_cache/**``, matched by no trigger entry at all).

So a push touching ``scripts/seed_falkordb.py`` (executed as a SUBPROCESS out of the
baked image by #1761's graph-emptiness sentinel) or ``data/kg_cache/*.json`` (bound by
``src/data/kg/activation.py`` on the adaptive-validity hot path) changed what the image
WOULD contain without triggering the build that would produce it. Production then ran a
stale baked artifact until some unrelated ``src/**`` push rebuilt the image incidentally.
The window closed on its own, nothing bounded it, and nothing announced it — the same
"silent divergence that self-heals" family as #1479's five-week mlflow pin drift.

Both are now trigger paths. This test is the standing invariant, so the NEXT production
``COPY`` added without a matching trigger fails here instead of shipping the same defect
a third time.

WHY A STRUCTURAL TEST RATHER THAN A LIVE CERTIFICATION: ``.github/**`` is itself absent
from ``deploy.yml``'s ``on.push.paths``, so merging a trigger fix deploys nothing and
there is no live artifact to certify against. This test is the only gate, which is why
it carries its own positive control below.

NON-VACUITY (the failure mode this repo keeps hitting — a guard that asserts a boolean
over a silently-corrupted computation and stays green):
  * ``test_parser_discovers_the_real_production_image_inputs`` pins WHAT WAS COMPUTED —
    the reachable stage set and a floor of known image-input paths, each asserted to
    exist on disk. A parser that degrades to emitting nothing (or garbage tokens) fails
    here loudly instead of reporting a vacuous "0 uncovered".
  * ``test_positive_control_*`` runs the SAME extract -> match -> report pipeline over a
    synthetic Dockerfile whose uncovered paths are known, and asserts the exact reported
    set. A matcher that can never report anything fails here.
  * ``test_trigger_globs_all_have_a_recognised_shape`` fails closed if a future trigger
    entry uses a glob form ``_glob_covers`` does not understand, rather than letting the
    matcher silently under- or over-cover.

FAITHFULNESS LIMIT: this proves the deploy TRIGGER set covers the Dockerfile's declared
image inputs. It does not prove GitHub's own path-filter engine agrees with
``_glob_covers`` on exotic glob forms — which is precisely why the shape test restricts
the trigger vocabulary to the two forms (literal path, ``prefix/**``) whose semantics are
unambiguous.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEPLOY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy.yml"
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"

#: The stage the production image is built from (``docker build --target production``,
#: cf. deploy.yml's build-and-push job).
PRODUCTION_STAGE = "production"

_WILDCARD = re.compile(r"[*?\[]")


# --------------------------------------------------------------------------- #
# deploy.yml trigger extraction
# --------------------------------------------------------------------------- #
def _trigger_paths() -> list[str]:
    """``on.push.paths`` from deploy.yml.

    PyYAML (YAML 1.1) parses a bare ``on:`` mapping key as the boolean ``True``.
    """
    wf = yaml.safe_load(DEPLOY_WORKFLOW.read_text())
    on = wf.get("on")
    if on is None:
        on = wf.get(True)
    return list((on or {}).get("push", {}).get("paths", []) or [])


# --------------------------------------------------------------------------- #
# Dockerfile stage graph
# --------------------------------------------------------------------------- #
@dataclass
class _Copy:
    lineno: int
    sources: list[str]
    from_stage: str | None


@dataclass
class _Stage:
    name: str
    parent: str | None  # `FROM <stage> AS <name>` inheritance, if <stage> is a stage
    copies: list[_Copy] = field(default_factory=list)


def _logical_lines(text: str) -> list[tuple[int, str]]:
    """(1-based lineno of the FIRST physical line, joined instruction text).

    Joins ``\\``-continuations and drops whole-line comments, which Docker permits
    *between* continuation lines.
    """
    out: list[tuple[int, str]] = []
    buf: list[str] = []
    start = 0
    for idx, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not buf and (not stripped or stripped.startswith("#")):
            continue
        if buf and stripped.startswith("#"):
            continue  # comment inside a continuation
        if not buf:
            start = idx
        if stripped.endswith("\\"):
            buf.append(stripped[:-1])
            continue
        buf.append(stripped)
        out.append((start, " ".join(p.strip() for p in buf if p.strip())))
        buf = []
    if buf:
        out.append((start, " ".join(p.strip() for p in buf if p.strip())))
    return out


def _parse_stages(dockerfile_text: str) -> dict[str, _Stage]:
    """Build the stage graph: ``FROM ... AS <name>`` nodes with their ``COPY`` edges."""
    stages: dict[str, _Stage] = {}
    current: _Stage | None = None
    for lineno, line in _logical_lines(dockerfile_text):
        tokens = line.split()
        if not tokens:
            continue
        head = tokens[0].upper()
        if head == "FROM":
            rest = [t for t in tokens[1:] if not t.startswith("--")]
            base = rest[0] if rest else ""
            name = None
            for i, tok in enumerate(rest):
                if tok.upper() == "AS" and i + 1 < len(rest):
                    name = rest[i + 1]
            if name is None:
                # An unnamed stage cannot be targeted by --target and nothing can
                # COPY --from it by name, so it is not reachable from production.
                current = None
                continue
            current = _Stage(name=name, parent=base)
            stages[name] = current
            continue
        if head in {"COPY", "ADD"} and current is not None:
            assert "[" not in line and '"' not in line, (
                f"{DOCKERFILE.name}:{lineno} uses the JSON/exec form of {head}, which "
                f"this parser does not tokenize. Extend _parse_stages before relying "
                f"on this guard again. Line: {line!r}"
            )
            from_stage: str | None = None
            args: list[str] = []
            for tok in tokens[1:]:
                if tok.startswith("--"):
                    if tok.startswith("--from="):
                        from_stage = tok.split("=", 1)[1]
                    continue
                args.append(tok)
            assert len(args) >= 2, (
                f"{DOCKERFILE.name}:{lineno} — {head} needs >=1 source and a dest; "
                f"parsed args={args!r} from {line!r}"
            )
            current.copies.append(_Copy(lineno=lineno, sources=args[:-1], from_stage=from_stage))
    return stages


def _reachable_stages(stages: dict[str, _Stage], root: str) -> set[str]:
    """Stages whose content ends up in the ``root`` image.

    Two edge kinds, both real: ``FROM <stage> AS ...`` inheritance (root inherits the
    parent's filesystem) and ``COPY --from=<stage>`` (root pulls artifacts built there —
    e.g. production's ``COPY --from=dependencies /app/.venv``, which is exactly why
    ``requirements.lock`` and ``patches/`` are already deploy triggers).
    """
    seen: set[str] = set()
    stack = [root]
    while stack:
        name = stack.pop()
        if name in seen or name not in stages:
            continue
        seen.add(name)
        stage = stages[name]
        if stage.parent:
            stack.append(stage.parent)
        for cp in stage.copies:
            if cp.from_stage:
                stack.append(cp.from_stage)
    return seen


def _image_input_paths(dockerfile_text: str, root: str = PRODUCTION_STAGE) -> dict[str, list[str]]:
    """Repo paths baked into the ``root`` image -> ["<stage>:<lineno>", ...] provenance.

    ``COPY --from=<stage>`` sources are deliberately EXCLUDED: those are container paths
    inside another build stage (``/app/.venv``), not repo paths, and treating them as
    repo paths would produce false failures.
    """
    stages = _parse_stages(dockerfile_text)
    assert root in stages, (
        f"no `AS {root}` stage found in the Dockerfile; parsed stages={sorted(stages)}"
    )
    inputs: dict[str, list[str]] = {}
    for name in sorted(_reachable_stages(stages, root)):
        for cp in stages[name].copies:
            if cp.from_stage is not None:
                continue  # stage copy, not a repo path
            for src in cp.sources:
                norm = src[2:] if src.startswith("./") else src
                inputs.setdefault(norm, []).append(f"{name}:{cp.lineno}")
    return inputs


# --------------------------------------------------------------------------- #
# Trigger-glob matcher
# --------------------------------------------------------------------------- #
def _is_directory_source(src: str) -> bool:
    """A trailing ``/`` declares a directory; otherwise ask the repo."""
    if src.endswith("/"):
        return True
    return (REPO_ROOT / src).is_dir()


def _glob_covers(trigger: str, src: str) -> bool:
    """Does ``trigger`` guarantee that EVERY file the ``COPY`` of ``src`` bakes in
    fires the deploy?

    Coverage is all-or-nothing on purpose. ``scripts/bentoml/**`` matches *some* files
    under ``scripts/``, but ``COPY scripts/ ./scripts/`` bakes in the whole tree — so a
    subtree trigger does NOT cover a directory source. That partial-coverage case IS the
    #1783 defect, and a matcher that scored it as covered would be the exact fail-open
    this test exists to prevent.

    Only two trigger forms are understood, and ``test_trigger_globs_all_have_a_recognised_shape``
    holds the workflow to them:
      * ``prefix/**`` — covers ``prefix`` and everything beneath it.
      * a literal path — covers exactly that one file.
    Anything else returns False (fail CLOSED: an unrecognised form surfaces as an
    uncovered path someone must look at, never as a silent pass).
    """
    trigger = trigger.strip()
    is_dir = _is_directory_source(src)
    path = src.rstrip("/")
    if trigger == "**":
        return True
    if trigger.startswith("!"):
        return False  # a negation can only ever REMOVE coverage
    if trigger.endswith("/**"):
        prefix = trigger[: -len("/**")]
        if _WILDCARD.search(prefix):
            return False
        return path == prefix or path.startswith(prefix + "/")
    if _WILDCARD.search(trigger):
        return False
    return (not is_dir) and path == trigger


def _uncovered(inputs: dict[str, list[str]], triggers: list[str]) -> dict[str, list[str]]:
    """Image-input paths that no trigger glob covers, with their provenance."""
    return {
        src: prov
        for src, prov in sorted(inputs.items())
        if not any(_glob_covers(t, src) for t in triggers)
    }


def _render(uncovered: dict[str, list[str]]) -> str:
    return "\n".join(f"  {src}  (Dockerfile {', '.join(prov)})" for src, prov in uncovered.items())


# --------------------------------------------------------------------------- #
# Matcher semantics — the logic the invariant rests on, exercised directly
# --------------------------------------------------------------------------- #
def test_glob_covers_semantics() -> None:
    """Table-driven: pin what ``_glob_covers`` decides, including the partial-coverage
    case that is the whole point of #1783."""
    covered = [
        ("src/**", "src/"),
        ("src/**", "src"),
        ("config/**", "config/"),
        ("data/**", "data/kg_cache/"),  # ancestor subtree covers a descendant dir
        ("scripts/**", "scripts/"),
        ("pyproject.toml", "pyproject.toml"),
        ("requirements.lock", "requirements.lock"),
        ("docker/**", "docker/Dockerfile"),  # ancestor subtree covers a file
        ("**", "anything/"),
    ]
    for trigger, src in covered:
        assert _glob_covers(trigger, src), f"{trigger!r} should cover {src!r}"

    not_covered = [
        # THE #1783 SHAPE: a subtree trigger does not cover the whole directory COPY.
        ("scripts/bentoml/**", "scripts/"),
        ("scripts/deploy/**", "scripts/"),
        ("data/kg_cache/**", "data/"),
        # A literal file trigger never covers a directory source.
        ("scripts/seed_falkordb.py", "scripts/"),
        # Sibling subtree.
        ("src/**", "config/"),
        # Prefix-string collision must not be mistaken for a path-component prefix.
        ("script/**", "scripts/"),
        ("src/**", "srcfoo/"),
        # Unrecognised glob forms fail CLOSED.
        ("scripts/*.py", "scripts/"),
        ("**/seed_falkordb.py", "scripts/"),
        ("!scripts/**", "scripts/"),
    ]
    for trigger, src in not_covered:
        assert not _glob_covers(trigger, src), f"{trigger!r} must NOT cover {src!r}"


def test_trigger_globs_all_have_a_recognised_shape() -> None:
    """Fail closed if deploy.yml grows a glob form ``_glob_covers`` cannot reason about.

    Without this, a future ``scripts/**/*.py`` entry would silently score as covering
    nothing (or, worse under a laxer matcher, as covering everything) and the invariant
    below would quietly stop meaning what it says.
    """
    triggers = _trigger_paths()
    assert triggers, "deploy.yml on.push.paths parsed as EMPTY — the parser is broken"
    unrecognised = [
        t
        for t in triggers
        if not (
            t == "**"
            or (t.endswith("/**") and not _WILDCARD.search(t[: -len("/**")]))
            or not _WILDCARD.search(t)
        )
    ]
    assert unrecognised == [], (
        "deploy.yml on.push.paths uses glob forms this test's matcher does not "
        f"understand: {unrecognised}. Extend _glob_covers (and this shape check) "
        "rather than leaving the coverage invariant ambiguous."
    )


# --------------------------------------------------------------------------- #
# Positive control — prove the pipeline CAN report an uncovered path
# --------------------------------------------------------------------------- #
_FIXTURE_DOCKERFILE = """
FROM python:3.12-slim AS base
# no repo COPY here

FROM base AS deps
COPY requirements.txt pyproject.toml ./
COPY patches/ ./patches/

# NOT reachable from production: its uncovered COPY must NOT be reported.
FROM base AS development
COPY tests/ ./tests/

FROM python:3.12-slim AS production
# A stage copy: a CONTAINER path, not a repo path. Must never be reported.
COPY --from=deps /app/.venv /app/.venv
COPY src/ ./src/
# Uncovered outright.
COPY docs/ ./docs/
# PARTIALLY covered by 'scripts/deploy/**' -> still uncovered (the #1783 shape).
COPY scripts/ \\
     ./scripts/
"""

_FIXTURE_TRIGGERS = [
    "src/**",
    "requirements.txt",
    "pyproject.toml",
    "patches/**",
    "scripts/deploy/**",
]


def test_positive_control_pipeline_reports_uncovered_paths() -> None:
    """The extract -> match -> report pipeline, run over a Dockerfile whose answer is
    known by construction. A guard that passes because the matcher never matched
    anything is worse than no guard; this is the disproof that it can fail."""
    inputs = _image_input_paths(_FIXTURE_DOCKERFILE, root="production")

    # The `development` stage is unreachable from production, so its uncovered
    # `tests/` COPY must be absent — scoping, not luck.
    assert "tests/" not in inputs, (
        f"unreachable stage's COPY leaked into the image inputs: {sorted(inputs)}"
    )
    # A `COPY --from=` source is a container path, never a repo path.
    assert "/app/.venv" not in inputs, (
        f"a stage copy was mistaken for a repo path: {sorted(inputs)}"
    )
    assert set(inputs) == {
        "requirements.txt",
        "pyproject.toml",
        "patches/",
        "src/",
        "docs/",
        "scripts/",
    }, f"fixture image inputs parsed as {sorted(inputs)}"

    uncovered = _uncovered(inputs, _FIXTURE_TRIGGERS)
    assert set(uncovered) == {"docs/", "scripts/"}, (
        "positive control did not reproduce the known-uncovered set; the matcher or "
        f"the parser is wrong. Reported: {_render(uncovered) or '(nothing)'}"
    )
    # Provenance must point at the FIRST physical line of the instruction, including
    # across a `\` continuation (fixture line 20 — the string opens with a newline).
    assert uncovered["scripts/"] == ["production:20"], (
        f"continuation-joined COPY reported the wrong line: {uncovered['scripts/']}"
    )
    assert _FIXTURE_DOCKERFILE.splitlines()[19].strip().startswith("COPY scripts/"), (
        "fixture drifted: line 20 is no longer the continuation-joined COPY"
    )


# --------------------------------------------------------------------------- #
# Non-vacuity — pin WHAT WAS COMPUTED against the real repo
# --------------------------------------------------------------------------- #
def test_parser_discovers_the_real_production_image_inputs() -> None:
    """Assert the computation, not just its verdict.

    A parser that silently degrades to finding nothing would make the invariant below
    report "0 uncovered" forever. This pins the stage closure exactly and the image
    inputs as a floor, and requires every discovered path to actually exist in the repo
    (garbage tokens — a leaked dest, a mis-split flag — fail here).
    """
    stages = _parse_stages(DOCKERFILE.read_text())
    assert set(_reachable_stages(stages, PRODUCTION_STAGE)) == {
        "production",
        "dependencies",
        "base",
    }, (
        "unexpected stage closure for the production image: "
        f"{sorted(_reachable_stages(stages, PRODUCTION_STAGE))} "
        f"(all stages: {sorted(stages)})"
    )
    # `development` is a dev-only stage; its COPYs are not production image inputs.
    assert "development" in stages, f"parsed stages={sorted(stages)}"

    inputs = _image_input_paths(DOCKERFILE.read_text())
    known = {
        "pyproject.toml",  # production: project-root marker (#1448)
        "src/",  # production: application code
        "config/",  # production: YAML config tree
        "data/kg_cache/",  # production: KG Layer-2 caches (#1607)
        "scripts/",  # production: ops/maintenance scripts run via docker exec
        "requirements.txt",  # dependencies -> venv -> COPY --from=dependencies
        "requirements.lock",  # dependencies: the hash-pinned install source (#534)
        "patches/",  # dependencies: pip path-installs baked into the venv
    }
    missing = known - set(inputs)
    assert missing == set(), (
        f"the Dockerfile parser did not find known image inputs {sorted(missing)}; "
        f"it found {sorted(inputs)}. Fix the parser — do not relax this floor."
    )
    for src, provenance in sorted(inputs.items()):
        assert (REPO_ROOT / src.rstrip("/")).exists(), (
            f"parsed image-input path {src!r} (from Dockerfile {provenance}) does not "
            "exist in the repo — the COPY tokenizer emitted garbage"
        )


# --------------------------------------------------------------------------- #
# The invariant
# --------------------------------------------------------------------------- #
def test_every_production_image_input_is_a_deploy_trigger() -> None:
    """#1783: every repo path baked into the production image must be in on.push.paths.

    Otherwise a push that changes what the image WOULD contain never runs the build that
    would produce it, and production keeps serving the previously-baked copy until an
    unrelated commit rebuilds incidentally.
    """
    triggers = _trigger_paths()
    inputs = _image_input_paths(DOCKERFILE.read_text())
    uncovered = _uncovered(inputs, triggers)
    assert uncovered == {}, (
        "these paths are COPYed into the production image but no on.push.paths entry "
        "covers them, so changing one deploys NOTHING:\n"
        + _render(uncovered)
        + "\n\non.push.paths = "
        + repr(triggers)
        + "\nall image inputs = "
        + repr(sorted(inputs))
    )
