"""Shared plumbing for the ``deploy.yml`` guards in this directory.

WHY THIS FILE EXISTS (#1796)
============================
``deploy.yml`` cannot be certified the way the rest of the platform is:

* ``.github/**`` is NOT in ``deploy.yml``'s own ``on.push.paths``, so merging a deploy
  fix triggers no deploy;
* ``deploy.yml`` is baked into no image, so there is no container content marker to
  probe.

The only way to gain confidence in a change to the inline droplet script is therefore
to **extract the shell out of the YAML and execute it against stubs**. Seven modules in
this directory do exactly that, and before #1796 each one re-implemented the plumbing:
its own ``DEPLOY_WORKFLOW`` constant, its own ``yaml.safe_load``, its own step lookup.
The sharpest evidence was a non-obvious algorithm written twice, weeks apart — slicing
a shell function out of the inline script by scanning for its closing brace *at its own
indent level* (:func:`extract_shell_function` below). Two independent solutions to the
same indentation-matching problem is the signal that it belongs in one place.

WHAT DELIBERATELY DOES **NOT** LIVE HERE
========================================
The **stubs**. ``docker``/``git``/``curl`` stubs are per-defect: their per-ref and
per-attempt behaviour is what a given probe actually exercises, and *stub fidelity is
where these harnesses go wrong* — a stub using ``exit`` instead of ``return`` tears the
harness down where a real binary only sets a status; an in-memory attempt counter
silently never advances because the probe runs inside ``$(…)``, a subshell. Both were
real, and both were caught only because the stub sat next to the test that needed it,
with its reasoning attached. Keep them there.

Likewise the *assertions* stay per-module: the helpers here locate and execute shipped
text, they never decide whether it is correct.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEPLOY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy.yml"

#: The job every droplet-side guard addresses.
DEPLOY_JOB = "deploy"
#: The gated ssh-action step (rollout + health gate + drift check).
ROLLOUT_ID = "rollout"
#: The always-running ssh-action step that owns the prune (#1784).
CLEANUP_ID = "cleanup"

#: How a marker is matched against a (stripped) line.
Match = Literal["exact", "prefix", "contains"]


# --------------------------------------------------------------------------- #
# The shipped artifact
# --------------------------------------------------------------------------- #
def workflow_text() -> str:
    """``deploy.yml`` verbatim, as bytes-on-disk text."""
    return DEPLOY_WORKFLOW.read_text()


def load_workflow() -> dict:
    """``deploy.yml`` parsed."""
    wf: dict = yaml.safe_load(workflow_text())
    return wf


def trigger_paths() -> list[str]:
    """``on.push.paths``.

    PyYAML (YAML 1.1) parses a bare ``on:`` mapping key as the boolean ``True``, so a
    naive ``wf["on"]`` silently returns nothing and every coverage assertion over it
    goes vacuous. Both consumers of this hit that; it is handled once, here.
    """
    wf = load_workflow()
    on = wf.get("on")
    if on is None:
        on = wf.get(True)
    return list((on or {}).get("push", {}).get("paths", []) or [])


def job_steps(job: str = DEPLOY_JOB) -> list[dict]:
    return list(load_workflow()["jobs"][job]["steps"])


def step_table(job: str = DEPLOY_JOB) -> list[tuple[int, str, str, str]]:
    """The DERIVED step order: ``(index, id, name, uses/run)``.

    Printed by every lookup failure, so a wrong verdict shows its own derivation
    rather than asserting over a hand-typed table that can drift from the workflow.
    """
    table = []
    for i, step in enumerate(job_steps(job)):
        kind = step.get("uses") or ("run:" if "run" in step else "?")
        table.append((i, str(step.get("id", "")), str(step.get("name", "")), str(kind)))
    return table


def format_step_table(table: list[tuple[int, str, str, str]]) -> str:
    return "\n".join(f"  [{i}] id={id_!r} name={name!r} {kind}" for i, id_, name, kind in table)


def step_index(step_id: str, job: str = DEPLOY_JOB) -> int:
    """Position of the step with this ``id``. Addressing by id, never by position."""
    for i, step in enumerate(job_steps(job)):
        if step.get("id") == step_id:
            return i
    raise AssertionError(
        f"job {job!r} has no step with id {step_id!r}. Derived step table:\n"
        + format_step_table(step_table(job))
    )


def step_by_id(step_id: str, job: str = DEPLOY_JOB) -> dict:
    return job_steps(job)[step_index(step_id, job)]


def step_with(step_id: str, key: str, *, job: str = DEPLOY_JOB, what: str = "") -> str:
    """One ``with:`` input of a step addressed by id, as a string."""
    with_ = step_by_id(step_id, job).get("with") or {}
    assert key in with_, (
        f"step id={step_id!r} in job {job!r} carries no `{key}:`{f' — {what}' if what else ''}. "
        f"Derived `with` keys: {sorted(with_)}\nDerived step table:\n"
        + format_step_table(step_table(job))
    )
    return str(with_[key])


def ssh_script(step_id: str = ROLLOUT_ID, *, job: str = DEPLOY_JOB) -> str:
    """The ssh-action ``with.script`` of the step with this id — the droplet script."""
    return step_with(step_id, "script", job=job, what="it is not an ssh-action step")


def run_script(step_id: str, *, job: str = DEPLOY_JOB) -> str:
    """The ``run:`` body of the step with this id."""
    step = step_by_id(step_id, job)
    assert "run" in step, (
        f"step id={step_id!r} in job {job!r} has no `run:`. Derived step table:\n"
        + format_step_table(step_table(job))
    )
    return str(step["run"])


# --------------------------------------------------------------------------- #
# Slicing shell out of the inline script
# --------------------------------------------------------------------------- #
def _matches(line: str, marker: str, how: Match) -> bool:
    stripped = line.strip()
    if how == "exact":
        return stripped == marker
    if how == "prefix":
        return stripped.startswith(marker)
    return marker in line


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def line_index(lines: list[str], marker: str, *, how: Match = "exact", what: str = "") -> int:
    """Index of the first line matching ``marker``, or an assertion naming what is missing."""
    idx = next((i for i, ln in enumerate(lines) if _matches(ln, marker, how)), None)
    assert idx is not None, (
        f"{what or 'marker not found'} — expected a line {how} {marker!r} in the script"
    )
    return idx


def extract_block(
    script: str,
    *,
    start: str,
    end: str,
    start_match: Match = "exact",
    end_match: Match = "exact",
    anchor: str | None = None,
    anchor_match: Match = "exact",
    own_indent: bool = True,
    include_end: bool = True,
    what: str = "block",
) -> str:
    """Slice a line range out of an inline script, verbatim, and dedent it.

    ``start``/``end`` are markers matched per-line. When ``own_indent`` is set the end
    marker must additionally sit at the indent of the *anchor* line (``anchor``, or the
    start line when no anchor is given) — this is what makes the extraction survive a
    nested ``}``/``fi`` inside the block. ``anchor`` also moves where the end search
    begins, for blocks whose closer belongs to a construct opened partway in.

    The returned text is dedented by the anchor's indent so it can be executed as-is.
    """
    lines = script.splitlines()
    start_idx = line_index(lines, start, how=start_match, what=f"{what}: no start marker")
    if anchor is None:
        anchor_idx = start_idx
    else:
        anchor_idx = line_index(lines, anchor, how=anchor_match, what=f"{what}: no anchor")
    indent = _indent(lines[anchor_idx])
    for j in range(anchor_idx + 1, len(lines)):
        if not _matches(lines[j], end, end_match):
            continue
        if own_indent and _indent(lines[j]) != indent:
            continue
        stop = j + 1 if include_end else j
        return "\n".join(ln[indent:] for ln in lines[start_idx:stop])
    raise AssertionError(
        f"{what}: no end marker {end!r}"
        + (f" at the anchor's indent ({indent})" if own_indent else "")
        + " after the start marker"
    )


def extract_shell_function(script: str, name: str) -> str:
    """Slice ONE shell function out of the script, verbatim.

    Anchored on the function's own opening line and on the first line that closes it at
    that line's OWN indent, so a nested ``}`` (a brace group, a ``${...}`` expansion on
    its own line) cannot end the slice early. This exact algorithm was written twice
    independently before #1796; it lives here now.
    """
    return extract_block(
        script,
        start=f"{name}() {{",
        start_match="prefix",
        end="}",
        own_indent=True,
        what=f"{name}() in the droplet script",
    )


def index_map(script: str, markers: Mapping[str, str]) -> dict[str, int]:
    """Derive each marker's character position. A missing marker surfaces as -1."""
    return {label: script.find(needle) for label, needle in markers.items()}


def prose(script: str) -> str:
    """Comment text with the ``#`` markers and line wrapping flattened away.

    A phrase in a wrapped comment is split across lines at an arbitrary column, so a
    literal search over the raw script silently misses it — the fail-open shape that
    has bitten this repo repeatedly. Flatten first, then match literally.
    """
    words: list[str] = []
    for line in script.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            words.extend(stripped.lstrip("#").split())
    return " ".join(words)


# --------------------------------------------------------------------------- #
# Executing what was extracted
# --------------------------------------------------------------------------- #
def bash_run(
    workdir: Path,
    fragment: str,
    *,
    preamble: str = "",
    trailer: str = "",
    set_e: bool = True,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    name: str = "fragment.sh",
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    """Run an extracted fragment under ``bash`` with a supplied stub preamble and env.

    ``preamble`` is where the caller's stubs go (shell functions shadowing ``docker``,
    ``git``, …); ``trailer`` is where a reached-the-end marker goes. ``env=None`` means
    inherit the ambient environment — pass an explicit dict for a hermetic run.
    """
    runner = workdir / name
    runner.write_text(("set -e\n" if set_e else "") + preamble + fragment + trailer)
    return subprocess.run(
        ["bash", str(runner)],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def run_fragment(
    workdir: Path,
    fragment: str,
    *,
    preamble: str = "",
    trailer: str = "",
    set_e: bool = True,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    name: str = "fragment.sh",
    timeout: int = 30,
) -> tuple[int, str]:
    """:func:`bash_run`, reduced to ``(rc, stdout + stderr)``.

    The two streams are merged because a shipped script's diagnostics land on either
    one and an assertion that reads only ``stdout`` silently misses ``echo … >&2``.
    """
    proc = bash_run(
        workdir,
        fragment,
        preamble=preamble,
        trailer=trailer,
        set_e=set_e,
        env=env,
        cwd=cwd,
        name=name,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout + proc.stderr


def write_stub_bin(bin_dir: Path, stubs: Mapping[str, str]) -> Path:
    """Materialise PATH-shadowing executables. Returns the directory to prepend to PATH."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    for stub_name, body in stubs.items():
        path = bin_dir / stub_name
        path.write_text(body)
        path.chmod(0o755)
    return bin_dir


def redirect_project_dir(script: str, project_dir: Path) -> str:
    """Point the droplet script's hardcoded ``PROJECT_DIR`` at a temp tree.

    The ONLY substitution made to the shipped text: an infrastructure constant, so the
    control flow under test stays verbatim. Asserts exactly one hit, and asserts the
    extraction carries no GitHub-Actions interpolation — ``${{ … }}`` surviving into the
    fragment would mean the harness is executing something the droplet never sees.
    """
    assert "${{" not in script, (
        "the deploy `script:` gained GitHub-Actions interpolation; the extracted "
        "fragment is no longer what the droplet runs"
    )
    redirected, n = re.subn(r'PROJECT_DIR="[^"]*"', f'PROJECT_DIR="{project_dir}"', script, count=1)
    assert n == 1, f"expected exactly one PROJECT_DIR assignment to redirect, found {n}"
    return redirected


def render_step_summary(
    workdir: Path,
    script: str,
    *,
    env: Mapping[str, str],
    name: str = "summary.sh",
    summary_name: str = "summary.md",
    path: str = "/usr/bin:/bin",
    timeout: int = 30,
) -> str:
    """Execute a ``run:`` step under a chosen env and read back what it wrote.

    Renders to a real ``$GITHUB_STEP_SUMMARY`` file and returns its contents, so the
    assertion is over what the step PRODUCED rather than over the source text of its
    branches — executing a guard is not the same as exercising it.
    """
    summary_file = workdir / summary_name
    summary_file.write_text("")
    proc = bash_run(
        workdir,
        script,
        set_e=False,
        env={"PATH": path, "GITHUB_STEP_SUMMARY": str(summary_file), **env},
        name=name,
        timeout=timeout,
    )
    assert proc.returncode == 0, (
        f"the summary script must never fail its own step; rc={proc.returncode}\n"
        f"{proc.stdout}{proc.stderr}"
    )
    return summary_file.read_text()
