"""#1780 — the sha the droplet deploys must always have a published GHCR image.

ROOT CAUSE (measured from deploy run 32507847667, 2026-08-21):

``deploy.yml`` builds images for the **triggering** sha but the droplet script
resolves ``origin/main`` at **run time**. PR #1778 merged as ``32259eb57`` touching
only ``scripts/`` and ``tests/`` — neither is in ``on.push.paths``, so the workflow
never fired and **no image was ever built for that sha**. The in-flight deploy for
PR #1777 (``b1aba1c8f``) then resolved ``origin/main`` = ``32259eb57`` and found no
manifest, so it fell back to a ~26-min droplet local build that blew the 30m SSH
``command_timeout``. Production ended up correct but running an image that exists
only on the box, and the run was reported FAILED.

Two independent holes, both pinned here:

1. **No image EVER.** #1431's ancestor-walk was designed for "no image *yet*" — a
   newer commit lands mid-deploy while its own build is queued, so the walk skips it
   and its own run deploys it later. A path-filtered commit has no queued run at all,
   so the walk waits for a build that will never come. Fixed by ``ensure-main-image``:
   the already-authenticated CI runner builds and pushes images for ``origin/main``
   HEAD, before the droplet step, whenever that sha has no manifest.

2. **The downgrade floor was anchored to the wrong thing.** It compared the walk's
   target against ``PREV_SHA`` = ``git rev-parse HEAD`` of the droplet checkout. But
   PROD == DEV == the same box, so ``$PROJECT_DIR`` is also a human working copy and a
   plain ``git pull`` there moves ``PREV_SHA`` without deploying anything. On
   2026-08-21 the checkout had been moved out of band to ``32259eb`` while the
   containers were still on ``9444237ae``, so the floor read ``b1aba1c`` — a CHILD of
   the running sha, with published images — as a "downgrade" and refused it. Fixed by
   deriving the running sha from the ``e2i_api`` container's compose-pinned image tag.

The selection/floor CONTROL FLOW is exercised end-to-end by the network-free
``scripts/test_deploy_sha_selection.sh``. This module pins the parts that live in the
workflow file itself: the new job's wiring, the running-sha helper's parsing and
validation (executed for real against stubbed ``docker``/``git``), and the fallback
warning that sent a human chasing a GHCR auth failure that had not happened.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEPLOY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy.yml"

ENSURE_JOB = "ensure-main-image"
HELPER = "running_image_sha"


# --------------------------------------------------------------------------- #
# Extraction (the SHIPPED artifact, verbatim)
# --------------------------------------------------------------------------- #
def _load_workflow() -> dict:
    wf: dict = yaml.safe_load(DEPLOY_WORKFLOW.read_text())
    return wf


def _extract_deploy_script() -> str:
    wf = _load_workflow()
    for step in wf["jobs"]["deploy"]["steps"]:
        with_ = step.get("with") or {}
        if "script" in with_:
            return str(with_["script"])
    raise AssertionError("deploy.yml has no ssh-action step carrying a `script:`")


def _extract_helper(script: str, name: str) -> str:
    """Slice ONE shell function out of the deploy script, verbatim.

    Anchored on the function's own opening line and the first line that closes it at
    column 0 of the (dedented) script body, so the extracted text is the shipped
    implementation rather than a copy that can drift.
    """
    lines = script.splitlines()
    start = next(
        (i for i, ln in enumerate(lines) if ln.strip().startswith(f"{name}() {{")),
        None,
    )
    assert start is not None, f"deploy.yml's droplet script defines no {name}() helper"
    indent = len(lines[start]) - len(lines[start].lstrip())
    for j in range(start + 1, len(lines)):
        if lines[j].strip() == "}" and (len(lines[j]) - len(lines[j].lstrip())) == indent:
            return "\n".join(ln[indent:] for ln in lines[start : j + 1])
    raise AssertionError(f"{name}() in deploy.yml has no closing brace at its own indent")


# --------------------------------------------------------------------------- #
# 1. `ensure-main-image` wiring — the "no image EVER" hole
# --------------------------------------------------------------------------- #
def test_ensure_main_image_job_exists_and_gates_the_deploy() -> None:
    """The guarantee is worthless unless the droplet step waits for it.

    RED before the fix: deploy.yml has no such job, so `origin/main` HEAD could be
    (and on 2026-08-21 was) imageless at the moment the droplet resolved it.
    """
    wf = _load_workflow()
    jobs = wf["jobs"]
    assert ENSURE_JOB in jobs, (
        f"deploy.yml defines no `{ENSURE_JOB}` job — nothing guarantees that the sha "
        "the droplet resolves at run time has a published image (#1780)"
    )
    needs = jobs["deploy"].get("needs") or []
    needs = [needs] if isinstance(needs, str) else list(needs)
    assert ENSURE_JOB in needs, (
        f"the deploy job must `needs: {ENSURE_JOB}` — otherwise the SSH step can start "
        f"before the image it depends on has been pushed (parsed needs: {needs})"
    )


def test_ensure_main_image_pushes_both_app_images() -> None:
    """`image_exists()` on the droplet requires BOTH e2i-api and e2i-frontend at the
    sha tag, so guaranteeing only one of them would not satisfy the walk."""
    wf = _load_workflow()
    job = wf["jobs"][ENSURE_JOB]
    text = yaml.safe_dump(job)
    assert "e2i-api" in text or "IMAGE_NAME" in text, f"{ENSURE_JOB} never references the api image"
    assert "e2i-frontend" in text or "IMAGE_NAME_FRONTEND" in text, (
        f"{ENSURE_JOB} never references the frontend image — the droplet's "
        "image_exists() requires BOTH manifests, so an api-only guarantee is vacuous"
    )
    assert (job.get("permissions") or {}).get("packages") == "write", (
        f"{ENSURE_JOB} needs `packages: write` to push the missing image"
    )


# --------------------------------------------------------------------------- #
# 2. The floor's anchor — executed, not just grepped
# --------------------------------------------------------------------------- #
_DOCKER_STUB = """#!/usr/bin/env bash
printf '%s\\n' "${STUB_IMAGE:-}"
exit "${STUB_DOCKER_RC:-0}"
"""

# `git cat-file -e <sha>^{commit}` succeeds only for shas listed in $STUB_COMMITS;
# `git rev-parse HEAD` reports the droplet CHECKOUT head.
_GIT_STUB = """#!/usr/bin/env bash
if [ "$1" = "cat-file" ]; then
  _want=${3%%^*}
  case " ${STUB_COMMITS:-} " in *" $_want "*) exit 0 ;; *) exit 1 ;; esac
fi
if [ "$1" = "rev-parse" ]; then
  printf '%s\\n' "${STUB_CHECKOUT:-}"
fi
exit 0
"""


def _run_shell(tmp_path: Path, body: str, **env: str) -> tuple[int, str]:
    """Run a slice of the SHIPPED deploy script against stubbed docker/git."""
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    for name, stub in (("docker", _DOCKER_STUB), ("git", _GIT_STUB)):
        p = stub_bin / name
        p.write_text(stub)
        p.chmod(0o755)

    runner = tmp_path / "run.sh"
    runner.write_text("set -e\n" + body)
    proc = subprocess.run(
        ["bash", str(runner)],
        env={"PATH": f"{stub_bin}:/usr/bin:/bin", **env},
        capture_output=True,
        text=True,
        timeout=30,
    )
    return proc.returncode, proc.stdout.strip()


def _run_helper(tmp_path: Path, **env: str) -> tuple[int, str]:
    """Execute the SHIPPED running_image_sha() against stubbed docker/git."""
    helper = _extract_helper(_extract_deploy_script(), HELPER)
    return _run_shell(tmp_path, helper + f"\n{HELPER}\n", **env)


SHA_A = "a" * 40
SHA_B = "b" * 40


@pytest.mark.parametrize(
    ("image", "commits", "expect_rc", "expect_out", "why"),
    [
        (
            f"ghcr.io/owner/e2i-api:{SHA_A}",
            SHA_A,
            0,
            SHA_A,
            "the happy path: a compose-pinned sha tag that is a commit here",
        ),
        (
            "ghcr.io/owner/e2i-api:latest",
            SHA_A,
            1,
            "",
            "`latest` names no commit — must degrade, not be compared against",
        ),
        (
            "e2i-causal-analytics-api",
            SHA_A,
            1,
            "",
            "a locally-built image has no sha tag at all (the #1780 end state)",
        ),
        (
            f"ghcr.io/owner/e2i-api:{SHA_A[:7]}",
            SHA_A,
            1,
            "",
            "a short tag is ambiguous — merge-base must never run on a guess",
        ),
        (
            f"ghcr.io/owner/e2i-api:{'z' * 40}",
            SHA_A,
            1,
            "",
            "40 chars but not hex",
        ),
        (
            f"ghcr.io/owner/e2i-api:{SHA_B}",
            SHA_A,
            1,
            "",
            "a well-formed sha that is NOT a commit in this checkout (rewritten "
            "history) — `git merge-base --is-ancestor` would error on it",
        ),
        ("", SHA_A, 1, "", "no api container running (first deploy)"),
    ],
)
def test_running_image_sha_only_accepts_a_resolvable_commit(
    tmp_path: Path,
    image: str,
    commits: str,
    expect_rc: int,
    expect_out: str,
    why: str,
) -> None:
    """The floor may only be re-anchored on a sha it can actually reason about.

    Every rejection here degrades the floor to the pre-#1780 checkout-HEAD behaviour,
    which is never worse than before. RED before the fix: no such helper exists.
    """
    rc, out = _run_helper(tmp_path, STUB_IMAGE=image, STUB_COMMITS=commits)
    assert (rc, out) == (expect_rc, expect_out), why


def test_running_image_sha_survives_a_docker_failure(tmp_path: Path) -> None:
    """A dead/absent docker must not abort the deploy under `set -e` — the helper
    reports "unknown" and the floor degrades."""
    rc, out = _run_helper(tmp_path, STUB_IMAGE="", STUB_DOCKER_RC="1", STUB_COMMITS=SHA_A)
    assert (rc, out) == (1, "")


def _extract_anchor_block(script: str) -> str:
    """The SHIPPED lines that decide what PREV_SHA is, verbatim."""
    lines = script.splitlines()
    start = next(
        (i for i, ln in enumerate(lines) if ln.strip().startswith("CHECKOUT_SHA=")),
        None,
    )
    assert start is not None, (
        "deploy.yml's droplet script never records the checkout sha separately, so "
        "PREV_SHA is still just `git rev-parse HEAD` (#1780)"
    )
    end = next(i for i, ln in enumerate(lines) if "==> Pre-deploy SHA:" in ln)
    ind = len(lines[start]) - len(lines[start].lstrip())
    return "\n".join(ln[ind:] for ln in lines[start : end + 1])


@pytest.mark.parametrize(
    ("image", "commits", "expected", "why"),
    [
        (
            f"ghcr.io/owner/e2i-api:{SHA_B}",
            f"{SHA_A} {SHA_B}",
            SHA_B,
            "the running container's tag wins over the checkout head",
        ),
        (
            "ghcr.io/owner/e2i-api:latest",
            SHA_A,
            SHA_A,
            "unusable tag -> degrade to the checkout head (the pre-#1780 value)",
        ),
    ],
)
def test_prev_sha_is_anchored_on_the_running_image(
    tmp_path: Path, image: str, commits: str, expected: str, why: str
) -> None:
    """Executed, not grepped: run the shipped anchor block and read PREV_SHA back.

    PREV_SHA is the single anchor the floor, rollback_to_prev (both its `git reset`
    target and its IMAGE_TAG), the baked-image-input diff and the no-delta re-run
    detector all consume, so getting it from the running container fixes all of them
    at once. RED before the fix: the block does not exist, and PREV_SHA was
    unconditionally `git rev-parse HEAD`.
    """
    script = _extract_deploy_script()
    block = _extract_helper(script, HELPER) + "\n" + _extract_anchor_block(script)
    rc, out = _run_shell(
        tmp_path,
        block + '\nprintf "PREV=%s\\n" "$PREV_SHA"\n',
        STUB_IMAGE=image,
        STUB_COMMITS=commits,
        STUB_CHECKOUT=SHA_A,
    )
    assert rc == 0, out
    assert f"PREV={expected}" in out, why + f"\ngot:\n{out}"


def test_rollback_and_diffs_inherit_the_running_anchor() -> None:
    """The anchor must be resolved BEFORE anything consumes it, and the consumers must
    keep reading it rather than re-deriving the checkout head.

    A rollback to the checkout head is the dangerous half: on 2026-08-21 that would have
    reset production to 32259eb — a tree the containers were never on, and one with no
    GHCR image, so `rollback_to_prev`'s pull would fail and local-build on an already
    stressed box (the #528-B OOM path the pulled tier exists to avoid).
    """
    script = _extract_deploy_script()
    assert HELPER in script, f"PREV_SHA must be derived from {HELPER}()"
    lines = script.splitlines()
    anchor_at = next(i for i, ln in enumerate(lines) if ln.strip() == 'PREV_SHA="$RUNNING_SHA"')
    first_use = next(
        i for i, ln in enumerate(lines) if i > anchor_at - 1 and "merge-base --is-ancestor" in ln
    )
    assert anchor_at < first_use, "the floor must run after the anchor is resolved"

    calls = [i for i, ln in enumerate(lines) if re.match(r"\s*rollback_to_prev\s", ln)]
    assert calls, "deploy.yml lost its rollback_to_prev call sites"
    assert anchor_at < min(calls), (
        "PREV_SHA must be re-anchored before any rollback can fire, or a failed deploy "
        "rolls production back to a tree it was never running"
    )
    body = _extract_helper(script, "rollback_to_prev")
    assert 'git reset --hard "$PREV_SHA"' in body, (
        "rollback_to_prev must reset to the shared PREV_SHA anchor"
    )
    assert 'export IMAGE_TAG="$PREV_SHA"' in body, (
        "the rollback must pull the image tag of what was RUNNING — that image exists "
        "by construction, whereas the checkout head's may never have been built"
    )
    assert 'git diff --name-only "$PREV_SHA" "$NEW_SHA"' in script, (
        "the baked-image-input diff must also measure against what is running"
    )


# --------------------------------------------------------------------------- #
# 3. The fallback warning — it named two causes, both false
# --------------------------------------------------------------------------- #
def test_fallback_warning_states_the_actual_reason() -> None:
    """On 2026-08-21 the script printed, one line after naming the built ancestor it
    had just found and rejected:

        no pre-built GHCR image confirmed for any recent origin/main ancestor
        (GHCR auth failed or none built in the 30-commit window)

    One WAS confirmed, auth was fine, and the real cause — the downgrade floor — went
    unmentioned. A human lost real time chasing a GHCR auth failure that never
    happened. The message must carry the reason the code actually took, and the three
    causes must be distinguishable.
    """
    script = _extract_deploy_script()
    assert "GHCR auth failed or none built in the 30-commit window" not in script, (
        "the fallback warning still offers two hardcoded causes regardless of which "
        "branch was taken (#1780)"
    )
    fallback = next(
        (
            ln
            for ln in script.splitlines()
            if "Falling back to origin/main" in ln or "blind reset to origin/main" in ln
        ),
        None,
    )
    assert fallback is not None, "deploy.yml lost its blind-fallback warning entirely"
    assert re.search(r"\$\{?FALLBACK_REASON\b", fallback), (
        "the fallback warning must interpolate the reason the code recorded, not "
        "restate a fixed guess: " + fallback.strip()
    )

    # All three reachable causes must be distinguishable in the emitted text.
    for cause in ("auth", "window", "downgrade"):
        assert re.search(rf"FALLBACK_REASON=.*{cause}", script, re.IGNORECASE), (
            f"no FALLBACK_REASON assignment mentions the {cause!r} cause — the three "
            "reasons (GHCR auth unavailable / nothing built in the walk window / the "
            "only built ancestor was refused as a downgrade) must be told apart"
        )


def test_fallback_warning_names_the_local_build_cost() -> None:
    """The blind fallback is the branch that costs ~26 min of droplet build and can
    exceed the 30m SSH command_timeout. Say so where the operator reads it."""
    script = _extract_deploy_script()
    fallback = next(
        (
            ln
            for ln in script.splitlines()
            if "Falling back to origin/main" in ln or "blind reset to origin/main" in ln
        ),
        None,
    )
    assert fallback is not None
    assert "LOCAL-BUILD" in fallback.upper(), (
        "the warning must say the pull may take the local-build path: " + fallback.strip()
    )
