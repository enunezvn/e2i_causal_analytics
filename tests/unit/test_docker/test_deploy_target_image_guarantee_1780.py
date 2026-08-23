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
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from tests.unit.test_docker.conftest import (
    ROLLOUT_ID,
    bash_run,
    extract_block,
    extract_shell_function,
    load_workflow,
    run_script,
    ssh_script,
    write_stub_bin,
)

ENSURE_JOB = "ensure-main-image"
HELPER = "running_image_sha"


# --------------------------------------------------------------------------- #
# Extraction (the SHIPPED artifact, verbatim)
# --------------------------------------------------------------------------- #
# Extraction is shared (#1796): `ssh_script` addresses the rollout step BY ID, and
# `extract_shell_function` is the own-indent closing-brace scan this module and
# test_deploy_branch_ref_safety_1787.py had each written independently.


# --------------------------------------------------------------------------- #
# 1. `ensure-main-image` wiring — the "no image EVER" hole
# --------------------------------------------------------------------------- #
def test_ensure_main_image_job_exists_and_gates_the_deploy() -> None:
    """The guarantee is worthless unless the droplet step waits for it.

    RED before the fix: deploy.yml has no such job, so `origin/main` HEAD could be
    (and on 2026-08-21 was) imageless at the moment the droplet resolved it.
    """
    wf = load_workflow()
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
    wf = load_workflow()
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
    """Run a slice of the SHIPPED deploy script against stubbed docker/git.

    Returns STDOUT only (stripped), deliberately: these cases read a VALUE the helper
    echoed, and folding stderr in would let a diagnostic satisfy an equality check.
    """
    stub_bin = write_stub_bin(tmp_path / "bin", {"docker": _DOCKER_STUB, "git": _GIT_STUB})
    proc = bash_run(tmp_path, body, env={"PATH": f"{stub_bin}:/usr/bin:/bin", **env}, name="run.sh")
    return proc.returncode, proc.stdout.strip()


def _run_helper(tmp_path: Path, **env: str) -> tuple[int, str]:
    """Execute the SHIPPED running_image_sha() against stubbed docker/git."""
    helper = extract_shell_function(ssh_script(ROLLOUT_ID), HELPER)
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
    """The SHIPPED lines that decide what PREV_SHA is, verbatim.

    Marker-delimited rather than brace-delimited: this is a run of statements, not a
    function, so the closer is the echo that announces the resolved value.
    """
    return extract_block(
        script,
        start="CHECKOUT_SHA=",
        start_match="prefix",
        end="==> Pre-deploy SHA:",
        end_match="contains",
        own_indent=False,
        what=(
            "#1780: deploy.yml's droplet script never records the checkout sha "
            "separately, so PREV_SHA is still just `git rev-parse HEAD`"
        ),
    )


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
    script = ssh_script(ROLLOUT_ID)
    block = extract_shell_function(script, HELPER) + "\n" + _extract_anchor_block(script)
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
    script = ssh_script(ROLLOUT_ID)
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
    body = extract_shell_function(script, "rollback_to_prev")
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
    script = ssh_script(ROLLOUT_ID)
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
    script = ssh_script(ROLLOUT_ID)
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


# --------------------------------------------------------------------------- #
# 5. `ensure-main-image`'s PROBE — the decision, executed
#
# The wiring assertions above prove the job exists and is gated correctly. They say
# nothing about what it DECIDES, and the decision is the whole point: `needed` gates
# the checkout + both build steps, and `sha` is what those steps check out and tag.
# A probe that answers `false` when an image is genuinely missing reintroduces #1780
# in full, silently, and the job still reports success.
#
# Executed against stubbed `git`/`docker` rather than asserted structurally, because
# every interesting case here is a BRANCH, not a string.
#
# NOTE: deliberately NOT run through `_run_shell`, which prepends `set -e`. The probe
# ships with `set -uo pipefail` and no `-e` on purpose — its fail-soft contract is that
# a hiccup degrades to `needed=false` instead of failing the deploy. Running it under
# `set -e` would test a script we do not ship and would hide exactly that property.
# --------------------------------------------------------------------------- #
# Records what was actually fetched, and — critically — only resolves FETCH_HEAD when
# `origin main` was the thing fetched. A stub that answers `rev-parse FETCH_HEAD`
# unconditionally lets the probe fetch the WRONG ref and still look correct: codex
# found that hole, and it was real — mutating the probe to `git fetch origin
# "$TRIGGER_SHA"` left all 23 tests green, while in production it would resolve
# MAIN_SHA to the trigger sha, decide "main has not moved" every time, and reintroduce
# #1780 in full with the job still reporting success.
_PROBE_GIT_STUB = """#!/usr/bin/env bash
_args_file="${STUB_STATE:-/tmp}/fetch_args"
case "$1" in
  fetch)
    shift
    printf '%s\\n' "$*" > "$_args_file"
    exit "${STUB_FETCH_RC:-0}"
    ;;
  rev-parse)
    if [ "$2" = "FETCH_HEAD" ]; then
      # git errors on an unpopulated FETCH_HEAD rather than echoing a stale value.
      # EXACT match, not a substring: `--quiet upstream origin main` contains
      # "origin main" but fetches from a remote that does not exist on the runner,
      # so the probe would take the fail-soft path and skip the guarantee entirely.
      [ "$(cat "$_args_file" 2>/dev/null)" = "--quiet origin main" ] || exit 1
      printf '%s\\n' "${STUB_MAIN_SHA:-}"
      exit 0
    fi
    ;;
esac
exit 0
"""

# Matches on the FULL reference, exactly as `docker manifest inspect` resolves one.
# An earlier version compared only the last path segment, which ignored registry,
# owner and TAG — so a probe querying the right repo at the WRONG sha would still have
# read as present. Registry/owner/tag are the whole question here.
_PROBE_DOCKER_STUB = """#!/usr/bin/env bash
if [ "$1" = "manifest" ] && [ "$2" = "inspect" ]; then
  case " ${STUB_PRESENT:-} " in *" $3 "*) exit 0 ;; *) exit 1 ;; esac
fi
exit 0
"""


def _refs(sha: str, *repos: str) -> str:
    """Full image refs, in the same shape the probe builds them."""
    return " ".join(f"ghcr.io/owner/{r}:{sha}" for r in repos)


def _run_probe(tmp_path: Path, **env: str) -> tuple[int, str, dict[str, str], str]:
    """Execute the shipped probe.

    Returns (rc, stdout, parsed $GITHUB_OUTPUT, the args `git fetch` was called with).
    """
    stub_bin = write_stub_bin(
        tmp_path / "bin", {"docker": _PROBE_DOCKER_STUB, "git": _PROBE_GIT_STUB}
    )

    state = tmp_path / "state"
    state.mkdir()

    out_file = tmp_path / "gh_output"
    out_file.touch()

    # set_e=False: the probe ships `set -uo pipefail` and NO `-e`, deliberately (see the
    # section note above). Prepending `set -e` would exercise a script we do not ship.
    proc = bash_run(
        tmp_path,
        run_script("probe", job=ENSURE_JOB),
        set_e=False,
        env={
            "PATH": f"{stub_bin}:/usr/bin:/bin",
            "GITHUB_OUTPUT": str(out_file),
            "STUB_STATE": str(state),
            "REGISTRY": "ghcr.io",
            "IMAGE_NAME": "owner/e2i-api",
            "IMAGE_NAME_FRONTEND": "owner/e2i-frontend",
            **env,
        },
        name="probe.sh",
    )
    outputs: dict[str, str] = {}
    for line in out_file.read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            outputs[k] = v  # last write wins, as GitHub does
    args_file = state / "fetch_args"
    fetch_args = args_file.read_text().strip() if args_file.exists() else ""
    return proc.returncode, proc.stdout, outputs, fetch_args


BOTH = "e2i-api e2i-frontend"


@pytest.mark.parametrize(
    ("trigger", "main_sha", "present", "expect_needed", "why"),
    [
        (
            SHA_A,
            SHA_A,
            "",
            "false",
            "main has not moved: this run just pushed both images, so probing GHCR "
            "would be a wasted round-trip against a tag we know exists",
        ),
        (
            SHA_A,
            SHA_B,
            BOTH,
            "false",
            "main moved but that sha is already fully built (its own run beat us) — "
            "rebuilding would burn a runner for nothing",
        ),
        (
            SHA_A,
            SHA_B,
            "e2i-frontend",
            "true",
            "api missing: the droplet's image_exists() requires BOTH, so guaranteeing "
            "only the frontend leaves the walk rejecting this sha anyway",
        ),
        (
            SHA_A,
            SHA_B,
            "e2i-api",
            "true",
            "frontend missing — same reason, mirrored",
        ),
        (
            SHA_A,
            SHA_B,
            "",
            "true",
            "the actual #1780 case: a path-filtered merge nobody ever built",
        ),
    ],
)
def test_probe_decides_needed_from_what_is_actually_published(
    tmp_path: Path,
    trigger: str,
    main_sha: str,
    present: str,
    expect_needed: str,
    why: str,
) -> None:
    """`needed` must track real manifest presence, per repo.

    RED before the fix: there is no ensure-main-image job at all.
    """
    rc, _stdout, outputs, _fetch = _run_probe(
        tmp_path,
        TRIGGER_SHA=trigger,
        STUB_MAIN_SHA=main_sha,
        STUB_PRESENT=_refs(main_sha, *present.split()),
    )
    assert rc == 0, "the probe must never fail the run — it is a best-effort guarantee"
    assert outputs.get("needed") == expect_needed, why


def test_probe_publishes_the_resolved_main_sha_not_the_trigger(tmp_path: Path) -> None:
    """`sha` is consumed by the checkout + both build steps.

    If it echoed TRIGGER_SHA the job would cheerfully rebuild and re-tag the sha that
    was ALREADY built, leave origin/main's sha still missing, and report success — the
    #1780 failure with an extra build bolted on.
    """
    _rc, _stdout, outputs, _fetch = _run_probe(
        tmp_path, TRIGGER_SHA=SHA_A, STUB_MAIN_SHA=SHA_B, STUB_PRESENT=""
    )
    assert outputs.get("sha") == SHA_B, (
        "the probe must publish the sha it RESOLVED, not the one this run was triggered by"
    )


def test_probe_is_fail_soft_when_it_cannot_resolve_main(tmp_path: Path) -> None:
    """A broken `git fetch` must degrade to the pre-#1780 behaviour, not block a deploy.

    Blocking production because a probe hiccuped trades a slow deploy for NO deploy.
    The step must still exit 0, still publish a usable `sha`, and say why in the log so
    the give-up is visible rather than silent.
    """
    rc, stdout, outputs, _fetch = _run_probe(
        tmp_path, TRIGGER_SHA=SHA_A, STUB_MAIN_SHA=SHA_B, STUB_FETCH_RC="1"
    )
    assert rc == 0, "a fetch failure must not fail the job"
    assert outputs.get("needed") == "false", "give up, do not guess that a build is needed"
    assert outputs.get("sha") == SHA_A, (
        "downstream steps still read `sha`; it must fall back to the trigger sha rather "
        "than the empty string a failed `rev-parse` would leave"
    )
    assert "WARN" in stdout, "a silent give-up is indistinguishable from a no-op probe"


def test_probe_names_every_missing_image_in_the_log(tmp_path: Path) -> None:
    """The operator-facing half: when this fires, which image was missing must be
    readable from the log. #1780 cost hours precisely because `manifest unknown`
    scrolled past with no indication of which sha or repo it referred to."""
    _rc, stdout, _outputs, _fetch = _run_probe(
        tmp_path,
        TRIGGER_SHA=SHA_A,
        STUB_MAIN_SHA=SHA_B,
        STUB_PRESENT=_refs(SHA_B, "e2i-api"),
    )
    assert f"MISSING: ghcr.io/owner/e2i-frontend:{SHA_B}" in stdout
    assert f"present: ghcr.io/owner/e2i-api:{SHA_B}" in stdout


def test_probe_resolves_origin_main_and_not_this_run_s_own_ref(tmp_path: Path) -> None:
    """The job exists to guarantee an image for **origin/main HEAD**.

    Everything downstream is keyed to whatever `git fetch` put in FETCH_HEAD, so the ref
    being fetched IS the guarantee. Fetching this run's own sha instead would make
    MAIN_SHA == TRIGGER_SHA on every run, take the "main has not moved" early return
    every time, and leave `needed=false` permanently — #1780 restored in full, with the
    job still green.

    Found by codex on the first revision of this module: the original stub answered
    `rev-parse FETCH_HEAD` unconditionally, and mutating the probe to fetch
    `"$TRIGGER_SHA"` left all 23 tests passing. Asserted directly rather than left to
    emerge from the stub's behaviour.
    """
    _rc, _stdout, outputs, fetch_args = _run_probe(
        tmp_path, TRIGGER_SHA=SHA_A, STUB_MAIN_SHA=SHA_B, STUB_PRESENT=""
    )
    assert fetch_args == "--quiet origin main", (
        "the probe must fetch exactly `origin main`; it fetched: "
        f"{fetch_args!r}. Substring matching is not enough here — codex iter-2 showed "
        "`--quiet upstream origin main` contains 'origin main', passes a substring "
        "check, and fetches a remote that does not exist on the runner, so the probe "
        "silently takes the fail-soft path and skips the #1780 guarantee. If you change "
        "the fetch deliberately, update this string deliberately."
    )
    assert outputs.get("sha") == SHA_B, "and must publish what that fetch resolved to"
