#!/bin/bash
# =============================================================================
# #1431 deploy SHA-selection + downgrade-floor regression harness
# =============================================================================
# Self-contained, network/LLM-free regression test for the deploy-time logic in
# .github/workflows/deploy.yml that decides WHICH main-ancestor SHA to deploy.
#
# Context (#1431): production deploys are serialized (concurrency: deploy-production,
# cancel-in-progress:false). A newer commit can land on main mid-deploy while its
# build-and-push is still queued, so no GHCR image exists for it yet. Blindly
# `reset --hard origin/main` would then fail the pull and fall back to a LOCAL
# droplet build — the #528-B React/esbuild OOM the prebuilt-image path avoids. The
# fix walks origin/main newest->oldest and deploys the newest ANCESTOR whose images
# are already published, with a FLOOR that refuses a strict downgrade below the
# running SHA.
#
# This harness proves two things WITHOUT touching GHCR (the image probe is stubbed):
#   Part 1 — select_built_sha (copied VERBATIM from deploy.yml) picks the newest
#            candidate whose images "exist", respects the walk bound, and returns
#            empty when none qualify.
#   Part 2 — the downgrade FLOOR (mirrored from deploy.yml) exercised against a REAL
#            throwaway git repo, so `git merge-base --is-ancestor` runs for real:
#            (a) PREV built + newer unbuilt        -> deploys PREV (no local build)
#            (b) PREV image-less + only OLDER built -> floor forces fallback (NO downgrade)
#            (c) normal HEAD built                  -> deploys HEAD
#   Part 3 — #1780: the floor must be anchored to the RUNNING image, not to the
#            droplet CHECKOUT head. PROD == DEV == the same box, so $PROJECT_DIR is
#            also a human working copy and a `git pull` there moves the checkout with
#            no deploy behind it. Anchored wrongly the floor both refuses valid
#            upgrades (the 2026-08-21 incident, run 32507847667) and permits real
#            downgrades. deploy.yml's running_image_sha() supplies the anchor and its
#            parsing/validation is covered by
#            tests/unit/test_docker/test_deploy_target_image_guarantee_1780.py.
#
# Exit 0 = all cases pass. Exit 1 = a regression.
#
# Usage: scripts/test_deploy_sha_selection.sh
# =============================================================================

set -uo pipefail

PASS=0
FAIL=0
check() { # $1=name $2=expected $3=got
  if [ "$2" = "$3" ]; then
    printf 'PASS  %-34s -> %s\n' "$1" "${3:-<empty>}"
    PASS=$((PASS + 1))
  else
    printf 'FAIL  %-34s expected [%s] got [%s]\n' "$1" "$2" "$3"
    FAIL=$((FAIL + 1))
  fi
}

# --- select_built_sha: copied VERBATIM from .github/workflows/deploy.yml -------
# (keep in sync if the deploy helper changes)
select_built_sha() {
  while read -r _cand; do
    [ -n "$_cand" ] || continue
    if image_exists "$_cand"; then
      echo "$_cand"
      return 0
    fi
  done
  return 1
}

# --- stubbed probe: "built" iff the SHA/label is in BUILT_SET ------------------
# Stands in for image_exists()/manifest_present() so no GHCR call is made. Works on
# logical labels (Part 1) or real SHAs (Part 2) — it is just membership matching.
image_exists() { case " $BUILT_SET " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

echo "===== Part 1: select_built_sha (pure walk, stubbed probe, labels) ====="
run1() { # $1=name $2=candidates(newest-first,\n) $3=built-set $4=expected
  BUILT_SET="$3"
  check "$1" "$4" "$(printf '%s\n' "$2" | head -n 30 | select_built_sha || true)"
}
run1 "race_newest_unbuilt"   "$(printf 'B\nA\nZ')"              "A Z"   "A"
run1 "newest_built"          "$(printf 'C\nB\nA')"             "C B A" "C"
run1 "two_ahead_unbuilt"     "$(printf 'D\nC\nB\nA')"          "B A"   "B"
run1 "partial_push_skip"     "$(printf 'E\nD')"                "D"     "D"
run1 "none_built"            "$(printf 'X\nY\nZ')"             ""      ""
run1 "prev_reachable"        "$(printf 'N4\nN3\nN2\nN1\nP')"   "P"     "P"
big=$(for i in $(seq 1 31); do echo "c$i"; done)  # c1..c31 newest-first
run1 "beyond_30_window"      "$big"                            "c31"   ""
run1 "edge_of_30_window"     "$big"                            "c30"   "c30"

echo
echo "===== Part 2: downgrade FLOOR on a REAL git repo (merge-base --is-ancestor) ====="
REPO="$(mktemp -d)"
trap 'rm -rf "$REPO"' EXIT
cd "$REPO" || exit 1
git init -q
git config user.email t@t.t; git config user.name t; git config commit.gpgsign false
for n in 1 2 3 4 5; do
  git commit -q --allow-empty -m "c$n"
  eval "C$n=$(git rev-parse HEAD)"   # C1=oldest ... C5=newest(HEAD)
done

# resolve_target: EXACT logic from deploy.yml (walk + downgrade floor). Uses the
# ambient $BUILT_SET (via the image_exists stub), $PREV_SHA (the droplet CHECKOUT
# head) and $RUNNING_SHA (#1780: what the api container is actually running, empty
# when deploy.yml's running_image_sha() could not establish it).
resolve_target() {
  _t=$(git rev-list --topo-order HEAD | head -n 30 | select_built_sha || true)
  _floor="${RUNNING_SHA:-$PREV_SHA}"
  if [ -n "$_t" ] && [ "$_t" != "$_floor" ] \
     && git merge-base --is-ancestor "$_t" "$_floor"; then
    _t=""   # strict downgrade -> refuse, fall back to blind origin/main
  fi
  echo "$_t"
}

# Cases (a)-(f) predate #1780 and exercise the floor with NO running-image signal
# available, so the floor falls back to the checkout HEAD exactly as it always did.
RUNNING_SHA=""
# (a) PREV=c3 built, newer c4/c5 unbuilt -> returns PREV (c3); no local build.
PREV_SHA="$C3"; BUILT_SET="$C3 $C1"
check "a_prev_built_newer_unbuilt" "$C3" "$(resolve_target)"
# (b) PREV=c4 image-less, only OLDER c2 built -> floor -> empty (fallback, NO downgrade).
PREV_SHA="$C4"; BUILT_SET="$C2"
check "b_floor_refuses_downgrade" "" "$(resolve_target)"
# (c) normal: HEAD c5 built -> returns HEAD (c5).
PREV_SHA="$C3"; BUILT_SET="$C5 $C3"
check "c_head_built" "$C5" "$(resolve_target)"
# (d) same-SHA redeploy allowed (PREV==target, not a downgrade).
PREV_SHA="$C5"; BUILT_SET="$C5"
check "d_same_sha_redeploy" "$C5" "$(resolve_target)"
# (e) nothing built -> empty (fallback).
PREV_SHA="$C2"; BUILT_SET=""
check "e_none_built_fallback" "" "$(resolve_target)"
# (f) newer built ancestor is NOT floored (PREV is its ancestor).
PREV_SHA="$C2"; BUILT_SET="$C4 $C2"
check "f_newer_not_floored" "$C4" "$(resolve_target)"

echo
echo "===== Part 3: #1780 — the floor must be anchored to the RUNNING image ====="
# PROD == DEV == the same droplet: $PROJECT_DIR is ALSO a human working copy, so a
# `git pull` there moves the checkout HEAD (PREV_SHA) without deploying anything.
# The floor's stated purpose is "never deploy code strictly OLDER than what is
# RUNNING", and the checkout HEAD is not a sound proxy for that — it can sit ahead
# of the running containers (blocking a valid upgrade) or behind them (waving a
# real downgrade through). Both directions are exercised below.
#
# (g) is the 2026-08-21 incident (deploy run 32507847667) in miniature: the checkout
#     had been pulled to 32259eb while the containers were still on 9444237ae, so the
#     floor read b1aba1c — a CHILD of the running sha, with published images — as a
#     "downgrade", refused it, and left a ~26-min droplet local build as the only path
#     (which then blew the 30m SSH timeout).
PREV_SHA="$C5"; RUNNING_SHA="$C3"; BUILT_SET="$C4 $C3"
check "g_running_anchor_allows_upgrade" "$C4" "$(resolve_target)"
# (h) the mirror image, and the more dangerous one: the checkout is BEHIND the running
#     containers (a human `git checkout` of an older commit). Anchored to the checkout
#     HEAD the floor waves through a target that is strictly OLDER than what is running
#     — the exact silent downgrade the floor exists to prevent.
PREV_SHA="$C2"; RUNNING_SHA="$C4"; BUILT_SET="$C2"
check "h_running_anchor_blocks_downgrade" "" "$(resolve_target)"
# (i) degradation: with no usable running-image signal (first deploy, a locally-built
#     image, a tag that is not a commit here) the floor falls back to the checkout HEAD
#     — i.e. exactly the pre-#1780 behaviour, never worse.
PREV_SHA="$C4"; RUNNING_SHA=""; BUILT_SET="$C2"
check "i_floor_degrades_to_checkout" "" "$(resolve_target)"
# (j) a same-sha redeploy of the RUNNING image is not a downgrade.
PREV_SHA="$C2"; RUNNING_SHA="$C4"; BUILT_SET="$C4"
check "j_running_sha_redeploy" "$C4" "$(resolve_target)"

echo
echo "===== $PASS passed, $FAIL failed ====="
[ "$FAIL" -eq 0 ]
