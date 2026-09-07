# Runbook — deploy operations

**Scope:** the operator-facing behaviour of `.github/workflows/deploy.yml` —
which commit prod actually lands on, what refuses a deploy, how to read a
FAILED run that nevertheless converged, and how to roll back. `DEPLOYMENT.md`
describes the deploy *pipeline*; this runbook is the *procedures* around it.

**Everything here is re-derived from `.github/workflows/deploy.yml`** (1355 lines,
read 2026-09-07). Section headings carry the issue numbers so a claim can be
traced back. Nothing was inferred from a run log.

**Ground rule that governs the whole file: PROD == DEV == this box.** The
production checkout `/home/enunez/Projects/e2i_causal_analytics` is also a shared
human working copy. Almost every surprise below comes from that.

---

## 1. Which commit prod lands on — it is not always `origin/main`

The deploy does **not** blindly `git reset --hard origin/main` (#1431). Production
deploys are serialised (§7), so a newer commit can land on `main` while a deploy
is mid-flight, with its own build still queued *behind* this deploy — no GHCR
image exists for it yet. Resetting onto that sha would fail the pull and fall back
to a ~26-minute local Docker build on a memory-pressured production box (#528-B).

So the droplet walks `git rev-list --topo-order origin/main | head -n 30` and
deploys the **newest ancestor that already has BOTH published images** —
`ghcr.io/<owner>/e2i-api:<sha>` *and* `ghcr.io/<owner>/e2i-frontend:<sha>`. The
walk stops at the first hit, normally `origin/main` HEAD (one probe).
`--topo-order` so ancestry cannot be perturbed by commit-date/clock skew.

**Consequence for operators: prod can legitimately sit one or more commits behind
`origin/main`.** The skipped newer commit deploys via its *own* queued run, so
main still converges — just never through a local OOM build. The log says so:

```
==> origin/main <sha> has NO published GHCR image yet (its build is queued behind this
    deploy); deploying newest BUILT ancestor <sha> instead.
```

### The downgrade floor

A separate guard refuses any target that is a **strict ancestor of what is
running**. Without it, the walk could skip past a running sha that has no GHCR
image (a prior deploy took the local-build fallback) and land on an
older-but-built ancestor — silently deploying *and migrating* strictly older
code.

`PREV_SHA` — the floor's anchor, the rollback target, and the baked-image-input
diff base — is the **running `e2i_api` container's image tag**, not
`git rev-parse HEAD` of the checkout (#1780). A human `git pull` in the checkout
moves HEAD with no deploy behind it; on 2026-08-21 that made the floor read a
*child* of the running sha as a "downgrade", refused it, and left a ~26-min local
build as the only path. `running_image_sha()` requires a 40-char lowercase-hex
tag that is a known commit here, and degrades to the checkout HEAD otherwise.

```bash
# What is ACTUALLY running (read-only, this is the anchor for everything below)
docker inspect --format '{{.Config.Image}}' e2i_api
docker inspect --format '{{.Config.Image}}' e2i_frontend
```

When the walk finds nothing, the run prints a `FALLBACK_REASON` naming the actual
branch taken — GHCR auth unavailable, nothing built in the 30-commit window, or
the downgrade floor firing — and falls back to the blind `origin/main` reset.

---

## 2. `ensure-main-image` and the published-image assertion (#1780/#1782/#1785)

Two distinct mechanisms, often confused:

**`ensure-main-image` (a GitHub job, fail-SOFT).** Runs after both build jobs and
before the SSH deploy. It re-resolves `origin/main`, probes GHCR for both images
at that sha, and builds+pushes them if either is missing — so the droplet never
has to. It tags the **sha only, never `latest`**. Both build steps are
`continue-on-error: true` and every give-up path leaves `needed=false`: blocking a
deploy because a probe hiccuped would trade a slow deploy for *no* deploy.

**The published-image assertion (on the droplet, fail-HARD).** After the target
sha is fixed and `reset --hard` has landed, and **before** migrations or any
flip, the script asserts both images exist. On a genuine miss:

```
==> ERROR: no published GHCR image for the resolved target <sha>
==>   Recover: gh workflow run deploy.yml   (builds + pushes, then redeploy).
==> Deploy FAILED before any change was made — nothing was flipped or migrated.
```

That last line is a guarantee, not a hope: the assertion sits before
`bash scripts/run_migrations.sh`. **If you see it, the box was not touched** —
re-dispatch and move on.

It refuses rather than local-builds because a locally-built image exists only on
that box: prod then has no rollback target (`rollback_to_prev` cannot pull it) and
the next deploy resolves the same imageless sha and repeats the build.

**The assertion stands down — deliberately — when the registry will not answer.**
A missing image and a refused manifest read are the same non-zero at a boolean
call site, so the gate asks a three-way verdict: published / definitively absent /
no answer. It refuses only on the middle one. `docker login` failing, or a login
that succeeds while `read:packages` is enforced at the manifest (scope-reduced
token, package unlinked from its repo, revoked org access, a 5xx), all produce
`==> WARN: SKIPPING the #1785 published-image assertion` and pre-#1785 behaviour.
**A stand-down warning is not a missing image** — do not send anyone after
`ensure-main-image` for it.

---

## 3. Converged-but-FAILED — verify with container content markers

**A `failure` conclusion on the deploy job does not mean prod did not flip.**
Measured on run 32507847667 (2026-08-21): all six services flipped and the health
gate passed, then the SSH `command_timeout` fired four minutes later mid-prune and
the run reported FAILED (#1780/#1784).

Never read the job conclusion as the state of the box. Read the box:

```bash
# 1. Image sha the app tier is actually on
docker inspect --format '{{.Config.Image}}' e2i_api
docker inspect --format '{{.Config.Image}}' e2i_frontend

# 2. CONTENT marker — prove the change you shipped is in the running image.
#    Back-to-back merges can deploy a NEWER main than the sha you merged, so a
#    matching sha is necessary, not sufficient; grep for the thing you changed.
docker exec e2i_api grep -c '<symbol you added>' /app/src/<path you changed>
```

The `Deployment summary` step is built for this. On a rollout failure it refuses
to name a gate (`outcome` is one word) and instead enumerates what the gated half
covers, in order: **(1)** the published-image assertion — *nothing flipped or
migrated*; **(2)** the DB migrations; **(3)** the service flip and its rollback;
**(4)** the post-deploy image-drift check, which runs only after a converged flip.
On `cancelled` it says plainly that how far it got is not derivable — check the
running image tags before deploying again. Only `skipped`/empty proves the step
never started.

### The gates, in the order they run

| # | Gate | On failure |
|---|---|---|
| 1 | published-image assertion | exit 1, nothing touched |
| 2 | `scripts/run_migrations.sh` | `set -e` fails the step |
| 3 | `feast` + `feast-materializer` recreate | roll both back to `PREV_SHA` |
| 4 | materializer fresh-heartbeat, 60 × 10 s | roll both back; **app never flipped** |
| 5 | app-tier flip (`api frontend worker_light worker_medium scheduler`) | roll feast + app back |
| 6 | `/health`, 30 × 2 s on `localhost:8000` | roll the **app tier only** back (feast stays at the new sha) |
| 7 | `bentoml` `POST /model_info` non-empty `available_models`, 30 attempts | roll app tier + bentoml back |
| 8 | image-drift check (§4) | fail the run, **no rollback** |

---

## 4. The image-drift gate (#1479/#1480)

Last command of the gated half:

```bash
python3 scripts/deploy/check_image_drift.py --compose-cmd "$COMPOSE_CMD"
```

It compares every compose-pinned service's **running** image against the pin the
deploy's own `$COMPOSE_CMD` resolves. It exists because mlflow pin bumps
(#442/#1477) never reached the live server — the rollout never recreates mlflow —
and nothing noticed for five weeks. A stale app container from a failed recreate
is caught here too.

Any mismatch not held by a dated, ticketed entry in
`scripts/deploy/image_drift_allowlist.json` fails the run. **That file is `[]` as
of 2026-09-07** — nothing is currently allowlisted.

There is deliberately **no rollback** on a drift failure: the rollout converged
and was health-gated. A pin mismatch on a never-recreated sidecar is fixed by a
deliberate operator recreate — and stateful services need a backup first (mlflow's
first v3.15.1 boot runs a one-way sqlite migration) — not by rolling back code.

Note `scripts/deploy/**` is itself a deploy trigger: the check and its allowlist
are consumed from the droplet checkout, never baked into an image, so a change
there needs a deploy of its own to take effect.

---

## 5. Abort conditions

**Dirty tracked files.** Before any reset:

```
ERROR: droplet has uncommitted changes to tracked files; refusing to reset --hard
```

`git status --porcelain --untracked-files=no` must be empty. **Untracked files are
tolerated on purpose** (`-uno`) — they survive `reset --hard`, and ops scratch
files in the checkout used to abort every deploy. Fix by committing/stashing or
reverting the hot-patched tracked file, then re-dispatch.

**`main` held by another worktree.** The deploy re-attaches HEAD to `main`
(`git checkout -B main`, no start point) before any reset, because
`reset --hard` moves the *checked-out branch ref*: reset while HEAD sits on
someone's branch and their committed-but-unpushed work is rewound to the reflog
while the dirty-file guard reports all clear (#1787, measured 2026-08-21). This
also repairs a detached HEAD — the state a 2026-07-21 rollback left for two days,
during which the unheld `main` was taken over by worktree
`gh pr merge --delete-branch` runs.

`git checkout -B main` **bypasses git's own worktree lock** (measured on git
2.43.0: plain `git checkout main` refuses, the `-B` form succeeds and silently
moves `main` out from under the holder), so the workflow checks explicitly and
refuses:

```
ERROR: main is checked out in another worktree (<path>).
```

Fix it in the *holding* worktree, not in the production checkout:

```bash
git worktree list --porcelain | awk '/^worktree /{wt=substr($0,10)} /^branch refs\/heads\/main$/{print wt}'
git worktree prune          # if the holder is a stale/removed worktree
```

**Never do branch work in `/home/enunez/Projects/e2i_causal_analytics`** — it is
the deploy target. Leaving it on a feature branch, or with a tracked file
modified, blocks production deploys for everyone.

---

## 6. Rollback

Rollback is automatic on gates 3–7 (§3) and always targets `PREV_SHA` — the
**running `e2i_api` image tag**, not the checkout. This matters: on 2026-08-21 a
checkout-anchored rollback would have reset to a tree the containers were never
on and whose image is not in GHCR, so the rollback pull would fail and local-build
on an already-stressed box.

`rollback_to_prev` does `git reset --hard "$PREV_SHA"` (not `git checkout <sha>`,
which detaches HEAD), recomputes the compose overlay for the `PREV_SHA` tree, then
splits the services: the GHCR-backed tier
(`api frontend worker_light worker_medium worker_heavy scheduler`) is **pulled**
at `PREV_SHA` and recreated `--no-build`; locally built sidecars
(`feast`, `feast-materializer`, `bentoml` — never pushed to GHCR) always
`--build`. Every `up` is best-effort (`|| echo WARN`) so `set -e` cannot abort
mid-rollback before the caller's `exit 1`. If the pull is unavailable it falls
back to `--build`, which is the pre-#563 behaviour and the OOM that turned the
2026-06-23 rollback into a double fault.

A `WARN: rollback 'up' … failed — droplet may be in a PARTIAL state` line means
manual intervention: read the running image tags per service before anything else.

---

## 7. Concurrency, and the prune

**`concurrency: group: deploy-production`, `cancel-in-progress: false`** — a
whole-workflow group, so it supersedes an older *pending* run before it burns a
test+build cycle, and folds `workflow_dispatch` into the same lane. Cancelling a
mid-flight SSH deploy is exactly how a real partial state gets created, so
in-progress runs are never cancelled. A batch merge queues rather than piles up;
the 2026-07-31 6-way pileup (#1412) produced git/compose collisions and a
force-recreate SIGKILL (exit 137, **not** kernel OOM) with every run concluding
`failure` while the droplet converged healthy anyway.

**The prune is a separate, best-effort step** (#1784). `Post-deploy prune via SSH`
has `if: ${{ !cancelled() }}`, `continue-on-error: true`, and its own 15 m
`command_timeout`. It was split out because cleanup downstream of a gate spends
the *gate's* budget, and an SSH `command_timeout` kills the command outright — so
the `|| true` that makes a prune best-effort never runs. It prunes images only
when `ROLLOUT_OUTCOME=success` (an empty or `skipped` outcome falls on the
conservative side); the builder-cache prune always runs. A prune that overruns
reports `Deployment Succeeded — post-deploy cleanup did not finish` and **is not a
failed deploy**.

---

## 8. Restart side-effects

Any deploy recreates `e2i_api`, and an **in-flight `discover-effects` run does not
survive it**. The task stamps a liveness heartbeat every **15 s**
(`_DISCOVERY_HEARTBEAT_INTERVAL_SECONDS`) with a **120 s** TTL
(`_DISCOVERY_HEARTBEAT_TTL_SECONDS`, `src/api/routes/causal.py`). A poll that
finds a non-terminal row whose stamp is older than the TTL — or absent — knows
the task is gone and **read-repairs the row to `failed`**, persisting it so every
later poll on any worker agrees and the page stops polling (#1899).

Not a startup sweep: prod runs 2 gunicorn workers, and a freshly restarted worker
cannot tell whether the *other* worker's jobs are still alive; the heartbeat can.
The TTL equals the gunicorn worker timeout (`--timeout 120`) — an event loop
stalled that long is killed anyway, so a gap that long means the worker is gone,
never a live run.

**Operator expectation:** after a deploy, a `/causal-analysis` discovery that was
running flips to **Failed** (not Cancelled) within ~2 minutes of the next poll.
That is correct behaviour, not a regression. Re-run it.

---

## 9. Quick reference

```bash
# Trigger a deploy by hand (also the recovery path for a missing GHCR image)
gh workflow run deploy.yml

# What is running right now
docker inspect --format '{{.Config.Image}}' e2i_api

# Prove a specific change is in the running image
docker exec e2i_api grep -c '<marker>' /app/src/<path>

# Is main free for the deploy to reset?
git branch --show-current \
  && git status --porcelain --untracked-files=no
```

Deploy triggers are the `on.push.paths` list in `deploy.yml`: `src/**`,
`config/**`, the Dockerfiles and compose files the script `-f`s, `frontend/**`,
`requirements.txt`/`requirements.lock`/`pyproject.toml`, `patches/**`,
`scripts/**` (the whole tree is `COPY`ed into the image, #1783) and
`data/kg_cache/**`. `docs/**` is not a trigger — a docs-only PR does not deploy.
