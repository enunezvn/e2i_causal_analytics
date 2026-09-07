# Runbook — droplet maintenance cron (`/etc/cron.d/e2i-maintenance`)

**Scope:** the `scripts/maintenance/` layer that keeps the droplet alive between
deploys — orphan-process cleanup, memory monitoring, log truncation, Docker disk
cleanup — plus the freshness alarm that proves those jobs are actually running.

**Why this runbook exists:** on 2026-06-30 `/etc/cron.d/e2i-maintenance` silently
stopped executing. Root's password entered a forced-change state, so `pam_unix`
rejected every job and cron logged `Authentication token is no longer valid` on
each tick *without running the command*. Nothing noticed for **eight weeks** —
including `memory_monitor.sh --auto-cleanup`, the memory relief valve on a box
that runs production and development on the same host (#1798). Everything below
exists because of that.

**Verified 2026-09-07** against `scripts/maintenance/`,
`.github/workflows/maintenance-freshness.yml`, and the live box (read-only).

---

## 1. What runs, and when

Derived from `scripts/maintenance/setup_cron.sh` (the heredoc that writes the
cron file) and cross-checked against the live `/etc/cron.d/e2i-maintenance` on
2026-09-07 — **the two matched line for line**.

| Schedule | Job | Log | Success stamp |
|---|---|---|---|
| `*/15 * * * *` | `cleanup_orphans.sh` | `/var/log/e2i/orphan_cleanup.log` | `.cleanup_orphans.success` |
| `*/5 * * * *` | `memory_monitor.sh --auto-cleanup` | `/var/log/e2i/memory_monitor.log` | `.memory_monitor.success` |
| `0 2 * * *` | `find /var/log/e2i -name '*.log' -size +10M -exec truncate -s 1M {} \;` | — (inline, no redirect) | — (none; see §4) |
| `0 3 * * 0` | `docker_cleanup.sh` (Sundays) | `/var/log/e2i/docker_cleanup.log` | `.docker_cleanup.success` |

All four run as `root`, with `SHELL=/bin/bash` and an explicit `PATH` set at the
top of the cron file. Every command is an **absolute path into the shared
production checkout** `/home/enunez/Projects/e2i_causal_analytics` — the cron
layer runs the code that is checked out there, not a copy.

`setup_cron.sh` also installs two things that are *not* cron jobs:

- `/etc/profile.d/e2i-session-check.sh` — prints an orphan-count warning at login.
- `e2i-*` aliases appended to `/home/enunez/.bash_aliases` (`e2i-cleanup`,
  `e2i-cleanup-dry`, `e2i-memcheck`, `e2i-docker-cleanup`,
  `e2i-docker-cleanup-dry`, `e2i-logs`, `e2i-orphans`).

### Not in this crontab

The box also has a **user** crontab (`crontab -l` as `enunez`) carrying the daily
backup (`scripts/backup_cron.sh`, 02:00), the codex-orphan reaper (every 2 h), and
the **Monday 03:00 synthetic reseed** (`scripts/reseed_synthetic.sh` — see
[`synthetic_reseed.md`](synthetic_reseed.md)). The freshness alarm in §4 audits
**only** `/etc/cron.d/e2i-maintenance`; nothing watches the user crontab.

---

## 2. What each script does

**`cleanup_orphans.sh [--dry-run|-n] [--verbose|-v]`** — kills orphan/zombie
processes that freeze terminals: `exec(eval…)` Node/esbuild orphans from Vite
HMR, defunct Python, orphaned npm/node. `LOG_FILE` is overridable via the
environment (default `/var/log/e2i/orphan_cleanup.log`).

**`memory_monitor.sh [-t|--threshold N] [--swap-threshold N] [-c|--auto-cleanup] [-w|--webhook URL]`**
— alerts above a memory threshold (default 80%; swap default 50%), lists the top
consumers, and with `--auto-cleanup` (what cron passes) invokes the orphan
cleanup on high memory. Watched hogs: `esbuild`, `node`, `python`, `vite`, each
flagged above 1500 MB. Alert cooldown 300 s via `/tmp/e2i_memory_alert_cooldown`.

**`docker_cleanup.sh [--dry-run] [--verbose]`** — *conservative* disk cleanup.
Prunes build cache, dangling images, exited containers older than 24 h, dangling
**anonymous** volumes, and unused networks. It deliberately **skips**
`docker image prune -a`, named volumes, and anything referenced by a running or
compose-defined container.

**`check_maintenance_freshness.sh`** — the alarm; see §4. Do **not** put it in the
crontab it audits: a staleness check driven by the crontab it watches is dead
exactly when the thing it watches is dead.

---

## 3. Logs and success stamps

Logs live in `/var/log/e2i/` (root-owned, `755`). Truncation is by the 02:00 cron
entry, not logrotate: any `*.log` over 10 MB is truncated to 1 MB in place.

The `.<script>.success` stamps are the load-bearing artefact. **The log's mtime is
not evidence a job ran.** The log is written by *anything* that invokes the
script — a `--dry-run`, an aborted run, a human debugging by hand. On 2026-08-23 a
hand-run `docker_cleanup.sh --dry-run` wrote four lines, died on an invalid
filter, and reset the log's mtime; a log-mtime freshness check then reported
`docker_cleanup.log: OK` for a job that had not actually run since 2026-06-28.

Only a **real, completed** run touches `<logdir>/.<script>.success` — each script
defines `write_success_stamp()` and calls it on the success path only
(`cleanup_orphans.sh` guards it with `if ! $DRY_RUN`).

```bash
# What the alarm reads (read-only)
ls -la --time-style=+%FT%TZ /var/log/e2i/.*.success
ls -la /var/log/e2i/
```

---

## 4. How freshness is judged

`scripts/maintenance/check_maintenance_freshness.sh` parses the crontab itself
and, for each line, derives the interval from the five schedule fields:

- `*/N * * * *` → N minutes
- `M */N * * *` → N hours
- `M H * * *` → daily (86 400 s)
- `M H * * D` → weekly (604 800 s)
- anything else → `UNKNOWN SCHEDULE … not checked` (it reports rather than
  fabricates an interval)

A job is **stale** when its *success stamp* is older than `interval × TOLERANCE`.
`TOLERANCE` defaults to **2** and is overridable with `--tolerance N`. It
deliberately keeps no interval table of its own — a second copy would drift from
the real schedule.

Lines with **no `>>` redirect** are skipped, not failed: the 02:00 log-truncation
entry writes nothing, so there is nothing to check. With the current crontab that
leaves **3 checked jobs**.

```bash
# Run it by hand (read-only, sub-second)
/home/enunez/Projects/e2i_causal_analytics/scripts/maintenance/check_maintenance_freshness.sh
/home/enunez/Projects/e2i_causal_analytics/scripts/maintenance/check_maintenance_freshness.sh --tolerance 3 --verbose
```

Exit codes: `0` all fresh · `1` at least one job stale / never completed ·
`2` the crontab could not be read (maintenance uninstalled).

Diagnostic messages worth recognising:

- `NEVER COMPLETED — the log exists but there is no success stamp` — something
  wrote the log without finishing: a `--dry-run`, an aborted run, or a script
  version predating the stamps.
- `MISSING — no log and no success stamp` — the job has never run here.
- `STALE — last SUCCESS Ns ago, limit Ms` — it ran once and then stopped.

### Where `health_check.sh` reports it

`scripts/health_check.sh` has a `--- Maintenance ---` section calling
`check_maintenance()`, which shells out to the freshness script
(`FRESHNESS_SCRIPT` is overridable) and maps its exit code:

| rc | health_check verdict |
|---|---|
| 0 | `OK   Maintenance cron freshness` (counts HEALTHY) |
| 2 | `SKIP Maintenance cron freshness (crontab unreadable)` (counts SKIPPED — an unreadable crontab does not fail the box) |
| else | `FAIL Maintenance cron freshness` + the indented script output (counts UNHEALTHY) |

Missing/non-executable script → `SKIP`.

---

## 5. The unattended alarm: `Maintenance Freshness` workflow

`.github/workflows/maintenance-freshness.yml` — daily at **07:30 UTC** plus
`workflow_dispatch`. `health_check.sh` is manual, so the workflow is the only
caller independent of the box's own cron. Concurrency group
`maintenance-freshness`, `cancel-in-progress: false`.

The `check` job runs **two** SSH steps on purpose:

1. **Preflight** — connect, `cd` to the checkout, `git rev-parse`/`branch`,
   `test -x` the script, print the crontab and the stamps. If *this* fails the
   check never ran, and the verdict is `unreachable`.
2. **Run the script** — a non-zero exit here (1 = stale, 2 = crontab unreadable)
   means the check itself said so. Tolerance is passed through `envs: TOLERANCE`
   (appleboy forwards only what `envs:` names).

A `classify` step turns the two step outcomes into a single verdict —
`unreachable | fresh | stale | unknown` — so an SSH outage is never filed as
"the cron is stale".

**On red** the `report-failure` job (scheduled runs always; dispatches only with
`file_issue=true`) opens or comments on a tracking issue titled
`[maintenance-freshness] …`. Dedup keys on that **title prefix among open
issues**, not on the label — `gh issue list --label <missing>` silently returns
nothing, which would file a new issue every day. The label
`maintenance-freshness-failure` is self-healed when possible and the issue is
filed **without** it when creation fails (the alarm landing matters more than its
colour). The issue body's verdict decides where it sends you: `stale` points at
PAM, `chage -l root` and the stamps; `unreachable` points at the SSH secrets and
the box.

### Live-cert recipe (from the workflow header — does not touch the box)

```bash
gh workflow run maintenance-freshness.yml                    # expect green, output visible
gh workflow run maintenance-freshness.yml -f tolerance=0     # every job reads STALE -> red
gh workflow run maintenance-freshness.yml -f tolerance=0 -f file_issue=true
                                                             # ...and exercise the reporter
```

A scheduled run always files; a dispatch files only when asked, so a forced-red
certification does not spam issues.

---

## 6. Install / repair

```bash
sudo /home/enunez/Projects/e2i_causal_analytics/scripts/maintenance/setup_cron.sh
```

Idempotent: it recreates `/etc/cron.d/e2i-maintenance` (mode 644), `chmod +x`s the
three job scripts, creates `/var/log/e2i`, re-writes the profile.d check and the
aliases, starts `cron` if inactive, and then runs an initial orphan cleanup and
memory check. **It must run as root** (it exits 1 otherwise), and it is a
*mutating* command — it is the one procedure in this runbook that changes the box.

After a rebuild a stamp is legitimately **MISSING** until that job's first
scheduled run: up to 5 min for the memory monitor, 15 min for the orphan cleanup,
and up to a week for the Docker cleanup. Do not diagnose that as a failure.

### When the alarm says STALE

```bash
grep 'Authentication token is no longer valid' /var/log/syslog   # PAM refusing the job's account (#1798)
chage -l root                                                    # forced password change?
ls -la --time-style=+%FT%TZ /var/log/e2i/.*.success              # which job, how stale
systemctl is-active cron
tail -50 /var/log/e2i/memory_monitor.log
```

The 2026-06-30 shape is the one to rule out first: cron fires every tick, logs a
PAM rejection, and runs nothing — so the job scripts are innocent and the crontab
looks perfect.
