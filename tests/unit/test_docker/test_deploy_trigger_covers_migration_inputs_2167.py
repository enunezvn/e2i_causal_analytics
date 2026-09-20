"""#2167 — every schema directory the deploy APPLIES must be a deploy trigger.

The #1783 family, one layer over. That issue closed the gap for paths baked into the
production IMAGE (``COPY`` sources absent from ``on.push.paths``). This is the same
defect for paths the deploy CONSUMES AT DEPLOY TIME out of the droplet checkout.

``deploy.yml``'s rollout script runs ``bash scripts/run_migrations.sh``, which applies
every pending forward ``*.sql`` under each of its ``MIGRATION_DIRS`` — all of them under
``database/``. ``database/**`` was in no trigger path. So a migration-only push changed
what the deploy WOULD APPLY without running the deploy that applies it, and the new
migration sat unapplied until some unrelated ``src/**`` push carried it incidentally.
Unbounded, unannounced, self-healing — #1783's own words for its version of this, and
#1479's five-week mlflow pin drift before that.

FOUND THE HARD WAY, 2026-09-20 (lane #2167). The per_hcp_rollup contract migration 146
was applied to production BY HAND and then moved into ``database/migrations/``. Checking
how the move would reach production surfaced that merging it would deploy nothing at all:
the whole diff was ``database/``, ``tests/`` and ``docs/``, none of which is a trigger.
The lane needed a manual ``workflow_dispatch``. That is a fine workaround for someone who
KNOWS; the defect is that nothing tells someone who does not.

WHY THIS IS THE RIGHT LAYER TO GUARD. The alternative reading is "migrations are applied
by the deploy, so of course a migration change should deploy" — true, but a comment
saying so is not a gate. The invariant is derived here rather than asserted: the runner's
``MIGRATION_DIRS`` is parsed out of the runner (its own header: "MIGRATION_DIRS is the
only source of that scope -- do not restate the dir count here, it has gone stale twice"),
and the premise that the deploy runs the runner at all is itself asserted. Add a ninth
schema directory outside ``database/`` and this fails rather than shipping the defect a
third time.

ACCEPTED COST, stated rather than hidden (the #1783 precedent does the same): every
migration-only push now runs the full ~25-minute deploy pipeline. That is the point — an
unapplied migration is worth more than a saved pipeline — but it is a real cost and it
should be a decision, not a surprise.

FAITHFULNESS LIMIT: this proves the trigger set covers the runner's declared schema
directories. It does not prove GitHub's path-filter engine agrees with ``_glob_covers``
on exotic glob forms, which is why ``test_deploy_trigger_covers_image_inputs_1783``'s
shape gate restricts the trigger vocabulary to the two unambiguous forms — and this
module reuses that same matcher rather than growing a second copy that could drift.
"""

from __future__ import annotations

import re

from tests.unit.test_docker.conftest import REPO_ROOT, ssh_script, trigger_paths
from tests.unit.test_docker.test_deploy_trigger_covers_image_inputs_1783 import _glob_covers

RUNNER = REPO_ROOT / "scripts" / "run_migrations.sh"


def _migration_dirs() -> list[str]:
    """``$PROJECT_ROOT``-relative schema directories ``run_migrations.sh`` applies.

    Parsed from the runner, never restated. A restated list is a snapshot of now, and
    this guard exists precisely to notice the day the two disagree.
    """
    block = re.search(r"^MIGRATION_DIRS=\((.*?)^\)", RUNNER.read_text(), re.M | re.S)
    assert block, "run_migrations.sh no longer declares MIGRATION_DIRS as an array literal"
    return re.findall(r'"\$PROJECT_ROOT/([^":]+)::', block.group(1))


def test_the_deploy_actually_runs_the_migration_runner():
    """THE PREMISE. Everything below is only a defect while the deploy applies
    migrations at all. If that call is ever removed, this test should say so loudly
    rather than keep asserting a coverage rule about a step that no longer exists."""
    assert "run_migrations.sh" in ssh_script(), (
        "deploy.yml's rollout script no longer invokes scripts/run_migrations.sh — "
        "re-derive this guard instead of deleting it: if migrations moved somewhere "
        "else, THAT place's inputs are what now need trigger coverage"
    )


def test_the_migration_dirs_parser_is_not_vacuous():
    """A parser that silently matched nothing would make the guard below pass by
    covering no directory at all — the same vacuity the runner's own suite and
    ``SCAN``'s per-root floors guard against elsewhere."""
    dirs = _migration_dirs()
    assert "database/migrations" in dirs, dirs
    assert len(dirs) >= 8, dirs
    missing = [d for d in dirs if not (REPO_ROOT / d).is_dir()]
    assert not missing, f"MIGRATION_DIRS names directories that do not exist: {missing}"


def test_every_migration_directory_is_covered_by_a_deploy_trigger():
    """THE GUARD. A schema directory the deploy applies, that no trigger path covers,
    is a migration that ships whenever something unrelated happens to push."""
    paths = trigger_paths()
    uncovered = [d for d in _migration_dirs() if not any(_glob_covers(t, d) for t in paths)]
    assert not uncovered, (
        f"{uncovered} is/are applied by scripts/run_migrations.sh during the deploy but "
        f"matched by no entry in deploy.yml's on.push.paths {paths} — a push changing "
        "only those files alters what the deploy WOULD APPLY without running the deploy "
        "that applies it"
    )


def test_the_matcher_reports_the_gap_this_issue_closed():
    """POSITIVE CONTROL, over the same extract -> match -> report pipeline.

    Without it, a matcher that can never report anything would pass the guard above
    forever. Drives the real ``MIGRATION_DIRS`` against a trigger list with the fix
    REMOVED, and pins the exact set that comes back — which is the state deploy.yml was
    actually in before #2167.
    """
    before_the_fix = [p for p in trigger_paths() if not _glob_covers(p, "database/migrations")]
    assert before_the_fix != trigger_paths(), (
        "removing database coverage from the trigger list changed nothing, so the fix "
        "is not in the list and this control proves nothing"
    )
    uncovered = [
        d for d in _migration_dirs() if not any(_glob_covers(t, d) for t in before_the_fix)
    ]
    assert uncovered == _migration_dirs(), (
        "with the fix removed, EVERY migration directory should be uncovered — the "
        f"matcher reported only {uncovered}"
    )
