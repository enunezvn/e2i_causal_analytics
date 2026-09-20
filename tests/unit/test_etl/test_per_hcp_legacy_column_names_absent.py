"""No live code path reads the legacy per_hcp_rollup column names (migrations 144/146).

``business_metrics.{trx_count,nrx_count,total_rx_count}`` never held prescriptions --
they hold trigger funnel counts -- so the lane gives them the honest names
``{triggers_delivered_count,triggers_accepted_count,triggers_total_count}``. This is the
census proof: it fails with the file and line of every remaining reader.

AMENDED 2026-09-18, and the amendment is why this guard matters MORE than when it was
written. The change was an in-place rename; codex iter1 HIGH-1 replaced it with an
expand/contract pair -- migration 144 ADDs the canonical columns beside the legacy ones
and keeps both true with a bidirectional trigger, and 146 retires the legacy three in a
LATER deploy.

Under the rename, a missed reader announced itself the moment the migration landed, as a
runtime ``column does not exist``. Under the expand it does NOT: the legacy columns are
still there, still correct, and a missed reader keeps working -- silently depending on a
deprecated column -- right up until someone applies the contract, which is a different
deploy with a different author on a different day. **This static census was the only
thing standing between a missed consumer and that far-away breakage.** Treat a name added
to ALLOWED accordingly.

AMENDED AGAIN 2026-09-20 (issue #2167). The contract WAS applied: the three legacy
columns, the sync trigger and its function were dropped from production at 01:51:04Z, and
``database/migrations/146_drop_legacy_per_hcp_count_columns.sql`` now ships like any other
migration. The far-away breakage is therefore no longer far away -- a reader that slips
past this census now fails at runtime on the next query it makes. That makes the census
CHEAPER to violate safely and more valuable to keep: it is what turns a production
``column does not exist`` into a red CI job.

The allowlist is the load-bearing part, and it is deliberately tiny. Anything added to
it is invisible to this guard FOREVER, which is the same rot shape as a permanently
skipped test -- so a name belongs here only when renaming it would make the code WRONG,
never to turn a red green.

Red-first evidence, and why it is recorded here rather than re-derivable
-----------------------------------------------------------------------
This test was written to be red and is committed green, because Task 25 landed between
the brief and the run. The transition IS the census proof, and the "before" half is no
longer reachable from the working tree -- only from git history -- so both halves are
written down. Measured independently twice (by the implementer over ``git show`` blobs
and by the dispatcher over checkouts), same command, same ALLOWED exclusions:

* ``8c7a61204`` (pre-merge): **12** offenders outside ALLOWED, across exactly Task 25's
  five files -- ``feature_repo/data_sources.py:46,47,48``,
  ``feature_repo/features/hcp_features.py:31,32,33``,
  ``feature_repo/features/market_features.py:28,29,30``,
  ``src/agents/drift_monitor/nodes/alert_aggregator.py:637,638``,
  ``src/feature_store/feature_analyzer_adapter.py:450``.
* ``41a37592e`` (post-merge): **0**.

They are green because they were FIXED. None of those five is in ALLOWED, and none may
be added: the allowlist has to stay able to catch a sixth file that appears later.

Why this file is worth more than its size suggests
--------------------------------------------------
CI's lint job runs over ``src/ tests/`` only, so **nothing in CI looks at
``feature_repo/`` at all** -- and ``feature_repo`` is where 9 of the 12 offenders above
lived. For this class of defect this test is the only CI-visible coverage of that
directory, which is why ``test_the_scan_is_not_vacuous`` guards every root rather than
just ``src``.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LEGACY = re.compile(r"\b(trx_count|nrx_count|total_rx_count)\b")

#: Scanned roots, with a measured floor apiece (2026-09-17: 1041 / 10 / 146 ``.py``
#: files; ``config`` added 2026-09-20 at 1). The floors exist because ``rglob`` on a
#: path that does not exist returns an empty iterator and the scan then passes by
#: covering nothing. ``feature_repo`` is the one that matters most -- it holds 9 of the
#: 12 real offenders -- and it is also the smallest, so a single guard tuned to ``src``
#: would not have protected it.
#:
#: ``config`` was added by issue #2167, and it was NOT added on a hunch: the new
#: ``test_every_python_bearing_directory_in_the_production_image_is_scanned`` below
#: parses the Dockerfile's production stage and went red on its first run naming it.
#: It holds exactly one ``.py`` -- ``config/gunicorn.conf.py``, which issues no SQL --
#: but gunicorn EXECUTES it at boot in the production image, so it is a live code path
#: this census had never looked at. The floor is 1 rather than 0 so that deleting the
#: file fails loudly instead of silently emptying the root.
SCAN: dict[str, int] = {"src": 200, "feature_repo": 5, "scripts": 50, "config": 1}

#: Deliberate, reasoned mentions. Each names the legacy string WITHOUT reading the
#: renamed column, so renaming it would introduce a defect rather than fix one.
ALLOWED = {
    # The refusal set itself: these strings are the INPUT this repository rejects.
    # Renaming them would make the repository start accepting "trx_count" again.
    "src/repositories/experiment_outcome.py",
    # A causal-API example naming an outcome VARIABLE, not a column -- see
    # test_the_causal_api_example_is_a_variable_name_not_a_column below, which pins
    # the evidence rather than leaving this comment to be taken on trust.
    "src/api/schemas/causal.py",
    # A historical note recording that rep_visits/trx_count "were not real columns".
    # Renaming it would falsify the history it exists to record.
    "src/api/routes/causal/catalog.py",
}


def _offenders() -> list[str]:
    found: list[str] = []
    for top in SCAN:
        for path in sorted((REPO / top).rglob("*.py")):
            rel = path.relative_to(REPO).as_posix()
            if "__pycache__" in path.parts or rel in ALLOWED:
                continue
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if LEGACY.search(line):
                    found.append(f"{rel}:{i}: {line.strip()}")
    return found


def test_no_legacy_per_hcp_column_name_outside_the_allowlist():
    offenders = _offenders()
    assert not offenders, (
        "legacy per_hcp_rollup names (superseded by migration 144's canonical columns):\n"
        + "\n".join(offenders)
    )


def test_the_scan_is_not_vacuous():
    """Every scanned root must exist and be populated.

    The plan's version guarded ``src`` alone. That is the least likely root to break
    and the least costly if it did: ``feature_repo`` is where most offenders live, and
    a typo there would have made this guard silently cover a third of what it claims.
    """
    for top, floor in SCAN.items():
        root = REPO / top
        assert root.is_dir(), (
            f"scan root {top} does not exist; the census covers less than it claims"
        )
        n = len([p for p in root.rglob("*.py") if "__pycache__" not in p.parts])
        assert n >= floor, f"{top}: {n} .py files, expected >= {floor}"


def test_the_allowlist_entries_all_exist_and_all_still_mention_a_legacy_name():
    """An allowlist entry that no longer applies is a hole nobody notices.

    If a file is deleted or renamed, or stops mentioning the legacy names at all, its
    exemption silently starts covering nothing -- or worse, a future file at the same
    path inherits a blanket exemption it was never assessed for.
    """
    for rel in sorted(ALLOWED):
        path = REPO / rel
        assert path.is_file(), f"allowlisted {rel} no longer exists; drop it from ALLOWED"
        assert LEGACY.search(path.read_text()), (
            f"allowlisted {rel} no longer mentions a legacy name; drop it from ALLOWED"
        )


def test_the_causal_api_example_is_a_variable_name_not_a_column():
    """Pins the evidence for the ``causal.py`` exemption, which is the only one whose
    justification is not self-evident from the file itself.

    ``RouteQueryRequest``'s example pairs ``treatment_var: "rep_visits"`` with
    ``outcome_var: "trx_count"``, and ``catalog.py`` records that exact pair as the
    causal-discovery page's old free-typed defaults, which "were not real columns".
    Measured 2026-09-17: ``rep_visits`` appears in ZERO files under ``database/``, so
    the pair is two placeholders, not two columns -- and renaming the outcome half to
    ``triggers_delivered_count`` would turn a placeholder into a FALSE column
    reference, creating the defect the canonical names are meant to remove.
    """
    schema = (REPO / "src/api/schemas/causal.py").read_text()
    catalog = (REPO / "src/api/routes/causal/catalog.py").read_text()

    # The two halves of the example travel together; neither is a column.
    assert '"treatment_var": "rep_visits"' in schema
    assert '"outcome_var": "trx_count"' in schema
    assert "defaults rep_visits/trx_count were not real columns" in catalog

    # The discriminating half: a real column would appear in the schema DDL. Neither
    # of these does, and the treatment half is the control -- 144 gives it no canonical
    # counterpart, so if it were a column this argument would be wrong about both.
    ddl = "\n".join(
        p.read_text() for p in sorted((REPO / "database").rglob("*.sql")) if p.is_file()
    )
    assert "rep_visits" not in ddl, "rep_visits IS a column after all; re-assess the exemption"


# ============================================================================
# Issue #2167 — SCAN must keep covering whatever the production image ships
# ============================================================================
#: The Dockerfile stage whose contents are what actually runs in production.
#: ``development`` also COPYs ``tests/``, and parsing the wrong stage would widen
#: this guard to directories that never reach a container.
_PROD_STAGE = "production"
DOCKERFILE = REPO / "docker" / "Dockerfile"


def _production_image_dirs() -> list[str]:
    """Repo-relative directories the production stage COPYs into the image.

    Parsed out of the Dockerfile rather than restated here, for the same reason
    ``_runner_dirs`` is parsed out of run_migrations.sh elsewhere in this suite: a
    restated list is a snapshot that goes stale silently, and the whole point of
    this guard is to notice the day the two drift apart.
    """
    body = DOCKERFILE.read_text()
    stage = re.search(
        rf"^FROM\s+\S+\s+AS\s+{_PROD_STAGE}\s*$(.*?)(?=^FROM\s|\Z)", body, re.M | re.S
    )
    assert stage, f"docker/Dockerfile no longer declares a `{_PROD_STAGE}` stage"
    # `COPY src/ ./src/` — a plain directory copy. `COPY --from=...` pulls from an
    # earlier stage (the venv) and is not repo source, so it is excluded.
    return re.findall(r"^COPY\s+(?!--)(\S+)/\s+\./", stage.group(1), re.M)


def test_the_dockerfile_parser_is_not_vacuous():
    """A parser that silently matched nothing would make the guard below pass by
    covering no directory at all — the same failure mode ``SCAN``'s own per-root
    floors exist to prevent."""
    dirs = _production_image_dirs()
    assert "src" in dirs, dirs
    assert "tests" not in dirs, (
        f"parsed `tests/` out of the {_PROD_STAGE} stage — that is the development "
        f"stage's COPY, so the parser is reading the wrong stage: {dirs}"
    )


def test_every_python_bearing_directory_in_the_production_image_is_scanned():
    """THE DRIFT GUARD.

    ``SCAN`` is the census's reach, and it is a hand-written dict. Nothing tied it
    to what actually ships. The census is now the ONLY thing standing between a
    re-introduced legacy reader and a production ``column does not exist`` — issue
    #2167 applied the contract, so the legacy columns are gone and a missed reader
    fails at runtime rather than silently reading a deprecated column.

    Red on its FIRST run, which is why it exists in this form: it named ``config``,
    a directory the production stage COPYs and this census had never scanned. The
    gap had been "measured" as empty by grepping ``config/`` for the legacy names
    and finding none — but "no offender in it today" and "covered by the census"
    are different claims, and only the second one survives someone adding a file
    tomorrow. That is the substitution this test removes.

    Python-bearing is the discriminator rather than a hardcoded exemption list for
    ``config``/``data``: the day someone puts a ``.py`` under a shipped data
    directory, it becomes a code path that ships unscanned, and this goes red and
    forces the decision instead of quietly widening the blind spot.

    ``SCAN`` may legitimately cover MORE than the image does — ``feature_repo`` is
    scanned and is not in this image (it runs in the feast container, and it held 9
    of the 12 original offenders). Superset is fine; a gap is not.
    """
    shipped = _production_image_dirs()
    python_bearing = [d for d in shipped if any((REPO / d).rglob("*.py"))]
    assert python_bearing, (
        f"no directory the {_PROD_STAGE} stage COPYs contains any .py file — either "
        f"the image stopped shipping python or the parse is wrong: {shipped}"
    )
    unscanned = sorted(set(python_bearing) - set(SCAN))
    assert not unscanned, (
        f"{unscanned} ship(s) in the production image and contain(s) python, but "
        f"is/are absent from SCAN {sorted(SCAN)} — a code path that reaches "
        "production without this census ever looking at it. Add it to SCAN with a "
        "measured floor, or stop shipping it."
    )
