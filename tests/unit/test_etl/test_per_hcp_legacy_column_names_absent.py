"""No live code path reads the legacy per_hcp_rollup column names (migration 144).

Migration 144 renames ``business_metrics.{trx_count,nrx_count,total_rx_count}`` to
``{triggers_delivered_count,triggers_accepted_count,triggers_total_count}``, because
those columns never held prescriptions -- they hold trigger funnel counts. This is the
census proof for that rename: it fails with the file and line of every remaining
reader, so a missed consumer cannot reach a deploy as a runtime ``column does not
exist``.

The allowlist is the load-bearing part, and it is deliberately tiny. Anything added to
it is invisible to this guard FOREVER, which is the same rot shape as a permanently
skipped test -- so a name belongs here only when renaming it would make the code WRONG,
never to turn a red green.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LEGACY = re.compile(r"\b(trx_count|nrx_count|total_rx_count)\b")

#: Scanned roots, with a measured floor apiece (2026-09-17: 1041 / 10 / 146 ``.py``
#: files). The floors exist because ``rglob`` on a path that does not exist returns an
#: empty iterator and the scan then passes by covering nothing. ``feature_repo`` is the
#: one that matters most -- it holds 9 of the 12 real offenders -- and it is also the
#: smallest, so a single guard tuned to ``src`` would not have protected it.
SCAN: dict[str, int] = {"src": 200, "feature_repo": 5, "scripts": 50}

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
    assert not offenders, "legacy per_hcp_rollup names (renamed by migration 144):\n" + "\n".join(
        offenders
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
    reference, creating the defect the rename is meant to remove.
    """
    schema = (REPO / "src/api/schemas/causal.py").read_text()
    catalog = (REPO / "src/api/routes/causal/catalog.py").read_text()

    # The two halves of the example travel together; neither is a column.
    assert '"treatment_var": "rep_visits"' in schema
    assert '"outcome_var": "trx_count"' in schema
    assert "defaults rep_visits/trx_count were not real columns" in catalog

    # The discriminating half: a real column would appear in the schema DDL. Neither
    # of these does, and the treatment half is the control -- it is not renamed by 144,
    # so if it were a column this argument would be wrong about both.
    ddl = "\n".join(
        p.read_text() for p in sorted((REPO / "database").rglob("*.sql")) if p.is_file()
    )
    assert "rep_visits" not in ddl, "rep_visits IS a column after all; re-assess the exemption"
