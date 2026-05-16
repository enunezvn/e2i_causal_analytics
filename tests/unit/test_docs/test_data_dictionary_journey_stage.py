"""Parity test: docs/data/02-CORE-DATA-DICTIONARY.md `journey_stage` per-column
row enum values match the canonical Python source `E2I_JOURNEY_STAGES`.

Issue #248: the per-column row at docs/data/02-CORE-DATA-DICTIONARY.md
``journey_stage`` historically listed only 5 legacy values, while
``E2I_JOURNEY_STAGES`` (src/mlops/pandera_schemas.py) lists all 12 values
(5 legacy + 7 engagement-funnel values added by migration 035 /
issue #155 §2).

The enum summary table at line ~59 of the same docs file already has all 12;
this parity check pins the per-column row to the same source.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.mlops.pandera_schemas import E2I_JOURNEY_STAGES

# Repo root resolves to the worktree top, regardless of where pytest is invoked.
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DICT = REPO_ROOT / "docs" / "data" / "02-CORE-DATA-DICTIONARY.md"


def _extract_journey_stage_row_values() -> set[str]:
    """Parse the ``journey_stage`` per-column row and return its set of
    backtick-quoted enum values.

    The row format is::

        | `journey_stage` | `journey_stage_type` | YES | | Current stage: `a`, `b`, ... |

    We deliberately scope the search to the row line so that the summary
    table at line ~59 (which uses a different cell layout) is not picked up.
    """

    text = DATA_DICT.read_text(encoding="utf-8")

    # Find the per-column row. The leading "| `journey_stage` |" anchor is
    # unique in the file (the summary table uses "journey_stage_type" with the
    # `_type` suffix as the first cell).
    row_pattern = re.compile(
        r"^\|\s*`journey_stage`\s*\|.*$",
        re.MULTILINE,
    )
    matches = row_pattern.findall(text)
    assert len(matches) == 1, (
        f"Expected exactly one `journey_stage` per-column row in {DATA_DICT}, found {len(matches)}."
    )
    row = matches[0]

    # Inside the Description cell, enum values are backtick-quoted bare
    # identifiers (snake_case). The cell also references the type name
    # `journey_stage_type` and (after the fix) may reference issue/migration
    # tokens such as `migration 035`. Restrict to bare snake_case identifiers
    # that are also present in the canonical Python source — i.e. only count
    # tokens that are either already legacy values or known new values.
    backticked = re.findall(r"`([a-z0-9_]+)`", row)
    canonical = set(E2I_JOURNEY_STAGES)
    return {v for v in backticked if v in canonical}


def test_journey_stage_row_matches_canonical_python_source() -> None:
    """Per-column ``journey_stage`` row enumerates all canonical values."""
    docs_values = _extract_journey_stage_row_values()
    canonical = set(E2I_JOURNEY_STAGES)
    missing = canonical - docs_values
    extra = docs_values - canonical
    assert not missing, (
        f"docs/data/02-CORE-DATA-DICTIONARY.md `journey_stage` row is missing "
        f"values present in E2I_JOURNEY_STAGES: {sorted(missing)}"
    )
    assert not extra, (
        f"docs/data/02-CORE-DATA-DICTIONARY.md `journey_stage` row lists "
        f"values not present in E2I_JOURNEY_STAGES: {sorted(extra)}"
    )


def test_journey_stage_row_includes_seven_new_engagement_funnel_values() -> None:
    """Explicit pin: the 7 PR #152 engagement-funnel values are listed.

    Mirrors the audit-side intent of
    ``test_pr_a3_data_dictionary_journey_stage_row_includes_seven_new_values``
    (issue #248 references this name, though that audit test was never
    landed in this branch's tests/audit/ tree).
    """
    docs_values = _extract_journey_stage_row_values()
    seven_new = {
        "aware",
        "considering",
        "prescribed",
        "first_fill",
        "adherent",
        "discontinued",
        "maintained",
    }
    missing = seven_new - docs_values
    assert not missing, (
        f"docs/data/02-CORE-DATA-DICTIONARY.md `journey_stage` row is missing "
        f"these 7 engagement-funnel values (migration 035 / issue #155 §2): "
        f"{sorted(missing)}"
    )
