"""The "How E2I Works" page names the four predictive cohorts and the eight
intervention channels — those lists must be DERIVED from the backend, not typed.

Sources of truth:
- ``src.api.schemas.causal.CohortName`` — the four cohorts (each selects one
  outcome column as the label).
- ``src.digital_twin.effect.provider.INTERVENTION_CATALOG`` — the eight
  intervention channels (value + human label) that the Digital Twin dropdown
  and ``/digital-twin/intervention-types`` already source from.

The page's ``PREDICTIVE_COHORTS`` / ``INTERVENTION_CHANNELS`` arrays live in
``frontend/src/components/documentation/content.ts``. Their vitest pins the
literal lists; THIS test pins those lists to the Python enums, so adding or
renaming a cohort or channel on the backend fails here until the page catches
up (same mechanism as tests/unit/test_agents/test_agent_roster_ssot_1638.py).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.api.schemas.causal import CohortName
from src.digital_twin.effect.provider import INTERVENTION_CATALOG, INTERVENTION_TREATMENT_MAP

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTENT_TS = REPO_ROOT / "frontend" / "src" / "components" / "documentation" / "content.ts"


def _array_block(name: str) -> str:
    """Return the source text of ``export const <name>: ... = [ ... ];``."""
    src = CONTENT_TS.read_text(encoding="utf-8")
    m = re.search(rf"export const {name}\b[^=]*=\s*\[(.*?)\n\];", src, re.DOTALL)
    assert m, f"{name} not found in {CONTENT_TS}"
    return m.group(1)


def _fields(block: str, field: str) -> list[str]:
    return re.findall(rf"\b{field}: '([^']+)'", block)


class TestPredictiveCohortsMirrorCohortName:
    def test_ids_match_the_enum_in_order(self):
        assert _fields(_array_block("PREDICTIVE_COHORTS"), "id") == [c.value for c in CohortName]

    def test_label_columns_match_the_enum_docstring(self):
        """CohortName's docstring is the only place the label columns are written
        down; the page prints them, so keep them equal."""
        doc = CohortName.__doc__ or ""
        documented = dict(re.findall(r"- (\w+)\s+-> ([\w.]+)", doc))
        assert set(documented) == {c.value for c in CohortName}
        block = _array_block("PREDICTIVE_COHORTS")
        page = dict(zip(_fields(block, "id"), _fields(block, "labelColumn"), strict=True))
        assert page == documented


class TestInterventionChannelsMirrorTheCatalog:
    def test_ids_match_the_catalog_in_order(self):
        assert _fields(_array_block("INTERVENTION_CHANNELS"), "id") == [
            v for v, _ in INTERVENTION_CATALOG
        ]

    def test_names_are_the_catalog_labels(self):
        assert _fields(_array_block("INTERVENTION_CHANNELS"), "name") == [
            label for _, label in INTERVENTION_CATALOG
        ]

    def test_every_channel_is_estimable(self):
        """The page says each channel is a treatment the engine can estimate —
        true only while every catalog entry has a treatment column."""
        assert set(INTERVENTION_TREATMENT_MAP) == {v for v, _ in INTERVENTION_CATALOG}
