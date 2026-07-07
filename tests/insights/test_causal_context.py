"""causal_context — shared causal-registry grounding for insight surfaces.

Why this exists (2026-07-07): the causal_paths registry gained a commercial
grain (TRx/NRx/NBRx/share/ROI/intent chains), and the user asked for that
coverage to span the strategic-insight surfaces, not just the chatbot. This
module is the ONE hop those surfaces share: a server-side, provenance-gated
fetch plus two formatters —

* ``format_causal_context``: figures included (resource-optimization's
  signature already consumes raw numbers in its other inputs).
* ``format_driver_names``: digit-free humanized names ONLY, because the
  executive brief's placeholder guard rejects ANY numeric character in LM
  output (``_placeholder_violation`` check 2) — a node name like
  ``persistent_180d`` fed raw would poison every sample into fallback.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.insights.causal_context import (
    fetch_commercial_drivers,
    format_causal_context,
    format_driver_names,
)

DRIVERS = [
    {
        "start": "rep_detailing_frequency",
        "end": "trx_volume",
        "effect": 0.2977,
        "confidence": 0.87,
        "synthetic": True,
    },
    {
        "start": "competitor_activity",
        "end": "trx_market_share",
        "effect": -0.15,
        "confidence": 0.83,
        "synthetic": True,
    },
    {
        "start": "persistent_180d",
        "end": "trx_volume",
        "effect": 0.31,
        "confidence": 0.9,
        "synthetic": True,
    },
]


def test_format_causal_context_includes_figures_and_provenance():
    text = format_causal_context(DRIVERS)
    assert "rep detailing frequency" in text
    assert "TRx volume" in text
    assert "+0.30" in text or "+0.298" in text  # effect surfaced
    assert "0.87" in text  # confidence surfaced
    assert "synthetic" in text.lower()  # provenance label rides along
    # Negative competitor pressure keeps its sign.
    assert "-0.15" in text


def test_format_causal_context_empty_is_honest():
    text = format_causal_context([])
    assert "no modeled causal drivers" in text.lower()


def test_format_driver_names_is_digit_free():
    names = format_driver_names(DRIVERS)
    assert names  # something survives
    joined = " ".join(names)
    assert not any(ch.isnumeric() for ch in joined), joined
    # persistent_180d must humanize (not carry its digits) — it maps to
    # "patient persistence".
    assert any("patient persistence" in n for n in names)
    assert any("TRx volume" in n for n in names)


def test_format_driver_names_drops_unmappable_digit_nodes():
    weird = [{"start": "weird_90d_thing", "end": "trx_volume", "effect": 0.1, "confidence": 0.8}]
    assert format_driver_names(weird) == []


@pytest.mark.asyncio
async def test_fetch_dedupes_across_outcomes_and_caps():
    row = {
        "path_id": "scp_c1",
        "start_node": "rep_detailing_frequency",
        "end_node": "trx_volume",
        "causal_effect_size": 0.3,
        "confidence_level": 0.9,
        "is_synthetic": True,
    }
    other = {**row, "path_id": "scp_c2", "start_node": "formulary_status", "confidence_level": 0.8}
    repo = MagicMock()
    # Same top row returned for both outcome terms -> must dedupe by path_id.
    repo.search_paths_for_outcome = AsyncMock(return_value=[row, other])
    drivers = await fetch_commercial_drivers(
        "Kisqali", outcomes=("TRx", "ROI"), limit=2, repo=repo, include_synthetic=True
    )
    assert len(drivers) == 2
    assert drivers[0]["confidence"] == 0.9  # sorted desc
    assert drivers[0]["start"] == "rep_detailing_frequency"
    assert repo.search_paths_for_outcome.await_count == 2


@pytest.mark.asyncio
async def test_fetch_swallows_errors_returns_empty():
    repo = MagicMock()
    repo.search_paths_for_outcome = AsyncMock(side_effect=RuntimeError("db down"))
    drivers = await fetch_commercial_drivers("Kisqali", repo=repo, include_synthetic=True)
    assert drivers == []
