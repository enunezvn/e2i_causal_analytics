"""Mid-responder bucket (segment-analysis clinical-HTE rebuild).

The page only ever surfaced high/low responders; the band between the 0.5x and
1.5x ATE thresholds was never emitted, so a "mid" segment (the user's "where is
the mid segment?") was invisible. segment_analyzer now also emits
``mid_responders`` (responder_type="average") for segments whose |CATE| sits
strictly between the low and high thresholds. High/low behaviour is unchanged,
and an empty mid bucket when nothing qualifies (the default for legacy callers).
"""

import pytest

from src.agents.heterogeneous_optimizer.nodes.segment_analyzer import SegmentAnalyzerNode


def _state(cate_by_segment, ate=0.10):
    return {
        "overall_ate": ate,
        "cate_by_segment": cate_by_segment,
        "top_segments_count": 10,
        "status": "analyzing",
        "warnings": [],
        "errors": [],
    }


def _cate(value, segment_value, n=200):
    return {
        "segment_name": "disease_severity_band",
        "segment_value": segment_value,
        "cate_estimate": value,
        "cate_ci_lower": value - 0.02,
        "cate_ci_upper": value + 0.02,
        "sample_size": n,
        "statistical_significance": True,
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mid_bucket_emitted_for_band_between_thresholds():
    # ate=0.10 -> high>=0.15, low<=0.05, mid in (0.05, 0.15).
    cbs = {
        "disease_severity_band": [
            _cate(0.30, "high"),  # ratio 3.0 -> high
            _cate(0.10, "medium"),  # ratio 1.0 -> mid (the bucket that was invisible)
            _cate(0.02, "low"),  # ratio 0.2 -> low
        ]
    }
    out = await SegmentAnalyzerNode().execute(_state(cbs))

    mids = out.get("mid_responders") or []
    assert len(mids) == 1, "the medium-severity segment must surface as a mid responder"
    assert mids[0]["responder_type"] == "average"
    assert mids[0]["segment_id"] == "disease_severity_band_medium"

    # High/low unchanged and disjoint from mid.
    high_ids = {h["segment_id"] for h in (out.get("high_responders") or [])}
    low_ids = {l["segment_id"] for l in (out.get("low_responders") or [])}
    mid_ids = {m["segment_id"] for m in mids}
    assert "disease_severity_band_high" in high_ids
    assert "disease_severity_band_low" in low_ids
    assert mid_ids.isdisjoint(high_ids) and mid_ids.isdisjoint(low_ids)

    # Comparison surfaces the mid count (so downstream narratives aren't 2-bucket).
    assert out["segment_comparison"]["mid_responder_count"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mid_bucket_empty_when_nothing_qualifies():
    # Only clear high and clear low -> mid is empty (not None-crashing).
    cbs = {
        "disease_severity_band": [
            _cate(0.40, "high"),
            _cate(0.01, "low"),
        ]
    }
    out = await SegmentAnalyzerNode().execute(_state(cbs))
    assert out.get("mid_responders") == []
    assert out["segment_comparison"]["mid_responder_count"] == 0
