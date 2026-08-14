"""Red-first pins for #1610 — ``cate_analyzer`` must not emit ``NaN`` or ``'nan'``.

Two adjacent NaN-handling defects, both measured on ``0cdd6bb0`` before the fix.

**1. The ``float("nan")`` CATE sentinel leaks.** A segment with rows on only one
side of the treatment cannot produce a CATE, so the tool set the value to
``float("nan")`` and put it in ``segments`` / ``effect_by_segment`` anyway.
``json.dumps(result, allow_nan=False)`` raises on it ("Out of range float values
are not JSON compliant"), and the lenient dump every other consumer uses emits
the bare token ``NaN``, which is not valid JSON for a strict reader and renders
as a plausible-looking blank in synthesis. The sentinel's INTENT — never
fabricate a CATE for a segment that has no contrast — is right and is preserved;
what changes is that the unestimable segment leaves the numeric results and is
disclosed in ``excluded_segments`` instead (the #1599 / PR #1604 treatment
applied to ``gap_calculator``: exclude-or-refuse, fail closed, disclose).

Measured pre-fix, wider than the issue describes: the ``len(treated) == 0`` guard
catches only the EMPTY-arm shape. A segment whose ``outcome`` column is entirely
null in one arm has BOTH arms populated, reaches ``treated.mean() -
control.mean()``, and produces ``NaN`` through the *else* branch — so the guard
has to be on the computed VALUE, exactly as #1599 found for group means.

**2. ``groupby(..., dropna=False)`` labels the null group ``'nan'``.** A null
segment key iterates as ``float('nan')`` and ``str()`` turns it into the literal
label ``"nan"`` — indistinguishable from a real category of that name, and (worse)
promotable: measured pre-fix, an all-null ``age_group`` column returned
``high_responders == ['nan']``, and ``gap_calculator`` on a ``{west, null}``
region column returned ``top_performer == 'nan'`` with ``gap=0.35``. Both name a
non-entity as the thing to act on, which is the fabricated-finding shape #1574
exists to forbid.

``dropna=False`` is KEPT (see the fix): the original choice was to not let null
rows vanish silently, and that intent survives — the rows are still seen and
still counted, but they are excluded from the numeric basis with an explicit,
disclosed count rather than silently mislabeled as a segment.

Tests build their OWN DataFrames (the anti-mock rule forbids fabricating data
inside tool bodies, not in tests).
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from src.agents.tool_composer import tool_registrations as tr
from src.agents.tool_composer.errors import ToolRefusalError

# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def _mixed_segment_frame() -> pd.DataFrame:
    """Two estimable segments, one un-estimable segment, one null-keyed group.

    * ``<50``  — both arms populated -> a real CATE of 1.0.
    * ``>65``  — both arms populated -> a real CATE of 0.6000000000000001.
    * ``solo`` — treated rows only   -> no contrast, the ``float("nan")`` site.
    * ``None`` — a null segment key  -> the ``'nan'`` label site.
    """
    return pd.DataFrame(
        {
            "age_group": ["<50", "<50", ">65", ">65", "solo", "solo", None, None],
            "high_engagement": [1, 0, 1, 0, 1, 1, 1, 0],
            "discontinuation_flag": [1.0, 0.0, 0.8, 0.2, 0.5, 0.5, 0.9, 0.1],
        }
    )


# The exact floats the pre-fix tool returned for the two estimable segments,
# transcribed from the measured red run. Compared with ``==`` (not approx): the
# fix must not perturb a real segment's CATE by even one ULP.
_PRE_FIX_CATES: Dict[str, float] = {"<50": 1.0, ">65": 0.6000000000000001}


def _cate(df: pd.DataFrame, **overrides: Any) -> Any:
    kwargs: Dict[str, Any] = {
        "treatment": "high_engagement",
        "outcome": "discontinuation_flag",
        "segments": ["age_group"],
        "estimation_data": df,
    }
    kwargs.update(overrides)
    return tr.cate_analyzer(**kwargs)


def _labels(dumped: Dict[str, Any]) -> List[str]:
    """Every segment label a consumer can read off the numeric results."""
    return (
        [str(s["name"]) for s in dumped["segments"]]
        + list(dumped["effect_by_segment"].keys())
        + list(dumped["high_responders"])
    )


# ---------------------------------------------------------------------------
# (a) no NaN anywhere in the serialized tool result
# ---------------------------------------------------------------------------
def test_cate_result_serializes_under_strict_json():
    """``allow_nan=False`` is the strict-consumer contract the sentinel broke."""
    dumped = _cate(_mixed_segment_frame()).model_dump()
    json.dumps(dumped, allow_nan=False)  # pre-fix: ValueError on the 'solo' NaN


def test_no_non_finite_float_survives_anywhere_in_the_result():
    dumped = _cate(_mixed_segment_frame()).model_dump()

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                _walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                _walk(value, f"{path}[{i}]")
        elif isinstance(node, float):
            assert math.isfinite(node), f"non-finite float at {path}: {node!r}"

    _walk(dumped, "result")


# ---------------------------------------------------------------------------
# (b) no 'nan' segment label
# ---------------------------------------------------------------------------
def test_null_segment_key_is_never_labeled_nan():
    dumped = _cate(_mixed_segment_frame()).model_dump()
    assert "nan" not in _labels(dumped)


def test_null_segment_key_is_not_promoted_to_a_high_responder():
    """Pre-fix, an all-null segment column returned ``high_responders == ['nan']``."""
    df = pd.DataFrame(
        {
            "age_group": [None, None, None, None],
            "high_engagement": [1, 0, 1, 0],
            "discontinuation_flag": [1.0, 0.0, 0.8, 0.2],
        }
    )
    # Nothing in this frame names a segment, so there is no per-segment finding
    # to report and the tool must refuse rather than return an empty result that
    # reads as "no heterogeneity found".
    with pytest.raises(ToolRefusalError) as excinfo:
        _cate(df)
    reason = str(excinfo.value)
    assert "age_group" in reason
    assert "cate_estimation_scope=" in reason


# ---------------------------------------------------------------------------
# (c) real segments' CATEs are byte-identical to the pre-fix values
# ---------------------------------------------------------------------------
def test_estimable_segment_cates_are_unchanged():
    dumped = _cate(_mixed_segment_frame()).model_dump()
    assert dumped["effect_by_segment"] == _PRE_FIX_CATES
    assert {s["name"]: s["cate"] for s in dumped["segments"]} == _PRE_FIX_CATES
    assert {s["name"]: s["n"] for s in dumped["segments"]} == {"<50": 2, ">65": 2}


def test_a_clean_frame_is_completely_unaffected():
    """No nulls, every segment estimable -> nothing is excluded, values stand."""
    df = pd.DataFrame(
        {
            "age_group": ["<50", "<50", ">65", ">65"],
            "high_engagement": [1, 0, 1, 0],
            "discontinuation_flag": [1.0, 0.0, 0.8, 0.2],
        }
    )
    dumped = _cate(df).model_dump()
    assert dumped["effect_by_segment"] == _PRE_FIX_CATES
    assert dumped["excluded_segments"] == []
    assert dumped["high_responders"] == ["<50"]


# ---------------------------------------------------------------------------
# (d) the exclusion is DISCLOSED
# ---------------------------------------------------------------------------
def test_unestimable_segment_is_disclosed_with_a_reason_and_a_count():
    dumped = _cate(_mixed_segment_frame()).model_dump()
    excluded = {str(e["name"]): e for e in dumped["excluded_segments"]}
    assert "solo" in excluded, "the no-contrast segment vanished without disclosure"
    assert excluded["solo"]["n"] == 2
    assert excluded["solo"]["reason"] == tr._CATE_EXCLUDED_NO_CONTRAST
    assert excluded["solo"]["detail"]


def test_missing_segment_key_is_disclosed_with_a_null_name():
    """The null group is named ``None``, never a string that could pass for a label."""
    dumped = _cate(_mixed_segment_frame()).model_dump()
    missing = [e for e in dumped["excluded_segments"] if e["reason"] == tr._CATE_EXCLUDED_MISSING]
    assert len(missing) == 1
    assert missing[0]["name"] is None
    assert missing[0]["n"] == 2
    assert "age_group" in missing[0]["detail"]


def test_disclosure_survives_strict_json():
    dumped = _cate(_mixed_segment_frame()).model_dump()
    round_tripped = json.loads(json.dumps(dumped, allow_nan=False))
    assert len(round_tripped["excluded_segments"]) == 2


# ---------------------------------------------------------------------------
# Wider than the issue: the NaN reaches the result through the *else* branch too
# ---------------------------------------------------------------------------
def test_segment_whose_outcome_is_all_null_in_one_arm_is_excluded():
    """Both arms are POPULATED, so the ``len(...) == 0`` guard never fires.

    ``treated.mean()`` is ``NaN`` over an all-null arm, so the difference is
    ``NaN`` and reached ``effect_by_segment`` through the branch that is supposed
    to hold only measured values.
    """
    df = pd.DataFrame(
        {
            "age_group": ["<50", "<50", ">65", ">65"],
            "high_engagement": [1, 0, 1, 0],
            "discontinuation_flag": [np.nan, np.nan, 0.8, 0.2],
        }
    )
    dumped = _cate(df).model_dump()
    json.dumps(dumped, allow_nan=False)
    assert list(dumped["effect_by_segment"]) == [">65"]
    excluded = {str(e["name"]): e for e in dumped["excluded_segments"]}
    assert excluded["<50"]["reason"] == tr._CATE_EXCLUDED_NON_FINITE


def test_nullable_float64_outcome_does_not_escape_as_a_bare_typeerror():
    """``float(pd.NA)`` raises ``TypeError`` — the #1599 codex iter-1 shape.

    A bare ``TypeError`` out of a tool body reaches the executor's RETRYING arm,
    so the refusal is charged to the tool's circuit breaker and the disclosure is
    lost. ``_coerce_finite`` funnels ``pd.NA`` to the same exclusion branch.
    """
    df = pd.DataFrame(
        {
            "age_group": ["<50", "<50", ">65", ">65"],
            "high_engagement": [1, 0, 1, 0],
            "discontinuation_flag": pd.array([None, None, 0.8, 0.2], dtype="Float64"),
        }
    )
    dumped = _cate(df).model_dump()
    json.dumps(dumped, allow_nan=False)
    assert list(dumped["effect_by_segment"]) == [">65"]


# ---------------------------------------------------------------------------
# The excluded segments must not steer the ranking chain
# ---------------------------------------------------------------------------
def test_threshold_and_high_responders_ignore_excluded_segments():
    """``high_responders`` is "CATE above the cross-segment mean".

    A null-keyed group with a real difference-in-means (0.8 here) shifted that
    mean pre-fix, so excluding it is not cosmetic — it changes which segments the
    tool recommends, and the recommendation is now computed over segments only.
    """
    dumped = _cate(_mixed_segment_frame()).model_dump()
    finite = list(dumped["effect_by_segment"].values())
    threshold = sum(finite) / len(finite)
    assert dumped["high_responders"] == [
        name
        for name, value in dumped["effect_by_segment"].items()
        if value >= threshold and value > 0
    ]
    assert dumped["high_responders"] == ["<50"]


def test_segment_ranker_over_the_fixed_result_never_ranks_a_non_segment():
    """The downstream consumer turns effects into ``recommended_targets``."""
    dumped = _cate(_mixed_segment_frame()).model_dump()
    ranked = tr.segment_ranker(cate_results=dumped).model_dump()
    assert [r["segment"] for r in ranked["ranking"]] == ["<50", ">65"]
    assert "nan" not in ranked["recommended_targets"]


# ---------------------------------------------------------------------------
# gap_calculator — the same null-key label at the second groupby site
# ---------------------------------------------------------------------------
def test_gap_null_region_key_is_not_a_performer():
    """Pre-fix: ``top_performer == 'nan'``, ``gap == 0.35`` — a non-entity won."""
    df = pd.DataFrame(
        {
            "geographic_region": ["west", "west", None, None],
            "market_share": [0.60, 0.60, 0.95, 0.95],
        }
    )
    # Only ONE real region remains, so this is the #1574 singleton shape and the
    # tool must refuse rather than compare 'west' against a non-entity.
    with pytest.raises(ToolRefusalError) as excinfo:
        tr.gap_calculator(
            metric="market_share", entity_type="region", entities=[], estimation_data=df
        )
    reason = str(excinfo.value)
    assert "'nan'" not in reason
    assert "'rows_missing_grouping_value': 2" in reason


def test_gap_ignores_the_null_group_and_compares_the_real_regions():
    df = pd.DataFrame(
        {
            "geographic_region": ["west", "west", "northeast", "northeast", None, None],
            "market_share": [0.60, 0.60, 0.80, 0.80, 0.95, 0.95],
        }
    )
    result = tr.gap_calculator(
        metric="market_share", entity_type="region", entities=[], estimation_data=df
    )
    dumped = result.model_dump()
    json.dumps(dumped, allow_nan=False)
    assert set(dumped["entity_values"]) == {"west", "northeast"}
    assert dumped["top_performer"] == "northeast"
    assert dumped["bottom_performer"] == "west"
    assert dumped["gap"] == pytest.approx(0.20)


def test_gap_reason_with_missing_rows_stays_inside_the_composer_carry_limit():
    """The composer truncates from the END, where ``estimation_data_scope`` sits.

    The missing-rows disclosure adds prose AND a scope key, so it needs the same
    bound #1574/#1599 measured for the other branches — in the WIDEST branch.
    """
    reason = tr._gap_comparability_reason(
        entity_type="territory",
        group_col="territory",
        groups_present=[f"territory_{i}_" + "x" * 200 for i in range(400)],
        groups_matched=[f"territory_{i}_" + "x" * 200 for i in range(400)],
        groups_non_finite=[f"territory_{i}_" + "x" * 200 for i in range(400)],
        entities=[f"requested_{i}_" + "y" * 200 for i in range(400)],
        row_count=10**9,
        rows_missing_group_key=10**9,
    )
    assert len(reason) < 2_000, f"reason is {len(reason)} chars"
    assert "'rows_missing_grouping_value': 1000000000" in reason
    assert "estimation_data_scope=" in reason


def test_gap_reason_is_unchanged_when_no_group_key_is_missing():
    """The #1574 / #1599 reasons stay byte-identical for frames with no nulls."""
    kwargs: Dict[str, Any] = {
        "entity_type": "brand",
        "group_col": "brand",
        "groups_present": ["Kisqali"],
        "groups_matched": ["Kisqali"],
        "entities": ["Kisqali", "Ibrance"],
        "row_count": 120,
    }
    assert tr._gap_comparability_reason(**kwargs) == tr._gap_comparability_reason(
        **kwargs, rows_missing_group_key=0
    )
    assert "rows_missing_grouping_value" not in tr._gap_comparability_reason(**kwargs)


# ---------------------------------------------------------------------------
# The null-key detector itself
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value",
    [None, float("nan"), np.nan, pd.NA, pd.NaT],
    ids=["none", "float_nan", "np_nan", "pd_na", "pd_nat"],
)
def test_missing_group_key_detector_covers_every_null_flavor(value):
    assert tr._is_missing_group_key(value) is True


@pytest.mark.parametrize(
    "value",
    ["west", "nan", "", 0, 0.0, False],
    ids=["str", "literal_nan_string", "empty_str", "zero_int", "zero_float", "false"],
)
def test_missing_group_key_detector_keeps_real_keys(value):
    """A real category literally NAMED "nan" is a key, not a null."""
    assert tr._is_missing_group_key(value) is False
