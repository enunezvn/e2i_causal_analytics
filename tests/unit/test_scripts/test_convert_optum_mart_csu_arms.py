"""Lane C (spec 2026-09-22 §3C.1): the mart converter recognises remibrutinib.

``select_csu_escalation_contrast`` canonicalises ``index_biologic_brand``
(vendor brand OR molecule spelling) through the shared vocabulary
(src/data/csu_biologics.py), keeps the three-arm contrast, derives
``treatment_remibrutinib`` (1 = RHAPSIDO, 0 = XOLAIR / DUPIXENT), reports every
other label as a loud attrition step, and refuses NULL brands and one-arm
frames. The synthetic backing builder and the post-launch real export both go
through it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.convert_optum_mart import (  # noqa: E402
    CSU_ESCALATION_ARMS,
    CSU_ESCALATION_COMPETITOR_ARMS,
    CSU_ESCALATION_TREATED_ARM,
    CSU_ESCALATION_TREATMENT_COL,
    select_csu_escalation_contrast,
)

pytestmark = pytest.mark.unit


def _frame(brands: list[str | None]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patid": list(range(1, len(brands) + 1)),
            "index_biologic_brand": brands,
            "age_at_index": [40 + i for i in range(len(brands))],
        }
    )


def test_arm_constants_match_the_vocabulary():
    assert CSU_ESCALATION_TREATMENT_COL == "treatment_remibrutinib"
    assert CSU_ESCALATION_TREATED_ARM == "RHAPSIDO"
    assert CSU_ESCALATION_COMPETITOR_ARMS == ("XOLAIR", "DUPIXENT")
    assert CSU_ESCALATION_ARMS == ("RHAPSIDO", "XOLAIR", "DUPIXENT")


def test_derives_the_treatment_from_brand_or_molecule_spellings():
    df = _frame(["RHAPSIDO", "XOLAIR", "DUPIXENT", "remibrutinib", "OMALIZUMAB", "Dupilumab"])
    out, attrition = select_csu_escalation_contrast(df)
    assert out[CSU_ESCALATION_TREATMENT_COL].tolist() == [1, 0, 0, 1, 0, 0]
    # The brand column is canonicalised so the API's brand filter sees one spelling per arm.
    assert out["index_biologic_brand"].tolist() == [
        "RHAPSIDO",
        "XOLAIR",
        "DUPIXENT",
        "RHAPSIDO",
        "XOLAIR",
        "DUPIXENT",
    ]
    assert out[CSU_ESCALATION_TREATMENT_COL].dtype == "int64"
    assert dict(attrition) == {
        "input_rows": 6,
        "in_contrast": 6,
        "treatment_remibrutinib=1": 2,
    }
    # Untouched columns ride along, index preserved.
    assert out["age_at_index"].tolist() == [40, 41, 42, 43, 44, 45]


def test_other_brands_leave_with_their_own_attrition_step():
    df = _frame(["RHAPSIDO", "XOLAIR", "no_treatment", "KISQALI", "KISQALI"])
    out, attrition = select_csu_escalation_contrast(df)
    assert out["patid"].tolist() == [1, 2]
    steps = dict(attrition)
    assert steps["excluded_arm:KISQALI"] == 2
    assert steps["excluded_arm:no_treatment"] == 1
    assert steps["in_contrast"] == 2
    assert [s for s, _ in attrition] == [
        "input_rows",
        "excluded_arm:KISQALI",
        "excluded_arm:no_treatment",
        "in_contrast",
        "treatment_remibrutinib=1",
    ]


def test_null_brand_is_a_data_integrity_failure():
    with pytest.raises(ValueError, match="NULL index_biologic_brand"):
        select_csu_escalation_contrast(_frame(["RHAPSIDO", None, "XOLAIR"]))


def test_one_arm_frame_is_refused():
    with pytest.raises(ValueError, match="constant"):
        select_csu_escalation_contrast(_frame(["XOLAIR", "DUPIXENT"]))
    with pytest.raises(ValueError, match="constant"):
        select_csu_escalation_contrast(_frame(["RHAPSIDO", "Rhapsido"]))


def test_input_frame_is_not_mutated():
    df = _frame(["RHAPSIDO", "xolair"])
    before = df.copy()
    select_csu_escalation_contrast(df)
    pd.testing.assert_frame_equal(df, before)
