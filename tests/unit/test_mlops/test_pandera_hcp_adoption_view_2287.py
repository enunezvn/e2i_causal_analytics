"""#2287: the HCP-adoption goldstd view has a Pandera schema, so its contract is checked.

Without a registered schema ``run_schema_validation`` returns "skipped" for the view —
the fail-OPEN shape #2320 closed for ``patient_journeys`` contracts. The schema below
declares the database's own guarantees (``ck_hcp_brand_adoption_adopted``, the
``data_split_type`` / ``brand_type`` / ``region_type`` enums) plus non-negative counts,
so a check fails only on a frame the tables could not validly have produced.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.mlops.gold_standard_eval.cohort_spec import make_hcp_spec
from src.mlops.pandera_schemas import get_schema, validate_dataframe

VIEW = "hcp_adoption_goldstd_v"
SPEC = make_hcp_spec("Kisqali")
CONTRACT_COLUMNS = list(SPEC.base_covariates) + [SPEC.label_column]


def _frame(**overrides) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "peer_influence_score": [1.31, 2.4, None],
            "influence_network_size": [3, 10, None],
            "years_experience": [26, 4, 12],
            "specialty": ["dermatology", "oncology", None],
            "geographic_region": ["northeast", "west", None],
            "adopted": [0, 1, 0],
            "data_split": ["train", "holdout", "validation"],
        }
    )
    for k, v in overrides.items():
        df[k] = v
    return df


def test_the_view_has_a_registered_schema() -> None:
    assert get_schema(VIEW) is not None


def test_a_contract_frame_the_view_can_produce_passes() -> None:
    result = validate_dataframe(_frame(), VIEW, columns=CONTRACT_COLUMNS)
    assert result["status"] == "passed", result["errors"]


@pytest.mark.parametrize(
    "overrides, column",
    [
        ({"adopted": [0, 2, 1]}, "adopted"),
        ({"adopted": [0, None, 1]}, "adopted"),
        ({"influence_network_size": [-1, 3, 4]}, "influence_network_size"),
        ({"years_experience": [-5, 3, 4]}, "years_experience"),
        ({"geographic_region": ["mars", "west", None]}, "geographic_region"),
    ],
)
def test_a_value_the_table_could_not_hold_fails(overrides: dict, column: str) -> None:
    result = validate_dataframe(_frame(**overrides), VIEW, columns=CONTRACT_COLUMNS)
    assert result["status"] == "failed"
    assert column in {e["column"] for e in result["errors"]}, result["errors"]


def test_an_unscoped_load_is_held_to_the_split_enum() -> None:
    # data_split is outside a contract's projection (the loader adds it and the split
    # consumes it), so only a whole-schema validation reaches this check.
    result = validate_dataframe(_frame(data_split=["train", "bogus", "test"]), VIEW)
    assert result["status"] == "failed"
    assert "data_split" in {e["column"] for e in result["errors"]}, result["errors"]


def test_a_projected_column_missing_from_the_frame_fails() -> None:
    result = validate_dataframe(
        _frame().drop(columns=["specialty"]), VIEW, columns=CONTRACT_COLUMNS
    )
    assert result["status"] == "failed"
