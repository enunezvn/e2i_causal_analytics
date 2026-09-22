"""#2207 follow-up (codex r3 HIGH-1): the manual trigger accepts the loader's file-source
dict as ``data_source``.

``data_loader`` recognises a file source only as ``{"type": "file_dir"|"files", ...}``;
every string is a Supabase table name. ``TriggerRetrainingRequest.data_source`` was
``Optional[str]`` (and its example a bare path), so an operator could not trigger a
file-sourced retrain — the very route meant to heal a registry row's contract.
"""

from __future__ import annotations

import pytest

from src.api.routes.monitoring import TriggerRetrainingRequest


@pytest.mark.unit
def test_file_source_dict_passes_through_the_cohort_contract_unchanged():
    files = {"type": "file_dir", "path": "data/rwd/optum/initiation"}
    req = TriggerRetrainingRequest(
        reason="manual", data_source=files, target_outcome="initiated_biologic_180d"
    )
    assert req.cohort_contract() == {
        "data_source": files,
        "target_outcome": "initiated_biologic_180d",
    }


@pytest.mark.unit
def test_table_name_string_still_accepted():
    req = TriggerRetrainingRequest(reason="manual", data_source="patient_journeys")
    assert req.cohort_contract() == {"data_source": "patient_journeys"}


@pytest.mark.unit
def test_empty_request_stays_permissive_and_none_free():
    req = TriggerRetrainingRequest(reason="data_drift")
    assert req.cohort_contract() == {}


@pytest.mark.unit
def test_schema_example_is_a_file_source_object():
    example = TriggerRetrainingRequest.model_config["json_schema_extra"]["example"]
    assert isinstance(example["data_source"], dict)
    assert example["data_source"]["type"] in ("file_dir", "files")
