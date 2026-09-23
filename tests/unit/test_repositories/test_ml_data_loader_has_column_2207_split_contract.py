"""#2207 split contract: ``MLDataLoader.has_column`` — the one place PostgREST's
"undefined column" (SQLSTATE 42703) is told apart from every other failure.

Codex r1 MED on PR #2241: ``data_loader``'s ``data_split`` presence probe went through
``load_table_sample``, which swallows EVERY error into an empty frame, so a transport /
auth / timeout failure was indistinguishable from "the column is absent" and could route
a table that DOES carry ``data_split`` down the holdout-less temporal path. Here a real
42703 is ``False``; anything else raises (fail closed); no client raises.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from postgrest.exceptions import APIError

from src.repositories.ml_data_loader import MLDataLoader


def _client(execute_side_effect=None, data=None) -> MagicMock:
    client = MagicMock()
    chain = client.table.return_value.select.return_value.limit.return_value
    if execute_side_effect is not None:
        chain.execute.side_effect = execute_side_effect
    else:
        chain.execute.return_value = MagicMock(data=data if data is not None else [])
    return client


def _undefined_column() -> APIError:
    return APIError(
        {
            "code": "42703",
            "message": "column patient_journeys.data_split does not exist",
            "details": None,
            "hint": None,
        }
    )


@pytest.mark.asyncio
async def test_present_column_is_true_even_when_no_rows() -> None:
    client = _client(data=[])
    loader = MLDataLoader(client)
    assert await loader.has_column("patient_journeys", "data_split") is True
    client.table.assert_called_once_with("patient_journeys")
    client.table.return_value.select.assert_called_once_with("data_split")
    client.table.return_value.select.return_value.limit.assert_called_once_with(1)


@pytest.mark.asyncio
async def test_undefined_column_42703_is_false() -> None:
    loader = MLDataLoader(_client(execute_side_effect=_undefined_column()))
    assert await loader.has_column("patient_journeys", "data_split") is False


@pytest.mark.asyncio
async def test_other_api_error_raises() -> None:
    err = APIError({"code": "PGRST301", "message": "JWT expired", "details": None, "hint": None})
    loader = MLDataLoader(_client(execute_side_effect=err))
    with pytest.raises(APIError):
        await loader.has_column("patient_journeys", "data_split")


@pytest.mark.asyncio
async def test_transport_error_raises() -> None:
    loader = MLDataLoader(_client(execute_side_effect=ConnectionError("boom")))
    with pytest.raises(ConnectionError):
        await loader.has_column("patient_journeys", "data_split")


@pytest.mark.asyncio
async def test_no_client_raises_instead_of_guessing() -> None:
    loader = MLDataLoader(MagicMock())
    loader.client = None
    with pytest.raises(RuntimeError, match="Supabase client"):
        await loader.has_column("patient_journeys", "data_split")


@pytest.mark.asyncio
async def test_table_allowlist_enforced() -> None:
    loader = MLDataLoader(_client())
    with pytest.raises(ValueError, match="not supported"):
        await loader.has_column("hcp_brand_adoption", "data_split")
