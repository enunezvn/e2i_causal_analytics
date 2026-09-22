"""ABResultsRepository.save_results under migration ml/046 (#2206, owner fix).

The partial unique index ``uq_ab_results_one_final_per_experiment`` makes "one
FINAL row per experiment" atomic at the database. PostgREST cannot emit an
``ON CONFLICT`` that matches a partial index (measured: "there is no unique or
exclusion constraint matching the ON CONFLICT specification"), so the writer
keeps a plain INSERT and classifies the loser's 23505: on a FINAL row it is
"another delivery already persisted the final row" and surfaces as
``FinalResultAlreadyPersisted`` carrying the winner's row; on any other row (an
interim, whose repeats are legitimate history and never hit this index) the
``APIError`` propagates exactly as before.

The raised error is the real ``postgrest.exceptions.APIError`` shape (a dict
with message/code/hint/details, ``code`` a string), read from the installed
package — not a guess.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from postgrest.exceptions import APIError

from src.repositories.ab_results import ABResultsRepository, FinalResultAlreadyPersisted
from src.services.results_analysis import AnalysisMethod, AnalysisType, ExperimentResults


def _results(analysis_type: AnalysisType, experiment_id=None) -> ExperimentResults:
    return ExperimentResults(
        experiment_id=experiment_id or uuid4(),
        analysis_type=analysis_type,
        analysis_method=AnalysisMethod.ITT,
        computed_at=datetime.now(timezone.utc),
        primary_metric="trx_count",
        control_mean=1.0,
        treatment_mean=1.2,
        effect_estimate=0.2,
        effect_ci_lower=0.1,
        effect_ci_upper=0.3,
        relative_lift=20.0,
        relative_lift_ci_lower=10.0,
        relative_lift_ci_upper=30.0,
        p_value=0.01,
        is_significant=True,
        sample_size_control=50,
        sample_size_treatment=50,
        statistical_power=0.8,
    )


def _unique_violation(experiment_id) -> APIError:
    """The exact PostgREST error body for the ml/046 index (rehearsed live)."""
    return APIError(
        {
            "message": (
                "duplicate key value violates unique constraint "
                '"uq_ab_results_one_final_per_experiment"'
            ),
            "code": "23505",
            "hint": None,
            "details": f"Key (experiment_id)=({experiment_id}) already exists.",
        }
    )


def _winner_row(experiment_id) -> dict:
    return {
        "id": str(uuid4()),
        "experiment_id": str(experiment_id),
        "analysis_type": "final",
        "analysis_method": "itt",
        "computed_at": "2026-09-22T10:00:00Z",
        "primary_metric": "trx_count",
        "control_mean": 1.0,
        "treatment_mean": 1.15,
        "effect_estimate": 0.15,
        "effect_ci_lower": 0.05,
        "effect_ci_upper": 0.25,
        "p_value": 0.02,
        "control_n": 50,
        "treatment_n": 50,
        "observed_power": 0.75,
        "is_significant": True,
    }


def _client(insert_error: Exception | None, select_rows: list[dict]):
    """A supabase client whose INSERT raises ``insert_error`` (when given) and whose
    SELECT chain returns ``select_rows`` — the re-read after a lost race."""
    chain = MagicMock()
    for name in ("select", "eq", "order", "limit"):
        getattr(chain, name).return_value = chain
    chain.execute.return_value = MagicMock(data=select_rows)
    if insert_error is not None:
        chain.insert.return_value.execute.side_effect = insert_error
    else:
        chain.insert.return_value.execute.return_value = MagicMock(data=select_rows)
    client = MagicMock()
    client.table.return_value = chain
    return client, chain


@pytest.mark.asyncio
async def test_a_final_insert_losing_the_race_raises_with_the_winner_row():
    experiment_id = uuid4()
    winner = _winner_row(experiment_id)
    client, chain = _client(_unique_violation(experiment_id), [winner])
    repo = ABResultsRepository(supabase_client=client)

    with pytest.raises(FinalResultAlreadyPersisted) as raised:
        await repo.save_results(_results(AnalysisType.FINAL, experiment_id))

    assert raised.value.experiment_id == experiment_id
    assert raised.value.existing is not None
    assert str(raised.value.existing.id) == winner["id"]
    assert raised.value.existing.analysis_type == "final"
    # The re-read is scoped to this experiment's FINAL row.
    chain.eq.assert_any_call("experiment_id", str(experiment_id))
    chain.eq.assert_any_call("analysis_type", "final")
    # And the original database error travels with it for the log.
    assert isinstance(raised.value.__cause__, APIError)
    assert raised.value.__cause__.code == "23505"


@pytest.mark.asyncio
async def test_a_final_insert_losing_the_race_reads_the_winner_regardless_of_provenance():
    """The winner is whatever row holds the slot; the re-read must not hide it
    behind the real-mode provenance filter (``is_synthetic = false``)."""
    experiment_id = uuid4()
    client, chain = _client(_unique_violation(experiment_id), [_winner_row(experiment_id)])
    repo = ABResultsRepository(supabase_client=client)

    with pytest.raises(FinalResultAlreadyPersisted):
        await repo.save_results(_results(AnalysisType.FINAL, experiment_id))

    eq_calls = [c.args for c in chain.eq.call_args_list]
    assert ("is_synthetic", False) not in eq_calls


@pytest.mark.asyncio
async def test_an_interim_unique_violation_propagates_unchanged():
    """Interim repeats are legitimate history and never hit the partial index; a
    23505 on an interim insert is some other constraint and must propagate as
    before, never be reinterpreted as a lost race."""
    experiment_id = uuid4()
    client, chain = _client(_unique_violation(experiment_id), [_winner_row(experiment_id)])
    repo = ABResultsRepository(supabase_client=client)

    with pytest.raises(APIError) as raised:
        await repo.save_results(_results(AnalysisType.INTERIM, experiment_id))

    assert raised.value.code == "23505"
    chain.select.assert_not_called()


@pytest.mark.asyncio
async def test_a_final_insert_failing_for_another_reason_propagates_unchanged():
    """Only a unique violation is a lost race; an FK failure (23503) or anything
    else on a FINAL insert is still an error."""
    experiment_id = uuid4()
    fk_error = APIError(
        {
            "message": 'insert or update on table "ab_experiment_results" violates foreign key '
            'constraint "ab_experiment_results_experiment_id_fkey"',
            "code": "23503",
            "hint": None,
            "details": f"Key (experiment_id)=({experiment_id}) is not present in table "
            '"ml_experiments".',
        }
    )
    client, chain = _client(fk_error, [])
    repo = ABResultsRepository(supabase_client=client)

    with pytest.raises(APIError) as raised:
        await repo.save_results(_results(AnalysisType.FINAL, experiment_id))

    assert raised.value.code == "23503"
    chain.select.assert_not_called()


@pytest.mark.asyncio
async def test_a_final_insert_that_wins_returns_the_created_record():
    """Positive control: the happy path is untouched — plain INSERT, record back."""
    experiment_id = uuid4()
    winner = _winner_row(experiment_id)
    client, chain = _client(None, [winner])
    repo = ABResultsRepository(supabase_client=client)

    record = await repo.save_results(_results(AnalysisType.FINAL, experiment_id))

    assert str(record.id) == winner["id"]
    chain.insert.assert_called_once()
    assert not chain.upsert.called


@pytest.mark.asyncio
async def test_a_final_23505_on_another_constraint_is_not_a_lost_race():
    """codex r1 #7: only a violation of uq_ab_results_one_final_per_experiment is the
    redelivery race; a pkey (or any other unique) collision on a FINAL insert propagates."""
    experiment_id = uuid4()
    pkey_error = APIError(
        {
            "message": 'duplicate key value violates unique constraint "ab_experiment_results_pkey"',
            "code": "23505",
            "hint": None,
            "details": "Key (id)=(…) already exists.",
        }
    )
    client, chain = _client(pkey_error, [_winner_row(experiment_id)])
    repo = ABResultsRepository(supabase_client=client)

    with pytest.raises(APIError) as raised:
        await repo.save_results(_results(AnalysisType.FINAL, experiment_id))
    assert raised.value.code == "23505"
    chain.select.assert_not_called()


@pytest.mark.asyncio
async def test_a_matching_23505_whose_reread_finds_no_winner_propagates():
    """A committed winner is always visible to the re-read (PostgREST commits per
    request); finding none means this was not the race — surface the original error."""
    experiment_id = uuid4()
    client, _chain = _client(_unique_violation(experiment_id), [])
    repo = ABResultsRepository(supabase_client=client)

    with pytest.raises(APIError) as raised:
        await repo.save_results(_results(AnalysisType.FINAL, experiment_id))
    assert raised.value.code == "23505"
