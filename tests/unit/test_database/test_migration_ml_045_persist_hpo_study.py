"""Migration ml/045 content lock (#2207, codex r5): the HPO study + trial set is
persisted by ONE SQL function, i.e. one transaction.

The previous writer issued a study upsert, a per-trial upsert and a stale-trial
delete as separate PostgREST statements and claimed the study was "never left
without its current trials" — which three transactions cannot guarantee. The
function is what makes the claim true: on any error the parent and the child rows
are both unchanged (rehearsed live on 2026-09-22: a duplicate trial_number in the
payload raised and left the previous run's two trials intact).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
M = REPO / "database" / "ml" / "045_persist_hpo_study_rpc.sql"


def _sql() -> str:
    return "\n".join(
        line for line in M.read_text().splitlines() if not line.strip().startswith("--")
    )


@pytest.mark.unit
def test_one_function_upserts_the_study_and_replaces_its_trials():
    s = _sql()
    assert re.search(r"CREATE OR REPLACE FUNCTION public\.persist_hpo_study\(\s*p_study JSONB", s)
    assert "RETURNS UUID" in s
    assert "LANGUAGE plpgsql" in s
    assert "ON CONFLICT (study_name) DO UPDATE SET" in s
    assert "DELETE FROM ml_hpo_trials WHERE study_id = v_study_id" in s
    assert "INSERT INTO ml_hpo_trials" in s
    assert "jsonb_array_elements" in s
    # the delete and the insert are inside the same function body (one transaction)
    body = s.split("AS $$", 1)[1].split("$$;", 1)[0]
    assert "DELETE FROM ml_hpo_trials" in body and "INSERT INTO ml_hpo_trials" in body


@pytest.mark.unit
def test_invoker_security_pinned_search_path_and_service_role_grant():
    s = _sql()
    assert "SECURITY INVOKER" in s
    assert "SET search_path = public" in s
    assert "GRANT EXECUTE ON FUNCTION public.persist_hpo_study(jsonb, jsonb) TO service_role" in s
    assert "SECURITY DEFINER" not in s


@pytest.mark.unit
def test_experiment_id_is_a_nullable_uuid_never_a_label():
    s = _sql()
    assert "NULLIF(p_study->>'experiment_id', '')::uuid" in s
