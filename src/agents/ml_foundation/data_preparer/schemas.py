"""Pydantic v2 schemas for ``data_preparer`` agent outputs.

Formalises the ``qc_report`` ``Dict[str, Any]`` that the agent emits
to downstream consumers (``model_trainer.check_qc_gate``,
``model_selector`` candidate filter). Fields enumerated from
``data_preparer/agent.py::run`` (lines 158-177) where the agent
constructs the qc_report dict from final state.

This module is part of the chore(schemas) scaffolding PR. Shard A
wires the schema into ``DataPreparerState`` after this PR merges.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from src.agents.ml_foundation._pydantic_utils import BaseAgentSchema


class QCReportSchema(BaseAgentSchema):
    """Quality-control report produced by data_preparer.

    Consumed by:

    - ``model_trainer.check_qc_gate`` — reads ``status``,
      ``blocking_issues``, ``overall_score`` to decide whether to
      proceed with training.
    - ``model_selector`` — reads ``overall_score`` and dimension
      scores to filter candidate algorithms.
    - The audit-chain repository (via ``data_preparer.agent._persist_qc_report``)
      maps a subset of these fields into a database record.

    Field-name stability is load-bearing — renaming a key here
    silently breaks downstream consumers. The shape-guard tests in
    ``tests/integration/test_agent_output_contracts.py`` (Shard A
    deliverable post-migration) pin this schema's field set.
    """

    # Identification
    report_id: Optional[str] = None
    experiment_id: Optional[str] = None

    # Top-line status
    status: Optional[Literal["passed", "failed", "warning", "skipped"]] = None
    overall_score: Optional[float] = None

    # Per-dimension scores (5 dimensions per data-quality conventions)
    completeness_score: Optional[float] = None
    validity_score: Optional[float] = None
    consistency_score: Optional[float] = None
    uniqueness_score: Optional[float] = None
    timeliness_score: Optional[float] = None

    # Expectation engine output (Great Expectations)
    expectation_results: Optional[List[Dict[str, Any]]] = None
    failed_expectations: Optional[List[str]] = None
    warnings: Optional[List[Dict[str, Any]]] = None

    # Remediation surface
    remediation_steps: Optional[List[str]] = None
    blocking_issues: Optional[List[str]] = None  # Non-empty → blocks training

    # Tabular metadata
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    validated_at: Optional[str] = None  # ISO 8601 string

    # D2.2: Consumer-contract fields. The QC gate at
    # ``model_trainer/nodes/qc_gate_checker.py:30-46`` and
    # ``model_selector/agent.py:162-166`` read these THREE fields specifically
    # (not the dimension scores or expectation_results above).
    #
    # Pre-D2.2 these fields were NOT in the producer's qc_report dict; the
    # runner ``scripts/run_tier0_test.py:2295-2300, 2558+`` patched them in
    # before forwarding to model_trainer/model_selector. That hidden coupling
    # is removed by D2.2 — the producer now writes them directly.
    #
    # Semantic mapping at the producer (data_preparer/agent.py:158-177):
    # - qc_passed   = final_state["qc_passed"]               (already in state)
    # - qc_errors   = final_state.get("blocking_issues", []) (alias)
    # - qc_warnings = final_state.get("warnings", [])        (alias of warnings field above)
    qc_passed: Optional[bool] = None
    qc_errors: Optional[List[str]] = None
    qc_warnings: Optional[List[Dict[str, Any]]] = None
