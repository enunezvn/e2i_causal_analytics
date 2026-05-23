"""Cross-library validation module — retired surface, namespace stub.

This package previously hosted the "B8 Validation Loop" core classes
introduced in commit ``26ce1fff`` (2025-12-29) as part of the Phase
B4-B10 causal inference expansion. The full surface has been retired
in two phases via the orphan-audit framework:

Phase 1 (#457, PR #463, merged 2026-05-23, commit ``e2156ad5``):
  * ``ConfidenceScorer`` (was at ``validation/confidence_scorer.py``)
  * ``CrossValidator`` (was at ``validation/cross_validator.py``)
  * ``ABReconciler`` (was at ``validation/ab_reconciler.py``)
  * ``compute_pipeline_confidence`` helper

The consensus-weighting and pairwise-agreement responsibilities that
these classes would have provided now live inline in the pipeline at
``src.causal_engine.pipeline.sequential._apply_consensus`` and
``_apply_pairwise_agreement``.

Phase 2 (#464, this PR):
  * ``ValidationReportGenerator`` (was at ``validation/report_generator.py``)
  * 9 TypedDict envelopes (was at ``validation/state.py``):
    ``LibraryEffectEstimate``, ``PairwiseValidation``,
    ``RefutationValidation``, ``ValidationSummary``,
    ``CrossValidationResult``, ``ABExperimentResult``,
    ``ABReconciliationResult``, ``ValidationReportSection``,
    ``ValidationReport``.

The Phase 2 retire reflects: zero ``src/`` consumers of the generator;
zero ``src/`` producers of its input envelopes after the Phase 1 retire;
zero concrete WIRE workstreams across gh issues, gh PRs, non-archived
``.claude/plans/``, TODOs, and ``git log --all --grep``. Full evidence
at ``.claude/research/464_validation_report_generator_audit.md``
(iter-0 RETIRE-WITH-TYPEDDICTS; codex iter-1 ACCEPT).

Reversibility (Phase 2): full restoration ~30 min via
``git restore --source=735465ce src/causal_engine/validation/report_generator.py``
``git restore --source=735465ce src/causal_engine/validation/state.py``
``git restore --source=735465ce tests/unit/test_causal_engine/test_validation/test_report_generator.py``
plus restoring this ``__init__.py``'s prior imports/__all__.

This stub remains as a namespace placeholder to keep
``import src.causal_engine.validation`` working without raising
``ModuleNotFoundError`` — preserves the package surface for the
drift-guard tests at ``tests/integration/test_validation_package.py``.
"""

__all__: list[str] = []
