-- ============================================================================
-- Migration 033: RAGAS persistence semantics (#1487)
-- ============================================================================
--
-- Comment-only. Adds no structure, changes no data, alters no function body.
--
-- WHY IT EXISTS
-- -------------
-- Migration 022 shipped the RAGAS half of the self-improvement schema with no
-- Python writer, so `learning_signals.ragas_scores` was permanently '{}' and
-- `evaluation_results` was permanently empty. #1487 wired the writers
-- (src/repositories/evaluation_results.py and
-- RubricNode._store_evaluation), and in doing so made a case reachable that
-- had never been possible before: a row that measured only SOME of the five
-- RAGAS metrics.
--
-- That case is normal, not exceptional. The real-pipeline evaluation (#1485)
-- reports only faithfulness and answer_relevancy — context_precision,
-- context_recall and answer_correctness need a ground truth the replay
-- deliberately refuses to fabricate. And a judge that NaNs a metric now
-- reports it as unmeasured rather than 0.0 (#1488).
--
-- Migration 022's SQL helpers predate both facts and COALESCE a missing metric
-- to 0. On a partial row that does not just misreport a number — it feeds
-- determine_improvement_priority(), which routes the improvement work. These
-- comments put the warning where a SQL-side reader will find it.
-- ============================================================================


-- ----------------------------------------------------------------------------
-- learning_signals: the RAGAS half
-- ----------------------------------------------------------------------------

COMMENT ON COLUMN learning_signals.ragas_scores IS
    'Judged RAGAS metrics for this signal, written by RubricNode._store_evaluation. '
    'Contains ONLY metrics the judge actually scored: an absent key means the metric '
    'was not measured, never a judged 0.0. Which metrics were attempted-and-failed '
    'versus never-requested is recorded in signal_details.ragas_coverage. '
    'Empty {} means no RAGAS bundle was available at all.';

COMMENT ON COLUMN learning_signals.ragas_weighted IS
    'Weighted RAGAS aggregate over the MEASURED metrics only, renormalised by the '
    'weight that was measured — NOT the COALESCE-to-zero sum that '
    'calculate_combined_score() and update_learning_signal_evaluation() compute. '
    'A complete five-metric bundle gives the identical value (the weights sum to 1); '
    'a partial one is scored on what was judged instead of being penalised for what '
    'was not. NULL when no metric was measured. Source of truth: '
    'src/agents/feedback_learner/ragas_scoring.py.';

COMMENT ON COLUMN learning_signals.combined_score IS
    'Combined RAGAS + Rubric score: (ragas_weighted * 0.4) + (rubric_normalised * 0.6), '
    'where rubric_normalised = (rubric_total - 1) / 4. Written ONLY when both halves '
    'are real. A rubric-only row leaves this NULL rather than publishing 40%-of-zero '
    'under the name of a measured blend; likewise a RAGAS-only row. Source of truth: '
    'src/agents/feedback_learner/ragas_scoring.py::combined_score.';


-- ----------------------------------------------------------------------------
-- evaluation_results: per-query-response detail
-- ----------------------------------------------------------------------------

COMMENT ON COLUMN evaluation_results.faithfulness IS
    'RAGAS faithfulness (answer vs retrieved contexts). NULL = not measured, '
    'never a judged 0.0.';

COMMENT ON COLUMN evaluation_results.answer_relevancy IS
    'RAGAS answer_relevancy (answer vs query). NULL = not measured, never a judged 0.0. '
    'Note this metric is largely an ABSTENTION RATE: ragas multiplies the score by zero '
    'for any answer its judge calls noncommittal, so a genuine 0.0 usually means the '
    'pipeline declined to answer rather than answered irrelevantly.';

COMMENT ON COLUMN evaluation_results.context_precision IS
    'RAGAS context_precision. NULL = not measured, never a judged 0.0. Producers '
    'without a ground truth (the real-pipeline replay) do not report this at all.';

COMMENT ON COLUMN evaluation_results.context_recall IS
    'RAGAS context_recall. NULL = not measured, never a judged 0.0. Producers '
    'without a ground truth (the real-pipeline replay) do not report this at all.';

COMMENT ON COLUMN evaluation_results.answer_correctness IS
    'RAGAS answer_correctness. NULL = not measured, never a judged 0.0. Requires a '
    'ground_truth reference.';

COMMENT ON COLUMN evaluation_results.ragas_aggregate IS
    'Weighted RAGAS aggregate over the MEASURED metrics only, renormalised by the '
    'measured weight — the same value learning_signals.ragas_weighted carries, from '
    'the same Python source of truth. NULL when no metric was measured. Read it '
    'alongside the individual columns: a NULL metric column tells you what the '
    'aggregate does NOT cover.';

COMMENT ON COLUMN evaluation_results.rubric_aggregate IS
    'Weighted rubric score on the 1-5 scale. The writer refuses rows whose rubric came '
    'from the heuristic fallback (#471 neutral 3.0s), because no column here could '
    'distinguish them afterwards — so every value in this column was LLM-judged.';


-- ----------------------------------------------------------------------------
-- The view built on those columns
-- ----------------------------------------------------------------------------

COMMENT ON VIEW v_ragas_performance_trends IS
    'Daily RAGAS metric trends for monitoring. IMPORTANT: AVG() skips NULLs, and an '
    'unmeasured metric is stored as NULL, so each avg_* column averages over its own '
    'sample count — evaluation_count is the row count, NOT the denominator of every '
    'column. A metric that stops being reported shows as a stable average over a '
    'shrinking sample rather than as a drop. Check per-metric coverage before reading '
    'a trend as a quality change.';


-- ----------------------------------------------------------------------------
-- The helper functions, and the trap in them
-- ----------------------------------------------------------------------------

COMMENT ON FUNCTION calculate_combined_score IS
    'Calculate combined RAGAS + Rubric score with configurable weights. '
    'WARNING (#1487): COALESCEs an absent metric to 0 and an absent rubric_total to the '
    'bottom of the scale, so on a PARTIAL bundle it silently understates the score — a '
    'row measuring only faithfulness and answer_relevancy can never exceed 0.45 here no '
    'matter how good it was. Safe only on a complete five-metric bundle. The writers use '
    'src/agents/feedback_learner/ragas_scoring.py, which renormalises over the measured '
    'weight and returns NULL rather than blending a half that does not exist.';

COMMENT ON FUNCTION update_learning_signal_evaluation IS
    'Update learning signal with evaluation results and determine improvement routing. '
    'WARNING (#1487): inherits the COALESCE-to-zero flaw described on '
    'calculate_combined_score, and then feeds the understated value to '
    'determine_improvement_priority() — so a partial bundle does not merely record a low '
    'score, it routes the improvement work as if the response had been judged and failed. '
    'It has no caller: the Python writers set these columns directly on INSERT. Do not '
    'wire it up for partial bundles without renormalising first.';
