-- ============================================================================
-- Migration 034: score partial RAGAS bundles honestly (#1489 deferral 4)
-- ============================================================================
--
-- Function bodies only. Adds no table, alters no column, changes no row.
--
-- WHY IT EXISTS
-- -------------
-- Migration 022 shipped calculate_combined_score() and
-- update_learning_signal_evaluation() under two assumptions that were true
-- when they landed and are false now:
--
--   1. every RAGAS bundle carries all five metrics, so COALESCE(metric, 0) is
--      harmless; and
--   2. nothing else on the row describes the bundle, so replacing the score
--      columns leaves the row self-consistent.
--
-- #1485 made a PARTIAL bundle the normal shape — the real-pipeline replay
-- reports only faithfulness and answer_relevancy, because the other three need
-- a ground truth it deliberately refuses to fabricate. #1488 made an unmeasured
-- metric a NULL rather than a judged 0.0. #1487 added
-- signal_details.ragas_coverage, which records WHICH metrics a row's bundle
-- holds.
--
-- Migration 033 documented both traps in COMMENTs and left the arithmetic in
-- place. A comment cannot stop a caller: #1489's remaining work wires a
-- per-turn RAGAS producer, and this is the SQL-side entry point it would reach
-- for. This migration makes the functions correct instead of merely annotated.
--
-- WHAT WAS MEASURED (PostgreSQL 15.8, throwaway database, before this change)
-- --------------------------------------------------------------------------
--   calculate_combined_score('{"faithfulness":1.0,"answer_relevancy":1.0}', 5.0)
--       -> 0.78   a PERFECT partial row; ragas_scoring.py gives 1.0
--   calculate_combined_score(<all five 1.0>, NULL)
--       -> 0.40   40%-of-nothing published as a two-half blend
--   calculate_combined_score('{}', 5.0)
--       -> 0.60   zero RAGAS measurement, still routes improvement work
--   determine_improvement_priority(NULL)     -> 'critical'
--   determine_improvement_type(NULL, 0.75)   -> 'workflow'
--
-- At the measured #1485 baseline (faithfulness 0.524, answer_relevancy 0.179,
-- rubric 4.0) the COALESCE moved the priority from 'medium' to 'high'; and a
-- bundle whose every judged metric was 1.0 scored 0.45, landing under the 0.7
-- retrieval threshold, so the row was routed to 'retrieval' — tune k, chunks,
-- RRF weights — for a retrieval that had served it perfectly.
--
-- The stale half, also measured: calling update_learning_signal_evaluation on
-- a row the #1487 Python writer had inserted left signal_value at the OLD
-- rubric total (4.0 while rubric_total became 2.0) and left
-- ragas_coverage.measured naming two metrics after the stored bundle had been
-- replaced by a one-metric one.
--
-- WHAT THIS MIGRATION DOES
-- ------------------------
--   * renormalises over the weight actually MEASURED, so a partial bundle is
--     scored on what was judged instead of penalised for what was not;
--   * returns NULL rather than publishing a blend of a half that does not
--     exist (no RAGAS metric measured, or no rubric total);
--   * makes the routing helpers NULL-safe, so an absent measurement yields an
--     absent verdict instead of the worst one in the vocabulary;
--   * refreshes the companion components the update used to leave behind.
--
-- Behaviour on a COMPLETE five-metric bundle is unchanged: the weights sum to
-- 1.0, so renormalising divides by 1.0.
--
-- SOURCE OF TRUTH
-- ---------------
-- src/agents/feedback_learner/ragas_scoring.py is the Python implementation of
-- this same blend and the writers use it. These definitions are now its exact
-- SQL counterpart; tests parse the weights out of this file, and
-- tests/integration/test_combined_score_sql_semantics_1489.py executes both
-- against real PostgreSQL and asserts they agree.
-- ============================================================================


-- ----------------------------------------------------------------------------
-- The weights, defined ONCE
-- ----------------------------------------------------------------------------
-- 022 copy-pasted the weighted-sum expression into both calculate_combined_score
-- and update_learning_signal_evaluation, so a future weight change had two
-- places to miss and no test could see them diverge. Everything below selects
-- from this one table.

CREATE OR REPLACE FUNCTION ragas_metric_weights()
RETURNS TABLE (metric TEXT, weight FLOAT) AS $$
    SELECT * FROM (VALUES
        ('faithfulness',       0.25),
        ('answer_relevancy',   0.20),
        ('context_precision',  0.20),
        ('context_recall',     0.20),
        ('answer_correctness', 0.15)
    ) AS w(metric, weight);
$$ LANGUAGE sql IMMUTABLE;

COMMENT ON FUNCTION ragas_metric_weights IS
    'Per-metric RAGAS weights, the single SQL-side definition (#1489). They sum '
    'to 1.0, so a complete bundle needs no renormalising. Mirrored by '
    'RAGAS_METRIC_WEIGHTS in src/agents/feedback_learner/ragas_scoring.py; a '
    'unit test parses them out of this migration so the two cannot drift.';


-- ----------------------------------------------------------------------------
-- Weighted RAGAS aggregate over the MEASURED metrics only
-- ----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION ragas_weighted_measured(p_ragas_scores JSONB)
RETURNS FLOAT AS $$
DECLARE
    v_weighted_sum   FLOAT := 0;
    v_measured_weight FLOAT := 0;
    v_metric TEXT;
    v_weight FLOAT;
    v_value  FLOAT;
BEGIN
    IF p_ragas_scores IS NULL THEN
        RETURN NULL;
    END IF;

    FOR v_metric, v_weight IN SELECT metric, weight FROM ragas_metric_weights() LOOP
        -- ->> yields SQL NULL both for an ABSENT key (#1485: this producer
        -- never asks for that metric) and for a JSON null (#1488: the judge
        -- tried and failed). Neither is a judged score, so neither contributes
        -- to the numerator OR the denominator.
        v_value := NULLIF(p_ragas_scores->>v_metric, '')::float;
        IF v_value IS NOT NULL THEN
            v_weighted_sum    := v_weighted_sum + (v_value * v_weight);
            v_measured_weight := v_measured_weight + v_weight;
        END IF;
    END LOOP;

    IF v_measured_weight = 0 THEN
        RETURN NULL;  -- nothing was measured; 0.0 would be a fabricated score
    END IF;

    RETURN v_weighted_sum / v_measured_weight;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION ragas_weighted_measured IS
    'Weighted RAGAS aggregate over the MEASURED metrics only, renormalised by '
    'the weight that was measured (#1489). An absent key and a null-valued key '
    'both mean unmeasured and are excluded from numerator and denominator '
    'alike. NULL when no metric was measured. Identical to a plain weighted sum '
    'on a complete five-metric bundle. Python counterpart: RagasBundle.weighted.';


-- ----------------------------------------------------------------------------
-- The blend
-- ----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION calculate_combined_score(
    p_ragas_scores JSONB,
    p_rubric_total FLOAT,
    p_ragas_weight FLOAT DEFAULT 0.4,
    p_rubric_weight FLOAT DEFAULT 0.6
) RETURNS FLOAT AS $$
DECLARE
    v_ragas_weighted    FLOAT;
    v_rubric_normalized FLOAT;
BEGIN
    v_ragas_weighted := ragas_weighted_measured(p_ragas_scores);

    -- combined_score names a blend of two halves. Publishing it when one half
    -- was never measured puts a number no reader can distinguish from a
    -- measurement into the column that routes the improvement work. 022
    -- returned 0.40 for a perfect RAGAS half with no rubric, and 0.60 for a
    -- perfect rubric with no RAGAS at all.
    IF v_ragas_weighted IS NULL OR p_rubric_total IS NULL THEN
        RETURN NULL;
    END IF;

    v_rubric_normalized := (p_rubric_total - 1) / 4.0;

    RETURN ROUND(
        ((v_ragas_weighted * p_ragas_weight) + (v_rubric_normalized * p_rubric_weight))::numeric,
        4
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION calculate_combined_score IS
    'Combined RAGAS + Rubric score: (ragas_weighted * 0.4) + (rubric_normalised * 0.6), '
    'where ragas_weighted renormalises over the MEASURED metrics (#1489, superseding '
    'the COALESCE-to-zero behaviour migration 033 warned about). Returns NULL when '
    'either half is absent rather than blending a half that does not exist. Agrees '
    'exactly with src/agents/feedback_learner/ragas_scoring.py::combined_score.';


-- ----------------------------------------------------------------------------
-- Routing, made NULL-safe
-- ----------------------------------------------------------------------------
-- Both helpers used to fall through to their ELSE branch on a NULL input and
-- return a verdict — 'critical' and 'workflow' respectively — that no
-- measurement supported. Now that the blend can legitimately be NULL, that
-- fall-through would fire routinely.

CREATE OR REPLACE FUNCTION determine_improvement_type(
    p_ragas_weighted FLOAT,
    p_rubric_normalized FLOAT,
    p_ragas_threshold FLOAT DEFAULT 0.7,
    p_rubric_threshold FLOAT DEFAULT 0.7
) RETURNS improvement_type AS $$
BEGIN
    IF p_ragas_weighted IS NULL OR p_rubric_normalized IS NULL THEN
        RETURN NULL;
    ELSIF p_ragas_weighted >= p_ragas_threshold AND p_rubric_normalized >= p_rubric_threshold THEN
        RETURN 'none';
    ELSIF p_ragas_weighted < p_ragas_threshold AND p_rubric_normalized >= p_rubric_threshold THEN
        RETURN 'retrieval';
    ELSIF p_ragas_weighted >= p_ragas_threshold AND p_rubric_normalized < p_rubric_threshold THEN
        RETURN 'prompt';
    ELSE
        RETURN 'workflow';
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION determine_improvement_type IS
    'Route to an improvement action from the RAGAS vs Rubric halves. NULL in, NULL '
    'out (#1489): before, a NULL half fell through to ''workflow''. Thresholds and '
    'branch boundaries are unchanged for measured inputs.';


CREATE OR REPLACE FUNCTION determine_improvement_priority(
    p_combined_score FLOAT
) RETURNS improvement_priority AS $$
BEGIN
    IF p_combined_score IS NULL THEN
        RETURN NULL;
    ELSIF p_combined_score >= 0.85 THEN
        RETURN 'none';
    ELSIF p_combined_score >= 0.70 THEN
        RETURN 'low';
    ELSIF p_combined_score >= 0.55 THEN
        RETURN 'medium';
    ELSIF p_combined_score >= 0.40 THEN
        RETURN 'high';
    ELSE
        RETURN 'critical';
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION determine_improvement_priority IS
    'Improvement priority from the combined score. NULL in, NULL out (#1489): before, '
    'an unmeasurable score returned ''critical'' — the worst verdict in the '
    'vocabulary, fabricated from an absent measurement. Bands are unchanged.';


-- ----------------------------------------------------------------------------
-- The update, with no components left behind
-- ----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION update_learning_signal_evaluation(
    p_signal_id UUID,
    p_ragas_scores JSONB,
    p_rubric_scores JSONB,
    p_rubric_total FLOAT
) RETURNS VOID AS $$
DECLARE
    v_ragas_weighted    FLOAT;
    v_rubric_normalized FLOAT;
    v_combined_score    FLOAT;
    v_measured          TEXT[];
BEGIN
    v_ragas_weighted := ragas_weighted_measured(p_ragas_scores);
    v_rubric_normalized := CASE
        WHEN p_rubric_total IS NULL THEN NULL
        ELSE (p_rubric_total - 1) / 4.0
    END;
    v_combined_score := calculate_combined_score(p_ragas_scores, p_rubric_total);

    -- Which metrics the bundle being STORED actually carries. Restricted to the
    -- weighted metrics, so coverage can never claim a metric that contributed
    -- nothing to ragas_weighted.
    SELECT COALESCE(array_agg(w.metric ORDER BY w.metric), ARRAY[]::text[])
      INTO v_measured
      FROM ragas_metric_weights() AS w
     WHERE NULLIF(COALESCE(p_ragas_scores, '{}'::jsonb)->>w.metric, '') IS NOT NULL;

    UPDATE learning_signals SET
        ragas_scores = COALESCE(p_ragas_scores, '{}'::jsonb),
        ragas_weighted = v_ragas_weighted,
        rubric_scores = COALESCE(p_rubric_scores, '{}'::jsonb),
        rubric_total = p_rubric_total,
        rubric_weighted = p_rubric_total,
        combined_score = v_combined_score,
        -- signal_value carried the rubric score at INSERT (rubric_node.py sets
        -- signal_value = evaluation.weighted_score). Updating rubric_total and
        -- not this left the row asserting two different rubric scores.
        signal_value = p_rubric_total,
        improvement_type = determine_improvement_type(v_ragas_weighted, v_rubric_normalized),
        improvement_priority = determine_improvement_priority(v_combined_score),
        -- ragas_coverage described the bundle this statement is REPLACING.
        -- Recomputed from the incoming bundle so it cannot name metrics the row
        -- no longer holds. The judge-provenance keys the Python writer records
        -- (evaluation_model, evaluation_method) are deliberately not carried
        -- over: this function was not told them, and keeping the old ones would
        -- attribute the new scores to the previous judge.
        signal_details = jsonb_set(
            COALESCE(signal_details, '{}'::jsonb),
            '{ragas_coverage}',
            jsonb_build_object(
                'measured', to_jsonb(v_measured),
                'source', 'update_learning_signal_evaluation'
            ),
            true
        ),
        processed_at = now()
    WHERE signal_id = p_signal_id;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION update_learning_signal_evaluation IS
    'Update a learning signal with evaluation results and route the improvement work. '
    '#1489 replaced the COALESCE-to-zero arithmetic migration 033 warned about with '
    'ragas_weighted_measured(), so a partial bundle is scored on what was judged; a '
    'row missing either half now stores NULL for combined_score and NULL routing '
    'rather than an understated score and a confident verdict. It also refreshes the '
    'companion components it used to leave stale — signal_value (which carried the '
    'rubric score from INSERT) and signal_details.ragas_coverage (which described the '
    'replaced bundle).';
