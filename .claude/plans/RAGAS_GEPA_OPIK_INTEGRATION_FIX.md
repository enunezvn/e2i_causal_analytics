# RAGAS-GEPA-Opik Integration Fix Plan

**Created**: 2026-01-24
**Status**: In Progress
**Branch**: `claude/check-ragas-opik-integration-3ryR7`

---

## Executive Summary

The RAGAS → GEPA → Opik integration has critical gaps preventing proper evaluation tracing:

| Issue | File | Impact |
|-------|------|--------|
| Missing `get_ragas_evaluator()` factory | `src/rag/evaluation.py` | RAGASFeedbackProvider falls back to mock |
| Wrong parameters in `create_ragas_metric()` call | `src/rag/cognitive_rag_dspy.py:971-974` | Runtime failure in CognitiveRAGOptimizer |
| No bridge between RAGASEvaluator and GEPA | `ragas_feedback.py` | Real RAGAS scores never reach GEPA |

---

## Phase 1: Add Factory Function to evaluation.py

**Goal**: Create `get_ragas_evaluator()` factory function that RAGASFeedbackProvider imports.

### Tasks

- [ ] **1.1** Read current `evaluation.py` exports at bottom of file
- [ ] **1.2** Add `get_ragas_evaluator()` singleton factory function
- [ ] **1.3** Update `__all__` exports to include `get_ragas_evaluator`
- [ ] **1.4** Run unit test on droplet: `pytest tests/rag/test_evaluation.py -v -k "get_ragas" --tb=short`

### Implementation Details

```python
# Add after RAGASEvaluator class (around line 812)

_ragas_evaluator_instance: Optional[RAGASEvaluator] = None

def get_ragas_evaluator(
    config: Optional[EvaluationConfig] = None,
    enable_opik_tracing: bool = True,
) -> RAGASEvaluator:
    """Get or create the singleton RAGASEvaluator instance.

    This factory function provides a consistent evaluator instance
    for use across the codebase, particularly by RAGASFeedbackProvider.

    Args:
        config: Optional evaluation configuration
        enable_opik_tracing: Whether to enable Opik tracing

    Returns:
        RAGASEvaluator singleton instance
    """
    global _ragas_evaluator_instance
    if _ragas_evaluator_instance is None:
        _ragas_evaluator_instance = RAGASEvaluator(
            config=config,
            enable_opik_tracing=enable_opik_tracing,
        )
    return _ragas_evaluator_instance
```

### Verification

```bash
# On droplet
ssh -i ~/.ssh/replit enunez@138.197.4.36 \
  "cd /opt/e2i_causal_analytics && \
   /opt/e2i_causal_analytics/venv/bin/python -c \"from src.rag.evaluation import get_ragas_evaluator; print('OK')\""
```

---

## Phase 2: Fix RAGASFeedbackProvider Integration

**Goal**: Update RAGASFeedbackProvider to properly use the RAGASEvaluator.

### Tasks

- [ ] **2.1** Read `ragas_feedback.py` evaluate method (lines 199-273)
- [ ] **2.2** Update `__post_init__` to use `get_ragas_evaluator()` correctly
- [ ] **2.3** Update `evaluate()` method to call evaluator's `evaluate_sample()` async method
- [ ] **2.4** Run unit test: `pytest tests/integration/test_gepa_integration.py -v -k "ragas_feedback" --tb=short`

### Implementation Details

The current code at line 112 does:
```python
from src.rag.evaluation import get_ragas_evaluator
self._ragas_evaluator = get_ragas_evaluator()
```

But the evaluator's method is `evaluate_sample(sample: EvaluationSample)`, so we need to adapt the interface:

```python
# In evaluate() method, update lines 220-234 to:
if self._ragas_evaluator:
    from src.rag.evaluation import EvaluationSample
    sample = EvaluationSample(
        query=question,
        ground_truth=ground_truth or answer,
        answer=answer,
        retrieved_contexts=contexts,
    )
    result = await self._ragas_evaluator.evaluate_sample(sample)
    scores = {
        "faithfulness": result.faithfulness or 0.0,
        "answer_relevancy": result.answer_relevancy or 0.0,
        "context_precision": result.context_precision or 0.0,
        "context_recall": result.context_recall or 0.0,
    }
```

---

## Phase 3: Fix CognitiveRAGOptimizer GEPA Integration

**Goal**: Fix the `create_ragas_metric()` call with correct parameters.

### Tasks

- [ ] **3.1** Read `cognitive_rag_dspy.py` lines 960-1010
- [ ] **3.2** Fix `create_ragas_metric()` call at lines 971-974
- [ ] **3.3** Run unit test: `pytest tests/rag/test_cognitive_rag_dspy.py -v --tb=short`

### Current (Broken) Code

```python
# Lines 971-974 in cognitive_rag_dspy.py
ragas_metric = create_ragas_metric(
    phase=phase,                              # ❌ Does not exist
    fallback_metric=self._get_phase_metric(phase),  # ❌ Does not exist
)
```

### Fixed Code

```python
# Use correct signature: create_ragas_metric(provider=None, agent_name="cognitive_rag", weights=None)
ragas_metric = create_ragas_metric(
    agent_name=f"cognitive_rag_{phase}",
    weights=self._get_phase_weights(phase),  # Optional: phase-specific weights
)
```

Also need to add a helper method or use default weights:

```python
def _get_phase_weights(self, phase: str) -> dict[str, float]:
    """Get RAGAS metric weights for a specific phase."""
    weights = {
        "summarizer": {"faithfulness": 0.2, "answer_relevancy": 0.4, "context_precision": 0.2, "context_recall": 0.2},
        "investigator": {"faithfulness": 0.3, "answer_relevancy": 0.2, "context_precision": 0.3, "context_recall": 0.2},
        "agent": {"faithfulness": 0.3, "answer_relevancy": 0.3, "context_precision": 0.2, "context_recall": 0.2},
    }
    return weights.get(phase, None)  # None uses default equal weights
```

---

## Phase 4: Add Opik Tracing to RAGASFeedbackProvider

**Goal**: Ensure RAGAS evaluations in GEPA are traced to Opik.

### Tasks

- [ ] **4.1** Check if RAGASEvaluator already traces to Opik (it does via `_evaluate_with_tracing`)
- [ ] **4.2** Verify OpikEvaluationTracer integration in `evaluation.py`
- [ ] **4.3** Add optional trace_id parameter to RAGASFeedbackProvider.evaluate()
- [ ] **4.4** Integration test: `pytest tests/integration/test_gepa_opik_integration.py -v --tb=short`

### Implementation

```python
# In ragas_feedback.py, update evaluate() signature
async def evaluate(
    self,
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: Optional[str] = None,
    run_id: Optional[str] = None,  # Add for Opik tracing
    **kwargs: Any,
) -> ScoreWithFeedback:
```

Then pass run_id to evaluator:
```python
result = await self._ragas_evaluator.evaluate_sample(sample, run_id=run_id)
```

---

## Phase 5: End-to-End Integration Test

**Goal**: Verify the full RAGAS → GEPA → Opik pipeline works.

### Tasks

- [ ] **5.1** Create integration test for full pipeline
- [ ] **5.2** Run on droplet with Opik service
- [ ] **5.3** Verify traces appear in Opik UI at http://138.197.4.36:5173

### Test Script

```python
# tests/integration/test_ragas_gepa_opik_e2e.py
import pytest
from src.optimization.gepa.integration.ragas_feedback import (
    RAGASFeedbackProvider,
    create_ragas_metric,
)

@pytest.mark.asyncio
async def test_ragas_gepa_opik_integration():
    """Test that RAGAS scores flow through GEPA to Opik."""
    # Create provider
    provider = RAGASFeedbackProvider()

    # Evaluate a sample
    result = await provider.evaluate(
        question="What is the TRx trend for Kisqali?",
        answer="Kisqali TRx increased 15% in Q4.",
        contexts=["Q4 report shows 15% TRx growth."],
        ground_truth="Kisqali TRx grew 15% in Q4.",
        run_id="test_ragas_gepa_opik_e2e",
    )

    # Verify result structure
    assert "score" in result
    assert "feedback" in result
    assert isinstance(result["score"], float)
    assert 0.0 <= result["score"] <= 1.0

    # Verify provider is using real RAGAS (not mock)
    assert provider.enabled, "RAGAS evaluator should be enabled"
```

---

## Phase 6: Commit and Push

### Tasks

- [ ] **6.1** Run full test suite: `make test-fast`
- [ ] **6.2** Commit changes with descriptive message
- [ ] **6.3** Push to branch `claude/check-ragas-opik-integration-3ryR7`

---

## File Change Summary

| File | Changes |
|------|---------|
| `src/rag/evaluation.py` | Add `get_ragas_evaluator()` factory, update exports |
| `src/optimization/gepa/integration/ragas_feedback.py` | Fix evaluator integration, add run_id tracing |
| `src/rag/cognitive_rag_dspy.py` | Fix `create_ragas_metric()` call parameters |
| `tests/integration/test_ragas_gepa_opik_e2e.py` | New E2E test |

---

## Testing Strategy

All tests run on droplet in small batches:

```bash
# Phase 1 test
pytest tests/rag/test_evaluation.py -v -k "get_ragas or factory" --tb=short -n 2

# Phase 2 test
pytest tests/integration/test_gepa_integration.py -v -k "ragas" --tb=short -n 2

# Phase 3 test
pytest tests/rag/test_cognitive_rag_dspy.py -v --tb=short -n 2

# Phase 5 E2E test
pytest tests/integration/test_ragas_gepa_opik_e2e.py -v --tb=short

# Full suite (memory-safe)
make test-fast
```

---

## Progress Tracking

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1 | ✅ Complete | Added `get_ragas_evaluator()` factory to evaluation.py |
| Phase 2 | ✅ Complete | Fixed RAGASFeedbackProvider to use EvaluationSample |
| Phase 3 | ✅ Complete | Fixed create_ragas_metric() call, added _get_phase_weights() |
| Phase 4 | ✅ Complete | Added run_id parameter for Opik tracing |
| Phase 5 | ✅ Complete | Added TestRAGASGEPAOpikIntegration class |
| Phase 6 | 🔄 In Progress | Running tests, will commit and push |

---

## Rollback Plan

If issues arise:
1. Revert changes: `git checkout HEAD~1 -- <file>`
2. Keep mock evaluation as fallback (already exists)
3. Feature flag: `ENABLE_REAL_RAGAS=true` environment variable

---

## Dependencies

- RAGAS library (in requirements-ragas.txt)
- Opik running on droplet (port 5173/8080)
- OpenAI API key for RAGAS evaluation (uses gpt-4o-mini)
