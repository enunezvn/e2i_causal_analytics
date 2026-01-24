# Cognitive Workflow Resilience Implementation Plan

**Created**: 2026-01-24
**Completed**: 2026-01-24
**Branch**: `claude/evaluate-cognitive-workflow-OQTpx`
**Status**: Complete

---

## Overview

This plan addresses four identified concerns in the cognitive workflow:

| # | Concern | Priority | Complexity | Status |
|---|---------|----------|------------|--------|
| 1 | Embedding Service SPOF | P1 | Medium | Complete |
| 2 | Untracked Async Reflector Tasks | P2 | Low | Complete |
| 3 | Graph Traversal Scalability | P2 | Low | Complete |
| 4 | Uncached LLM Evidence Evaluation | P3 | Low | Complete |

---

## Phase 1: Embedding Fallback Chain (P1)

**Goal**: Prevent total cognitive workflow failure when primary embedding service is unavailable.

### Files to Modify
- `src/memory/services/factories.py`

### Implementation Steps

1.1. Add `LocalEmbeddingService` class
   - Uses sentence-transformers (`all-MiniLM-L6-v2`)
   - Lazy model loading (only loads if needed)
   - Same `EmbeddingService` interface

1.2. Add `FallbackEmbeddingService` class
   - Wraps primary (OpenAI/Bedrock) with fallback (Local)
   - In-memory cache for embeddings
   - Logs fallback activation

1.3. Update `get_embedding_service()` factory
   - Add `E2I_EMBEDDING_FALLBACK` env var control
   - Default to fallback-enabled behavior
   - Preserve backward compatibility

### Tests
- Unit test: `tests/unit/test_memory/test_embedding_fallback.py`

---

## Phase 2: Reflector Task Manager (P2)

**Goal**: Track async reflector tasks with timeout, error logging, and graceful shutdown.

### Files to Modify
- `src/memory/cognitive_integration.py`

### Implementation Steps

2.1. Add `ReflectorTaskManager` class
   - Task set tracking with weak references
   - Configurable timeout (default 30s)
   - Success/failure counters
   - `submit()` method with wrapped coroutine

2.2. Update `CognitiveService` class
   - Initialize `ReflectorTaskManager` in `__init__`
   - Replace `asyncio.create_task()` with manager submission
   - Add `shutdown()` method for graceful termination
   - Add `reflector_stats` property

2.3. Add health stats property
   - Expose pending/succeeded/failed counts

### Tests
- Unit test: `tests/unit/test_memory/test_reflector_manager.py`

---

## Phase 3: Graph Pagination (P2)

**Goal**: Prevent memory exhaustion on large graph traversals.

### Files to Modify
- `src/memory/semantic_memory.py`

### Implementation Steps

3.1. Update `get_patient_network()` method
   - Add `limit` parameter (default 100, max 500)
   - Add `offset` parameter for pagination
   - Add `pagination` metadata to response

3.2. Update `get_hcp_influence_network()` method
   - Same pagination parameters as patient network

3.3. Update `traverse_causal_chain()` method
   - Add `limit` parameter (default 50)

3.4. Add count methods
   - `count_patient_network()`
   - `count_hcp_influence_network()`

### Tests
- Unit test: `tests/unit/test_memory/test_semantic_pagination.py`

---

## Phase 4: Evidence Evaluation Cache (P3)

**Goal**: Reduce redundant LLM calls during multi-hop investigation.

### Files to Modify
- `src/memory/004_cognitive_workflow.py`

### Implementation Steps

4.1. Add `EvidenceEvaluationCache` class
   - Hash-based key generation
   - TTL support (default 1 hour)
   - LRU eviction (max 1000 entries)

4.2. Update `evaluate_evidence()` function
   - Check cache before LLM call
   - Store results after LLM evaluation
   - Log cache hits for observability

### Tests
- Unit test: `tests/unit/test_memory/test_evidence_cache.py`

---

## Testing Strategy

All testing will be performed on the droplet during branch merge:

```bash
# Run specific test files
ssh -i ~/.ssh/replit enunez@138.197.4.36 \
  "cd /opt/e2i_causal_analytics && \
   /opt/e2i_causal_analytics/.venv/bin/pytest \
   tests/unit/test_memory/test_embedding_fallback.py \
   tests/unit/test_memory/test_reflector_manager.py \
   tests/unit/test_memory/test_semantic_pagination.py \
   tests/unit/test_memory/test_evidence_cache.py \
   -v"
```

---

## Rollback Plan

All changes are controlled by environment variables:

| Feature | Env Var | Default | Disable Value |
|---------|---------|---------|---------------|
| Embedding Fallback | `E2I_EMBEDDING_FALLBACK` | `true` | `false` |
| Reflector Timeout | `E2I_REFLECTOR_TIMEOUT` | `30` | N/A |
| Graph Pagination | Uses parameter defaults | N/A | N/A |
| Evidence Cache | `E2I_EVIDENCE_CACHE` | `true` | `false` |

---

## Completion Checklist

- [x] Phase 1: Embedding Fallback Chain
- [x] Phase 2: Reflector Task Manager
- [x] Phase 3: Graph Pagination
- [x] Phase 4: Evidence Evaluation Cache
- [x] All unit tests created
- [x] Code committed and pushed
- [ ] Documentation updated (optional)

---

## Notes

- All changes preserve backward compatibility
- No breaking changes to existing interfaces
- Pagination parameters have sensible defaults
- Cache implementations use in-memory storage (no new dependencies)
