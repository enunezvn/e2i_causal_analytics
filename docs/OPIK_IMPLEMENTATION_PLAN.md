# Opik Implementation Plan

**Project**: E2I Causal Analytics
**Component**: LLM/Agent Observability
**Status**: Planning Complete, Implementation Not Started
**Last Updated**: 2025-12-21

---

## Executive Summary

This document provides a comprehensive implementation plan for integrating Opik as the primary LLM/agent observability platform for E2I Causal Analytics. Opik will provide distributed tracing, latency monitoring, token usage tracking, and error analysis across all 18 agents in 6 tiers.

**Current State**: 0% SDK implementation (configuration and mocks complete)
**Target State**: 100% production-ready with batching, circuit breaker, and caching
**Estimated Effort**: 14-19 hours across 3 phases

---

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Architecture Overview](#architecture-overview)
3. [Phased Implementation](#phased-implementation)
4. [File Structure](#file-structure)
5. [Dependencies & Environment](#dependencies--environment)
6. [Testing Strategy](#testing-strategy)
7. [Success Criteria](#success-criteria)
8. [Risk Mitigation](#risk-mitigation)

---

## Current State Analysis

### What Already Exists (Complete)

| Component | Location | Status |
|-----------|----------|--------|
| Agent Structure | `src/agents/ml_foundation/observability_connector/agent.py` | Complete (453 lines) |
| Span Context Manager | `agent.py:110-166` | Complete (async with pattern) |
| LLM Call Tracking | `agent.py:167-232` | Complete (token tracking) |
| W3C Trace Context | `nodes/context_manager.py` | Complete (traceparent/tracestate) |
| LangGraph Workflow | `graph.py` | Complete (2-node pipeline) |
| State Definition | `state.py` | Complete (TypedDict) |
| Metrics Aggregation Logic | `nodes/metrics_aggregator.py` | Complete (latency/error calculations) |
| Database Schema | `database/ml/mlops_tables.sql:425-479` | Complete (table + indexes) |
| Configuration | `config/agent_config.yaml:838-841` | Complete |
| Specialist Docs | `.claude/specialists/ml_foundation/observability_connector.md` | Complete |
| Unit Tests | 65 tests | Complete |

### What Needs Implementation (Missing)

| Component | Current State | Priority | Est. Hours |
|-----------|---------------|----------|------------|
| OpikConnector class | Does not exist | HIGH | 2-3 |
| ObservabilitySpanRepository | Does not exist | HIGH | 1-2 |
| Pydantic models | Does not exist | HIGH | 0.5 |
| Replace mock Opik calls | Commented code in `span_emitter.py:48-62` | HIGH | 1-2 |
| Replace mock DB writes | Commented code in `span_emitter.py:65-84` | HIGH | 1-2 |
| Real metrics queries | Mock data in `metrics_aggregator.py:111-172` | MEDIUM | 1-2 |
| Batch processing | Not implemented | MEDIUM | 2-3 |
| Circuit breaker | Not implemented | MEDIUM | 1-2 |
| Metrics caching | Not implemented | LOW | 1-2 |
| Configuration file | Hardcoded defaults | LOW | 0.5 |

---

## Architecture Overview

### Component Diagram

```
                                    ┌─────────────────────────────────────┐
                                    │           Opik Cloud/Self-Hosted    │
                                    │  (Traces, Spans, Metrics Dashboard) │
                                    └─────────────────▲───────────────────┘
                                                      │
                                                      │ HTTPS
                                                      │
┌─────────────────────────────────────────────────────┼─────────────────────────────────────────────────────┐
│                                                     │                                                      │
│  E2I Causal Analytics                              │                                                      │
│                                                     │                                                      │
│  ┌──────────────────────┐    ┌──────────────────────┼────────────────────┐    ┌──────────────────────┐   │
│  │                      │    │                      │                    │    │                      │   │
│  │   All 18 Agents      │───▶│  observability_connector (Tier 0)        │───▶│   Supabase DB        │   │
│  │                      │    │                      │                    │    │  ml_observability_   │   │
│  │  - orchestrator      │    │  ┌────────────────┐  │  ┌───────────────┐ │    │  spans               │   │
│  │  - causal_impact     │    │  │ OpikConnector  │──┘  │ Span Emitter  │ │    │                      │   │
│  │  - gap_analyzer      │    │  │ (SDK Wrapper)  │     │ (Node)        │─┼───▶│  Indexes:            │   │
│  │  - etc...            │    │  └────────────────┘     └───────────────┘ │    │  - trace_id          │   │
│  │                      │    │                                           │    │  - agent_name        │   │
│  └──────────────────────┘    │  ┌────────────────┐     ┌───────────────┐ │    │  - start_time        │   │
│                              │  │ Context Mgr    │     │ Metrics Aggr  │ │    │                      │   │
│                              │  │ (W3C Trace)    │     │ (Node)        │─┼───▶│  Views:              │   │
│                              │  └────────────────┘     └───────────────┘ │    │  - v_agent_latency_  │   │
│                              │                                           │    │    summary           │   │
│                              │  ┌────────────────────────────────────┐   │    │                      │   │
│                              │  │ ObservabilitySpanRepository        │   │    └──────────────────────┘   │
│                              │  │ (Data Access Layer)                │───┘                               │
│                              │  └────────────────────────────────────┘                                   │
│                              │                                                                            │
│                              └────────────────────────────────────────────────────────────────────────────┘
│                                                                                                           │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Agent executes operation** → Calls `observability_connector.trace_agent()` context manager
2. **Context Manager** → Creates span with W3C Trace Context, tracks timing
3. **Span Emitter Node** → Sends span to Opik SDK AND persists to database
4. **Metrics Aggregator Node** → Queries database for latency/error stats
5. **Dashboard** → Opik UI displays traces; local queries for custom dashboards

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Opik as primary (not MLflow) | Specialized for LLM observability, better trace visualization |
| Dual persistence (Opik + DB) | Opik for visualization, DB for custom queries and compliance |
| W3C Trace Context | Industry standard, enables distributed tracing |
| Async emission | Non-blocking, doesn't impact agent latency |
| Circuit breaker | Graceful degradation when Opik unavailable |

---

## Phased Implementation

### Phase 1: Core Infrastructure (4-6 hours)

**Objective**: Create foundational components for Opik and database integration.

#### 1.1 Create OpikConnector (`src/mlops/opik_connector.py`)

Centralized Opik SDK wrapper used by all observability operations.

```python
"""Opik SDK connector for LLM/agent observability."""

from contextlib import asynccontextmanager
from typing import Any, Dict, Optional
import os
from datetime import datetime

from opik import Opik, track


class OpikConnector:
    """Singleton wrapper for Opik SDK operations."""

    _instance: Optional["OpikConnector"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        project_name: str = "e2i-causal-analytics",
        workspace: str = "default",
        api_key: Optional[str] = None,
    ):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.project_name = project_name
        self.workspace = workspace
        self.api_key = api_key or os.getenv("OPIK_API_KEY")

        # Initialize Opik client
        self.client = Opik(
            project_name=self.project_name,
            workspace=self.workspace,
        )

        self._initialized = True

    @asynccontextmanager
    async def trace_agent(
        self,
        agent_name: str,
        operation: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for tracing agent operations.

        Usage:
            async with opik.trace_agent("gap_analyzer", "analyze_gaps") as span:
                # Do work
                span.set_attribute("result_count", 10)
        """
        start_time = datetime.utcnow()
        span_data = {
            "agent_name": agent_name,
            "operation": operation,
            "trace_id": trace_id,
            "parent_span_id": parent_span_id,
            "metadata": metadata or {},
            "start_time": start_time,
            "status": "started",
        }

        try:
            yield span_data
            span_data["status"] = "completed"
        except Exception as e:
            span_data["status"] = "error"
            span_data["error"] = str(e)
            span_data["error_type"] = type(e).__name__
            raise
        finally:
            span_data["end_time"] = datetime.utcnow()
            span_data["duration_ms"] = (
                span_data["end_time"] - start_time
            ).total_seconds() * 1000

            # Emit to Opik (async, non-blocking)
            await self._emit_span(span_data)

    async def trace_llm_call(
        self,
        model: str,
        prompt: str,
        response: str,
        input_tokens: int,
        output_tokens: int,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Track an LLM API call with token usage."""
        # Implementation for LLM-specific tracking
        pass

    async def log_metric(
        self,
        name: str,
        value: float,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log a custom metric to Opik."""
        pass

    async def _emit_span(self, span_data: Dict[str, Any]):
        """Emit span to Opik SDK."""
        # Actual Opik SDK call
        self.client.trace(
            name=f"{span_data['agent_name']}.{span_data['operation']}",
            trace_id=span_data.get("trace_id"),
            parent_span_id=span_data.get("parent_span_id"),
            start_time=span_data["start_time"],
            end_time=span_data.get("end_time"),
            metadata={
                "agent_name": span_data["agent_name"],
                "status": span_data["status"],
                "duration_ms": span_data.get("duration_ms"),
                **span_data.get("metadata", {}),
            },
        )
```

#### 1.2 Create ObservabilitySpanRepository (`src/repositories/observability_span.py`)

Data access layer for `ml_observability_spans` table.

```python
"""Repository for observability spans."""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

from src.repositories.base import BaseRepository
from src.agents.ml_foundation.observability_connector.models import ObservabilitySpan


class ObservabilitySpanRepository(BaseRepository[ObservabilitySpan]):
    """Repository for ml_observability_spans table."""

    table_name = "ml_observability_spans"
    model_class = ObservabilitySpan

    async def insert_span(self, span: ObservabilitySpan) -> ObservabilitySpan:
        """Insert a single span."""
        return await self.create(span)

    async def insert_spans_batch(
        self,
        spans: List[ObservabilitySpan],
    ) -> int:
        """Batch insert spans for performance."""
        if not self.client or not spans:
            return 0

        data = [s.model_dump() for s in spans]
        result = await self.client.table(self.table_name).insert(data).execute()
        return len(result.data)

    async def get_spans_by_time_window(
        self,
        window: str = "24h",
        agent_name: Optional[str] = None,
        limit: int = 1000,
    ) -> List[ObservabilitySpan]:
        """Get spans within a time window."""
        # Parse window (1h, 24h, 7d)
        hours = self._parse_window(window)
        since = datetime.utcnow() - timedelta(hours=hours)

        query = (
            self.client.table(self.table_name)
            .select("*")
            .gte("start_time", since.isoformat())
            .order("start_time", desc=True)
            .limit(limit)
        )

        if agent_name:
            query = query.eq("agent_name", agent_name)

        result = await query.execute()
        return [self._to_model(row) for row in result.data]

    async def get_latency_stats(
        self,
        window: str = "24h",
        agent_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get latency percentiles using SQL aggregation."""
        # Use the v_agent_latency_summary view for efficiency
        query = self.client.table("v_agent_latency_summary").select("*")

        if agent_name:
            query = query.eq("agent_name", agent_name)

        result = await query.execute()
        return result.data

    async def delete_old_spans(
        self,
        days: int = 90,
        batch_size: int = 1000,
    ) -> int:
        """Delete spans older than retention period."""
        cutoff = datetime.utcnow() - timedelta(days=days)

        result = await (
            self.client.table(self.table_name)
            .delete()
            .lt("start_time", cutoff.isoformat())
            .limit(batch_size)
            .execute()
        )

        return len(result.data)

    @staticmethod
    def _parse_window(window: str) -> int:
        """Parse time window string to hours."""
        if window.endswith("h"):
            return int(window[:-1])
        elif window.endswith("d"):
            return int(window[:-1]) * 24
        return 24  # Default to 24 hours
```

#### 1.3 Create Pydantic Models (`src/agents/ml_foundation/observability_connector/models.py`)

```python
"""Pydantic models for observability data."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid


class ObservabilitySpan(BaseModel):
    """Database model for ml_observability_spans."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str
    trace_id: str
    parent_span_id: Optional[str] = None
    operation_name: str
    agent_name: str
    agent_tier: int = 0
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: str = "started"
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)
    llm_model: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    events: List[Dict[str, Any]] = Field(default_factory=list)

    class Config:
        from_attributes = True


class LatencyStats(BaseModel):
    """Latency percentile statistics."""

    agent_name: str
    p50_ms: float
    p95_ms: float
    p99_ms: float
    avg_ms: float
    count: int
    time_window: str


class QualityMetrics(BaseModel):
    """Aggregated quality metrics."""

    total_spans: int
    successful_spans: int
    failed_spans: int
    error_rate: float
    avg_latency_ms: float
    p95_latency_ms: float
    time_window: str
    agents: Dict[str, LatencyStats]
```

---

### Phase 2: Agent Integration (3-4 hours)

**Objective**: Replace mock implementations with real Opik and database calls.

#### 2.1 Update `span_emitter.py`

**File**: `src/agents/ml_foundation/observability_connector/nodes/span_emitter.py`

**Changes**:
1. Import `OpikConnector` from `src/mlops/opik_connector.py`
2. Import `ObservabilitySpanRepository` from `src/repositories/`
3. Replace commented mock code (lines 48-84) with actual SDK calls
4. Add error handling with fallback to database-only

#### 2.2 Update `metrics_aggregator.py`

**File**: `src/agents/ml_foundation/observability_connector/nodes/metrics_aggregator.py`

**Changes**:
1. Replace `_get_mock_spans()` with repository queries
2. Use `v_agent_latency_summary` view for fast percentile queries
3. Add time window parsing (1h, 24h, 7d)

#### 2.3 Update `agent.py`

**File**: `src/agents/ml_foundation/observability_connector/agent.py`

**Changes**:
1. Initialize `OpikConnector` in `__init__`
2. Initialize `ObservabilitySpanRepository` with Supabase client
3. Pass dependencies to nodes

---

### Phase 3: Production Features (4-5 hours)

**Objective**: Production hardening with performance, reliability, and monitoring.

#### 3.1 Batch Processing

**New File**: `src/agents/ml_foundation/observability_connector/batch_processor.py`

- Buffer spans in memory (max 100 or 5 seconds)
- Emit batches to Opik and database
- Handle partial failures gracefully

#### 3.2 Circuit Breaker

**Add to**: `src/mlops/opik_connector.py`

- Track consecutive failures (threshold: 5)
- Open circuit after threshold
- Half-open after 30 seconds
- Fall back to database-only when circuit open

#### 3.3 Metrics Caching

**New File**: `src/agents/ml_foundation/observability_connector/cache.py`

- Cache metrics in Redis (or in-memory fallback)
- TTL: 60s for "1h" window, 300s for "24h"
- Invalidate on new span insertion

#### 3.4 Configuration File

**New File**: `config/observability.yaml`

```yaml
observability:
  opik:
    enabled: true
    project_name: "e2i-causal-analytics"
    workspace: "default"
    api_key_env: "OPIK_API_KEY"

  sampling:
    default_rate: 1.0
    production_rate: 0.1
    always_sample_errors: true

  batching:
    enabled: true
    max_batch_size: 100
    max_batch_wait_seconds: 5

  circuit_breaker:
    failure_threshold: 5
    reset_timeout_seconds: 30

  retention:
    span_ttl_days: 90
    cleanup_batch_size: 1000
```

---

## File Structure

### New Files to Create

```
src/
├── mlops/
│   ├── opik_connector.py           # NEW: Opik SDK wrapper
│   └── __init__.py                 # UPDATE: Export OpikConnector
├── repositories/
│   └── observability_span.py       # NEW: Span repository
├── agents/ml_foundation/observability_connector/
│   ├── models.py                   # NEW: Pydantic models
│   ├── batch_processor.py          # NEW (Phase 3): Batch emission
│   └── cache.py                    # NEW (Phase 3): Metrics caching
config/
└── observability.yaml              # NEW (Phase 3): Configuration
```

### Files to Modify

```
src/agents/ml_foundation/observability_connector/
├── agent.py                        # UPDATE: Add connector/repository init
├── nodes/
│   ├── span_emitter.py             # UPDATE: Replace mock with real calls
│   └── metrics_aggregator.py       # UPDATE: Replace mock with DB queries
src/mlops/
└── __init__.py                     # UPDATE: Export OpikConnector
src/repositories/
└── __init__.py                     # UPDATE: Export ObservabilitySpanRepository
```

---

## Dependencies & Environment

### Python Dependencies

Already installed in requirements:
```
opik>=0.2.0
```

### Environment Variables

```bash
# Required
OPIK_API_KEY=your-opik-api-key

# Optional (with defaults)
OPIK_ENDPOINT=https://www.comet.com/opik
OPIK_PROJECT_NAME=e2i-causal-analytics
OPIK_WORKSPACE=default
```

### Database Prerequisites

Schema already defined in `database/ml/mlops_tables.sql:425-479`.

Verify with:
```bash
# Check table exists
psql -c "SELECT count(*) FROM ml_observability_spans;"

# Check view exists
psql -c "SELECT * FROM v_agent_latency_summary LIMIT 1;"
```

---

## Testing Strategy

### Unit Tests (Update Existing)

**Location**: `tests/unit/test_agents/test_ml_foundation/test_observability_connector/`

**Updates**:
1. Mock `OpikConnector` for isolation
2. Mock `ObservabilitySpanRepository` for database tests
3. Add tests for new components

### Integration Tests (New)

**Location**: `tests/integration/test_observability_integration.py`

**Test Cases**:
- End-to-end span emission to Opik
- Database write and query round-trip
- Metrics computation from real data
- Cross-agent context propagation

### Load Tests (New)

**Location**: `tests/load/test_observability_load.py`

**Scenarios**:
- 100 concurrent span emissions
- 1000 spans in batch
- Recovery from Opik outage

---

## Success Criteria

### Phase 1 Complete When:
- [ ] `OpikConnector` class created and unit tested
- [ ] `ObservabilitySpanRepository` created and unit tested
- [ ] Pydantic models defined and validated
- [ ] All 65 existing tests pass

### Phase 2 Complete When:
- [ ] Spans successfully emit to Opik dashboard
- [ ] Spans persist to `ml_observability_spans` table
- [ ] Metrics query returns real data
- [ ] Integration tests pass

### Phase 3 Complete When:
- [ ] Batch processing handles 100+ spans/second
- [ ] Circuit breaker protects against Opik outages
- [ ] Metrics caching reduces query latency by 80%
- [ ] Configuration file works

### Production Ready When:
- [ ] CONTRACT_VALIDATION.md shows 100% compliance
- [ ] Integration tests pass in CI
- [ ] Load tests pass (100 concurrent spans)
- [ ] Documentation updated

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Opik API unavailable | Medium | High | Circuit breaker + database fallback |
| High span volume | Medium | Medium | Batch processing + sampling |
| Database write failures | Low | Medium | Async retry with exponential backoff |
| Memory pressure | Low | Medium | Bounded buffers + graceful degradation |
| Configuration errors | Low | Low | Validation on startup + sensible defaults |
| API key exposure | Low | High | Environment variables, never in code |

---

## Related Documentation

- **Specialist**: `.claude/specialists/ml_foundation/observability_connector.md`
- **MLOps Integration**: `.claude/specialists/MLOps_Integration/mlops_integration.md`
- **Context**: `.claude/context/mlops-tools.md`
- **Contract Validation**: `src/agents/ml_foundation/observability_connector/CONTRACT_VALIDATION.md`
- **Database Schema**: `database/ml/mlops_tables.sql:425-479`

---

## Appendix: Database Schema Reference

```sql
-- ml_observability_spans table (from mlops_tables.sql)
CREATE TABLE IF NOT EXISTS ml_observability_spans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    span_id VARCHAR(64) NOT NULL,
    trace_id VARCHAR(64) NOT NULL,
    parent_span_id VARCHAR(64),
    operation_name VARCHAR(255) NOT NULL,
    agent_name VARCHAR(100) NOT NULL,
    agent_tier INTEGER DEFAULT 0,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ,
    duration_ms DECIMAL(12, 3),
    status VARCHAR(50) DEFAULT 'started',
    error_type VARCHAR(100),
    error_message TEXT,
    attributes JSONB DEFAULT '{}',
    llm_model VARCHAR(100),
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    events JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_obs_spans_trace ON ml_observability_spans(trace_id);
CREATE INDEX idx_obs_spans_agent ON ml_observability_spans(agent_name);
CREATE INDEX idx_obs_spans_time ON ml_observability_spans(start_time);
CREATE INDEX idx_obs_spans_status ON ml_observability_spans(status);
```
