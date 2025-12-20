# CLAUDE.md - Observability Connector Agent

## Overview

The **Observability Connector** provides telemetry and monitoring across all 18 agents. It emits spans to Opik for LLM/agent observability, collects quality metrics, and propagates trace context throughout the system.

| Attribute | Value |
|-----------|-------|
| **Tier** | 0 (ML Foundation) |
| **Type** | Standard |
| **SLA** | <100ms (async) |
| **Primary Output** | Spans, QualityMetrics, ObservabilityContext |
| **Database Table** | `ml_observability_spans` |
| **Memory Types** | Working, Episodic |
| **MLOps Tools** | Opik |

## Responsibilities

1. **Span Emission**: Emit spans for all agent operations to Opik
2. **Latency Tracking**: Track p50, p95, p99 latencies per agent
3. **Token Counting**: Track LLM token usage per agent
4. **Error Rate Monitoring**: Track error rates and types
5. **Context Propagation**: Propagate trace_id and span_id across agents
6. **Quality Metrics**: Collect and aggregate quality signals

## Position in Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│  OBSERVABILITY CONNECTOR (Cross-Cutting)                                │
│                                                                         │
│  Wraps ALL agent operations with spans:                                 │
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │  scope_definer  │  │  data_preparer  │  │  model_trainer  │  ...    │
│  │    [span]       │  │    [span]       │  │    [span]       │         │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
│            │                   │                    │                   │
│            └───────────────────┼────────────────────┘                   │
│                                ▼                                        │
│                    observability_connector ◀── YOU ARE HERE             │
│                                │                                        │
│                                ▼                                        │
│                    ┌─────────────────┐                                  │
│                    │      Opik       │                                  │
│                    │   (Telemetry)   │                                  │
│                    └─────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Outputs

### ObservabilityContext

```python
@dataclass
class ObservabilityContext:
    """Distributed tracing context."""
    trace_id: str                    # Root trace ID
    span_id: str                     # Current span ID
    parent_span_id: Optional[str]    # Parent span (if nested)
    
    # Baggage (propagated metadata)
    experiment_id: Optional[str]
    user_id: Optional[str]
    request_id: str
    
    # Sampling
    sampled: bool                    # Whether this trace is sampled
    
    def child(self, operation_name: str) -> "ObservabilityContext":
        """Create child context for nested operations."""
        return ObservabilityContext(
            trace_id=self.trace_id,
            span_id=generate_span_id(),
            parent_span_id=self.span_id,
            experiment_id=self.experiment_id,
            user_id=self.user_id,
            request_id=self.request_id,
            sampled=self.sampled
        )
```

### Span

```python
@dataclass
class Span:
    """Observability span for an operation."""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    
    # Operation
    operation_name: str              # "scope_definer.execute"
    agent_name: str                  # "scope_definer"
    agent_tier: int                  # 0
    
    # Timing
    start_time: datetime
    end_time: Optional[datetime]
    duration_ms: Optional[float]
    
    # Status
    status: SpanStatus               # ok, error, timeout
    error_type: Optional[str]
    error_message: Optional[str]
    
    # Attributes
    attributes: Dict[str, Any]
    # {"experiment_id": "exp_123", "model_name": "CausalForest", ...}
    
    # LLM-specific (for Hybrid/Deep agents)
    llm_model: Optional[str]         # "claude-sonnet-4-20250514"
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    
    # Events within span
    events: List[SpanEvent]
```

### QualityMetrics

```python
@dataclass
class QualityMetrics:
    """Aggregated quality metrics."""
    time_window: str                 # "1h", "24h", "7d"
    computed_at: datetime
    
    # Latency (per agent)
    latency_by_agent: Dict[str, LatencyStats]
    # {"scope_definer": {"p50": 2.1, "p95": 4.5, "p99": 8.2}, ...}
    
    # Error Rates (per agent)
    error_rate_by_agent: Dict[str, float]
    # {"scope_definer": 0.02, "data_preparer": 0.05, ...}
    
    # Token Usage (per agent)
    token_usage_by_agent: Dict[str, TokenUsage]
    # {"causal_impact": {"input": 50000, "output": 12000}, ...}
    
    # Tier-level Metrics
    latency_by_tier: Dict[int, LatencyStats]
    error_rate_by_tier: Dict[int, float]
    
    # System Health
    overall_success_rate: float
    overall_p95_latency_ms: float
    fallback_invocation_rate: float
```

## Database Schema

### ml_observability_spans Table

```sql
CREATE TABLE ml_observability_spans (
    span_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    parent_span_id TEXT,
    
    -- Operation
    operation_name TEXT NOT NULL,
    agent_name agent_name_enum NOT NULL,
    agent_tier agent_tier_enum NOT NULL,
    
    -- Timing
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ,
    duration_ms NUMERIC(12,3),
    
    -- Status
    status TEXT NOT NULL,           -- ok, error, timeout
    error_type TEXT,
    error_message TEXT,
    
    -- Attributes
    attributes JSONB DEFAULT '{}',
    
    -- LLM Metrics
    llm_model TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    
    -- Events
    events JSONB DEFAULT '[]'
);

-- Indexes for common queries
CREATE INDEX idx_span_trace ON ml_observability_spans(trace_id);
CREATE INDEX idx_span_agent ON ml_observability_spans(agent_name);
CREATE INDEX idx_span_time ON ml_observability_spans(start_time);
CREATE INDEX idx_span_status ON ml_observability_spans(status);

-- View for agent latency stats
CREATE OR REPLACE VIEW v_agent_latency AS
SELECT 
    agent_name,
    agent_tier,
    COUNT(*) as span_count,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY duration_ms) as p50_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) as p95_ms,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration_ms) as p99_ms,
    AVG(duration_ms) as avg_ms,
    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as error_rate
FROM ml_observability_spans
WHERE start_time > NOW() - INTERVAL '24 hours'
GROUP BY agent_name, agent_tier;
```

## Implementation

### agent.py

```python
from src.agents.base_agent import BaseAgent
from src.mlops.opik_connector import OpikConnector
from .opik_emitter import OpikEmitter
from .metrics_collector import MetricsCollector
from .context_propagator import ContextPropagator

class ObservabilityConnectorAgent(BaseAgent):
    """
    Observability Connector: Telemetry and monitoring across all agents.
    
    Operates asynchronously to avoid blocking agent operations.
    """
    
    tier = 0
    tier_name = "ml_foundation"
    agent_type = "standard"
    sla_seconds = 0.1  # 100ms async
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.opik = OpikConnector()
        self.emitter = OpikEmitter()
        self.metrics_collector = MetricsCollector()
        self.context_propagator = ContextPropagator()
        self.span_repo = MLObservabilitySpanRepository()
    
    # ═══════════════════════════════════════════════════════════════
    # Context Management
    # ═══════════════════════════════════════════════════════════════
    
    def create_context(
        self,
        request_id: str,
        experiment_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> ObservabilityContext:
        """Create new observability context for a request."""
        return ObservabilityContext(
            trace_id=generate_trace_id(),
            span_id=generate_span_id(),
            parent_span_id=None,
            experiment_id=experiment_id,
            user_id=user_id,
            request_id=request_id,
            sampled=self._should_sample()
        )
    
    def extract_context(self, headers: Dict) -> ObservabilityContext:
        """Extract context from incoming request headers."""
        return self.context_propagator.extract(headers)
    
    def inject_context(
        self, 
        context: ObservabilityContext, 
        headers: Dict
    ) -> Dict:
        """Inject context into outgoing request headers."""
        return self.context_propagator.inject(context, headers)
    
    # ═══════════════════════════════════════════════════════════════
    # Span Operations
    # ═══════════════════════════════════════════════════════════════
    
    @asynccontextmanager
    async def span(
        self,
        operation_name: str,
        agent_name: str,
        agent_tier: int,
        context: ObservabilityContext,
        attributes: Dict = None
    ):
        """
        Context manager for creating spans.
        
        Usage:
            async with observability.span("execute", "scope_definer", 0, ctx) as span:
                # Agent operation
                span.set_attribute("experiment_id", "exp_123")
        """
        span = Span(
            span_id=context.span_id,
            trace_id=context.trace_id,
            parent_span_id=context.parent_span_id,
            operation_name=operation_name,
            agent_name=agent_name,
            agent_tier=agent_tier,
            start_time=datetime.utcnow(),
            end_time=None,
            duration_ms=None,
            status=SpanStatus.OK,
            error_type=None,
            error_message=None,
            attributes=attributes or {},
            events=[]
        )
        
        try:
            yield span
            span.status = SpanStatus.OK
        except Exception as e:
            span.status = SpanStatus.ERROR
            span.error_type = type(e).__name__
            span.error_message = str(e)
            raise
        finally:
            span.end_time = datetime.utcnow()
            span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000
            
            # Emit to Opik (async, non-blocking)
            asyncio.create_task(self._emit_span(span))
    
    async def _emit_span(self, span: Span):
        """Emit span to Opik and persist to database."""
        try:
            # Send to Opik
            await self.emitter.emit(span)
            
            # Persist to database
            await self.span_repo.create(span)
        except Exception as e:
            # Log but don't fail - observability should not break operations
            logger.error(f"Failed to emit span: {e}")
    
    # ═══════════════════════════════════════════════════════════════
    # LLM-specific Tracking (for Hybrid/Deep agents)
    # ═══════════════════════════════════════════════════════════════
    
    async def track_llm_call(
        self,
        span: Span,
        model: str,
        input_tokens: int,
        output_tokens: int
    ):
        """Track LLM usage within a span."""
        span.llm_model = model
        span.input_tokens = input_tokens
        span.output_tokens = output_tokens
        span.total_tokens = input_tokens + output_tokens
        
        # Add event
        span.events.append(SpanEvent(
            timestamp=datetime.utcnow(),
            name="llm_call",
            attributes={
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
        ))
    
    # ═══════════════════════════════════════════════════════════════
    # Metrics Collection
    # ═══════════════════════════════════════════════════════════════
    
    async def get_quality_metrics(
        self,
        time_window: str = "24h"
    ) -> QualityMetrics:
        """Collect aggregated quality metrics."""
        return await self.metrics_collector.collect(time_window)
    
    async def execute(self, state: AgentState) -> AgentState:
        """
        Main execution: Collect and return quality metrics.
        
        This agent is usually called via its helper methods (span, track_llm_call),
        but can also be invoked directly to get metrics.
        """
        metrics = await self.get_quality_metrics(
            time_window=state.get("time_window", "24h")
        )
        
        return state.with_updates(quality_metrics=metrics)
```

### opik_emitter.py

```python
from opik import Opik

class OpikEmitter:
    """Emit spans to Opik for LLM/agent observability."""
    
    def __init__(self):
        self.opik = Opik()
    
    async def emit(self, span: Span):
        """Emit span to Opik."""
        opik_span = self.opik.trace(
            name=span.operation_name,
            trace_id=span.trace_id,
            span_id=span.span_id,
            parent_span_id=span.parent_span_id,
            start_time=span.start_time,
            end_time=span.end_time,
            status=span.status.value,
            attributes={
                "agent_name": span.agent_name,
                "agent_tier": span.agent_tier,
                **span.attributes
            }
        )
        
        # Add LLM metrics if present
        if span.llm_model:
            opik_span.log_llm_call(
                model=span.llm_model,
                input_tokens=span.input_tokens,
                output_tokens=span.output_tokens
            )
        
        # Add events
        for event in span.events:
            opik_span.add_event(
                name=event.name,
                timestamp=event.timestamp,
                attributes=event.attributes
            )
```

### metrics_collector.py

```python
class MetricsCollector:
    """Collect and aggregate observability metrics."""
    
    async def collect(self, time_window: str) -> QualityMetrics:
        """Collect quality metrics for time window."""
        
        # Query span data
        spans = await self._query_spans(time_window)
        
        # Compute latency by agent
        latency_by_agent = self._compute_latency_stats(spans, group_by="agent_name")
        
        # Compute error rates by agent
        error_rate_by_agent = self._compute_error_rates(spans, group_by="agent_name")
        
        # Compute token usage by agent
        token_usage_by_agent = self._compute_token_usage(spans)
        
        # Compute tier-level metrics
        latency_by_tier = self._compute_latency_stats(spans, group_by="agent_tier")
        error_rate_by_tier = self._compute_error_rates(spans, group_by="agent_tier")
        
        # Overall metrics
        total_spans = len(spans)
        error_spans = sum(1 for s in spans if s.status == "error")
        
        return QualityMetrics(
            time_window=time_window,
            computed_at=datetime.utcnow(),
            latency_by_agent=latency_by_agent,
            error_rate_by_agent=error_rate_by_agent,
            token_usage_by_agent=token_usage_by_agent,
            latency_by_tier=latency_by_tier,
            error_rate_by_tier=error_rate_by_tier,
            overall_success_rate=1 - (error_spans / total_spans) if total_spans > 0 else 1.0,
            overall_p95_latency_ms=self._compute_percentile(spans, 0.95),
            fallback_invocation_rate=self._compute_fallback_rate(spans)
        )
    
    def _compute_latency_stats(
        self,
        spans: List[Span],
        group_by: str
    ) -> Dict[str, LatencyStats]:
        """Compute latency statistics grouped by field."""
        groups = defaultdict(list)
        
        for span in spans:
            key = getattr(span, group_by)
            groups[key].append(span.duration_ms)
        
        stats = {}
        for key, durations in groups.items():
            durations = sorted(durations)
            stats[key] = LatencyStats(
                p50=durations[int(len(durations) * 0.50)],
                p95=durations[int(len(durations) * 0.95)],
                p99=durations[int(len(durations) * 0.99)],
                avg=sum(durations) / len(durations)
            )
        
        return stats
```

## Usage in Other Agents

### Wrapping Agent Operations

```python
# In any agent's execute method:

class ScopeDefinerAgent(BaseAgent):
    
    async def execute(self, state: AgentState) -> AgentState:
        ctx = state.observability_context
        
        async with observability.span(
            operation_name="scope_definer.execute",
            agent_name="scope_definer",
            agent_tier=0,
            context=ctx,
            attributes={"experiment_id": state.experiment_id}
        ) as span:
            # Main operation
            scope_spec = await self._build_scope(state)
            
            # Add attributes
            span.set_attribute("problem_type", scope_spec.problem_type)
            
            return state.with_updates(scope_spec=scope_spec)
```

### Tracking LLM Calls in Hybrid Agents

```python
# In feature_analyzer (Hybrid agent):

async def _interpret_shap(self, shap_analysis):
    ctx = self.current_context
    
    async with observability.span(
        operation_name="feature_analyzer.interpret",
        agent_name="feature_analyzer",
        agent_tier=0,
        context=ctx.child("interpret")
    ) as span:
        # LLM call
        response = await self.llm_client.generate(prompt)
        
        # Track LLM usage
        await observability.track_llm_call(
            span=span,
            model="claude-sonnet-4-20250514",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens
        )
        
        return response.content
```

## Downstream Integration

### health_score (Tier 3)

```python
# In health_score.execute():

# Get quality metrics from observability_connector
metrics = await observability.get_quality_metrics(time_window="1h")

# Factor into health score
agent_health = 1 - metrics.overall_error_rate
latency_health = 1 if metrics.overall_p95_latency_ms < 5000 else 0.5

health_score = (agent_health * 0.6) + (latency_health * 0.4)
```

### orchestrator (Tier 1)

```python
# In orchestrator for routing decisions:

# Check agent latency before routing
metrics = await observability.get_quality_metrics(time_window="1h")

if metrics.latency_by_agent["causal_impact"].p95 > 30000:
    # Route to faster alternative if causal_impact is slow
    logger.warning("causal_impact slow, considering fallback")
```

## Error Handling

```python
class ObservabilityError(AgentError):
    """Base error for observability_connector."""
    pass

class SpanEmissionError(ObservabilityError):
    """Failed to emit span."""
    # Non-fatal: logged but doesn't block operations
    pass

class MetricsCollectionError(ObservabilityError):
    """Failed to collect metrics."""
    pass
```

## Testing

```python
class TestObservabilityConnector:
    
    async def test_span_creation(self):
        """Test span is created and emitted."""
        ctx = observability.create_context(request_id="test_123")
        
        async with observability.span(
            "test_operation", "test_agent", 0, ctx
        ) as span:
            pass
        
        # Verify span was persisted
        stored = await span_repo.get(span.span_id)
        assert stored is not None
        assert stored.status == "ok"
    
    async def test_error_capture(self):
        """Test errors are captured in span."""
        ctx = observability.create_context(request_id="test_456")
        
        with pytest.raises(ValueError):
            async with observability.span(
                "failing_operation", "test_agent", 0, ctx
            ) as span:
                raise ValueError("Test error")
        
        stored = await span_repo.get(span.span_id)
        assert stored.status == "error"
        assert stored.error_type == "ValueError"
    
    async def test_llm_tracking(self):
        """Test LLM usage is tracked."""
        # ... verify token counts are recorded
```

## Key Principles

1. **Non-Blocking**: Observability should never block agent operations
2. **Context Propagation**: trace_id flows through all operations
3. **Graceful Degradation**: Failures logged but don't break agents
4. **LLM Tracking**: All Hybrid/Deep agent LLM calls tracked
5. **Aggregation**: Metrics pre-computed for fast dashboard queries
6. **Sampling**: Support sampling for high-volume traces
