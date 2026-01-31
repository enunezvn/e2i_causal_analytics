# Workers Module Tests

## Test Files Created

### 1. test_event_consumer.py (50+ tests)
Comprehensive tests for `src/workers/event_consumer.py`:
- `CeleryMetrics` initialization:
  - Prometheus metrics setup
  - Metric registration
  - Conditional initialization based on availability
- `TaskTiming` dataclass functionality
- `CeleryEventConsumer` event handlers:
  - Task lifecycle events (sent, received, started, succeeded, failed)
  - Task retry and rejection handling
  - Task revocation
  - Worker online/offline events
  - Worker heartbeat tracking
- Queue extraction logic
- Task timing creation and cleanup
- Worker type inference
- Event handler routing
- Trace ID propagation helpers:
  - Injection into headers
  - Extraction from headers
- `traced_task` context manager
- Integration tests for complete task lifecycles

**Coverage Areas:**
- Event consumption and processing
- Prometheus metrics recording
- Task latency and runtime tracking
- Worker availability monitoring
- Trace ID propagation through task execution
- Error handling in event processing
- Complete task lifecycle flows (success and failure paths)

## Running the Tests

```bash
# Run all workers tests
pytest tests/unit/test_workers/ -v

# Run specific test file
pytest tests/unit/test_workers/test_event_consumer.py -v

# Run with coverage
pytest tests/unit/test_workers/ --cov=src/workers --cov-report=term-missing

# Run specific test class
pytest tests/unit/test_workers/test_event_consumer.py::TestCeleryEventConsumer -v

# Run integration tests only
pytest tests/unit/test_workers/test_event_consumer.py::TestEventHandlerIntegration -v
```

## Test Coverage

### Key Source Files Covered:
- ✅ `src/workers/event_consumer.py` (~586 lines) - Comprehensive coverage

### Areas with High Test Coverage:
- Event handler methods for all task lifecycle events
- Prometheus metrics initialization and recording
- Task timing tracking
- Worker type inference
- Trace ID propagation
- Event routing and handler mapping
- Complete task lifecycle flows

### Mock Dependencies:
All external dependencies are properly mocked:
- Celery application
- Prometheus client (Counter, Histogram, Gauge)
- Event receiver
- Worker connections

## Test Organization

### Test Classes:
1. **TestCeleryMetrics** - Metrics dataclass and initialization
2. **TestTaskTiming** - Task timing dataclass
3. **TestCeleryEventConsumer** - Main consumer functionality
4. **TestTraceIDPropagation** - Trace ID helper functions
5. **TestTracedTask** - Context manager for traced tasks
6. **TestEventHandlerIntegration** - End-to-end lifecycle tests

## Notes

- Prometheus client is mocked to avoid import dependency
- All event handlers are tested with realistic event data
- Timing calculations are verified
- Metrics recording is validated via mock assertions
- Integration tests cover complete task flows from sent to completion
- Both success and failure paths are tested
