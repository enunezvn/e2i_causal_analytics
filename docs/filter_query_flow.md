# E2I Filter → Query Flow Technical Specification

**Version:** 1.0
**Date:** 2025-12-15
**Purpose:** Technical specification for dashboard filter control integration with query pipeline

---

## Overview

The E2I dashboard provides filter controls (brand, region, date range, etc.) that need to integrate seamlessly with the natural language query pipeline. This document specifies how filter state transforms into database queries and interacts with the NLP extraction system.

---

## Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERACTION                              │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
              ┌─────▼──────┐              ┌──────▼──────┐
              │ Filter     │              │ Text Query  │
              │ Selection  │              │ Input       │
              └─────┬──────┘              └──────┬──────┘
                    │                             │
┌───────────────────┴──────────────┬──────────────┴──────────────────┐
│                                  │                                  │
│  FilterState {                   │  User Query String               │
│    brand: "kisqali"              │  "Why did adoption drop?"        │
│    region: "northeast"           │                                  │
│    date_range: {...}             │                                  │
│  }                               │                                  │
│                                  │                                  │
└─────────┬────────────────────────┴──────────────┬───────────────────┘
          │                                       │
          │                                       │
    ┌─────▼──────┐                         ┌─────▼──────────┐
    │ Mode:      │                         │ Mode:          │
    │ WITHOUT    │                         │ WITH           │
    │ NLP        │                         │ NLP            │
    └─────┬──────┘                         └─────┬──────────┘
          │                                      │
          │                                      │
    ┌─────▼──────────┐                    ┌─────▼──────────────┐
    │ SQL WHERE      │                    │ NLP Extraction     │
    │ Generator      │                    │ Pipeline           │
    └─────┬──────────┘                    └─────┬──────────────┘
          │                                     │
          │                              ┌──────▼──────────┐
          │                              │ ExtractedEntities│
          │                              │ {               │
          │                              │   brand: ?      │
          │                              │   intent: ?     │
          │                              │ }               │
          │                              └──────┬──────────┘
          │                                     │
          │                              ┌──────▼──────────┐
          │                              │ Filter Injection│
          │                              │ (Override NLP)  │
          │                              └──────┬──────────┘
          │                                     │
          │                              ┌──────▼──────────┐
          │                              │ EnrichedQuery   │
          │                              │ {               │
          │                              │   entities: {   │
          │                              │     brand: "kisqali"│
          │                              │     region: "northeast"│
          │                              │     intent: "causal"│
          │                              │   }             │
          │                              │ }               │
          │                              └──────┬──────────┘
          │                                     │
    ┌─────▼─────────────────────────────────────▼──────────┐
    │              Orchestrator Agent                      │
    │  - Route to appropriate agents                       │
    │  - Apply filters to data access                      │
    │  - Generate SQL with WHERE clauses                   │
    └─────┬────────────────────────────────────────────────┘
          │
    ┌─────▼──────────┐
    │  Database      │
    │  Query         │
    │  Execution     │
    └─────┬──────────┘
          │
    ┌─────▼──────────┐
    │  Filtered      │
    │  Results       │
    └─────┬──────────┘
          │
    ┌─────▼──────────┐
    │  Dashboard     │
    │  Visualization │
    │  Update        │
    └────────────────┘
```

---

## Component Specifications

### 1. FilterState Object

**Location:** Frontend Redux Store (`filterSlice`)

**Schema:**
```typescript
interface FilterState {
  brand: string | null;           // "remibrutinib" | "fabhalta" | "kisqali" | "all" | null
  region: string | null;          // "northeast" | "south" | "midwest" | "west" | "all" | null
  dateRange: {
    startDate: string;            // ISO 8601: "YYYY-MM-DD"
    endDate: string;
  };
  specialty: string | null;       // "allergy" | "hematology" | ... | "all" | null
  hcpSegment: string | null;      // "high_prescriber" | ... | "all" | null
}
```

**State Management:**
- Stored in Redux `filterSlice`
- Persisted to URL query parameters
- Synced across browser tabs via localStorage events
- Restored on page refresh from URL

**Default Values:**
```typescript
const defaultFilterState: FilterState = {
  brand: null,
  region: null,
  dateRange: {
    startDate: subtractDays(today(), 90),  // 90 days ago
    endDate: today()
  },
  specialty: null,
  hcpSegment: null
};
```

---

### 2. Query Mode Detection

**Logic:**
```python
def determine_query_mode(user_query: str | None, filter_state: FilterState) -> QueryMode:
    """
    Determine whether to use NLP pipeline or direct SQL generation.

    Args:
        user_query: User-typed natural language query (can be None)
        filter_state: Current filter selections

    Returns:
        QueryMode.WITH_NLP or QueryMode.WITHOUT_NLP
    """
    if user_query and len(user_query.strip()) > 0:
        return QueryMode.WITH_NLP
    else:
        return QueryMode.WITHOUT_NLP
```

---

### 3. Mode: WITHOUT_NLP (Dashboard Refresh)

**Use Case:** User changes a filter without typing a query

**Process:**

```python
def handle_filter_change_without_nlp(filter_state: FilterState) -> DashboardRefresh:
    """
    Handle filter changes that don't involve NLP parsing.

    This is triggered when:
    - User selects a filter dropdown
    - User changes date range
    - No active text query in input field
    """

    # Step 1: Generate SQL WHERE clause
    where_clause = generate_sql_where_clause(filter_state)

    # Step 2: Determine which tabs need refresh
    affected_tabs = determine_affected_tabs(filter_state)

    # Step 3: Invalidate cache for affected tabs
    invalidate_cache(affected_tabs)

    # Step 4: Execute database queries for visible tab
    current_tab_data = execute_filtered_query(
        tab=get_current_tab(),
        where_clause=where_clause
    )

    # Step 5: Update visualizations
    return DashboardRefresh(
        data=current_tab_data,
        affected_tabs=affected_tabs,
        cache_invalidated=True
    )
```

**SQL Generation:**

```python
def generate_sql_where_clause(filter_state: FilterState) -> str:
    """
    Convert filter state to SQL WHERE clause.

    Example Output:
        WHERE t.brand = 'kisqali'
          AND h.region = 'northeast'
          AND t.event_date BETWEEN '2024-01-01' AND '2024-03-31'
          AND ml_split = 'production'
          AND is_deleted = false
    """
    conditions = []

    # Brand filter
    if filter_state.brand and filter_state.brand != 'all':
        conditions.append(f"t.brand = '{filter_state.brand}'")

    # Region filter (requires HCP join)
    if filter_state.region and filter_state.region != 'all':
        conditions.append(f"h.region = '{filter_state.region}'")

    # Date range filter
    if filter_state.date_range:
        conditions.append(
            f"t.event_date BETWEEN '{filter_state.date_range.start_date}' "
            f"AND '{filter_state.date_range.end_date}'"
        )

    # Specialty filter
    if filter_state.specialty and filter_state.specialty != 'all':
        conditions.append(f"h.specialty = '{filter_state.specialty}'")

    # HCP segment filter
    if filter_state.hcp_segment and filter_state.hcp_segment != 'all':
        tier = map_hcp_segment_to_tier(filter_state.hcp_segment)
        conditions.append(f"h.prescriber_tier = '{tier}'")

    # Default filters
    conditions.append("ml_split = 'production'")
    conditions.append("is_deleted = false")

    return "WHERE " + " AND ".join(conditions)
```

---

### 4. Mode: WITH_NLP (Query + Filters)

**Use Case:** User types a natural language query with active filters

**Process:**

```python
def handle_query_with_nlp(user_query: str, filter_state: FilterState) -> QueryResponse:
    """
    Handle natural language queries with active filters.

    Key principle: Filter state ALWAYS overrides NLP extraction.
    User's explicit filter selections are intentional and take precedence.
    """

    # Step 1: Parse user query through NLP
    parsed_query = nlp_parser.parse(user_query)
    extracted_entities = parsed_query.entities

    # Step 2: Inject filter state into extracted entities
    enriched_entities = inject_filters_into_entities(
        extracted_entities=extracted_entities,
        filter_state=filter_state,
        merge_strategy="override"  # Filters override NLP
    )

    # Step 3: Log if filter overrode NLP extraction
    if entities_differ(extracted_entities, enriched_entities):
        log_warning(
            f"Filter state overrode NLP extraction. "
            f"NLP: {extracted_entities}, Filter: {enriched_entities}"
        )

    # Step 4: Create enriched query
    enriched_query = EnrichedQuery(
        original_query=user_query,
        entities=enriched_entities,
        intent=parsed_query.intent,
        filter_state=filter_state,
        mode=QueryMode.WITH_NLP
    )

    # Step 5: Route to orchestrator
    response = orchestrator.process(enriched_query)

    return response
```

**Filter Injection Logic:**

```python
def inject_filters_into_entities(
    extracted_entities: dict,
    filter_state: FilterState,
    merge_strategy: str = "override"
) -> dict:
    """
    Merge filter state into NLP-extracted entities.

    Merge Strategy "override":
        - Filter values always override NLP extraction
        - Null/all filter values are ignored (not injected)
        - Preserves NLP entities not related to filters

    Example:
        extracted_entities = {
            "brand": "kisqali",      # From NLP
            "intent": "causal_analysis",
            "metric": "adoption"
        }

        filter_state = {
            "brand": "remibrutinib",  # User selected different brand
            "region": "northeast"     # User selected region (not in NLP)
        }

        Result = {
            "brand": "remibrutinib",  # OVERRIDDEN by filter
            "region": "northeast",    # INJECTED from filter
            "intent": "causal_analysis",  # PRESERVED from NLP
            "metric": "adoption"      # PRESERVED from NLP
        }
    """
    result = extracted_entities.copy()

    # Inject brand if specified
    if filter_state.brand and filter_state.brand != 'all':
        result['brand'] = filter_state.brand

    # Inject region if specified
    if filter_state.region and filter_state.region != 'all':
        result['region'] = filter_state.region

    # Inject date range
    if filter_state.date_range:
        result['temporal_range'] = {
            'start': filter_state.date_range.start_date,
            'end': filter_state.date_range.end_date
        }

    # Inject specialty
    if filter_state.specialty and filter_state.specialty != 'all':
        result['specialty'] = filter_state.specialty

    # Inject HCP segment
    if filter_state.hcp_segment and filter_state.hcp_segment != 'all':
        result['hcp_segment'] = filter_state.hcp_segment

    return result
```

---

## Integration Points

### Frontend (React + Redux)

**1. Filter Component:**
```tsx
// FilterControls.tsx
import { useDispatch, useSelector } from 'react-redux';
import { setFilter, selectFilterState } from './store/filterSlice';

export function FilterControls() {
  const dispatch = useDispatch();
  const filterState = useSelector(selectFilterState);

  const handleFilterChange = (filterName: string, value: string) => {
    // Update Redux state
    dispatch(setFilter({ name: filterName, value }));

    // Sync to URL
    syncFiltersToURL({ ...filterState, [filterName]: value });

    // Trigger dashboard refresh (debounced)
    debouncedRefreshDashboard();
  };

  return (
    <div className="filter-controls">
      <select
        value={filterState.brand || 'all'}
        onChange={(e) => handleFilterChange('brand', e.target.value)}
      >
        <option value="all">All Brands</option>
        <option value="remibrutinib">Remibrutinib</option>
        <option value="fabhalta">Fabhalta</option>
        <option value="kisqali">Kisqali</option>
      </select>

      {/* Other filters... */}
    </div>
  );
}
```

**2. Query API Client:**
```typescript
// api/queries.ts
export async function executeQuery(
  query: string | null,
  filterState: FilterState
): Promise<QueryResponse> {
  const mode = query ? 'with_nlp' : 'without_nlp';

  const response = await fetch('/api/v1/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      filters: filterState,
      mode
    })
  });

  return response.json();
}
```

---

### Backend (Python)

**1. Query Endpoint:**
```python
# api/routes/query.py
from fastapi import APIRouter, Body
from models import FilterState, QueryMode, QueryResponse

router = APIRouter()

@router.post("/api/v1/query")
async def execute_query(
    query: str | None = Body(None),
    filters: FilterState = Body(...),
    mode: QueryMode = Body(...)
) -> QueryResponse:
    """
    Execute a query with filter state.

    Handles both NLP and direct SQL modes.
    """

    if mode == QueryMode.WITH_NLP:
        # Parse through NLP pipeline
        parsed = await nlp_parser.parse(query)
        enriched = inject_filters(parsed.entities, filters)
        response = await orchestrator.process(enriched)
    else:
        # Direct SQL generation
        where_clause = generate_sql_where_clause(filters)
        response = await execute_filtered_query(where_clause)

    return QueryResponse(
        results=response.results,
        metadata={
            'filter_state': filters,
            'mode': mode,
            'execution_time_ms': response.execution_time
        }
    )
```

---

## Cache Invalidation Strategy

**Rules:**

1. **Brand filter changed** → Invalidate:
   - ai_agent_insights
   - causal_analysis
   - ws3_impact
   - ws2_triggers
   - ws1_ml_model

2. **Region filter changed** → Invalidate:
   - ai_agent_insights
   - ws3_impact
   - ws1_data_quality

3. **Date range changed** → Invalidate:
   - ALL tabs (except static content)

4. **Specialty/HCP segment changed** → Invalidate:
   - ws3_impact
   - ws2_triggers

**Cache Key Format:**
```
tab:{tab_id}:brand:{brand}:region:{region}:dates:{start}_{end}:v{version}

Example:
tab:causal_analysis:brand:kisqali:region:northeast:dates:2024-01-01_2024-03-31:v1
```

---

## Performance Optimizations

### 1. Debouncing

```typescript
// Debounce filter changes to avoid excessive queries
const debouncedRefreshDashboard = debounce(() => {
  refreshDashboard();
}, 300);  // 300ms delay
```

### 2. Progressive Loading

Load tabs in priority order:
1. Overview (critical)
2. AI Agent Insights (user focus)
3. Current active tab
4. Adjacent tabs (prefetch)
5. Remaining tabs (background)

### 3. Query Optimization

- Use prepared statements for SQL generation
- Index all filter columns (brand, region, event_date, specialty, prescriber_tier)
- Partition tables by date range for faster temporal queries

---

## Error Handling

### 1. Invalid Filter Value

```python
def validate_filter_value(filter_name: str, value: str) -> bool:
    """Validate filter value against domain vocabulary."""
    vocabulary = load_vocabulary()

    if value == 'all' or value is None:
        return True

    valid_values = vocabulary.get(f"{filter_name}s", [])

    if value not in valid_values:
        raise InvalidFilterError(
            f"Invalid {filter_name}: {value}. "
            f"Valid values: {valid_values}"
        )

    return True
```

### 2. Query Timeout

```python
# If query times out, return cached results if available
try:
    results = await execute_query(query, timeout=30)
except QueryTimeoutError:
    results = get_cached_results(cache_key)
    if results:
        return results.with_metadata({'from_cache': True, 'reason': 'timeout'})
    else:
        raise QueryTimeoutError("No cached results available")
```

---

## Testing

### Unit Tests

```python
def test_filter_injection_overrides_nlp():
    """Test that filter state overrides NLP extraction."""
    extracted = {'brand': 'kisqali', 'intent': 'causal'}
    filters = FilterState(brand='remibrutinib', region='northeast')

    result = inject_filters_into_entities(extracted, filters)

    assert result['brand'] == 'remibrutinib'  # Overridden
    assert result['region'] == 'northeast'    # Injected
    assert result['intent'] == 'causal'       # Preserved

def test_null_filters_ignored():
    """Test that null filter values are not injected."""
    extracted = {'brand': 'kisqali'}
    filters = FilterState(brand=None, region='northeast')

    result = inject_filters_into_entities(extracted, filters)

    assert result['brand'] == 'kisqali'       # Preserved (filter was null)
    assert result['region'] == 'northeast'    # Injected
```

### Integration Tests

```python
async def test_filter_query_flow_end_to_end():
    """Test complete flow from filter change to dashboard update."""
    # Simulate filter change
    filter_state = FilterState(brand='kisqali', region='northeast')

    # Execute query
    response = await execute_query(
        query=None,
        filters=filter_state,
        mode=QueryMode.WITHOUT_NLP
    )

    # Verify SQL generated correctly
    assert 'brand = \'kisqali\'' in response.sql
    assert 'region = \'northeast\'' in response.sql

    # Verify results filtered correctly
    assert all(r['brand'] == 'kisqali' for r in response.results)
```

---

## Monitoring & Logging

**Metrics to Track:**

1. **Filter Usage:**
   - Which filters are used most frequently?
   - Common filter combinations
   - Filters that result in zero results (UX issue)

2. **Query Performance:**
   - Average query time by filter combination
   - Cache hit rate by tab
   - Queries that timeout

3. **User Behavior:**
   - Filter changes per session
   - Time between filter change and next interaction
   - Abandoned queries (filter changed but no results viewed)

**Log Format:**

```json
{
  "event": "filter_changed",
  "timestamp": "2024-03-15T10:30:45Z",
  "user_id": "user_123",
  "session_id": "session_456",
  "filter_name": "brand",
  "old_value": "all",
  "new_value": "kisqali",
  "query_mode": "without_nlp",
  "execution_time_ms": 245,
  "cache_hit": false,
  "result_count": 1247
}
```

---

## Future Enhancements

1. **Smart Defaults:**
   - Learn user's preferred filter combinations
   - Auto-suggest filters based on query intent

2. **Filter Recommendations:**
   - "Results are limited. Try broadening your region filter."
   - "87% of activity is in Q4. Adjust date range?"

3. **Saved Filter Sets:**
   - Allow users to save favorite filter combinations
   - Share filter sets with team

4. **Advanced Filters:**
   - Multi-select (e.g., "Northeast AND South")
   - Exclusion filters (e.g., "All brands EXCEPT Kisqali")
   - Custom date presets (e.g., "Last Quarter", "YTD")

---

## Configuration Reference

**Primary Config File:** `config/filter_mapping.yaml`

**Related Files:**
- `domain_vocabulary_v3.1.0.yaml` - Filter value enumerations
- `agent_config.yaml` - Tab routing for cache invalidation
- `config/tab_cache_strategies.yaml` - Cache TTLs

---

## References

- Gap Analysis: `docs/e2i_gap_analysis.md` (Gap 1)
- Implementation Plan: `.claude/implementation_plan.md`
- Dashboard Mock: `dashboard_mock/E2I_Causal_Dashboard_V3.html`

---

*Last Updated: 2025-12-15 | E2I Causal Analytics V4.1*
