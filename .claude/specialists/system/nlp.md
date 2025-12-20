# NLP Specialist Instructions

## Domain Scope
You are the NLP specialist for E2I Causal Analytics. Your scope is LIMITED to:
- `src/nlp/` - All NLP processing modules
- `src/nlp/models/` - Pydantic models for queries
- `config/domain_vocabulary.yaml` - Fixed entity vocabularies

## Critical Constraints

### This is NOT Medical NER
The NLP layer extracts **domain-specific business entities**, not medical terms:

```python
# ✅ CORRECT - Business entities
entities = {
    "brands": ["Remibrutinib", "Fabhalta", "Kisqali"],
    "regions": ["northeast", "south", "midwest", "west"],
    "kpis": ["TRx", "NRx", "conversion_rate", "market_share"],
    "journey_stages": ["diagnosis", "initial_treatment", "maintenance"],
    "time_periods": ["Q1 2024", "last 30 days", "YTD"],
    "hcp_ids": ["HCP-12345"]
}

# ❌ WRONG - Do NOT extract medical entities
# diagnoses, medications, symptoms, procedures
```

### Entity Extraction Strategy
1. **Fuzzy matching** against `domain_vocabulary.yaml` (primary)
2. **Claude API** for complex/ambiguous queries (fallback)
3. **Never** use medical NER libraries (scispaCy, BioBERT, etc.)

## Module Responsibilities

### query_processor.py
Main pipeline orchestrating:
1. Intent classification
2. Entity extraction
3. Query rewriting
4. Ambiguity resolution

### intent_classifier.py
Routes queries to appropriate agents:
```python
class IntentType(Enum):
    CAUSAL = "causal"           # → Tier 2 agents
    EXPLORATORY = "exploratory" # → Orchestrator decides
    COMPARATIVE = "comparative" # → Gap Analyzer
    TREND = "trend"             # → Drift Monitor
    WHAT_IF = "what_if"         # → Causal Impact (counterfactual)
```

### entity_extractor.py
- Load vocabularies from `config/domain_vocabulary.yaml`
- Use `rapidfuzz` for fuzzy matching (threshold: 85%)
- Return `E2IEntities` Pydantic model

### query_rewriter.py
Optimize queries for CausalRAG retrieval:
- Expand abbreviations (TRx → Total Prescriptions)
- Add temporal context if missing
- Normalize entity references

### ambiguity_resolver.py
Handle under-specified queries:
- Missing time range → Default to "last 90 days"
- Missing brand → Ask for clarification
- Missing region → Apply to all regions

## Pydantic Models (src/nlp/models/)

### query_models.py
```python
class ParsedQuery(BaseModel):
    raw_query: str
    intent: IntentType
    entities: E2IEntities
    time_range: TimeRange
    filters: Dict[str, Any]
    confidence: float
    requires_clarification: bool
    clarification_prompt: Optional[str]
```

### entity_models.py
```python
class E2IEntities(BaseModel):
    brands: List[str] = []
    regions: List[str] = []
    kpis: List[str] = []
    time_periods: List[str] = []
    hcp_ids: List[str] = []
    journey_stages: List[str] = []
    workstreams: List[str] = []  # WS1, WS2, WS3
```

## Testing Requirements
All changes must pass:
- `tests/unit/test_nlp/test_entity_extractor.py`
- `tests/unit/test_nlp/test_intent_classifier.py`
- `tests/unit/test_nlp/test_query_processor.py`

## Integration Contracts

### Output Contract (to Orchestrator)
```python
# NLP must return ParsedQuery to orchestrator
parsed = query_processor.process(raw_query)
assert isinstance(parsed, ParsedQuery)
assert parsed.intent in IntentType
```

### Vocabulary Contract
```yaml
# All extracted entities must exist in domain_vocabulary.yaml
# or be flagged as unknown
```

## Example Implementations

### Good Query Processing
```python
def process_query(raw: str) -> ParsedQuery:
    # 1. Classify intent
    intent = intent_classifier.classify(raw)
    
    # 2. Extract entities (NO medical NER)
    entities = entity_extractor.extract(raw, vocab=DOMAIN_VOCAB)
    
    # 3. Check for ambiguity
    if not entities.brands and "brand" not in raw.lower():
        return ParsedQuery(
            requires_clarification=True,
            clarification_prompt="Which brand are you asking about?"
        )
    
    # 4. Rewrite for RAG
    optimized = query_rewriter.rewrite(raw, entities)
    
    return ParsedQuery(...)
```

## Handoff Format
When handing off to other specialists:
```yaml
nlp_handoff:
  parsed_query: <ParsedQuery as dict>
  routing_decision: <IntentType>
  target_agent: <agent_name>
  confidence: <0.0-1.0>
```
