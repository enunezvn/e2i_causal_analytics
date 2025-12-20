# E2I RAG Pipeline: NLP Library Analysis

## Your Requirements Recap

1. **Query Robustness**: Handle typos, abbreviations, vague references, zero context
2. **Cache Invalidation**: Ensure no stale data is ever used

## Current E2I Architecture (from Project Knowledge)

Your existing `entity_extractor.py` already uses:
- **rapidfuzz** for fuzzy matching against `domain_vocabulary.yaml`
- **Claude** for complex/ambiguous query resolution
- Fixed vocabulary lookup with alias support

```python
# Current approach (from project_structure_v2)
from rapidfuzz import fuzz, process

def _extract_from_vocab(self, query: str, vocab: set, threshold: int = 80):
    matches = process.extract(term.lower(), words, scorer=fuzz.ratio)
    if matches and matches[0][1] >= threshold:
        found.append(term)
```

---

## Library Comparison Matrix

| Library | Primary Use Case | Typo Handling | Speed | E2I Relevance | Recommendation |
|---------|-----------------|---------------|-------|---------------|----------------|
| **TextBlob** | Sentiment, basic NLP | ✅ Built-in spell correction | Medium | ⚠️ Low | Skip |
| **Sumy** | Text summarization | ❌ None | Fast | ❌ Not relevant | Skip |
| **Pattern** | Web mining, NLP | ⚠️ Basic | Medium | ❌ Outdated (Python 2 legacy) | Skip |
| **spaCy Matcher** | Rule-based matching | ❌ Exact match only | Very Fast | ✅ Useful for patterns | Consider |
| **fastText** | Embeddings, OOV handling | ✅ Subword-aware | Fast | ✅ High | **Recommended** |
| **KeyBERT** | Keyword extraction | ❌ None | Slow | ⚠️ Limited | Skip |
| **PollyNLP** | ❌ Does not exist | N/A | N/A | N/A | Invalid |
| **Flair** | NER, embeddings | ✅ Via embeddings | Medium | ⚠️ Overkill | Skip |
| **YAKE** | Unsupervised keywords | ❌ None | Fast | ⚠️ Limited | Skip |
| **Stanza** | Multi-lingual NLP | ❌ None | Slow | ❌ Overkill | Skip |

---

## Detailed Analysis

### ✅ Recommended: fastText

**Why it fits E2I:**

1. **Subword Embeddings Handle Typos Naturally**
   ```python
   # "Remibrutinib" typo as "Remibritunib" 
   # fastText breaks into character n-grams:
   # ["rem", "emi", "mib", "ibr", "bru", "rut", "uti", "tin", "ini", "nib"]
   # Similar n-grams = similar vectors = matched!
   ```

2. **Zero-Shot Abbreviation Handling**
   - "ROI" → clusters near "return on investment" 
   - "MW" → clusters near "Midwest"
   - No explicit alias mapping needed

3. **Extremely Fast at Inference**
   - 10,000 queries/second on CPU
   - No GPU required

4. **Small Memory Footprint**
   - Can use compressed models (~50MB)

**Implementation for E2I:**

```python
# src/nlp/query_normalizer.py

import fasttext
from typing import Optional, List, Tuple

class E2IQueryNormalizer:
    """
    Handles typos/abbreviations using fastText embeddings
    trained on E2I domain vocabulary.
    """
    
    def __init__(self, model_path: str = "models/e2i_fasttext.bin"):
        self.model = fasttext.load_model(model_path)
        self.vocab_embeddings = self._precompute_vocab_embeddings()
    
    def _precompute_vocab_embeddings(self) -> dict:
        """Cache embeddings for all known terms."""
        vocab_terms = [
            # Brands
            "Remibrutinib", "Fabhalta", "Kisqali",
            # Regions  
            "Northeast", "Southeast", "Midwest", "West",
            # KPIs
            "conversion_rate", "hcp_coverage", "time_to_therapy",
            # Agents
            "causal_impact", "gap_analyzer", "orchestrator",
            # ... etc
        ]
        return {term: self.model.get_word_vector(term) for term in vocab_terms}
    
    def normalize_term(
        self, 
        query_term: str, 
        threshold: float = 0.75
    ) -> Tuple[Optional[str], float]:
        """
        Find closest vocabulary term for potentially misspelled input.
        
        Returns: (matched_term, confidence_score)
        """
        query_vec = self.model.get_word_vector(query_term)
        
        best_match = None
        best_score = 0.0
        
        for vocab_term, vocab_vec in self.vocab_embeddings.items():
            # Cosine similarity
            score = self._cosine_sim(query_vec, vocab_vec)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = vocab_term
        
        return best_match, best_score
    
    def expand_abbreviations(self, query: str) -> str:
        """
        Expand abbreviations using embedding proximity.
        """
        # E2I-specific abbreviation patterns
        known_expansions = {
            "ROI": "return on investment",
            "HCP": "healthcare provider", 
            "Rx": "prescription",
            "TtT": "time to therapy",
            "CFR": "counterfactual failure rate",
        }
        
        for abbr, expansion in known_expansions.items():
            if abbr.lower() in query.lower():
                # Only expand if context suggests abbreviation use
                query = query.replace(abbr, expansion)
        
        return query
```

---

### ⚠️ Consider: spaCy Matcher (Complementary)

**Use Case:** Pattern-based ID extraction (already in your design)

```python
# Enhance your existing pattern extraction
from spacy.matcher import Matcher
import spacy

nlp = spacy.blank("en")
matcher = Matcher(nlp.vocab)

# E2I-specific patterns
patterns = {
    "HCP_ID": [[{"TEXT": {"REGEX": r"HCP-\d{5}"}}]],
    "PATIENT_ID": [[{"TEXT": {"REGEX": r"PAT-\d{5}"}}]],
    "QUARTER": [[{"LOWER": {"IN": ["q1", "q2", "q3", "q4"]}}]],
    "BRAND": [[{"LOWER": {"IN": ["remibrutinib", "fabhalta", "kisqali"]}}]],
}

for name, pattern in patterns.items():
    matcher.add(name, pattern)
```

**Verdict:** Already covered by your regex patterns in `_extract_patterns()`. Only add if you need more complex linguistic patterns.

---

### ❌ Skip These Libraries

| Library | Why Skip for E2I |
|---------|------------------|
| **TextBlob** | Generic spell correction won't know "Remibrutinib". Your fastText approach is domain-aware. |
| **Sumy** | Text summarization ≠ query understanding. You're not summarizing documents. |
| **Pattern** | Outdated (Python 2 era), poor maintenance, replaced by spaCy. |
| **KeyBERT** | Keyword extraction from documents, not query normalization. |
| **Flair** | Heavy transformer models for NER—overkill when you have fixed vocabulary. |
| **YAKE** | Unsupervised keyword extraction—you have supervised vocabulary. |
| **Stanza** | Stanford NLP's multi-lingual suite—unnecessary complexity for English-only pharma terms. |
| **PollyNLP** | **Does not exist**—possibly confused with AWS Polly (text-to-speech) or a typo. |

---

## Cache Invalidation Strategy

This is architectural, not library-dependent. Here's a recommended approach:

### 1. Embedding Cache with Version Keys

```python
# src/rag/cache_manager.py

from datetime import datetime
from typing import Optional
import hashlib
import redis

class E2IRagCache:
    """
    Time-aware cache with automatic invalidation.
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.default_ttl = 3600  # 1 hour
        
    def _make_key(self, query: str, data_version: str) -> str:
        """
        Key includes data version to auto-invalidate on schema changes.
        """
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        return f"rag:v{data_version}:{query_hash}"
    
    def get(
        self, 
        query: str, 
        data_version: str,
        max_age_seconds: Optional[int] = None
    ) -> Optional[dict]:
        """
        Get cached result, respecting freshness requirements.
        """
        key = self._make_key(query, data_version)
        cached = self.redis.hgetall(key)
        
        if not cached:
            return None
            
        # Check age if max_age specified
        if max_age_seconds:
            cached_at = float(cached.get(b'timestamp', 0))
            age = datetime.utcnow().timestamp() - cached_at
            if age > max_age_seconds:
                self.redis.delete(key)
                return None
        
        return {
            'results': cached.get(b'results'),
            'timestamp': cached.get(b'timestamp'),
        }
    
    def set(
        self, 
        query: str, 
        data_version: str, 
        results: dict,
        ttl: Optional[int] = None
    ):
        """Store with timestamp for age checking."""
        key = self._make_key(query, data_version)
        self.redis.hset(key, mapping={
            'results': results,
            'timestamp': datetime.utcnow().timestamp(),
        })
        self.redis.expire(key, ttl or self.default_ttl)
    
    def invalidate_all(self, data_version: str):
        """Nuclear option: clear all cache for a version."""
        pattern = f"rag:v{data_version}:*"
        for key in self.redis.scan_iter(pattern):
            self.redis.delete(key)
```

### 2. Data Version Tracking

```python
# Track when underlying data changes
class DataVersionTracker:
    """
    Tracks data freshness for cache invalidation.
    """
    
    def __init__(self, supabase_client):
        self.client = supabase_client
        
    def get_current_version(self) -> str:
        """
        Version based on latest data modification.
        """
        # Query max updated_at across key tables
        tables = ['causal_paths', 'agent_activities', 'business_metrics']
        max_updated = None
        
        for table in tables:
            result = self.client.table(table)\
                .select("updated_at")\
                .order("updated_at", desc=True)\
                .limit(1)\
                .execute()
            
            if result.data:
                ts = result.data[0]['updated_at']
                if not max_updated or ts > max_updated:
                    max_updated = ts
        
        # Version = date hash
        return hashlib.md5(str(max_updated).encode()).hexdigest()[:8]
```

### 3. Query-Level Freshness

```python
# In your RAG retriever
class CausalRAG:
    def retrieve(
        self, 
        query: str, 
        require_fresh: bool = False,
        max_cache_age: int = 3600
    ):
        data_version = self.version_tracker.get_current_version()
        
        if not require_fresh:
            cached = self.cache.get(
                query, 
                data_version,
                max_age_seconds=max_cache_age
            )
            if cached:
                return cached['results']
        
        # Fresh retrieval
        results = self._retrieve_fresh(query)
        self.cache.set(query, data_version, results)
        return results
```

---

## Final Recommendations

### For Query Robustness (Typos, Abbreviations, Vague References)

| Layer | Tool | Purpose |
|-------|------|---------|
| 1. Pre-processing | **fastText** (new) | Normalize typos via subword embeddings |
| 2. Entity Extraction | **rapidfuzz** (existing) | Fuzzy match against vocabulary |
| 3. Ambiguity Resolution | **Claude** (existing) | Handle complex/zero-context queries |

### For Cache Invalidation

| Component | Implementation |
|-----------|----------------|
| Cache Backend | Redis with TTL |
| Version Key | Include data version hash in cache keys |
| Freshness Check | Timestamp-based age validation |
| Nuclear Invalidation | Pattern-based key deletion |

### Implementation Priority

1. **Phase 1**: Add fastText for typo normalization (high impact, low effort)
2. **Phase 2**: Implement versioned cache with Redis
3. **Phase 3**: Add real-time invalidation triggers on data updates

---

## Training a Domain-Specific fastText Model

```bash
# 1. Prepare training corpus from your domain vocabulary + synonyms
cat > e2i_corpus.txt << EOF
Remibrutinib brand therapy drug treatment
Fabhalta brand therapy drug treatment
Kisqali brand therapy drug treatment
Northeast region NE territory area
Southeast region SE territory area
Midwest region MW territory area
West region territory area Pacific
conversion rate KPI metric performance
hcp coverage healthcare provider KPI metric
time to therapy TtT KPI metric latency
causal impact agent analyzer inference
gap analyzer agent performance metric
orchestrator agent coordinator router
EOF

# 2. Train fastText model
fasttext supervised -input e2i_corpus.txt -output e2i_model -dim 50 -epoch 25 -wordNgrams 3

# Or using Python:
import fasttext
model = fasttext.train_unsupervised(
    'e2i_corpus.txt',
    model='skipgram',
    dim=50,
    epoch=25,
    wordNgrams=3,
    minCount=1
)
model.save_model('models/e2i_fasttext.bin')
```

---

## Summary

| Your Requirement | Solution |
|-----------------|----------|
| Handle typos | fastText subword embeddings |
| Handle abbreviations | Explicit mapping + fastText clustering |
| Handle vague references | Claude for disambiguation (existing) |
| Handle zero context | Default to broad search + ask for clarification |
| Cache invalidation | Redis + data version keys + TTL |
| No stale data | Timestamp-based freshness checks |

**Net New Libraries to Add:** 
- `fasttext` (for typo handling)
- `redis` (for caching, if not already present)

**Libraries to Skip:** TextBlob, Sumy, Pattern, KeyBERT, Flair, YAKE, Stanza, PollyNLP
