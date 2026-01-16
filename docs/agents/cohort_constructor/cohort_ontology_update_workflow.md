# CohortConstructor Ontology Update Workflow

## Overview

The cohort ontology (`cohort_vocabulary.yaml`) requires systematic updates as new clinical codes, drugs, and usage patterns emerge. This document outlines the update process, validation, and deployment workflow.

---

## Update Triggers & Sources

### 1. **Scheduled Clinical Code Updates** (Quarterly)

**Source of Truth:**
- **ICD-10-CM:** CMS releases (annual, October 1)
- **CPT/HCPCS:** AMA releases (annual, January 1)
- **NDC:** FDA daily updates via OpenFDA API
- **LOINC:** Regenstrief Institute releases (biannual)

**Process:**
```python
# Automated sync script
def sync_clinical_codes():
    """
    Quarterly job to sync external code systems
    """
    # 1. Fetch latest codes from authoritative sources
    icd10_codes = fetch_cms_icd10_updates()
    cpt_codes = fetch_ama_cpt_updates()
    ndc_codes = fetch_fda_ndc_updates()
    loinc_codes = fetch_loinc_updates()
    
    # 2. Validate against existing ontology
    new_codes = validate_new_codes(
        existing=load_ontology(),
        incoming={
            'diagnosis_codes': icd10_codes,
            'procedure_codes': cpt_codes,
            'drug_codes': ndc_codes,
            'lab_codes': loinc_codes
        }
    )
    
    # 3. Generate change proposal
    proposal = generate_ontology_update_proposal(new_codes)
    
    # 4. Require human review before merge
    notify_data_governance_team(proposal)
    
    return proposal
```

---

### 2. **New Drug/Brand Additions** (Event-Driven)

**Trigger:** FDA approval, internal product launch

**Process:**
1. **Product team notification** → Jira ticket created
2. **Data governance review** → Brand name, aliases, NDC codes
3. **Update ontology** → Add to `brands` section with aliases
4. **Schema sync** → Update `brand_type` ENUM in Supabase
5. **Deploy** → Version bump, regression tests, deploy

**Example Update:**
```yaml
# Before (v1.0.0)
brands:
  values:
    - Remibrutinib
    - Fabhalta
    - Kisqali

# After (v1.1.0) - New brand added
brands:
  values:
    - Remibrutinib
    - Fabhalta
    - Kisqali
    - Cosentyx  # NEW: Added 2025-01-15
  aliases:
    Cosentyx: [cos, cosentyx, secukinumab]  # NEW
```

---

### 3. **User-Driven Pattern Discovery** (Continuous)

**Source:** Query logs, user feedback, support tickets

**Process:**
```python
class OntologyEnrichmentPipeline:
    """
    Continuous learning from user queries
    """
    
    def analyze_query_logs(self, days=30):
        """
        Mine query logs for new patterns
        """
        queries = fetch_user_queries(last_n_days=days)
        
        # 1. Extract unrecognized terms
        unknown_terms = []
        for query in queries:
            parsed = parse_query(query, ontology=self.vocab)
            if parsed.has_unrecognized_entities:
                unknown_terms.append(parsed.unrecognized)
        
        # 2. Cluster similar terms
        clusters = cluster_similar_terms(unknown_terms)
        
        # 3. Generate alias suggestions
        suggestions = []
        for cluster in clusters:
            canonical = find_canonical_term(cluster, self.vocab)
            if canonical:
                suggestions.append({
                    'canonical': canonical,
                    'aliases': cluster,
                    'frequency': len(cluster),
                    'example_queries': get_example_queries(cluster)
                })
        
        return suggestions
    
    def generate_enrichment_pr(self, suggestions):
        """
        Auto-generate PR for ontology enrichment
        """
        if len(suggestions) < 5:
            return None  # Wait for more signal
        
        # Sort by frequency
        top_suggestions = sorted(
            suggestions, 
            key=lambda x: x['frequency'], 
            reverse=True
        )[:10]
        
        # Generate YAML additions
        yaml_additions = self._format_as_yaml(top_suggestions)
        
        # Create PR with human review required
        pr = create_github_pr(
            branch='ontology-enrichment-auto',
            title='[Auto] Ontology enrichment from user queries',
            body=self._generate_pr_description(top_suggestions),
            files={'cohort_vocabulary.yaml': yaml_additions},
            reviewers=['data-governance-team']
        )
        
        return pr
```

**Example Enrichment:**
```yaml
# User queries revealed "bloodwork" used frequently
# Add to aliases section:

aliases:
  lab_test:
    - lab
    - test
    - laboratory
    - bloodwork     # NEW: Added from query logs (freq=247)
    - blood panel   # NEW: Added from query logs (freq=89)
```

---

### 4. **Schema Evolution** (As Needed)

**Trigger:** Database schema changes (new tables, ENUMs, columns)

**Process:**
1. Schema migration runs (e.g., `011_add_biomarker_table.sql`)
2. **Automatic detection** via schema diff
3. **Ontology update required** before deployment gate passes
4. Update `database_tables` section in vocabulary
5. Add any new ENUMs to relevant sections

**Enforcement:**
```python
# CI/CD pipeline check
def validate_ontology_schema_sync():
    """
    Block deployment if ontology is out of sync with schema
    """
    db_enums = fetch_supabase_enums()
    vocab_enums = load_vocabulary_enums()
    
    mismatches = compare_enums(db_enums, vocab_enums)
    
    if mismatches:
        raise DeploymentBlockedError(
            f"Ontology out of sync with schema: {mismatches}\n"
            f"Update cohort_vocabulary.yaml before deploying."
        )
```

---

## Validation Pipeline

### Pre-Merge Checks (Automated)

```python
class OntologyValidator:
    """
    Run before merging ontology updates
    """
    
    def validate(self, old_vocab, new_vocab):
        """
        Comprehensive validation suite
        """
        checks = [
            self.check_no_breaking_changes(old_vocab, new_vocab),
            self.check_no_duplicates(new_vocab),
            self.check_valid_yaml_structure(new_vocab),
            self.check_enum_sync_with_db(new_vocab),
            self.check_alias_consistency(new_vocab),
            self.check_example_queries_valid(new_vocab),
            self.check_version_bump(old_vocab, new_vocab)
        ]
        
        failures = [c for c in checks if not c.passed]
        if failures:
            raise ValidationError(failures)
        
        return True
    
    def check_no_breaking_changes(self, old, new):
        """
        Ensure backward compatibility
        """
        # Check: no removals from enum values
        for enum_name in old.keys():
            old_values = set(old[enum_name]['values'])
            new_values = set(new[enum_name]['values'])
            
            removed = old_values - new_values
            if removed:
                return ValidationResult(
                    passed=False,
                    error=f"Breaking change: Removed values from {enum_name}: {removed}"
                )
        
        return ValidationResult(passed=True)
    
    def check_enum_sync_with_db(self, vocab):
        """
        Ensure ENUMs match Supabase schema
        """
        db_enums = fetch_db_enums()
        
        for enum_name, enum_data in vocab.items():
            if 'values' not in enum_data:
                continue
            
            vocab_values = set(enum_data['values'])
            db_values = set(db_enums.get(enum_name, []))
            
            if vocab_values != db_values:
                return ValidationResult(
                    passed=False,
                    error=f"ENUM mismatch for {enum_name}:\n"
                          f"  Vocab: {vocab_values}\n"
                          f"  DB: {db_values}"
                )
        
        return ValidationResult(passed=True)
```

### Post-Merge Validation (Automated + Manual)

```python
def post_deployment_validation():
    """
    Run after ontology update is deployed
    """
    # 1. Smoke tests
    run_smoke_tests()
    
    # 2. Query regression tests
    regression_results = run_query_regression_suite()
    
    # 3. Alert if query parsing degraded
    current_parse_rate = calculate_parse_success_rate()
    baseline_parse_rate = fetch_baseline_parse_rate()
    
    if current_parse_rate < baseline_parse_rate - 0.05:
        alert_team(
            f"Query parsing degraded after ontology update:\n"
            f"  Before: {baseline_parse_rate:.2%}\n"
            f"  After: {current_parse_rate:.2%}"
        )
```

---

## Version Control Strategy

### Semantic Versioning

```
cohort_vocabulary v[MAJOR].[MINOR].[PATCH]

MAJOR: Breaking changes (removed values, schema incompatibility)
MINOR: New features (added values, new sections)
PATCH: Fixes (typos, alias additions, documentation)
```

**Examples:**
- `v1.0.0` → `v1.0.1`: Added aliases for "bloodwork"
- `v1.0.1` → `v1.1.0`: Added new brand "Cosentyx"
- `v1.1.0` → `v2.0.0`: Removed deprecated `journey_stages` values

### Git Workflow

```bash
# Branch naming convention
feature/ontology-add-cosentyx
fix/ontology-typo-diagnosis-codes
chore/ontology-quarterly-icd10-sync

# Commit message format
feat(ontology): Add Cosentyx brand and aliases

- Added Cosentyx to brands.values
- Added aliases: cos, cosentyx, secukinumab
- Updated schema ENUM brand_type
- Version bump: v1.0.0 -> v1.1.0

Closes JIRA-1234
```

---

## Backward Compatibility

### Graceful Handling of Unknown Values

```python
class CohortParser:
    """
    Gracefully handle ontology mismatches
    """
    
    def parse_criterion(self, criterion_dict, vocab):
        """
        Parse criterion with fallback for unknown values
        """
        criterion_type = criterion_dict.get('type')
        
        # Check if type is in current ontology
        if criterion_type not in vocab['criterion_types']['values']:
            # Log for enrichment pipeline
            log_unknown_value(
                section='criterion_types',
                value=criterion_type,
                context=criterion_dict
            )
            
            # Attempt fuzzy match
            fuzzy_match = find_fuzzy_match(
                criterion_type, 
                vocab['criterion_types']['values']
            )
            
            if fuzzy_match and fuzzy_match.confidence > 0.8:
                logger.info(
                    f"Fuzzy matched '{criterion_type}' -> '{fuzzy_match.value}'"
                )
                criterion_type = fuzzy_match.value
            else:
                # Fail gracefully with informative error
                raise UnknownCriterionTypeError(
                    f"Unknown criterion type: '{criterion_type}'. "
                    f"Valid types: {vocab['criterion_types']['values']}. "
                    f"This may indicate an outdated ontology."
                )
        
        return criterion_type
```

---

## Deployment Checklist

### Before Deploying Ontology Update

- [ ] Version bumped in YAML header
- [ ] Changelog updated
- [ ] All validation checks passed
- [ ] Schema ENUMs synced (if applicable)
- [ ] Regression tests passed
- [ ] Human review completed (for new clinical codes)
- [ ] Documentation updated
- [ ] Stakeholders notified (for MAJOR versions)

### After Deployment

- [ ] Smoke tests passed
- [ ] Query parse rate monitored (24 hours)
- [ ] User feedback monitored
- [ ] Performance metrics stable
- [ ] Rollback plan ready (if needed)

---

## Monitoring & Observability

### Key Metrics

```python
class OntologyHealthMetrics:
    """
    Track ontology effectiveness over time
    """
    
    def calculate_metrics(self):
        return {
            # Coverage
            'query_parse_success_rate': self._parse_success_rate(),
            'unknown_term_frequency': self._unknown_term_rate(),
            
            # Usage
            'most_used_enums': self._most_used_enums(),
            'unused_enums': self._unused_enums(),
            
            # Quality
            'alias_match_rate': self._alias_match_rate(),
            'fuzzy_match_rate': self._fuzzy_match_rate(),
            
            # Freshness
            'days_since_last_update': self._days_since_update(),
            'pending_code_updates': self._pending_updates(),
        }
    
    def _parse_success_rate(self):
        """
        % of queries successfully parsed
        """
        total = count_queries(last_30_days=True)
        successful = count_queries(
            last_30_days=True, 
            parsed_successfully=True
        )
        return successful / total if total > 0 else 0
    
    def _unknown_term_rate(self):
        """
        % of queries with unrecognized terms
        """
        total = count_queries(last_30_days=True)
        with_unknown = count_queries(
            last_30_days=True,
            has_unknown_terms=True
        )
        return with_unknown / total if total > 0 else 0
```

### Alerts

```python
# Set up alerts for ontology health
ALERT_THRESHOLDS = {
    'query_parse_success_rate': 0.85,  # Alert if < 85%
    'unknown_term_frequency': 0.15,     # Alert if > 15%
    'days_since_last_update': 120,      # Alert if > 4 months
}

def check_ontology_health():
    metrics = OntologyHealthMetrics().calculate_metrics()
    
    alerts = []
    
    if metrics['query_parse_success_rate'] < ALERT_THRESHOLDS['query_parse_success_rate']:
        alerts.append({
            'severity': 'high',
            'message': f"Query parse rate dropped to {metrics['query_parse_success_rate']:.2%}"
        })
    
    if metrics['unknown_term_frequency'] > ALERT_THRESHOLDS['unknown_term_frequency']:
        alerts.append({
            'severity': 'medium',
            'message': f"High unknown term rate: {metrics['unknown_term_frequency']:.2%}. "
                      f"Consider running enrichment pipeline."
        })
    
    if alerts:
        notify_data_governance_team(alerts)
```

---

## Governance

### Roles & Responsibilities

| Role | Responsibility |
|------|----------------|
| **Data Governance Team** | Approve all ontology changes; quarterly review |
| **Clinical SMEs** | Validate new clinical codes and aliases |
| **Engineering Team** | Implement automated sync; maintain validation pipeline |
| **Product Team** | Notify of new drug launches; prioritize user feedback |

### Review Cadence

- **Weekly:** Review auto-generated enrichment suggestions
- **Monthly:** Review unknown term reports
- **Quarterly:** Sync with external code systems (ICD-10, CPT, NDC, LOINC)
- **Annually:** Comprehensive ontology audit

---

## Example: Full Update Workflow

### Scenario: Quarterly ICD-10 Update

```bash
# 1. Automated sync runs (cron job)
python scripts/sync_clinical_codes.py --source icd10 --dry-run

# 2. Review changes
cat proposals/icd10_update_2025_q1.yaml

# Output:
# New Codes (23):
#   L50.8 - Other urticaria (new subtype)
#   D59.8 - Other acquired hemolytic anemias
#   ...
# 
# Deprecated Codes (2):
#   L50.9 - Urticaria, unspecified (use L50.8 instead)

# 3. Human review by clinical SME
# SME marks L50.8 as relevant for CSU cohorts

# 4. Apply changes
python scripts/sync_clinical_codes.py --source icd10 --apply

# 5. Auto-generates PR
git checkout -b chore/ontology-icd10-q1-2025
# ... updates cohort_vocabulary.yaml ...
# ... updates migration 012_add_new_icd10_codes.sql ...

# 6. CI/CD runs validation
pytest tests/test_ontology_validation.py
pytest tests/test_schema_sync.py
pytest tests/test_query_regression.py

# 7. Manual review + approval
# Data Governance reviews PR

# 8. Merge to main
git merge chore/ontology-icd10-q1-2025

# 9. Deploy
# Version: v1.1.0 -> v1.2.0
# Changelog updated
# Deployed to staging -> production

# 10. Post-deployment monitoring
python scripts/monitor_ontology_health.py --days 7
```

---

## Best Practices

### ✅ DO

- **Version control everything** - Track all changes in Git
- **Validate before merge** - Run comprehensive validation suite
- **Sync with schema** - Keep ENUMs aligned with database
- **Document changes** - Maintain detailed changelog
- **Monitor impact** - Track query parse rates after updates
- **Learn from users** - Mine query logs for enrichment opportunities
- **Use semantic versioning** - Signal breaking vs. additive changes

### ❌ DON'T

- **Don't remove values** - Without major version bump and deprecation notice
- **Don't bypass validation** - Always run validation pipeline
- **Don't ignore unknown terms** - They're signals for ontology gaps
- **Don't forget aliases** - User terminology varies widely
- **Don't update manually in production** - Always go through CI/CD
- **Don't skip human review** - Especially for clinical codes

---

## Tools & Scripts

### Required Scripts

```bash
scripts/
├── sync_clinical_codes.py          # Quarterly code sync
├── validate_ontology.py            # Pre-merge validation
├── generate_enrichment_pr.py       # Auto-generate alias PRs
├── monitor_ontology_health.py      # Post-deployment monitoring
├── compare_ontology_versions.py    # Version diff tool
└── migrate_data_for_ontology.py    # Data migration helper
```

### GitHub Actions Workflow

```yaml
# .github/workflows/ontology_validation.yml
name: Validate Ontology Update

on:
  pull_request:
    paths:
      - 'cohort_vocabulary.yaml'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Validate YAML syntax
        run: python scripts/validate_ontology.py --syntax
      
      - name: Check version bump
        run: python scripts/validate_ontology.py --version-bump
      
      - name: Validate schema sync
        run: python scripts/validate_ontology.py --schema-sync
        env:
          SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
          SUPABASE_KEY: ${{ secrets.SUPABASE_KEY }}
      
      - name: Run regression tests
        run: pytest tests/test_query_regression.py
      
      - name: Check for breaking changes
        run: python scripts/validate_ontology.py --breaking-changes
      
      - name: Generate change summary
        run: python scripts/compare_ontology_versions.py --pr-comment
```

---

## Summary

Ontology updates follow a **structured, validated, and monitored** process:

1. **Triggers:** Scheduled (quarterly), event-driven (new drugs), continuous (user patterns)
2. **Validation:** Automated checks + human review
3. **Versioning:** Semantic versioning with changelog
4. **Deployment:** CI/CD with rollback capability
5. **Monitoring:** Parse rates, unknown terms, usage metrics

This ensures the ontology stays **current, accurate, and backward-compatible** while supporting rapid evolution of clinical knowledge and user needs.
