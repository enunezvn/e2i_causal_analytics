# E2I Causal Analytics: MIPROv2 → GEPA Migration Plan
## DSPy Optimizer Upgrade for 18-Agent 6-Tier Architecture
### Version 1.0 • December 2025

---

## Implementation Status

| Field | Value |
|-------|-------|
| **Status** | ✅ Complete |
| **Started** | 2025-12-27 |
| **Completed** | 2025-12-27 |
| **Progress Tracker** | [GEPA_Migration_Progress.md](./GEPA_Migration_Progress.md) |
| **Implementation Plan** | [.claude/plans/tranquil-wiggling-salamander.md](../.claude/plans/tranquil-wiggling-salamander.md) |

### Migration Summary
- **Phases Completed**: 11 (0-10)
- **Files Created**: 27
- **Files Modified**: 4
- **Tests Added**: 60+
- **Issues Encountered**: 0

### Key Deliverables
- `src/optimization/gepa/` - Complete GEPA optimizer module
- `database/ml/023_gepa_optimization_tables.sql` - Database schema
- `config/gepa_config.yaml` - GEPA configuration
- `scripts/gepa_pilot.py` - Pilot optimization script
- Integration with Feedback Learner and Cognitive RAG
- MLOps integration (MLflow, Opik, RAGAS)

---

## Executive Summary

This document outlines the migration from DSPy's MIPROv2 optimizer to GEPA (Generative Evolutionary Prompting with AI) for the E2I Causal Analytics platform. GEPA's reflective evolution approach aligns exceptionally well with E2I's existing feedback loop architecture, particularly the Feedback Learner ↔ Experiment Designer ↔ Causal Impact cycle.

**Key Benefits:**
- **10%+ performance improvement** over MIPROv2 (per arxiv:2507.19457)
- **Fewer training examples required** - critical for expensive pharmaceutical domain labels
- **Rich textual feedback utilization** - leverages existing ExperimentKnowledgeStore
- **Joint tool optimization** - improves DoWhy/EconML tool selection
- **Pareto frontier for multi-objective KPIs** - handles competing brand metrics naturally

**Migration Scope:**
- 18 agents across 6 tiers
- Primary focus: 5 Hybrid/Deep agents (highest GEPA ROI)
- Secondary: 13 Standard agents (lighter optimization)

---

## Part 1: Architecture Alignment Analysis

### 1.1 Agent Classification → GEPA Strategy

| Agent Type | Count | GEPA Approach | Feedback Richness |
|------------|-------|---------------|-------------------|
| **Hybrid** | 4 | Full GEPA + Tool Optimization | High (computation traces + LLM) |
| **Deep** | 2 | Full GEPA with extended reflection | Very High (reasoning chains) |
| **Standard** | 12 | GEPA `auto="light"` | Medium (SLA + metrics) |

### 1.2 Tier-to-GEPA Mapping

```
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 0: ML Foundation (7 Standard Agents)                         │
│  └─ GEPA Strategy: auto="light", no tool optimization              │
│     Focus: Feature store queries, model deployment accuracy        │
├─────────────────────────────────────────────────────────────────────┤
│  TIER 1: Coordination (1 Standard Agent)                           │
│  └─ GEPA Strategy: auto="medium", routing accuracy feedback        │
│     Focus: Agent selection precision, synthesis quality            │
├─────────────────────────────────────────────────────────────────────┤
│  TIER 2: Causal Analytics (3 Agents: 1 Hybrid, 2 Standard)         │
│  └─ GEPA Strategy: FULL for causal_impact, light for others        │
│     Focus: Refutation pass rate, CATE validity, gap identification │
├─────────────────────────────────────────────────────────────────────┤
│  TIER 3: Monitoring (3 Agents: 1 Hybrid, 2 Standard)               │
│  └─ GEPA Strategy: FULL for experiment_designer, light for others  │
│     Focus: Power analysis accuracy, drift detection precision      │
├─────────────────────────────────────────────────────────────────────┤
│  TIER 4: ML Predictions (2 Standard Agents)                        │
│  └─ GEPA Strategy: auto="light"                                    │
│     Focus: AUC targets, ROI optimization                           │
├─────────────────────────────────────────────────────────────────────┤
│  TIER 5: Self-Improvement (2 Deep Agents)                          │
│  └─ GEPA Strategy: FULL with extended reflection                   │
│     Focus: Explanation clarity, prompt optimization quality        │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 Existing Feedback Loop → GEPA Integration

```
┌──────────────────┐    Design Context    ┌──────────────────┐
│                  │◄────────────────────│                  │
│   Experiment     │                      │    Feedback      │
│    Designer      │                      │    Learner       │
│    (Hybrid)      │                      │    (Deep)        │
│                  │────────────────────►│                  │
└────────┬─────────┘    Validation        └────────▲─────────┘
         │             Outcomes                    │
         │                                         │
         │  A/B Test                    Validation │
         │  Design                       Results   │
         ▼                                         │
┌──────────────────┐                    ┌──────────┴─────────┐
│                  │──────────────────►│                    │
│   Causal Impact  │   Causal Estimate  │   Experiment       │
│     (Hybrid)     │   + Refutation     │   KnowledgeStore   │
│                  │◄──────────────────│                    │
└──────────────────┘   Past Lessons     └────────────────────┘

                    ┌─────────────────────┐
                    │  GEPA Optimizer     │
                    │  ─────────────────  │
                    │  • Captures traces  │
                    │  • Reflects on      │
                    │    validation fails │
                    │  • Proposes better  │
                    │    instructions     │
                    │  • Evolves tools    │
                    └─────────────────────┘
```

---

## Part 2: GEPA Feedback Metrics Implementation

### 2.1 Core Metric Architecture

```python
# src/optimization/gepa/metrics/__init__.py

from typing import Union, Optional
from dspy import Example, Prediction
from dspy.teleprompt.gepa.gepa import ScoreWithFeedback

# Type alias for GEPA traces
DSPyTrace = list[tuple]

class E2IGEPAMetric:
    """Base class for E2I GEPA feedback metrics."""
    
    def __call__(
        self,
        gold: Example,
        pred: Prediction,
        trace: Optional[DSPyTrace],
        pred_name: Optional[str],
        pred_trace: Optional[DSPyTrace]
    ) -> Union[float, ScoreWithFeedback]:
        raise NotImplementedError
```

### 2.2 Tier 2: Causal Impact Metric (Primary Priority)

```python
# src/optimization/gepa/metrics/causal_impact_metric.py

from dataclasses import dataclass
from typing import Optional, Union
import dspy
from dspy import Example, Prediction

@dataclass
class CausalFeedbackComponents:
    """Structured feedback for causal inference."""
    refutation_feedback: str
    sensitivity_feedback: str
    methodology_feedback: str
    business_feedback: str

class CausalImpactGEPAMetric:
    """
    GEPA metric for Causal Impact Hybrid agent.
    
    Scoring breakdown:
    - Refutation tests (0.30): All 5 DoWhy tests must pass
    - Sensitivity analysis (0.25): E-value robustness
    - Methodology (0.25): DAG validity, estimation method selection
    - Business relevance (0.20): KPI attribution, actionability
    """
    
    REFUTATION_WEIGHT = 0.30
    SENSITIVITY_WEIGHT = 0.25
    METHODOLOGY_WEIGHT = 0.25
    BUSINESS_WEIGHT = 0.20
    
    # Refutation test types from domain_vocabulary.yaml
    REFUTATION_TESTS = [
        "placebo_treatment",
        "random_common_cause", 
        "data_subset",
        "bootstrap",
        "sensitivity_e_value"
    ]
    
    def __call__(
        self,
        gold: Example,
        pred: Prediction,
        trace: Optional[list] = None,
        pred_name: Optional[str] = None,
        pred_trace: Optional[list] = None
    ) -> dict:
        """
        Compute score and feedback for causal impact estimation.
        
        Args:
            gold: Ground truth example with expected outcomes
            pred: Model prediction with causal estimate
            trace: Full execution trace
            pred_name: Name of predictor being optimized
            pred_trace: Sub-trace for specific predictor
            
        Returns:
            ScoreWithFeedback dict with score and textual feedback
        """
        feedback_parts = []
        
        # === Component 1: Refutation Tests ===
        refutation_score, refutation_feedback = self._score_refutation(pred)
        feedback_parts.append(refutation_feedback)
        
        # === Component 2: Sensitivity Analysis ===
        sensitivity_score, sensitivity_feedback = self._score_sensitivity(pred)
        feedback_parts.append(sensitivity_feedback)
        
        # === Component 3: Methodology ===
        methodology_score, methodology_feedback = self._score_methodology(
            pred, gold, trace
        )
        feedback_parts.append(methodology_feedback)
        
        # === Component 4: Business Relevance ===
        business_score, business_feedback = self._score_business_relevance(
            pred, gold
        )
        feedback_parts.append(business_feedback)
        
        # === Aggregate Score ===
        total_score = (
            self.REFUTATION_WEIGHT * refutation_score +
            self.SENSITIVITY_WEIGHT * sensitivity_score +
            self.METHODOLOGY_WEIGHT * methodology_score +
            self.BUSINESS_WEIGHT * business_score
        )
        
        # === Construct Feedback ===
        combined_feedback = self._construct_feedback(
            total_score,
            feedback_parts,
            pred_name,
            pred_trace
        )
        
        return {"score": total_score, "feedback": combined_feedback}
    
    def _score_refutation(self, pred: Prediction) -> tuple[float, str]:
        """Score refutation test results from Node 3."""
        if not hasattr(pred, 'refutation_results'):
            return 0.0, "CRITICAL: No refutation tests executed"
        
        results = pred.refutation_results
        passed_tests = []
        failed_tests = []
        
        for test_type in self.REFUTATION_TESTS:
            test_result = results.get(test_type, {})
            status = test_result.get('status', 'skipped')
            
            if status == 'passed':
                passed_tests.append(test_type)
            elif status == 'failed':
                failed_tests.append({
                    'test': test_type,
                    'original_effect': test_result.get('original_effect'),
                    'refuted_effect': test_result.get('refuted_effect'),
                    'p_value': test_result.get('p_value')
                })
        
        score = len(passed_tests) / len(self.REFUTATION_TESTS)
        
        if failed_tests:
            feedback = f"REFUTATION FAILURES ({len(failed_tests)}/{len(self.REFUTATION_TESTS)}): "
            for f in failed_tests:
                feedback += (
                    f"[{f['test']}] effect changed from {f['original_effect']:.3f} "
                    f"to {f['refuted_effect']:.3f} (p={f['p_value']:.4f}); "
                )
        else:
            feedback = f"All {len(self.REFUTATION_TESTS)} refutation tests passed"
        
        return score, feedback
    
    def _score_sensitivity(self, pred: Prediction) -> tuple[float, str]:
        """Score sensitivity analysis robustness."""
        if not hasattr(pred, 'sensitivity_analysis'):
            return 0.0, "CRITICAL: No sensitivity analysis performed"
        
        sensitivity = pred.sensitivity_analysis
        e_value = sensitivity.get('e_value', 0)
        robustness_value = sensitivity.get('robustness_value', 0)
        
        # E-value > 2 is generally considered robust
        if e_value >= 3:
            score = 1.0
            feedback = f"Strong robustness: E-value={e_value:.2f} (threshold: 2.0)"
        elif e_value >= 2:
            score = 0.8
            feedback = f"Acceptable robustness: E-value={e_value:.2f}"
        elif e_value >= 1.5:
            score = 0.5
            feedback = f"MARGINAL robustness: E-value={e_value:.2f}. Consider larger sample or stronger instruments"
        else:
            score = 0.2
            feedback = f"WEAK robustness: E-value={e_value:.2f}. Estimate may not survive unobserved confounding"
        
        return score, feedback
    
    def _score_methodology(
        self, 
        pred: Prediction, 
        gold: Example,
        trace: Optional[list]
    ) -> tuple[float, str]:
        """Score methodology selection appropriateness."""
        score = 0.0
        feedback_parts = []
        
        # DAG validity check
        if hasattr(pred, 'dag_approved') and pred.dag_approved:
            score += 0.4
        else:
            feedback_parts.append("DAG not expert-approved")
        
        # Estimation method appropriateness
        if hasattr(pred, 'estimation_method'):
            method = pred.estimation_method
            data_characteristics = getattr(gold, 'data_characteristics', {})
            
            # Check method-data alignment
            if self._method_appropriate(method, data_characteristics):
                score += 0.6
            else:
                feedback_parts.append(
                    f"Method '{method}' may not be optimal for data characteristics: "
                    f"{data_characteristics}"
                )
                score += 0.3
        
        if not feedback_parts:
            feedback = "Methodology validated: DAG approved, estimation method appropriate"
        else:
            feedback = "METHODOLOGY CONCERNS: " + "; ".join(feedback_parts)
        
        return score, feedback
    
    def _method_appropriate(self, method: str, data_chars: dict) -> bool:
        """Check if estimation method matches data characteristics."""
        # CausalForest best for heterogeneous effects
        if data_chars.get('heterogeneous', False) and method == 'CausalForest':
            return True
        # LinearDML for high-dimensional confounders
        if data_chars.get('high_dim_confounders', False) and method == 'LinearDML':
            return True
        # Standard methods for simple cases
        if not data_chars.get('complex', False) and method in ['OLS', 'IPW']:
            return True
        return False
    
    def _score_business_relevance(
        self, 
        pred: Prediction, 
        gold: Example
    ) -> tuple[float, str]:
        """Score business relevance and actionability."""
        score = 0.0
        feedback_parts = []
        
        # KPI attribution
        if hasattr(pred, 'kpi_attribution'):
            attributed_kpis = pred.kpi_attribution
            expected_kpis = getattr(gold, 'expected_kpis', [])
            
            if set(attributed_kpis) & set(expected_kpis):
                score += 0.5
            else:
                feedback_parts.append(
                    f"KPI attribution mismatch. Expected: {expected_kpis}, "
                    f"Got: {attributed_kpis}"
                )
        
        # Actionability
        if hasattr(pred, 'recommendations') and pred.recommendations:
            score += 0.5
        else:
            feedback_parts.append("No actionable recommendations generated")
        
        if not feedback_parts:
            feedback = "Business relevance validated"
        else:
            feedback = "BUSINESS CONCERNS: " + "; ".join(feedback_parts)
        
        return score, feedback
    
    def _construct_feedback(
        self,
        total_score: float,
        feedback_parts: list[str],
        pred_name: Optional[str],
        pred_trace: Optional[list]
    ) -> str:
        """Construct final feedback string for GEPA reflection."""
        lines = [f"Overall Score: {total_score:.3f}"]
        
        # Add component feedback
        for part in feedback_parts:
            lines.append(f"• {part}")
        
        # Add predictor-specific context if available
        if pred_name and pred_trace:
            lines.append(f"\n[Predictor: {pred_name}]")
            # Extract relevant trace info
            if pred_trace:
                last_call = pred_trace[-1] if pred_trace else None
                if last_call:
                    _, inputs, outputs = last_call
                    lines.append(f"Inputs: {list(inputs.keys())}")
                    lines.append(f"Outputs: {list(outputs.keys()) if isinstance(outputs, dict) else type(outputs)}")
        
        return "\n".join(lines)
```

### 2.3 Tier 3: Experiment Designer Metric

```python
# src/optimization/gepa/metrics/experiment_designer_metric.py

class ExperimentDesignerGEPAMetric:
    """
    GEPA metric for Experiment Designer Hybrid agent.
    
    Scoring breakdown:
    - Power analysis validity (0.35): Correct power calculations
    - Design validity (0.30): Proper randomization, controls
    - Past learning integration (0.20): Uses ExperimentKnowledgeStore
    - Pre-registration quality (0.15): Complete protocol generation
    """
    
    POWER_WEIGHT = 0.35
    DESIGN_WEIGHT = 0.30
    LEARNING_WEIGHT = 0.20
    PREREG_WEIGHT = 0.15
    
    def __call__(
        self,
        gold: Example,
        pred: Prediction,
        trace: Optional[list] = None,
        pred_name: Optional[str] = None,
        pred_trace: Optional[list] = None
    ) -> dict:
        feedback_parts = []
        
        # === Component 1: Power Analysis ===
        power_score, power_feedback = self._score_power_analysis(pred, gold)
        feedback_parts.append(power_feedback)
        
        # === Component 2: Design Validity ===
        design_score, design_feedback = self._score_design_validity(pred)
        feedback_parts.append(design_feedback)
        
        # === Component 3: Past Learning Integration ===
        learning_score, learning_feedback = self._score_learning_integration(
            pred, trace
        )
        feedback_parts.append(learning_feedback)
        
        # === Component 4: Pre-registration Quality ===
        prereg_score, prereg_feedback = self._score_preregistration(pred)
        feedback_parts.append(prereg_feedback)
        
        total_score = (
            self.POWER_WEIGHT * power_score +
            self.DESIGN_WEIGHT * design_score +
            self.LEARNING_WEIGHT * learning_score +
            self.PREREG_WEIGHT * prereg_score
        )
        
        combined_feedback = "\n".join([
            f"Score: {total_score:.3f}",
            *[f"• {part}" for part in feedback_parts]
        ])
        
        return {"score": total_score, "feedback": combined_feedback}
    
    def _score_power_analysis(
        self, 
        pred: Prediction, 
        gold: Example
    ) -> tuple[float, str]:
        """Score power analysis accuracy."""
        if not hasattr(pred, 'power_calculation'):
            return 0.0, "CRITICAL: No power analysis performed"
        
        calculated_power = pred.power_calculation.get('power', 0)
        target_power = 0.80  # Standard threshold
        
        if calculated_power >= target_power:
            # Check sample size reasonableness
            sample_size = pred.power_calculation.get('required_n', 0)
            expected_effect = gold.expected_effect_size
            
            if self._sample_size_reasonable(sample_size, expected_effect):
                return 1.0, f"Power={calculated_power:.2f} with reasonable n={sample_size}"
            else:
                return 0.7, f"Power achieved but sample size ({sample_size}) may be over/under-estimated for effect={expected_effect}"
        else:
            return 0.3, f"UNDERPOWERED: Power={calculated_power:.2f} < {target_power}. Increase sample size or adjust MDE"
    
    def _sample_size_reasonable(self, n: int, effect_size: float) -> bool:
        """Check if sample size is reasonable for expected effect."""
        # Rule of thumb: smaller effects need larger samples
        if effect_size < 0.1:
            return n >= 1000
        elif effect_size < 0.3:
            return n >= 200
        else:
            return n >= 50
    
    def _score_design_validity(self, pred: Prediction) -> tuple[float, str]:
        """Score experimental design validity."""
        score = 0.0
        issues = []
        
        if hasattr(pred, 'randomization_method'):
            if pred.randomization_method in ['stratified', 'blocked', 'cluster']:
                score += 0.4
            elif pred.randomization_method == 'simple':
                score += 0.3
                issues.append("Consider stratified randomization for balance")
        else:
            issues.append("No randomization method specified")
        
        if hasattr(pred, 'control_group') and pred.control_group:
            score += 0.3
        else:
            issues.append("Control group not defined")
        
        if hasattr(pred, 'blinding') and pred.blinding:
            score += 0.3
        else:
            issues.append("No blinding specified")
        
        if issues:
            return score, f"DESIGN ISSUES: {'; '.join(issues)}"
        return score, "Design validity confirmed"
    
    def _score_learning_integration(
        self, 
        pred: Prediction,
        trace: Optional[list]
    ) -> tuple[float, str]:
        """Score integration with ExperimentKnowledgeStore."""
        if not hasattr(pred, 'past_learnings_applied'):
            return 0.0, "ExperimentKnowledgeStore not consulted"
        
        learnings = pred.past_learnings_applied
        if not learnings:
            return 0.3, "ExperimentKnowledgeStore consulted but no applicable learnings found"
        
        # Check if learnings were actually applied
        applied_count = sum(1 for l in learnings if l.get('applied', False))
        
        if applied_count == len(learnings):
            return 1.0, f"All {applied_count} past learnings applied to design"
        elif applied_count > 0:
            return 0.7, f"{applied_count}/{len(learnings)} past learnings applied"
        else:
            return 0.4, "Past learnings retrieved but not applied to design"
    
    def _score_preregistration(self, pred: Prediction) -> tuple[float, str]:
        """Score pre-registration protocol completeness."""
        if not hasattr(pred, 'preregistration'):
            return 0.0, "No pre-registration protocol generated"
        
        prereg = pred.preregistration
        required_fields = [
            'hypothesis', 'primary_outcome', 'sample_size_justification',
            'analysis_plan', 'stopping_rules'
        ]
        
        present = [f for f in required_fields if prereg.get(f)]
        missing = [f for f in required_fields if not prereg.get(f)]
        
        score = len(present) / len(required_fields)
        
        if missing:
            return score, f"Pre-registration missing: {', '.join(missing)}"
        return score, "Pre-registration complete"
```

### 2.4 Tier 5: Feedback Learner Metric

```python
# src/optimization/gepa/metrics/feedback_learner_metric.py

class FeedbackLearnerGEPAMetric:
    """
    GEPA metric for Feedback Learner Deep agent.
    
    This is meta-optimization: optimizing the optimizer.
    
    Scoring breakdown:
    - Learning extraction quality (0.40): Useful patterns identified
    - Storage efficiency (0.20): Compact, retrievable storage
    - Application success (0.40): Learnings improve downstream agents
    """
    
    EXTRACTION_WEIGHT = 0.40
    STORAGE_WEIGHT = 0.20
    APPLICATION_WEIGHT = 0.40
    
    def __call__(
        self,
        gold: Example,
        pred: Prediction,
        trace: Optional[list] = None,
        pred_name: Optional[str] = None,
        pred_trace: Optional[list] = None
    ) -> dict:
        feedback_parts = []
        
        # === Component 1: Learning Extraction ===
        extraction_score, extraction_feedback = self._score_extraction(pred, gold)
        feedback_parts.append(extraction_feedback)
        
        # === Component 2: Storage Efficiency ===
        storage_score, storage_feedback = self._score_storage(pred)
        feedback_parts.append(storage_feedback)
        
        # === Component 3: Application Success ===
        application_score, application_feedback = self._score_application(pred, gold)
        feedback_parts.append(application_feedback)
        
        total_score = (
            self.EXTRACTION_WEIGHT * extraction_score +
            self.STORAGE_WEIGHT * storage_score +
            self.APPLICATION_WEIGHT * application_score
        )
        
        combined_feedback = "\n".join([
            f"Score: {total_score:.3f}",
            *[f"• {part}" for part in feedback_parts]
        ])
        
        return {"score": total_score, "feedback": combined_feedback}
    
    def _score_extraction(
        self, 
        pred: Prediction, 
        gold: Example
    ) -> tuple[float, str]:
        """Score quality of extracted learnings."""
        if not hasattr(pred, 'extracted_learnings'):
            return 0.0, "CRITICAL: No learnings extracted"
        
        learnings = pred.extracted_learnings
        expected_patterns = getattr(gold, 'expected_patterns', [])
        
        # Check pattern recall
        matched = 0
        for expected in expected_patterns:
            for learning in learnings:
                if self._pattern_match(expected, learning):
                    matched += 1
                    break
        
        if expected_patterns:
            recall = matched / len(expected_patterns)
        else:
            recall = 1.0 if learnings else 0.0
        
        # Check for false positives (spurious patterns)
        spurious = len(learnings) - matched if learnings else 0
        precision_penalty = min(0.3, spurious * 0.1)
        
        score = max(0, recall - precision_penalty)
        
        if score >= 0.8:
            return score, f"High-quality extraction: {matched}/{len(expected_patterns)} patterns, {spurious} spurious"
        else:
            return score, f"Extraction needs improvement: {matched}/{len(expected_patterns)} patterns found, {spurious} spurious patterns"
    
    def _pattern_match(self, expected: dict, learning: dict) -> bool:
        """Check if extracted learning matches expected pattern."""
        # Semantic similarity check would go here
        # For now, simple keyword overlap
        expected_keywords = set(str(expected).lower().split())
        learning_keywords = set(str(learning).lower().split())
        overlap = len(expected_keywords & learning_keywords)
        return overlap / len(expected_keywords) > 0.5
    
    def _score_storage(self, pred: Prediction) -> tuple[float, str]:
        """Score storage efficiency."""
        if not hasattr(pred, 'storage_format'):
            return 0.5, "Default storage format used"
        
        storage = pred.storage_format
        
        # Check compactness
        if storage.get('compressed', False):
            score = 0.8
        else:
            score = 0.6
        
        # Check retrievability
        if storage.get('indexed', False):
            score += 0.2
        
        return min(1.0, score), f"Storage: compressed={storage.get('compressed')}, indexed={storage.get('indexed')}"
    
    def _score_application(
        self, 
        pred: Prediction, 
        gold: Example
    ) -> tuple[float, str]:
        """Score downstream application success."""
        if not hasattr(pred, 'application_results'):
            return 0.0, "No downstream application measured"
        
        results = pred.application_results
        improvement = results.get('performance_delta', 0)
        
        if improvement > 0.05:
            return 1.0, f"Strong downstream improvement: +{improvement:.1%}"
        elif improvement > 0:
            return 0.7, f"Modest downstream improvement: +{improvement:.1%}"
        elif improvement == 0:
            return 0.4, "No measurable downstream impact"
        else:
            return 0.1, f"NEGATIVE downstream impact: {improvement:.1%}. Review extracted learnings"
```

### 2.5 Standard Agent Metric (Tiers 0, 1, 4)

```python
# src/optimization/gepa/metrics/standard_agent_metric.py

class StandardAgentGEPAMetric:
    """
    Generic GEPA metric for Standard agents.
    
    Uses SLA compliance and basic accuracy metrics.
    Lighter optimization for tool-heavy, SLA-bound operations.
    """
    
    def __init__(self, sla_threshold_ms: int = 2000):
        self.sla_threshold = sla_threshold_ms
    
    def __call__(
        self,
        gold: Example,
        pred: Prediction,
        trace: Optional[list] = None,
        pred_name: Optional[str] = None,
        pred_trace: Optional[list] = None
    ) -> dict:
        feedback_parts = []
        
        # SLA compliance (0.4)
        sla_score = self._score_sla(pred)
        
        # Accuracy (0.6)
        accuracy_score, accuracy_feedback = self._score_accuracy(pred, gold)
        feedback_parts.append(accuracy_feedback)
        
        total_score = 0.4 * sla_score + 0.6 * accuracy_score
        
        sla_feedback = f"SLA: {'PASS' if sla_score == 1.0 else 'FAIL'} ({getattr(pred, 'latency_ms', 'N/A')}ms)"
        feedback_parts.insert(0, sla_feedback)
        
        return {
            "score": total_score,
            "feedback": f"Score: {total_score:.3f} | " + " | ".join(feedback_parts)
        }
    
    def _score_sla(self, pred: Prediction) -> float:
        """Score SLA compliance."""
        latency = getattr(pred, 'latency_ms', self.sla_threshold)
        return 1.0 if latency <= self.sla_threshold else 0.5
    
    def _score_accuracy(
        self, 
        pred: Prediction, 
        gold: Example
    ) -> tuple[float, str]:
        """Score prediction accuracy."""
        if hasattr(gold, 'expected_output') and hasattr(pred, 'output'):
            # Exact match or similarity
            if pred.output == gold.expected_output:
                return 1.0, "Exact match"
            # Partial credit for similar outputs
            similarity = self._compute_similarity(pred.output, gold.expected_output)
            return similarity, f"Similarity: {similarity:.2f}"
        return 0.5, "No ground truth for comparison"
    
    def _compute_similarity(self, pred_output, gold_output) -> float:
        """Compute output similarity."""
        # Implement based on output type
        if isinstance(pred_output, (int, float)) and isinstance(gold_output, (int, float)):
            # Relative error
            if gold_output == 0:
                return 1.0 if pred_output == 0 else 0.0
            rel_error = abs(pred_output - gold_output) / abs(gold_output)
            return max(0, 1 - rel_error)
        return 0.5  # Default
```

---

## Part 3: Tool Optimization Configuration

### 3.1 DoWhy/EconML Tool Descriptions

```python
# src/optimization/gepa/tools/causal_tools.py

from dspy import Tool

# Define tools with optimizable descriptions
CAUSAL_TOOLS = [
    Tool(
        name="causal_forest",
        description="""
        Use CausalForest for heterogeneous treatment effect estimation when:
        - You need to estimate CATE (Conditional Average Treatment Effect)
        - The treatment effect varies across subgroups
        - You have sufficient sample size (n > 500)
        - Confounders are moderate dimensional
        
        NOT suitable when:
        - Treatment effect is homogeneous
        - Sample size is small (n < 200)
        - You need interpretable coefficients
        """,
        func=lambda **kwargs: run_causal_forest(**kwargs)
    ),
    Tool(
        name="linear_dml",
        description="""
        Use LinearDML (Double Machine Learning) when:
        - You have high-dimensional confounders
        - You want interpretable coefficients
        - Treatment is continuous or binary
        - You suspect linear treatment effect
        
        NOT suitable when:
        - Treatment effect is highly non-linear
        - Very small sample size
        """,
        func=lambda **kwargs: run_linear_dml(**kwargs)
    ),
    Tool(
        name="ols_regression",
        description="""
        Use OLS regression when:
        - Simple confounding structure
        - Small set of known confounders
        - Need baseline comparison
        - Quick estimation needed
        
        NOT suitable when:
        - Unobserved confounding suspected
        - High-dimensional data
        - Non-linear relationships
        """,
        func=lambda **kwargs: run_ols(**kwargs)
    ),
    Tool(
        name="ipw_estimator",
        description="""
        Use Inverse Probability Weighting when:
        - Binary treatment
        - Propensity scores can be estimated
        - Want to adjust for selection bias
        
        NOT suitable when:
        - Extreme propensity scores (near 0 or 1)
        - Treatment is continuous
        """,
        func=lambda **kwargs: run_ipw(**kwargs)
    ),
    Tool(
        name="refutation_suite",
        description="""
        Run the refutation test suite to validate causal estimates.
        Always run this after estimation to check robustness.
        
        Returns: ValidationSuite with pass/fail for each test
        - placebo_treatment: Effect should disappear with fake treatment
        - random_common_cause: Effect stable with random confounders
        - data_subset: Effect consistent across data subsets
        - bootstrap: Effect stable across resamples
        - sensitivity_e_value: Robustness to unobserved confounding
        """,
        func=lambda **kwargs: run_refutation_suite(**kwargs)
    )
]
```

### 3.2 GEPA Tool Optimization Setup

```python
# src/optimization/gepa/optimizer_setup.py

import dspy
from dspy import GEPA

def create_causal_impact_optimizer(
    trainset: list,
    valset: list,
    reflection_lm: dspy.LM
) -> GEPA:
    """
    Create GEPA optimizer for Causal Impact agent with tool optimization.
    """
    from .metrics.causal_impact_metric import CausalImpactGEPAMetric
    
    metric = CausalImpactGEPAMetric()
    
    gepa = GEPA(
        metric=metric,
        
        # Budget: medium for thorough optimization
        auto="medium",
        
        # Reflection configuration
        reflection_lm=reflection_lm,
        reflection_minibatch_size=3,  # Small batches for domain-specific examples
        
        # Enable tool optimization for DoWhy/EconML tools
        enable_tool_optimization=True,
        
        # Pareto for multi-objective (refutation + sensitivity + business)
        candidate_selection_strategy="pareto",
        
        # Allow longer exploration
        skip_perfect_score=True,
        
        # Merge successful variants
        use_merge=True,
        max_merge_invocations=5,
        
        # Logging
        log_dir="./gepa_logs/causal_impact",
        track_stats=True,
        use_mlflow=True,  # Integrate with existing MLflow
        
        seed=42
    )
    
    return gepa
```

---

## Part 4: Migration Phases

### Phase 1: Pilot (Week 1-2)

**Scope:** Single agent - Causal Impact Hybrid

```python
# scripts/gepa_pilot.py

import dspy
from src.agents.tier2.causal_impact import CausalImpactAgent
from src.optimization.gepa.optimizer_setup import create_causal_impact_optimizer
from src.data.training_data import load_causal_trainset, load_causal_valset

def run_pilot():
    """Run GEPA pilot on Causal Impact agent."""
    
    # 1. Load existing training data
    trainset = load_causal_trainset(limit=100)  # Start small
    valset = load_causal_valset(limit=50)
    
    # 2. Configure reflection LM
    reflection_lm = dspy.LM(
        model="anthropic/claude-sonnet-4-20250514",
        temperature=1.0,
        max_tokens=16000
    )
    
    # 3. Create optimizer
    gepa = create_causal_impact_optimizer(
        trainset=trainset,
        valset=valset,
        reflection_lm=reflection_lm
    )
    
    # 4. Get student module
    student = CausalImpactAgent().as_dspy_module()
    
    # 5. Compile
    print("Starting GEPA optimization...")
    optimized = gepa.compile(
        student=student,
        trainset=trainset,
        valset=valset
    )
    
    # 6. Evaluate
    from dspy import Evaluate
    evaluator = Evaluate(
        devset=valset,
        metric=gepa.metric_fn,
        num_threads=4
    )
    
    baseline_score = evaluator(student)
    optimized_score = evaluator(optimized)
    
    print(f"\nPilot Results:")
    print(f"  Baseline:  {baseline_score:.3f}")
    print(f"  Optimized: {optimized_score:.3f}")
    print(f"  Δ:         {optimized_score - baseline_score:+.3f} ({(optimized_score/baseline_score - 1)*100:+.1f}%)")
    
    # 7. Save optimized module
    optimized.save("./models/gepa/causal_impact_pilot_v1.json")
    
    return optimized, gepa

if __name__ == "__main__":
    run_pilot()
```

**Success Criteria:**
- [ ] GEPA runs without errors
- [ ] Optimization completes within budget
- [ ] ≥5% improvement over baseline
- [ ] Tool selection patterns improve

### Phase 2: Hybrid Agents (Week 3-4)

**Scope:** All 4 Hybrid agents

| Agent | Tier | Priority | Expected Improvement |
|-------|------|----------|---------------------|
| causal_impact | 2 | P0 | 10-15% |
| experiment_designer | 3 | P0 | 8-12% |
| feature_analyzer | 0 | P1 | 5-8% |
| (other hybrid if any) | - | P1 | 5-8% |

```python
# scripts/gepa_phase2_hybrid.py

HYBRID_AGENTS = {
    "causal_impact": {
        "tier": 2,
        "metric": CausalImpactGEPAMetric,
        "auto": "medium",
        "enable_tool_optimization": True
    },
    "experiment_designer": {
        "tier": 3,
        "metric": ExperimentDesignerGEPAMetric,
        "auto": "medium",
        "enable_tool_optimization": True
    },
    "feature_analyzer": {
        "tier": 0,
        "metric": FeatureAnalyzerGEPAMetric,
        "auto": "light",
        "enable_tool_optimization": False
    }
}

def optimize_hybrid_agents():
    """Phase 2: Optimize all Hybrid agents."""
    results = {}
    
    for agent_name, config in HYBRID_AGENTS.items():
        print(f"\n{'='*60}")
        print(f"Optimizing: {agent_name} (Tier {config['tier']})")
        print(f"{'='*60}")
        
        result = optimize_single_agent(
            agent_name=agent_name,
            metric_class=config['metric'],
            auto=config['auto'],
            enable_tool_optimization=config['enable_tool_optimization']
        )
        
        results[agent_name] = result
    
    # Summary
    print("\n" + "="*60)
    print("Phase 2 Summary: Hybrid Agents")
    print("="*60)
    for name, result in results.items():
        delta = result['optimized_score'] - result['baseline_score']
        print(f"  {name}: {result['baseline_score']:.3f} → {result['optimized_score']:.3f} (Δ={delta:+.3f})")
    
    return results
```

### Phase 3: Deep Agents (Week 5)

**Scope:** 2 Deep agents (Tier 5)

```python
# Deep agents need extended reflection
DEEP_AGENTS = {
    "explainer": {
        "tier": 5,
        "metric": ExplainerGEPAMetric,
        "auto": "heavy",  # More budget for complex reasoning
        "reflection_minibatch_size": 2  # Smaller batches, deeper analysis
    },
    "feedback_learner": {
        "tier": 5,
        "metric": FeedbackLearnerGEPAMetric,
        "auto": "heavy",
        "reflection_minibatch_size": 2
    }
}
```

### Phase 4: Standard Agents (Week 6)

**Scope:** 12 Standard agents

```python
# Standard agents get light optimization
STANDARD_AGENTS = [
    # Tier 0 (7)
    "scope_definer", "data_preparer", "model_selector", "model_trainer",
    "model_deployer", "observability_connector",  # feature_analyzer is Hybrid
    
    # Tier 1 (1)
    "orchestrator",
    
    # Tier 2 (2)
    "gap_analyzer", "heterogeneous_optimizer",
    
    # Tier 3 (2)
    "drift_monitor", "health_score",
    
    # Tier 4 (2)
    "prediction_synthesizer", "resource_optimizer"
]

def optimize_standard_agents():
    """Phase 4: Light optimization for Standard agents."""
    metric = StandardAgentGEPAMetric()
    
    for agent_name in STANDARD_AGENTS:
        optimize_single_agent(
            agent_name=agent_name,
            metric_class=lambda: metric,
            auto="light",
            enable_tool_optimization=False
        )
```

### Phase 5: Integration & Validation (Week 7-8)

```python
# scripts/gepa_integration_test.py

def run_integration_test():
    """Test full 18-agent system with GEPA-optimized modules."""
    
    # 1. Load all optimized modules
    optimized_agents = load_all_optimized_agents()
    
    # 2. Run end-to-end test scenarios
    test_scenarios = [
        "analyze_remibrutinib_midwest_performance",
        "design_ab_test_kisqali_targeting",
        "investigate_fabhalta_conversion_drop"
    ]
    
    results = []
    for scenario in test_scenarios:
        result = run_scenario(scenario, optimized_agents)
        results.append(result)
    
    # 3. Compare to baseline
    baseline_agents = load_baseline_agents()
    baseline_results = []
    for scenario in test_scenarios:
        result = run_scenario(scenario, baseline_agents)
        baseline_results.append(result)
    
    # 4. Report
    print("\nIntegration Test Results")
    print("="*60)
    for i, scenario in enumerate(test_scenarios):
        baseline = baseline_results[i]
        optimized = results[i]
        print(f"\n{scenario}:")
        print(f"  Baseline:  score={baseline['score']:.3f}, latency={baseline['latency_ms']}ms")
        print(f"  Optimized: score={optimized['score']:.3f}, latency={optimized['latency_ms']}ms")
```

---

## Part 5: MLOps Integration

### 5.1 MLflow + GEPA Logging

```python
# src/optimization/gepa/mlflow_integration.py

import mlflow
from mlflow.tracking import MlflowClient

def setup_gepa_mlflow():
    """Configure MLflow for GEPA experiment tracking."""
    
    mlflow.set_tracking_uri("http://localhost:5000")  # Or your MLflow server
    mlflow.set_experiment("e2i-gepa-optimization")
    
    # Set tags for GEPA runs
    mlflow.set_tags({
        "optimizer": "GEPA",
        "framework": "DSPy",
        "version": "1.0"
    })

def log_gepa_run(agent_name: str, gepa_result, baseline_score: float):
    """Log GEPA optimization results to MLflow."""
    
    with mlflow.start_run(run_name=f"gepa_{agent_name}"):
        # Log parameters
        mlflow.log_params({
            "agent": agent_name,
            "auto_budget": gepa_result.auto,
            "enable_tool_optimization": gepa_result.enable_tool_optimization,
            "reflection_minibatch_size": gepa_result.reflection_minibatch_size
        })
        
        # Log metrics
        detailed = gepa_result.detailed_results
        mlflow.log_metrics({
            "baseline_score": baseline_score,
            "optimized_score": detailed.val_aggregate_scores[detailed.best_idx],
            "improvement": detailed.val_aggregate_scores[detailed.best_idx] - baseline_score,
            "total_metric_calls": detailed.total_metric_calls,
            "num_candidates_explored": len(detailed.candidates)
        })
        
        # Log artifacts
        mlflow.log_artifact(f"./gepa_logs/{agent_name}/")
```

### 5.2 Opik + GEPA Tracing

```python
# src/optimization/gepa/opik_integration.py

from opik import track

@track(name="gepa_optimization")
def tracked_gepa_compile(gepa, student, trainset, valset):
    """GEPA compile with Opik tracing."""
    return gepa.compile(
        student=student,
        trainset=trainset,
        valset=valset
    )

@track(name="gepa_evaluation")
def tracked_evaluation(evaluator, module):
    """Evaluation with Opik tracing."""
    return evaluator(module)
```

### 5.3 RAGAS Integration for RAG Agents

```python
# src/optimization/gepa/ragas_feedback.py

from ragas.metrics import faithfulness, answer_relevancy, context_precision

class RAGAgentGEPAMetric:
    """GEPA metric for RAG-based agents using RAGAS metrics."""
    
    def __init__(self):
        self.ragas_metrics = [
            faithfulness,
            answer_relevancy,
            context_precision
        ]
    
    def __call__(
        self,
        gold: Example,
        pred: Prediction,
        trace: Optional[list] = None,
        pred_name: Optional[str] = None,
        pred_trace: Optional[list] = None
    ) -> dict:
        # Compute RAGAS scores
        ragas_scores = self._compute_ragas(pred, gold)
        
        # Build feedback
        feedback_parts = []
        for metric_name, score in ragas_scores.items():
            if score < 0.7:
                feedback_parts.append(f"{metric_name}={score:.2f} (BELOW THRESHOLD)")
            else:
                feedback_parts.append(f"{metric_name}={score:.2f}")
        
        avg_score = sum(ragas_scores.values()) / len(ragas_scores)
        
        return {
            "score": avg_score,
            "feedback": " | ".join(feedback_parts)
        }
```

---

## Part 6: Cost & Budget Analysis

### 6.1 Estimated Token Costs

| Phase | Agents | Examples | Est. Metric Calls | Est. Tokens | Est. Cost |
|-------|--------|----------|-------------------|-------------|-----------|
| Pilot | 1 | 150 | ~2,000 | ~10M | ~$30 |
| Phase 2 | 4 | 600 | ~8,000 | ~40M | ~$120 |
| Phase 3 | 2 | 200 | ~4,000 | ~20M | ~$60 |
| Phase 4 | 12 | 1,200 | ~6,000 | ~30M | ~$90 |
| Phase 5 | 18 | 500 | ~3,000 | ~15M | ~$45 |
| **Total** | | | **~23,000** | **~115M** | **~$345** |

*Assumes Claude Sonnet at ~$3/M tokens for reflection LM*

### 6.2 Budget Configurations

```python
# Recommended budget settings per agent type

BUDGET_CONFIGS = {
    "hybrid": {
        "auto": "medium",
        "estimated_metric_calls": 2000,
        "estimated_cost": "$30-50"
    },
    "deep": {
        "auto": "heavy",
        "estimated_metric_calls": 2000,
        "estimated_cost": "$30-50"
    },
    "standard": {
        "auto": "light",
        "estimated_metric_calls": 500,
        "estimated_cost": "$5-10"
    }
}
```

---

## Part 7: Rollback Plan

### 7.1 Versioned Module Storage

```python
# src/optimization/gepa/versioning.py

from datetime import datetime
import json

def save_optimized_module(
    agent_name: str,
    module,
    gepa_result,
    baseline_score: float
):
    """Save optimized module with metadata for rollback."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = f"gepa_v1_{timestamp}"
    
    metadata = {
        "agent": agent_name,
        "version": version,
        "optimizer": "GEPA",
        "baseline_score": baseline_score,
        "optimized_score": gepa_result.detailed_results.val_aggregate_scores[
            gepa_result.detailed_results.best_idx
        ],
        "total_metric_calls": gepa_result.detailed_results.total_metric_calls,
        "timestamp": timestamp
    }
    
    # Save module
    module_path = f"./models/gepa/{agent_name}/{version}.json"
    module.save(module_path)
    
    # Save metadata
    meta_path = f"./models/gepa/{agent_name}/{version}_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Update latest symlink
    latest_path = f"./models/gepa/{agent_name}/latest.json"
    # ... symlink logic
    
    return version

def rollback_agent(agent_name: str, target_version: str = "mipro_baseline"):
    """Rollback agent to previous version."""
    
    if target_version == "mipro_baseline":
        module_path = f"./models/mipro/{agent_name}/latest.json"
    else:
        module_path = f"./models/gepa/{agent_name}/{target_version}.json"
    
    # Update production symlink
    prod_path = f"./models/production/{agent_name}.json"
    # ... symlink to module_path
    
    print(f"Rolled back {agent_name} to {target_version}")
```

### 7.2 A/B Testing Infrastructure

```python
# src/optimization/gepa/ab_test.py

class GEPAABTest:
    """A/B test GEPA vs baseline in production."""
    
    def __init__(self, agent_name: str, traffic_split: float = 0.1):
        self.agent_name = agent_name
        self.traffic_split = traffic_split
        self.baseline = load_baseline_agent(agent_name)
        self.gepa = load_gepa_agent(agent_name)
    
    def route(self, request_id: str):
        """Route request to baseline or GEPA."""
        # Deterministic routing based on request_id hash
        if hash(request_id) % 100 < self.traffic_split * 100:
            return "gepa", self.gepa
        return "baseline", self.baseline
    
    def record_outcome(self, request_id: str, variant: str, outcome: dict):
        """Record outcome for analysis."""
        # Log to analytics
        pass
```

---

## Part 8: Success Metrics

### 8.1 Key Performance Indicators

| Metric | Baseline (MIPROv2) | Target (GEPA) | Measurement |
|--------|-------------------|---------------|-------------|
| Causal Impact accuracy | 0.78 | 0.85+ | Refutation pass rate |
| Experiment Designer power | 0.75 | 0.85+ | Achieved vs. designed power |
| Feedback Learner improvement | +5%/quarter | +8%/quarter | Downstream agent gains |
| Average agent latency | 1.8s | <2.0s | P95 latency |
| Training data efficiency | 500 examples | 200 examples | For equivalent performance |

### 8.2 Monitoring Dashboard

```sql
-- GEPA optimization tracking query
SELECT 
    agent_name,
    optimizer_version,
    AVG(score) as avg_score,
    AVG(latency_ms) as avg_latency,
    COUNT(*) as request_count,
    SUM(CASE WHEN score > 0.8 THEN 1 ELSE 0 END)::float / COUNT(*) as success_rate
FROM agent_executions
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY agent_name, optimizer_version
ORDER BY agent_name, optimizer_version DESC;
```

---

## Appendix A: File Structure

```
src/optimization/gepa/
├── __init__.py
├── optimizer_setup.py           # GEPA configuration
├── versioning.py                # Module version management
├── ab_test.py                   # A/B testing infrastructure
│
├── metrics/
│   ├── __init__.py
│   ├── base.py                  # E2IGEPAMetric base class
│   ├── causal_impact_metric.py  # Tier 2 Hybrid
│   ├── experiment_designer_metric.py  # Tier 3 Hybrid
│   ├── feedback_learner_metric.py     # Tier 5 Deep
│   ├── explainer_metric.py            # Tier 5 Deep
│   ├── standard_agent_metric.py       # Generic Standard
│   └── rag_agent_metric.py            # RAG-based agents
│
├── tools/
│   ├── __init__.py
│   └── causal_tools.py          # DoWhy/EconML tool definitions
│
├── integration/
│   ├── __init__.py
│   ├── mlflow_integration.py    # MLflow logging
│   ├── opik_integration.py      # Opik tracing
│   └── ragas_feedback.py        # RAGAS metrics

scripts/
├── gepa_pilot.py                # Phase 1 pilot
├── gepa_phase2_hybrid.py        # Phase 2 hybrid agents
├── gepa_phase3_deep.py          # Phase 3 deep agents
├── gepa_phase4_standard.py      # Phase 4 standard agents
├── gepa_integration_test.py     # Phase 5 integration
└── gepa_rollback.py             # Rollback utilities

config/
└── gepa_config.yaml             # GEPA configuration
```

---

## Appendix B: Quick Start

```bash
# 1. Install GEPA
pip install gepa dspy-ai

# 2. Run pilot
python scripts/gepa_pilot.py

# 3. Review results
mlflow ui  # http://localhost:5000

# 4. If successful, proceed to Phase 2
python scripts/gepa_phase2_hybrid.py
```

---

## Appendix C: Configuration Template

```yaml
# config/gepa_config.yaml

gepa:
  reflection_lm:
    model: "anthropic/claude-sonnet-4-20250514"
    temperature: 1.0
    max_tokens: 16000
  
  default_settings:
    seed: 42
    track_stats: true
    use_mlflow: true
    use_merge: true
    max_merge_invocations: 5
  
  agent_configs:
    # Tier 2 Hybrid
    causal_impact:
      auto: "medium"
      enable_tool_optimization: true
      reflection_minibatch_size: 3
    
    # Tier 3 Hybrid
    experiment_designer:
      auto: "medium"
      enable_tool_optimization: true
      reflection_minibatch_size: 3
    
    # Tier 5 Deep
    explainer:
      auto: "heavy"
      enable_tool_optimization: false
      reflection_minibatch_size: 2
    
    feedback_learner:
      auto: "heavy"
      enable_tool_optimization: false
      reflection_minibatch_size: 2
    
    # Standard agents (default)
    _default:
      auto: "light"
      enable_tool_optimization: false
      reflection_minibatch_size: 5
```

---

*Document Version: 1.0*  
*Last Updated: December 2025*  
*Author: E2I Causal Analytics Team*
