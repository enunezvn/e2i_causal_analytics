"""Template Generator Node.

This node generates DoWhy-compatible outputs and pre-registration documents
for the finalized experiment design.

Algorithm: .claude/specialists/Agent_Specialists_Tiers 1-5/experiment-designer.md lines 708-862
Contract: .claude/contracts/tier3-contracts.md lines 82-142
"""

import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, cast

from src.agents.experiment_designer.state import (
    DoWhySpec,
    ErrorDetails,
    ExperimentDesignState,
    ExperimentTemplate,
)


class TemplateGeneratorNode:
    """Generates DoWhy-compatible outputs and pre-registration documents.

    This node:
    1. Builds DoWhy causal DAG specification
    2. Generates Python analysis code template
    3. Creates pre-registration document
    4. Produces monitoring dashboard specification

    Pure computation - structured code generation.
    Performance Target: <500ms for template generation
    """

    def __init__(self):
        """Initialize template generator node."""
        self._template_version = "1.0.0"

    async def execute(self, state: ExperimentDesignState) -> ExperimentDesignState:
        """Execute template generation.

        Args:
            state: Current agent state with finalized design

        Returns:
            Updated state with generated templates
        """
        start_time = time.time()

        # Skip if status is failed
        if state.get("status") == "failed":
            return state

        try:
            # Update status
            state["status"] = "generating"

            # Build DoWhy specification
            dowhy_spec = self._build_dowhy_spec(state)
            state["dowhy_spec"] = dowhy_spec
            state["causal_graph_dot"] = dowhy_spec["graph_dot"]

            # Generate analysis code
            analysis_code = self._generate_analysis_code(state, dowhy_spec)
            state["analysis_code"] = analysis_code

            # Build experiment template
            experiment_template = self._build_experiment_template(state)
            state["experiment_template"] = experiment_template

            # Generate monitoring dashboard spec
            monitoring_spec = self._generate_monitoring_spec(state)
            state["monitoring_dashboard_spec"] = monitoring_spec

            # Calculate total latency
            latency_ms = int((time.time() - start_time) * 1000)
            node_latencies = state.get("node_latencies_ms", {})
            node_latencies["template_generator"] = latency_ms
            state["node_latencies_ms"] = node_latencies

            # Calculate total latency across all nodes
            total_latency = sum(node_latencies.values())
            state["node_latencies_ms"]["total"] = total_latency

            # Contract-required output fields
            state["total_latency_ms"] = total_latency
            state["timestamp"] = datetime.now(timezone.utc).isoformat()

            # Ensure errors and warnings are always set (required fields, v4.3 fix)
            if "errors" not in state:
                state["errors"] = []
            if "warnings" not in state:
                state["warnings"] = []

            # Update status to completed
            state["status"] = "completed"

        except Exception as e:
            error: ErrorDetails = {
                "node": "template_generator",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            state["errors"] = state.get("errors", []) + [error]
            state["status"] = "failed"
            # Ensure contract-required fields are set even on failure
            latency_ms = int((time.time() - start_time) * 1000)
            node_latencies = state.get("node_latencies_ms", {})
            node_latencies["template_generator"] = latency_ms
            state["total_latency_ms"] = sum(node_latencies.values())
            state["timestamp"] = datetime.now(timezone.utc).isoformat()

        return state

    def _build_dowhy_spec(self, state: ExperimentDesignState) -> DoWhySpec:
        """Build DoWhy-compatible DAG specification.

        Args:
            state: Current agent state

        Returns:
            DoWhySpec TypedDict
        """
        # Extract treatment variable
        treatments = state.get("treatments", [])
        treatment_var = treatments[0].get("name", "treatment") if treatments else "treatment"

        # Extract outcome variable
        outcomes = state.get("outcomes", [])
        primary_outcome = None
        for outcome in outcomes:
            if outcome.get("is_primary", False):
                primary_outcome = outcome.get("name", "outcome")
                break
        if not primary_outcome:
            primary_outcome = outcomes[0].get("name", "outcome") if outcomes else "outcome"

        # Get confounders from design
        stratification = state.get("stratification_variables", [])
        blocking = state.get("blocking_variables", [])
        common_causes = list(set(stratification + blocking))

        # Build effect modifiers
        effect_modifiers = state.get("stratification_variables", [])

        # Build DOT graph
        edges = [(treatment_var, primary_outcome)]
        for cause in common_causes:
            edges.extend([(cause, treatment_var), (cause, primary_outcome)])

        dot_lines = ["digraph {"]
        dot_lines.append(f'    "{treatment_var}" [label="{treatment_var}"];')
        dot_lines.append(f'    "{primary_outcome}" [label="{primary_outcome}"];')
        for cause in common_causes:
            dot_lines.append(f'    "{cause}" [label="{cause}"];')
        for src, tgt in edges:
            dot_lines.append(f'    "{src}" -> "{tgt}";')
        dot_lines.append("}")
        dot_string = "\n".join(dot_lines)

        # Determine identification strategy
        design_type = state.get("design_type", "RCT")
        if design_type in ["RCT", "cluster_rct"]:
            identification_strategy = "backdoor"
        elif design_type == "instrumental_variable":
            identification_strategy = "iv"
        elif design_type == "regression_discontinuity":
            identification_strategy = "rdd"
        else:
            identification_strategy = "backdoor"

        spec: DoWhySpec = {
            "treatment_variable": treatment_var,
            "outcome_variable": primary_outcome,
            "common_causes": common_causes,
            "graph_dot": dot_string,
            "identification_strategy": identification_strategy,
        }
        # Add optional fields only if they have values
        if effect_modifiers:
            spec["effect_modifiers"] = effect_modifiers
        return spec

    def _generate_analysis_code(self, state: ExperimentDesignState, dag_spec: DoWhySpec) -> str:
        """Generate Python analysis template using DoWhy.

        Args:
            state: Current agent state
            dag_spec: DoWhy DAG specification

        Returns:
            Python code template string
        """
        power_analysis = state.get("power_analysis", {})
        design_type = state.get("design_type", "RCT")
        treatments = state.get("treatments", [])
        treatment_desc = treatments[0].get("description", "") if treatments else ""

        # Choose estimator based on design
        if design_type in ["RCT", "cluster_rct"]:
            estimator = "backdoor.linear_regression"
        elif design_type == "difference_in_differences":
            estimator = "backdoor.difference_in_differences"
        else:
            estimator = "backdoor.linear_regression"

        return f'''"""
E2I Experiment Analysis Template
================================
Design: {design_type}
Treatment: {dag_spec["treatment_variable"]} - {treatment_desc}
Outcome: {dag_spec["outcome_variable"]}
Sample Size: {power_analysis.get("required_sample_size", "Not calculated")}
Generated: {datetime.now().isoformat()}

This template provides a starting point for causal analysis.
Modify as needed for your specific experiment.
"""

import pandas as pd
import numpy as np
from dowhy import CausalModel
from econml.dml import CausalForestDML
from econml.inference import BootstrapInference

# === DATA LOADING ===
# Replace with your actual data source
df = pd.read_parquet("experiment_results.parquet")

# === DATA VALIDATION ===
print("Data shape:", df.shape)
print("Treatment distribution:")
print(df["{dag_spec["treatment_variable"]}"].value_counts())
print("\\nOutcome summary:")
print(df["{dag_spec["outcome_variable"]}"].describe())

# === CAUSAL MODEL SPECIFICATION ===
model = CausalModel(
    data=df,
    treatment="{dag_spec["treatment_variable"]}",
    outcome="{dag_spec["outcome_variable"]}",
    common_causes={dag_spec["common_causes"]},
    effect_modifiers={dag_spec.get("effect_modifiers", [])},
    graph="""
{dag_spec["graph_dot"]}
"""
)

# Visualize the causal graph
model.view_model()

# === IDENTIFICATION ===
identified_estimand = model.identify_effect(
    proceed_when_unidentifiable=True
)
print("\\n=== IDENTIFIED ESTIMAND ===")
print(identified_estimand)

# === PRIMARY ANALYSIS: Average Treatment Effect ===
estimate_ate = model.estimate_effect(
    identified_estimand,
    method_name="{estimator}"
)
print("\\n=== ATE ESTIMATE ===")
print(f"ATE: {{estimate_ate.value:.4f}}")
print(f"95% CI: {{estimate_ate.get_confidence_intervals()}}")

# === HETEROGENEOUS TREATMENT EFFECTS ===
# Using EconML CausalForestDML for CATE estimation
X = df[{dag_spec.get("effect_modifiers", [])}].values
T = df["{dag_spec["treatment_variable"]}"].values
Y = df["{dag_spec["outcome_variable"]}"].values
W = df[{dag_spec["common_causes"]}].values if {len(dag_spec["common_causes"])} > 0 else None

cf = CausalForestDML(
    model_y="auto",
    model_t="auto",
    n_estimators=100,
    random_state=42
)

if W is not None:
    cf.fit(Y, T, X=X, W=W, inference=BootstrapInference(n_bootstrap_samples=100))
else:
    cf.fit(Y, T, X=X, inference=BootstrapInference(n_bootstrap_samples=100))

# CATE predictions
cate = cf.effect(X)
print("\\n=== CATE SUMMARY ===")
print(f"Mean CATE: {{cate.mean():.4f}}")
print(f"CATE Std: {{cate.std():.4f}}")
print(f"CATE Range: [{{cate.min():.4f}}, {{cate.max():.4f}}]")

# === REFUTATION TESTS ===
print("\\n=== REFUTATION TESTS ===")

# 1. Placebo treatment
refute_placebo = model.refute_estimate(
    identified_estimand,
    estimate_ate,
    method_name="placebo_treatment_refuter",
    placebo_type="permute"
)
print(f"Placebo test: {{refute_placebo}}")

# 2. Random common cause
refute_random = model.refute_estimate(
    identified_estimand,
    estimate_ate,
    method_name="random_common_cause"
)
print(f"Random common cause: {{refute_random}}")

# 3. Data subset
refute_subset = model.refute_estimate(
    identified_estimand,
    estimate_ate,
    method_name="data_subset_refuter",
    subset_fraction=0.8
)
print(f"Subset refutation: {{refute_subset}}")

# === SAVE RESULTS ===
results = {{
    "ate": estimate_ate.value,
    "ate_ci": estimate_ate.get_confidence_intervals(),
    "cate_mean": float(cate.mean()),
    "cate_std": float(cate.std()),
    "refutation_passed": all([
        refute_placebo.refutation_result is not None,
        refute_random.refutation_result is not None,
        refute_subset.refutation_result is not None
    ])
}}

import json
with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\\n=== ANALYSIS COMPLETE ===")
print("Results saved to analysis_results.json")
'''

    def _build_experiment_template(self, state: ExperimentDesignState) -> ExperimentTemplate:
        """Build complete experiment template.

        Args:
            state: Current agent state

        Returns:
            ExperimentTemplate TypedDict
        """
        power_analysis = state.get("power_analysis", {})
        treatments = state.get("treatments", [])
        outcomes = state.get("outcomes", [])

        # Generate template ID
        template_id = f"exp_{uuid.uuid4().hex[:8]}"

        # Build design summary
        design_summary = (
            f"{state.get('design_type', 'RCT')} experiment with "
            f"n={power_analysis.get('required_sample_size', 'TBD')} "
            f"to achieve {power_analysis.get('achieved_power', 0.8) * 100:.0f}% power"
        )

        # Build monitoring checkpoints
        duration_days = state.get("duration_estimate_days", 56)
        checkpoints = []
        for week in [1, 2, 4]:
            checkpoint_day = week * 7
            if checkpoint_day <= duration_days:
                checkpoints.append(
                    {
                        "day": checkpoint_day,
                        "type": "interim",
                        "checks": [
                            "enrollment_rate",
                            "treatment_fidelity",
                            "outcome_collection_rate",
                        ],
                    }
                )
        checkpoints.append(
            {
                "day": duration_days,
                "type": "final",
                "checks": [
                    "primary_analysis",
                    "refutation_tests",
                    "heterogeneity_analysis",
                ],
            }
        )

        # Generate pre-registration document
        prereg_doc = self._generate_preregistration(state)

        return ExperimentTemplate(
            template_id=template_id,
            template_version=self._template_version,
            design_summary=design_summary,
            treatments=treatments,
            outcomes=outcomes,
            sample_size=power_analysis.get("required_sample_size", 0),
            duration_days=duration_days,
            randomization_unit=state.get("randomization_unit", "individual"),
            randomization_method=state.get("randomization_method", "simple"),
            blocking_variables=state.get("blocking_variables") or [],
            stratification_variables=state.get("stratification_variables") or [],
            pre_registration_document=prereg_doc,
            analysis_code_template=state.get("analysis_code") or "",
            monitoring_checkpoints=checkpoints,
        )

    def _generate_preregistration(self, state: ExperimentDesignState) -> str:
        """Generate pre-registration document.

        Args:
            state: Current agent state

        Returns:
            Pre-registration markdown document
        """
        power_analysis: dict[str, Any] = cast(dict[str, Any], state.get("power_analysis") or {})
        treatments = state.get("treatments", [])
        treatment_name = treatments[0].get("name", "treatment") if treatments else "treatment"
        treatment_desc = treatments[0].get("description", "") if treatments else ""

        outcomes = state.get("outcomes", [])
        primary_outcome = None
        secondary_outcomes = []
        for outcome in outcomes:
            if outcome.get("is_primary", False):
                primary_outcome = outcome.get("name", "outcome")
            else:
                secondary_outcomes.append(outcome.get("name", ""))

        formality = state.get("preregistration_formality", "medium")
        outcome_str = primary_outcome or "outcome"

        # Build document based on formality level
        if formality == "light":
            return self._generate_light_prereg(
                state, power_analysis, treatment_name, outcome_str
            )
        elif formality == "heavy":
            return self._generate_heavy_prereg(
                state,
                power_analysis,
                treatment_name,
                treatment_desc,
                outcome_str,
                secondary_outcomes,
            )
        else:  # medium
            return self._generate_medium_prereg(
                state,
                power_analysis,
                treatment_name,
                treatment_desc,
                outcome_str,
                secondary_outcomes,
            )

    def _generate_light_prereg(
        self, state: ExperimentDesignState, power: dict[str, Any], treatment: str, outcome: str
    ) -> str:
        """Generate light pre-registration."""
        return f"""# Experiment Pre-Registration (Light)

**Date:** {datetime.now().strftime("%Y-%m-%d")}
**Design:** {state.get("design_type", "RCT")}

## Hypothesis
Testing effect of {treatment} on {outcome}.

## Sample Size
n = {power.get("required_sample_size", "TBD")}

## Primary Analysis
Comparison of {outcome} between treatment and control groups.

---
*Auto-generated by E2I Experiment Designer*
"""

    def _generate_medium_prereg(
        self,
        state: ExperimentDesignState,
        power: dict[str, Any],
        treatment: str,
        treatment_desc: str,
        outcome: str,
        secondary: list[str],
    ) -> str:
        """Generate medium pre-registration."""
        validity_score = state.get("overall_validity_score", 0.5)
        threats = state.get("validity_threats", [])
        threat_summary = (
            ", ".join([t.get("threat_name", "") for t in threats[:3]]) or "None identified"
        )

        return f"""# Experiment Pre-Registration

## Study Information
- **Registration Date:** {datetime.now().strftime("%Y-%m-%d")}
- **Design Type:** {state.get("design_type", "RCT")}
- **Validity Score:** {validity_score:.2f}

## Hypotheses
**Primary Hypothesis:**
{state.get("causal_assumptions", ["Treatment affects outcome"])[0] if state.get("causal_assumptions") else "Treatment affects outcome"}

## Design
- **Treatment:** {treatment} - {treatment_desc}
- **Primary Outcome:** {outcome}
- **Secondary Outcomes:** {", ".join(secondary) if secondary else "None"}
- **Sample Size:** {power.get("required_sample_size", "TBD")} (Power: {power.get("achieved_power", 0.8) * 100:.0f}%)
- **Duration:** {state.get("duration_estimate_days", "TBD")} days
- **Randomization:** {state.get("randomization_method", "simple")} at {state.get("randomization_unit", "individual")} level

## Validity Considerations
- **Identified Threats:** {threat_summary}
- **Stratification Variables:** {", ".join(state.get("stratification_variables", [])) or "None"}
- **Blocking Variables:** {", ".join(state.get("blocking_variables", [])) or "None"}

## Analysis Plan
1. Primary: Comparison of {outcome} between arms
2. Heterogeneity: CATE estimation using CausalForestDML
3. Refutation: Placebo, random common cause, subset tests

---
*Pre-registration auto-generated by E2I Experiment Designer v{self._template_version}*
"""

    def _generate_heavy_prereg(
        self,
        state: ExperimentDesignState,
        power: dict[str, Any],
        treatment: str,
        treatment_desc: str,
        outcome: str,
        secondary: list[str],
    ) -> str:
        """Generate comprehensive pre-registration (OSF-style)."""
        medium = self._generate_medium_prereg(
            state, power, treatment, treatment_desc, outcome, secondary
        )

        threats = state.get("validity_threats", [])
        mitigations = state.get("mitigations", [])

        threat_details = (
            "\n".join(
                [
                    f"- **{t.get('threat_name', 'Unknown')}** ({t.get('severity', 'medium')}): {t.get('description', '')}"
                    for t in threats
                ]
            )
            or "No significant threats identified"
        )

        mitigation_details = (
            "\n".join(
                [
                    f"- **{m.get('threat_addressed', 'Unknown')}**: {m.get('strategy', '')}"
                    for m in mitigations
                ]
            )
            or "No specific mitigations required"
        )

        sensitivity = power.get("sensitivity_analysis", {})
        sensitivity_text = ""
        if sensitivity:
            sensitivity_text = "\n### Sensitivity Analysis\n"
            for key, values in sensitivity.items():
                sensitivity_text += f"\n**{key}:**\n"
                for subkey, data in values.items():
                    if isinstance(data, dict):
                        sensitivity_text += f"- {subkey}: n={data.get('sample_size', 'N/A')}\n"

        return f"""{medium}

## Detailed Validity Assessment

### Identified Threats
{threat_details}

### Planned Mitigations
{mitigation_details}

## Power Analysis Details
- **Effect Size:** {power.get("minimum_detectable_effect", "TBD")} ({power.get("effect_size_type", "cohens_d")})
- **Alpha:** {power.get("alpha", 0.05)}
- **Power:** {power.get("achieved_power", 0.8) * 100:.0f}%
- **N per arm:** {power.get("required_sample_size_per_arm", "TBD")}

### Assumptions
{chr(10).join(["- " + a for a in power.get("assumptions", ["Standard assumptions"])])}

{sensitivity_text}

## Causal Graph
```
{state.get("causal_graph_dot", "digraph { treatment -> outcome; }")}
```

## Stopping Rules
- Early stopping for efficacy: Monitored at interim analyses
- Early stopping for futility: If conditional power < 10%
- Early stopping for harm: Safety monitoring committee review

## Data Management
- Data will be collected and stored in compliance with organizational policies
- Primary analysis dataset will be locked after all data collection complete
- Analysis code will be version controlled

---
*Comprehensive pre-registration auto-generated by E2I Experiment Designer v{self._template_version}*
*For OSF registration, copy relevant sections to the appropriate template*
"""

    def _generate_monitoring_spec(self, state: ExperimentDesignState) -> dict[str, Any]:
        """Generate monitoring dashboard specification.

        Args:
            state: Current agent state

        Returns:
            Dashboard specification dictionary
        """
        power_analysis = state.get("power_analysis", {})
        duration_days = state.get("duration_estimate_days", 56)

        return {
            "dashboard_id": f"monitor_{uuid.uuid4().hex[:8]}",
            "refresh_interval_minutes": 60,
            "panels": [
                {
                    "name": "Enrollment Progress",
                    "type": "progress_bar",
                    "target": power_analysis.get("required_sample_size", 500),
                    "metric": "enrolled_count",
                },
                {
                    "name": "Treatment Fidelity",
                    "type": "gauge",
                    "target": 0.95,
                    "metric": "treatment_adherence_rate",
                },
                {
                    "name": "Outcome Collection",
                    "type": "gauge",
                    "target": 0.90,
                    "metric": "outcome_collection_rate",
                },
                {
                    "name": "Balance Check",
                    "type": "table",
                    "metrics": state.get("stratification_variables", []),
                },
                {
                    "name": "Timeline",
                    "type": "timeline",
                    "start_date": datetime.now().isoformat(),
                    "end_date": (
                        (datetime.now() + timedelta(days=duration_days)).isoformat()
                        if duration_days
                        else None
                    ),
                },
            ],
            "alerts": [
                {
                    "name": "Low Enrollment",
                    "condition": "enrollment_rate < 0.5 * expected_rate",
                    "severity": "warning",
                },
                {
                    "name": "High Attrition",
                    "condition": "attrition_rate > 0.15",
                    "severity": "critical",
                },
                {
                    "name": "Treatment Contamination",
                    "condition": "cross_arm_exposure > 0.05",
                    "severity": "critical",
                },
            ],
        }
