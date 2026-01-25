# E2I Causal Analytics: Skills vs MCP Server Evaluation

**Date**: 2025-01-25
**Version**: 1.0
**Status**: Analysis Complete

---

## Executive Summary

This document evaluates the integration of Skills and MCP (Model Context Protocol) servers into the E2I Causal Analytics 21-agent architecture. Based on analysis of your current architecture, LangChain Deep Agents patterns, and available healthcare/pharma MCP tools, I recommend a **hybrid approach** where:

1. **MCP servers** provide data connectivity to external pharmaceutical and healthcare data sources
2. **Skills** encode domain-specific procedures and workflows for your agents
3. Your existing **tool registry system** remains the foundation for agent-to-agent capability composition

---

## Part 1: Current E2I Architecture Analysis

### Agent Tier Summary (21 Agents, 6 Tiers)

| Tier | Agents | Complexity | Current Tool Usage |
|------|--------|------------|-------------------|
| **0** (Foundation) | 8 agents | Standard | Great Expectations, Feast, Optuna, MLflow, BentoML |
| **1** (Coordination) | 2 agents | Complex | Tool composition via ToolRegistry (17 files, 7,923 LOC) |
| **2** (Causal) | 3 agents | Hybrid | DoWhy, EconML, internal composable tools |
| **3** (Monitoring) | 4 agents | Hybrid | Digital Twin simulation, PSI drift detection |
| **4** (ML) | 2 agents | Standard | Ensemble models, linear programming |
| **5** (Learning) | 2 agents | **Deep** | DSPy/GEPA optimization, LangGraph workflows |

### Identified Deep Agents

Based on GEPA budget allocation and architectural complexity:

| Agent | Type | GEPA Budget | Nodes | Characteristics |
|-------|------|-------------|-------|-----------------|
| `feedback_learner` | Deep | 4,000 calls | 8 nodes | Self-improvement, rubric evaluation, signal collection |
| `explainer` | Deep | 4,000 calls | Multi | Extended reasoning, visualization explanation |
| `causal_impact` | Hybrid | 2,000 calls | 5 nodes | DoWhy/EconML + LLM interpretation |
| `experiment_designer` | Hybrid | 2,000 calls | 5 nodes | Power analysis, Digital Twin pre-screening |
| `tool_composer` | Hybrid | 2,000 calls | 4-phase | Multi-tool orchestration, dependency planning |

### Current Tool Architecture

Your system uses a **capability composition model** where:
- Agents expose composable tools via `@composable_tool` decorator
- Central `ToolRegistry` (singleton) manages tool discovery
- `tool_composer` agent orchestrates multi-tool workflows
- No MCP protocol implementation exists currently

---

## Part 2: Skills vs MCP - When to Use Each

### Key Insight from LangChain Deep Agents

> "Skills define procedures; subagents execute complex multi-step work. Your subagents can use skills to effectively manage their context windows."

### The Fundamental Distinction

| Aspect | MCP (Model Context Protocol) | Skills |
|--------|------------------------------|--------|
| **Purpose** | Data connectivity - the **what** | Procedural knowledge - the **how** |
| **Provides** | Access to external data sources, APIs, databases | Instructions, methodology, domain expertise |
| **Complexity** | Full protocol (servers, clients, transports) | Markdown + YAML metadata + optional scripts |
| **Token Cost** | Thousands of tokens per server | Dozens for metadata; full load on-demand |
| **Maintenance** | Server updates, auth management | Update markdown file |

### Decision Framework for E2I

**Use MCP when you need to:**
- Query external pharmaceutical databases (ChEMBL, FDA, ClinicalTrials.gov)
- Access real-time regulatory data (CMS Coverage, ICD-10)
- Retrieve scientific literature (PubMed, bioRxiv)
- Connect to clinical trial registries

**Use Skills when you need to:**
- Encode causal inference workflows (how to interpret DoWhy outputs)
- Define KPI calculation procedures for Remibrutinib, Fabhalta, Kisqali
- Standardize HCP targeting analysis patterns
- Guide agents through pharma commercial analytics methodology

---

## Part 3: MCP Server/Tool Evaluation Matrix

### Tier 1: High Value for E2I (Recommended)

| Tool | Value | Relevance | E2I Agents | Recommended Tools |
|------|-------|-----------|------------|-------------------|
| **ToolUniverse** | **Critical** | Drug discovery, 211+ biomedical tools | `causal_impact`, `experiment_designer`, `explainer` | `drug_info`, `target_validation`, `disease_analysis` |
| **ChEMBL Connector** | **High** | Bioactive compounds, drug properties | `gap_analyzer`, `heterogeneous_optimizer` | `compound_search`, `target_search`, `get_bioactivity` |
| **ClinicalTrials.gov** | **High** | Trial design, patient recruitment | `experiment_designer`, `prediction_synthesizer` | `search_studies`, `get_study_details`, `analyze_endpoints` |
| **PubMed MCP** | **High** | Literature for causal evidence | `causal_impact`, `explainer`, `feedback_learner` | `search_literature`, `get_citations` |

### Tier 2: Medium Value (Selective Integration)

| Tool | Value | Relevance | E2I Agents | Recommended Tools |
|------|-------|-----------|------------|-------------------|
| **ICD-10 Connector** | Medium | Disease classification for cohorts | `cohort_constructor`, `gap_analyzer` | `search_codes`, `verify_code` |
| **CMS Coverage** | Medium | Formulary analysis, coverage gaps | `gap_analyzer`, `resource_optimizer` | `check_coverage`, `get_determination` |
| **OpenPharma FDA MCP** | Medium | Drug labels, adverse events | `drift_monitor`, `health_score` | `search_drugs`, `get_adverse_events` |
| **DS-Star Framework** | Medium | Data science automation | Tier 0 agents | `analyze_data`, `plan_analysis` |

### Tier 3: Lower Priority / Redundant

| Tool | Assessment | Reason |
|------|------------|--------|
| **FHIR Developer Skill** | Low | E2I is commercial analytics, not clinical EHR |
| **Medical MCP (JamesANZ)** | Redundant | Overlaps with ToolUniverse + OpenPharma |
| **Healthcare MCP Public** | Redundant | Overlaps with official Anthropic connectors |
| **DeepSense MCP** | Evaluate | Depends on specific data science needs |
| **reading-plus-ai servers** | Specialized | Deep research, may overlap with existing RAG |

---

## Part 4: Detailed Tool Recommendations by Agent

### Deep Agents (Tier 5) - Highest Value

#### `feedback_learner` (Deep)
**Current**: 8-node LangGraph workflow for self-improvement

**MCP Recommendations**:
| MCP Server | Tools to Expose | Use Case |
|------------|-----------------|----------|
| PubMed | `search_literature` | Find evidence for pattern validation |
| ToolUniverse | `literature_search` | Cross-reference learned patterns |

**Skills Recommendations**:
- `pharma-feedback-patterns.md` - Domain-specific feedback interpretation
- `commercial-analytics-rubric.md` - Evaluation criteria for pharma KPIs

#### `explainer` (Deep)
**Current**: Extended reasoning, visualization generation

**MCP Recommendations**:
| MCP Server | Tools to Expose | Use Case |
|------------|-----------------|----------|
| ChEMBL | `compound_search`, `get_bioactivity` | Explain drug mechanisms |
| ToolUniverse | `drug_info`, `disease_info` | Enrich explanations |
| PubMed | `get_article_details` | Cite supporting evidence |

**Skills Recommendations**:
- `causal-explanation-template.md` - Standard causal narrative structure
- `kpi-explanation-methodology.md` - How to explain TRx/NRx impacts

---

### Hybrid Agents (Tiers 2-3) - High Value

#### `causal_impact` (Hybrid)
**Current**: DoWhy/EconML for causal inference

**MCP Recommendations**:
| MCP Server | Tools to Expose | Use Case |
|------------|-----------------|----------|
| ToolUniverse | `target_validation`, `genetic_evidence` | Validate causal hypotheses |
| ClinicalTrials.gov | `search_studies`, `analyze_endpoints` | Find RCT evidence for causal claims |
| PubMed | `search_literature` | Literature-backed causal evidence |

**Skills Recommendations**:
- `dowhy-interpretation.md` - How to interpret refutation tests
- `causal-chain-validation.md` - Procedures for validating causal paths

#### `experiment_designer` (Hybrid)
**Current**: A/B test design with Digital Twin pre-screening

**MCP Recommendations**:
| MCP Server | Tools to Expose | Use Case |
|------------|-----------------|----------|
| ClinicalTrials.gov | `search_studies`, `get_study_details` | Reference trial designs |
| ToolUniverse | `clinical_trial_tools` | Protocol drafting support |

**Skills Recommendations**:
- `ab-test-design-protocol.md` - Standard A/B test methodology
- `digital-twin-validation.md` - Twin fidelity validation procedures

#### `gap_analyzer` (Hybrid)
**Current**: ROI opportunity detection

**MCP Recommendations**:
| MCP Server | Tools to Expose | Use Case |
|------------|-----------------|----------|
| ChEMBL | `compound_search`, `target_search` | Identify market gaps |
| CMS Coverage | `check_coverage` | Coverage gap analysis |
| ICD-10 | `search_codes` | Diagnosis classification for gap analysis |

---

### Standard Agents (Tiers 0, 1, 4) - Selective Value

#### `tool_composer` (Tier 1)
**Current**: Multi-tool orchestration

**MCP Recommendations**:
- Make ALL MCP tools available through `tool_composer`
- `tool_composer` should be the gateway for MCP tool access

**Skills Recommendations**:
- `tool-composition-patterns.md` - Standard composition workflows
- `mcp-tool-selection.md` - When to use which MCP tool

#### Tier 0 Foundation Agents
**MCP Recommendations**:
| Agent | MCP Server | Tools |
|-------|------------|-------|
| `cohort_constructor` | ICD-10 | `search_codes`, `verify_code` |
| `data_preparer` | DS-Star | `analyze_data` (optional) |

---

## Part 5: Implementation Architecture

### Recommended Integration Pattern

```
                    ┌─────────────────────────────────────────┐
                    │          User Query                      │
                    └─────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │         Orchestrator (Tier 1)            │
                    │  - Intent classification                 │
                    │  - Agent routing                         │
                    └─────────────────┬───────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
┌─────────▼─────────┐   ┌─────────────▼─────────────┐   ┌────────▼────────┐
│  Tool Composer    │   │   Domain Agents           │   │  Deep Agents    │
│  (Tier 1)         │   │   (Tiers 2-4)             │   │  (Tier 5)       │
│                   │   │                           │   │                 │
│  Orchestrates:    │   │  Use directly:            │   │  Extended       │
│  - Internal tools │   │  - DoWhy/EconML           │   │  reasoning +    │
│  - MCP tools      │   │  - Digital Twin           │   │  Skills         │
│  - Composable     │   │  - Domain-specific        │   │                 │
└─────────┬─────────┘   └─────────────┬─────────────┘   └────────┬────────┘
          │                           │                           │
          └───────────────────────────┼───────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │          Tool & Data Layer              │
                    │                                         │
                    │  ┌─────────────┐  ┌─────────────────┐   │
                    │  │ToolRegistry │  │  MCP Gateway    │   │
                    │  │(Internal)   │  │                 │   │
                    │  │             │  │  - ToolUniverse │   │
                    │  │@composable  │  │  - ChEMBL       │   │
                    │  │_tool        │  │  - PubMed       │   │
                    │  │             │  │  - ClinicalTrials│  │
                    │  └─────────────┘  └─────────────────┘   │
                    │                                         │
                    │  ┌─────────────────────────────────┐    │
                    │  │        Skills Repository        │    │
                    │  │  .claude/skills/                │    │
                    │  │  - pharma-analytics/            │    │
                    │  │  - causal-inference/            │    │
                    │  │  - experiment-design/           │    │
                    │  └─────────────────────────────────┘    │
                    └─────────────────────────────────────────┘
```

### MCP Gateway Pattern

Instead of exposing all MCP tools to all agents, create an **MCP Gateway** layer:

```python
# src/mcp/gateway.py

class MCPGateway:
    """Centralized MCP tool access with agent permissions."""

    AGENT_PERMISSIONS = {
        "tool_composer": ["*"],  # Full access
        "causal_impact": ["tooluniverse", "pubmed", "clinicaltrials"],
        "experiment_designer": ["clinicaltrials", "tooluniverse"],
        "gap_analyzer": ["chembl", "cms_coverage", "icd10"],
        "explainer": ["chembl", "tooluniverse", "pubmed"],
        "feedback_learner": ["pubmed"],
        "cohort_constructor": ["icd10"],
    }

    def get_tools_for_agent(self, agent_name: str) -> list[Tool]:
        """Return only permitted MCP tools for an agent."""
        permissions = self.AGENT_PERMISSIONS.get(agent_name, [])
        if "*" in permissions:
            return self.all_mcp_tools
        return [t for t in self.all_mcp_tools if t.server in permissions]
```

### Skills Directory Structure

```
.claude/skills/
├── pharma-analytics/
│   ├── SKILL.md                    # Master skill index
│   ├── kpi-calculation.md          # TRx, NRx, conversion rate procedures
│   ├── brand-context.md            # Remibrutinib, Fabhalta, Kisqali specifics
│   └── hcp-targeting.md            # HCP analysis methodology
├── causal-inference/
│   ├── SKILL.md
│   ├── dowhy-workflow.md           # DoWhy interpretation procedures
│   ├── refutation-guide.md         # How to interpret refutation tests
│   └── sensitivity-analysis.md     # Sensitivity analysis procedures
├── experiment-design/
│   ├── SKILL.md
│   ├── ab-test-protocol.md         # A/B test design methodology
│   ├── power-analysis.md           # Power calculation procedures
│   └── digital-twin-validation.md  # Twin fidelity procedures
└── data-science/
    ├── SKILL.md
    ├── drift-detection.md          # PSI calculation procedures
    └── model-validation.md         # Model validation workflows
```

---

## Part 6: User Question Integration

### How MCP Tools Complement User Questions

| User Question Type | MCP Tools Used | Example |
|-------------------|----------------|---------|
| "What's the causal impact of rep visits on Kisqali TRx?" | Internal DoWhy + PubMed for evidence | Agent uses internal tools, PubMed validates |
| "Compare our trial design to similar studies" | ClinicalTrials.gov | `search_studies` with similar endpoints |
| "Why did Remibrutinib market share drop in Q3?" | ChEMBL + ToolUniverse | Drug mechanism context for explanation |
| "Find coverage gaps for Fabhalta" | CMS Coverage + ICD-10 | Coverage determination + diagnosis mapping |
| "What biomarkers predict patient response?" | ToolUniverse + ChEMBL | Target validation, bioactivity data |

### Query Routing Enhancement

Add MCP awareness to your orchestrator:

```python
# Enhanced intent classification
INTENT_MCP_MAPPING = {
    "LITERATURE_EVIDENCE": ["pubmed"],
    "DRUG_MECHANISM": ["chembl", "tooluniverse"],
    "TRIAL_REFERENCE": ["clinicaltrials"],
    "COVERAGE_ANALYSIS": ["cms_coverage", "icd10"],
    "TARGET_VALIDATION": ["tooluniverse"],
}
```

---

## Part 7: Redundancy Analysis

### Redundant Tools to Avoid

| Tool Set 1 | Tool Set 2 | Recommendation |
|------------|------------|----------------|
| Medical MCP (JamesANZ) | Healthcare MCP Public | Use **OpenPharma FDA MCP** instead |
| Healthcare MCP Public | Official Anthropic Connectors | Use **official connectors** (better support) |
| Multiple ICD-10 implementations | - | Use **DeepSense ICD-10 Connector** |
| DS-Star | DeepSense MCP | Evaluate based on specific needs |

### Recommended Primary Stack

| Category | Primary Choice | Reason |
|----------|----------------|--------|
| Drug Data | **ChEMBL Connector** (official) | 2M+ compounds, EMBL-EBI quality |
| Biomedical Tools | **ToolUniverse** | 211+ tools, Harvard quality |
| Clinical Trials | **ClinicalTrials.gov** (official) | 500K+ studies, NIH source |
| Literature | **PubMed** (official) | 35M+ articles, gold standard |
| Medical Codes | **ICD-10 Connector** (DeepSense) | CMS 2026 data, unified search |
| Coverage | **CMS Coverage** (official) | LCD/NCD determinations |

---

## Part 8: Implementation Roadmap

### Phase 1: Foundation (2-3 weeks)
1. Create `.claude/skills/` directory structure
2. Write initial skills for `causal_impact` and `experiment_designer`
3. Implement MCP Gateway pattern in `src/mcp/`
4. Add `langchain-mcp-adapters` dependency

### Phase 2: High-Value MCP Integration (2-3 weeks)
1. Integrate ToolUniverse MCP (211+ biomedical tools)
2. Integrate ChEMBL Connector
3. Integrate PubMed MCP
4. Update `tool_composer` to orchestrate MCP tools

### Phase 3: Skills Development (2-3 weeks)
1. Write pharma-analytics skills (KPI procedures)
2. Write causal-inference skills (DoWhy workflows)
3. Write experiment-design skills
4. Test skills with deep agents

### Phase 4: Full Integration (2-3 weeks)
1. Add ClinicalTrials.gov connector
2. Add ICD-10 + CMS Coverage for gap analysis
3. Implement user question routing with MCP awareness
4. Performance optimization and caching

---

## Part 9: Cost-Benefit Summary

### Token Economics

| Approach | Token Cost | Value |
|----------|------------|-------|
| Skills only | Low (~50-200 tokens metadata) | High procedural value |
| MCP only | High (~1,000-10,000 tokens per server) | High data access value |
| **Hybrid (recommended)** | Medium | Maximum value |

### Expected Benefits

1. **Enhanced Causal Evidence**: PubMed + ClinicalTrials.gov for validating causal claims
2. **Richer Explanations**: ChEMBL + ToolUniverse for drug mechanism context
3. **Better Experiment Design**: Reference similar trials from ClinicalTrials.gov
4. **Gap Analysis**: CMS Coverage + ICD-10 for coverage opportunity identification
5. **Standardized Procedures**: Skills ensure consistent methodology across agents

### Risks to Mitigate

| Risk | Mitigation |
|------|------------|
| Context bloat from MCP | Use MCP Gateway with agent permissions |
| Tool selection confusion | Let `tool_composer` be the orchestrator |
| Maintenance burden | Start with official Anthropic connectors |
| Token cost explosion | Cache MCP responses, use skills for procedures |

---

## Appendix: Sources

### LangChain Deep Agents
- [LangChain Deep Agents GitHub](https://github.com/langchain-ai/deepagents)
- [Deep Agents Overview](https://docs.langchain.com/oss/python/deepagents/overview)
- [Building Multi-Agent Applications](https://www.blog.langchain.com/building-multi-agent-applications-with-deep-agents/)

### Skills vs MCP
- [Skills Explained - Claude Blog](https://claude.com/blog/skills-explained)
- [Claude Skills vs MCP Comparison](https://skywork.ai/blog/ai-agent/claude-skills-vs-mcp-vs-llm-tools-comparison-2025/)
- [Extending Claude with Skills and MCP](https://claude.com/blog/extending-claude-capabilities-with-skills-mcp-servers)

### Healthcare MCP Tools
- [Anthropic Healthcare Initiative](https://www.anthropic.com/news/healthcare-life-sciences)
- [ToolUniverse - Harvard](https://github.com/mims-harvard/ToolUniverse)
- [OpenPharma MCP Servers](https://github.com/openpharma-org)
- [ChEMBL Connector](https://claude.com/connectors/chembl)
- [ClinicalTrials.gov Connector](https://claude.com/resources/tutorials/using-the-clinicaltrials-gov-connector-in-claude)

### Data Science Tools
- [DS-Star Framework](https://github.com/JulesLscx/DS-Star)
- [reading-plus-ai MCP Servers](https://github.com/reading-plus-ai)

---

## Conclusion

The E2I Causal Analytics system is well-architected for MCP and Skills integration. The key recommendations are:

1. **Adopt the hybrid model**: MCP for data connectivity, Skills for procedural knowledge
2. **Prioritize high-value tools**: ToolUniverse, ChEMBL, PubMed, ClinicalTrials.gov
3. **Use MCP Gateway pattern**: Control tool exposure per agent
4. **Build domain skills**: Encode pharma commercial analytics procedures
5. **Route through tool_composer**: Make it the MCP orchestration hub

This approach maximizes value while managing context costs and maintaining your existing tool registry architecture.

---

## Related Documents

- **[Domain Skills Framework](../../skills/SKILL_INTEGRATION.md)**: Expanded guide on building pharma commercial analytics skills, including:
  - Complete skill directory structure
  - 10+ skill file templates (KPI calculation, brand analytics, causal inference, etc.)
  - Agent-skill mapping patterns
  - SkillLoader implementation
  - Token efficiency strategies
  - MCP + Skills integration patterns
