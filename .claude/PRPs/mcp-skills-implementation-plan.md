# MCP & Skills Implementation Plan

**Date**: 2025-01-26
**Status**: Draft
**Priority**: High
**Related**: [Skills vs MCP Evaluation](./skills-vs-mcp-evaluation.md)

---

## Executive Summary

This plan implements the hybrid approach for extending E2I agent capabilities:
- **Phase 1**: Remote Anthropic Connectors (zero infrastructure)
- **Phase 2**: Domain Skills Framework (procedural knowledge)
- **Phase 3**: Agent Integration (wire up connectors + skills)
- **Phase 4**: Self-Hosted MCP (optional, if needed)

**Key Decision**: Start with Anthropic's hosted connectors to avoid infrastructure overhead. The droplet already runs MLflow, Opik, Redis, FalkorDB, and the E2I API.

---

## Current State

### Infrastructure (DigitalOcean Droplet)
| Service | Port | Status |
|---------|------|--------|
| E2I API | 8000 | Running |
| MLflow | 5000 | Running |
| Opik | 5173/8080 | Running |
| Redis | 6382 | Running |
| FalkorDB | 6381 | Running |

### Agent Architecture
- 21 agents across 6 tiers
- Central ToolRegistry with `@composable_tool` pattern
- No MCP implementation currently
- No formal skills system

---

## Phase 1: Remote Anthropic Connectors

**Duration**: 1-2 weeks
**Infrastructure**: None (use Anthropic's hosted services)
**Cost**: Claude Team/Enterprise subscription

### 1.1 Prerequisites

- [ ] Verify Claude API access with connector capabilities
- [ ] Confirm subscription tier supports connectors (Pro/Max/Team/Enterprise)
- [ ] Test connector availability via Claude API

```bash
# Test connector access
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "content-type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Using ChEMBL, search for ribociclib"}]
  }'
```

### 1.2 Create Connector Gateway

**File**: `src/mcp/connector_gateway.py`

```python
"""
Gateway for accessing Anthropic's remote connectors.
Provides unified interface for E2I agents to query external data sources.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any
import anthropic


class Connector(Enum):
    """Available Anthropic connectors for E2I."""
    CHEMBL = "chembl"
    CLINICALTRIALS = "clinicaltrials_gov"
    PUBMED = "pubmed"
    ICD10 = "icd10"
    CMS_COVERAGE = "cms_coverage"


@dataclass
class ConnectorQuery:
    """Query to send to a connector."""
    connector: Connector
    query: str
    parameters: dict[str, Any] | None = None


@dataclass
class ConnectorResult:
    """Result from a connector query."""
    connector: Connector
    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    tokens_used: int = 0


class AnthropicConnectorGateway:
    """
    Gateway for Anthropic's hosted connectors.

    Usage:
        gateway = AnthropicConnectorGateway()
        result = await gateway.query(
            Connector.CHEMBL,
            "Find bioactivity data for ribociclib"
        )
    """

    # Agent permissions - which agents can use which connectors
    AGENT_PERMISSIONS: dict[str, list[Connector]] = {
        "tool_composer": list(Connector),  # Full access
        "causal_impact": [Connector.CHEMBL, Connector.PUBMED, Connector.CLINICALTRIALS],
        "experiment_designer": [Connector.CLINICALTRIALS, Connector.PUBMED],
        "gap_analyzer": [Connector.CHEMBL, Connector.CMS_COVERAGE, Connector.ICD10],
        "heterogeneous_optimizer": [Connector.CHEMBL, Connector.ICD10],
        "explainer": [Connector.CHEMBL, Connector.PUBMED],
        "feedback_learner": [Connector.PUBMED],
        "cohort_constructor": [Connector.ICD10],
        "prediction_synthesizer": [Connector.CLINICALTRIALS],
        "drift_monitor": [Connector.PUBMED],  # For literature on drift patterns
    }

    def __init__(self, api_key: str | None = None):
        self.client = anthropic.Anthropic(api_key=api_key)
        self._cache: dict[str, ConnectorResult] = {}

    def check_permission(self, agent_name: str, connector: Connector) -> bool:
        """Check if agent has permission to use connector."""
        allowed = self.AGENT_PERMISSIONS.get(agent_name, [])
        return connector in allowed

    async def query(
        self,
        connector: Connector,
        query: str,
        agent_name: str | None = None,
        use_cache: bool = True,
    ) -> ConnectorResult:
        """
        Query a connector with permission checking.

        Args:
            connector: Which connector to query
            query: Natural language query
            agent_name: Requesting agent (for permission check)
            use_cache: Whether to use cached results

        Returns:
            ConnectorResult with data or error
        """
        # Permission check
        if agent_name and not self.check_permission(agent_name, connector):
            return ConnectorResult(
                connector=connector,
                success=False,
                error=f"Agent '{agent_name}' not permitted to use {connector.value}"
            )

        # Cache check
        cache_key = f"{connector.value}:{query}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # Build connector-specific prompt
        prompt = self._build_prompt(connector, query)

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )

            result = ConnectorResult(
                connector=connector,
                success=True,
                data={"response": response.content[0].text},
                tokens_used=response.usage.input_tokens + response.usage.output_tokens
            )

            # Cache successful results
            if use_cache:
                self._cache[cache_key] = result

            return result

        except anthropic.APIError as e:
            return ConnectorResult(
                connector=connector,
                success=False,
                error=str(e)
            )

    def _build_prompt(self, connector: Connector, query: str) -> str:
        """Build connector-specific prompt."""
        prefixes = {
            Connector.CHEMBL: "Using the ChEMBL connector, ",
            Connector.CLINICALTRIALS: "Using the ClinicalTrials.gov connector, ",
            Connector.PUBMED: "Using the PubMed connector, ",
            Connector.ICD10: "Using the ICD-10 connector, ",
            Connector.CMS_COVERAGE: "Using the CMS Coverage connector, ",
        }
        return f"{prefixes[connector]}{query}"

    async def query_multiple(
        self,
        queries: list[ConnectorQuery],
        agent_name: str | None = None,
    ) -> list[ConnectorResult]:
        """Query multiple connectors in parallel."""
        import asyncio
        tasks = [
            self.query(q.connector, q.query, agent_name)
            for q in queries
        ]
        return await asyncio.gather(*tasks)


# Singleton instance
_gateway: AnthropicConnectorGateway | None = None


def get_connector_gateway() -> AnthropicConnectorGateway:
    """Get singleton gateway instance."""
    global _gateway
    if _gateway is None:
        _gateway = AnthropicConnectorGateway()
    return _gateway
```

### 1.3 Create Connector-Specific Helpers

**File**: `src/mcp/connectors/__init__.py`

```python
from .chembl import ChEMBLConnector
from .clinicaltrials import ClinicalTrialsConnector
from .pubmed import PubMedConnector

__all__ = ["ChEMBLConnector", "ClinicalTrialsConnector", "PubMedConnector"]
```

**File**: `src/mcp/connectors/chembl.py`

```python
"""ChEMBL connector helpers for drug discovery queries."""

from dataclasses import dataclass
from ..connector_gateway import get_connector_gateway, Connector


@dataclass
class CompoundInfo:
    """Structured compound information from ChEMBL."""
    chembl_id: str
    name: str
    molecular_formula: str | None = None
    molecular_weight: float | None = None
    max_phase: int | None = None  # Clinical phase (4 = approved)
    indication: str | None = None


@dataclass
class BioactivityData:
    """Bioactivity measurement from ChEMBL."""
    target_name: str
    activity_type: str  # IC50, EC50, Ki, etc.
    value: float
    units: str
    assay_type: str


class ChEMBLConnector:
    """Helper for ChEMBL-specific queries."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.gateway = get_connector_gateway()

    async def search_compound(self, name: str) -> CompoundInfo | None:
        """Search for a compound by name."""
        result = await self.gateway.query(
            Connector.CHEMBL,
            f"Search for compound '{name}' and return its ChEMBL ID, molecular formula, "
            f"molecular weight, max clinical phase, and primary indication.",
            agent_name=self.agent_name
        )
        if result.success:
            # Parse response into structured data
            return self._parse_compound_info(result.data)
        return None

    async def get_bioactivity(
        self,
        compound_name: str,
        target_name: str | None = None
    ) -> list[BioactivityData]:
        """Get bioactivity data for a compound."""
        query = f"Get bioactivity data (IC50, EC50, Ki values) for '{compound_name}'"
        if target_name:
            query += f" against target '{target_name}'"

        result = await self.gateway.query(
            Connector.CHEMBL,
            query,
            agent_name=self.agent_name
        )
        if result.success:
            return self._parse_bioactivity(result.data)
        return []

    async def find_similar_compounds(self, compound_name: str) -> list[str]:
        """Find structurally similar compounds."""
        result = await self.gateway.query(
            Connector.CHEMBL,
            f"Find compounds structurally similar to '{compound_name}' "
            f"that are in clinical development or approved.",
            agent_name=self.agent_name
        )
        if result.success:
            return self._parse_compound_list(result.data)
        return []

    def _parse_compound_info(self, data: dict) -> CompoundInfo | None:
        """Parse compound info from response."""
        # Implementation depends on actual response format
        # This is a placeholder
        return None

    def _parse_bioactivity(self, data: dict) -> list[BioactivityData]:
        """Parse bioactivity data from response."""
        return []

    def _parse_compound_list(self, data: dict) -> list[str]:
        """Parse list of compound names from response."""
        return []
```

**File**: `src/mcp/connectors/clinicaltrials.py`

```python
"""ClinicalTrials.gov connector helpers for trial queries."""

from dataclasses import dataclass
from ..connector_gateway import get_connector_gateway, Connector


@dataclass
class ClinicalTrial:
    """Structured clinical trial information."""
    nct_id: str
    title: str
    status: str  # Recruiting, Completed, etc.
    phase: str  # Phase 1, Phase 2, etc.
    conditions: list[str]
    interventions: list[str]
    enrollment: int | None = None
    primary_outcome: str | None = None
    start_date: str | None = None
    completion_date: str | None = None


class ClinicalTrialsConnector:
    """Helper for ClinicalTrials.gov queries."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.gateway = get_connector_gateway()

    async def search_trials(
        self,
        condition: str,
        intervention: str | None = None,
        phase: str | None = None,
        status: str = "Recruiting",
        limit: int = 10
    ) -> list[ClinicalTrial]:
        """Search for clinical trials."""
        query = f"Search for clinical trials for condition '{condition}'"
        if intervention:
            query += f" with intervention '{intervention}'"
        if phase:
            query += f" in {phase}"
        query += f" that are {status}. Return up to {limit} results."

        result = await self.gateway.query(
            Connector.CLINICALTRIALS,
            query,
            agent_name=self.agent_name
        )
        if result.success:
            return self._parse_trials(result.data)
        return []

    async def get_trial_details(self, nct_id: str) -> ClinicalTrial | None:
        """Get detailed information about a specific trial."""
        result = await self.gateway.query(
            Connector.CLINICALTRIALS,
            f"Get full details for clinical trial {nct_id} including "
            f"endpoints, eligibility criteria, and study design.",
            agent_name=self.agent_name
        )
        if result.success:
            return self._parse_trial_details(result.data)
        return None

    async def find_similar_trials(
        self,
        nct_id: str,
        same_condition: bool = True
    ) -> list[ClinicalTrial]:
        """Find trials with similar design."""
        result = await self.gateway.query(
            Connector.CLINICALTRIALS,
            f"Find clinical trials with similar design to {nct_id}, "
            f"focusing on {'same condition' if same_condition else 'similar methodology'}.",
            agent_name=self.agent_name
        )
        if result.success:
            return self._parse_trials(result.data)
        return []

    async def analyze_endpoints(self, condition: str) -> dict:
        """Analyze common endpoints used in trials for a condition."""
        result = await self.gateway.query(
            Connector.CLINICALTRIALS,
            f"Analyze the primary and secondary endpoints commonly used "
            f"in clinical trials for {condition}. Summarize the most frequent "
            f"choices and their measurement methods.",
            agent_name=self.agent_name
        )
        if result.success:
            return result.data
        return {}

    def _parse_trials(self, data: dict) -> list[ClinicalTrial]:
        """Parse trials from response."""
        return []

    def _parse_trial_details(self, data: dict) -> ClinicalTrial | None:
        """Parse detailed trial info."""
        return None
```

### 1.4 Unit Tests

**File**: `tests/unit/test_mcp/test_connector_gateway.py`

```python
"""Tests for Anthropic connector gateway."""

import pytest
from unittest.mock import AsyncMock, patch
from src.mcp.connector_gateway import (
    AnthropicConnectorGateway,
    Connector,
    ConnectorResult
)


class TestConnectorPermissions:
    """Test agent permission checking."""

    def test_tool_composer_has_full_access(self):
        gateway = AnthropicConnectorGateway()
        for connector in Connector:
            assert gateway.check_permission("tool_composer", connector)

    def test_causal_impact_permissions(self):
        gateway = AnthropicConnectorGateway()
        assert gateway.check_permission("causal_impact", Connector.CHEMBL)
        assert gateway.check_permission("causal_impact", Connector.PUBMED)
        assert not gateway.check_permission("causal_impact", Connector.CMS_COVERAGE)

    def test_unknown_agent_no_permissions(self):
        gateway = AnthropicConnectorGateway()
        assert not gateway.check_permission("unknown_agent", Connector.CHEMBL)


class TestConnectorQueries:
    """Test connector query execution."""

    @pytest.mark.asyncio
    async def test_query_with_permission_denied(self):
        gateway = AnthropicConnectorGateway()
        result = await gateway.query(
            Connector.CMS_COVERAGE,
            "test query",
            agent_name="causal_impact"
        )
        assert not result.success
        assert "not permitted" in result.error

    @pytest.mark.asyncio
    @patch.object(AnthropicConnectorGateway, 'client')
    async def test_query_success(self, mock_client):
        mock_client.messages.create = AsyncMock(return_value=MockResponse())
        gateway = AnthropicConnectorGateway()
        result = await gateway.query(
            Connector.CHEMBL,
            "Find ribociclib",
            agent_name="causal_impact"
        )
        assert result.success

    @pytest.mark.asyncio
    async def test_query_caching(self):
        gateway = AnthropicConnectorGateway()
        # Pre-populate cache
        cached_result = ConnectorResult(
            connector=Connector.CHEMBL,
            success=True,
            data={"cached": True}
        )
        gateway._cache["chembl:test query"] = cached_result

        result = await gateway.query(
            Connector.CHEMBL,
            "test query",
            use_cache=True
        )
        assert result.data == {"cached": True}
```

### 1.5 Deliverables Checklist

- [ ] `src/mcp/__init__.py`
- [ ] `src/mcp/connector_gateway.py`
- [ ] `src/mcp/connectors/__init__.py`
- [ ] `src/mcp/connectors/chembl.py`
- [ ] `src/mcp/connectors/clinicaltrials.py`
- [ ] `src/mcp/connectors/pubmed.py`
- [ ] `src/mcp/connectors/icd10.py`
- [ ] `src/mcp/connectors/cms_coverage.py`
- [ ] `tests/unit/test_mcp/test_connector_gateway.py`
- [ ] `tests/unit/test_mcp/test_connectors.py`
- [ ] `tests/integration/test_mcp_integration.py`

---

## Phase 2: Domain Skills Framework

**Duration**: 2-3 weeks
**Dependencies**: None (can run parallel to Phase 1)

### 2.1 Create Skills Directory Structure

```bash
mkdir -p .claude/skills/{pharma-commercial,causal-inference,experiment-design,gap-analysis,data-quality}
```

### 2.2 Implement SkillLoader

**File**: `src/skills/__init__.py`

```python
from .loader import SkillLoader, Skill, SkillMetadata
from .matcher import SkillMatcher

__all__ = ["SkillLoader", "Skill", "SkillMetadata", "SkillMatcher"]
```

**File**: `src/skills/loader.py`

```python
"""
Skill loading and caching for E2I agents.
Skills are markdown files with YAML frontmatter that encode domain procedures.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml
import re


@dataclass
class SkillMetadata:
    """Metadata from skill YAML frontmatter."""
    name: str
    version: str
    description: str
    triggers: list[str] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)


@dataclass
class Skill:
    """A loaded skill with metadata and content."""
    path: str
    metadata: SkillMetadata
    content: str
    sections: dict[str, str] = field(default_factory=dict)

    def get_section(self, name: str) -> str:
        """Get a specific section by name (case-insensitive)."""
        name_lower = name.lower()
        for key, value in self.sections.items():
            if key.lower() == name_lower:
                return value
        return ""

    def matches_trigger(self, query: str) -> bool:
        """Check if query matches any skill trigger."""
        query_lower = query.lower()
        return any(
            trigger.lower() in query_lower
            for trigger in self.metadata.triggers
        )

    def is_for_agent(self, agent_name: str) -> bool:
        """Check if skill is intended for an agent."""
        return agent_name in self.metadata.agents or not self.metadata.agents

    @property
    def token_estimate(self) -> int:
        """Estimate token count (rough: 4 chars per token)."""
        return len(self.content) // 4


class SkillLoader:
    """
    Loads and caches domain skills for E2I agents.

    Usage:
        loader = SkillLoader()
        skill = await loader.load("pharma-commercial/kpi-calculation.md")
        trx_section = skill.get_section("TRx (Total Prescriptions)")
    """

    SKILLS_DIR = Path(".claude/skills")
    _cache: dict[str, Skill] = {}

    @classmethod
    async def load(cls, skill_path: str) -> Skill:
        """
        Load a skill by relative path.

        Args:
            skill_path: Path relative to .claude/skills/

        Returns:
            Loaded Skill object

        Raises:
            FileNotFoundError: If skill file doesn't exist
        """
        if skill_path in cls._cache:
            return cls._cache[skill_path]

        full_path = cls.SKILLS_DIR / skill_path
        if not full_path.exists():
            raise FileNotFoundError(f"Skill not found: {skill_path}")

        content = full_path.read_text()
        skill = cls._parse_skill(skill_path, content)
        cls._cache[skill_path] = skill
        return skill

    @classmethod
    async def load_section(cls, skill_path: str, section_name: str) -> str:
        """Load only a specific section from a skill (more token-efficient)."""
        skill = await cls.load(skill_path)
        return skill.get_section(section_name)

    @classmethod
    async def list_skills(cls, category: str | None = None) -> list[str]:
        """List available skill paths, optionally filtered by category."""
        skills = []
        search_dir = cls.SKILLS_DIR / category if category else cls.SKILLS_DIR

        for path in search_dir.rglob("*.md"):
            if path.name != "SKILL.md":  # Skip index files
                rel_path = path.relative_to(cls.SKILLS_DIR)
                skills.append(str(rel_path))

        return sorted(skills)

    @classmethod
    async def get_metadata_only(cls, skill_path: str) -> SkillMetadata:
        """Load only metadata (efficient for matching)."""
        skill = await cls.load(skill_path)
        return skill.metadata

    @classmethod
    def clear_cache(cls):
        """Clear the skill cache."""
        cls._cache.clear()

    @classmethod
    def _parse_skill(cls, path: str, content: str) -> Skill:
        """Parse skill markdown with YAML frontmatter."""
        # Split frontmatter from content
        frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)

        if frontmatter_match:
            frontmatter_text = frontmatter_match.group(1)
            body = content[frontmatter_match.end():]
            try:
                frontmatter = yaml.safe_load(frontmatter_text) or {}
            except yaml.YAMLError:
                frontmatter = {}
        else:
            frontmatter = {}
            body = content

        metadata = SkillMetadata(
            name=frontmatter.get("name", path),
            version=frontmatter.get("version", "1.0"),
            description=frontmatter.get("description", ""),
            triggers=frontmatter.get("triggers", []),
            agents=frontmatter.get("agents", []),
        )

        sections = cls._parse_sections(body)

        return Skill(
            path=path,
            metadata=metadata,
            content=body,
            sections=sections,
        )

    @classmethod
    def _parse_sections(cls, body: str) -> dict[str, str]:
        """Parse markdown sections into a dictionary."""
        sections = {}
        current_section = None
        current_content = []

        for line in body.split("\n"):
            # Match ## or ### headers
            header_match = re.match(r'^(#{2,3})\s+(.+)$', line)
            if header_match:
                # Save previous section
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = header_match.group(2).strip()
                current_content = []
            else:
                current_content.append(line)

        # Save final section
        if current_section:
            sections[current_section] = "\n".join(current_content).strip()

        return sections
```

**File**: `src/skills/matcher.py`

```python
"""Skill matching for automatic skill discovery."""

from dataclasses import dataclass
from .loader import SkillLoader, Skill


@dataclass
class SkillMatch:
    """A matched skill with relevance score."""
    skill_path: str
    score: float
    matched_triggers: list[str]


class SkillMatcher:
    """
    Matches queries to relevant skills.

    Usage:
        matcher = SkillMatcher()
        matches = await matcher.find_matches("calculate TRx for Kisqali")
        # Returns: [SkillMatch(skill_path="pharma-commercial/kpi-calculation.md", ...)]
    """

    # Keyword weights for scoring
    KEYWORD_WEIGHTS = {
        # KPI terms
        "trx": 2.0, "nrx": 2.0, "nbrx": 2.0, "prescription": 1.5,
        "market share": 2.0, "conversion": 1.5, "adherence": 1.5,
        "persistence": 1.5, "pdc": 1.5, "roi": 2.0,

        # Brand terms
        "kisqali": 2.0, "fabhalta": 2.0, "remibrutinib": 2.0,
        "ribociclib": 1.5, "iptacopan": 1.5,

        # Causal terms
        "causal": 2.0, "confounder": 2.0, "dowhy": 2.0, "econml": 1.5,
        "refutation": 1.5, "sensitivity": 1.5, "ate": 1.5, "cate": 1.5,

        # Experiment terms
        "experiment": 1.5, "validity": 2.0, "power analysis": 2.0,
        "sample size": 1.5, "randomization": 1.5,

        # Gap analysis terms
        "gap": 1.5, "opportunity": 1.5, "revenue": 1.5, "cost": 1.0,
    }

    # Skill path to keyword mappings
    SKILL_KEYWORDS = {
        "pharma-commercial/kpi-calculation.md": [
            "trx", "nrx", "nbrx", "prescription", "market share",
            "conversion", "adherence", "persistence", "pdc", "roi"
        ],
        "pharma-commercial/brand-analytics.md": [
            "kisqali", "fabhalta", "remibrutinib", "brand", "competitor",
            "ribociclib", "iptacopan", "cdk4/6", "pnh", "csu"
        ],
        "pharma-commercial/patient-journey.md": [
            "patient journey", "funnel", "stage", "aware", "considering",
            "prescribed", "first fill", "adherent", "discontinued"
        ],
        "causal-inference/confounder-identification.md": [
            "confounder", "confounding", "adjustment", "control for",
            "instrumental variable", "bias"
        ],
        "causal-inference/dowhy-workflow.md": [
            "causal effect", "ate", "cate", "dowhy", "estimation",
            "refutation", "sensitivity", "e-value"
        ],
        "experiment-design/validity-threats.md": [
            "validity", "selection bias", "contamination", "attrition",
            "temporal", "measurement error", "experiment design"
        ],
        "experiment-design/power-analysis.md": [
            "power", "sample size", "mde", "detectable effect",
            "statistical power"
        ],
        "gap-analysis/roi-estimation.md": [
            "roi", "revenue", "cost", "opportunity", "gap",
            "payback", "quick win", "strategic"
        ],
    }

    async def find_matches(
        self,
        query: str,
        agent_name: str | None = None,
        max_results: int = 3,
        min_score: float = 0.5
    ) -> list[SkillMatch]:
        """
        Find skills matching a query.

        Args:
            query: User query or task description
            agent_name: Filter to skills for this agent
            max_results: Maximum matches to return
            min_score: Minimum relevance score

        Returns:
            List of SkillMatch sorted by score descending
        """
        query_lower = query.lower()
        matches = []

        for skill_path, keywords in self.SKILL_KEYWORDS.items():
            score = 0.0
            matched = []

            for keyword in keywords:
                if keyword in query_lower:
                    weight = self.KEYWORD_WEIGHTS.get(keyword, 1.0)
                    score += weight
                    matched.append(keyword)

            if score >= min_score:
                # Check agent filter
                if agent_name:
                    try:
                        skill = await SkillLoader.load(skill_path)
                        if not skill.is_for_agent(agent_name):
                            continue
                    except FileNotFoundError:
                        continue

                matches.append(SkillMatch(
                    skill_path=skill_path,
                    score=score,
                    matched_triggers=matched
                ))

        # Sort by score descending
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:max_results]

    async def get_best_skill(
        self,
        query: str,
        agent_name: str | None = None
    ) -> Skill | None:
        """Get the single best matching skill."""
        matches = await self.find_matches(query, agent_name, max_results=1)
        if matches:
            return await SkillLoader.load(matches[0].skill_path)
        return None
```

### 2.3 Create Core Skill Files

Create the following skill files based on templates in `SKILL_INTEGRATION.md`:

- [ ] `.claude/skills/SKILL.md` - Master index
- [ ] `.claude/skills/pharma-commercial/SKILL.md` - Category index
- [ ] `.claude/skills/pharma-commercial/kpi-calculation.md`
- [ ] `.claude/skills/pharma-commercial/brand-analytics.md`
- [ ] `.claude/skills/pharma-commercial/patient-journey.md`
- [ ] `.claude/skills/causal-inference/SKILL.md` - Category index
- [ ] `.claude/skills/causal-inference/confounder-identification.md`
- [ ] `.claude/skills/causal-inference/dowhy-workflow.md`
- [ ] `.claude/skills/experiment-design/SKILL.md` - Category index
- [ ] `.claude/skills/experiment-design/validity-threats.md`
- [ ] `.claude/skills/experiment-design/power-analysis.md`
- [ ] `.claude/skills/gap-analysis/SKILL.md` - Category index
- [ ] `.claude/skills/gap-analysis/roi-estimation.md`

### 2.4 Unit Tests

**File**: `tests/unit/test_skills/test_loader.py`

```python
"""Tests for skill loading."""

import pytest
from src.skills import SkillLoader


class TestSkillLoader:
    @pytest.mark.asyncio
    async def test_load_skill(self, tmp_path, monkeypatch):
        # Create test skill
        skill_dir = tmp_path / ".claude" / "skills"
        skill_dir.mkdir(parents=True)
        skill_file = skill_dir / "test-skill.md"
        skill_file.write_text("""---
name: Test Skill
version: 1.0
description: A test skill
triggers:
  - test
  - example
agents:
  - test_agent
---

# Test Skill

## Section One

Content for section one.

## Section Two

Content for section two.
""")

        monkeypatch.setattr(SkillLoader, "SKILLS_DIR", skill_dir)

        skill = await SkillLoader.load("test-skill.md")

        assert skill.metadata.name == "Test Skill"
        assert skill.metadata.version == "1.0"
        assert "test" in skill.metadata.triggers
        assert "test_agent" in skill.metadata.agents
        assert "Content for section one" in skill.get_section("Section One")

    @pytest.mark.asyncio
    async def test_skill_caching(self, tmp_path, monkeypatch):
        skill_dir = tmp_path / ".claude" / "skills"
        skill_dir.mkdir(parents=True)
        (skill_dir / "cached.md").write_text("---\nname: Cached\n---\n# Cached")

        monkeypatch.setattr(SkillLoader, "SKILLS_DIR", skill_dir)
        SkillLoader.clear_cache()

        skill1 = await SkillLoader.load("cached.md")
        skill2 = await SkillLoader.load("cached.md")

        assert skill1 is skill2  # Same object from cache
```

### 2.5 Deliverables Checklist

- [ ] `src/skills/__init__.py`
- [ ] `src/skills/loader.py`
- [ ] `src/skills/matcher.py`
- [ ] All skill markdown files (12+ files)
- [ ] `tests/unit/test_skills/test_loader.py`
- [ ] `tests/unit/test_skills/test_matcher.py`

---

## Phase 3: Agent Integration

**Duration**: 2-3 weeks
**Dependencies**: Phase 1 and Phase 2

### 3.1 Add Skill Support to Base Agent

**File**: `src/agents/base.py` (modify existing)

```python
# Add to existing BaseAgent class

from src.skills import SkillLoader, SkillMatcher, Skill
from src.mcp import get_connector_gateway, Connector


class BaseAgent:
    """Base class for all E2I agents."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.skill_loader = SkillLoader
        self.skill_matcher = SkillMatcher()
        self.connector_gateway = get_connector_gateway()
        self._loaded_skills: list[str] = []

    # === Skill Methods ===

    async def load_skill(self, skill_path: str) -> Skill:
        """Load a skill and track it."""
        skill = await self.skill_loader.load(skill_path)
        self._loaded_skills.append(skill_path)
        return skill

    async def load_skill_section(self, skill_path: str, section: str) -> str:
        """Load only a specific section (token-efficient)."""
        return await self.skill_loader.load_section(skill_path, section)

    async def find_relevant_skills(self, query: str) -> list[Skill]:
        """Find skills relevant to a query."""
        matches = await self.skill_matcher.find_matches(
            query,
            agent_name=self.agent_name
        )
        return [
            await self.skill_loader.load(m.skill_path)
            for m in matches
        ]

    def get_skill_context(self) -> str:
        """Get loaded skill paths for context tracking."""
        return ", ".join(self._loaded_skills)

    # === MCP Connector Methods ===

    async def query_connector(
        self,
        connector: Connector,
        query: str,
        use_cache: bool = True
    ):
        """Query an MCP connector with permission checking."""
        return await self.connector_gateway.query(
            connector=connector,
            query=query,
            agent_name=self.agent_name,
            use_cache=use_cache
        )

    def can_use_connector(self, connector: Connector) -> bool:
        """Check if this agent can use a connector."""
        return self.connector_gateway.check_permission(
            self.agent_name,
            connector
        )

    def list_available_connectors(self) -> list[Connector]:
        """List connectors this agent can use."""
        return [
            c for c in Connector
            if self.can_use_connector(c)
        ]
```

### 3.2 Integrate with Causal Impact Agent

**File**: `src/agents/causal_impact/agent.py` (modify existing)

```python
# Add skill and connector usage to CausalImpactAgent

class CausalImpactAgent(BaseAgent):
    """Causal impact estimation agent with skill and connector support."""

    async def build_dag(
        self,
        treatment: str,
        outcome: str,
        brand: str | None = None
    ) -> CausalDAG:
        """Build causal DAG with skill-guided confounder selection."""

        # Load confounder identification skill
        conf_skill = await self.load_skill(
            "causal-inference/confounder-identification.md"
        )

        # Determine analysis type and get confounders
        if "hcp" in treatment.lower():
            section = "HCP Targeting → Prescription Impact"
        elif "patient" in treatment.lower():
            section = "Patient Journey → Outcome Analysis"
        else:
            section = "Trigger → Conversion Analysis"

        confounder_guidance = conf_skill.get_section(section)

        # If brand specified, add brand-specific confounders
        if brand:
            brand_skill = await self.load_skill(
                "pharma-commercial/brand-analytics.md"
            )
            brand_section = brand_skill.get_section(brand.title())
            # Extract brand-specific confounders from section

        # Build DAG with confounders
        return self._construct_dag(
            treatment=treatment,
            outcome=outcome,
            confounder_guidance=confounder_guidance
        )

    async def enrich_with_external_evidence(
        self,
        treatment: str,
        outcome: str
    ) -> dict:
        """Enrich causal analysis with external evidence from connectors."""

        evidence = {}

        # Query PubMed for supporting literature
        if self.can_use_connector(Connector.PUBMED):
            pubmed_result = await self.query_connector(
                Connector.PUBMED,
                f"Find systematic reviews or meta-analyses examining "
                f"the effect of {treatment} on {outcome}"
            )
            if pubmed_result.success:
                evidence["literature"] = pubmed_result.data

        # Query ClinicalTrials for RCT evidence
        if self.can_use_connector(Connector.CLINICALTRIALS):
            trials_result = await self.query_connector(
                Connector.CLINICALTRIALS,
                f"Find completed randomized trials studying {treatment} "
                f"with {outcome} as an endpoint"
            )
            if trials_result.success:
                evidence["trials"] = trials_result.data

        return evidence

    async def interpret_results(
        self,
        estimate: CausalEstimate,
        user_level: str = "analyst"
    ) -> str:
        """Generate interpretation using skill templates."""

        # Load interpretation skill
        workflow_skill = await self.load_skill(
            "causal-inference/dowhy-workflow.md"
        )

        # Get audience-appropriate template
        template_section = f"For {user_level.title()}s"
        template = workflow_skill.get_section(template_section)

        # Apply template to results
        return self._apply_interpretation_template(
            template=template,
            estimate=estimate
        )
```

### 3.3 Integrate with Experiment Designer Agent

**File**: `src/agents/experiment_designer/agent.py` (modify existing)

```python
class ExperimentDesignerAgent(BaseAgent):
    """Experiment design agent with skill and connector support."""

    async def audit_validity(self, design: ExperimentDesign) -> ValidityAudit:
        """Audit experiment validity using skill-defined threat taxonomy."""

        # Load validity threats skill
        validity_skill = await self.load_skill(
            "experiment-design/validity-threats.md"
        )

        threats = [
            "Selection Bias",
            "Confounding",
            "Measurement Error",
            "Contamination",
            "Temporal Effects",
            "Attrition"
        ]

        assessments = []
        for threat in threats:
            threat_guidance = validity_skill.get_section(threat)
            assessment = self._assess_threat(design, threat, threat_guidance)
            assessments.append(assessment)

        # Get scoring framework
        scoring = validity_skill.get_section("Validity Scoring Framework")
        overall_score = self._calculate_validity_score(assessments, scoring)

        return ValidityAudit(
            assessments=assessments,
            overall_score=overall_score
        )

    async def find_reference_trials(
        self,
        condition: str,
        intervention_type: str
    ) -> list[ClinicalTrial]:
        """Find reference trials via ClinicalTrials.gov connector."""

        if not self.can_use_connector(Connector.CLINICALTRIALS):
            return []

        result = await self.query_connector(
            Connector.CLINICALTRIALS,
            f"Find completed Phase 3 trials for {condition} "
            f"with {intervention_type} interventions. "
            f"Focus on trials with clear primary endpoints and "
            f"published results."
        )

        if result.success:
            return self._parse_trial_references(result.data)
        return []
```

### 3.4 Integrate with Tool Composer

**File**: `src/agents/tool_composer/agent.py` (modify existing)

```python
class ToolComposerAgent(BaseAgent):
    """Tool composition agent - the MCP orchestration hub."""

    async def plan_with_connectors(
        self,
        query: str,
        sub_questions: list[str]
    ) -> ExecutionPlan:
        """Plan tool execution including connector queries."""

        plan_steps = []

        for question in sub_questions:
            # Check if question needs external data
            connector_need = self._identify_connector_need(question)

            if connector_need:
                plan_steps.append(ConnectorStep(
                    connector=connector_need,
                    query=question,
                    priority="high" if "evidence" in question.lower() else "medium"
                ))

            # Check for internal tool needs
            tool_need = self._identify_tool_need(question)
            if tool_need:
                plan_steps.append(ToolStep(
                    tool=tool_need,
                    input_query=question
                ))

        return ExecutionPlan(steps=plan_steps)

    def _identify_connector_need(self, question: str) -> Connector | None:
        """Identify if question needs a connector."""
        question_lower = question.lower()

        connector_triggers = {
            Connector.CHEMBL: ["compound", "drug", "molecule", "bioactivity"],
            Connector.CLINICALTRIALS: ["trial", "study", "clinical", "endpoint"],
            Connector.PUBMED: ["literature", "evidence", "research", "publication"],
            Connector.ICD10: ["diagnosis", "icd", "disease code"],
            Connector.CMS_COVERAGE: ["coverage", "medicare", "medicaid", "formulary"],
        }

        for connector, triggers in connector_triggers.items():
            if any(t in question_lower for t in triggers):
                return connector

        return None
```

### 3.5 Update Configuration

**File**: `config/mcp_config.yaml`

```yaml
# MCP and Skills Configuration

mcp:
  # Use Anthropic's hosted connectors
  provider: anthropic

  # Connector enablement
  connectors:
    chembl:
      enabled: true
      cache_ttl: 3600  # 1 hour
    clinicaltrials:
      enabled: true
      cache_ttl: 3600
    pubmed:
      enabled: true
      cache_ttl: 3600
    icd10:
      enabled: true
      cache_ttl: 86400  # 24 hours (codes don't change often)
    cms_coverage:
      enabled: true
      cache_ttl: 86400

  # Rate limiting
  rate_limits:
    requests_per_minute: 60
    tokens_per_minute: 100000

skills:
  # Skills directory
  directory: .claude/skills

  # Loading behavior
  loading:
    cache_enabled: true
    max_cached_skills: 50
    preload_on_startup: false

  # Token budgets by agent tier
  token_budgets:
    deep: 3000      # Tier 5: feedback_learner, explainer
    hybrid: 2000    # Tiers 2-3: causal_impact, experiment_designer
    standard: 1000  # Tiers 0-1, 4
```

### 3.6 Deliverables Checklist

- [ ] Update `src/agents/base.py` with skill and connector methods
- [ ] Update `src/agents/causal_impact/agent.py`
- [ ] Update `src/agents/experiment_designer/agent.py`
- [ ] Update `src/agents/gap_analyzer/agent.py`
- [ ] Update `src/agents/tool_composer/agent.py`
- [ ] Update `src/agents/explainer/agent.py`
- [ ] Create `config/mcp_config.yaml`
- [ ] Integration tests for skill + connector workflows

---

## Phase 4: Self-Hosted MCP (Optional)

**Duration**: 1-2 weeks
**Dependencies**: Phase 3 complete
**Trigger**: Only if remote connectors don't meet needs

### 4.1 When to Consider Self-Hosted

Implement self-hosted MCP servers if:
- [ ] Anthropic connectors lack specific tools you need
- [ ] You need offline/air-gapped operation
- [ ] You require custom tool behavior or data sources
- [ ] Rate limits on hosted connectors are insufficient

### 4.2 ToolUniverse Integration (If Needed)

```bash
# On droplet
cd /opt
git clone https://github.com/mims-harvard/ToolUniverse
cd ToolUniverse
pip install -r requirements.txt

# Create systemd service
sudo tee /etc/systemd/system/tooluniverse.service << EOF
[Unit]
Description=ToolUniverse MCP Server
After=network.target

[Service]
User=enunez
WorkingDirectory=/opt/ToolUniverse
ExecStart=/opt/e2i_causal_analytics/.venv/bin/python -m tooluniverse.server --port 8002
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable tooluniverse
sudo systemctl start tooluniverse
```

### 4.3 Hybrid Gateway (Remote + Self-Hosted)

```python
# src/mcp/hybrid_gateway.py

class HybridMCPGateway:
    """Gateway supporting both remote Anthropic and self-hosted MCPs."""

    def __init__(self):
        self.anthropic_gateway = AnthropicConnectorGateway()
        self.self_hosted: dict[str, SelfHostedMCP] = {}

    def register_self_hosted(self, name: str, url: str):
        """Register a self-hosted MCP server."""
        self.self_hosted[name] = SelfHostedMCP(name, url)

    async def query(
        self,
        source: str,
        query: str,
        agent_name: str
    ):
        """Route query to appropriate backend."""
        if source in [c.value for c in Connector]:
            return await self.anthropic_gateway.query(
                Connector(source),
                query,
                agent_name
            )
        elif source in self.self_hosted:
            return await self.self_hosted[source].query(query)
        else:
            raise ValueError(f"Unknown MCP source: {source}")
```

---

## Testing Strategy

### Unit Tests
- Connector gateway permission checking
- Skill loading and parsing
- Skill matching accuracy
- Agent method integration

### Integration Tests
- End-to-end connector queries (with mocks)
- Skill + agent workflow tests
- Multi-skill loading scenarios

### Manual Testing
- Verify connector access on Claude subscription
- Test actual connector responses
- Validate skill content accuracy

---

## Rollout Plan

### Week 1-2
- [ ] Implement Phase 1 (Connector Gateway)
- [ ] Verify Claude API connector access
- [ ] Basic unit tests

### Week 3-4
- [ ] Implement Phase 2 (Skills Framework)
- [ ] Create initial skill files
- [ ] Skill loader tests

### Week 5-6
- [ ] Implement Phase 3 (Agent Integration)
- [ ] Update key agents (causal_impact, experiment_designer, gap_analyzer)
- [ ] Integration tests

### Week 7+
- [ ] Production deployment
- [ ] Monitor usage and performance
- [ ] Evaluate Phase 4 necessity

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Connector query success rate | >95% | Logging |
| Skill match accuracy | >80% | Manual evaluation |
| Agent context efficiency | <3000 tokens/skill | Token counting |
| Query enrichment rate | >30% queries use connectors | Analytics |

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Connector unavailability | Low | Medium | Graceful fallback, caching |
| Skill content errors | Medium | Medium | Review process, testing |
| Token budget exceeded | Medium | Low | Strict budgets, section loading |
| API rate limits | Low | Medium | Caching, request batching |

---

## References

- [Skills vs MCP Evaluation](./skills-vs-mcp-evaluation.md)
- [Domain Skills Framework](../../skills/SKILL_INTEGRATION.md)
- [Anthropic Connectors](https://claude.com/connectors)
- [MCP Documentation](https://docs.anthropic.com/mcp)
