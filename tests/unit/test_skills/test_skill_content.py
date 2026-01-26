"""Unit tests for skill content validation.

Tests that all domain skill files:
1. Load correctly
2. Have properly formatted metadata
3. Contain required sections
4. Have valid triggers and agent references
"""

import pytest

from src.skills.loader import SkillLoader


class TestBrandAnalyticsSkill:
    """Tests for the brand-analytics.md skill file."""

    @pytest.fixture
    def loader(self):
        """Create a SkillLoader."""
        return SkillLoader()

    def test_load_brand_analytics(self, loader):
        """Test loading the brand analytics skill."""
        skill = loader.load("pharma-commercial/brand-analytics.md")

        assert skill.metadata.name == "Brand-Specific Analytics"
        assert skill.metadata.version == "1.0"

    def test_brand_analytics_triggers(self, loader):
        """Test that brand analytics has correct triggers."""
        skill = loader.load("pharma-commercial/brand-analytics.md")

        expected_triggers = [
            "Kisqali analysis",
            "Fabhalta analysis",
            "Remibrutinib analysis",
            "brand context",
        ]
        for trigger in expected_triggers:
            assert trigger in skill.metadata.triggers

    def test_brand_analytics_agents(self, loader):
        """Test that brand analytics references correct agents."""
        skill = loader.load("pharma-commercial/brand-analytics.md")

        expected_agents = ["causal_impact", "gap_analyzer", "experiment_designer"]
        for agent in expected_agents:
            assert agent in skill.metadata.agents

    def test_brand_analytics_categories(self, loader):
        """Test that brand analytics has correct categories."""
        skill = loader.load("pharma-commercial/brand-analytics.md")

        expected_categories = ["oncology", "rare-disease", "immunology"]
        for category in expected_categories:
            assert category in skill.metadata.categories

    def test_kisqali_section(self, loader):
        """Test that Kisqali section exists with key content."""
        skill = loader.load("pharma-commercial/brand-analytics.md")

        # Check section exists
        sections = skill.get_sections_matching(r".*Kisqali.*")
        assert len(sections) >= 1

        # Check content contains key info
        assert "HR+/HER2-" in skill.content or "breast cancer" in skill.content.lower()

    def test_fabhalta_section(self, loader):
        """Test that Fabhalta section exists with key content."""
        skill = loader.load("pharma-commercial/brand-analytics.md")

        sections = skill.get_sections_matching(r".*Fabhalta.*")
        assert len(sections) >= 1

        assert "PNH" in skill.content or "hemoglobinuria" in skill.content.lower()

    def test_remibrutinib_section(self, loader):
        """Test that Remibrutinib section exists with key content."""
        skill = loader.load("pharma-commercial/brand-analytics.md")

        sections = skill.get_sections_matching(r".*Remibrutinib.*")
        assert len(sections) >= 1

        assert "CSU" in skill.content or "urticaria" in skill.content.lower()

    def test_brand_specific_confounders(self, loader):
        """Test that brand-specific confounders are documented."""
        skill = loader.load("pharma-commercial/brand-analytics.md")

        # Each brand should have confounders section
        assert "confounder" in skill.content.lower()


class TestDoWhyWorkflowSkill:
    """Tests for the dowhy-workflow.md skill file."""

    @pytest.fixture
    def loader(self):
        """Create a SkillLoader."""
        return SkillLoader()

    def test_load_dowhy_workflow(self, loader):
        """Test loading the DoWhy workflow skill."""
        skill = loader.load("causal-inference/dowhy-workflow.md")

        assert skill.metadata.name == "DoWhy Causal Estimation Workflow"
        assert skill.metadata.version == "1.0"

    def test_dowhy_triggers(self, loader):
        """Test that DoWhy skill has correct triggers."""
        skill = loader.load("causal-inference/dowhy-workflow.md")

        expected_triggers = [
            "causal estimation",
            "DoWhy analysis",
            "ATE calculation",
            "CATE analysis",
        ]
        for trigger in expected_triggers:
            assert trigger in skill.metadata.triggers

    def test_dowhy_agents(self, loader):
        """Test that DoWhy skill references correct agents."""
        skill = loader.load("causal-inference/dowhy-workflow.md")

        assert "causal_impact" in skill.metadata.agents

    def test_dag_construction_phase(self, loader):
        """Test that DAG construction phase exists."""
        skill = loader.load("causal-inference/dowhy-workflow.md")

        assert "DAG Construction" in skill.content

    def test_estimation_phase(self, loader):
        """Test that estimation phase exists with energy score."""
        skill = loader.load("causal-inference/dowhy-workflow.md")

        assert "Energy Score" in skill.content
        assert "0.35" in skill.content  # Treatment balance weight
        assert "0.45" in skill.content  # Outcome fit weight

    def test_refutation_testing_phase(self, loader):
        """Test that refutation testing phase exists."""
        skill = loader.load("causal-inference/dowhy-workflow.md")

        assert "Refutation Testing" in skill.content
        assert "placebo" in skill.content.lower()
        assert "random_common_cause" in skill.content

    def test_sensitivity_analysis_phase(self, loader):
        """Test that sensitivity analysis phase exists."""
        skill = loader.load("causal-inference/dowhy-workflow.md")

        assert "Sensitivity Analysis" in skill.content
        assert "E-value" in skill.content

    def test_interpretation_phase(self, loader):
        """Test that interpretation phase exists with audience sections."""
        skill = loader.load("causal-inference/dowhy-workflow.md")

        assert "Interpretation" in skill.content
        assert "Executive" in skill.content
        assert "Analyst" in skill.content
        assert "Data Scientist" in skill.content

    def test_code_templates(self, loader):
        """Test that code templates are included."""
        skill = loader.load("causal-inference/dowhy-workflow.md")

        assert "CausalModel" in skill.content
        assert "CausalForestDML" in skill.content


class TestValidityThreatsSkill:
    """Tests for the validity-threats.md skill file."""

    @pytest.fixture
    def loader(self):
        """Create a SkillLoader."""
        return SkillLoader()

    def test_load_validity_threats(self, loader):
        """Test loading the validity threats skill."""
        skill = loader.load("experiment-design/validity-threats.md")

        assert skill.metadata.name == "Experiment Validity Threat Assessment"
        assert skill.metadata.version == "1.0"

    def test_validity_threats_triggers(self, loader):
        """Test that validity threats has correct triggers."""
        skill = loader.load("experiment-design/validity-threats.md")

        expected_triggers = [
            "validity threats",
            "experiment validation",
            "internal validity",
            "external validity",
        ]
        for trigger in expected_triggers:
            assert trigger in skill.metadata.triggers

    def test_validity_threats_agents(self, loader):
        """Test that validity threats references correct agents."""
        skill = loader.load("experiment-design/validity-threats.md")

        assert "experiment_designer" in skill.metadata.agents

    def test_six_threat_taxonomy(self, loader):
        """Test that all 6 threats are documented."""
        skill = loader.load("experiment-design/validity-threats.md")

        expected_threats = [
            "Selection Bias",
            "Confounding",
            "Measurement Error",
            "Contamination",
            "Temporal Effects",
            "Attrition",
        ]
        for threat in expected_threats:
            assert threat in skill.content

    def test_pharma_manifestations(self, loader):
        """Test that pharma-specific manifestations are included."""
        skill = loader.load("experiment-design/validity-threats.md")

        # Should mention pharma-specific examples
        assert "HCP" in skill.content
        assert "patient" in skill.content.lower()

    def test_mitigations_included(self, loader):
        """Test that mitigations are included for threats."""
        skill = loader.load("experiment-design/validity-threats.md")

        assert "Mitigation" in skill.content
        assert "Randomization" in skill.content

    def test_validity_scoring_framework(self, loader):
        """Test that validity scoring framework exists."""
        skill = loader.load("experiment-design/validity-threats.md")

        assert "Validity Score" in skill.content
        assert "Likelihood" in skill.content
        assert "Severity" in skill.content


class TestPowerAnalysisSkill:
    """Tests for the power-analysis.md skill file."""

    @pytest.fixture
    def loader(self):
        """Create a SkillLoader."""
        return SkillLoader()

    def test_load_power_analysis(self, loader):
        """Test loading the power analysis skill."""
        skill = loader.load("experiment-design/power-analysis.md")

        assert skill.metadata.name == "Power Analysis and Sample Size Calculation"
        assert skill.metadata.version == "1.0"

    def test_power_analysis_triggers(self, loader):
        """Test that power analysis has correct triggers."""
        skill = loader.load("experiment-design/power-analysis.md")

        expected_triggers = [
            "sample size",
            "power analysis",
            "statistical power",
            "minimum detectable effect",
        ]
        for trigger in expected_triggers:
            assert trigger in skill.metadata.triggers

    def test_power_analysis_agents(self, loader):
        """Test that power analysis references correct agents."""
        skill = loader.load("experiment-design/power-analysis.md")

        assert "experiment_designer" in skill.metadata.agents

    def test_sample_size_formulas(self, loader):
        """Test that sample size formulas are included."""
        skill = loader.load("experiment-design/power-analysis.md")

        assert "sample_size" in skill.content
        assert "t-test" in skill.content.lower() or "ttest" in skill.content.lower()

    def test_cluster_randomization(self, loader):
        """Test that cluster randomization is covered."""
        skill = loader.load("experiment-design/power-analysis.md")

        assert "cluster" in skill.content.lower()
        assert "ICC" in skill.content

    def test_pharma_benchmarks(self, loader):
        """Test that pharma-specific benchmarks are included."""
        skill = loader.load("experiment-design/power-analysis.md")

        # Should have pharma-specific metrics
        assert "TRx" in skill.content or "NRx" in skill.content

    def test_mde_calculation(self, loader):
        """Test that MDE calculation is covered."""
        skill = loader.load("experiment-design/power-analysis.md")

        assert "MDE" in skill.content or "minimum detectable effect" in skill.content.lower()


class TestROIEstimationSkill:
    """Tests for the roi-estimation.md skill file."""

    @pytest.fixture
    def loader(self):
        """Create a SkillLoader."""
        return SkillLoader()

    def test_load_roi_estimation(self, loader):
        """Test loading the ROI estimation skill."""
        skill = loader.load("gap-analysis/roi-estimation.md")

        assert skill.metadata.name == "ROI Estimation Procedures"
        assert skill.metadata.version == "1.0"

    def test_roi_estimation_triggers(self, loader):
        """Test that ROI estimation has correct triggers."""
        skill = loader.load("gap-analysis/roi-estimation.md")

        expected_triggers = [
            "ROI calculation",
            "revenue impact",
            "cost to close",
            "opportunity sizing",
        ]
        for trigger in expected_triggers:
            assert trigger in skill.metadata.triggers

    def test_roi_estimation_agents(self, loader):
        """Test that ROI estimation references correct agents."""
        skill = loader.load("gap-analysis/roi-estimation.md")

        assert "gap_analyzer" in skill.metadata.agents

    def test_revenue_multipliers(self, loader):
        """Test that revenue multipliers are documented."""
        skill = loader.load("gap-analysis/roi-estimation.md")

        assert "TRx" in skill.content
        assert "NRx" in skill.content
        assert "$500" in skill.content  # TRx multiplier

    def test_cost_formulas(self, loader):
        """Test that cost formulas are included."""
        skill = loader.load("gap-analysis/roi-estimation.md")

        assert "Cost to Close" in skill.content or "Cost-to-Close" in skill.content

    def test_roi_interpretation(self, loader):
        """Test that ROI interpretation thresholds exist."""
        skill = loader.load("gap-analysis/roi-estimation.md")

        assert "3.0" in skill.content  # Target ROI threshold
        assert "Excellent" in skill.content or "Strong" in skill.content

    def test_opportunity_categorization(self, loader):
        """Test that opportunity categories are defined."""
        skill = loader.load("gap-analysis/roi-estimation.md")

        assert "Quick Win" in skill.content
        assert "Strategic Bet" in skill.content

    def test_payback_period(self, loader):
        """Test that payback period calculation exists."""
        skill = loader.load("gap-analysis/roi-estimation.md")

        assert "Payback" in skill.content
        assert "months" in skill.content.lower()

    def test_brand_specific_multipliers(self, loader):
        """Test that brand-specific multipliers exist."""
        skill = loader.load("gap-analysis/roi-estimation.md")

        # Should have brand-specific sections
        assert "Kisqali" in skill.content
        assert "Fabhalta" in skill.content
        assert "Remibrutinib" in skill.content


class TestCategoryIndexFiles:
    """Tests for category SKILL.md index files."""

    @pytest.fixture
    def loader(self):
        """Create a SkillLoader."""
        return SkillLoader()

    def test_experiment_design_category_index(self, loader):
        """Test experiment-design category index."""
        skills = loader.list_skills("experiment-design")

        assert len(skills) >= 2
        assert any("validity-threats.md" in s for s in skills)
        assert any("power-analysis.md" in s for s in skills)

    def test_gap_analysis_category_index(self, loader):
        """Test gap-analysis category index."""
        skills = loader.list_skills("gap-analysis")

        assert len(skills) >= 1
        assert any("roi-estimation.md" in s for s in skills)


class TestSkillCrossReferences:
    """Tests for cross-references between skills."""

    @pytest.fixture
    def loader(self):
        """Create a SkillLoader."""
        return SkillLoader()

    def test_all_skills_load(self, loader):
        """Test that all skill files can be loaded."""
        skill_paths = [
            "pharma-commercial/kpi-calculation.md",
            "pharma-commercial/brand-analytics.md",
            "causal-inference/confounder-identification.md",
            "causal-inference/dowhy-workflow.md",
            "experiment-design/validity-threats.md",
            "experiment-design/power-analysis.md",
            "gap-analysis/roi-estimation.md",
        ]

        for path in skill_paths:
            skill = loader.load(path)
            assert skill is not None
            assert skill.metadata.name != "Unknown"

    def test_all_skills_have_triggers(self, loader):
        """Test that all skills have at least one trigger."""
        skill_paths = [
            "pharma-commercial/kpi-calculation.md",
            "pharma-commercial/brand-analytics.md",
            "causal-inference/confounder-identification.md",
            "causal-inference/dowhy-workflow.md",
            "experiment-design/validity-threats.md",
            "experiment-design/power-analysis.md",
            "gap-analysis/roi-estimation.md",
        ]

        for path in skill_paths:
            skill = loader.load(path)
            assert len(skill.metadata.triggers) > 0, f"{path} has no triggers"

    def test_all_skills_have_agents(self, loader):
        """Test that all skills reference at least one agent."""
        skill_paths = [
            "pharma-commercial/kpi-calculation.md",
            "pharma-commercial/brand-analytics.md",
            "causal-inference/confounder-identification.md",
            "causal-inference/dowhy-workflow.md",
            "experiment-design/validity-threats.md",
            "experiment-design/power-analysis.md",
            "gap-analysis/roi-estimation.md",
        ]

        for path in skill_paths:
            skill = loader.load(path)
            assert len(skill.metadata.agents) > 0, f"{path} has no agents"
