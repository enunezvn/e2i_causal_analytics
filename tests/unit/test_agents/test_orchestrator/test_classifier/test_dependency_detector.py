"""Tests for Stage 3: DependencyDetector."""

from src.agents.orchestrator.classifier.dependency_detector import DependencyDetector
from src.agents.orchestrator.classifier.domain_mapper import DomainMapper
from src.agents.orchestrator.classifier.feature_extractor import FeatureExtractor
from src.agents.orchestrator.classifier.schemas import DependencyType


class _ExplodingLLMClient:
    """Any attribute access fails the test — the LLM must never be touched."""

    def __getattr__(self, name):  # pragma: no cover - failure path
        raise AssertionError("LLM client must not be invoked by the dependency detector")


class TestDependencyDetector:
    def setup_method(self):
        self.extractor = FeatureExtractor()
        self.mapper = DomainMapper()

    async def _detect(self, query: str, detector: DependencyDetector | None = None, **kwargs):
        detector = detector or DependencyDetector(llm_client=None)
        features = self.extractor.extract(query)
        mapping = self.mapper.map_domains(features)
        return await detector.detect(query, features, mapping, **kwargs)

    async def test_single_question_no_dependencies(self):
        analysis = await self._detect("What is TRx for Kisqali?")
        assert len(analysis.sub_questions) == 1
        assert analysis.dependencies == []
        assert not analysis.has_dependencies
        assert analysis.is_parallelizable
        assert analysis.dependency_depth == 0
        assert analysis.used_llm is False

    async def test_reference_chain_detected(self):
        analysis = await self._detect("What drove the drop in TRx? and how should we act on that")
        assert len(analysis.sub_questions) == 2
        assert analysis.has_dependencies
        assert analysis.dependencies[0].dependency_type == DependencyType.REFERENCE_CHAIN
        assert not analysis.is_parallelizable
        assert analysis.dependency_depth == 1

    async def test_dependency_ids_link_adjacent_subquestions(self):
        analysis = await self._detect(
            "Which segments respond best? and how should we target those segments"
        )
        assert analysis.has_dependencies
        dep = analysis.dependencies[0]
        assert dep.from_id == "Q1"
        assert dep.to_id == "Q2"

    async def test_used_llm_false_without_client(self):
        analysis = await self._detect(
            "What drove TRx? and which regions lag? and what would a fix cost?",
            use_llm=True,
        )
        assert analysis.used_llm is False

    async def test_llm_client_never_invoked(self):
        """Regression: the scaffold made a BLOCKING sync Anthropic call from
        async code and discarded the response. Until the async implementation
        lands, _detect_with_llm must be a no-op that reports used_llm=False
        even when a client is present and escalation is requested."""
        detector = DependencyDetector(llm_client=_ExplodingLLMClient())
        analysis = await self._detect(
            # 3+ sub-questions with no rule-based deps -> _needs_llm_analysis True
            "What is NRx for Fabhalta? and which regions lag? and who are the top HCPs?",
            detector=detector,
            use_llm=True,
        )
        assert analysis.used_llm is False
