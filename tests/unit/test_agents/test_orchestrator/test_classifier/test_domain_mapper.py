"""Tests for Stage 2: DomainMapper."""

from src.agents.orchestrator.classifier.domain_mapper import DomainMapper
from src.agents.orchestrator.classifier.feature_extractor import FeatureExtractor
from src.agents.orchestrator.classifier.schemas import Domain


class TestDomainMapper:
    def setup_method(self):
        self.extractor = FeatureExtractor()
        self.mapper = DomainMapper()

    def _map(self, query: str):
        return self.mapper.map_domains(self.extractor.extract(query))

    def test_causal_query_detects_causal_domain(self):
        mapping = self._map("What caused the decline — was it due to fewer rep visits?")
        domains = [dm.domain for dm in mapping.domains_detected]
        assert Domain.CAUSAL_ANALYSIS in domains

    def test_keyword_free_query_detects_no_domains(self):
        """Regression: base weights were added unconditionally, so MONITORING
        (base 0.4) and EXPLANATION (base 0.3) cleared the 0.3 threshold with
        ZERO evidence — every query classified multi-domain. Base must only
        count when the domain has real evidence."""
        mapping = self._map("Good morning team")
        assert mapping.domain_count == 0
        assert mapping.primary_domain is None
        assert not mapping.is_multi_domain

    def test_evidence_free_domains_not_auto_detected(self):
        """A pure-causal query must not drag MONITORING/EXPLANATION along."""
        mapping = self._map("The campaign resulted in higher NRx — what drove it? due to reps?")
        domains = [dm.domain for dm in mapping.domains_detected]
        assert Domain.MONITORING not in domains

    def test_every_detected_domain_has_evidence(self):
        mapping = self._map(
            "Compare segment performance and predict which HCP group will grow next quarter"
        )
        assert mapping.domain_count >= 1
        for dm in mapping.domains_detected:
            assert dm.evidence, f"{dm.domain} detected without evidence"

    def test_multi_domain_ordering_by_confidence(self):
        mapping = self._map(
            "Design an experiment to test the hypothesis and predict the expected lift"
        )
        confidences = [dm.confidence for dm in mapping.domains_detected]
        assert confidences == sorted(confidences, reverse=True)
        if mapping.domain_count > 1:
            assert mapping.is_multi_domain
            assert mapping.primary_domain == mapping.domains_detected[0].domain

    def test_monitoring_still_reachable_with_evidence(self):
        mapping = self._map("Is there any drift or anomaly in the Kisqali model data quality?")
        domains = [dm.domain for dm in mapping.domains_detected]
        assert Domain.MONITORING in domains
