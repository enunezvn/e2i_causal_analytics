"""Tests for Stage 1: FeatureExtractor."""

from src.agents.orchestrator.classifier.feature_extractor import FeatureExtractor


class TestStructuralFeatures:
    def setup_method(self):
        self.extractor = FeatureExtractor()

    def test_simple_question(self):
        features = self.extractor.extract("What is TRx for Kisqali?")
        assert features.structural.question_count == 1
        assert features.structural.word_count == 5
        assert not features.structural.has_conditional
        assert not features.structural.has_comparison

    def test_conditional_detection(self):
        features = self.extractor.extract(
            "If conversion rate in the west is below 15%, which patient segments should we prioritize?"
        )
        assert features.structural.has_conditional

    def test_comparison_detection(self):
        features = self.extractor.extract("Compare TRx for Kisqali vs its competitors")
        assert features.structural.has_comparison

    def test_sequence_detection(self):
        features = self.extractor.extract("First measure the effect, then reallocate the budget")
        assert features.structural.has_sequence

    def test_compound_question_count(self):
        features = self.extractor.extract(
            "Which regions underperform, and what would be the ROI of shifting capacity?"
        )
        assert features.structural.question_count >= 2

    def test_empty_query(self):
        features = self.extractor.extract("")
        assert features.structural.word_count == 0
        assert features.raw_query == ""


class TestTemporalFeatures:
    def setup_method(self):
        self.extractor = FeatureExtractor()

    def test_multigroup_pattern_returns_full_match(self):
        """Regression: '(last|this|next) (week|month|quarter|year)' has two
        capture groups; findall returned tuples and crashed pydantic
        validation. finditer + group(0) must yield the full phrase."""
        features = self.extractor.extract("Forecast TRx volume for the next quarter")
        assert "next quarter" in features.temporal.time_references
        assert all(isinstance(ref, str) for ref in features.temporal.time_references)

    def test_quarter_and_year_references(self):
        features = self.extractor.extract("Why did TRx drop in Q1 2026?")
        refs = [r.lower() for r in features.temporal.time_references]
        assert "q1" in refs
        assert "2026" in refs

    def test_future_and_past_markers(self):
        past = self.extractor.extract("The campaign resulted in higher NRx")
        assert past.temporal.has_past
        future = self.extractor.extract("Predict what will happen next year")
        assert future.temporal.has_future


class TestEntityFeatures:
    def setup_method(self):
        self.extractor = FeatureExtractor()

    def test_territory_full_mention(self):
        """Regression: 'territor(y|ies)' has a capture group; findall returned
        the fragment 'y'/'ies' as the mention instead of the full word."""
        features = self.extractor.extract("Show performance by territory")
        assert "region" in features.entities.entity_types
        mentions_lower = [m.lower() for m in features.entities.entity_mentions]
        assert "territory" in mentions_lower
        assert "y" not in mentions_lower

    def test_brand_and_hcp_entities(self):
        features = self.extractor.extract("Which HCPs prescribe Kisqali in the Northeast?")
        assert "HCP" in features.entities.entity_types
        assert "drug" in features.entities.entity_types
        assert "region" in features.entities.entity_types


class TestIntentSignals:
    def setup_method(self):
        self.extractor = FeatureExtractor()

    def test_causal_keywords(self):
        features = self.extractor.extract("What was the impact and effect of rep visits?")
        assert "impact" in features.intent_signals.causal_keywords
        assert "effect" in features.intent_signals.causal_keywords

    def test_cohort_keywords(self):
        features = self.extractor.extract("Build a cohort of patients with CSU")
        assert "cohort" in features.intent_signals.cohort_keywords
        assert "patients with" in features.intent_signals.cohort_keywords

    def test_keyword_free_query_has_no_signals(self):
        features = self.extractor.extract("Good morning team")
        signals = features.intent_signals
        assert not signals.causal_keywords
        assert not signals.monitoring_keywords
        assert not signals.explanation_keywords
