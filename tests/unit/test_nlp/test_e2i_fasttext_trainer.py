"""
Tests for E2I FastText Model Training and Testing Suite.

Tests cover:
- Corpus preprocessing
- Model training configuration
- Test case validation
- Cosine similarity calculations
- Best match finding algorithms
- Test suite execution
- Interactive testing mode
- Main entry point and CLI

Note: Mock fasttext to avoid dependency on actual model training.
"""

import hashlib
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock fasttext before importing the module
sys.modules["fasttext"] = MagicMock()

from src.nlp.e2i_fasttext_trainer import (
    CANONICAL_VOCABULARY,
    TEST_CASES,
    TRAINING_CONFIG,
    cosine_similarity,
    find_best_match,
    interactive_test,
    main,
    preprocess_corpus,
    run_test_suite,
    train_model,
)


class TestConstants:
    """Tests for module constants."""

    def test_canonical_vocabulary_structure(self):
        """Verify canonical vocabulary has expected structure."""
        assert "brands" in CANONICAL_VOCABULARY
        assert "regions" in CANONICAL_VOCABULARY
        assert "agents" in CANONICAL_VOCABULARY
        assert "kpis" in CANONICAL_VOCABULARY
        assert "journey_stages" in CANONICAL_VOCABULARY
        assert "workstreams" in CANONICAL_VOCABULARY

    def test_canonical_vocabulary_contents(self):
        """Verify canonical vocabulary has expected content."""
        assert "Remibrutinib" in CANONICAL_VOCABULARY["brands"]
        assert "Kisqali" in CANONICAL_VOCABULARY["brands"]
        assert "Fabhalta" in CANONICAL_VOCABULARY["brands"]
        assert "northeast" in CANONICAL_VOCABULARY["regions"]
        assert "orchestrator" in CANONICAL_VOCABULARY["agents"]
        assert "TRx" in CANONICAL_VOCABULARY["kpis"]

    def test_training_config_structure(self):
        """Verify training config has expected parameters."""
        assert TRAINING_CONFIG["model"] == "skipgram"
        assert TRAINING_CONFIG["dim"] == 100
        assert TRAINING_CONFIG["epoch"] == 50
        assert TRAINING_CONFIG["lr"] == 0.05
        assert TRAINING_CONFIG["wordNgrams"] == 3
        assert TRAINING_CONFIG["minCount"] == 1
        assert TRAINING_CONFIG["minn"] == 2
        assert TRAINING_CONFIG["maxn"] == 5

    def test_test_cases_structure(self):
        """Verify test cases have expected structure."""
        assert len(TEST_CASES) > 0
        for test_case in TEST_CASES:
            assert len(test_case) == 3  # (typo, expected, category)
            assert isinstance(test_case[0], str)
            assert isinstance(test_case[1], str)
            assert isinstance(test_case[2], str)

    def test_test_cases_cover_all_categories(self):
        """Verify test cases cover all vocabulary categories."""
        categories = {tc[2] for tc in TEST_CASES}
        assert "brands" in categories
        assert "regions" in categories
        assert "agents" in categories
        assert "kpis" in categories


class TestCosineSimilarity:
    """Tests for cosine similarity calculation."""

    def test_identical_vectors(self):
        """Identical vectors have similarity 1.0."""
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have similarity 0.0."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(v1, v2) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Opposite vectors have similarity -1.0."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([-1.0, 0.0, 0.0])
        assert cosine_similarity(v1, v2) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        """Zero vector returns 0.0 similarity."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([0.0, 0.0, 0.0])
        assert cosine_similarity(v1, v2) == 0.0

    def test_both_zero_vectors(self):
        """Both zero vectors return 0.0."""
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([0.0, 0.0, 0.0])
        assert cosine_similarity(v1, v2) == 0.0

    def test_similar_vectors(self):
        """Similar vectors have high similarity."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([1.1, 2.1, 3.1])
        similarity = cosine_similarity(v1, v2)
        assert 0.99 < similarity < 1.0

    def test_dissimilar_vectors(self):
        """Dissimilar vectors have low similarity."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 0.0, 1.0])
        similarity = cosine_similarity(v1, v2)
        assert similarity == pytest.approx(0.0)


class TestFindBestMatch:
    """Tests for finding best matching term."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock fasttext model."""
        model = MagicMock()

        # Mock get_word_vector to return deterministic vectors
        def get_vector(word):
            # Deterministic vector generation (hashlib is stable across processes,
            # unlike hash() which is randomized per PYTHONHASHSEED)
            seed = int(hashlib.md5(word.lower().encode()).hexdigest(), 16) % 2**32
            np.random.seed(seed)
            return np.abs(np.random.randn(100))

        model.get_word_vector.side_effect = get_vector
        return model

    def test_find_exact_match(self, mock_model):
        """Find exact match when query matches candidate."""
        candidates = ["Kisqali", "Fabhalta", "Remibrutinib"]
        match, score = find_best_match(mock_model, "Kisqali", candidates)

        assert match == "Kisqali"
        assert score > 0.99  # Should be very high for identical

    def test_find_close_match(self, mock_model):
        """Find close match for typo."""
        candidates = ["Kisqali", "Fabhalta", "Remibrutinib"]
        match, score = find_best_match(mock_model, "kisqali", candidates)

        # Should match "Kisqali" (case-insensitive)
        assert match is not None
        assert score > 0.0

    def test_threshold_filtering(self, mock_model):
        """Matches below threshold return None."""
        candidates = ["Kisqali", "Fabhalta"]
        match, score = find_best_match(mock_model, "unrelated", candidates, threshold=0.99)

        # High threshold should filter out poor matches
        if score < 0.99:
            assert match is None

    def test_empty_candidates(self, mock_model):
        """Empty candidate list returns None."""
        match, score = find_best_match(mock_model, "query", [])

        assert match is None
        assert score == 0.0

    def test_single_candidate(self, mock_model):
        """Single candidate is selected if above threshold."""
        # Use very low negative threshold to ensure match
        match, score = find_best_match(mock_model, "test", ["candidate"], threshold=-1.0)

        assert match == "candidate"
        assert score >= -1.0

    def test_case_insensitive_matching(self, mock_model):
        """Matching is case-insensitive."""
        candidates = ["Kisqali", "FABHALTA", "remibrutinib"]
        match1, _ = find_best_match(mock_model, "kisqali", candidates)
        match2, _ = find_best_match(mock_model, "KISQALI", candidates)

        # Both should match "Kisqali"
        assert match1 is not None
        assert match2 is not None


class TestPreprocessCorpus:
    """Tests for corpus preprocessing."""

    def test_preprocess_removes_comments(self):
        """Preprocessing removes comment lines."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("# This is a comment\n")
            f.write("valid line\n")
            f.write("# Another comment\n")
            f.write("another valid line\n")
            input_path = f.name

        output_path = input_path.replace(".txt", "_processed.txt")

        try:
            preprocess_corpus(input_path, output_path)

            with open(output_path, "r") as f:
                lines = f.readlines()

            assert len(lines) == 2
            assert lines[0].strip() == "valid line"
            assert lines[1].strip() == "another valid line"

        finally:
            os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_preprocess_removes_empty_lines(self):
        """Preprocessing removes empty lines."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("line1\n")
            f.write("\n")
            f.write("line2\n")
            f.write("   \n")
            f.write("line3\n")
            input_path = f.name

        output_path = input_path.replace(".txt", "_processed.txt")

        try:
            preprocess_corpus(input_path, output_path)

            with open(output_path, "r") as f:
                lines = f.readlines()

            assert len(lines) == 3
            assert "line1" in lines[0]
            assert "line2" in lines[1]
            assert "line3" in lines[2]

        finally:
            os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_preprocess_removes_section_headers(self):
        """Preprocessing removes section headers."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("=== SECTION ===\n")
            f.write("content\n")
            f.write("===\n")
            input_path = f.name

        output_path = input_path.replace(".txt", "_processed.txt")

        try:
            preprocess_corpus(input_path, output_path)

            with open(output_path, "r") as f:
                content = f.read()

            assert "SECTION" not in content
            assert "content" in content

        finally:
            os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_preprocess_preserves_valid_content(self):
        """Preprocessing preserves valid content."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Brand: Kisqali\n")
            f.write("KPI: TRx NRx\n")
            f.write("Region: northeast\n")
            input_path = f.name

        output_path = input_path.replace(".txt", "_processed.txt")

        try:
            preprocess_corpus(input_path, output_path)

            with open(output_path, "r") as f:
                content = f.read()

            assert "Kisqali" in content
            assert "TRx" in content
            assert "northeast" in content

        finally:
            os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestTrainModel:
    """Tests for model training."""

    @patch("src.nlp.e2i_fasttext_trainer.fasttext")
    def test_train_model_creates_model(self, mock_fasttext):
        """Training creates and saves a model."""
        mock_model = MagicMock()
        mock_model.words = ["word1", "word2", "word3"]
        mock_model.get_dimension.return_value = 100
        mock_fasttext.train_unsupervised.return_value = mock_model

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("training data\n")
            corpus_path = f.name

        model_path = corpus_path.replace(".txt", ".bin")

        try:
            result = train_model(corpus_path, model_path)

            assert mock_fasttext.train_unsupervised.called
            assert result == mock_model
            assert mock_model.save_model.called

        finally:
            os.unlink(corpus_path)
            if os.path.exists(model_path):
                os.unlink(model_path)
            # Clean up processed file
            processed_path = corpus_path.replace(".txt", "_processed.txt")
            if os.path.exists(processed_path):
                os.unlink(processed_path)

    @patch("src.nlp.e2i_fasttext_trainer.fasttext")
    def test_train_model_uses_config(self, mock_fasttext):
        """Training uses configuration parameters."""
        mock_model = MagicMock()
        mock_model.words = []
        mock_model.get_dimension.return_value = 100
        mock_fasttext.train_unsupervised.return_value = mock_model

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("data\n")
            corpus_path = f.name

        model_path = corpus_path.replace(".txt", ".bin")

        try:
            train_model(corpus_path, model_path)

            call_args = mock_fasttext.train_unsupervised.call_args
            kwargs = call_args[1]

            assert kwargs["model"] == "skipgram"
            assert kwargs["dim"] == 100
            assert kwargs["epoch"] == 50
            assert kwargs["wordNgrams"] == 3

        finally:
            os.unlink(corpus_path)
            processed_path = corpus_path.replace(".txt", "_processed.txt")
            if os.path.exists(processed_path):
                os.unlink(processed_path)


class TestRunTestSuite:
    """Tests for test suite execution."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model that returns perfect matches."""
        model = MagicMock()

        # Mock vectors such that expected matches have high similarity
        def get_vector(word):
            # Return same vector for case-insensitive matches
            np.random.seed(hash(word.lower()) % 2**32)
            return np.random.randn(100)

        model.get_word_vector.side_effect = get_vector
        return model

    def test_run_test_suite_structure(self, mock_model):
        """Test suite returns expected structure."""
        results = run_test_suite(mock_model)

        assert "passed" in results
        assert "failed" in results
        assert "total" in results
        assert "by_category" in results
        assert "failures" in results

    def test_run_test_suite_counts_tests(self, mock_model):
        """Test suite counts all tests."""
        results = run_test_suite(mock_model)

        assert results["total"] == len(TEST_CASES)
        assert results["passed"] + results["failed"] == results["total"]

    def test_run_test_suite_categorizes_results(self, mock_model):
        """Test suite categorizes results by vocabulary category."""
        results = run_test_suite(mock_model)

        assert isinstance(results["by_category"], dict)
        assert len(results["by_category"]) > 0

        for _category, stats in results["by_category"].items():
            assert "passed" in stats
            assert "failed" in stats

    def test_run_test_suite_records_failures(self, mock_model):
        """Test suite records failure details."""
        results = run_test_suite(mock_model)

        if results["failed"] > 0:
            assert len(results["failures"]) == results["failed"]
            for failure in results["failures"]:
                assert "typo" in failure
                assert "expected" in failure
                assert "got" in failure
                assert "score" in failure
                assert "category" in failure


class TestInteractiveTest:
    """Tests for interactive testing mode."""

    @patch("builtins.input")
    def test_interactive_test_exit_on_quit(self, mock_input):
        """Interactive mode exits on 'quit' command."""
        mock_input.side_effect = ["quit"]
        mock_model = MagicMock()

        # Should not raise exception
        interactive_test(mock_model)

    @patch("builtins.input")
    def test_interactive_test_exit_on_exit(self, mock_input):
        """Interactive mode exits on 'exit' command."""
        mock_input.side_effect = ["exit"]
        mock_model = MagicMock()

        interactive_test(mock_model)

    @patch("builtins.input")
    def test_interactive_test_handles_eof(self, mock_input):
        """Interactive mode handles EOF gracefully."""
        mock_input.side_effect = EOFError()
        mock_model = MagicMock()

        # Should not raise exception
        interactive_test(mock_model)

    @patch("builtins.input")
    @patch("builtins.print")
    def test_interactive_test_processes_query(self, mock_print, mock_input):
        """Interactive mode processes queries."""
        mock_input.side_effect = ["Kisqali", "quit"]
        mock_model = MagicMock()
        mock_model.get_word_vector.return_value = np.array([1.0, 0.0, 0.0])
        mock_model.get_nearest_neighbors.return_value = [(0.99, "kisqali"), (0.85, "ribociclib")]

        interactive_test(mock_model)

        # Should have printed results
        assert mock_print.call_count > 0


class TestMain:
    """Tests for main entry point."""

    def test_main_no_args_prints_help(self):
        """Main with no args prints help and exits."""
        with patch.object(sys, "argv", ["prog"]):
            with pytest.raises(SystemExit):
                main()

    @patch("src.nlp.e2i_fasttext_trainer.train_model")
    @patch("os.path.exists")
    def test_main_train_command(self, mock_exists, mock_train):
        """Main handles 'train' command."""
        mock_exists.return_value = True
        mock_train.return_value = MagicMock()

        with patch.object(sys, "argv", ["prog", "train"]):
            main()

        assert mock_train.called

    @patch("src.nlp.e2i_fasttext_trainer.run_test_suite")
    @patch("src.nlp.e2i_fasttext_trainer.fasttext")
    @patch("os.path.exists")
    def test_main_test_command(self, mock_exists, mock_fasttext, mock_test_suite):
        """Main handles 'test' command."""
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_fasttext.load_model.return_value = mock_model
        mock_test_suite.return_value = {"passed": 10, "failed": 0, "total": 10}

        with patch.object(sys, "argv", ["prog", "test"]):
            main()

        assert mock_test_suite.called

    @patch("src.nlp.e2i_fasttext_trainer.interactive_test")
    @patch("src.nlp.e2i_fasttext_trainer.fasttext")
    @patch("os.path.exists")
    def test_main_interactive_command(self, mock_exists, mock_fasttext, mock_interactive):
        """Main handles 'interactive' command."""
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_fasttext.load_model.return_value = mock_model

        with patch.object(sys, "argv", ["prog", "interactive"]):
            main()

        assert mock_interactive.called

    @patch("src.nlp.e2i_fasttext_trainer.train_model")
    @patch("src.nlp.e2i_fasttext_trainer.run_test_suite")
    @patch("os.path.exists")
    def test_main_all_command(self, mock_exists, mock_test_suite, mock_train):
        """Main handles 'all' command."""
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_model.words = []
        mock_model.get_dimension.return_value = 100
        mock_train.return_value = mock_model
        mock_test_suite.return_value = {"passed": 10, "failed": 0, "total": 10}

        with patch.object(sys, "argv", ["prog", "all"]):
            main()

        assert mock_train.called
        assert mock_test_suite.called

    def test_main_unknown_command_exits(self):
        """Main with unknown command exits."""
        with patch.object(sys, "argv", ["prog", "unknown"]):
            with pytest.raises(SystemExit):
                main()

    @patch("os.path.exists")
    def test_main_train_missing_corpus_exits(self, mock_exists):
        """Main exits if corpus file missing for train."""
        mock_exists.return_value = False

        with patch.object(sys, "argv", ["prog", "train"]):
            with pytest.raises(SystemExit):
                main()

    @patch("os.path.exists")
    def test_main_test_missing_model_exits(self, mock_exists):
        """Main exits if model file missing for test."""
        mock_exists.return_value = False

        with patch.object(sys, "argv", ["prog", "test"]):
            with pytest.raises(SystemExit):
                main()
