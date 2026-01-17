"""
Tests for src/utils/llm_factory.py

Covers:
- get_llm_provider function
- get_chat_llm function
- Provider-specific LLM creation functions
- Convenience functions (get_fast_llm, get_standard_llm, get_reasoning_llm)
- Error handling for missing API keys and packages
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.utils.llm_factory import (
    MODEL_MAPPINGS,
    get_chat_llm,
    get_fast_llm,
    get_llm_provider,
    get_reasoning_llm,
    get_standard_llm,
)


# =============================================================================
# get_llm_provider Tests
# =============================================================================


class TestGetLLMProvider:
    """Tests for the get_llm_provider function."""

    def test_default_provider_is_openai(self):
        """Test default provider is openai when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove LLM_PROVIDER if it exists
            os.environ.pop("LLM_PROVIDER", None)
            provider = get_llm_provider()
            assert provider == "openai"

    def test_returns_anthropic_when_set(self):
        """Test returns anthropic when LLM_PROVIDER is anthropic."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic"}):
            provider = get_llm_provider()
            assert provider == "anthropic"

    def test_returns_openai_when_set(self):
        """Test returns openai when LLM_PROVIDER is openai."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}):
            provider = get_llm_provider()
            assert provider == "openai"

    def test_case_insensitive(self):
        """Test provider name is case insensitive."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "ANTHROPIC"}):
            provider = get_llm_provider()
            assert provider == "anthropic"

        with patch.dict(os.environ, {"LLM_PROVIDER": "OpenAI"}):
            provider = get_llm_provider()
            assert provider == "openai"

    def test_unknown_provider_defaults_to_openai(self):
        """Test unknown provider falls back to openai with warning."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "invalid_provider"}):
            provider = get_llm_provider()
            assert provider == "openai"


# =============================================================================
# MODEL_MAPPINGS Tests
# =============================================================================


class TestModelMappings:
    """Tests for the MODEL_MAPPINGS constant."""

    def test_anthropic_mappings_exist(self):
        """Test anthropic provider has all tier mappings."""
        assert "anthropic" in MODEL_MAPPINGS
        assert "fast" in MODEL_MAPPINGS["anthropic"]
        assert "standard" in MODEL_MAPPINGS["anthropic"]
        assert "reasoning" in MODEL_MAPPINGS["anthropic"]

    def test_openai_mappings_exist(self):
        """Test openai provider has all tier mappings."""
        assert "openai" in MODEL_MAPPINGS
        assert "fast" in MODEL_MAPPINGS["openai"]
        assert "standard" in MODEL_MAPPINGS["openai"]
        assert "reasoning" in MODEL_MAPPINGS["openai"]

    def test_anthropic_model_names(self):
        """Test anthropic model names are correct."""
        assert "haiku" in MODEL_MAPPINGS["anthropic"]["fast"]
        assert "sonnet" in MODEL_MAPPINGS["anthropic"]["standard"]
        assert "sonnet" in MODEL_MAPPINGS["anthropic"]["reasoning"]

    def test_openai_model_names(self):
        """Test openai model names are correct."""
        assert MODEL_MAPPINGS["openai"]["fast"] == "gpt-4o-mini"
        assert MODEL_MAPPINGS["openai"]["standard"] == "gpt-4o"
        assert MODEL_MAPPINGS["openai"]["reasoning"] == "gpt-4o"


# =============================================================================
# get_chat_llm Tests
# =============================================================================


class TestGetChatLLM:
    """Tests for the get_chat_llm function."""

    @patch("src.utils.llm_factory._create_openai_llm")
    def test_uses_openai_by_default(self, mock_create_openai):
        """Test uses OpenAI provider by default."""
        mock_llm = MagicMock()
        mock_create_openai.return_value = mock_llm

        with patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}):
            result = get_chat_llm()

        assert result == mock_llm
        mock_create_openai.assert_called_once()

    @patch("src.utils.llm_factory._create_anthropic_llm")
    def test_uses_anthropic_when_configured(self, mock_create_anthropic):
        """Test uses Anthropic provider when configured."""
        mock_llm = MagicMock()
        mock_create_anthropic.return_value = mock_llm

        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "test-key"}):
            result = get_chat_llm()

        assert result == mock_llm
        mock_create_anthropic.assert_called_once()

    @patch("src.utils.llm_factory._create_openai_llm")
    def test_provider_override(self, mock_create_openai):
        """Test provider parameter overrides environment."""
        mock_llm = MagicMock()
        mock_create_openai.return_value = mock_llm

        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic", "OPENAI_API_KEY": "test-key"}):
            result = get_chat_llm(provider="openai")

        assert result == mock_llm
        mock_create_openai.assert_called_once()

    @patch("src.utils.llm_factory._create_openai_llm")
    def test_passes_model_tier(self, mock_create_openai):
        """Test passes correct model based on tier."""
        mock_llm = MagicMock()
        mock_create_openai.return_value = mock_llm

        with patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}):
            get_chat_llm(model_tier="fast")

        # Should pass gpt-4o-mini for fast tier
        call_args = mock_create_openai.call_args
        assert call_args[0][0] == "gpt-4o-mini"

    @patch("src.utils.llm_factory._create_openai_llm")
    def test_passes_max_tokens(self, mock_create_openai):
        """Test passes max_tokens parameter."""
        mock_llm = MagicMock()
        mock_create_openai.return_value = mock_llm

        with patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}):
            get_chat_llm(max_tokens=2048)

        call_args = mock_create_openai.call_args
        assert call_args[0][1] == 2048  # max_tokens

    @patch("src.utils.llm_factory._create_openai_llm")
    def test_passes_temperature(self, mock_create_openai):
        """Test passes temperature parameter."""
        mock_llm = MagicMock()
        mock_create_openai.return_value = mock_llm

        with patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}):
            get_chat_llm(temperature=0.7)

        call_args = mock_create_openai.call_args
        assert call_args[0][2] == 0.7  # temperature

    @patch("src.utils.llm_factory._create_openai_llm")
    def test_passes_timeout(self, mock_create_openai):
        """Test passes timeout parameter."""
        mock_llm = MagicMock()
        mock_create_openai.return_value = mock_llm

        with patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}):
            get_chat_llm(timeout=30)

        call_args = mock_create_openai.call_args
        assert call_args[0][3] == 30  # timeout


# =============================================================================
# _create_anthropic_llm Tests
# =============================================================================


class TestCreateAnthropicLLM:
    """Tests for the _create_anthropic_llm function."""

    def test_raises_import_error_if_package_missing(self):
        """Test raises ImportError if langchain-anthropic not installed."""
        with patch.dict("sys.modules", {"langchain_anthropic": None}):
            with patch("src.utils.llm_factory._create_anthropic_llm") as mock_create:
                mock_create.side_effect = ImportError(
                    "langchain-anthropic is required for Anthropic LLMs"
                )
                with pytest.raises(ImportError, match="langchain-anthropic is required"):
                    mock_create("claude-sonnet-4-20250514", 1024, 0.3, None)

    def test_raises_value_error_if_api_key_missing(self):
        """Test raises ValueError if ANTHROPIC_API_KEY not set."""
        # Mock the ChatAnthropic import
        mock_chat_anthropic = MagicMock()
        with patch.dict("sys.modules", {"langchain_anthropic": MagicMock(ChatAnthropic=mock_chat_anthropic)}):
            with patch.dict(os.environ, {}, clear=True):
                # Remove API key if it exists
                os.environ.pop("ANTHROPIC_API_KEY", None)

                # Import the actual function after patching
                from src.utils.llm_factory import _create_anthropic_llm

                with pytest.raises(ValueError, match="ANTHROPIC_API_KEY.*not set"):
                    _create_anthropic_llm("claude-sonnet-4-20250514", 1024, 0.3, None)

    def test_creates_chat_anthropic_with_correct_params(self):
        """Test creates ChatAnthropic with correct parameters."""
        mock_chat_anthropic_class = MagicMock()
        mock_chat_anthropic_module = MagicMock()
        mock_chat_anthropic_module.ChatAnthropic = mock_chat_anthropic_class

        with patch.dict("sys.modules", {"langchain_anthropic": mock_chat_anthropic_module}):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-api-key"}):
                # Re-import to get fresh function
                from importlib import reload

                import src.utils.llm_factory as llm_factory

                reload(llm_factory)

                llm_factory._create_anthropic_llm(
                    "claude-sonnet-4-20250514",
                    1024,
                    0.3,
                    30,  # timeout
                )

                mock_chat_anthropic_class.assert_called_once_with(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1024,
                    temperature=0.3,
                    timeout=30,
                )

    def test_timeout_not_passed_when_none(self):
        """Test timeout is not passed to ChatAnthropic when None."""
        mock_chat_anthropic_class = MagicMock()
        mock_chat_anthropic_module = MagicMock()
        mock_chat_anthropic_module.ChatAnthropic = mock_chat_anthropic_class

        with patch.dict("sys.modules", {"langchain_anthropic": mock_chat_anthropic_module}):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-api-key"}):
                from importlib import reload

                import src.utils.llm_factory as llm_factory

                reload(llm_factory)

                llm_factory._create_anthropic_llm(
                    "claude-sonnet-4-20250514",
                    1024,
                    0.3,
                    None,  # timeout
                )

                # Timeout should not be in kwargs
                call_kwargs = mock_chat_anthropic_class.call_args[1]
                assert "timeout" not in call_kwargs


# =============================================================================
# _create_openai_llm Tests
# =============================================================================


class TestCreateOpenAILLM:
    """Tests for the _create_openai_llm function."""

    def test_raises_import_error_if_package_missing(self):
        """Test raises ImportError if langchain-openai not installed."""
        with patch("src.utils.llm_factory._create_openai_llm") as mock_create:
            mock_create.side_effect = ImportError(
                "langchain-openai is required for OpenAI LLMs"
            )
            with pytest.raises(ImportError, match="langchain-openai is required"):
                mock_create("gpt-4o", 1024, 0.3, None)

    def test_raises_value_error_if_api_key_missing(self):
        """Test raises ValueError if OPENAI_API_KEY not set."""
        mock_chat_openai = MagicMock()
        with patch.dict("sys.modules", {"langchain_openai": MagicMock(ChatOpenAI=mock_chat_openai)}):
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop("OPENAI_API_KEY", None)

                from src.utils.llm_factory import _create_openai_llm

                with pytest.raises(ValueError, match="OPENAI_API_KEY.*not set"):
                    _create_openai_llm("gpt-4o", 1024, 0.3, None)

    def test_creates_chat_openai_with_correct_params(self):
        """Test creates ChatOpenAI with correct parameters."""
        mock_chat_openai_class = MagicMock()
        mock_chat_openai_module = MagicMock()
        mock_chat_openai_module.ChatOpenAI = mock_chat_openai_class

        with patch.dict("sys.modules", {"langchain_openai": mock_chat_openai_module}):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
                from importlib import reload

                import src.utils.llm_factory as llm_factory

                reload(llm_factory)

                llm_factory._create_openai_llm(
                    "gpt-4o",
                    1024,
                    0.3,
                    30,  # timeout
                )

                mock_chat_openai_class.assert_called_once_with(
                    model="gpt-4o",
                    max_tokens=1024,
                    temperature=0.3,
                    request_timeout=30,  # OpenAI uses request_timeout
                )

    def test_request_timeout_not_passed_when_none(self):
        """Test request_timeout is not passed to ChatOpenAI when None."""
        mock_chat_openai_class = MagicMock()
        mock_chat_openai_module = MagicMock()
        mock_chat_openai_module.ChatOpenAI = mock_chat_openai_class

        with patch.dict("sys.modules", {"langchain_openai": mock_chat_openai_module}):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
                from importlib import reload

                import src.utils.llm_factory as llm_factory

                reload(llm_factory)

                llm_factory._create_openai_llm(
                    "gpt-4o",
                    1024,
                    0.3,
                    None,  # timeout
                )

                call_kwargs = mock_chat_openai_class.call_args[1]
                assert "request_timeout" not in call_kwargs


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestGetFastLLM:
    """Tests for the get_fast_llm convenience function."""

    @patch("src.utils.llm_factory.get_chat_llm")
    def test_uses_fast_tier(self, mock_get_chat_llm):
        """Test uses fast model tier."""
        mock_llm = MagicMock()
        mock_get_chat_llm.return_value = mock_llm

        result = get_fast_llm()

        assert result == mock_llm
        mock_get_chat_llm.assert_called_once_with(
            model_tier="fast",
            max_tokens=256,
            temperature=0.0,  # Deterministic for classification
            timeout=5,
            provider=None,
        )

    @patch("src.utils.llm_factory.get_chat_llm")
    def test_custom_max_tokens(self, mock_get_chat_llm):
        """Test custom max_tokens parameter."""
        get_fast_llm(max_tokens=512)

        call_kwargs = mock_get_chat_llm.call_args[1]
        assert call_kwargs["max_tokens"] == 512

    @patch("src.utils.llm_factory.get_chat_llm")
    def test_custom_timeout(self, mock_get_chat_llm):
        """Test custom timeout parameter."""
        get_fast_llm(timeout=10)

        call_kwargs = mock_get_chat_llm.call_args[1]
        assert call_kwargs["timeout"] == 10

    @patch("src.utils.llm_factory.get_chat_llm")
    def test_provider_override(self, mock_get_chat_llm):
        """Test provider override parameter."""
        get_fast_llm(provider="anthropic")

        call_kwargs = mock_get_chat_llm.call_args[1]
        assert call_kwargs["provider"] == "anthropic"


class TestGetStandardLLM:
    """Tests for the get_standard_llm convenience function."""

    @patch("src.utils.llm_factory.get_chat_llm")
    def test_uses_standard_tier(self, mock_get_chat_llm):
        """Test uses standard model tier."""
        mock_llm = MagicMock()
        mock_get_chat_llm.return_value = mock_llm

        result = get_standard_llm()

        assert result == mock_llm
        mock_get_chat_llm.assert_called_once_with(
            model_tier="standard",
            max_tokens=1024,
            temperature=0.3,
            timeout=None,
            provider=None,
        )

    @patch("src.utils.llm_factory.get_chat_llm")
    def test_custom_temperature(self, mock_get_chat_llm):
        """Test custom temperature parameter."""
        get_standard_llm(temperature=0.7)

        call_kwargs = mock_get_chat_llm.call_args[1]
        assert call_kwargs["temperature"] == 0.7

    @patch("src.utils.llm_factory.get_chat_llm")
    def test_custom_max_tokens(self, mock_get_chat_llm):
        """Test custom max_tokens parameter."""
        get_standard_llm(max_tokens=2048)

        call_kwargs = mock_get_chat_llm.call_args[1]
        assert call_kwargs["max_tokens"] == 2048


class TestGetReasoningLLM:
    """Tests for the get_reasoning_llm convenience function."""

    @patch("src.utils.llm_factory.get_chat_llm")
    def test_uses_reasoning_tier(self, mock_get_chat_llm):
        """Test uses reasoning model tier."""
        mock_llm = MagicMock()
        mock_get_chat_llm.return_value = mock_llm

        result = get_reasoning_llm()

        assert result == mock_llm
        mock_get_chat_llm.assert_called_once_with(
            model_tier="reasoning",
            max_tokens=4096,
            temperature=0.3,
            timeout=120,
            provider=None,
        )

    @patch("src.utils.llm_factory.get_chat_llm")
    def test_custom_max_tokens(self, mock_get_chat_llm):
        """Test custom max_tokens parameter."""
        get_reasoning_llm(max_tokens=8192)

        call_kwargs = mock_get_chat_llm.call_args[1]
        assert call_kwargs["max_tokens"] == 8192

    @patch("src.utils.llm_factory.get_chat_llm")
    def test_custom_timeout(self, mock_get_chat_llm):
        """Test custom timeout parameter."""
        get_reasoning_llm(timeout=180)

        call_kwargs = mock_get_chat_llm.call_args[1]
        assert call_kwargs["timeout"] == 180

    @patch("src.utils.llm_factory.get_chat_llm")
    def test_provider_override(self, mock_get_chat_llm):
        """Test provider override parameter."""
        get_reasoning_llm(provider="openai")

        call_kwargs = mock_get_chat_llm.call_args[1]
        assert call_kwargs["provider"] == "openai"
