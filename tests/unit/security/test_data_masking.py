"""Tests for PII data masking utilities.

This module tests the data masking functions that protect
patient and HCP identifiers in API responses.
"""

from unittest.mock import MagicMock

import pytest

from src.api.utils.data_masking import (
    DEFAULT_PII_FIELDS,
    PIIMaskingMiddleware,
    create_masked_model_response,
    mask_identifier,
    mask_pii,
    mask_response,
    mask_response_dict,
)


class TestMaskIdentifier:
    """Tests for the mask_identifier function."""

    def test_mask_patient_id_standard_format(self):
        """Test masking standard patient ID format."""
        result = mask_identifier("PAT-2024-001234")
        assert result.startswith("PAT-")
        assert result.endswith("1234")
        assert "****" in result
        assert "2024" not in result

    def test_mask_hcp_id_standard_format(self):
        """Test masking standard HCP ID format."""
        result = mask_identifier("HCP-NE-5678")
        assert result.startswith("HCP-")
        assert result.endswith("5678")
        assert "*" in result
        assert "NE" not in result

    def test_mask_preserves_prefix(self):
        """Test that identifier prefix is preserved for sufficiently long IDs."""
        # Long IDs preserve the prefix
        assert mask_identifier("PAT-2024-001234").startswith("PAT-")
        assert mask_identifier("HCP-NORTHEAST-5678").startswith("HCP-")
        assert mask_identifier("ID-123456789").startswith("ID-")
        # Short IDs may be more aggressively masked for security

    def test_mask_preserves_last_four_chars(self):
        """Test that last 4 characters are preserved."""
        result = mask_identifier("PAT-2024-001234")
        assert result.endswith("1234")

        result = mask_identifier("HCP-NORTHEAST-9999")
        assert result.endswith("9999")

    def test_mask_short_string(self):
        """Test masking very short strings."""
        result = mask_identifier("PAT-1")
        # Short strings are aggressively masked for security
        # The result should contain asterisks
        assert "*" in result
        # Some original characters should remain
        assert len(result) == len("PAT-1")

    def test_mask_single_char(self):
        """Test masking single character returns unchanged."""
        assert mask_identifier("X") == "X"

    def test_mask_empty_string(self):
        """Test empty string returns unchanged."""
        assert mask_identifier("") == ""

    def test_mask_none_returns_none(self):
        """Test None input returns None."""
        assert mask_identifier(None) is None

    def test_mask_without_prefix_pattern(self):
        """Test masking string without hyphen prefix pattern."""
        result = mask_identifier("12345678901234")
        # Should mask middle portion
        assert result.startswith("1234")
        assert result.endswith("1234")
        assert "*" in result

    def test_mask_with_multiple_segments(self):
        """Test masking ID with multiple hyphen-separated segments."""
        result = mask_identifier("PAT-US-EAST-2024-001234")
        assert result.startswith("PAT-")
        assert result.endswith("1234")
        # Middle segments should be masked
        assert "US" not in result or "*" in result


class TestMaskPii:
    """Tests for the mask_pii function."""

    def test_masks_patient_id_field(self):
        """Test that patient_id field is masked."""
        result = mask_pii("PAT-2024-001234", "patient_id")
        assert "2024" not in result
        assert result.endswith("1234")

    def test_masks_hcp_id_field(self):
        """Test that hcp_id field is masked."""
        result = mask_pii("HCP-NE-5678", "hcp_id")
        assert "NE" not in result
        assert result.endswith("5678")

    def test_does_not_mask_non_pii_field(self):
        """Test that non-PII fields are not masked."""
        result = mask_pii("sensitive-data-here", "model_version")
        assert result == "sensitive-data-here"

    def test_masks_with_custom_pii_fields(self):
        """Test masking with custom PII field set."""
        custom_fields = {"ssn", "dob"}
        result = mask_pii("123-45-6789", "ssn", pii_fields=custom_fields)
        # SSN is in custom fields, should be masked
        assert "*" in result

    def test_does_not_mask_with_custom_fields_if_not_included(self):
        """Test that default PII fields are not masked with custom set."""
        custom_fields = {"ssn"}
        result = mask_pii("PAT-2024-001234", "patient_id", pii_fields=custom_fields)
        # patient_id is NOT in custom fields
        assert result == "PAT-2024-001234"

    def test_handles_none_value(self):
        """Test handling None values."""
        result = mask_pii(None, "patient_id")
        assert result is None

    def test_masks_list_of_ids(self):
        """Test masking a list of IDs."""
        ids = ["PAT-001-1111", "PAT-002-2222", "PAT-003-3333"]
        result = mask_pii(ids, "patient_id")
        assert len(result) == 3
        for masked_id in result:
            assert "*" in masked_id


class TestMaskResponseDict:
    """Tests for the mask_response_dict function."""

    def test_masks_simple_dict(self):
        """Test masking a simple dictionary."""
        data = {
            "patient_id": "PAT-2024-001234",
            "hcp_id": "HCP-NE-5678",
            "name": "Test Result",
        }
        result = mask_response_dict(data)

        assert result["name"] == "Test Result"
        assert "2024" not in result["patient_id"]
        assert result["patient_id"].endswith("1234")
        assert result["hcp_id"].endswith("5678")

    def test_masks_nested_dict(self):
        """Test masking nested dictionaries."""
        data = {
            "response": {
                "patient_id": "PAT-2024-001234",
                "details": {
                    "hcp_id": "HCP-NE-5678",
                },
            },
        }
        result = mask_response_dict(data)

        assert "2024" not in result["response"]["patient_id"]
        assert "NE" not in result["response"]["details"]["hcp_id"]

    def test_masks_list_of_dicts(self):
        """Test masking a list of dictionaries."""
        data = {
            "results": [
                {"patient_id": "PAT-001-1111"},
                {"patient_id": "PAT-002-2222"},
            ]
        }
        result = mask_response_dict(data)

        for item in result["results"]:
            assert "*" in item["patient_id"]

    def test_preserves_non_pii_fields(self):
        """Test that non-PII fields are preserved."""
        data = {
            "patient_id": "PAT-2024-001234",
            "prediction_score": 0.85,
            "model_name": "propensity_v2",
            "timestamp": "2024-01-15T10:30:00Z",
        }
        result = mask_response_dict(data)

        assert result["prediction_score"] == 0.85
        assert result["model_name"] == "propensity_v2"
        assert result["timestamp"] == "2024-01-15T10:30:00Z"

    def test_non_recursive_mode(self):
        """Test non-recursive masking."""
        data = {
            "patient_id": "PAT-2024-001234",
            "nested": {
                "patient_id": "PAT-NESTED-9999",
            },
        }
        result = mask_response_dict(data, recursive=False)

        # Top-level should be masked
        assert "*" in result["patient_id"]
        # Nested should NOT be masked in non-recursive mode
        assert result["nested"]["patient_id"] == "PAT-NESTED-9999"

    def test_handles_empty_dict(self):
        """Test handling empty dictionary."""
        result = mask_response_dict({})
        assert result == {}

    def test_handles_non_dict_input(self):
        """Test handling non-dict input."""
        result = mask_response_dict("not a dict")
        assert result == "not a dict"


class TestCreateMaskedModelResponse:
    """Tests for the create_masked_model_response function."""

    def test_masks_pydantic_v2_model(self):
        """Test masking a Pydantic v2 model with model_dump()."""
        mock_model = MagicMock()
        mock_model.model_dump.return_value = {
            "patient_id": "PAT-2024-001234",
            "result": "success",
        }

        result = create_masked_model_response(mock_model)

        assert "*" in result["patient_id"]
        assert result["result"] == "success"

    def test_masks_pydantic_v1_model(self):
        """Test masking a Pydantic v1 model with dict()."""
        mock_model = MagicMock(spec=["dict"])
        mock_model.dict.return_value = {
            "hcp_id": "HCP-NE-5678",
            "status": "active",
        }
        del mock_model.model_dump  # Ensure it uses dict()

        result = create_masked_model_response(mock_model)

        assert "*" in result["hcp_id"]
        assert result["status"] == "active"

    def test_raises_for_non_model(self):
        """Test that non-Pydantic objects raise TypeError."""
        with pytest.raises(TypeError):
            create_masked_model_response({"plain": "dict"})


class TestPIIMaskingMiddleware:
    """Tests for the PIIMaskingMiddleware class."""

    def test_middleware_masks_dict(self):
        """Test middleware masks dictionary data."""
        masker = PIIMaskingMiddleware()
        data = {"patient_id": "PAT-2024-001234"}

        result = masker.mask(data)

        assert "*" in result["patient_id"]

    def test_middleware_masks_list(self):
        """Test middleware masks list data."""
        masker = PIIMaskingMiddleware()
        data = [
            {"patient_id": "PAT-001-1111"},
            {"patient_id": "PAT-002-2222"},
        ]

        result = masker.mask(data)

        for item in result:
            assert "*" in item["patient_id"]

    def test_middleware_masks_model(self):
        """Test middleware masks Pydantic model."""
        masker = PIIMaskingMiddleware()
        mock_model = MagicMock()
        mock_model.model_dump.return_value = {"patient_id": "PAT-2024-001234"}

        result = masker.mask(mock_model)

        assert "*" in result["patient_id"]

    def test_middleware_disabled(self):
        """Test middleware does not mask when disabled."""
        masker = PIIMaskingMiddleware(enabled=False)
        data = {"patient_id": "PAT-2024-001234"}

        result = masker.mask(data)

        assert result["patient_id"] == "PAT-2024-001234"

    def test_middleware_custom_pii_fields(self):
        """Test middleware with custom PII fields."""
        masker = PIIMaskingMiddleware(pii_fields={"ssn"})
        data = {
            "patient_id": "PAT-2024-001234",  # Not in custom fields
            "ssn": "123-45-6789",  # In custom fields
        }

        result = masker.mask(data)

        # patient_id should NOT be masked (not in custom fields)
        assert result["patient_id"] == "PAT-2024-001234"
        # ssn SHOULD be masked
        assert "*" in result["ssn"]


class TestMaskResponse:
    """Tests for the convenience mask_response function."""

    def test_masks_dict(self):
        """Test convenience function masks dict."""
        data = {"patient_id": "PAT-2024-001234"}
        result = mask_response(data)
        assert "*" in result["patient_id"]

    def test_masks_list(self):
        """Test convenience function masks list."""
        data = [{"hcp_id": "HCP-NE-5678"}]
        result = mask_response(data)
        assert "*" in result[0]["hcp_id"]


class TestDefaultPiiFields:
    """Tests for default PII field configuration."""

    def test_default_includes_patient_id(self):
        """Test that patient_id is in default PII fields."""
        assert "patient_id" in DEFAULT_PII_FIELDS

    def test_default_includes_hcp_id(self):
        """Test that hcp_id is in default PII fields."""
        assert "hcp_id" in DEFAULT_PII_FIELDS

    def test_default_is_frozen(self):
        """Test that default PII fields cannot be modified."""
        assert isinstance(DEFAULT_PII_FIELDS, frozenset)


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_mask_unicode_string(self):
        """Test masking strings with unicode characters."""
        result = mask_identifier("PAT-日本語-1234")
        assert result.startswith("PAT-")
        assert result.endswith("1234")

    def test_mask_very_long_id(self):
        """Test masking very long identifiers."""
        long_id = "PAT-" + "X" * 100 + "-1234"
        result = mask_identifier(long_id)
        assert result.startswith("PAT-")
        assert result.endswith("1234")
        assert len(result) == len(long_id)

    def test_mask_integer_not_masked(self):
        """Test that integer values are not masked."""
        result = mask_pii(12345, "patient_id")
        assert result == 12345

    def test_deeply_nested_structure(self):
        """Test masking deeply nested structures."""
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "patient_id": "PAT-DEEP-9999",
                    }
                }
            }
        }
        result = mask_response_dict(data)
        assert "*" in result["level1"]["level2"]["level3"]["patient_id"]

    def test_mixed_list_items(self):
        """Test list with mixed types."""
        data = {
            "items": [
                {"patient_id": "PAT-001-1111"},
                "plain string",
                123,
                None,
                {"other_field": "value"},
            ]
        }
        result = mask_response_dict(data)

        # Dict with patient_id should be masked
        assert "*" in result["items"][0]["patient_id"]
        # Other items should be unchanged
        assert result["items"][1] == "plain string"
        assert result["items"][2] == 123
        assert result["items"][3] is None
        assert result["items"][4] == {"other_field": "value"}
