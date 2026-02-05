"""Unit tests for JWT authentication dependency.

Tests cover:
- JWT token verification with Supabase
- Role-based access control (RBAC)
- Role hierarchy (ADMIN > OPERATOR > ANALYST > VIEWER)
- Authentication dependencies (require_auth, require_viewer, etc.)
- Testing mode bypass
- Error handling for invalid tokens
- User role extraction

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException, status

from src.api.dependencies.auth import (
    UserRole,
    get_user_role,
    has_role,
)


@pytest.mark.unit
class TestUserRole:
    """Test suite for UserRole enum and role utilities."""

    def test_user_role_enum_values(self):
        """Test UserRole enum has expected values."""
        assert UserRole.VIEWER == "viewer"
        assert UserRole.ANALYST == "analyst"
        assert UserRole.OPERATOR == "operator"
        assert UserRole.ADMIN == "admin"

    def test_role_levels_hierarchy(self):
        """Test role levels follow hierarchy."""
        from src.api.dependencies.auth import ROLE_LEVELS

        assert ROLE_LEVELS[UserRole.VIEWER] < ROLE_LEVELS[UserRole.ANALYST]
        assert ROLE_LEVELS[UserRole.ANALYST] < ROLE_LEVELS[UserRole.OPERATOR]
        assert ROLE_LEVELS[UserRole.OPERATOR] < ROLE_LEVELS[UserRole.ADMIN]

    def test_get_user_role_from_app_metadata(self):
        """Test get_user_role extracts role from app_metadata (preferred)."""
        user = {
            "id": "user-123",
            "app_metadata": {"role": "analyst"},
        }

        role = get_user_role(user)

        assert role == UserRole.ANALYST

    def test_get_user_role_from_top_level(self):
        """Test get_user_role falls back to top-level role field."""
        user = {
            "id": "user-123",
            "role": "operator",
        }

        role = get_user_role(user)

        assert role == UserRole.OPERATOR

    def test_get_user_role_prefers_app_metadata(self):
        """Test get_user_role prefers app_metadata over top-level role."""
        user = {
            "id": "user-123",
            "app_metadata": {"role": "admin"},
            "role": "viewer",  # Should be ignored
        }

        role = get_user_role(user)

        assert role == UserRole.ADMIN

    def test_get_user_role_legacy_is_admin_flag(self):
        """Test get_user_role handles legacy is_admin flag."""
        user = {
            "id": "user-123",
            "app_metadata": {"is_admin": True},
        }

        role = get_user_role(user)

        assert role == UserRole.ADMIN

    def test_get_user_role_defaults_to_viewer(self):
        """Test get_user_role defaults to viewer when no role found."""
        user = {
            "id": "user-123",
        }

        role = get_user_role(user)

        assert role == UserRole.VIEWER

    def test_get_user_role_handles_invalid_role(self):
        """Test get_user_role handles invalid role strings."""
        user = {
            "id": "user-123",
            "app_metadata": {"role": "superadmin"},  # Invalid
        }

        role = get_user_role(user)

        assert role == UserRole.VIEWER  # Defaults to viewer

    def test_get_user_role_case_insensitive(self):
        """Test get_user_role handles case variations."""
        user = {
            "id": "user-123",
            "app_metadata": {"role": "ANALYST"},
        }

        role = get_user_role(user)

        assert role == UserRole.ANALYST

    def test_has_role_same_level(self):
        """Test has_role returns True for same role level."""
        user = {
            "app_metadata": {"role": "analyst"},
        }

        assert has_role(user, UserRole.ANALYST) is True

    def test_has_role_higher_level(self):
        """Test has_role returns True for higher role level."""
        user = {
            "app_metadata": {"role": "admin"},
        }

        assert has_role(user, UserRole.ANALYST) is True
        assert has_role(user, UserRole.OPERATOR) is True
        assert has_role(user, UserRole.VIEWER) is True

    def test_has_role_lower_level(self):
        """Test has_role returns False for lower role level."""
        user = {
            "app_metadata": {"role": "viewer"},
        }

        assert has_role(user, UserRole.ANALYST) is False
        assert has_role(user, UserRole.OPERATOR) is False
        assert has_role(user, UserRole.ADMIN) is False

    def test_has_role_hierarchy(self):
        """Test complete role hierarchy."""
        # Viewer can only access viewer endpoints
        viewer = {"app_metadata": {"role": "viewer"}}
        assert has_role(viewer, UserRole.VIEWER) is True
        assert has_role(viewer, UserRole.ANALYST) is False

        # Analyst can access viewer and analyst endpoints
        analyst = {"app_metadata": {"role": "analyst"}}
        assert has_role(analyst, UserRole.VIEWER) is True
        assert has_role(analyst, UserRole.ANALYST) is True
        assert has_role(analyst, UserRole.OPERATOR) is False

        # Operator can access viewer, analyst, and operator endpoints
        operator = {"app_metadata": {"role": "operator"}}
        assert has_role(operator, UserRole.VIEWER) is True
        assert has_role(operator, UserRole.ANALYST) is True
        assert has_role(operator, UserRole.OPERATOR) is True
        assert has_role(operator, UserRole.ADMIN) is False

        # Admin can access all endpoints
        admin = {"app_metadata": {"role": "admin"}}
        assert has_role(admin, UserRole.VIEWER) is True
        assert has_role(admin, UserRole.ANALYST) is True
        assert has_role(admin, UserRole.OPERATOR) is True
        assert has_role(admin, UserRole.ADMIN) is True


@pytest.mark.unit
class TestAuthenticationFunctions:
    """Test suite for authentication dependency functions."""

    @pytest.mark.asyncio
    async def test_verify_supabase_token_success(self):
        """Test successful token verification."""
        from src.api.dependencies.auth import verify_supabase_token

        mock_user = MagicMock()
        mock_user.id = "user-123"
        mock_user.email = "test@example.com"
        mock_user.role = "authenticated"
        mock_user.aud = "authenticated"
        mock_user.created_at = None
        mock_user.app_metadata = {"role": "analyst"}
        mock_user.user_metadata = {"name": "Test User"}

        mock_response = MagicMock()
        mock_response.user = mock_user

        mock_client = MagicMock()
        mock_client.auth.get_user.return_value = mock_response

        with patch.dict(
            "os.environ",
            {
                "SUPABASE_URL": "https://test.supabase.co",
                "SUPABASE_ANON_KEY": "test-key",
            },
        ):
            with patch("supabase.create_client") as mock_create:
                mock_create.return_value = mock_client

                user_data = await verify_supabase_token("test-token")

                assert user_data is not None
                assert user_data["id"] == "user-123"
                assert user_data["email"] == "test@example.com"
                assert user_data["app_metadata"]["role"] == "analyst"
                mock_client.auth.get_user.assert_called_once_with("test-token")

    @pytest.mark.asyncio
    async def test_verify_supabase_token_invalid(self):
        """Test token verification with invalid token."""
        from src.api.dependencies.auth import verify_supabase_token

        mock_client = MagicMock()
        mock_client.auth.get_user.return_value = None

        with patch.dict(
            "os.environ",
            {
                "SUPABASE_URL": "https://test.supabase.co",
                "SUPABASE_ANON_KEY": "test-key",
            },
        ):
            with patch("supabase.create_client") as mock_create:
                mock_create.return_value = mock_client

                user_data = await verify_supabase_token("invalid-token")

                assert user_data is None

    @pytest.mark.asyncio
    async def test_verify_supabase_token_no_credentials(self):
        """Test token verification without Supabase credentials."""
        from src.api.dependencies.auth import verify_supabase_token

        with patch.dict("os.environ", {}, clear=True):
            user_data = await verify_supabase_token("test-token")

            assert user_data is None

    @pytest.mark.asyncio
    async def test_verify_supabase_token_error(self):
        """Test token verification handles errors gracefully."""
        from src.api.dependencies.auth import verify_supabase_token

        with patch.dict(
            "os.environ",
            {
                "SUPABASE_URL": "https://test.supabase.co",
                "SUPABASE_ANON_KEY": "test-key",
            },
        ):
            with patch("supabase.create_client") as mock_create:
                mock_create.side_effect = Exception("Connection failed")

                user_data = await verify_supabase_token("test-token")

                assert user_data is None

    @pytest.mark.asyncio
    async def test_get_current_user_with_valid_token(self):
        """Test get_current_user with valid token."""
        from src.api.dependencies.auth import get_current_user

        mock_request = MagicMock()
        mock_credentials = MagicMock()
        mock_credentials.credentials = "valid-token"

        user_data = {"id": "user-123", "email": "test@example.com"}

        with patch("src.api.dependencies.auth.verify_supabase_token") as mock_verify:
            mock_verify.return_value = user_data

            result = await get_current_user(mock_request, mock_credentials)

            assert result == user_data
            assert mock_request.state.user == user_data

    @pytest.mark.asyncio
    async def test_get_current_user_no_credentials(self):
        """Test get_current_user without credentials returns None."""
        from src.api.dependencies.auth import get_current_user

        mock_request = MagicMock()

        result = await get_current_user(mock_request, None)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self):
        """Test get_current_user with invalid token returns None."""
        from src.api.dependencies.auth import get_current_user

        mock_request = MagicMock()
        mock_credentials = MagicMock()
        mock_credentials.credentials = "invalid-token"

        with patch("src.api.dependencies.auth.verify_supabase_token") as mock_verify:
            mock_verify.return_value = None

            result = await get_current_user(mock_request, mock_credentials)

            assert result is None

    @pytest.mark.asyncio
    async def test_require_auth_success(self):
        """Test require_auth with valid credentials."""
        from src.api.dependencies.auth import require_auth

        mock_request = MagicMock()
        mock_credentials = MagicMock()
        mock_credentials.credentials = "valid-token"

        user_data = {"id": "user-123", "email": "test@example.com"}

        # Disable testing mode so it uses actual auth verification
        with patch("src.api.dependencies.auth.TESTING_MODE", False):
            with patch("src.api.dependencies.auth.verify_supabase_token") as mock_verify:
                mock_verify.return_value = user_data

                result = await require_auth(mock_request, mock_credentials)

                assert result == user_data
                assert mock_request.state.user == user_data

    @pytest.mark.asyncio
    async def test_require_auth_testing_mode(self):
        """Test require_auth returns test user in testing mode."""
        from src.api.dependencies.auth import require_auth

        mock_request = MagicMock()

        with patch("src.api.dependencies.auth.TESTING_MODE", True):
            result = await require_auth(mock_request, None)

            assert result is not None
            assert result["id"] == "test-user-id"
            assert result["app_metadata"]["role"] == "admin"

    @pytest.mark.asyncio
    async def test_require_auth_no_credentials(self):
        """Test require_auth raises error without credentials."""
        from src.api.dependencies.auth import require_auth

        mock_request = MagicMock()

        with patch("src.api.dependencies.auth.TESTING_MODE", False):
            with pytest.raises(HTTPException) as exc_info:
                await require_auth(mock_request, None)

            assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
            assert "Missing authorization header" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_require_auth_invalid_token(self):
        """Test require_auth raises error with invalid token."""
        from src.api.dependencies.auth import require_auth

        mock_request = MagicMock()
        mock_credentials = MagicMock()
        mock_credentials.credentials = "invalid-token"

        with patch("src.api.dependencies.auth.TESTING_MODE", False):
            with patch("src.api.dependencies.auth.verify_supabase_token") as mock_verify:
                mock_verify.return_value = None

                with pytest.raises(HTTPException) as exc_info:
                    await require_auth(mock_request, mock_credentials)

                assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
                assert "Invalid or expired token" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_require_viewer(self):
        """Test require_viewer allows viewer and higher roles."""
        from src.api.dependencies.auth import require_viewer

        viewer_user = {"app_metadata": {"role": "viewer"}}
        analyst_user = {"app_metadata": {"role": "analyst"}}

        # Viewer should pass
        result = await require_viewer(viewer_user)
        assert result == viewer_user

        # Analyst (higher) should pass
        result = await require_viewer(analyst_user)
        assert result == analyst_user

    @pytest.mark.asyncio
    async def test_require_analyst(self):
        """Test require_analyst allows analyst and higher roles."""
        from src.api.dependencies.auth import require_analyst

        analyst_user = {"app_metadata": {"role": "analyst"}}
        admin_user = {"app_metadata": {"role": "admin"}}
        viewer_user = {"app_metadata": {"role": "viewer"}}

        # Analyst should pass
        result = await require_analyst(analyst_user)
        assert result == analyst_user

        # Admin (higher) should pass
        result = await require_analyst(admin_user)
        assert result == admin_user

        # Viewer (lower) should fail
        with pytest.raises(HTTPException) as exc_info:
            await require_analyst(viewer_user)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Analyst privileges required" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_require_operator(self):
        """Test require_operator allows operator and higher roles."""
        from src.api.dependencies.auth import require_operator

        operator_user = {"app_metadata": {"role": "operator"}}
        admin_user = {"app_metadata": {"role": "admin"}}
        analyst_user = {"app_metadata": {"role": "analyst"}}

        # Operator should pass
        result = await require_operator(operator_user)
        assert result == operator_user

        # Admin (higher) should pass
        result = await require_operator(admin_user)
        assert result == admin_user

        # Analyst (lower) should fail
        with pytest.raises(HTTPException) as exc_info:
            await require_operator(analyst_user)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Operator privileges required" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_require_admin(self):
        """Test require_admin only allows admin role."""
        from src.api.dependencies.auth import require_admin

        admin_user = {"app_metadata": {"role": "admin"}}
        operator_user = {"app_metadata": {"role": "operator"}}

        # Admin should pass
        result = await require_admin(admin_user)
        assert result == admin_user

        # Operator (lower) should fail
        with pytest.raises(HTTPException) as exc_info:
            await require_admin(operator_user)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Admin privileges required" in str(exc_info.value.detail)


@pytest.mark.unit
class TestAuthUtilityFunctions:
    """Test suite for auth utility functions."""

    def test_is_auth_enabled_with_credentials(self):
        """Test is_auth_enabled returns True with credentials."""
        from src.api.dependencies.auth import is_auth_enabled

        with patch("src.api.dependencies.auth.TESTING_MODE", False):
            with patch("src.api.dependencies.auth.SUPABASE_URL", "https://test.supabase.co"):
                with patch("src.api.dependencies.auth.SUPABASE_ANON_KEY", "test-key"):
                    assert is_auth_enabled() is True

    def test_is_auth_enabled_without_credentials(self):
        """Test is_auth_enabled returns False without credentials."""
        from src.api.dependencies.auth import is_auth_enabled

        with patch("src.api.dependencies.auth.TESTING_MODE", False):
            with patch("src.api.dependencies.auth.SUPABASE_URL", ""):
                with patch("src.api.dependencies.auth.SUPABASE_ANON_KEY", ""):
                    assert is_auth_enabled() is False

    def test_is_auth_enabled_testing_mode(self):
        """Test is_auth_enabled returns False in testing mode."""
        from src.api.dependencies.auth import is_auth_enabled

        with patch("src.api.dependencies.auth.TESTING_MODE", True):
            with patch("src.api.dependencies.auth.SUPABASE_URL", "https://test.supabase.co"):
                with patch("src.api.dependencies.auth.SUPABASE_ANON_KEY", "test-key"):
                    assert is_auth_enabled() is False

    def test_is_testing_mode(self):
        """Test is_testing_mode detection."""
        from src.api.dependencies.auth import is_testing_mode

        with patch("src.api.dependencies.auth.TESTING_MODE", True):
            assert is_testing_mode() is True

        with patch("src.api.dependencies.auth.TESTING_MODE", False):
            assert is_testing_mode() is False
