"""
Unit tests for RBAC (Role-Based Access Control) in auth.py.

Tests cover:
- UserRole enum values
- get_user_role() function with various JWT claim formats
- has_role() hierarchical logic
- require_* dependencies (viewer, analyst, operator, admin)

Author: E2I Causal Analytics Team
"""

import pytest

from src.api.dependencies.auth import (
    ROLE_LEVELS,
    AuthError,
    UserRole,
    get_user_role,
    has_role,
    require_admin,
    require_analyst,
    require_operator,
    require_viewer,
)

# =============================================================================
# UserRole Enum Tests
# =============================================================================


class TestUserRoleEnum:
    """Tests for the UserRole enum."""

    def test_enum_values(self):
        """Test that all expected roles exist."""
        assert UserRole.VIEWER.value == "viewer"
        assert UserRole.ANALYST.value == "analyst"
        assert UserRole.OPERATOR.value == "operator"
        assert UserRole.ADMIN.value == "admin"

    def test_enum_count(self):
        """Test that exactly 4 roles exist."""
        assert len(UserRole) == 4

    def test_enum_from_string(self):
        """Test enum can be created from string."""
        assert UserRole("viewer") == UserRole.VIEWER
        assert UserRole("analyst") == UserRole.ANALYST
        assert UserRole("operator") == UserRole.OPERATOR
        assert UserRole("admin") == UserRole.ADMIN

    def test_enum_is_string(self):
        """Test that enum values are strings."""
        assert isinstance(UserRole.VIEWER.value, str)
        # UserRole inherits from str, so .value gives the string
        assert UserRole.VIEWER.value == "viewer"


# =============================================================================
# ROLE_LEVELS Tests
# =============================================================================


class TestRoleLevels:
    """Tests for the ROLE_LEVELS hierarchy."""

    def test_all_roles_have_levels(self):
        """Test that all roles have a level assigned."""
        for role in UserRole:
            assert role in ROLE_LEVELS

    def test_hierarchy_order(self):
        """Test that hierarchy is ADMIN > OPERATOR > ANALYST > VIEWER."""
        assert ROLE_LEVELS[UserRole.ADMIN] > ROLE_LEVELS[UserRole.OPERATOR]
        assert ROLE_LEVELS[UserRole.OPERATOR] > ROLE_LEVELS[UserRole.ANALYST]
        assert ROLE_LEVELS[UserRole.ANALYST] > ROLE_LEVELS[UserRole.VIEWER]

    def test_specific_levels(self):
        """Test specific level values."""
        assert ROLE_LEVELS[UserRole.VIEWER] == 1
        assert ROLE_LEVELS[UserRole.ANALYST] == 2
        assert ROLE_LEVELS[UserRole.OPERATOR] == 3
        assert ROLE_LEVELS[UserRole.ADMIN] == 4


# =============================================================================
# get_user_role() Tests
# =============================================================================


class TestGetUserRole:
    """Tests for the get_user_role function."""

    def test_role_from_app_metadata(self):
        """Test extracting role from app_metadata.role (preferred)."""
        user = {"app_metadata": {"role": "analyst"}}
        assert get_user_role(user) == UserRole.ANALYST

    def test_role_from_top_level(self):
        """Test fallback to top-level role field."""
        user = {"role": "operator"}
        assert get_user_role(user) == UserRole.OPERATOR

    def test_app_metadata_takes_precedence(self):
        """Test that app_metadata.role takes precedence over top-level."""
        user = {"app_metadata": {"role": "admin"}, "role": "viewer"}
        assert get_user_role(user) == UserRole.ADMIN

    def test_legacy_is_admin_flag(self):
        """Test legacy is_admin flag migration."""
        user = {"app_metadata": {"is_admin": True}}
        assert get_user_role(user) == UserRole.ADMIN

    def test_default_to_viewer(self):
        """Test default to viewer when no role specified."""
        assert get_user_role({}) == UserRole.VIEWER
        assert get_user_role({"app_metadata": {}}) == UserRole.VIEWER

    def test_unknown_role_defaults_to_viewer(self):
        """Test unknown role string defaults to viewer."""
        user = {"app_metadata": {"role": "superuser"}}
        assert get_user_role(user) == UserRole.VIEWER

    def test_case_insensitive(self):
        """Test role matching is case insensitive."""
        assert get_user_role({"app_metadata": {"role": "ADMIN"}}) == UserRole.ADMIN
        assert get_user_role({"app_metadata": {"role": "Analyst"}}) == UserRole.ANALYST
        assert get_user_role({"app_metadata": {"role": "OPERATOR"}}) == UserRole.OPERATOR


# =============================================================================
# has_role() Tests
# =============================================================================


class TestHasRole:
    """Tests for the has_role hierarchical check."""

    def test_admin_has_all_roles(self):
        """Test that admin has access to all role levels."""
        admin_user = {"app_metadata": {"role": "admin"}}
        assert has_role(admin_user, UserRole.ADMIN) is True
        assert has_role(admin_user, UserRole.OPERATOR) is True
        assert has_role(admin_user, UserRole.ANALYST) is True
        assert has_role(admin_user, UserRole.VIEWER) is True

    def test_operator_has_operator_and_below(self):
        """Test that operator has access to operator, analyst, viewer."""
        operator_user = {"app_metadata": {"role": "operator"}}
        assert has_role(operator_user, UserRole.ADMIN) is False
        assert has_role(operator_user, UserRole.OPERATOR) is True
        assert has_role(operator_user, UserRole.ANALYST) is True
        assert has_role(operator_user, UserRole.VIEWER) is True

    def test_analyst_has_analyst_and_below(self):
        """Test that analyst has access to analyst, viewer."""
        analyst_user = {"app_metadata": {"role": "analyst"}}
        assert has_role(analyst_user, UserRole.ADMIN) is False
        assert has_role(analyst_user, UserRole.OPERATOR) is False
        assert has_role(analyst_user, UserRole.ANALYST) is True
        assert has_role(analyst_user, UserRole.VIEWER) is True

    def test_viewer_has_only_viewer(self):
        """Test that viewer only has viewer access."""
        viewer_user = {"app_metadata": {"role": "viewer"}}
        assert has_role(viewer_user, UserRole.ADMIN) is False
        assert has_role(viewer_user, UserRole.OPERATOR) is False
        assert has_role(viewer_user, UserRole.ANALYST) is False
        assert has_role(viewer_user, UserRole.VIEWER) is True

    def test_no_role_user_has_viewer_access(self):
        """Test user with no role defaults to viewer access."""
        no_role_user = {}
        assert has_role(no_role_user, UserRole.VIEWER) is True
        assert has_role(no_role_user, UserRole.ANALYST) is False


# =============================================================================
# require_viewer() Tests
# =============================================================================


class TestRequireViewer:
    """Tests for require_viewer dependency."""

    @pytest.mark.asyncio
    async def test_viewer_passes(self):
        """Test that viewer role passes."""
        user = {"app_metadata": {"role": "viewer"}}
        result = await require_viewer(user)
        assert result == user

    @pytest.mark.asyncio
    async def test_analyst_passes(self):
        """Test that analyst role passes (higher than viewer)."""
        user = {"app_metadata": {"role": "analyst"}}
        result = await require_viewer(user)
        assert result == user

    @pytest.mark.asyncio
    async def test_admin_passes(self):
        """Test that admin role passes."""
        user = {"app_metadata": {"role": "admin"}}
        result = await require_viewer(user)
        assert result == user


# =============================================================================
# require_analyst() Tests
# =============================================================================


class TestRequireAnalyst:
    """Tests for require_analyst dependency."""

    @pytest.mark.asyncio
    async def test_viewer_fails(self):
        """Test that viewer role fails with 403."""
        user = {"app_metadata": {"role": "viewer"}}
        with pytest.raises(AuthError) as exc_info:
            await require_analyst(user)
        assert exc_info.value.status_code == 403
        assert "Analyst privileges required" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_analyst_passes(self):
        """Test that analyst role passes."""
        user = {"app_metadata": {"role": "analyst"}}
        result = await require_analyst(user)
        assert result == user

    @pytest.mark.asyncio
    async def test_operator_passes(self):
        """Test that operator role passes (higher than analyst)."""
        user = {"app_metadata": {"role": "operator"}}
        result = await require_analyst(user)
        assert result == user

    @pytest.mark.asyncio
    async def test_admin_passes(self):
        """Test that admin role passes."""
        user = {"app_metadata": {"role": "admin"}}
        result = await require_analyst(user)
        assert result == user


# =============================================================================
# require_operator() Tests
# =============================================================================


class TestRequireOperator:
    """Tests for require_operator dependency."""

    @pytest.mark.asyncio
    async def test_viewer_fails(self):
        """Test that viewer role fails with 403."""
        user = {"app_metadata": {"role": "viewer"}}
        with pytest.raises(AuthError) as exc_info:
            await require_operator(user)
        assert exc_info.value.status_code == 403
        assert "Operator privileges required" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_analyst_fails(self):
        """Test that analyst role fails with 403."""
        user = {"app_metadata": {"role": "analyst"}}
        with pytest.raises(AuthError) as exc_info:
            await require_operator(user)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_operator_passes(self):
        """Test that operator role passes."""
        user = {"app_metadata": {"role": "operator"}}
        result = await require_operator(user)
        assert result == user

    @pytest.mark.asyncio
    async def test_admin_passes(self):
        """Test that admin role passes."""
        user = {"app_metadata": {"role": "admin"}}
        result = await require_operator(user)
        assert result == user


# =============================================================================
# require_admin() Tests
# =============================================================================


class TestRequireAdmin:
    """Tests for require_admin dependency."""

    @pytest.mark.asyncio
    async def test_viewer_fails(self):
        """Test that viewer role fails with 403."""
        user = {"app_metadata": {"role": "viewer"}}
        with pytest.raises(AuthError) as exc_info:
            await require_admin(user)
        assert exc_info.value.status_code == 403
        assert "Admin privileges required" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_analyst_fails(self):
        """Test that analyst role fails with 403."""
        user = {"app_metadata": {"role": "analyst"}}
        with pytest.raises(AuthError) as exc_info:
            await require_admin(user)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_operator_fails(self):
        """Test that operator role fails with 403."""
        user = {"app_metadata": {"role": "operator"}}
        with pytest.raises(AuthError) as exc_info:
            await require_admin(user)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_admin_passes(self):
        """Test that admin role passes."""
        user = {"app_metadata": {"role": "admin"}}
        result = await require_admin(user)
        assert result == user


# =============================================================================
# AuthError Tests
# =============================================================================


class TestAuthError:
    """Tests for the AuthError exception."""

    def test_default_status_code(self):
        """Test default 401 status code."""
        error = AuthError("Test message")
        assert error.status_code == 401

    def test_custom_status_code(self):
        """Test custom 403 status code."""
        error = AuthError("Forbidden", status_code=403)
        assert error.status_code == 403

    def test_detail_format(self):
        """Test error detail format."""
        error = AuthError("Test message")
        assert error.detail == {"error": "authentication_error", "message": "Test message"}

    def test_www_authenticate_header(self):
        """Test WWW-Authenticate header is set."""
        error = AuthError("Test message")
        assert error.headers == {"WWW-Authenticate": "Bearer"}
