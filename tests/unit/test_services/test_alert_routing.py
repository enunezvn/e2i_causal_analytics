"""
Unit Tests for Alert Routing Service (Phase 14).

Tests cover:
- Alert severity routing
- Notification channel configuration
- Alert payload creation
- Deduplication and rate limiting
- Provider integration
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.alert_routing import (
    AlertPayload,
    AlertRouter,
    AlertRoutingConfig,
    NotificationChannel,
    SlackNotificationProvider,
    EmailNotificationProvider,
    WebhookNotificationProvider,
    get_alert_router,
    route_drift_alerts,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def default_config() -> AlertRoutingConfig:
    """Create default alert routing configuration."""
    return AlertRoutingConfig(
        email_enabled=True,
        smtp_host="smtp.example.com",
        smtp_port=587,
        smtp_username="user",
        smtp_password="pass",
        email_from="alerts@example.com",
        email_recipients=["ml-team@example.com"],
        slack_enabled=True,
        slack_webhook_url="https://hooks.slack.com/test",
        slack_channel="#ml-alerts",
        webhook_enabled=True,
        webhook_url="https://hooks.example.com/alerts",
        min_interval_seconds=60,
        max_alerts_per_hour=100,
    )


@pytest.fixture
def alert_router(default_config: AlertRoutingConfig) -> AlertRouter:
    """Create alert router instance."""
    return AlertRouter(config=default_config)


@pytest.fixture
def sample_alert_payload() -> AlertPayload:
    """Create sample alert payload."""
    return AlertPayload(
        alert_id="alert-123",
        model_id="propensity_v2.1.0",
        alert_type="drift_data",
        severity="high",
        title="Data Drift Detected",
        description="Significant drift detected in feature 'days_since_last_visit'",
        triggered_at=datetime.now(timezone.utc),
        drift_score=0.75,
        features_affected=["days_since_last_visit", "rx_count"],
        recommended_actions=["Review feature distributions", "Consider retraining"],
        metadata={
            "baseline_period": "2024-12-01 to 2024-12-08",
            "current_period": "2024-12-08 to 2024-12-15",
        },
    )


# =============================================================================
# NOTIFICATION CHANNEL TESTS
# =============================================================================


class TestNotificationChannel:
    """Tests for NotificationChannel enum."""

    def test_email_channel_exists(self):
        """Test EMAIL channel is defined."""
        assert NotificationChannel.EMAIL == "email"

    def test_slack_channel_exists(self):
        """Test SLACK channel is defined."""
        assert NotificationChannel.SLACK == "slack"

    def test_webhook_channel_exists(self):
        """Test WEBHOOK channel is defined."""
        assert NotificationChannel.WEBHOOK == "webhook"

    def test_pagerduty_channel_exists(self):
        """Test PAGERDUTY channel is defined."""
        assert NotificationChannel.PAGERDUTY == "pagerduty"


# =============================================================================
# ALERT ROUTING CONFIG TESTS
# =============================================================================


class TestAlertRoutingConfig:
    """Tests for AlertRoutingConfig dataclass."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = AlertRoutingConfig()

        assert config.email_enabled is False
        assert config.slack_enabled is False
        assert config.webhook_enabled is False
        assert config.email_recipients == []
        assert config.slack_channel == "#ml-alerts"
        assert config.webhook_url == ""
        assert config.min_interval_seconds == 300
        assert config.max_alerts_per_hour == 20

    def test_custom_config_values(self, default_config: AlertRoutingConfig):
        """Test custom configuration values."""
        assert default_config.email_enabled is True
        assert default_config.slack_enabled is True
        assert default_config.webhook_enabled is True
        assert "ml-team@example.com" in default_config.email_recipients
        assert default_config.slack_channel == "#ml-alerts"

    def test_severity_channel_defaults(self):
        """Test default severity channel configuration."""
        config = AlertRoutingConfig()

        # Critical severity should include all major channels
        assert NotificationChannel.SLACK in config.critical_channels
        assert NotificationChannel.EMAIL in config.critical_channels
        assert NotificationChannel.PAGERDUTY in config.critical_channels

        # High severity should include Slack and Email
        assert NotificationChannel.SLACK in config.high_channels
        assert NotificationChannel.EMAIL in config.high_channels

        # Medium severity should include Slack
        assert NotificationChannel.SLACK in config.medium_channels

        # Low severity should be empty by default
        assert config.low_channels == []


# =============================================================================
# ALERT PAYLOAD TESTS
# =============================================================================


class TestAlertPayload:
    """Tests for AlertPayload dataclass."""

    def test_payload_creation(self, sample_alert_payload: AlertPayload):
        """Test alert payload creation."""
        assert sample_alert_payload.alert_id == "alert-123"
        assert sample_alert_payload.model_id == "propensity_v2.1.0"
        assert sample_alert_payload.alert_type == "drift_data"
        assert sample_alert_payload.severity == "high"

    def test_payload_drift_score(self, sample_alert_payload: AlertPayload):
        """Test drift score in payload."""
        assert sample_alert_payload.drift_score == 0.75

    def test_payload_features_affected(self, sample_alert_payload: AlertPayload):
        """Test features affected list."""
        assert len(sample_alert_payload.features_affected) == 2
        assert "days_since_last_visit" in sample_alert_payload.features_affected

    def test_payload_recommended_actions(self, sample_alert_payload: AlertPayload):
        """Test recommended actions list."""
        assert len(sample_alert_payload.recommended_actions) == 2
        assert "Review feature distributions" in sample_alert_payload.recommended_actions

    def test_payload_metadata(self, sample_alert_payload: AlertPayload):
        """Test payload metadata."""
        assert "baseline_period" in sample_alert_payload.metadata
        assert "current_period" in sample_alert_payload.metadata

    def test_dedup_key_generation(self, sample_alert_payload: AlertPayload):
        """Test deduplication key generation."""
        key = sample_alert_payload.get_dedup_key()
        assert isinstance(key, str)
        assert len(key) == 32  # MD5 hash length

    def test_dedup_key_consistency(self, sample_alert_payload: AlertPayload):
        """Test that same alert generates same dedup key."""
        key1 = sample_alert_payload.get_dedup_key()
        key2 = sample_alert_payload.get_dedup_key()
        assert key1 == key2

    def test_dedup_key_uniqueness(self):
        """Test that different alerts generate different dedup keys."""
        alert1 = AlertPayload(
            alert_id="alert-1",
            model_id="model_a",
            alert_type="drift",
            severity="high",
            title="Alert 1",
            description="Desc 1",
            triggered_at=datetime.now(timezone.utc),
        )
        alert2 = AlertPayload(
            alert_id="alert-2",
            model_id="model_b",
            alert_type="drift",
            severity="high",
            title="Alert 2",
            description="Desc 2",
            triggered_at=datetime.now(timezone.utc),
        )

        assert alert1.get_dedup_key() != alert2.get_dedup_key()


# =============================================================================
# NOTIFICATION PROVIDER TESTS
# =============================================================================


class TestSlackNotificationProvider:
    """Tests for Slack notification provider."""

    def test_is_enabled_when_configured(self, default_config: AlertRoutingConfig):
        """Test provider is enabled with valid config."""
        provider = SlackNotificationProvider(default_config)
        assert provider.is_enabled() is True

    def test_is_disabled_when_not_enabled(self):
        """Test provider is disabled when slack_enabled is False."""
        config = AlertRoutingConfig(slack_enabled=False)
        provider = SlackNotificationProvider(config)
        assert provider.is_enabled() is False

    def test_is_disabled_without_webhook_url(self):
        """Test provider is disabled without webhook URL."""
        config = AlertRoutingConfig(slack_enabled=True, slack_webhook_url="")
        provider = SlackNotificationProvider(config)
        assert provider.is_enabled() is False

    def test_get_severity_color(self, default_config: AlertRoutingConfig):
        """Test severity color mapping."""
        provider = SlackNotificationProvider(default_config)

        assert provider._get_severity_color("critical") == "#dc3545"
        assert provider._get_severity_color("high") == "#fd7e14"
        assert provider._get_severity_color("medium") == "#ffc107"
        assert provider._get_severity_color("low") == "#28a745"
        assert provider._get_severity_color("unknown") == "#6c757d"


class TestEmailNotificationProvider:
    """Tests for Email notification provider."""

    def test_is_enabled_when_configured(self, default_config: AlertRoutingConfig):
        """Test provider is enabled with valid config."""
        provider = EmailNotificationProvider(default_config)
        assert provider.is_enabled() is True

    def test_is_disabled_when_not_enabled(self):
        """Test provider is disabled when email_enabled is False."""
        config = AlertRoutingConfig(email_enabled=False)
        provider = EmailNotificationProvider(config)
        assert provider.is_enabled() is False

    def test_is_disabled_without_smtp_host(self):
        """Test provider is disabled without SMTP host."""
        config = AlertRoutingConfig(
            email_enabled=True,
            smtp_host="",
            email_recipients=["test@example.com"],
        )
        provider = EmailNotificationProvider(config)
        assert provider.is_enabled() is False

    def test_is_disabled_without_recipients(self):
        """Test provider is disabled without recipients."""
        config = AlertRoutingConfig(
            email_enabled=True,
            smtp_host="smtp.example.com",
            email_recipients=[],
        )
        provider = EmailNotificationProvider(config)
        assert provider.is_enabled() is False


class TestWebhookNotificationProvider:
    """Tests for Webhook notification provider."""

    def test_is_enabled_when_configured(self, default_config: AlertRoutingConfig):
        """Test provider is enabled with valid config."""
        provider = WebhookNotificationProvider(default_config)
        assert provider.is_enabled() is True

    def test_is_disabled_when_not_enabled(self):
        """Test provider is disabled when webhook_enabled is False."""
        config = AlertRoutingConfig(webhook_enabled=False)
        provider = WebhookNotificationProvider(config)
        assert provider.is_enabled() is False

    def test_is_disabled_without_webhook_url(self):
        """Test provider is disabled without webhook URL."""
        config = AlertRoutingConfig(webhook_enabled=True, webhook_url="")
        provider = WebhookNotificationProvider(config)
        assert provider.is_enabled() is False

    def test_format_payload(self, default_config: AlertRoutingConfig, sample_alert_payload: AlertPayload):
        """Test webhook payload formatting."""
        provider = WebhookNotificationProvider(default_config)
        payload = provider._format_payload(sample_alert_payload)

        assert payload["alert_id"] == "alert-123"
        assert payload["model_id"] == "propensity_v2.1.0"
        assert payload["severity"] == "high"
        assert payload["drift_score"] == 0.75


# =============================================================================
# ALERT ROUTER TESTS
# =============================================================================


class TestAlertRouter:
    """Tests for AlertRouter class."""

    def test_router_initialization(self, alert_router: AlertRouter):
        """Test router initialization."""
        assert alert_router is not None
        assert alert_router.config.email_enabled is True
        assert alert_router.config.slack_enabled is True

    def test_providers_initialized(self, alert_router: AlertRouter):
        """Test that all providers are initialized."""
        assert NotificationChannel.SLACK in alert_router._providers
        assert NotificationChannel.EMAIL in alert_router._providers
        assert NotificationChannel.WEBHOOK in alert_router._providers

    def test_get_channels_for_severity_critical(self, alert_router: AlertRouter):
        """Test channel routing for critical severity."""
        channels = alert_router._get_channels_for_severity("critical")
        assert len(channels) >= 1
        assert NotificationChannel.SLACK in channels

    def test_get_channels_for_severity_high(self, alert_router: AlertRouter):
        """Test channel routing for high severity."""
        channels = alert_router._get_channels_for_severity("high")
        assert len(channels) >= 1

    def test_get_channels_for_severity_medium(self, alert_router: AlertRouter):
        """Test channel routing for medium severity."""
        channels = alert_router._get_channels_for_severity("medium")
        assert len(channels) >= 1

    def test_get_channels_for_severity_low(self, alert_router: AlertRouter):
        """Test channel routing for low severity."""
        channels = alert_router._get_channels_for_severity("low")
        # Low severity may have no channels by default
        assert isinstance(channels, list)

    def test_get_channels_for_unknown_severity(self, alert_router: AlertRouter):
        """Test channel routing for unknown severity."""
        channels = alert_router._get_channels_for_severity("unknown")
        assert channels == []

    @pytest.mark.asyncio
    async def test_route_alert_success(
        self, alert_router: AlertRouter, sample_alert_payload: AlertPayload
    ):
        """Test successful alert routing."""
        # Mock all providers
        for provider in alert_router._providers.values():
            provider.send = AsyncMock(return_value=True)
            provider.is_enabled = MagicMock(return_value=True)

        result = await alert_router.route_alert(sample_alert_payload)

        assert result["alert_id"] == "alert-123"
        assert result["deduplicated"] is False
        assert result["rate_limited"] is False
        assert "channels" in result

    @pytest.mark.asyncio
    async def test_route_alert_deduplication(
        self, alert_router: AlertRouter, sample_alert_payload: AlertPayload
    ):
        """Test alert deduplication."""
        # Mock providers
        for provider in alert_router._providers.values():
            provider.send = AsyncMock(return_value=True)
            provider.is_enabled = MagicMock(return_value=True)

        # Send first alert
        await alert_router.route_alert(sample_alert_payload)

        # Send same alert again - should be deduplicated
        result = await alert_router.route_alert(sample_alert_payload)

        assert result["deduplicated"] is True

    @pytest.mark.asyncio
    async def test_route_alert_with_disabled_channels(self):
        """Test routing with all channels disabled."""
        config = AlertRoutingConfig(
            email_enabled=False,
            slack_enabled=False,
            webhook_enabled=False,
        )
        router = AlertRouter(config=config)

        payload = AlertPayload(
            alert_id="test-123",
            model_id="test_v1.0",
            alert_type="drift",
            severity="high",
            title="Test Alert",
            description="Test description",
            triggered_at=datetime.now(timezone.utc),
        )

        result = await router.route_alert(payload)

        # Should succeed but with no channels in result
        assert result["alert_id"] == "test-123"

    @pytest.mark.asyncio
    async def test_route_batch(
        self, alert_router: AlertRouter
    ):
        """Test routing multiple alerts."""
        # Mock providers
        for provider in alert_router._providers.values():
            provider.send = AsyncMock(return_value=True)
            provider.is_enabled = MagicMock(return_value=True)

        alerts = [
            AlertPayload(
                alert_id=f"alert-{i}",
                model_id="test_v1.0",
                alert_type="drift",
                severity="high",
                title=f"Alert {i}",
                description=f"Description {i}",
                triggered_at=datetime.now(timezone.utc),
            )
            for i in range(3)
        ]

        results = await alert_router.route_batch(alerts)

        assert len(results) == 3
        for result in results:
            assert "alert_id" in result


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_get_alert_router_default(self):
        """Test getting default alert router."""
        router = get_alert_router()
        assert isinstance(router, AlertRouter)

    def test_get_alert_router_with_config(self, default_config: AlertRoutingConfig):
        """Test getting alert router with custom config."""
        router = get_alert_router(config=default_config)
        assert router.config.email_enabled is True

    @pytest.mark.asyncio
    async def test_route_drift_alerts_with_no_drift(self):
        """Test routing when no drift is detected."""
        drift_results = [
            {"feature": "feature_1", "drift_detected": False}
        ]

        result = await route_drift_alerts(
            model_version="test_v1.0",
            drift_results=drift_results,
            overall_score=0.1,
            summary="No drift detected",
            recommended_actions=[],
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_route_drift_alerts_with_drift(self):
        """Test routing when drift is detected."""
        drift_results = [
            {
                "feature": "feature_1",
                "drift_detected": True,
                "severity": "high",
                "drift_type": "data",
                "test_statistic": 0.75,
                "p_value": 0.01,
            }
        ]

        with patch("src.services.alert_routing.get_alert_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.route_batch = AsyncMock(return_value=[{"success": True}])
            mock_get_router.return_value = mock_router

            result = await route_drift_alerts(
                model_version="test_v1.0",
                drift_results=drift_results,
                overall_score=0.75,
                summary="High drift detected",
                recommended_actions=["Consider retraining"],
            )

            assert len(result) == 1
            mock_router.route_batch.assert_called_once()


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_route_alert_with_empty_features(self, alert_router: AlertRouter):
        """Test routing alert with no features affected."""
        # Mock providers
        for provider in alert_router._providers.values():
            provider.send = AsyncMock(return_value=True)
            provider.is_enabled = MagicMock(return_value=True)

        payload = AlertPayload(
            alert_id="empty-features",
            model_id="test_v1.0",
            alert_type="drift",
            severity="low",
            title="Minor Drift",
            description="Minimal drift detected",
            triggered_at=datetime.now(timezone.utc),
            features_affected=[],
        )

        result = await alert_router.route_alert(payload)

        assert result["alert_id"] == "empty-features"

    @pytest.mark.asyncio
    async def test_route_alert_with_none_severity(self, alert_router: AlertRouter):
        """Test routing alert with 'none' severity gets no channels."""
        # Create alert with low severity (which has no channels by default)
        payload = AlertPayload(
            alert_id="none-severity",
            model_id="test_v1.0",
            alert_type="drift",
            severity="none",
            title="No Drift",
            description="No significant drift",
            triggered_at=datetime.now(timezone.utc),
        )

        result = await alert_router.route_alert(payload)

        assert result["channels"] == {}

    def test_payload_with_optional_fields(self):
        """Test payload creation with only required fields."""
        payload = AlertPayload(
            alert_id="minimal",
            model_id="test_v1.0",
            alert_type="drift",
            severity="low",
            title="Minimal Alert",
            description="Test",
            triggered_at=datetime.now(timezone.utc),
        )

        assert payload.drift_score is None
        assert payload.features_affected == []
        assert payload.recommended_actions == []
        assert payload.metadata == {}


# =============================================================================
# INTEGRATION-STYLE TESTS
# =============================================================================


class TestAlertRoutingWorkflow:
    """Tests for complete alert routing workflows."""

    @pytest.mark.asyncio
    async def test_full_drift_alert_workflow(self, alert_router: AlertRouter):
        """Test complete workflow from drift detection to alert routing."""
        # Mock providers
        for provider in alert_router._providers.values():
            provider.send = AsyncMock(return_value=True)
            provider.is_enabled = MagicMock(return_value=True)

        payload = AlertPayload(
            alert_id="workflow-test",
            model_id="propensity_v2.1.0",
            alert_type="drift_data",
            severity="high",
            title="Workflow Test Alert",
            description="Testing full workflow",
            triggered_at=datetime.now(timezone.utc),
            drift_score=0.72,
            features_affected=["days_since_last_visit", "rx_count", "engagement_score"],
            recommended_actions=["Review distributions", "Consider retraining"],
        )

        result = await alert_router.route_alert(payload)

        assert result["alert_id"] == "workflow-test"
        assert result["deduplicated"] is False
        assert result["rate_limited"] is False

    @pytest.mark.asyncio
    async def test_multiple_alerts_sequential(self, alert_router: AlertRouter):
        """Test routing multiple alerts sequentially."""
        # Mock providers
        for provider in alert_router._providers.values():
            provider.send = AsyncMock(return_value=True)
            provider.is_enabled = MagicMock(return_value=True)

        severities = ["low", "medium", "high", "critical"]

        for i, severity in enumerate(severities):
            payload = AlertPayload(
                alert_id=f"seq-alert-{i}",
                model_id=f"model_{i}",  # Different models to avoid dedup
                alert_type="drift",
                severity=severity,
                title=f"Alert {i}",
                description=f"Description {i}",
                triggered_at=datetime.now(timezone.utc),
            )

            result = await alert_router.route_alert(payload)

            assert result["alert_id"] == f"seq-alert-{i}"
