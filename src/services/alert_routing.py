"""Alert Routing Service.

Phase 14: Model Monitoring & Drift Detection

This service handles routing of drift alerts to various notification channels:
- Email (SMTP)
- Slack (Webhook)
- PagerDuty (Optional)
- Custom webhooks

Features:
- Alert deduplication
- Escalation policies
- Rate limiting
- Channel-specific formatting
"""

import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


class NotificationChannel(str, Enum):
    """Supported notification channels."""

    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"


@dataclass
class AlertRoutingConfig:
    """Configuration for alert routing."""

    # Email settings
    email_enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    email_from: str = "alerts@e2i-analytics.com"
    email_recipients: List[str] = field(default_factory=list)

    # Slack settings
    slack_enabled: bool = False
    slack_webhook_url: str = ""
    slack_channel: str = "#ml-alerts"

    # PagerDuty settings
    pagerduty_enabled: bool = False
    pagerduty_routing_key: str = ""

    # Custom webhook
    webhook_enabled: bool = False
    webhook_url: str = ""
    webhook_headers: Dict[str, str] = field(default_factory=dict)

    # Rate limiting
    min_interval_seconds: int = 300  # 5 minutes between same alerts
    max_alerts_per_hour: int = 20

    # Severity routing
    critical_channels: List[NotificationChannel] = field(
        default_factory=lambda: [
            NotificationChannel.SLACK,
            NotificationChannel.EMAIL,
            NotificationChannel.PAGERDUTY,
        ]
    )
    high_channels: List[NotificationChannel] = field(
        default_factory=lambda: [NotificationChannel.SLACK, NotificationChannel.EMAIL]
    )
    medium_channels: List[NotificationChannel] = field(
        default_factory=lambda: [NotificationChannel.SLACK]
    )
    low_channels: List[NotificationChannel] = field(default_factory=list)


@dataclass
class AlertPayload:
    """Alert payload for routing."""

    alert_id: str
    model_id: str
    alert_type: str
    severity: str  # critical, high, medium, low
    title: str
    description: str
    triggered_at: datetime
    drift_score: Optional[float] = None
    features_affected: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_dedup_key(self) -> str:
        """Generate deduplication key for the alert."""
        key_parts = [self.model_id, self.alert_type, self.severity]
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()


# =============================================================================
# NOTIFICATION PROVIDERS
# =============================================================================


class NotificationProvider(ABC):
    """Base class for notification providers."""

    @abstractmethod
    async def send(self, alert: AlertPayload) -> bool:
        """Send alert notification.

        Args:
            alert: Alert payload

        Returns:
            True if sent successfully
        """
        pass

    @abstractmethod
    def is_enabled(self) -> bool:
        """Check if provider is enabled and configured."""
        pass


class SlackNotificationProvider(NotificationProvider):
    """Slack webhook notification provider."""

    def __init__(self, config: AlertRoutingConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None

    def is_enabled(self) -> bool:
        return self.config.slack_enabled and bool(self.config.slack_webhook_url)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _get_severity_color(self, severity: str) -> str:
        """Map severity to Slack color."""
        colors = {
            "critical": "#dc3545",  # Red
            "high": "#fd7e14",  # Orange
            "medium": "#ffc107",  # Yellow
            "low": "#28a745",  # Green
        }
        return colors.get(severity, "#6c757d")

    def _format_message(self, alert: AlertPayload) -> Dict[str, Any]:
        """Format alert as Slack message."""
        color = self._get_severity_color(alert.severity)

        fields = [
            {"title": "Model", "value": alert.model_id, "short": True},
            {"title": "Severity", "value": alert.severity.upper(), "short": True},
            {"title": "Alert Type", "value": alert.alert_type, "short": True},
        ]

        if alert.drift_score is not None:
            fields.append(
                {"title": "Drift Score", "value": f"{alert.drift_score:.2f}", "short": True}
            )

        if alert.features_affected:
            fields.append(
                {
                    "title": "Affected Features",
                    "value": ", ".join(alert.features_affected[:5]),
                    "short": False,
                }
            )

        if alert.recommended_actions:
            actions_text = "\n".join(f"• {a}" for a in alert.recommended_actions[:3])
            fields.append(
                {"title": "Recommended Actions", "value": actions_text, "short": False}
            )

        return {
            "channel": self.config.slack_channel,
            "attachments": [
                {
                    "color": color,
                    "title": alert.title,
                    "text": alert.description,
                    "fields": fields,
                    "footer": "E2I Drift Monitor",
                    "ts": int(alert.triggered_at.timestamp()),
                }
            ],
        }

    async def send(self, alert: AlertPayload) -> bool:
        if not self.is_enabled():
            return False

        try:
            session = await self._get_session()
            message = self._format_message(alert)

            async with session.post(
                self.config.slack_webhook_url,
                json=message,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status == 200:
                    logger.info(f"Slack alert sent for {alert.alert_id}")
                    return True
                else:
                    logger.warning(
                        f"Slack alert failed: {response.status} - {await response.text()}"
                    )
                    return False

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False


class EmailNotificationProvider(NotificationProvider):
    """Email (SMTP) notification provider."""

    def __init__(self, config: AlertRoutingConfig):
        self.config = config

    def is_enabled(self) -> bool:
        return (
            self.config.email_enabled
            and bool(self.config.smtp_host)
            and bool(self.config.email_recipients)
        )

    def _format_html_body(self, alert: AlertPayload) -> str:
        """Format alert as HTML email body."""
        severity_colors = {
            "critical": "#dc3545",
            "high": "#fd7e14",
            "medium": "#ffc107",
            "low": "#28a745",
        }
        color = severity_colors.get(alert.severity, "#6c757d")

        features_html = ""
        if alert.features_affected:
            features_list = "".join(f"<li>{f}</li>" for f in alert.features_affected)
            features_html = f"<h4>Affected Features</h4><ul>{features_list}</ul>"

        actions_html = ""
        if alert.recommended_actions:
            actions_list = "".join(f"<li>{a}</li>" for a in alert.recommended_actions)
            actions_html = f"<h4>Recommended Actions</h4><ul>{actions_list}</ul>"

        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background-color: {color}; color: white; padding: 15px; border-radius: 5px 5px 0 0;">
                <h2 style="margin: 0;">{alert.severity.upper()} Alert: {alert.title}</h2>
            </div>
            <div style="padding: 20px; border: 1px solid #ddd; border-top: none;">
                <p><strong>Model:</strong> {alert.model_id}</p>
                <p><strong>Alert Type:</strong> {alert.alert_type}</p>
                <p><strong>Triggered:</strong> {alert.triggered_at.isoformat()}</p>
                {f'<p><strong>Drift Score:</strong> {alert.drift_score:.2f}</p>' if alert.drift_score else ''}

                <h4>Description</h4>
                <p>{alert.description}</p>

                {features_html}
                {actions_html}
            </div>
            <div style="padding: 10px; background-color: #f8f9fa; text-align: center; font-size: 12px;">
                E2I Causal Analytics - Model Monitoring
            </div>
        </body>
        </html>
        """

    async def send(self, alert: AlertPayload) -> bool:
        if not self.is_enabled():
            return False

        try:
            import smtplib
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText

            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[{alert.severity.upper()}] {alert.title} - {alert.model_id}"
            msg["From"] = self.config.email_from
            msg["To"] = ", ".join(self.config.email_recipients)

            # Plain text version
            plain_text = f"""
{alert.title}

Model: {alert.model_id}
Severity: {alert.severity.upper()}
Alert Type: {alert.alert_type}
Triggered: {alert.triggered_at.isoformat()}
{f'Drift Score: {alert.drift_score:.2f}' if alert.drift_score else ''}

{alert.description}

{'Affected Features: ' + ', '.join(alert.features_affected) if alert.features_affected else ''}

{'Recommended Actions:' + chr(10) + chr(10).join(f'- {a}' for a in alert.recommended_actions) if alert.recommended_actions else ''}

---
E2I Causal Analytics - Model Monitoring
            """

            html_body = self._format_html_body(alert)

            msg.attach(MIMEText(plain_text, "plain"))
            msg.attach(MIMEText(html_body, "html"))

            # Run SMTP in thread pool to avoid blocking
            def send_email():
                with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                    if self.config.smtp_username and self.config.smtp_password:
                        server.starttls()
                        server.login(self.config.smtp_username, self.config.smtp_password)
                    server.send_message(msg)

            await asyncio.get_event_loop().run_in_executor(None, send_email)

            logger.info(f"Email alert sent for {alert.alert_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False


class WebhookNotificationProvider(NotificationProvider):
    """Generic webhook notification provider."""

    def __init__(self, config: AlertRoutingConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None

    def is_enabled(self) -> bool:
        return self.config.webhook_enabled and bool(self.config.webhook_url)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _format_payload(self, alert: AlertPayload) -> Dict[str, Any]:
        """Format alert as webhook payload."""
        return {
            "alert_id": alert.alert_id,
            "model_id": alert.model_id,
            "alert_type": alert.alert_type,
            "severity": alert.severity,
            "title": alert.title,
            "description": alert.description,
            "triggered_at": alert.triggered_at.isoformat(),
            "drift_score": alert.drift_score,
            "features_affected": alert.features_affected,
            "recommended_actions": alert.recommended_actions,
            "metadata": alert.metadata,
        }

    async def send(self, alert: AlertPayload) -> bool:
        if not self.is_enabled():
            return False

        try:
            session = await self._get_session()
            payload = self._format_payload(alert)
            headers = {"Content-Type": "application/json", **self.config.webhook_headers}

            async with session.post(
                self.config.webhook_url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if 200 <= response.status < 300:
                    logger.info(f"Webhook alert sent for {alert.alert_id}")
                    return True
                else:
                    logger.warning(
                        f"Webhook alert failed: {response.status} - {await response.text()}"
                    )
                    return False

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False


# =============================================================================
# ALERT ROUTER
# =============================================================================


class AlertRouter:
    """Main alert routing service.

    Handles routing alerts to appropriate channels based on severity,
    with deduplication and rate limiting.
    """

    def __init__(self, config: AlertRoutingConfig):
        self.config = config
        self._providers: Dict[NotificationChannel, NotificationProvider] = {}
        self._dedup_cache: Dict[str, datetime] = {}
        self._hourly_count = 0
        self._hourly_reset: Optional[datetime] = None

        # Initialize providers
        self._providers[NotificationChannel.SLACK] = SlackNotificationProvider(config)
        self._providers[NotificationChannel.EMAIL] = EmailNotificationProvider(config)
        self._providers[NotificationChannel.WEBHOOK] = WebhookNotificationProvider(config)

    def _get_channels_for_severity(self, severity: str) -> List[NotificationChannel]:
        """Get notification channels based on severity."""
        channel_map = {
            "critical": self.config.critical_channels,
            "high": self.config.high_channels,
            "medium": self.config.medium_channels,
            "low": self.config.low_channels,
        }
        return channel_map.get(severity, [])

    def _should_deduplicate(self, alert: AlertPayload) -> bool:
        """Check if alert should be deduplicated (already sent recently)."""
        dedup_key = alert.get_dedup_key()
        now = datetime.now(timezone.utc)

        if dedup_key in self._dedup_cache:
            last_sent = self._dedup_cache[dedup_key]
            if now - last_sent < timedelta(seconds=self.config.min_interval_seconds):
                return True

        return False

    def _check_rate_limit(self) -> bool:
        """Check if within rate limit."""
        now = datetime.now(timezone.utc)

        # Reset hourly counter
        if self._hourly_reset is None or now - self._hourly_reset > timedelta(hours=1):
            self._hourly_reset = now
            self._hourly_count = 0

        return self._hourly_count < self.config.max_alerts_per_hour

    def _record_sent(self, alert: AlertPayload) -> None:
        """Record that an alert was sent."""
        dedup_key = alert.get_dedup_key()
        self._dedup_cache[dedup_key] = datetime.now(timezone.utc)
        self._hourly_count += 1

        # Clean old dedup entries
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        self._dedup_cache = {
            k: v for k, v in self._dedup_cache.items() if v > cutoff
        }

    async def route_alert(self, alert: AlertPayload) -> Dict[str, Any]:
        """Route an alert to appropriate channels.

        Args:
            alert: Alert payload

        Returns:
            Routing result with status for each channel
        """
        result = {
            "alert_id": alert.alert_id,
            "deduplicated": False,
            "rate_limited": False,
            "channels": {},
        }

        # Check deduplication
        if self._should_deduplicate(alert):
            logger.info(f"Alert {alert.alert_id} deduplicated")
            result["deduplicated"] = True
            return result

        # Check rate limit
        if not self._check_rate_limit():
            logger.warning(f"Alert {alert.alert_id} rate limited")
            result["rate_limited"] = True
            return result

        # Get channels for severity
        channels = self._get_channels_for_severity(alert.severity)

        if not channels:
            logger.info(f"No channels configured for severity {alert.severity}")
            return result

        # Send to each channel
        tasks = []
        for channel in channels:
            provider = self._providers.get(channel)
            if provider and provider.is_enabled():
                tasks.append((channel, provider.send(alert)))

        # Execute sends in parallel
        for channel, task in tasks:
            try:
                success = await task
                result["channels"][channel.value] = {
                    "sent": success,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            except Exception as e:
                result["channels"][channel.value] = {
                    "sent": False,
                    "error": str(e),
                }

        # Record successful send
        if any(ch.get("sent") for ch in result["channels"].values()):
            self._record_sent(alert)

        return result

    async def route_batch(self, alerts: List[AlertPayload]) -> List[Dict[str, Any]]:
        """Route multiple alerts.

        Args:
            alerts: List of alert payloads

        Returns:
            List of routing results
        """
        results = []
        for alert in alerts:
            result = await self.route_alert(alert)
            results.append(result)
        return results


# =============================================================================
# FACTORY
# =============================================================================


def get_alert_router(config: Optional[AlertRoutingConfig] = None) -> AlertRouter:
    """Get configured alert router instance.

    Args:
        config: Optional configuration. If not provided, loads from environment/config file.

    Returns:
        AlertRouter instance
    """
    if config is None:
        import os

        config = AlertRoutingConfig(
            # Email
            email_enabled=os.getenv("ALERT_EMAIL_ENABLED", "false").lower() == "true",
            smtp_host=os.getenv("SMTP_HOST", ""),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            smtp_username=os.getenv("SMTP_USERNAME", ""),
            smtp_password=os.getenv("SMTP_PASSWORD", ""),
            email_from=os.getenv("ALERT_EMAIL_FROM", "alerts@e2i-analytics.com"),
            email_recipients=os.getenv("ALERT_EMAIL_RECIPIENTS", "").split(","),
            # Slack
            slack_enabled=os.getenv("ALERT_SLACK_ENABLED", "false").lower() == "true",
            slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL", ""),
            slack_channel=os.getenv("SLACK_CHANNEL", "#ml-alerts"),
            # Webhook
            webhook_enabled=os.getenv("ALERT_WEBHOOK_ENABLED", "false").lower() == "true",
            webhook_url=os.getenv("ALERT_WEBHOOK_URL", ""),
        )

    return AlertRouter(config)


# =============================================================================
# INTEGRATION WITH TASKS
# =============================================================================


async def route_drift_alerts(
    model_version: str,
    drift_results: List[Dict[str, Any]],
    overall_score: float,
    summary: str,
    recommended_actions: List[str],
) -> List[Dict[str, Any]]:
    """Route drift detection results as alerts.

    Called from Celery tasks after drift detection completes.

    Args:
        model_version: Model that was checked
        drift_results: List of drift results
        overall_score: Overall drift score
        summary: Human-readable summary
        recommended_actions: Recommended actions

    Returns:
        List of routing results
    """
    router = get_alert_router()

    # Create alerts from drift results
    alerts = []
    for result in drift_results:
        if not result.get("drift_detected"):
            continue

        # Determine severity from result
        severity = result.get("severity", "medium")
        if severity == "none":
            continue

        alert = AlertPayload(
            alert_id=f"drift-{model_version}-{result.get('feature', 'unknown')}",
            model_id=model_version,
            alert_type=f"drift_{result.get('drift_type', 'data')}",
            severity=severity,
            title=f"Drift Detected: {result.get('feature', 'Unknown Feature')}",
            description=summary or f"Drift detected in {result.get('feature')}",
            triggered_at=datetime.now(timezone.utc),
            drift_score=result.get("test_statistic"),
            features_affected=[result.get("feature", "unknown")],
            recommended_actions=recommended_actions,
            metadata={
                "p_value": result.get("p_value"),
                "drift_type": result.get("drift_type"),
                "overall_score": overall_score,
            },
        )
        alerts.append(alert)

    if not alerts:
        return []

    return await router.route_batch(alerts)
