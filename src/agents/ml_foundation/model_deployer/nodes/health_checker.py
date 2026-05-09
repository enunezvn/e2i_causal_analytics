"""Health Checker Node - Verify deployment health.

Handles:
1. Endpoint health checks
2. Response time validation
3. Readiness verification
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import HTTP clients with graceful fallback
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

HTTP_CLIENT_AVAILABLE = HTTPX_AVAILABLE or AIOHTTP_AVAILABLE
if not HTTP_CLIENT_AVAILABLE:
    logger.warning("No HTTP client available (httpx/aiohttp), using simulated health checks")


async def _check_health_httpx(
    url: str, timeout: float = 5.0
) -> Tuple[bool, Optional[int], Optional[str]]:
    """Perform health check using httpx.

    Args:
        url: Health check URL
        timeout: Request timeout in seconds

    Returns:
        Tuple of (success, status_code, error_message)
    """
    if not HTTPX_AVAILABLE:
        return False, None, "httpx not available"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            return response.status_code == 200, response.status_code, None
    except httpx.TimeoutException:
        return False, None, "Health check timed out"
    except httpx.ConnectError:
        return False, None, "Failed to connect to endpoint"
    except Exception as e:
        return False, None, str(e)


async def _check_health_aiohttp(
    url: str, timeout: float = 5.0
) -> Tuple[bool, Optional[int], Optional[str]]:
    """Perform health check using aiohttp.

    Args:
        url: Health check URL
        timeout: Request timeout in seconds

    Returns:
        Tuple of (success, status_code, error_message)
    """
    if not AIOHTTP_AVAILABLE:
        return False, None, "aiohttp not available"

    try:
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_obj) as session:
            async with session.get(url) as response:
                return response.status == 200, response.status, None
    except TimeoutError:
        return False, None, "Health check timed out"
    except aiohttp.ClientConnectorError:
        return False, None, "Failed to connect to endpoint"
    except Exception as e:
        return False, None, str(e)


async def _perform_http_health_check(
    url: str, timeout: float = 5.0
) -> Tuple[bool, Optional[int], Optional[str]]:
    """Perform HTTP health check using available client.

    Args:
        url: Health check URL
        timeout: Request timeout in seconds

    Returns:
        Tuple of (success, status_code, error_message)
    """
    # Prefer httpx for better async support
    if HTTPX_AVAILABLE:
        return await _check_health_httpx(url, timeout)
    elif AIOHTTP_AVAILABLE:
        return await _check_health_aiohttp(url, timeout)
    else:
        return False, None, "No HTTP client available"


async def check_health(state: Dict[str, Any]) -> Dict[str, Any]:
    """Perform health check on deployed endpoint.

    Args:
        state: Current agent state with endpoint_url

    Returns:
        State updates with health check results
    """
    start_time = time.time()

    try:
        endpoint_url = state.get("endpoint_url")
        deployment_status = state.get("deployment_status", "pending")

        if not endpoint_url:
            return {
                "error": "Missing endpoint_url for health check",
                "error_type": "missing_endpoint_url",
                "health_check_passed": False,
            }

        # Skip health check if deployment failed
        if deployment_status == "failed":
            return {
                "health_check_passed": False,
                "health_check_error": "Deployment failed, skipping health check",
            }

        # Generate health check and metrics URLs
        health_check_url = f"{endpoint_url}/health"
        metrics_url = f"{endpoint_url}/metrics"

        # Try real HTTP health check first
        http_check_performed = False
        http_status_code = None
        http_error = None

        if HTTP_CLIENT_AVAILABLE and endpoint_url.startswith(("http://", "https://")):
            http_check_performed = True
            health_check_passed, http_status_code, http_error = await _perform_http_health_check(
                health_check_url, timeout=5.0
            )
            if http_error:
                logger.warning(f"HTTP health check failed: {http_error}")
        else:
            # Fallback to simulation based on deployment status
            logger.info("Using simulated health check (no HTTP client or invalid URL)")
            health_check_passed = deployment_status == "healthy"

        response_time_ms = (time.time() - start_time) * 1000

        result = {
            "health_check_passed": health_check_passed,
            "health_check_url": health_check_url,
            "metrics_url": metrics_url,
            "health_check_response_time_ms": response_time_ms,
            "http_check_performed": http_check_performed,
        }

        if http_status_code is not None:
            result["http_status_code"] = http_status_code

        if not health_check_passed:
            result["health_check_error"] = http_error or "Endpoint health check failed"

        return result

    except Exception as e:
        return {
            "error": f"Health check failed: {str(e)}",
            "error_type": "health_check_error",
            "error_details": {"exception": str(e)},
            "health_check_passed": False,
        }
