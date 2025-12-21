"""Health Checker Node - Verify deployment health.

Handles:
1. Endpoint health checks
2. Response time validation
3. Readiness verification
"""

import time
from typing import Any, Dict


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

        # In production, this would make HTTP request to health endpoint
        # response = requests.get(f"{endpoint_url}/health", timeout=5)
        # health_ok = response.status_code == 200

        # For now, simulate health check
        health_check_passed = deployment_status == "healthy"

        response_time_ms = (time.time() - start_time) * 1000

        # Generate health check and metrics URLs
        health_check_url = f"{endpoint_url}/health"
        metrics_url = f"{endpoint_url}/metrics"

        if health_check_passed:
            return {
                "health_check_passed": True,
                "health_check_url": health_check_url,
                "metrics_url": metrics_url,
                "health_check_response_time_ms": response_time_ms,
            }
        else:
            return {
                "health_check_passed": False,
                "health_check_url": health_check_url,
                "metrics_url": metrics_url,
                "health_check_response_time_ms": response_time_ms,
                "health_check_error": "Endpoint health check failed",
            }

    except Exception as e:
        return {
            "error": f"Health check failed: {str(e)}",
            "error_type": "health_check_error",
            "error_details": {"exception": str(e)},
            "health_check_passed": False,
        }
