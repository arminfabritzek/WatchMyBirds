"""
Health Service - Web Layer Service for System Health.

Exposes system health status to the web interface.
"""

from core import health_core


def get_system_health() -> dict:
    """
    Get current system health status.

    Returns:
        Dictionary with system health metrics.
    """
    return health_core.get_system_health()
