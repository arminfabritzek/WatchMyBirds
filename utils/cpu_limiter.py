# utils/cpu_limiter.py
import multiprocessing
import os
import platform

import psutil

from config import get_config
from logging_config import get_logger

logger = get_logger(__name__)


def _available_cpu_ids(process: psutil.Process) -> list[int]:
    """
    Return CPU ids currently available to this process.

    On Linux/Windows with affinity support, this reflects cgroup/cpuset limits.
    Falls back to [0..cpu_count-1] if affinity introspection fails.
    """
    try:
        current = process.cpu_affinity()
        if current:
            return sorted(int(cpu) for cpu in current)
    except Exception:
        pass
    return list(range(multiprocessing.cpu_count()))


def restrict_to_cpus(cpu_limit=None):
    """
    Restrict the process to only use the first `cpu_limit` CPUs.
    If no limit is provided, it uses the shared config `CPU_LIMIT`.
    """
    try:
        cfg = get_config()
        if cpu_limit is None:
            cpu_limit = int(cfg.get("CPU_LIMIT", 2))
        if cpu_limit <= 0:
            logger.info("CPU affinity disabled (CPU_LIMIT <= 0).")
            return
        if platform.system() in ("Linux", "Windows"):
            p = psutil.Process(os.getpid())
            available_cpus = _available_cpu_ids(p)
            if not available_cpus:
                logger.warning("Could not determine available CPUs for affinity.")
                return
            allowed_cpus = available_cpus[: min(cpu_limit, len(available_cpus))]
            p.cpu_affinity(allowed_cpus)
            logger.info(
                "Restricted process to CPUs: %s (requested=%s, available=%s)",
                allowed_cpus,
                cpu_limit,
                available_cpus,
            )
        else:
            logger.debug(
                f"CPU affinity is not supported on {platform.system()}. Skipping restriction."
            )
    except Exception as e:
        logger.warning(f"Could not set CPU affinity: {e}")
