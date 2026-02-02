# utils/cpu_limiter.py
import multiprocessing
import os
import platform

import psutil

from config import get_config
from logging_config import get_logger

logger = get_logger(__name__)


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
            logger.warning("CPU limit not applied (<=0).")
            return
        if platform.system() in ("Linux", "Windows"):
            p = psutil.Process(os.getpid())
            available_cpus = list(range(multiprocessing.cpu_count()))
            allowed_cpus = available_cpus[:cpu_limit]
            p.cpu_affinity(allowed_cpus)
            logger.info(f"Restricted process to CPUs: {allowed_cpus}")
        else:
            logger.debug(
                f"CPU affinity is not supported on {platform.system()}. Skipping restriction."
            )
    except Exception as e:
        logger.warning(f"Could not set CPU affinity: {e}")
