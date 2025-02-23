# utils/cpu_limiter.py
import os
import psutil
import multiprocessing
import platform

def restrict_to_cpus(cpu_limit=None):
    """
    Restrict the process to only use the first `cpu_limit` CPUs.
    If no limit is provided, it uses the environment variable `CPU_LIMIT`.
    """
    try:
        if cpu_limit is None:
            cpu_limit = int(float(os.getenv("CPU_LIMIT", 2)))  # Safe conversion
        if platform.system() == "Linux" or platform.system() == "Windows":  # ✅ Works on Linux & Windows
            p = psutil.Process(os.getpid())
            available_cpus = list(range(multiprocessing.cpu_count()))
            allowed_cpus = available_cpus[:cpu_limit]  # Restrict to first N CPUs
            p.cpu_affinity(allowed_cpus)  # Set CPU affinity
            print(f"🔧 Restricted process to CPUs: {allowed_cpus}")
        else:
            print(f"⚠️ CPU affinity is not supported on {platform.system()}. Skipping restriction.")
    except Exception as e:
        print(f"⚠️ Could not set CPU affinity: {e}")