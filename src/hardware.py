"""Thread caps and a small hardware snapshot."""

from __future__ import annotations

import os
import platform

import psutil


def limit_threads(n: int = 4) -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(n)


def snapshot(device: str) -> dict:
    memory = psutil.virtual_memory()
    cpu = platform.processor() or platform.machine()
    gpu = None
    if device == "cuda":
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    return {
        "cpu": cpu,
        "ram_gb": round(memory.total / (1024**3), 2),
        "gpu": gpu,
        "python": platform.python_version(),
        "os": platform.platform(),
        "peak_rss_gb": None,
    }


def peak_rss_gb() -> float:
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)
    if peak is None:
        peak = info.rss
    return float(peak) / (1024**3)
