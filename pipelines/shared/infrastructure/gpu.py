from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GpuInfo:
    index: int
    free_mib: int
    total_mib: int


def list_gpus() -> list[GpuInfo]:
    """Return GPUs ordered by free memory descending."""
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("Could not query GPUs: %s", exc)
        return []

    gpus: list[GpuInfo] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            gpus.append(
                GpuInfo(
                    index=int(parts[0]),
                    free_mib=int(float(parts[1])),
                    total_mib=int(float(parts[2])),
                )
            )
        except ValueError:
            continue
    gpus.sort(key=lambda g: g.free_mib, reverse=True)
    return gpus


def freest_device() -> str | int:
    """Return CUDA device index with most free VRAM, or 'cpu'."""
    gpus = list_gpus()
    if not gpus:
        logger.info("No NVIDIA GPU found; using CPU")
        return "cpu"
    best = gpus[0]
    logger.info(
        "Using GPU %s (free=%s MiB / total=%s MiB)",
        best.index,
        best.free_mib,
        best.total_mib,
    )
    return best.index


def is_cuda_oom(error: BaseException) -> bool:
    message = str(error).lower()
    patterns = (
        r"out of memory",
        r"cuda.*oom",
        r"cudnn_status_alloc_failed",
    )
    return any(re.search(p, message) for p in patterns)
