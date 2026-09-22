"""Cost accounting (section 21). INR = cpu_hours * rate + gpu_hours * rate. Rates are ASSUMPTIONS: set via env to
your actual node price (e.g. instance INR/hour / vCPUs)."""
from __future__ import annotations
import os, resource
from dataclasses import dataclass, asdict
from typing import Optional

CPU_INR_PER_HOUR = float(os.environ.get("COST_INR_PER_CPU_HOUR", "3.0"))
GPU_INR_PER_HOUR = float(os.environ.get("COST_INR_PER_GPU_HOUR", "60.0"))


def cpu_seconds() -> float:
    a, b = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime   # children = ffmpeg subprocesses


@dataclass
class ProcessingCost:
    cpuSeconds: float
    gpuSeconds: float
    model: str
    inputDurationSeconds: float
    outputDurationSeconds: float
    outputs: int = 1
    inr: float = 0.0
    inrPerSourceHour: float = 0.0
    inrPerReel: float = 0.0

    def finalize(self) -> "ProcessingCost":
        self.inr = self.cpuSeconds / 3600 * CPU_INR_PER_HOUR + self.gpuSeconds / 3600 * GPU_INR_PER_HOUR
        self.inrPerSourceHour = self.inr / (self.inputDurationSeconds / 3600) if self.inputDurationSeconds else 0.0
        self.inrPerReel = self.inr / max(1, self.outputs)
        for k in ("cpuSeconds", "inr", "inrPerSourceHour", "inrPerReel"):
            setattr(self, k, round(getattr(self, k), 4))
        return self

    def dict(self):
        return asdict(self)
