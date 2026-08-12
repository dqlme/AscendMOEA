"""Ascend-native tensorized multi-objective evolutionary optimization."""

from __future__ import annotations

import os as _os

_os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

from .algorithms import (
    ALGORITHMS,
    create_algorithm,
    get_algorithm,
    list_algorithms,
    register_algorithm,
)
from .core import Algorithm, Population, Problem
from .device import (
    configure_torch,
    npu_is_available,
    seed_everything,
    select_device,
    synchronize,
)
from .problems import (
    PROBLEMS,
    create_problem,
    get_problem,
    list_problems,
    register_problem,
)
from .registry import ComponentRegistry
from .workflow import (
    GenerationRecord,
    HistoryMonitor,
    Monitor,
    OptimizationResult,
    RunConfig,
    Workflow,
    optimize,
)

__version__ = "1.0.0"

__all__ = [
    "ALGORITHMS",
    "PROBLEMS",
    "Algorithm",
    "ComponentRegistry",
    "GenerationRecord",
    "HistoryMonitor",
    "Monitor",
    "OptimizationResult",
    "Population",
    "Problem",
    "RunConfig",
    "Workflow",
    "__version__",
    "configure_torch",
    "create_algorithm",
    "create_problem",
    "get_algorithm",
    "get_problem",
    "list_algorithms",
    "list_problems",
    "npu_is_available",
    "optimize",
    "register_algorithm",
    "register_problem",
    "seed_everything",
    "select_device",
    "synchronize",
]

del _os
