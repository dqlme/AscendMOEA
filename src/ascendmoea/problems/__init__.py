"""Benchmark, constrained, multimodal, sparse, and real-world problems."""

from __future__ import annotations

import keyword

from ..registry import ComponentRegistry
from .cf import CF_CLASSES
from .dascmop import DAS_CLASSES
from .lircmop import LIR_CLASSES
from .mmo_registry import MMO_CLASSES
from .real_world_registry import REAL_WORLD_CLASSES
from .sparse_registry import SPARSE_CLASSES
from .standard_registry import STANDARD_CLASSES

PROBLEMS = ComponentRegistry("problem")
for _mapping in (
    CF_CLASSES,
    DAS_CLASSES,
    LIR_CLASSES,
    MMO_CLASSES,
    SPARSE_CLASSES,
    REAL_WORLD_CLASSES,
    STANDARD_CLASSES,
):
    for _name, _component in _mapping.items():
        PROBLEMS.register(_component, name=_name)

register_problem = PROBLEMS.register


def get_problem(name: str):
    """Return a problem class by its registered name."""
    return PROBLEMS[name]


def create_problem(name: str, /, **kwargs):
    """Construct a registered problem with the supplied keyword arguments."""
    return PROBLEMS.create(name, **kwargs)


def list_problems() -> tuple[str, ...]:
    """Return all built-in and user-registered problem names."""
    return PROBLEMS.names()


_exported_problem_names: set[str] = set()
for _name, _component in PROBLEMS.items():
    for _export_name in (_name, _component.__name__):
        if _export_name.isidentifier() and not keyword.iskeyword(_export_name):
            globals().setdefault(_export_name, _component)
            _exported_problem_names.add(_export_name)

__all__ = [
    "PROBLEMS",
    "create_problem",
    "get_problem",
    "list_problems",
    "register_problem",
    *sorted(_exported_problem_names),
]

del _component, _export_name, _mapping, _name
