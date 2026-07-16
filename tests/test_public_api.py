"""Tests for release-level imports and built-in catalogs."""

from importlib.resources import files

from ascendmoea import list_algorithms, list_problems
from ascendmoea.algorithms import NSGAII
from ascendmoea.metrics import METRICS
from ascendmoea.operators import __all__ as operator_exports
from ascendmoea.problems import DTLZ2, MMF1, RWMOP2, SMOP1


def _resource_files(root):
    """Yield files recursively from an importlib resource tree."""
    for child in root.iterdir():
        if child.is_dir():
            yield from _resource_files(child)
        else:
            yield child


def test_catalogs_and_direct_imports_are_available() -> None:
    """Expose all major problem families and the complete algorithm catalog."""
    algorithm_names = list_algorithms()
    problem_names = list_problems()
    assert len(algorithm_names) == len(set(algorithm_names)) == 20
    assert "NSGAII" in algorithm_names
    assert len(problem_names) == len(set(problem_names)) == 155
    assert all(component is not None for component in (NSGAII, DTLZ2, MMF1, RWMOP2, SMOP1))


def test_public_operator_metric_and_resource_catalogs_are_complete() -> None:
    """Keep the documented operator, metric, and packaged-data counts stable."""
    assert len(operator_exports) == len(set(operator_exports)) == 19
    assert len(METRICS) == 12
    assert METRICS["runtime"] == (None, "min")
    resources = list(_resource_files(files("ascendmoea").joinpath("vendor")))
    assert sum(resource.name.endswith(".mat") for resource in resources) == 91
