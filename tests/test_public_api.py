"""Tests for release-level imports and built-in catalogs."""

from ascendmoea import list_algorithms, list_problems
from ascendmoea.algorithms import NSGAII
from ascendmoea.problems import DTLZ2, MMF1, RWMOP2, SMOP1


def test_catalogs_and_direct_imports_are_available() -> None:
    """Expose all major problem families and the complete algorithm catalog."""
    assert len(list_algorithms()) == 20
    assert "NSGAII" in list_algorithms()
    assert len(list_problems()) == 155
    assert all(component is not None for component in (NSGAII, DTLZ2, MMF1, RWMOP2, SMOP1))
