"""Built-in multi-objective evolutionary algorithms and extension helpers."""

from ..registry import ComponentRegistry
from .bico import BiCo
from .ccmo import CCMO
from .cmegl import CMEGL
from .cmmo import CMMO
from .cmoeacd import CMOEACD
from .cmoead import CMOEAD
from .cmoeams import CMOEAMS
from .commea import CoMMEA
from .emcmo import EMCMO
from .hrea import HREA
from .lmea import LMEA
from .moea_psl import MOEAPSL
from .moead import MOEAD
from .nsga2 import NSGAII
from .sparse_ea import SparseEA
from .sparse_ea2 import SparseEA2
from .spea2 import SPEA2
from .ts_sparse_ea import TSSparseEA

ALGORITHMS = ComponentRegistry(
    "algorithm",
    (
        NSGAII,
        MOEAD,
        SPEA2,
        BiCo,
        CMOEAD,
        CCMO,
        CMEGL,
        CMOEACD,
        CMOEAMS,
        EMCMO,
        HREA,
        CMMO,
        CoMMEA,
        SparseEA,
        SparseEA2,
        TSSparseEA,
        LMEA,
        MOEAPSL,
    ),
)

register_algorithm = ALGORITHMS.register


def get_algorithm(name: str):
    """Return an algorithm class by its registered name."""
    return ALGORITHMS[name]


def create_algorithm(name: str, /, **kwargs):
    """Construct a registered algorithm with the supplied keyword arguments."""
    return ALGORITHMS.create(name, **kwargs)


def list_algorithms() -> tuple[str, ...]:
    """Return all built-in and user-registered algorithm names."""
    return ALGORITHMS.names()


__all__ = [
    "ALGORITHMS",
    "BiCo",
    "CCMO",
    "CMEGL",
    "CMMO",
    "CMOEACD",
    "CMOEAD",
    "CMOEAMS",
    "CoMMEA",
    "EMCMO",
    "HREA",
    "LMEA",
    "MOEAD",
    "MOEAPSL",
    "NSGAII",
    "SPEA2",
    "SparseEA",
    "SparseEA2",
    "TSSparseEA",
    "create_algorithm",
    "get_algorithm",
    "list_algorithms",
    "register_algorithm",
]
