"""Vectorized implementation of the standard_registry problem."""

from .dtlz1 import DTLZ1
from .dtlz2 import DTLZ2
from .dtlz3 import DTLZ3
from .dtlz4 import DTLZ4
from .dtlz5 import DTLZ5
from .dtlz6 import DTLZ6
from .dtlz7 import DTLZ7
from .zdt1 import ZDT1
from .zdt2 import ZDT2
from .zdt3 import ZDT3
from .zdt4 import ZDT4
from .zdt5 import ZDT5
from .zdt6 import ZDT6

STANDARD_CLASSES = {
    cls.name: cls
    for cls in (
        ZDT1,
        ZDT2,
        ZDT3,
        ZDT4,
        ZDT5,
        ZDT6,
        DTLZ1,
        DTLZ2,
        DTLZ3,
        DTLZ4,
        DTLZ5,
        DTLZ6,
        DTLZ7,
    )
}
