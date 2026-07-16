"""Vectorized implementation of the sparse_registry problem."""

from .lsmop1 import LSMOP1
from .lsmop2 import LSMOP2
from .lsmop3 import LSMOP3
from .lsmop4 import LSMOP4
from .lsmop5 import LSMOP5
from .lsmop6 import LSMOP6
from .lsmop7 import LSMOP7
from .lsmop8 import LSMOP8
from .lsmop9 import LSMOP9
from .smop1 import SMOP1
from .smop2 import SMOP2
from .smop3 import SMOP3
from .smop4 import SMOP4
from .smop5 import SMOP5
from .smop6 import SMOP6
from .smop7 import SMOP7
from .smop8 import SMOP8

SPARSE_CLASSES = {
    "SMOP1": SMOP1,
    "SMOP2": SMOP2,
    "SMOP3": SMOP3,
    "SMOP4": SMOP4,
    "SMOP5": SMOP5,
    "SMOP6": SMOP6,
    "SMOP7": SMOP7,
    "SMOP8": SMOP8,
    "LSMOP1": LSMOP1,
    "LSMOP2": LSMOP2,
    "LSMOP3": LSMOP3,
    "LSMOP4": LSMOP4,
    "LSMOP5": LSMOP5,
    "LSMOP6": LSMOP6,
    "LSMOP7": LSMOP7,
    "LSMOP8": LSMOP8,
    "LSMOP9": LSMOP9,
}
