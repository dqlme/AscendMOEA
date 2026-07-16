"""Vectorized implementation of the real_world_registry problem."""

from .monrp import MONRP
from .motsp import MOTSP
from .mpdmp import MPDMP
from .mqap import MQAP
from .rwmop1 import RWMOP1
from .rwmop2 import RWMOP2
from .rwmop3 import RWMOP3
from .rwmop4 import RWMOP4
from .rwmop5 import RWMOP5
from .rwmop6 import RWMOP6
from .rwmop7 import RWMOP7
from .rwmop8 import RWMOP8
from .rwmop9 import RWMOP9
from .rwmop10 import RWMOP10
from .sparse_kp import SparseKP
from .sparse_pm import SparsePM
from .sparse_po import SparsePO
from .sparse_sr import SparseSR

REAL_WORLD_CLASSES = {
    "RWMOP1": RWMOP1,
    "RWMOP2": RWMOP2,
    "RWMOP3": RWMOP3,
    "RWMOP4": RWMOP4,
    "RWMOP5": RWMOP5,
    "RWMOP6": RWMOP6,
    "RWMOP7": RWMOP7,
    "RWMOP8": RWMOP8,
    "RWMOP9": RWMOP9,
    "RWMOP10": RWMOP10,
    "Sparse_KP": SparseKP,
    "Sparse_PM": SparsePM,
    "Sparse_PO": SparsePO,
    "Sparse_SR": SparseSR,
    "MOTSP": MOTSP,
    "mQAP": MQAP,
    "MONRP": MONRP,
    "MPDMP": MPDMP,
}
