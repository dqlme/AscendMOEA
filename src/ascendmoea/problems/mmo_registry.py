"""Vectorized implementation of the mmo_registry problem."""

from .application import ApplicationProblem
from .idmpm2t1 import IDMPM2T1
from .idmpm2t1_e import IDMPM2T1E
from .idmpm2t2 import IDMPM2T2
from .idmpm2t2_e import IDMPM2T2E
from .idmpm2t3 import IDMPM2T3
from .idmpm2t3_e import IDMPM2T3E
from .idmpm2t4 import IDMPM2T4
from .idmpm2t4_e import IDMPM2T4E
from .idmpm3t1 import IDMPM3T1
from .idmpm3t1_e import IDMPM3T1E
from .idmpm3t2 import IDMPM3T2
from .idmpm3t2_e import IDMPM3T2E
from .idmpm3t3 import IDMPM3T3
from .idmpm3t3_e import IDMPM3T3E
from .idmpm3t4 import IDMPM3T4
from .idmpm3t4_e import IDMPM3T4E
from .idmpm4t1 import IDMPM4T1
from .idmpm4t2 import IDMPM4T2
from .idmpm4t3 import IDMPM4T3
from .idmpm4t4 import IDMPM4T4
from .map_problem import MapProblem
from .mmf1 import MMF1
from .mmf1_e import MMF1E
from .mmf1_z import MMF1Z
from .mmf2 import MMF2
from .mmf3 import MMF3
from .mmf4 import MMF4
from .mmf5 import MMF5
from .mmf6 import MMF6
from .mmf7 import MMF7
from .mmf8 import MMF8
from .mmf9 import MMF9
from .mmf10 import MMF10
from .mmf11 import MMF11
from .mmf12 import MMF12
from .mmf13 import MMF13
from .mmf14 import MMF14
from .mmf14_a import MMF14A
from .mmf15 import MMF15
from .mmf15_a import MMF15A
from .mmmop1a import MMMOP1A
from .mmmop1b import MMMOP1B
from .mmmop2a import MMMOP2A
from .mmmop2b import MMMOP2B
from .mmmop3a import MMMOP3A
from .mmmop3b import MMMOP3B
from .mmmop3c import MMMOP3C
from .mmmop3d import MMMOP3D
from .mmmop4a import MMMOP4A
from .mmmop4b import MMMOP4B
from .mmmop4c import MMMOP4C
from .mmmop4d import MMMOP4D
from .mmmop5a import MMMOP5A
from .mmmop5b import MMMOP5B
from .mmmop5c import MMMOP5C
from .mmmop5d import MMMOP5D
from .mmmop6a import MMMOP6A
from .mmmop6b import MMMOP6B
from .mmmop6c import MMMOP6C
from .mmmop6d import MMMOP6D
from .omni_test import OmniTest
from .poly_2 import Poly2
from .poly_3 import Poly3
from .poly_4 import Poly4
from .poly_5 import Poly5
from .poly_6 import Poly6
from .poly_8 import Poly8
from .poly_10 import Poly10
from .poly_12 import Poly12
from .rlp import RLP
from .sym_part_rotated import SYMPARTRotated
from .sym_part_simple import SYMPARTSimple
from .ueclp import UECLP

MMO_CLASSES = {
    "MMF1": MMF1,
    "MMF1_e": MMF1E,
    "MMF1_z": MMF1Z,
    "MMF2": MMF2,
    "MMF3": MMF3,
    "MMF4": MMF4,
    "MMF5": MMF5,
    "MMF6": MMF6,
    "MMF7": MMF7,
    "MMF8": MMF8,
    "MMF9": MMF9,
    "MMF10": MMF10,
    "MMF11": MMF11,
    "MMF12": MMF12,
    "MMF13": MMF13,
    "MMF14": MMF14,
    "MMF14_a": MMF14A,
    "MMF15": MMF15,
    "MMF15_a": MMF15A,
    "SYM_PART_simple": SYMPARTSimple,
    "SYM_PART_rotated": SYMPARTRotated,
    "Omni_test": OmniTest,
    "IDMPM2T1": IDMPM2T1,
    "IDMPM2T2": IDMPM2T2,
    "IDMPM2T3": IDMPM2T3,
    "IDMPM2T4": IDMPM2T4,
    "IDMPM3T1": IDMPM3T1,
    "IDMPM3T2": IDMPM3T2,
    "IDMPM3T3": IDMPM3T3,
    "IDMPM3T4": IDMPM3T4,
    "IDMPM4T1": IDMPM4T1,
    "IDMPM4T2": IDMPM4T2,
    "IDMPM4T3": IDMPM4T3,
    "IDMPM4T4": IDMPM4T4,
    "IDMPM2T1_e": IDMPM2T1E,
    "IDMPM2T2_e": IDMPM2T2E,
    "IDMPM2T3_e": IDMPM2T3E,
    "IDMPM2T4_e": IDMPM2T4E,
    "IDMPM3T1_e": IDMPM3T1E,
    "IDMPM3T2_e": IDMPM3T2E,
    "IDMPM3T3_e": IDMPM3T3E,
    "IDMPM3T4_e": IDMPM3T4E,
    "MMMOP1A": MMMOP1A,
    "MMMOP1B": MMMOP1B,
    "MMMOP2A": MMMOP2A,
    "MMMOP2B": MMMOP2B,
    "MMMOP3A": MMMOP3A,
    "MMMOP3B": MMMOP3B,
    "MMMOP3C": MMMOP3C,
    "MMMOP3D": MMMOP3D,
    "MMMOP4A": MMMOP4A,
    "MMMOP4B": MMMOP4B,
    "MMMOP4C": MMMOP4C,
    "MMMOP4D": MMMOP4D,
    "MMMOP5A": MMMOP5A,
    "MMMOP5B": MMMOP5B,
    "MMMOP5C": MMMOP5C,
    "MMMOP5D": MMMOP5D,
    "MMMOP6A": MMMOP6A,
    "MMMOP6B": MMMOP6B,
    "MMMOP6C": MMMOP6C,
    "MMMOP6D": MMMOP6D,
    "poly_2": Poly2,
    "poly_3": Poly3,
    "poly_4": Poly4,
    "poly_5": Poly5,
    "poly_6": Poly6,
    "poly_8": Poly8,
    "poly_10": Poly10,
    "poly_12": Poly12,
    "map": MapProblem,
    "application": ApplicationProblem,
    "RLP": RLP,
    "UECLP": UECLP,
}
