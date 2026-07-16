"""Implementation utilities for load_functions."""

import os

import numpy as np
import scipy
import torch

from ..TestFunctions import APP, CEC2019, IDMP, MMMOP, IDMP_e, Poly

_TEST_SUITES = {
    "APP": APP,
    "CEC2019": CEC2019,
    "IDMP": IDMP,
    "IDMP_e": IDMP_e,
    "MMMOP": MMMOP,
    "Poly": Poly,
}


def load_functions(function_num):
    """Load functions."""
    refpoint = None
    PS = None
    PF = None

    if function_num == 1:
        function_name = "MMF1"
        n_var = 2
        n_obj = 2
        xl = torch.tensor([[1, -1]])
        xu = torch.tensor([[3, 1]])
        refpoint = torch.tensor([[2, 2]])
        mat_filename = "MMF1truePSPF.mat"
        mat_folder = "CEC2019"
        load_PF = "PF"
        load_PS = "PS"
    elif function_num == 2:
        function_name = "MMF1_e"
        n_var = 2
        n_obj = 2
        xl = torch.tensor([[1, -20]])
        xu = torch.tensor([[3, 20]])
        refpoint = torch.tensor([[1.1, 1.1]])
        mat_filename = "MMF1_e_Reference_PSPF_data.mat"
        mat_folder = "CEC2019"
        load_PF = "PF"
        load_PS = "PS"
    elif function_num == 3:
        function_name = "MMF1_z"
        n_var = 2
        n_obj = 2
        xl = torch.tensor([[1, -1]])
        xu = torch.tensor([[3, 1]])
        refpoint = torch.tensor([[1.1, 1.1]])
        mat_filename = "MMF1_z_Reference_PSPF_data.mat"
        mat_folder = "CEC2019"
        load_PF = "PF"
        load_PS = "PS"
    elif function_num == 4:
        function_name = "MMF2"
        n_var = 2
        n_obj = 2
        xl = torch.tensor([[0, 0]])
        xu = torch.tensor([[1, 2]])
        refpoint = torch.tensor([[2, 2]])
        mat_filename = "MMF2truePSPF.mat"
        mat_folder = "CEC2019"
        load_PF = "PF"
        load_PS = "PS"
    elif function_num == 5:
        function_name = "MMF3"
        n_var = 2
        n_obj = 2
        xl = torch.tensor([[0, 0]])
        xu = torch.tensor([[1, 1.5]])
        refpoint = torch.tensor([[2, 2]])
        mat_filename = "MMF3truePSPF.mat"
        mat_folder = "CEC2019"
        load_PF = "PF"
        load_PS = "PS"
    elif function_num == 6:
        function_name = "MMF4"
        n_var = 2
        n_obj = 2
        xl = torch.tensor([[-1, 0]])
        xu = torch.tensor([[1, 2]])
        refpoint = torch.tensor([[2, 2]])
        mat_filename = "MMF4truePSPF.mat"
        mat_folder = "CEC2019"
        load_PF = "PF"
        load_PS = "PS"
    elif function_num == 7:
        function_name = "MMF5"
        n_var = 2
        n_obj = 2
        xl = torch.tensor([[1, -1]])
        xu = torch.tensor([[3, 3]])
        refpoint = torch.tensor([[2, 2]])
        mat_filename = "MMF5truePSPF.mat"
        mat_folder = "CEC2019"
        load_PF = "PF"
        load_PS = "PS"
    elif function_num == 8:
        function_name = "MMF6"
        n_var = 2
        n_obj = 2
        xl = torch.tensor([[1, -1]])
        xu = torch.tensor([[3, 2]])
        refpoint = torch.tensor([[2, 2]])
        mat_filename = "MMF6truePSPF.mat"
        mat_folder = "CEC2019"
        load_PF = "PF"
        load_PS = "PS"
    elif function_num == 9:
        function_name = "MMF7"
        n_var = 2
        n_obj = 2
        xl = torch.tensor([[1, -1]])
        xu = torch.tensor([[3, 1]])
        refpoint = torch.tensor([[2, 2]])
        mat_filename = "MMF7truePSPF.mat"
        mat_folder = "CEC2019"
        load_PF = "PF"
        load_PS = "PS"
    elif function_num == 10:
        function_name = "MMF8"
        n_var = 2
        n_obj = 2
        xl = torch.tensor([[-np.pi, 0]])
        xu = torch.tensor([[np.pi, 9]])
        refpoint = torch.tensor([[2, 2]])
        mat_filename = "MMF8truePSPF.mat"
        mat_folder = "CEC2019"
        load_PF = "PF"
        load_PS = "PS"
    elif function_num == 11:
        function_name = "MMF9"
        n_var = 2
        n_obj = 2
        xl = torch.tensor([[0.1, 0.1]])
        xu = torch.tensor([[1.1, 1.1]])
        refpoint = torch.tensor([[1.21, 11]])
        mat_filename = "MMF9_Reference_PSPF_data.mat"
        mat_folder = "CEC2019"
        load_PF = "PF"
        load_PS = "PS"
    elif function_num == 12:
        function_name = "MMF10"
        n_var = 2
        n_obj = 2
        xl = torch.tensor([[0.1, 0.1]])
        xu = torch.tensor([[1.1, 1.1]])
        refpoint = torch.tensor([[1.21, 13.2]])
        mat_filename = "MMF10_Reference_PSPF_data.mat"
        mat_folder = "CEC2019"
        load_PF = "PF"
        load_PS = "PS"
    elif function_num == 13:
        function_name = "MMF11"
        n_var = 2
        n_obj = 2
        xl = torch.tensor([[0.1, 0.1]])
        xu = torch.tensor([[1.1, 1.1]])
        refpoint = torch.tensor([[1.21, 15.4]])
        mat_filename = "MMF11_Reference_PSPF_data.mat"
        mat_folder = "CEC2019"
        load_PF = "PF"
        load_PS = "PS"
    elif function_num == 14:
        function_name = "MMF12"
        n_var = 2
        n_obj = 2
        xl = torch.tensor([[0, 0]])
        xu = torch.tensor([[1, 1]])
        refpoint = torch.tensor([[1.54, 1.1]])
        mat_filename = "MMF12_Reference_PSPF_data.mat"
        mat_folder = "CEC2019"
        load_PF = "PF"
        load_PS = "PS"
    elif function_num == 15:
        function_name = "MMF13"
        n_var = 3
        n_obj = 2
        xl = torch.tensor([[0.1, 0.1, 0.1]])
        xu = torch.tensor([[1.1, 1.1, 1.1]])
        refpoint = torch.tensor([[1.54, 15.4]])
        mat_filename = "MMF13_Reference_PSPF_data.mat"
        mat_folder = "CEC2019"
        load_PF = "PF"
        load_PS = "PS"
    elif function_num == 16:
        function_name = "MMF14"
        n_var = 3
        n_obj = 3
        xl = torch.tensor([[0, 0, 0]])
        xu = torch.tensor([[1, 1, 1]])
        refpoint = torch.tensor([[2.2, 2.2, 2.2]])
        mat_filename = "MMF14_Reference_PSPF_data.mat"
        mat_folder = "CEC2019"
        load_PF = "PF"
        load_PS = "PS"
    elif function_num == 17:
        function_name = "MMF14_a"
        n_var = 3
        n_obj = 3
        xl = torch.tensor([[0, 0, 0]])
        xu = torch.tensor([[1, 1, 1]])
        refpoint = torch.tensor([[2.2, 2.2, 2.2]])
        mat_filename = "MMF14_a_Reference_PSPF_data.mat"
        mat_folder = "CEC2019"
        load_PF = "PF"
        load_PS = "PS"
    elif function_num == 18:
        function_name = "MMF15"
        n_var = 3
        n_obj = 3
        xl = torch.tensor([[0, 0, 0]])
        xu = torch.tensor([[1, 1, 1]])
        refpoint = torch.tensor([[2.5, 2.5, 2.5]])
        mat_filename = "MMF15_Reference_PSPF_data.mat"
        mat_folder = "CEC2019"
        load_PF = "PF"
        load_PS = "PS"
    elif function_num == 19:
        function_name = "MMF15_a"
        n_var = 3
        n_obj = 3
        xl = torch.tensor([[0, 0, 0]])
        xu = torch.tensor([[1, 1, 1]])
        refpoint = torch.tensor([[2.5, 2.5, 2.5]])
        mat_filename = "MMF15_a_Reference_PSPF_data.mat"
        mat_folder = "CEC2019"
        load_PF = "PF"
        load_PS = "PS"
    elif function_num == 20:
        function_name = "SYM_PART_simple"
        n_var = 2
        n_obj = 2
        xl = torch.tensor([[-20, -20]])
        xu = torch.tensor([[20, 20]])
        refpoint = torch.tensor([[2, 2]])
        mat_filename = "SYM_PART_simple_turePSPF.mat"
        mat_folder = "CEC2019"
        load_PF = "PF"
        load_PS = "PS"
    elif function_num == 21:
        function_name = "SYM_PART_rotated"
        n_var = 2
        n_obj = 2
        xl = torch.tensor([[-20, -20]])
        xu = torch.tensor([[20, 20]])
        refpoint = torch.tensor([[2, 2]])
        mat_filename = "SYM_PART_rotatedtruePSPF.mat"
        mat_folder = "CEC2019"
        load_PF = "PF"
        load_PS = "PS"
    elif function_num == 22:
        function_name = "Omni_test"
        n_var = 3
        n_obj = 2
        xl = torch.tensor([[0, 0, 0]])
        xu = torch.tensor([[6, 6, 6]])
        refpoint = torch.tensor([[5, 5]])
        mat_filename = "Omni_testtruePSPF.mat"
        mat_folder = "CEC2019"
        load_PF = "PF"
        load_PS = "PS"
    elif function_num == 23:
        function_name = "IDMPM2T1"
        n_var = 2
        n_obj = 2
        xl = torch.tensor([[-1, -1]])
        xu = torch.tensor([[1, 1]])
        refpoint = torch.tensor([[1, 1]])
        mat_filename = "IDMPM2T1Reference.mat"
        mat_folder = "IDMP"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 24:
        function_name = "IDMPM2T2"
        n_var = 2
        n_obj = 2
        xl = torch.tensor([[-1, -1]])
        xu = torch.tensor([[1, 1]])
        refpoint = torch.tensor([[1, 1]])
        mat_filename = "IDMPM2T2Reference.mat"
        mat_folder = "IDMP"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 25:
        function_name = "IDMPM2T3"
        n_var = 2
        n_obj = 2
        xl = torch.tensor([[-1, -1]])
        xu = torch.tensor([[1, 1]])
        refpoint = torch.tensor([[1, 1]])
        mat_filename = "IDMPM2T3Reference.mat"
        mat_folder = "IDMP"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 26:
        function_name = "IDMPM2T4"
        n_var = 2
        n_obj = 2
        xl = torch.tensor([[-1, -1]])
        xu = torch.tensor([[1, 1]])
        refpoint = torch.tensor([[1, 1]])
        mat_filename = "IDMPM2T4Reference.mat"
        mat_folder = "IDMP"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 27:
        function_name = "IDMPM3T1"
        n_var = 3
        n_obj = 3
        xl = torch.tensor([[-1, -1, -1]])
        xu = torch.tensor([[1, 1, 1]])
        refpoint = torch.tensor([[1, 1, 1]])
        mat_filename = "IDMPM3T1Reference.mat"
        mat_folder = "IDMP"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 28:
        function_name = "IDMPM3T2"
        n_var = 3
        n_obj = 3
        xl = torch.tensor([[-1, -1, -1]])
        xu = torch.tensor([[1, 1, 1]])
        refpoint = torch.tensor([[1, 1, 1]])
        mat_filename = "IDMPM3T2Reference.mat"
        mat_folder = "IDMP"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 29:
        function_name = "IDMPM3T3"
        n_var = 3
        n_obj = 3
        xl = torch.tensor([[-1, -1, -1]])
        xu = torch.tensor([[1, 1, 1]])
        refpoint = torch.tensor([[1, 1, 1]])
        mat_filename = "IDMPM3T3Reference.mat"
        mat_folder = "IDMP"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 30:
        function_name = "IDMPM3T4"
        n_var = 3
        n_obj = 3
        xl = torch.tensor([[-1, -1, -1]])
        xu = torch.tensor([[1, 1, 1]])
        refpoint = torch.tensor([[1, 1, 1]])
        mat_filename = "IDMPM3T4Reference.mat"
        mat_folder = "IDMP"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 31:
        function_name = "IDMPM4T1"
        n_var = 4
        n_obj = 4
        xl = torch.tensor([[-1, -1, -1, -1]])
        xu = torch.tensor([[1, 1, 1, 1]])
        refpoint = torch.tensor([[1, 1, 1, 1]])
        mat_filename = "IDMPM4T1Reference.mat"
        mat_folder = "IDMP"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 32:
        function_name = "IDMPM4T2"
        n_var = 4
        n_obj = 4
        xl = torch.tensor([[-1, -1, -1, -1]])
        xu = torch.tensor([[1, 1, 1, 1]])
        refpoint = torch.tensor([[1, 1, 1, 1]])
        mat_filename = "IDMPM4T2Reference.mat"
        mat_folder = "IDMP"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 33:
        function_name = "IDMPM4T3"
        n_var = 4
        n_obj = 4
        xl = torch.tensor([[-1, -1, -1, -1]])
        xu = torch.tensor([[1, 1, 1, 1]])
        refpoint = torch.tensor([[1, 1, 1, 1]])
        mat_filename = "IDMPM4T3Reference.mat"
        mat_folder = "IDMP"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 34:
        function_name = "IDMPM4T4"
        n_var = 4
        n_obj = 4
        xl = torch.tensor([[-1, -1, -1, -1]])
        xu = torch.tensor([[1, 1, 1, 1]])
        refpoint = torch.tensor([[1, 1, 1, 1]])
        mat_filename = "IDMPM4T4Reference.mat"
        mat_folder = "IDMP"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 35:
        function_name = "IDMPM2T1_e"
        n_var = 2
        n_obj = 2
        xl = torch.tensor([[-1, -1]])
        xu = torch.tensor([[1, 1]])
        refpoint = torch.tensor([[1, 1]])
        mat_filename = "IDMPM2T1_e_Reference.mat"
        mat_folder = "IDMP_e"
        load_PF = "PFs"
        load_PS = "PSs"
    elif function_num == 36:
        function_name = "IDMPM2T2_e"
        n_var = 2
        n_obj = 2
        xl = torch.tensor([[-1, -1]])
        xu = torch.tensor([[1, 1]])
        refpoint = torch.tensor([[1, 1]])
        mat_filename = "IDMPM2T2_e_Reference.mat"
        mat_folder = "IDMP_e"
        load_PF = "PFs"
        load_PS = "PSs"
    elif function_num == 37:
        function_name = "IDMPM2T3_e"
        n_var = 2
        n_obj = 2
        xl = torch.tensor([[-1, -1]])
        xu = torch.tensor([[1, 1]])
        refpoint = torch.tensor([[1, 1]])
        mat_filename = "IDMPM2T3_e_Reference.mat"
        mat_folder = "IDMP_e"
        load_PF = "PFs"
        load_PS = "PSs"
    elif function_num == 38:
        function_name = "IDMPM2T4_e"
        n_var = 2
        n_obj = 2
        xl = torch.tensor([[-1, -1]])
        xu = torch.tensor([[1, 1]])
        refpoint = torch.tensor([[1, 1]])
        mat_filename = "IDMPM2T4_e_Reference.mat"
        mat_folder = "IDMP_e"
        load_PF = "PFs"
        load_PS = "PSs"
    elif function_num == 39:
        function_name = "IDMPM3T1_e"
        n_var = 3
        n_obj = 3
        xl = torch.tensor([[-1, -1, -1]])
        xu = torch.tensor([[1, 1, 1]])
        refpoint = torch.tensor([[1, 1, 1]])
        mat_filename = "IDMPM3T1_e_Reference.mat"
        mat_folder = "IDMP_e"
        load_PF = "PFs"
        load_PS = "PSs"
    elif function_num == 40:
        function_name = "IDMPM3T2_e"
        n_var = 3
        n_obj = 3
        xl = torch.tensor([[-1, -1, -1]])
        xu = torch.tensor([[1, 1, 1]])
        refpoint = torch.tensor([[1, 1, 1]])
        mat_filename = "IDMPM3T2_e_Reference.mat"
        mat_folder = "IDMP_e"
        load_PF = "PFs"
        load_PS = "PSs"
    elif function_num == 41:
        function_name = "IDMPM3T3_e"
        n_var = 3
        n_obj = 3
        xl = torch.tensor([[-1, -1, -1]])
        xu = torch.tensor([[1, 1, 1]])
        refpoint = torch.tensor([[1, 1, 1]])
        mat_filename = "IDMPM3T3_e_Reference.mat"
        mat_folder = "IDMP_e"
        load_PF = "PFs"
        load_PS = "PSs"
    elif function_num == 42:
        function_name = "IDMPM3T4_e"
        n_var = 3
        n_obj = 3
        xl = torch.tensor([[-1, -1, -1]])
        xu = torch.tensor([[1, 1, 1]])
        refpoint = torch.tensor([[1, 1, 1]])
        mat_filename = "IDMPM3T4_e_Reference.mat"
        mat_folder = "IDMP_e"
        load_PF = "PFs"
        load_PS = "PSs"
    elif function_num == 43:
        function_name = "MMMOP1A"
        n_var = 3
        n_obj = 2
        xl = torch.tensor([[0, 0, 0]])
        xu = torch.tensor([[1, 1, 1]])
        refpoint = None
        mat_filename = "MMMOP1A_PFPS.mat"
        mat_folder = "MMMOP"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 44:
        function_name = "MMMOP1B"
        n_var = 7
        n_obj = 3
        xl = torch.tensor([[0, 0, 0, 0, 0, 0, 0]])
        xu = torch.tensor([[1, 1, 1, 1, 1, 1, 1]])
        refpoint = None
        mat_filename = "MMMOP1B_PFPS.mat"
        mat_folder = "MMMOP"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 45:
        function_name = "MMMOP2A"
        n_var = 3
        n_obj = 2
        xl = torch.tensor([[0, 0, 0]])
        xu = torch.tensor([[1, 1, 1]])
        refpoint = None
        mat_filename = "MMMOP2A_PFPS.mat"
        mat_folder = "MMMOP"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 46:
        function_name = "MMMOP2B"
        n_var = 7
        n_obj = 3
        xl = torch.tensor([[0, 0, 0, 0, 0, 0, 0]])
        xu = torch.tensor([[1, 1, 1, 1, 1, 1, 1]])
        refpoint = None
        mat_filename = "MMMOP2B_PFPS.mat"
        mat_folder = "MMMOP"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 47:
        function_name = "MMMOP3A"
        n_var = 2
        n_obj = 2
        xl = torch.tensor([[0, 0]])
        xu = torch.tensor([[1, 1]])
        refpoint = None
        mat_filename = "MMMOP3A_PFPS.mat"
        mat_folder = "MMMOP"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 48:
        function_name = "MMMOP3B"
        n_var = 7
        n_obj = 3
        xl = torch.tensor([[0, 0, 0, 0, 0, 0, 0]])
        xu = torch.tensor([[1, 1, 1, 1, 1, 1, 1]])
        refpoint = None
        mat_filename = "MMMOP3B_PFPS.mat"
        mat_folder = "MMMOP"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 49:
        function_name = "MMMOP3C"
        n_var = 6
        n_obj = 2
        xl = torch.tensor([[0, 0, 0, 0, 0, 0]])
        xu = torch.tensor([[1, 1, 1, 1, 1, 1]])
        refpoint = None
        mat_filename = "MMMOP3C_PFPS.mat"
        mat_folder = "MMMOP"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 50:
        function_name = "MMMOP3D"
        n_var = 7
        n_obj = 3
        xl = torch.tensor([[0, 0, 0, 0, 0, 0, 0]])
        xu = torch.tensor([[1, 1, 1, 1, 1, 1, 1]])
        refpoint = None
        mat_filename = "MMMOP3D_PFPS.mat"
        mat_folder = "MMMOP"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 51:
        function_name = "MMMOP4A"
        n_var = 2
        n_obj = 2
        xl = torch.tensor([[0, 0]])
        xu = torch.tensor([[1, 1]])
        refpoint = None
        mat_filename = "MMMOP4A_PFPS.mat"
        mat_folder = "MMMOP"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 52:
        function_name = "MMMOP4B"
        n_var = 7
        n_obj = 3
        xl = torch.tensor([[0, 0, 0, 0, 0, 0, 0]])
        xu = torch.tensor([[1, 1, 1, 1, 1, 1, 1]])
        refpoint = None
        mat_filename = "MMMOP4B_PFPS.mat"
        mat_folder = "MMMOP"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 53:
        function_name = "MMMOP4C"
        n_var = 6
        n_obj = 2
        xl = torch.tensor([[0, 0, 0, 0, 0, 0]])
        xu = torch.tensor([[1, 1, 1, 1, 1, 1]])
        refpoint = None
        mat_filename = "MMMOP4C_PFPS.mat"
        mat_folder = "MMMOP"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 54:
        function_name = "MMMOP4D"
        n_var = 7
        n_obj = 3
        xl = torch.tensor([[0, 0, 0, 0, 0, 0, 0]])
        xu = torch.tensor([[1, 1, 1, 1, 1, 1, 1]])
        refpoint = None
        mat_filename = "MMMOP4D_PFPS.mat"
        mat_folder = "MMMOP"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 55:
        function_name = "MMMOP5A"
        n_var = 2
        n_obj = 2
        xl = torch.tensor([[0, 0]])
        xu = torch.tensor([[1, 1]])
        refpoint = None
        mat_filename = "MMMOP5A_PFPS.mat"
        mat_folder = "MMMOP"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 56:
        function_name = "MMMOP5B"
        n_var = 7
        n_obj = 3
        xl = torch.tensor([[0, 0, 0, 0, 0, 0, 0]])
        xu = torch.tensor([[1, 1, 1, 1, 1, 1, 1]])
        refpoint = None
        mat_filename = "MMMOP5B_PFPS.mat"
        mat_folder = "MMMOP"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 57:
        function_name = "MMMOP5C"
        n_var = 6
        n_obj = 2
        xl = torch.tensor([[0, 0, 0, 0, 0, 0]])
        xu = torch.tensor([[1, 1, 1, 1, 1, 1]])
        refpoint = None
        mat_filename = "MMMOP5C_PFPS.mat"
        mat_folder = "MMMOP"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 58:
        function_name = "MMMOP5D"
        n_var = 7
        n_obj = 3
        xl = torch.tensor([[0, 0, 0, 0, 0, 0, 0]])
        xu = torch.tensor([[1, 1, 1, 1, 1, 1, 1]])
        refpoint = None
        mat_filename = "MMMOP5D_PFPS.mat"
        mat_folder = "MMMOP"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 59:
        function_name = "MMMOP6A"
        n_var = 2
        n_obj = 2
        xl = torch.tensor([[0, 0]])
        xu = torch.tensor([[1, 1]])
        refpoint = None
        mat_filename = "MMMOP6A_PFPS.mat"
        mat_folder = "MMMOP"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 60:
        function_name = "MMMOP6B"
        n_var = 4
        n_obj = 3
        xl = torch.tensor([[0, 0, 0, 0]])
        xu = torch.tensor([[1, 1, 1, 1]])
        refpoint = None
        mat_filename = "MMMOP6B_PFPS.mat"
        mat_folder = "MMMOP"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 61:
        function_name = "MMMOP6C"
        n_var = 4
        n_obj = 2
        xl = torch.tensor([[0, 0, 0, 0]])
        xu = torch.tensor([[1, 1, 1, 1]])
        refpoint = None
        mat_filename = "MMMOP6C_PFPS.mat"
        mat_folder = "MMMOP"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 62:
        function_name = "MMMOP6D"
        n_var = 5
        n_obj = 3
        xl = torch.tensor([[0, 0, 0, 0, 0]])
        xu = torch.tensor([[1, 1, 1, 1, 1]])
        refpoint = None
        mat_filename = "MMMOP6D_PFPS.mat"
        mat_folder = "MMMOP"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 63:
        function_name = "poly_2"
        n_var = 2
        n_obj = 2
        xl = torch.tensor([[-1, -1]])
        xu = torch.tensor([[1, 1]])
        refpoint = None
        mat_filename = "poly_2.mat"
        mat_folder = "Poly"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 64:
        function_name = "poly_3"
        n_var = 2
        n_obj = 3
        xl = torch.tensor([[-1, -1]])
        xu = torch.tensor([[1, 1]])
        refpoint = None
        mat_filename = "poly_3.mat"
        mat_folder = "Poly"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 65:
        function_name = "poly_4"
        n_var = 2
        n_obj = 4
        xl = torch.tensor([[-1, -1]])
        xu = torch.tensor([[1, 1]])
        refpoint = None
        mat_filename = "poly_4.mat"
        mat_folder = "Poly"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 66:
        function_name = "poly_5"
        n_var = 2
        n_obj = 5
        xl = torch.tensor([[-1, -1]])
        xu = torch.tensor([[1, 1]])
        refpoint = None
        mat_filename = "poly_5.mat"
        mat_folder = "Poly"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 67:
        function_name = "poly_6"
        n_var = 2
        n_obj = 6
        xl = torch.tensor([[-1, -1]])
        xu = torch.tensor([[1, 1]])
        refpoint = None
        mat_filename = "poly_6.mat"
        mat_folder = "Poly"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 68:
        function_name = "poly_8"
        n_var = 2
        n_obj = 8
        xl = torch.tensor([[-1, -1]])
        xu = torch.tensor([[1, 1]])
        refpoint = None
        mat_filename = "poly_8.mat"
        mat_folder = "Poly"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 69:
        function_name = "poly_10"
        n_var = 2
        n_obj = 10
        xl = torch.tensor([[-1, -1]])
        xu = torch.tensor([[1, 1]])
        refpoint = None
        mat_filename = "poly_10.mat"
        mat_folder = "Poly"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 70:
        function_name = "poly_12"
        n_var = 2
        n_obj = 12
        xl = torch.tensor([[-1, -1]])
        xu = torch.tensor([[1, 1]])
        refpoint = None
        mat_filename = "poly_12.mat"
        mat_folder = "Poly"
        load_PF = "PF"
        load_PS = "PSS"
    elif function_num == 71:
        function_name = "map"
        n_var = 2
        n_obj = 4
        xl = torch.tensor([[0, 0]])
        xu = torch.tensor([[100, 100]])
        refpoint = None
        mat_filename = "maptruePSPF.mat"
        mat_folder = "APP"
        load_PF = "PF"
        load_PS = "PS"
    elif function_num == 72:
        function_name = "application"
        n_var = 2
        n_obj = 3
        xl = torch.tensor([[0, 0]])
        xu = torch.tensor([[2.5, 1.5]])
        refpoint = None
        mat_filename = None
        mat_folder = "APP"
        load_PF = None
        load_PS = None

    test_suit = _TEST_SUITES[mat_folder]
    function = getattr(test_suit, function_name)

    if function_num != 72:
        current_file_path = os.path.dirname(__file__)
        mat_path = os.path.join(current_file_path, "..", "TruePSPFdata", mat_folder, mat_filename)
        try:
            data = scipy.io.loadmat(mat_path)
            PF = torch.from_numpy(data[load_PF])
            PS = torch.from_numpy(data[load_PS])
        except NotImplementedError:
            try:
                import h5py
            except ModuleNotFoundError:
                PF = None
                PS = None
            else:
                with h5py.File(mat_path, "r") as f:
                    PF = torch.from_numpy(np.array(f[load_PF]))
                    PS = torch.from_numpy(np.array(f[load_PS]))
                    if PF.ndim > 1:
                        PF = PF.T
                    if PS.ndim > 1:
                        PS = PS.T

    return function, function_name, n_var, n_obj, xl, xu, refpoint, PS, PF
