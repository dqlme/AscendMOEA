"""Implementation utilities for CEC2019."""

import numpy as np
import torch


def MMF1(x: torch.Tensor):
    """Evaluate the MMF1 benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    y = torch.zeros((x.shape[0], 2))
    x1_abs = torch.abs(x[:, 0] - 2.0)
    y[:, 0] = x1_abs
    y[:, 1] = (
        1.0
        - torch.sqrt(x1_abs)
        + 2.0 * (x[:, 1] - torch.sin(6.0 * torch.pi * x1_abs + torch.pi)) ** 2
    )
    return y


def MMF1_e(x: torch.Tensor):
    """Evaluate the MMF1_e benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    y = torch.zeros((x.shape[0], 2))

    left_index = x[:, 0] < 2.0
    right_index = x[:, 0] >= 2.0

    y[left_index, 0] = 2.0 - x[left_index, 0]
    y[right_index, 0] = x[right_index, 0] - 2.0

    f1_val = y[:, 0]
    term1 = 1.0 - torch.sqrt(f1_val)

    term2_left = (
        2.0 * (x[left_index, 1] - torch.sin(6.0 * torch.pi * f1_val[left_index] + torch.pi)) ** 2
    )
    term2_right = (
        2.0
        * (
            x[right_index, 1]
            - torch.exp(x[right_index, 0])
            * torch.sin(6.0 * torch.pi * f1_val[right_index] + torch.pi)
        )
        ** 2
    )

    y[left_index, 1] = term1[left_index] + term2_left
    y[right_index, 1] = term1[right_index] + term2_right

    return y


def MMF1_z(x: torch.Tensor):
    """Evaluate the MMF1_z benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    y = torch.zeros((x.shape[0], 2))

    left_index = x[:, 0] < 2.0
    right_index = x[:, 0] >= 2.0

    y[left_index, 0] = 2.0 - x[left_index, 0]
    y[right_index, 0] = x[right_index, 0] - 2.0

    f1_val = y[:, 0]
    term1 = 1.0 - torch.sqrt(f1_val)

    term2_left = (
        2.0 * (x[left_index, 1] - torch.sin(6.0 * torch.pi * f1_val[left_index] + torch.pi)) ** 2
    )
    term2_right = (
        2.0 * (x[right_index, 1] - torch.sin(2.0 * torch.pi * f1_val[right_index] + torch.pi)) ** 2
    )

    y[left_index, 1] = term1[left_index] + term2_left
    y[right_index, 1] = term1[right_index] + term2_right

    return y


def MMF2(x: torch.Tensor):
    """Evaluate the MMF2 benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    f = torch.zeros((x.shape[0], 2))
    x_mod = x.clone()
    x_mod[:, 1] -= (x_mod[:, 1] > 1.0).type_as(x_mod)

    f[:, 0] = x_mod[:, 0]
    y2 = x_mod[:, 1] - torch.sqrt(x_mod[:, 0])
    f[:, 1] = (
        1.0
        - torch.sqrt(x_mod[:, 0])
        + 2.0 * ((4.0 * y2**2) - 2.0 * torch.cos(20.0 * y2 * torch.pi / np.sqrt(2.0)) + 2.0)
    )
    return f


def MMF3(x: torch.Tensor):
    """Evaluate the MMF3 benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    f = torch.zeros((x.shape[0], 2))
    f[:, 0] = x[:, 0]

    x1 = x[:, 0]
    x2 = x[:, 1]
    y2 = torch.zeros(x.shape[0])
    sqrt_x1 = torch.sqrt(x1)

    cond1 = (x2 >= 0) & (x2 <= 0.5)
    y2[cond1] = x2[cond1] - sqrt_x1[cond1]

    cond2 = (x2 > 0.5) & (x2 < 1) & (x1 >= 0) & (x1 <= 0.25)
    y2[cond2] = x2[cond2] - 0.5 - sqrt_x1[cond2]

    cond3 = (x2 > 0.5) & (x2 < 1) & (x1 > 0.25)
    y2[cond3] = x2[cond3] - sqrt_x1[cond3]

    cond4 = (x2 >= 1) & (x2 <= 1.5)
    y2[cond4] = x2[cond4] - 0.5 - sqrt_x1[cond4]

    f[:, 1] = (
        1.0
        - sqrt_x1
        + 2.0 * ((4.0 * y2**2) - 2.0 * torch.cos(20.0 * y2 * torch.pi / np.sqrt(2.0)) + 2.0)
    )
    return f


def MMF4(x: torch.Tensor):
    """Evaluate the MMF4 benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    f = torch.zeros((x.shape[0], 2))
    x_mod = x.clone()
    x_mod[:, 1] -= (x_mod[:, 1] > 1.0).type_as(x_mod)
    abs_x1 = torch.abs(x_mod[:, 0])

    f[:, 0] = abs_x1
    f[:, 1] = 1.0 - x_mod[:, 0] ** 2 + 2.0 * (x_mod[:, 1] - torch.sin(torch.pi * abs_x1)) ** 2
    return f


def MMF5(x: torch.Tensor):
    """Evaluate the MMF5 benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    y = torch.zeros((x.shape[0], 2))
    x_mod = x.clone()
    x_mod[:, 1] = x_mod[:, 1] - (x_mod[:, 1] > 1.0) * 2.0

    abs_x1_minus_2 = torch.abs(x_mod[:, 0] - 2.0)
    y[:, 0] = abs_x1_minus_2
    y[:, 1] = (
        1.0
        - torch.sqrt(abs_x1_minus_2)
        + 2.0 * (x_mod[:, 1] - torch.sin(6.0 * torch.pi * abs_x1_minus_2 + torch.pi)) ** 2
    )
    return y


def MMF6(x: torch.Tensor):
    """Evaluate the MMF6 benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    y = torch.zeros((x.shape[0], 2))
    x_mod = x.clone()
    x1 = x_mod[:, 0]
    x2 = x_mod[:, 1]

    cond1_x1 = (
        ((x1 > 13 / 6) & (x1 <= 14 / 6))
        | ((x1 > 15 / 6) & (x1 <= 16 / 6))
        | ((x1 > 17 / 6) & (x1 <= 3))
    )
    cond1 = (x2 > 1.0) & cond1_x1
    x_mod[cond1, 1] -= 1.0

    cond2_x1 = (
        ((x1 > 7 / 6) & (x1 <= 8 / 6))
        | ((x1 > 9 / 6) & (x1 <= 10 / 6))
        | ((x1 > 11 / 6) & (x1 <= 2))
        | ((x1 > 2) & (x1 <= 13 / 6))
        | ((x1 > 14 / 6) & (x1 <= 15 / 6))
        | ((x1 > 16 / 6) & (x1 <= 17 / 6))
    )
    cond2 = (x2 > 0) & (x2 <= 1.0) & cond2_x1
    x_mod[cond2, 1] -= 1.0

    cond3_x1 = (
        ((x1 > 1) & (x1 <= 7 / 6))
        | ((x1 > 4 / 3) & (x1 <= 3 / 2))
        | ((x1 > 5 / 3) & (x1 <= 11 / 6))
    )
    cond3 = (x2 > 1.0) & (x2 <= 2.0) & cond3_x1
    x_mod[cond3, 1] -= 1.0

    abs_x1_minus_2 = torch.abs(x_mod[:, 0] - 2.0)
    y[:, 0] = abs_x1_minus_2
    y[:, 1] = (
        1.0
        - torch.sqrt(abs_x1_minus_2)
        + 2.0 * (x_mod[:, 1] - torch.sin(6.0 * torch.pi * abs_x1_minus_2 + torch.pi)) ** 2
    )
    return y


def MMF7(x: torch.Tensor):
    """Evaluate the MMF7 benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    y = torch.zeros((x.shape[0], 2))
    x_mod = x.clone()
    x_mod[:, 0] = torch.abs(x_mod[:, 0] - 2.0)
    x1_mod = x_mod[:, 0]
    x2 = x_mod[:, 1]

    y[:, 0] = x1_mod
    term_in_paren = (
        0.3 * (x1_mod**2) * torch.cos(24.0 * torch.pi * x1_mod + 4.0 * torch.pi) + 0.6 * x1_mod
    )
    y[:, 1] = (
        1.0
        - torch.sqrt(x1_mod)
        + (x2 - term_in_paren * torch.sin(6.0 * torch.pi * x1_mod + torch.pi)) ** 2
    )
    return y


def MMF8(x: torch.Tensor):
    """Evaluate the MMF8 benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    y = torch.zeros((x.shape[0], 2))
    x_mod = x.clone()
    x_mod[:, 1] = x_mod[:, 1] - (x_mod[:, 1] > 4.0) * 4.0

    abs_x1 = torch.abs(x_mod[:, 0])
    sin_abs_x1 = torch.sin(abs_x1)

    y[:, 0] = sin_abs_x1
    sqrt_term_arg = torch.clamp(1.0 - sin_abs_x1**2, min=0.0)
    y[:, 1] = torch.sqrt(sqrt_term_arg) + 2.0 * (x_mod[:, 1] - (sin_abs_x1 + abs_x1)) ** 2
    return y


def MMF9(x: torch.Tensor):
    """Evaluate the MMF9 benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    y = torch.zeros((x.shape[0], 2))
    num_of_peak = 2.0

    x1 = x[:, 0]
    x2 = x[:, 1]

    temp1 = (torch.sin(num_of_peak * torch.pi * x2)) ** 6
    g = 2.0 - temp1
    x1_safe = torch.clamp(x1, min=1e-9)
    y[:, 0] = x1
    y[:, 1] = g / x1_safe
    return y


def MMF10(x: torch.Tensor):
    """Evaluate the MMF10 benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    y = torch.zeros((x.shape[0], 2))

    x1 = x[:, 0]
    x2 = x[:, 1]

    term1 = torch.exp(-(((x2 - 0.2) / 0.004) ** 2))
    term2 = 0.8 * torch.exp(-(((x2 - 0.6) / 0.4) ** 2))
    g = 2.0 - term1 - term2
    x1_safe = torch.clamp(x1, min=1e-9)
    y[:, 0] = x1
    y[:, 1] = g / x1_safe
    return y


def MMF11(x: torch.Tensor):
    """Evaluate the MMF11 benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    y = torch.zeros((x.shape[0], 2))
    num_of_peak = 2.0

    x1 = x[:, 0]
    x2 = x[:, 1]

    temp1 = (torch.sin(num_of_peak * torch.pi * x2)) ** 6
    temp2 = torch.exp(-2.0 * np.log10(2.0) * ((x2 - 0.1) / 0.8) ** 2)
    g = 2.0 - temp2 * temp1
    x1_safe = torch.clamp(x1, min=1e-9)
    y[:, 0] = x1
    y[:, 1] = g / x1_safe
    return y


def MMF12(x: torch.Tensor):
    """Evaluate the MMF12 benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    y = torch.zeros((x.shape[0], 2))
    q = 4.0
    Alfa = 2.0
    num_of_peak = 2.0

    x1 = x[:, 0]
    x2 = x[:, 1]

    term_g1 = (torch.sin(num_of_peak * torch.pi * x2)) ** 6
    term_g2 = torch.exp(-2.0 * np.log10(2.0) * ((x2 - 0.1) / 0.8) ** 2)
    g = 2.0 - term_g1 * term_g2

    y[:, 0] = x1

    g_safe = torch.clamp(g, min=1e-9)
    ratio = y[:, 0] / g_safe
    h = 1.0 - ratio**Alfa - ratio * torch.sin(2.0 * torch.pi * q * y[:, 0])

    y[:, 1] = g * h
    return y


def MMF13(x: torch.Tensor):
    """Evaluate the MMF13 benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    y = torch.zeros((x.shape[0], 2))

    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]

    term_g_arg = x2 + torch.sqrt(x3)
    term_g1 = torch.exp(-2.0 * np.log10(2.0) * ((term_g_arg - 0.1) / 0.8) ** 2)
    term_g2 = (torch.sin(2.0 * torch.pi * term_g_arg)) ** 6
    g = 2.0 - term_g1 * term_g2

    x1_safe = torch.clamp(x1, min=1e-9)
    y[:, 0] = x1
    y[:, 1] = g / x1_safe
    return y


def MMF14(x: torch.Tensor, M: int = 3, num_of_peak: int = 2):
    """Evaluate the MMF14 benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    N = x.shape[0]

    g = 2.0 - (torch.sin(num_of_peak * torch.pi * x[:, -1])) ** 2

    cos_terms = torch.cos(x[:, : M - 1] * torch.pi / 2.0)
    cumprod_cos = torch.fliplr(
        torch.cumprod(torch.cat((torch.ones((N, 1)), cos_terms), dim=1), dim=1)
    )

    sin_terms = torch.sin(x[:, torch.arange(M - 2, -1, -1)] * torch.pi / 2.0)
    shape_terms = torch.cat((torch.ones((N, 1)), sin_terms), dim=1)

    f = cumprod_cos * shape_terms

    y = f * (1.0 + g.unsqueeze(-1))
    return y


def MMF14_a(x: torch.Tensor, M: int = 3, num_of_peak: int = 2):
    """Evaluate the MMF14_a benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    N = x.shape[0]

    x_g = x[:, -1] - 0.5 * torch.sin(torch.pi * x[:, -2])
    g = 2.0 - (torch.sin(num_of_peak * torch.pi * (x_g + 1.0 / (2.0 * num_of_peak)))) ** 2

    cos_terms = torch.cos(x[:, : M - 1] * torch.pi / 2.0)
    cumprod_cos = torch.fliplr(
        torch.cumprod(torch.cat((torch.ones((N, 1)), cos_terms), dim=1), dim=1)
    )
    sin_terms = torch.sin(x[:, torch.arange(M - 2, -1, -1)] * torch.pi / 2.0)
    shape_terms = torch.cat((torch.ones((N, 1)), sin_terms), dim=1)
    f = cumprod_cos * shape_terms

    y = f * (1.0 + g.unsqueeze(-1))
    return y


def MMF15(x: torch.Tensor, M: int = 3, num_of_peak: int = 2):
    """Evaluate the MMF15 benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    N = x.shape[0]

    x_last = x[:, -1]
    term_g1 = torch.exp(-2.0 * np.log10(2.0) * ((x_last - 0.1) / 0.8) ** 2)
    term_g2 = (torch.sin(num_of_peak * torch.pi * x_last)) ** 2
    g = 2.0 - term_g1 * term_g2

    cos_terms = torch.cos(x[:, : M - 1] * torch.pi / 2.0)
    cumprod_cos = torch.fliplr(
        torch.cumprod(torch.cat((torch.ones((N, 1)), cos_terms), dim=1), dim=1)
    )
    sin_terms = torch.sin(x[:, torch.arange(M - 2, -1, -1)] * torch.pi / 2.0)
    shape_terms = torch.cat((torch.ones((N, 1)), sin_terms), dim=1)
    f = cumprod_cos * shape_terms

    y = f * (1.0 + g.unsqueeze(-1))
    return y


def MMF15_a(x: torch.Tensor, M: int = 3, num_of_peak: int = 2):
    """Evaluate the MMF15_a benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    N = x.shape[0]

    t = x[:, -1] - 0.5 * torch.sin(torch.pi * x[:, -2])
    t_shifted = t + 1.0 / (2.0 * num_of_peak)
    term_g1 = torch.exp(-2.0 * np.log10(2.0) * ((t_shifted - 0.1) / 0.8) ** 2)
    term_g2 = (torch.sin(num_of_peak * torch.pi * t_shifted)) ** 2
    g = 2.0 - term_g1 * term_g2

    cos_terms = torch.cos(x[:, : M - 1] * torch.pi / 2.0)
    cumprod_cos = torch.fliplr(
        torch.cumprod(torch.cat((torch.ones((N, 1)), cos_terms), dim=1), dim=1)
    )
    sin_terms = torch.sin(x[:, torch.arange(M - 2, -1, -1)] * torch.pi / 2.0)
    shape_terms = torch.cat((torch.ones((N, 1)), sin_terms), dim=1)
    f = cumprod_cos * shape_terms

    y = f * (1.0 + g.unsqueeze(-1))
    return y


def Omni_test(x: torch.Tensor):
    """Evaluate the Omni_test benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    f = torch.zeros((x.shape[0], 2))
    f[:, 0] = torch.sum(torch.sin(torch.pi * x), dim=1)
    f[:, 1] = torch.sum(torch.cos(torch.pi * x), dim=1)
    return f


def _fun_sym_part(x: torch.Tensor, a: float):
    """Helper function used by SYM_PART problems."""
    y = torch.zeros((x.shape[0], 2))
    y[:, 0] = (x[:, 0] + a) ** 2 + x[:, 1] ** 2
    y[:, 1] = (x[:, 0] - a) ** 2 + x[:, 1] ** 2
    return y


def SYM_PART_simple(x: torch.Tensor):
    """Evaluate the SYM_PART_simple benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    a = 1.0
    b = 10.0
    c = 8.0

    x1_orig = x[:, 0]
    x2_orig = x[:, 1]

    temp_t1_arg = (torch.abs(x1_orig) - (a + c / 2.0)) / (2.0 * a + c)
    temp_t1 = torch.sign(x1_orig) * torch.ceil(temp_t1_arg)
    t1 = torch.sign(temp_t1) * torch.clamp(torch.abs(temp_t1), max=1.0)

    temp_t2_arg = (torch.abs(x2_orig) - b / 2.0) / b
    temp_t2 = torch.sign(x2_orig) * torch.ceil(temp_t2_arg)
    t2 = torch.sign(temp_t2) * torch.clamp(torch.abs(temp_t2), max=1.0)

    x1_mod = x1_orig - t1 * (c + 2.0 * a)
    x2_mod = x2_orig - t2 * b

    y = _fun_sym_part(torch.stack((x1_mod, x2_mod), dim=1), a)
    return y


def SYM_PART_rotated(x: torch.Tensor):
    """Evaluate the SYM_PART_rotated benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    a = 1.0
    b = 10.0
    c = 8.0
    w = np.pi / 4.0

    x1_orig = x[:, 0]
    x2_orig = x[:, 1]

    cos_w = np.cos(w)
    sin_w = np.sin(w)
    xx1 = cos_w * x1_orig - sin_w * x2_orig
    xx2 = sin_w * x1_orig + cos_w * x2_orig

    temp_t1_arg = (torch.abs(xx1) - (a + c / 2.0)) / (2.0 * a + c)
    temp_t1 = torch.sign(xx1) * torch.ceil(temp_t1_arg)
    t1 = torch.sign(temp_t1) * torch.clamp(torch.abs(temp_t1), max=1.0)

    temp_t2_arg = (torch.abs(xx2) - b / 2.0) / b
    temp_t2 = torch.sign(xx2) * torch.ceil(temp_t2_arg)
    t2 = torch.sign(temp_t2) * torch.clamp(torch.abs(temp_t2), max=1.0)

    x1_mod = xx1 - t1 * (c + 2.0 * a)
    x2_mod = xx2 - t2 * b

    y = _fun_sym_part(torch.stack((x1_mod, x2_mod), dim=1), a)
    return y
