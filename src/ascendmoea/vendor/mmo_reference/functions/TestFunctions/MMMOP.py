"""Implementation utilities for MMMOP."""

import torch


def MMMOP1A(x: torch.Tensor):
    """Evaluate the MMMOP1A benchmark formula."""
    N = x.shape[0]
    M = 2
    kA = 1
    kB = 1

    term1 = torch.sum(torch.sin(5.0 * torch.pi * x[:, M - 1 : M + kA - 1]) ** 6, dim=1)
    g_vars = x[:, M + kA - 1 :]
    term2 = torch.sum((g_vars - 0.5) ** 2 - torch.cos(20.0 * torch.pi * (g_vars - 0.5)), dim=1)
    g = 100.0 * (kA + kB - term1 + term2)

    prod_terms = x[:, : M - 1]
    cumprod_rev = torch.fliplr(torch.cumprod(torch.hstack((torch.ones((N, 1)), prod_terms)), dim=1))

    one_minus_x_rev = 1.0 - x[:, : M - 1].flip(dims=[1])
    shape_terms = torch.hstack((torch.ones((N, 1)), one_minus_x_rev))

    f = cumprod_rev * shape_terms

    PopObj = f * (1.0 + g)[:, None]
    return PopObj


def MMMOP1B(x: torch.Tensor):
    """Evaluate the MMMOP1B benchmark formula."""
    N = x.shape[0]
    M = 3
    kA = 1
    kB = 4

    term1 = torch.sum(torch.sin(5.0 * torch.pi * x[:, M - 1 : M + kA - 1]) ** 6, dim=1)
    g_vars = x[:, M + kA - 1 :]
    term2 = torch.sum((g_vars - 0.5) ** 2 - torch.cos(20.0 * torch.pi * (g_vars - 0.5)), dim=1)
    g = 100.0 * (kA + kB - term1 + term2)

    prod_terms = x[:, : M - 1]
    cumprod_rev = torch.fliplr(torch.cumprod(torch.hstack((torch.ones((N, 1)), prod_terms)), dim=1))
    one_minus_x_rev = 1.0 - x[:, : M - 1].flip(dims=[1])
    shape_terms = torch.hstack((torch.ones((N, 1)), one_minus_x_rev))
    f = cumprod_rev * shape_terms

    PopObj = f * (1.0 + g)[:, None]
    return PopObj


def MMMOP2A(x: torch.Tensor):
    """Evaluate the MMMOP2A benchmark formula."""
    N = x.shape[0]
    M = 2
    kA = 1
    alpha = 100.0

    PopDec = x.clone()
    PopDec[:, : M - 1] = PopDec[:, : M - 1] ** alpha

    y_g = 9.75 * x[:, M - 1 : M + kA - 1] + 0.25
    term1 = torch.sum(torch.sin(10.0 * torch.log(y_g)), dim=1)
    term2 = torch.sum((x[:, M + kA - 1 :] - 0.5) ** 2, dim=1)
    g = kA - term1 + term2

    cos_terms = torch.cos(PopDec[:, : M - 1] * torch.pi / 2.0)
    cumprod_cos = torch.fliplr(torch.cumprod(torch.hstack((torch.ones((N, 1)), cos_terms)), dim=1))
    sin_terms = torch.sin(PopDec[:, : M - 1].flip(dims=[1]) * torch.pi / 2.0)
    shape_terms = torch.hstack((torch.ones((N, 1)), sin_terms))
    f = cumprod_cos * shape_terms

    PopObj = f * (1.0 + g)[:, None]
    return PopObj


def MMMOP2B(x: torch.Tensor):
    """Evaluate the MMMOP2B benchmark formula."""
    N = x.shape[0]
    M = 3
    kA = 1
    alpha = 100.0

    PopDec = x.clone()
    PopDec[:, : M - 1] = PopDec[:, : M - 1] ** alpha

    y_g = 9.75 * x[:, M - 1 : M + kA - 1] + 0.25
    term1 = torch.sum(torch.sin(10.0 * torch.log(y_g)), dim=1)
    term2 = torch.sum((x[:, M + kA - 1 :] - 0.5) ** 2, dim=1)
    g = kA - term1 + term2

    cos_terms = torch.cos(PopDec[:, : M - 1] * torch.pi / 2.0)
    cumprod_cos = torch.fliplr(torch.cumprod(torch.hstack((torch.ones((N, 1)), cos_terms)), dim=1))
    sin_terms = torch.sin(PopDec[:, : M - 1].flip(dims=[1]) * torch.pi / 2.0)
    shape_terms = torch.hstack((torch.ones((N, 1)), sin_terms))
    f = cumprod_cos * shape_terms

    PopObj = f * (1.0 + g)[:, None]
    return PopObj


def MMMOP3A(x: torch.Tensor):
    """Evaluate the MMMOP3A benchmark formula."""
    N = x.shape[0]
    M = 2
    kA = 0
    c = 0.0
    d = 3.0

    PopDec = x
    y_trans = PopDec[:, : M - 1] * d - torch.floor(PopDec[:, : M - 1] * d)

    term1 = (
        torch.sum(torch.cos(2.0 * torch.pi * c * PopDec[:, M - 1 : M + kA - 1]), dim=1)
        if kA > 0
        else torch.tensor(0.0)
    )
    term2 = torch.sum((PopDec[:, M + kA - 1 :] - 0.5) ** 2, dim=1)
    g = kA + term1 + term2

    cos_terms = torch.cos(y_trans * torch.pi / 2.0)
    cumprod_cos = torch.fliplr(torch.cumprod(torch.hstack((torch.ones((N, 1)), cos_terms)), dim=1))
    sin_terms = torch.sin(y_trans.flip(dims=[1]) * torch.pi / 2.0)
    shape_terms = torch.hstack((torch.ones((N, 1)), sin_terms))
    f = cumprod_cos * shape_terms

    PopObj = f * (1.0 + g)[:, None]
    return PopObj


def MMMOP3B(x: torch.Tensor):
    """Evaluate the MMMOP3B benchmark formula."""
    N = x.shape[0]
    M = 3
    kA = 0
    c = 0.0
    d = 1.0

    PopDec = x
    y_trans = PopDec[:, : M - 1] * d - torch.floor(PopDec[:, : M - 1] * d)

    term1 = (
        torch.sum(torch.cos(2.0 * torch.pi * c * PopDec[:, M - 1 : M + kA - 1]), dim=1)
        if kA > 0
        else torch.tensor(0.0)
    )
    term2 = torch.sum((PopDec[:, M + kA - 1 :] - 0.5) ** 2, dim=1)
    g = kA + term1 + term2

    cos_terms = torch.cos(y_trans * torch.pi / 2.0)
    cumprod_cos = torch.fliplr(torch.cumprod(torch.hstack((torch.ones((N, 1)), cos_terms)), dim=1))
    sin_terms = torch.sin(y_trans.flip(dims=[1]) * torch.pi / 2.0)
    shape_terms = torch.hstack((torch.ones((N, 1)), sin_terms))
    f = cumprod_cos * shape_terms

    PopObj = f * (1.0 + g)[:, None]
    return PopObj


def MMMOP3C(x: torch.Tensor):
    """Evaluate the MMMOP3C benchmark formula."""
    N = x.shape[0]
    M = 2
    kA = 1
    c = 3.0
    d = 3.0

    PopDec = x
    y_trans = PopDec[:, : M - 1] * d - torch.floor(PopDec[:, : M - 1] * d)

    term1 = (
        torch.sum(torch.cos(2.0 * torch.pi * c * PopDec[:, M - 1 : M + kA - 1]), dim=1)
        if kA > 0
        else torch.tensor(0.0)
    )
    term2 = torch.sum((PopDec[:, M + kA - 1 :] - 0.5) ** 2, dim=1)
    g = kA + term1 + term2

    cos_terms = torch.cos(y_trans * torch.pi / 2.0)
    cumprod_cos = torch.fliplr(torch.cumprod(torch.hstack((torch.ones((N, 1)), cos_terms)), dim=1))
    sin_terms = torch.sin(y_trans.flip(dims=[1]) * torch.pi / 2.0)
    shape_terms = torch.hstack((torch.ones((N, 1)), sin_terms))
    f = cumprod_cos * shape_terms

    PopObj = f * (1.0 + g)[:, None]
    return PopObj


def MMMOP3D(x: torch.Tensor):
    """Evaluate the MMMOP3D benchmark formula."""
    N = x.shape[0]
    M = 3
    kA = 1
    c = 2.0
    d = 2.0

    PopDec = x
    y_trans = PopDec[:, : M - 1] * d - torch.floor(PopDec[:, : M - 1] * d)

    term1 = (
        torch.sum(torch.cos(2.0 * torch.pi * c * PopDec[:, M - 1 : M + kA - 1]), dim=1)
        if kA > 0
        else torch.tensor(0.0)
    )
    term2 = torch.sum((PopDec[:, M + kA - 1 :] - 0.5) ** 2, dim=1)
    g = kA + term1 + term2

    cos_terms = torch.cos(y_trans * torch.pi / 2.0)
    cumprod_cos = torch.fliplr(torch.cumprod(torch.hstack((torch.ones((N, 1)), cos_terms)), dim=1))
    sin_terms = torch.sin(y_trans.flip(dims=[1]) * torch.pi / 2.0)
    shape_terms = torch.hstack((torch.ones((N, 1)), sin_terms))
    f = cumprod_cos * shape_terms

    PopObj = f * (1.0 + g)[:, None]
    return PopObj


def _transform_y_mmmop4(PopDec_subset, d):
    """Evaluate the _transform_y_mmmop4 benchmark formula."""
    N, num_vars = PopDec_subset.shape
    y = torch.zeros_like(PopDec_subset)
    d_sum_factor = d * (d + 1) * 0.5
    m_vals = torch.arange(d, 0, -1)
    d_sums = torch.cumsum(m_vals, dim=0)

    for i in range(N):
        for j in range(num_vars):
            y_val = PopDec_subset[i, j] * d_sum_factor
            for k in range(d):
                m = m_vals[k]
                current_d_sum = d_sums[k]
                if y_val <= current_d_sum:
                    y[i, j] = (y_val - current_d_sum + m) / d
                    break
    return y


def _transform_y_mmmop5(PopDec_subset, d):
    """Evaluate the _transform_y_mmmop5 benchmark formula."""
    N, num_vars = PopDec_subset.shape
    y = torch.zeros_like(PopDec_subset)
    d_sum_factor = (2 ** (d + 1)) - 1
    m_vals = torch.arange(d, -1, -1)
    two_pow_m = 2.0**m_vals
    d_sums = torch.cumsum(two_pow_m, dim=0)

    for i in range(N):
        for j in range(num_vars):
            y_val = PopDec_subset[i, j] * d_sum_factor
            for k in range(d + 1):
                m_pow = two_pow_m[k]
                current_d_sum = d_sums[k]
                if y_val <= current_d_sum:
                    y[i, j] = (y_val - current_d_sum + m_pow) / m_pow
                    break
    return y


def MMMOP4A(x: torch.Tensor):
    """Evaluate the MMMOP4A benchmark formula."""
    N = x.shape[0]
    M = 2
    kA = 0
    kB = 1
    c = 0.0
    d = 4

    PopDec = x
    y_trans = _transform_y_mmmop4(PopDec[:, : M - 1], d)

    term1 = (
        torch.sum(torch.cos(2.0 * torch.pi * c * PopDec[:, M - 1 : M + kA - 1]), dim=1)
        if kA > 0
        else torch.tensor(0.0)
    )
    g_vars = PopDec[:, M + kA - 1 :]
    term2 = torch.sum((g_vars - 0.5) ** 2 - torch.cos(20.0 * torch.pi * (g_vars - 0.5)), dim=1)
    g = 100.0 * (kA + kB + term1 + term2)

    cos_terms = torch.cos(y_trans * torch.pi / 2.0)
    cumprod_cos = torch.fliplr(torch.cumprod(torch.hstack((torch.ones((N, 1)), cos_terms)), dim=1))
    sin_terms = torch.sin(y_trans.flip(dims=[1]) * torch.pi / 2.0)
    shape_terms = torch.hstack((torch.ones((N, 1)), sin_terms))
    f = cumprod_cos * shape_terms

    PopObj = f * (1.0 + g)[:, None]
    return PopObj


def MMMOP4B(x: torch.Tensor):
    """Evaluate the MMMOP4B benchmark formula."""
    N = x.shape[0]
    M = 3
    kA = 0
    kB = 5
    c = 0.0
    d = 3

    PopDec = x
    y_trans = _transform_y_mmmop4(PopDec[:, : M - 1], d)

    term1 = (
        torch.sum(torch.cos(2.0 * torch.pi * c * PopDec[:, M - 1 : M + kA - 1]), dim=1)
        if kA > 0
        else torch.tensor(0.0)
    )
    g_vars = PopDec[:, M + kA - 1 :]
    term2 = torch.sum((g_vars - 0.5) ** 2 - torch.cos(20.0 * torch.pi * (g_vars - 0.5)), dim=1)
    g = 100.0 * (kA + kB + term1 + term2)

    cos_terms = torch.cos(y_trans * torch.pi / 2.0)
    cumprod_cos = torch.fliplr(torch.cumprod(torch.hstack((torch.ones((N, 1)), cos_terms)), dim=1))
    sin_terms = torch.sin(y_trans.flip(dims=[1]) * torch.pi / 2.0)
    shape_terms = torch.hstack((torch.ones((N, 1)), sin_terms))
    f = cumprod_cos * shape_terms

    PopObj = f * (1.0 + g)[:, None]
    return PopObj


def MMMOP4C(x: torch.Tensor):
    """Evaluate the MMMOP4C benchmark formula."""
    N = x.shape[0]
    M = 2
    kA = 1
    kB = 4
    c = 2.0
    d = 4

    PopDec = x
    y_trans = _transform_y_mmmop4(PopDec[:, : M - 1], d)

    term1 = (
        torch.sum(torch.cos(2.0 * torch.pi * c * PopDec[:, M - 1 : M + kA - 1]), dim=1)
        if kA > 0
        else torch.tensor(0.0)
    )
    g_vars = PopDec[:, M + kA - 1 :]
    term2 = torch.sum((g_vars - 0.5) ** 2 - torch.cos(20.0 * torch.pi * (g_vars - 0.5)), dim=1)
    g = 100.0 * (kA + kB + term1 + term2)

    cos_terms = torch.cos(y_trans * torch.pi / 2.0)
    cumprod_cos = torch.fliplr(torch.cumprod(torch.hstack((torch.ones((N, 1)), cos_terms)), dim=1))
    sin_terms = torch.sin(y_trans.flip(dims=[1]) * torch.pi / 2.0)
    shape_terms = torch.hstack((torch.ones((N, 1)), sin_terms))
    f = cumprod_cos * shape_terms

    PopObj = f * (1.0 + g)[:, None]
    return PopObj


def MMMOP4D(x: torch.Tensor):
    """Evaluate the MMMOP4D benchmark formula."""
    N = x.shape[0]
    M = 3
    kA = 1
    kB = 4
    c = 2.0
    d = 3

    PopDec = x
    y_trans = _transform_y_mmmop4(PopDec[:, : M - 1], d)

    term1 = (
        torch.sum(torch.cos(2.0 * torch.pi * c * PopDec[:, M - 1 : M + kA - 1]), dim=1)
        if kA > 0
        else torch.tensor(0.0)
    )
    g_vars = PopDec[:, M + kA - 1 :]
    term2 = torch.sum((g_vars - 0.5) ** 2 - torch.cos(20.0 * torch.pi * (g_vars - 0.5)), dim=1)
    g = 100.0 * (kA + kB + term1 + term2)

    cos_terms = torch.cos(y_trans * torch.pi / 2.0)
    cumprod_cos = torch.fliplr(torch.cumprod(torch.hstack((torch.ones((N, 1)), cos_terms)), dim=1))
    sin_terms = torch.sin(y_trans.flip(dims=[1]) * torch.pi / 2.0)
    shape_terms = torch.hstack((torch.ones((N, 1)), sin_terms))
    f = cumprod_cos * shape_terms

    PopObj = f * (1.0 + g)[:, None]
    return PopObj


def MMMOP5A(x: torch.Tensor):
    """Evaluate the MMMOP5A benchmark formula."""
    N = x.shape[0]
    M = 2
    kA = 0
    kB = 1
    c = 0.0
    d = 3

    PopDec = x
    y_trans = _transform_y_mmmop5(PopDec[:, : M - 1], d)

    term1 = (
        torch.sum(torch.cos(2.0 * torch.pi * c * PopDec[:, M - 1 : M + kA - 1]), dim=1)
        if kA > 0
        else torch.tensor(0.0)
    )
    g_vars = PopDec[:, M + kA - 1 :]
    term2 = torch.sum((g_vars - 0.5) ** 2 - torch.cos(20.0 * torch.pi * (g_vars - 0.5)), dim=1)
    g = 100.0 * (kA + kB + term1 + term2)

    cos_terms = torch.cos(y_trans * torch.pi / 2.0)
    cumprod_cos = torch.fliplr(torch.cumprod(torch.hstack((torch.ones((N, 1)), cos_terms)), dim=1))
    sin_terms = torch.sin(y_trans.flip(dims=[1]) * torch.pi / 2.0)
    shape_terms = torch.hstack((torch.ones((N, 1)), sin_terms))
    f = cumprod_cos * shape_terms

    PopObj = f * (1.0 + g)[:, None]
    return PopObj


def MMMOP5B(x: torch.Tensor):
    """Evaluate the MMMOP5B benchmark formula."""
    N = x.shape[0]
    M = 3
    kA = 0
    kB = 5
    c = 0.0
    d = 1

    PopDec = x
    y_trans = _transform_y_mmmop5(PopDec[:, : M - 1], d)

    term1 = (
        torch.sum(torch.cos(2.0 * torch.pi * c * PopDec[:, M - 1 : M + kA - 1]), dim=1)
        if kA > 0
        else torch.tensor(0.0)
    )
    g_vars = PopDec[:, M + kA - 1 :]
    term2 = torch.sum((g_vars - 0.5) ** 2 - torch.cos(20.0 * torch.pi * (g_vars - 0.5)), dim=1)
    g = 100.0 * (kA + kB + term1 + term2)

    cos_terms = torch.cos(y_trans * torch.pi / 2.0)
    cumprod_cos = torch.fliplr(torch.cumprod(torch.hstack((torch.ones((N, 1)), cos_terms)), dim=1))
    sin_terms = torch.sin(y_trans.flip(dims=[1]) * torch.pi / 2.0)
    shape_terms = torch.hstack((torch.ones((N, 1)), sin_terms))
    f = cumprod_cos * shape_terms

    PopObj = f * (1.0 + g)[:, None]
    return PopObj


def MMMOP5C(x: torch.Tensor):
    """Evaluate the MMMOP5C benchmark formula."""
    N = x.shape[0]
    M = 2
    kA = 1
    kB = 4
    c = 2.0
    d = 2

    PopDec = x
    y_trans = _transform_y_mmmop5(PopDec[:, : M - 1], d)

    term1 = (
        torch.sum(torch.cos(2.0 * torch.pi * c * PopDec[:, M - 1 : M + kA - 1]), dim=1)
        if kA > 0
        else torch.tensor(0.0)
    )
    g_vars = PopDec[:, M + kA - 1 :]
    term2 = torch.sum((g_vars - 0.5) ** 2 - torch.cos(20.0 * torch.pi * (g_vars - 0.5)), dim=1)
    g = 100.0 * (kA + kB + term1 + term2)

    cos_terms = torch.cos(y_trans * torch.pi / 2.0)
    cumprod_cos = torch.fliplr(torch.cumprod(torch.hstack((torch.ones((N, 1)), cos_terms)), dim=1))
    sin_terms = torch.sin(y_trans.flip(dims=[1]) * torch.pi / 2.0)
    shape_terms = torch.hstack((torch.ones((N, 1)), sin_terms))
    f = cumprod_cos * shape_terms

    PopObj = f * (1.0 + g)[:, None]
    return PopObj


def MMMOP5D(x: torch.Tensor):
    """Evaluate the MMMOP5D benchmark formula."""
    N = x.shape[0]
    M = 3
    kA = 1
    kB = 4
    c = 2.0
    d = 1

    PopDec = x
    y_trans = _transform_y_mmmop5(PopDec[:, : M - 1], d)

    term1 = (
        torch.sum(torch.cos(2.0 * torch.pi * c * PopDec[:, M - 1 : M + kA - 1]), dim=1)
        if kA > 0
        else torch.tensor(0.0)
    )
    g_vars = PopDec[:, M + kA - 1 :]
    term2 = torch.sum((g_vars - 0.5) ** 2 - torch.cos(20.0 * torch.pi * (g_vars - 0.5)), dim=1)
    g = 100.0 * (kA + kB + term1 + term2)

    cos_terms = torch.cos(y_trans * torch.pi / 2.0)
    cumprod_cos = torch.fliplr(torch.cumprod(torch.hstack((torch.ones((N, 1)), cos_terms)), dim=1))
    sin_terms = torch.sin(y_trans.flip(dims=[1]) * torch.pi / 2.0)
    shape_terms = torch.hstack((torch.ones((N, 1)), sin_terms))
    f = cumprod_cos * shape_terms

    PopObj = f * (1.0 + g)[:, None]
    return PopObj


def MMMOP6A(x: torch.Tensor):
    """Evaluate the MMMOP6A benchmark formula."""
    N = x.shape[0]
    M = 2
    kA = 0
    kB = 1
    c = 2.0

    PopDec = x
    y_g = (PopDec[:, M - 1 : M + kA - 1] - 0.5) * 12.0 if kA > 0 else torch.empty((N, 0))
    z_vars = PopDec[:, M + kA - 1 :]
    z = 2.0 * c * z_vars - 2.0 * torch.floor(c * z_vars) - 1.0

    i_vals = torch.arange(kB, dtype=PopDec.dtype, device=PopDec.device)
    j_vals = torch.arange(M - 1)
    sin_base = torch.sin(
        2.0 * torch.pi * PopDec[:, j_vals, None] + i_vals[None, None, :] * torch.pi / kB
    )
    t = torch.prod(sin_base, dim=1)

    g_term1 = torch.tensor(0.0)
    if kA >= 2:
        y1 = y_g[:, 0::2]
        y2 = y_g[:, 1::2]
        g_term1 = torch.sum((y1**2 + y2 - 11.0) ** 2 + (y1 + y2**2 - 7.0) ** 2, dim=1)

    g_term2 = torch.sum((z - t) ** 2, dim=1)
    g = g_term1 + g_term2

    cos_terms = torch.cos(PopDec[:, : M - 1] * torch.pi / 2.0)
    cumprod_cos = torch.fliplr(torch.cumprod(torch.hstack((torch.ones((N, 1)), cos_terms)), dim=1))
    sin_terms = torch.sin(PopDec[:, : M - 1].flip(dims=[1]) * torch.pi / 2.0)
    shape_terms = torch.hstack((torch.ones((N, 1)), sin_terms))
    f = cumprod_cos * shape_terms

    PopObj = f * (1.0 + g)[:, None]
    return PopObj


def MMMOP6B(x: torch.Tensor):
    """Evaluate the MMMOP6B benchmark formula."""
    N = x.shape[0]
    M = 3
    kA = 0
    kB = 2
    c = 2.0

    PopDec = x
    y_g = (PopDec[:, M - 1 : M + kA - 1] - 0.5) * 12.0 if kA > 0 else torch.empty((N, 0))
    z_vars = PopDec[:, M + kA - 1 :]
    z = 2.0 * c * z_vars - 2.0 * torch.floor(c * z_vars) - 1.0

    i_vals = torch.arange(kB, dtype=PopDec.dtype, device=PopDec.device)
    j_vals = torch.arange(M - 1)
    sin_base = torch.sin(
        2.0 * torch.pi * PopDec[:, j_vals, None] + i_vals[None, None, :] * torch.pi / kB
    )
    t = torch.prod(sin_base, dim=1)

    g_term1 = torch.tensor(0.0)
    if kA >= 2:
        y1 = y_g[:, 0::2]
        y2 = y_g[:, 1::2]
        g_term1 = torch.sum((y1**2 + y2 - 11.0) ** 2 + (y1 + y2**2 - 7.0) ** 2, dim=1)
    g_term2 = torch.sum((z - t) ** 2, dim=1)
    g = g_term1 + g_term2

    cos_terms = torch.cos(PopDec[:, : M - 1] * torch.pi / 2.0)
    cumprod_cos = torch.fliplr(torch.cumprod(torch.hstack((torch.ones((N, 1)), cos_terms)), dim=1))
    sin_terms = torch.sin(PopDec[:, : M - 1].flip(dims=[1]) * torch.pi / 2.0)
    shape_terms = torch.hstack((torch.ones((N, 1)), sin_terms))
    f = cumprod_cos * shape_terms

    PopObj = f * (1.0 + g)[:, None]
    return PopObj


def MMMOP6C(x: torch.Tensor):
    """Evaluate the MMMOP6C benchmark formula."""
    N = x.shape[0]
    M = 2
    kA = 2
    kB = 1
    c = 2.0

    PopDec = x
    y_g = (PopDec[:, M - 1 : M + kA - 1] - 0.5) * 12.0 if kA > 0 else torch.empty((N, 0))
    z_vars = PopDec[:, M + kA - 1 :]
    z = 2.0 * c * z_vars - 2.0 * torch.floor(c * z_vars) - 1.0

    i_vals = torch.arange(kB, dtype=PopDec.dtype, device=PopDec.device)
    j_vals = torch.arange(M - 1)
    sin_base = torch.sin(
        2.0 * torch.pi * PopDec[:, j_vals, None] + i_vals[None, None, :] * torch.pi / kB
    )
    t = torch.prod(sin_base, dim=1)

    g_term1 = torch.tensor(0.0)
    if kA >= 2:
        y1 = y_g[:, 0::2]
        y2 = y_g[:, 1::2]
        g_term1 = torch.sum((y1**2 + y2 - 11.0) ** 2 + (y1 + y2**2 - 7.0) ** 2, dim=1)
    g_term2 = torch.sum((z - t) ** 2, dim=1)
    g = g_term1 + g_term2

    cos_terms = torch.cos(PopDec[:, : M - 1] * torch.pi / 2.0)
    cumprod_cos = torch.fliplr(torch.cumprod(torch.hstack((torch.ones((N, 1)), cos_terms)), dim=1))
    sin_terms = torch.sin(PopDec[:, : M - 1].flip(dims=[1]) * torch.pi / 2.0)
    shape_terms = torch.hstack((torch.ones((N, 1)), sin_terms))
    f = cumprod_cos * shape_terms

    PopObj = f * (1.0 + g)[:, None]
    return PopObj


def MMMOP6D(x: torch.Tensor):
    """Evaluate the MMMOP6D benchmark formula."""
    N = x.shape[0]
    M = 3
    kA = 2
    kB = 1
    c = 2.0

    PopDec = x
    y_g = (PopDec[:, M - 1 : M + kA - 1] - 0.5) * 12.0 if kA > 0 else torch.empty((N, 0))
    z_vars = PopDec[:, M + kA - 1 :]
    z = 2.0 * c * z_vars - 2.0 * torch.floor(c * z_vars) - 1.0

    i_vals = torch.arange(kB, dtype=PopDec.dtype, device=PopDec.device)
    j_vals = torch.arange(M - 1)
    sin_base = torch.sin(
        2.0 * torch.pi * PopDec[:, j_vals, None] + i_vals[None, None, :] * torch.pi / kB
    )
    t = torch.prod(sin_base, dim=1)

    g_term1 = torch.tensor(0.0)
    if kA >= 2:
        y1 = y_g[:, 0::2]
        y2 = y_g[:, 1::2]
        g_term1 = torch.sum((y1**2 + y2 - 11.0) ** 2 + (y1 + y2**2 - 7.0) ** 2, dim=1)
    g_term2 = torch.sum((z - t) ** 2, dim=1)
    g = g_term1 + g_term2

    cos_terms = torch.cos(PopDec[:, : M - 1] * torch.pi / 2.0)
    cumprod_cos = torch.fliplr(torch.cumprod(torch.hstack((torch.ones((N, 1)), cos_terms)), dim=1))
    sin_terms = torch.sin(PopDec[:, : M - 1].flip(dims=[1]) * torch.pi / 2.0)
    shape_terms = torch.hstack((torch.ones((N, 1)), sin_terms))
    f = cumprod_cos * shape_terms

    PopObj = f * (1.0 + g)[:, None]
    return PopObj
