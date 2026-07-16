"""Implementation utilities for IDMP."""

import torch


def _nsidedpoly(n):
    """Generates vertices for a regular n-sided polygon centered at origin with radius 1."""
    if n < 3:
        return torch.empty((0, 2))
    angles = torch.linspace(-n + 1, n - 1, n) / n
    vertices = torch.stack((torch.sin(torch.pi * angles), torch.cos(torch.pi * angles)), dim=1)
    return vertices


def IDMPM2T1(x: torch.Tensor):
    """Evaluate the IDMPM2T1 benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    y = torch.zeros((x.shape[0], 2))
    mu = 3.0
    x1 = x[:, 0]
    x2 = x[:, 1]

    g1 = torch.abs(x2 + 0.5)
    g2 = mu * torch.abs(x2 - 0.5)

    f1_comp1 = torch.abs(x1 + 0.6) + g1
    f1_comp2 = torch.abs(x1 - 0.4) + g2
    y[:, 0] = torch.minimum(f1_comp1, f1_comp2)

    f2_comp1 = torch.abs(x1 + 0.4) + g1
    f2_comp2 = torch.abs(x1 - 0.6) + g2
    y[:, 1] = torch.minimum(f2_comp1, f2_comp2)

    return y


def IDMPM2T2(x: torch.Tensor):
    """Evaluate the IDMPM2T2 benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    M = 2
    N = x.shape[0]
    y = torch.full((N, M), float("nan"))

    NP = 2
    psize = torch.tensor([0.10] * NP)
    center = torch.tensor([-0.50, 0.50])
    Points = torch.stack((center - psize, center + psize), dim=0)

    a = 0.4
    x1 = x[:, 0]
    x2 = x[:, 1]

    temp1 = torch.abs(x1[:, None] - Points[0, :])
    term_g1 = 100.0 * (torch.abs(x2 + 0.5)) ** 2
    term_g2 = 100.0 * (torch.abs(x2 - 0.5)) ** (2.0 - a)
    temp1[:, 0] += term_g1
    temp1[:, 1] += term_g2
    y[:, 0] = torch.min(temp1, dim=1).values

    temp2 = torch.abs(x1[:, None] - Points[1, :])
    temp2[:, 0] += term_g1
    temp2[:, 1] += term_g2
    y[:, 1] = torch.min(temp2, dim=1).values

    return y


def IDMPM2T3(x: torch.Tensor):
    """Evaluate the IDMPM2T3 benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    N = x.shape[0]
    M = 2
    y = torch.full((N, M), float("nan"))

    NP = 2
    psize = torch.tensor([0.10] * NP)
    center = torch.tensor([-0.50, 0.50])
    Points = torch.stack((center - psize, center + psize), dim=0)

    a = 0.4
    PopDec = x
    x1 = PopDec[:, 0]
    x2 = PopDec[:, 1]

    temp1_base = torch.abs(x1[:, None] - Points[0, :])
    temp1 = temp1_base.clone()
    term_g1 = 100.0 * torch.abs(x2 + 0.5) ** 2
    term_g2 = 100.0 * (x2 - 0.5 + a * (x1 - 0.5)) ** 2
    temp1[:, 0] += term_g1
    temp1[:, 1] += term_g2
    y[:, 0] = torch.min(temp1, dim=1).values

    temp2_base = torch.abs(x1[:, None] - Points[1, :])
    temp2 = temp2_base.clone()
    temp2[:, 0] += term_g1
    temp2[:, 1] += term_g2
    y[:, 1] = torch.min(temp2, dim=1).values

    return y


def IDMPM2T4(x: torch.Tensor):
    """Evaluate the IDMPM2T4 benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    M = 2
    N = x.shape[0]
    y = torch.full((N, M), float("nan"))

    NP = 2
    psize = torch.tensor([0.10] * NP)
    center = torch.tensor([-0.50, 0.50])
    Points = torch.stack((center - psize, center + psize), dim=0)

    a = 4.0
    PopDec = x
    x1 = PopDec[:, 0]
    x2 = PopDec[:, 1]

    g1_term = 100.0 * (
        torch.abs(x2 + 0.50) ** 2 + 1.0 - torch.cos(1.0 * 2.0 * torch.pi * (x2 + 0.5))
    )
    g2_term = 100.0 * (torch.abs(x2 - 0.50) ** 2 + 1.0 - torch.cos(a * 2.0 * torch.pi * (x2 - 0.5)))

    temp1_base = torch.abs(x1[:, None] - Points[0, :])
    temp1 = temp1_base.clone()
    temp1[:, 0] += g1_term
    temp1[:, 1] += g2_term
    y[:, 0] = torch.min(temp1, dim=1).values

    temp2_base = torch.abs(x1[:, None] - Points[1, :])
    temp2 = temp2_base.clone()
    temp2[:, 0] += g1_term
    temp2[:, 1] += g2_term
    y[:, 1] = torch.min(temp2, dim=1).values

    return y


def IDMPM3T1(x: torch.Tensor):
    """Evaluate the IDMPM3T1 benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] < 3:
        raise ValueError("IDMPM3T1 requires at least 3 dimensions")
    N = x.shape[0]
    M = 3
    NP = 4
    y = torch.full((N, M), float("nan"))

    pgon_verts = _nsidedpoly(M)

    psize = torch.tensor([0.1] * NP)
    center = torch.tensor([[-0.50, -0.50], [0.50, -0.50], [0.50, 0.50], [-0.50, 0.50]])

    Points = torch.zeros((M, 2, NP))
    for i in range(NP):
        Points[:, :, i] = pgon_verts * psize[i] + center[i, :]

    PopDec = x
    x12 = PopDec[:, 0:2]
    x3 = PopDec[:, 2]

    g_terms = torch.stack(
        [
            torch.abs(x3 + 0.6),
            2.0 * torch.abs(x3 + 0.2),
            3.0 * torch.abs(x3 - 0.2),
            4.0 * torch.abs(x3 - 0.6),
        ],
        dim=1,
    )

    for i in range(M):
        points_for_obj_i = Points[i, :, :].T
        dist_base = torch.cdist(x12, points_for_obj_i)
        temp = dist_base + g_terms
        y[:, i] = torch.min(temp, dim=1).values

    return y


def IDMPM3T2(x: torch.Tensor):
    """Evaluate the IDMPM3T2 benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] < 3:
        raise ValueError("IDMPM3T2 requires at least 3 dimensions")
    N = x.shape[0]
    M = 3
    NP = 4
    y = torch.full((N, M), float("nan"))

    pgon_verts = _nsidedpoly(M)

    psize = torch.tensor([0.1] * NP)
    center = torch.tensor([[-0.50, -0.50], [0.50, -0.50], [0.50, 0.50], [-0.50, 0.50]])

    Points = torch.zeros((M, 2, NP))
    for i in range(NP):
        Points[:, :, i] = pgon_verts * psize[i] + center[i, :]

    PopDec = x
    x12 = PopDec[:, 0:2]
    x3 = PopDec[:, 2]

    g_terms = 100.0 * torch.stack(
        [
            (torch.abs(x3 + 0.6)) ** 2.0,
            (torch.abs(x3 + 0.2)) ** 1.8,
            (torch.abs(x3 - 0.2)) ** 1.6,
            (torch.abs(x3 - 0.6)) ** 1.4,
        ],
        dim=1,
    )

    for i in range(M):
        points_for_obj_i = Points[i, :, :].T
        dist_base = torch.cdist(x12, points_for_obj_i)
        temp = dist_base + g_terms
        y[:, i] = torch.min(temp, dim=1).values

    return y


def IDMPM3T3(x: torch.Tensor):
    """Evaluate the IDMPM3T3 benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] < 3:
        raise ValueError("IDMPM3T3 requires at least 3 dimensions")
    N = x.shape[0]
    M = 3
    NP = 4
    y = torch.full((N, M), float("nan"))

    pgon_verts = _nsidedpoly(M)

    psize = torch.tensor([0.1] * NP)
    center = torch.tensor([[-0.50, -0.50], [0.50, -0.50], [0.50, 0.50], [-0.50, 0.50]])

    Points = torch.zeros((M, 2, NP))
    for i in range(NP):
        Points[:, :, i] = pgon_verts * psize[i] + center[i, :]

    PopDec = x
    x1 = PopDec[:, 0]
    x2 = PopDec[:, 1]
    x3 = PopDec[:, 2]
    x12 = PopDec[:, 0:2]

    t2 = x1 - 0.5 + x2 + 0.5
    t3 = x1 - 0.5 + x2 - 0.5
    t4 = x1 + 0.5 + x2 - 0.5

    g_terms = 100.0 * torch.stack(
        [
            (x3 + 0.6) ** 2,
            (x3 + 0.2 + 0.1 * t2) ** 2,
            (x3 - 0.2 + 0.2 * t3) ** 2,
            (x3 - 0.6 + 0.3 * t4) ** 2,
        ],
        dim=1,
    )

    for i in range(M):
        points_for_obj_i = Points[i, :, :].T
        dist_base = torch.cdist(x12, points_for_obj_i)
        temp = dist_base + g_terms
        y[:, i] = torch.min(temp, dim=1).values

    return y


def IDMPM3T4(x: torch.Tensor):
    """Evaluate the IDMPM3T4 benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] < 3:
        raise ValueError("IDMPM3T4 requires at least 3 dimensions")
    N = x.shape[0]
    M = 3
    NP = 4
    y = torch.full((N, M), float("nan"))

    pgon_verts = _nsidedpoly(M)

    psize = torch.tensor([0.1] * NP)
    center = torch.tensor([[-0.50, -0.50], [0.50, -0.50], [0.50, 0.50], [-0.50, 0.50]])

    Points = torch.zeros((M, 2, NP))
    for i in range(NP):
        Points[:, :, i] = pgon_verts * psize[i] + center[i, :]

    PopDec = x
    x12 = PopDec[:, 0:2]
    x3 = PopDec[:, 2]

    g_terms = 100.0 * torch.stack(
        [
            ((x3 + 0.6) ** 2 + 1.0 - torch.cos(1.0 * 2.0 * torch.pi * (x3 + 0.6))),
            ((x3 + 0.2) ** 2 + 1.0 - torch.cos(2.0 * 2.0 * torch.pi * (x3 + 0.2))),
            ((x3 - 0.2) ** 2 + 1.0 - torch.cos(3.0 * 2.0 * torch.pi * (x3 - 0.2))),
            ((x3 - 0.6) ** 2 + 1.0 - torch.cos(4.0 * 2.0 * torch.pi * (x3 - 0.6))),
        ],
        dim=1,
    )

    for i in range(M):
        points_for_obj_i = Points[i, :, :].T
        dist_base = torch.cdist(x12, points_for_obj_i)
        temp = dist_base + g_terms
        y[:, i] = torch.min(temp, dim=1).values

    return y


def IDMPM4T1(x: torch.Tensor):
    """Evaluate the IDMPM4T1 benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] < 4:
        raise ValueError("IDMPM4T1 requires at least 4 dimensions")
    N = x.shape[0]
    M = 4
    NP = 4
    y = torch.full((N, M), float("nan"))

    pgon_verts = _nsidedpoly(M)

    psize = torch.tensor([0.1] * NP)
    center = torch.tensor([[-0.50, -0.50], [0.50, -0.50], [0.50, 0.50], [-0.50, 0.50]])

    Points = torch.zeros((M, 2, NP))
    for i in range(NP):
        Points[:, :, i] = pgon_verts * psize[i] + center[i, :]

    PopDec = x
    x12 = PopDec[:, 0:2]
    x3 = PopDec[:, 2]
    x4 = PopDec[:, 3]

    g_terms = torch.stack(
        [
            1.0 * (torch.abs(x3 + 0.6) + torch.abs(x4 + 0.6)),
            2.0 * (torch.abs(x3 + 0.2) + torch.abs(x4 + 0.2)),
            3.0 * (torch.abs(x3 - 0.2) + torch.abs(x4 - 0.2)),
            4.0 * (torch.abs(x3 - 0.6) + torch.abs(x4 - 0.6)),
        ],
        dim=1,
    )

    for i in range(M):
        points_for_obj_i = Points[i, :, :].T
        dist_base = torch.cdist(x12, points_for_obj_i)
        temp = dist_base + g_terms
        y[:, i] = torch.min(temp, dim=1).values

    return y


def IDMPM4T2(x: torch.Tensor):
    """Evaluate the IDMPM4T2 benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] < 4:
        raise ValueError("IDMPM4T2 requires at least 4 dimensions")
    N = x.shape[0]
    M = 4
    NP = 4
    y = torch.full((N, M), float("nan"))

    pgon_verts = _nsidedpoly(M)

    psize = torch.tensor([0.1] * NP)
    center = torch.tensor([[-0.50, -0.50], [0.50, -0.50], [0.50, 0.50], [-0.50, 0.50]])

    Points = torch.zeros((M, 2, NP))
    for i in range(NP):
        Points[:, :, i] = pgon_verts * psize[i] + center[i, :]

    PopDec = x
    x12 = PopDec[:, 0:2]
    x3 = PopDec[:, 2]
    x4 = PopDec[:, 3]

    g_terms = 100.0 * torch.stack(
        [
            ((torch.abs(x3 + 0.6)) ** 2.0 + (torch.abs(x4 + 0.6)) ** 2.0),
            ((torch.abs(x3 + 0.2)) ** 1.8 + (torch.abs(x4 + 0.2)) ** 1.8),
            ((torch.abs(x3 - 0.2)) ** 1.6 + (torch.abs(x4 - 0.2)) ** 1.6),
            ((torch.abs(x3 - 0.6)) ** 1.4 + (torch.abs(x4 - 0.6)) ** 1.4),
        ],
        dim=1,
    )

    for i in range(M):
        points_for_obj_i = Points[i, :, :].T
        dist_base = torch.cdist(x12, points_for_obj_i)
        temp = dist_base + g_terms
        y[:, i] = torch.min(temp, dim=1).values

    return y


def IDMPM4T3(x: torch.Tensor):
    """Evaluate the IDMPM4T3 benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] < 4:
        raise ValueError("IDMPM4T3 requires at least 4 dimensions")
    N = x.shape[0]
    M = 4
    NP = 4
    y = torch.full((N, M), float("nan"))

    pgon_verts = _nsidedpoly(M)

    psize = torch.tensor([0.1] * NP)
    center = torch.tensor([[-0.50, -0.50], [0.50, -0.50], [0.50, 0.50], [-0.50, 0.50]])

    Points = torch.zeros((M, 2, NP))
    for i in range(NP):
        Points[:, :, i] = pgon_verts * psize[i] + center[i, :]

    PopDec = x
    x1 = PopDec[:, 0]
    x2 = PopDec[:, 1]
    x3 = PopDec[:, 2]
    x4 = PopDec[:, 3]
    x12 = PopDec[:, 0:2]

    t2 = x1 - 0.5 + x2 + 0.5
    t3 = x1 - 0.5 + x2 - 0.5
    t4 = x1 + 0.5 + x2 - 0.5
    a2, a3, a4 = 0.05, 0.1, 0.15

    g_terms = 100.0 * torch.stack(
        [
            ((x3 + 0.6) ** 2 + (x4 + 0.6) ** 2),
            ((x3 + 0.2 + a2 * t2) ** 2 + (x4 + 0.2 + a2 * t2) ** 2),
            ((x3 - 0.2 + a3 * t3) ** 2 + (x4 - 0.2 + a3 * t3) ** 2),
            ((x3 - 0.6 + a4 * t4) ** 2 + (x4 - 0.6 + a4 * t4) ** 2),
        ],
        dim=1,
    )

    for i in range(M):
        points_for_obj_i = Points[i, :, :].T
        dist_base = torch.cdist(x12, points_for_obj_i)
        temp = dist_base + g_terms
        y[:, i] = torch.min(temp, dim=1).values

    return y


def IDMPM4T4(x: torch.Tensor):
    """Evaluate the IDMPM4T4 benchmark formula."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] < 4:
        raise ValueError("IDMPM4T4 requires at least 4 dimensions")
    N = x.shape[0]
    M = 4
    NP = 4
    y = torch.full((N, M), float("nan"))

    pgon_verts = _nsidedpoly(M)

    psize = torch.tensor([0.1] * NP)
    center = torch.tensor([[-0.50, -0.50], [0.50, -0.50], [0.50, 0.50], [-0.50, 0.50]])

    Points = torch.zeros((M, 2, NP))
    for i in range(NP):
        Points[:, :, i] = pgon_verts * psize[i] + center[i, :]

    PopDec = x
    x12 = PopDec[:, 0:2]
    x3 = PopDec[:, 2]
    x4 = PopDec[:, 3]

    cos_term_x4_zero = 1.0

    g_terms = 100.0 * torch.stack(
        [
            ((x3 + 0.6) ** 2 + 1.0 - torch.cos(1.0 * 2.0 * torch.pi * (x3 + 0.6)))
            + ((x4 + 0.6) ** 2 + 1.0 - cos_term_x4_zero),
            ((x3 + 0.2) ** 2 + 1.0 - torch.cos(2.0 * 2.0 * torch.pi * (x3 + 0.2)))
            + ((x4 + 0.2) ** 2 + 1.0 - cos_term_x4_zero),
            ((x3 - 0.2) ** 2 + 1.0 - torch.cos(3.0 * 2.0 * torch.pi * (x3 - 0.2)))
            + ((x4 - 0.2) ** 2 + 1.0 - cos_term_x4_zero),
            ((x3 - 0.6) ** 2 + 1.0 - torch.cos(4.0 * 2.0 * torch.pi * (x3 - 0.6)))
            + ((x4 - 0.6) ** 2 + 1.0 - cos_term_x4_zero),
        ],
        dim=1,
    )

    for i in range(M):
        points_for_obj_i = Points[i, :, :].T
        dist_base = torch.cdist(x12, points_for_obj_i)
        temp = dist_base + g_terms
        y[:, i] = torch.min(temp, dim=1).values

    return y
