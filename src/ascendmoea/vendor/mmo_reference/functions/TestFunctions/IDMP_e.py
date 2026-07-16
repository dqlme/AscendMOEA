"""Implementation utilities for IDMP_e."""

import torch


def _nsidedpoly(n):
    """Generates vertices for a regular n-sided polygon centered at origin with radius 1."""
    if n < 3:
        return torch.empty((0, 2))
    angles = torch.linspace(-n + 1, n - 1, n) / n
    vertices = torch.stack((torch.sin(torch.pi * angles), torch.cos(torch.pi * angles)), dim=1)
    return vertices


def IDMPM2T1_e(x: torch.Tensor):
    """Evaluate the IDMPM2T1_e benchmark formula."""
    N = x.shape[0]
    M = 2
    NP = 2
    y = torch.full((N, M), float("nan"))

    a = 3.0
    psize = 0.10 * torch.ones(NP)
    center = torch.tensor([-0.50, 0.50])
    Points = torch.stack((center - psize, center + psize), dim=0)

    PopDec = x
    x1 = PopDec[:, 0]
    x2 = PopDec[:, 1]

    for i in range(M):
        temp_base = torch.abs(x1[:, None] - Points[i, :])
        temp = temp_base.clone()
        temp[:, 0] += 1.0 * (torch.abs(x2 + 0.5)) ** 1.0
        temp[:, 1] += a * (torch.abs(x2 - 0.5)) ** 1.0
        y[:, i] = torch.min(temp, dim=1).values

    index = PopDec[:, 1] > 0
    y[index, :] += 0.01

    return y


def IDMPM2T2_e(x: torch.Tensor):
    """Evaluate the IDMPM2T2_e benchmark formula."""
    N = x.shape[0]
    M = 2
    NP = 2
    y = torch.full((N, M), float("nan"))

    a = 0.4
    psize = 0.10 * torch.ones(NP)
    center = torch.tensor([-0.50, 0.50])
    Points = torch.stack((center - psize, center + psize), dim=0)

    PopDec = x
    x1 = PopDec[:, 0]
    x2 = PopDec[:, 1]

    for i in range(M):
        temp_base = torch.abs(x1[:, None] - Points[i, :])
        temp = temp_base.clone()
        temp[:, 0] += 100.0 * (torch.abs(x2 + 0.5)) ** 2.0
        temp[:, 1] += 100.0 * (torch.abs(x2 - 0.5)) ** (2.0 - a)
        y[:, i] = torch.min(temp, dim=1).values

    index = PopDec[:, 1] > 0
    y[index, :] += 0.01

    return y


def IDMPM2T3_e(x: torch.Tensor):
    """Evaluate the IDMPM2T3_e benchmark formula."""
    N = x.shape[0]
    M = 2
    NP = 2
    y = torch.full((N, M), float("nan"))

    a = 0.4
    psize = 0.10 * torch.ones(NP)
    center = torch.tensor([-0.50, 0.50])
    Points = torch.stack((center - psize, center + psize), dim=0)

    PopDec = x
    x1 = PopDec[:, 0]
    x2 = PopDec[:, 1]

    for i in range(M):
        temp_base = torch.abs(x1[:, None] - Points[i, :])
        temp = temp_base.clone()
        temp[:, 0] += 100.0 * (1.0 - torch.cos(2.0 * torch.pi * (x2 + 0.5)))
        temp[:, 1] += 100.0 * ((x2 - 0.5 + a * (x1 - 0.5)) ** 2)
        y[:, i] = torch.min(temp, dim=1).values

    index = PopDec[:, 0] > 0
    y[index, :] += 0.01

    return y


def IDMPM2T4_e(x: torch.Tensor):
    """Evaluate the IDMPM2T4_e benchmark formula."""
    N = x.shape[0]
    M = 2
    NP = 2
    y = torch.full((N, M), float("nan"))

    psize = 0.10 * torch.ones(NP)
    center = torch.tensor([-0.50, 0.50])
    Points = torch.stack((center - psize, center + psize), dim=0)

    PopDec = x
    x1 = PopDec[:, 0]
    x2 = PopDec[:, 1]

    for i in range(M):
        temp_base = torch.abs(x1[:, None] - Points[i, :])
        temp = temp_base.clone()
        temp[:, 0] += 100.0 * (1.0 - torch.cos(1.0 * 2.0 * torch.pi * (x2 + 0.5)))
        temp[:, 1] += 100.0 * (1.0 - torch.cos(4.0 * torch.pi * (x2 - 0.5)))
        y[:, i] = torch.min(temp, dim=1).values

    index1 = PopDec[:, 0] > 0
    y[index1, :] += 0.01
    index2 = PopDec[:, 1] > 0.2
    y[index2, :] += 0.01

    return y


def IDMPM3T1_e(x: torch.Tensor):
    """Evaluate the IDMPM3T1_e benchmark formula."""
    if x.shape[1] < 3:
        raise ValueError("IDMPM3T1_e requires at least 3 dimensions")
    N = x.shape[0]
    M = 3
    NP = 4
    y = torch.full((N, M), float("nan"))

    pgon_verts = _nsidedpoly(M)
    psize = torch.tensor([0.1, 0.1, 0.1, 0.1])
    center = torch.tensor([[-0.50, -0.50], [0.50, -0.50], [0.50, 0.50], [-0.50, 0.50]])
    Points = torch.zeros((M, 2, NP))
    for i in range(NP):
        Points[:, :, i] = pgon_verts * psize[i] + center[i, :]

    PopDec = x
    x12 = PopDec[:, 0:2]
    x3 = PopDec[:, 2]

    for i in range(M):
        points_for_obj_i = Points[i, :, :].T
        dist_base = torch.cdist(x12, points_for_obj_i)
        temp = dist_base.clone()
        temp[:, 0] += 1.0 * (torch.abs(x3 + 0.6))
        temp[:, 1] += 2.0 * (torch.abs(x3 + 0.2))
        temp[:, 2] += 1.0 * (torch.abs(x3 - 0.2))
        temp[:, 3] += 2.0 * (torch.abs(x3 - 0.6))
        y[:, i] = torch.min(temp, dim=1).values

    index = PopDec[:, 1] > 0
    y[index, :] += 0.03

    return y


def IDMPM3T2_e(x: torch.Tensor):
    """Evaluate the IDMPM3T2_e benchmark formula."""
    if x.shape[1] < 3:
        raise ValueError("IDMPM3T2_e requires at least 3 dimensions")
    N = x.shape[0]
    M = 3
    NP = 4
    y = torch.full((N, M), float("nan"))

    pgon_verts = _nsidedpoly(M)
    psize = torch.tensor([0.1, 0.1, 0.1, 0.1])
    center = torch.tensor([[-0.50, -0.50], [0.50, -0.50], [0.50, 0.50], [-0.50, 0.50]])
    Points = torch.zeros((M, 2, NP))
    for i in range(NP):
        Points[:, :, i] = pgon_verts * psize[i] + center[i, :]

    PopDec = x
    x12 = PopDec[:, 0:2]
    x3 = PopDec[:, 2]

    for i in range(M):
        points_for_obj_i = Points[i, :, :].T
        dist_base = torch.cdist(x12, points_for_obj_i)
        temp = dist_base.clone()
        temp[:, 0] += 100.0 * (torch.abs(x3 + 0.6)) ** 2.0
        temp[:, 1] += 100.0 * (torch.abs(x3 + 0.2)) ** 1.8
        temp[:, 2] += 100.0 * (torch.abs(x3 - 0.2)) ** 1.6
        temp[:, 3] += 100.0 * (torch.abs(x3 - 0.6)) ** 1.4
        y[:, i] = torch.min(temp, dim=1).values

    index1 = PopDec[:, 1] > 0
    index2 = PopDec[:, 0] > 0
    y[index1 & index2, :] += 0.03
    y[~index1 & ~index2, :] += 0.06

    return y


def IDMPM3T3_e(x: torch.Tensor):
    """Evaluate the IDMPM3T3_e benchmark formula."""
    if x.shape[1] < 3:
        raise ValueError("IDMPM3T3_e requires at least 3 dimensions")
    N = x.shape[0]
    M = 3
    NP = 4
    y = torch.full((N, M), float("nan"))

    pgon_verts = _nsidedpoly(M)
    psize = torch.tensor([0.1, 0.1, 0.1, 0.1])
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

    for i in range(M):
        points_for_obj_i = Points[i, :, :].T
        dist_base = torch.cdist(x12, points_for_obj_i)
        temp = dist_base.clone()
        temp[:, 0] += 100.0 * (x3 + 0.6) ** 2
        temp[:, 1] += 100.0 * (x3 + 0.2 + 0.1 * t2) ** 2
        temp[:, 2] += 100.0 * (x3 - 0.2 + 0.2 * t3) ** 2
        temp[:, 3] += 100.0 * (x3 - 0.6 + 0.3 * t4) ** 2
        y[:, i] = torch.min(temp, dim=1).values

    index1 = PopDec[:, 1] < 0
    index2 = PopDec[:, 0] > 0
    y[index1 & index2, :] += 0.03
    y[~index1 & ~index2, :] += 0.06

    return y


def IDMPM3T4_e(x: torch.Tensor):
    """Evaluate the IDMPM3T4_e benchmark formula."""
    if x.shape[1] < 3:
        raise ValueError("IDMPM3T4_e requires at least 3 dimensions")
    N = x.shape[0]
    M = 3
    NP = 4
    y = torch.full((N, M), float("nan"))

    pgon_verts = _nsidedpoly(M)
    psize = torch.tensor([0.1, 0.1, 0.1, 0.1])
    center = torch.tensor([[-0.50, -0.50], [0.50, -0.50], [0.50, 0.50], [-0.50, 0.50]])
    Points = torch.zeros((M, 2, NP))
    for i in range(NP):
        Points[:, :, i] = pgon_verts * psize[i] + center[i, :]

    PopDec = x
    x12 = PopDec[:, 0:2]
    x3 = PopDec[:, 2]

    for i in range(M):
        points_for_obj_i = Points[i, :, :].T
        dist_base = torch.cdist(x12, points_for_obj_i)
        temp = dist_base.clone()
        temp[:, 0] += 100.0 * (1.0 - torch.cos(1.0 * 2.0 * torch.pi * (x3 + 0.6)))
        temp[:, 1] += 100.0 * (1.0 - torch.cos(1.0 * 2.0 * torch.pi * (x3 + 0.2)))
        temp[:, 2] += 100.0 * (1.0 - torch.cos(2.0 * torch.pi * (x3 - 0.2)))
        temp[:, 3] += 100.0 * (1.0 - torch.cos(2.0 * torch.pi * (x3 - 0.6)))
        y[:, i] = torch.min(temp, dim=1).values

    index1 = PopDec[:, 1] > 0
    index2 = PopDec[:, 0] > 0
    y[index1 & index2, :] += 0.03
    y[~index1 & ~index2, :] += 0.06
    y[index1 & ~index2, :] += 0.06

    return y
