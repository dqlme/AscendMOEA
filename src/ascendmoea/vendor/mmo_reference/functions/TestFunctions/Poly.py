"""Implementation utilities for Poly."""

import torch


def _nsidedpoly(n):
    """Generates vertices for a regular n-sided polygon centered at origin with radius 1."""
    if n < 3:
        return torch.empty((0, 2))
    angles = torch.linspace(-n + 1, n - 1, n) / n
    vertices = torch.stack((torch.sin(torch.pi * angles), torch.cos(torch.pi * angles)), dim=1)
    return vertices


def poly_2(PopDec: torch.Tensor):
    """Evaluate the poly_2 benchmark formula."""
    N = PopDec.shape[0]
    M = 2
    PopObj = torch.full((N, M), float("nan"))

    Points = torch.tensor([[-0.6, 0.3], [-0.4, 0.7]])
    x1 = PopDec[:, 0]
    x2 = PopDec[:, 1]

    for i in range(M):
        temp = torch.abs(x1[:, None] - Points[i, :])
        temp[:, 0] += torch.abs(x2 + 0.5)
        temp[:, 1] += torch.abs(x2 - 0.5)
        PopObj[:, i] = torch.min(temp, dim=1).values

    return PopObj


def _create_poly_func(M_obj):
    """Creates a polygon test function for M_obj objectives."""

    def poly_func(PopDec: torch.Tensor):
        """Evaluate the poly_func benchmark formula."""
        if PopDec.shape[1] < 2:
            raise ValueError(f"poly_{M_obj} requires at least 2 dimensions")
        N = PopDec.shape[0]
        M = M_obj
        NP = 4
        PopObj = torch.full((N, M), float("nan"))

        pgon_verts = _nsidedpoly(M)
        if M <= 4:
            psize = torch.tensor([0.1, 0.2, 0.3, 0.15])
        else:
            psize = torch.tensor([0.1, 0.15, 0.2, 0.225])
        center = torch.tensor([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])

        Points = torch.zeros((M, 2, NP))
        for i in range(NP):
            Points[:, :, i] = pgon_verts * psize[i] + center[i, :]

        x12 = PopDec[:, 0:2]

        for i in range(M):
            points_for_obj_i = Points[i, :, :].T
            dist_base = torch.cdist(x12, points_for_obj_i)
            PopObj[:, i] = torch.min(dist_base, dim=1).values

        return PopObj

    poly_func.__name__ = f"poly_{M_obj}"
    return poly_func


poly_3 = _create_poly_func(3)
poly_4 = _create_poly_func(4)
poly_5 = _create_poly_func(5)
poly_6 = _create_poly_func(6)
poly_8 = _create_poly_func(8)
poly_10 = _create_poly_func(10)
poly_12 = _create_poly_func(12)
