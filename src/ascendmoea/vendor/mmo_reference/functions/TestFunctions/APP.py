"""Implementation utilities for APP."""

import numpy as np
import torch


def _euclidean_distances(x: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Compute small 2-D distance tables without cdist cancellation."""
    difference = x[:, None, :] - points[None, :, :]
    return torch.sqrt(torch.sum(difference * difference, dim=2))


def map(x: torch.Tensor):
    """Evaluate the map benchmark formula."""
    es = torch.tensor([[3.0, 37], [42, 96], [45, 60], [50, 25], [83, 72], [98, 38]])
    js = torch.tensor([[40.0, 20], [51, 60], [95, 51]])
    cs = torch.tensor(
        [
            [10.0, 55],
            [15, 15],
            [15, 78],
            [15, 88],
            [20, 23],
            [20, 70],
            [32, 42],
            [35, 60],
            [40, 76],
            [52, 78],
            [52, 96],
            [55, 33],
            [75, 27],
        ]
    )
    rs = torch.tensor([[17.5, 82.5], [55.5, 82.5], [94.5, 6.5]])

    if x.ndim == 1:
        x = x.reshape(1, -1)

    dist_e = _euclidean_distances(x, es)
    dist_j = _euclidean_distances(x, js)
    dist_c = _euclidean_distances(x, cs)
    dist_r = _euclidean_distances(x, rs)

    y = torch.stack(
        (
            torch.min(dist_e, dim=1).values,
            torch.min(dist_j, dim=1).values,
            torch.min(dist_c, dim=1).values,
            torch.min(dist_r, dim=1).values,
        ),
        dim=0,
    ).T

    return y


def application(x: torch.Tensor):
    """Evaluate the application benchmark formula."""
    A = torch.tensor(
        [
            [0.1, 1.4],
            [0.78, 1.48],
            [2.02, 0.56],
            [
                2.02 - np.cos(np.pi * 5 / 12) / 2 + np.sqrt(2) / 4,
                0.56 + np.sin(np.pi * 5 / 12) / 2 + np.sqrt(2) / 4,
            ],
            [0.75, 0.47],
        ]
    )
    B = torch.tensor(
        [
            [0.1, 0.9],
            [0.78 + np.sin(np.pi / 9) / 2, 1.48 - np.cos(np.pi / 9) / 2],
            [2.02 + np.sqrt(2) / 4, 0.56 + np.sqrt(2) / 4],
            [0.34, 0.095],
        ]
    )
    C = torch.tensor(
        [
            [0.533, 1.15],
            [
                0.78 + np.sin(np.pi / 9) / 2 + np.cos(np.pi * 5 / 18) / 2,
                1.48 - np.cos(np.pi / 9) / 2 + np.sin(np.pi * 5 / 18) / 2,
            ],
            [2.02 - np.cos(np.pi * 5 / 12) / 2, 0.56 + np.sin(np.pi * 5 / 12) / 2],
            [0.995, 0.775],
            [0.925, 0.05],
            [1.965, 0.165],
            [1.62, 1.34],
            [1.45, 0.445],
            [2.335, 0.2],
            [0.325, 0.715],
        ]
    )

    if x.ndim == 1:
        x = x.reshape(1, -1)
    N = x.shape[0]
    y = torch.full((N, 3), float("nan"))

    x_coords = x[:, :2]

    dist_A = _euclidean_distances(x_coords, A)
    dist_B = _euclidean_distances(x_coords, B)
    dist_C = _euclidean_distances(x_coords, C)

    y[:, 0] = torch.min(dist_A, dim=1).values
    y[:, 1] = torch.min(dist_B, dim=1).values
    y[:, 2] = torch.min(dist_C, dim=1).values

    return y
