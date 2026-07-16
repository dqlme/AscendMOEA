"""Vectorized implementation of the rwmop_base problem."""

from __future__ import annotations

import torch

from ..core import Problem, to_tensor


class RWMOPBase(Problem):
    """Shared plumbing for the independently implemented RWMOP problems."""

    reference_point: tuple[float, ...] = ()

    def __init__(self, n=100, max_fe=10000):
        """Initialize the instance."""
        super().__init__(n=n, m=None, d=None, max_fe=max_fe)

    def evaluation(self, dec):
        # These engineering problems evaluate the supplied decisions directly.
        """Repair and evaluate a complete decision batch."""
        dec = to_tensor(dec, device=self.device, dtype=self.dtype)
        if dec.ndim == 1:
            dec = dec.unsqueeze(0)
        if dec.shape[0] == 0:
            return self._empty_population()
        return self._pack_population(dec, self.cal_obj(dec), self.cal_con(dec))

    def get_optimum(self, n: int = 1000):
        """Generate objective-space reference points for metrics."""
        del n
        return torch.tensor([self.reference_point], dtype=self.dtype, device=self.device)

    def get_pf(self):
        """Generate a drawable Pareto-front representation."""
        return None
