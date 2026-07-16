"""Shape and construction tests for the optional plotting API."""

import pytest
import torch

pytest.importorskip("plotly")

from ascendmoea.plotting import draw_dec, draw_metric_curve, draw_obj


def test_decision_plots_support_one_two_and_three_dimensions() -> None:
    """Construct each low-dimensional decision-space figure."""
    for dimension in (1, 2, 3):
        figure = draw_dec(torch.zeros((4, dimension)))
        assert len(figure.data) == 1


def test_plotting_rejects_ambiguous_shapes() -> None:
    """Fail early for unsupported objective and metric inputs."""
    with pytest.raises(ValueError, match="obj must have shape"):
        draw_obj(torch.zeros((3, 4)))
    with pytest.raises(ValueError, match="same number"):
        draw_metric_curve([1, 2], [1], "IGD")
