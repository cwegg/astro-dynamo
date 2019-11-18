from astro_dynamo.grid import ForceGrid, Grid
import numpy as np
import torch
import pytest


@pytest.fixture(scope="module")
def three_by_three_grid():
    return Grid(n=(3, 3, 3), data=torch.zeros(3, 3, 3))


def test_grid_to(three_by_three_grid):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    new_grid = three_by_three_grid.to(device)
    assert new_grid.min.device == device
    assert new_grid.max.device == device
    assert new_grid.data.device == device
    assert new_grid.dx.device == device


def test_smoothing_pot_limits():
    np.testing.assert_array_almost_equal(ForceGrid.smoothing_pot(torch.tensor([0., np.inf])), torch.tensor([-1.4, 0.]))


@pytest.mark.parametrize('r', [1, 2])
def test_smoothing_pot_continuity(r):
    eps = 1e-5
    np.testing.assert_array_almost_equal(ForceGrid.smoothing_pot(torch.tensor([r - eps])),
                                         ForceGrid.smoothing_pot(torch.tensor([r + eps])), decimal=4)


def test_smoothing_pot_large_r():
    np.testing.assert_array_almost_equal(ForceGrid.smoothing_pot(torch.tensor([10.])), torch.tensor([-1. / 10]))


def test_complex_mul():
    np.testing.assert_array_almost_equal(ForceGrid.complex_mul(torch.tensor([[1., 0.], [0., 1.]]),
                                                               torch.tensor([[0., 1.], [0., 1.]])),
                                         torch.tensor([[0., 1.], [-1., 0.]]))
