import numpy as np
import pytest
import torch
from astro_dynamo.grid import ForceGrid, Grid


@pytest.fixture(scope="module")
def three_by_three_grid():
    return Grid(n=(3, 3, 3), grid_edges=torch.tensor((1., 1., 1.)))


def test_uncentered_grid():
    grid = Grid(n=(3, 3, 3), grid_edges=torch.tensor(((0., 1.), (-1, 1), (-1, 1))))
    np.testing.assert_array_almost_equal(grid.x, torch.tensor((0., 0.5, 1)))


def test_grid_to(three_by_three_grid):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    new_grid = three_by_three_grid.to(device)
    for var, value in vars(new_grid).items():
        if type(value) == torch.Tensor:
            assert value.device == device


def test_grid_to_cpu(three_by_three_grid):
    device = torch.device("cpu")
    new_grid = three_by_three_grid.to(device)
    for var, value in vars(new_grid).items():
        if type(value) == torch.Tensor:
            assert value.device == device


def test_coordinates(three_by_three_grid):
    np.testing.assert_array_almost_equal(three_by_three_grid.x, [-1., 0., 1.])
    np.testing.assert_array_almost_equal(three_by_three_grid.y, [-1., 0., 1.])
    np.testing.assert_array_almost_equal(three_by_three_grid.z, [-1., 0., 1.])


@pytest.mark.parametrize('h', [None, 0.5])
def test_ingrid_outside_grid(three_by_three_grid, h):
    assert three_by_three_grid.ingrid(torch.tensor([[2., 0., 0.]]), h=h) == torch.tensor([0])


@pytest.mark.parametrize('h', [None, 0.5])
def test_ingrid_in_grid(three_by_three_grid, h):
    assert three_by_three_grid.ingrid(torch.tensor([[0., 0., 0.]]), h=h) == torch.tensor([1])


def test_ingrid_h(three_by_three_grid):
    assert three_by_three_grid.ingrid(torch.tensor([[0., 0., 0.6], ]), h=0.5) == torch.tensor([0])


@pytest.mark.parametrize('position', [torch.tensor([[0., 0., 0.]]), torch.tensor([[0.1, -0.1, 0.1]])])
def test_grid_data_nearest_center(three_by_three_grid, position):
    center = torch.zeros(tuple(three_by_three_grid.n))
    center[1, 1, 1] = 1.0
    np.testing.assert_array_equal(three_by_three_grid.grid_data(position, method='nearest'), center)


@pytest.mark.parametrize('position, ix, iy, iz', [(torch.tensor([[0.9, 0.9, 0.9]]), 2, 2, 2),
                                                  (torch.tensor([[-0.9, 0.8, 0.9]]), 0, 2, 2)])
def test_grid_data_nearest_corner(three_by_three_grid, position, ix, iy, iz):
    corner = torch.zeros(tuple(three_by_three_grid.n))
    corner[ix, iy, iz] = 1.0
    np.testing.assert_array_equal(three_by_three_grid.grid_data(position, method='nearest'), corner)


def test_grid_data_cic_center(three_by_three_grid):
    position = torch.tensor([[0., 0., 0.]])
    center = torch.zeros(tuple(three_by_three_grid.n))
    center[1, 1, 1] = 1.0
    np.testing.assert_array_equal(three_by_three_grid.grid_data(position, method='cic'), center)


def test_grid_data_cic_offset_half(three_by_three_grid):
    eps = 1e-6
    position = torch.tensor([[-0.5 + eps, -0.5 + eps, -0.5 + eps]])
    center = torch.zeros(tuple(three_by_three_grid.n))
    center[0, 0, 0] = center[0, 0, 1] = center[0, 1, 0] = center[0, 1, 1] = 1 / 8.
    center[1, 0, 0] = center[1, 0, 1] = center[1, 1, 0] = center[1, 1, 1] = 1 / 8.
    np.testing.assert_array_almost_equal(three_by_three_grid.grid_data(position, method='cic'), center)


def test_grid_data_unrecognised_method(three_by_three_grid):
    position = torch.tensor([[0., 0., 0.]])
    with pytest.raises(ValueError):
        _ = three_by_three_grid.grid_data(position, method='blah')


@pytest.fixture(scope="module")
def three_by_three_forcegrid():
    forcegrid = ForceGrid(n=(3, 3, 3), grid_edges=torch.tensor((1., 1., 1.)), rho=torch.zeros(3, 3, 3))
    forcegrid.grid_accelerations()
    return forcegrid


def test_forcegrid_to(three_by_three_forcegrid):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    new_grid = three_by_three_forcegrid.to(device)
    for var, value in vars(new_grid).items():
        if type(value) == torch.Tensor:
            assert value.device == device


def test_forcegrid_to_cpu(three_by_three_forcegrid):
    device = torch.device("cpu")
    new_grid = three_by_three_forcegrid.to(device)
    for var, value in vars(new_grid).items():
        if type(value) == torch.Tensor:
            assert value.device == device


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


@pytest.mark.parametrize('position, expected', [(torch.tensor([[0., 0., 0.]]), torch.tensor([1., 1., 0.])),
                                                (torch.tensor([[0., 0., 0.5]]), torch.tensor([0.5, 0.5, 0.])),
                                                (torch.tensor([[0., 0.5, 0.]]), torch.tensor([1.0, 0.5, 0.])),
                                                (torch.tensor([[-0.5, 0., 0.]]), torch.tensor([0.5, 0.5, -0.5]))])
def test_get_accelerations(three_by_three_forcegrid, position, expected):
    # setup the acceration grid by hand and try going in all three directions
    three_by_three_forcegrid.acc = torch.zeros_like(three_by_three_forcegrid.acc)
    three_by_three_forcegrid.acc[1, 1, 1, 0:2] = 1.
    three_by_three_forcegrid.acc[1, 2, 1, 0] = 1.
    three_by_three_forcegrid.acc[0, 1, 1, 2] = -1
    np.testing.assert_array_almost_equal(three_by_three_forcegrid.get_accelerations(position), expected)


@pytest.mark.parametrize('position, expected', [(torch.tensor([[0., 0., 0.]]), torch.tensor([1.])),
                                                (torch.tensor([[0., 0., 0.5]]), torch.tensor([0.5])),
                                                (torch.tensor([[0., 0.5, 0.]]), torch.tensor([1.0])),
                                                (torch.tensor([[-0.5, 0., 0.]]), torch.tensor([0.]))])
def test_get_potential(three_by_three_forcegrid, position, expected):
    # setup the acceration grid by hand and try going in all three directions
    three_by_three_forcegrid.pot = torch.zeros_like(three_by_three_forcegrid.pot)
    three_by_three_forcegrid.pot[1, 1, 1] = 1.
    three_by_three_forcegrid.pot[1, 2, 1] = 1.
    three_by_three_forcegrid.pot[0, 1, 1] = -1
    np.testing.assert_array_almost_equal(three_by_three_forcegrid.get_potential(position), expected)


@pytest.fixture(scope="module")
def almost_oned_force_grid():
    # make a long thin grid and test at the distant edges that we get the analytic potential
    forcegrid = ForceGrid(n=(4096 * 4, 3, 3), grid_edges=torch.tensor((100., 1., 1.))).double()
    source_position = torch.tensor([[0., 0., 0.]], dtype=torch.double)
    forcegrid.grid_accelerations(source_position, method='cic')
    return forcegrid


def test_grid_accelerations_pot(almost_oned_force_grid):
    np.testing.assert_allclose([almost_oned_force_grid.pot[0, 1, 1], almost_oned_force_grid.pot[-1, 1, 1]],
                               [-1 / almost_oned_force_grid.max[0], 1 / almost_oned_force_grid.min[0]],
                               rtol=1e-4, atol=1e-8)


def test_grid_accelerations_acc_negativex(almost_oned_force_grid):
    min_x = almost_oned_force_grid.min[0]
    np.testing.assert_allclose(almost_oned_force_grid.acc[0, 1, 1, :], [1 / (min_x * min_x), 0, 0],
                               rtol=1e-4, atol=1e-8)


def test_grid_accelerations_acc_positivex(almost_oned_force_grid):
    max_x = almost_oned_force_grid.max[0]
    np.testing.assert_allclose(almost_oned_force_grid.acc[-1, 1, 1, :], [-1 / (max_x * max_x), 0, 0],
                               rtol=1e-4, atol=1e-8)


@pytest.mark.parametrize('fractional_update', [0., 0.1, 0.5, 1.0])
def test_force_grid_fractional_update(three_by_three_forcegrid, fractional_update):
    position = torch.tensor([[0., 0., 0.]])
    center = torch.zeros(tuple(three_by_three_forcegrid.n))
    center[1, 1, 1] = 1.0
    three_by_three_forcegrid.rho = torch.zeros_like(three_by_three_forcegrid.rho)
    three_by_three_forcegrid.update_density(position, method='nearest', fractional_update=fractional_update)
    np.testing.assert_array_almost_equal(three_by_three_forcegrid.rho, center * fractional_update)
