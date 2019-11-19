from astro_dynamo.analytic_potentials import SpheroidalPotential
import numpy as np
import torch
import pytest
import astro_dynamo.analytic_potentials
import galpy.potential


@pytest.mark.parametrize('quad_func', [astro_dynamo.analytic_potentials.fixed_quad, SpheroidalPotential._fixedquad])
def test_fixed_quad(quad_func):
    np.testing.assert_almost_equal(quad_func(lambda x: x ** 2, n=5), 1 / 3)


@pytest.fixture(scope="module")
def astro_dynamo_and_galpy_potentials():
    # Getting units correct is painful. with ro=1 vo=1 and turn_physical_off then everything should be just G=1
    q = 0.5
    galpy_pot = galpy.potential.TwoPowerTriaxialPotential(c=q, ro=1, vo=1)
    galpy_pot.turn_physical_off()
    pot = astro_dynamo.analytic_potentials.SpheroidalPotential(lambda m: galpy_pot._amp * galpy_pot._mdens(m), q=q)
    return pot, galpy_pot


@pytest.mark.parametrize('device_str', ["cuda" if torch.cuda.is_available() else "cpu", 'cpu'])
def test_grid_to_device(astro_dynamo_and_galpy_potentials, device_str):
    pot, _ = astro_dynamo_and_galpy_potentials
    device = torch.device(device_str)
    new_pot = pot.to(device)
    for var, value in vars(new_pot).items():
        if type(value) == torch.Tensor:
            assert value.device == device


@pytest.mark.parametrize('dtype', [torch.double, torch.float])
def test_grid_to_dtype(astro_dynamo_and_galpy_potentials, dtype):
    pot, _ = astro_dynamo_and_galpy_potentials
    new_pot = pot.to(dtype=dtype)
    for var, value in vars(new_pot).items():
        if type(value) == torch.Tensor:
            assert value.dtype == dtype


@pytest.mark.parametrize('astro_dynamo_method, galpy_method, rtol',
                         [('f_r_cyl', 'Rforce', 1e-7), ('f_z', 'zforce', 1e-7),
                          ('f_r', 'rforce', 1e-7), ('pot_ellip', 'flattening', 1e-5)])
@pytest.mark.parametrize('z', [-5., -1., 0., 0.5, 10.])
def test_astro_dynamo_vs_galpy(astro_dynamo_and_galpy_potentials, astro_dynamo_method, galpy_method, z, rtol):
    pot, galpy_pot = astro_dynamo_and_galpy_potentials
    r_cyl = np.logspace(-2., 2., 100, dtype=np.float32)
    galpy_func = eval(f'galpy_pot.{galpy_method}')
    galpy_result = list(map(lambda x: galpy_func(x, z), r_cyl))
    astro_dynamo_function = eval(f'pot.{astro_dynamo_method}')
    astro_dynamo_result = astro_dynamo_function(torch.as_tensor(r_cyl), torch.as_tensor([z]), rel_tol=1e-7)
    return np.testing.assert_allclose(astro_dynamo_result, galpy_result, atol=1e-6, rtol=rtol)


@pytest.mark.parametrize('z', [0.])
def test_astro_dynamo_vs_galpy_vc(astro_dynamo_and_galpy_potentials, z, rtol=1e-5):
    pot, galpy_pot = astro_dynamo_and_galpy_potentials
    r_cyl = np.logspace(-2., 2., 100, dtype=np.float32)
    galpy_func = galpy_pot.vcirc
    galpy_result = list(map(lambda x: galpy_func(x, z), r_cyl))
    astro_dynamo_function = pot.vc2
    astro_dynamo_result = astro_dynamo_function(torch.as_tensor(r_cyl), torch.as_tensor([z]), rel_tol=1e-8).sqrt()
    return np.testing.assert_allclose(astro_dynamo_result, galpy_result, atol=1e-5, rtol=rtol)


def test_f_compute_raises(astro_dynamo_and_galpy_potentials):
    pot, _ = astro_dynamo_and_galpy_potentials
    with pytest.raises(ValueError):
        pot._f_compute(torch.as_tensor([1.]), torch.as_tensor([1.]), direction='theta')


def test_f_theta_in_plane(astro_dynamo_and_galpy_potentials):
    pot, _ = astro_dynamo_and_galpy_potentials
    r_cyl = np.linspace(0., 10., 100, dtype=np.float32)
    return np.testing.assert_allclose(pot.f_z(r_cyl, [0.]), pot.f_theta(r_cyl, [0.]), atol=1e-6)


def test_f_theta_zaxis(astro_dynamo_and_galpy_potentials):
    pot, _ = astro_dynamo_and_galpy_potentials
    z = np.linspace(0., 10., 100, dtype=np.float32)
    return np.testing.assert_allclose(pot.f_r_cyl([0.], z), pot.f_theta([0.], z), atol=1e-6)


@pytest.mark.parametrize('r, ang, r_cyl, z',
                         [(0, 0, 0, 0), (1, 0, 0, 1), (1, 90, 1, 0), (1, 45, 1 / np.sqrt(2), 1 / np.sqrt(2))])
def test_spherical_to_cylindrical(r, ang, r_cyl, z):
    r_cyl_computed, z_computed = SpheroidalPotential.spherical_to_cylindrical(torch.as_tensor(r), torch.as_tensor(ang))
    np.testing.assert_allclose([r_cyl_computed, z_computed], [r_cyl, z])
