import numpy as np
import pytest
from astro_dynamo.imfs import PowerLawIMF


@pytest.fixture(scope="module")
def simple_power_law_imf():
    return PowerLawIMF(mass_breaks=[0.5],power_law_indicies=[0., -2.])

def test_continuity_number(simple_power_law_imf):
    eps = 1e-8
    np.testing.assert_array_almost_equal(simple_power_law_imf.number(simple_power_law_imf.mass_breaks[0]-eps),
                                         simple_power_law_imf.number(simple_power_law_imf.mass_breaks[0]+eps))

def test_continuity_integral(simple_power_law_imf):
    eps = 1e-8
    np.testing.assert_array_almost_equal(simple_power_law_imf.integral(simple_power_law_imf.mass_breaks[0]-eps),
                                         simple_power_law_imf.integral(simple_power_law_imf.mass_breaks[0]+eps))

def test_integral_limits(simple_power_law_imf):
    np.testing.assert_array_almost_equal(simple_power_law_imf.integral([0, np.inf]), [0, 1.0])

def test_number(simple_power_law_imf):
    np.testing.assert_array_almost_equal(simple_power_law_imf.number([0.5]), [1.])

def test_integral(simple_power_law_imf):
    np.testing.assert_array_almost_equal(simple_power_law_imf.integral([0.5]), [0.5])
