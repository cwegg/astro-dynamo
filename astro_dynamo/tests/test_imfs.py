import numpy as np
import pytest
from astro_dynamo.imfs import PowerLawIMF


@pytest.fixture(scope="module")
def simple_power_law_imf():
    return PowerLawIMF(mass_breaks=[0.5], power_law_indicies=[0., -3.])

@pytest.mark.parametrize('method',
                         [('number'), ('number_integral'),('mass_integral')])
def test_continuity(simple_power_law_imf, method):
    eps = 1e-8
    test_function = eval(f'simple_power_law_imf.{method}')
    np.testing.assert_array_almost_equal(test_function(simple_power_law_imf.mass_breaks[0] - eps),
                                         test_function(simple_power_law_imf.mass_breaks[0] + eps))

def test_integral_limits(simple_power_law_imf):
    np.testing.assert_array_almost_equal(simple_power_law_imf.mass_integral([0, np.inf]), [0, 1.0])


def test_number(simple_power_law_imf):
    np.testing.assert_array_almost_equal(simple_power_law_imf.number([0.5]), [8/3])


def test_integral(simple_power_law_imf):
    np.testing.assert_array_almost_equal(simple_power_law_imf.number_integral([0.5]), [4/3])
