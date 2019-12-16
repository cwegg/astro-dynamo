import astro_dynamo.imfs
import astro_dynamo.luminosityfunction
import numpy as np
import pytest


@pytest.fixture(scope="module")
def lf_from_isochrones():
    lf_obj = astro_dynamo.luminosityfunction.ParsecLuminosityFunction(
        isochrone_file='../data/parsec_isochrones_gaia_2mass.dat',
        mag_range=(-20,20), d_mag=0.2)
    return lf_obj

def test_construction(lf_from_isochrones):
    assert lf_from_isochrones.grids['Ksmag'][11,10,:].shape==(201,)

def test_z(lf_from_isochrones):
    y = 0.2485 + 1.78 * lf_from_isochrones.zs
    x = 1 - y - lf_from_isochrones.zs
    m_h_solar = 0.0207
    m_h = np.log10(lf_from_isochrones.zs / x) - np.log10(m_h_solar)
    np.testing.assert_array_almost_equal(m_h,lf_from_isochrones.mhs)


