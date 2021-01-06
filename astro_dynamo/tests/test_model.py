import numpy as np
import pytest
import torch

from astro_dynamo.model import MilkyWayModel
from astro_dynamo.snap import SnapShot


@pytest.fixture(scope="module")
def mwmodel():
    snap = SnapShot(positions=torch.as_tensor([[0.0, 0.0, 0.0]]),
                    velocities=torch.as_tensor([[0.0, 0.0, 0.0]]),
                    masses=torch.as_tensor([1.0])).double()
    model = MilkyWayModel(snap=snap, potentials=[], targets=[], v_scale=1.0,
                          d_scale=1.0, r_0=8.0, z_0=0.1).double()
    return model


def test_xyz(mwmodel):
    galatic_center = [[np.sqrt(8.0 ** 2 - 0.1 ** 2), 0.0, -0.1]]
    np.testing.assert_array_almost_equal(mwmodel.xyz, galatic_center)


def test_l_b_mu_galactic_center(mwmodel):
    galatic_center = [[0.0, 0.0, 5 * np.log10(100 * 8.0)]]
    np.testing.assert_array_almost_equal(mwmodel.l_b_mu, galatic_center)


def test_l_b_mu_bar_endr():
    r_0, bar_angle, bar_len = 8.0, 30.0, 5.0
    snap = SnapShot(positions=torch.as_tensor([[-bar_len, 0.0, 0.0]]),
                    velocities=torch.as_tensor([[0.0, 0.0, 0.0]]),
                    masses=torch.as_tensor([1.0])).double()
    model = MilkyWayModel(snap=snap, potentials=[], targets=[], v_scale=1.0,
                          d_scale=1.0, r_0=r_0, z_0=0.0,
                          bar_angle=bar_angle).double()
    d = np.sqrt(bar_len ** 2 + r_0 ** 2 - 2 * bar_len * r_0 * np.cos(
        np.pi * bar_angle / 180))
    gal_l = 180 * np.arcsin(bar_len / d * np.sin(np.pi * bar_angle / 180)) / np.pi
    bar_end = [[gal_l, 0.0, 5 * np.log10(100 * d)]]
    np.testing.assert_array_almost_equal(model.l_b_mu, bar_end)


def test_get_masses():
    d_scale, v_scale = 2.0, 5.0
    snap = SnapShot(positions=torch.as_tensor([[0.0, 0.0, 0.0]]),
                    velocities=torch.as_tensor([[0.0, 0.0, 0.0]]),
                    masses=torch.as_tensor([1.0]))
    model = MilkyWayModel(snap=snap, potentials=[], targets=[], v_scale=v_scale,
                          d_scale=d_scale).double().eval()
    np.testing.assert_array_almost_equal(model.masses.detach()[0],
                                         d_scale * 1e3 * v_scale ** 2 / 4.302E-3)
