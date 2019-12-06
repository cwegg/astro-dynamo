import matplotlib
import matplotlib.artist
import matplotlib.axes
import matplotlib.pyplot as plt
import pytest
import torch
from astro_dynamo.model import MilkyWayModel
from astro_dynamo.plot import plot_disk_kinematics, plot_snap_projections, plot_surface_density_profile
from astro_dynamo.snap import SnapShot
from astro_dynamo.targets import DiskKinematics, SurfaceDensity

"""For the plotting tests we just run each code and make sure that it produces something. Alternatives would be to
either do a comparison of the produced image to previous versions, or to examine programmatically the lines drawn.
But both seem like complete overkill for what are just quick plots."""

@pytest.fixture(scope="module")
def mwmodel():
    snap = SnapShot(positions=torch.as_tensor([[1.0, 0.0, 0.0]]),
                    velocities=torch.as_tensor([[1.0, 0.0, 0.0]]),
                    masses=torch.as_tensor([1.0])).double()
    targets = [DiskKinematics(r_range=(0,2), r_bins=1)]
    targets += [SurfaceDensity(r_range=(0,2), r_bins=1)]
    model = MilkyWayModel(snap=snap, potentials=[], targets=targets, v_scale=1.0, d_scale=1.0,r_0=8.0,z_0=0.0)
    return model

@pytest.fixture(scope="module")
def snap():
    snap = SnapShot(positions=torch.as_tensor([[1.0, 0.0, 0.0]]),
                    velocities=torch.as_tensor([[1.0, 0.0, 0.0]]),
                    masses=torch.as_tensor([1.0])).double()
    return snap


def test_plot_disk_kinematics(mwmodel):
    assert type(plot_disk_kinematics(mwmodel)) == matplotlib.figure.Figure

def test_plot_snap_projections(snap):
    assert type(plot_snap_projections(snap)) == matplotlib.figure.Figure

def test_plot_surface_density_profile(mwmodel):
    f,ax = plt.subplots(1,1)
    assert isinstance(plot_surface_density_profile(mwmodel, ax, mwmodel()[1]),matplotlib.axes.SubplotBase)
