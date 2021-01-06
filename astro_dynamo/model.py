import math
from typing import List, Union, Tuple

import torch
import torch.nn as nn

from astro_dynamo.snap import SnapShot
from .snaptools import align_bar


def _symmetrize_matrix(x, dim):
    """Symmetrize a tensor along dimension dim"""
    return (x + x.flip(dims=[dim])) / 2


class DynamicalModel(nn.Module):
    """DynamicalModels class. This containts a snapshot of the particles, the potentials
    in which they move, and the targets to which the model should be fitted.

    Attributes:
        snap:
            Should be a SnapShot whose masses will be optimised

        potentials:
            The potentials add. If self gravity is not required set self_gravity_update to None.
            If self gravity is required then the potential of the snapshot should be in potentials[0]
            and self_gravity_update represents how much to update the running average of the density on
            each iteration. Default value is 0.2 which is then exponential averages the density with timescale
            5 snapshots(=1/0.2).

        targets:
            A list of targets. Running
                model = DynamicalModel(snap, potentials, targets)
                current_target_list = model()
            will provide an list of theDynamicalModelse targets evaluated with the present model. These are then
            typically combined to a loss that pytorch can optimise.

    Methods:
        forward()
            Computes the targets by evaluating them on the current snapshot. Can also be called as DynamicalModel()
        integrate(steps=256)
            Integrates the model forward by steps. Updates potential the density assocaiates to potential[0]
        update_potential()
            Recomputes the accelerations from potential[0]. Adjust each snapshots velocity by a factor vc_new/vc_old
        resample()
            Resamples the snapshot to equal mass particles.
    """

    def __init__(self, snap, potentials, targets, self_gravity_update=0.2):
        super(DynamicalModel, self).__init__()
        self.snap = snap
        self.targets = nn.ModuleList(targets)
        self.potentials = nn.ModuleList(potentials)
        self.self_gravity_update = self_gravity_update

    def forward(self):
        return [target(self) for target in self.targets]

    def integrate(self, steps=256):
        with torch.no_grad():
            self.snap.leapfrog_steps(potentials=self.potentials, steps=steps)
            if self.self_gravity_update is not None:
                self.potentials[0].update_density(self.snap.positions,
                                                  self.snap.masses.detach(),
                                                  fractional_update=self.self_gravity_update)

    def update_potential(self, dm_potential=None, update_velocities=True):
        with torch.no_grad():
            if update_velocities:
                old_accelerations = self.snap.get_accelerations(self.potentials,
                                                                self.snap.positions)
                old_vc = torch.sum(-old_accelerations * self.snap.positions,
                                   dim=-1).sqrt()
            self.potentials[0].rho = _symmetrize_matrix(
                _symmetrize_matrix(
                    _symmetrize_matrix(self.potentials[0].rho, 0), 1), 2)
            self.potentials[0].grid_accelerations()
            if dm_potential is not None:
                self.potentials[1] = dm_potential
            if update_velocities:
                new_accelerations = self.snap.get_accelerations(self.potentials,
                                                                self.snap.positions)
                new_vc = torch.sum(-new_accelerations * self.snap.positions,
                                   dim=-1).sqrt()
                gd = torch.isfinite(new_vc / old_vc) & (new_vc / old_vc > 0)
                self.snap.velocities[gd, :] *= (new_vc / old_vc)[gd, None]
            align_bar(self.snap)

    def resample(self, velocity_perturbation=0.01):
        """Resample the model to equal mass particles.

        Note that the snapshot changes and so the parameters of
        the model also change in a way that any optimiser that keeps parameter-by-parameter information e.g.
        gradients must also be update."""
        with torch.no_grad():
            self.snap = self.snap.resample(self.potentials,
                                           velocity_perturbation=velocity_perturbation)
            align_bar(self.snap)

    def vc(self, components=False, r=torch.linspace(0, 9),
           phi=torch.linspace(0, math.pi)):
        """Returns (r,vc) the circular velocity of the model in physical units and locations at which it was evaluated.

        If components=True then return list containing the vc of each component, otherwise just return the total.
        r optionally specifies the physical radii at which to compute vc
        phi specifies the azimuths over which to average."""
        phi_grid, r_grid = torch.meshgrid(phi, r)
        phi_grid, r_grid = phi_grid.flatten(), r_grid.flatten()
        pos = torch.stack((r_grid * torch.cos(phi_grid),
                           r_grid * torch.sin(phi_grid), 0 * phi_grid)).t()
        pos = pos.to(device=self.d_scale.device)
        pos /= self.d_scale
        vc = []
        for potential in self.potentials:
            device = next(potential.buffers()).device
            acc = potential.get_accelerations(pos.to(device=device)).to(
                device=pos.device)
            vc += [torch.sum(-acc * pos, dim=1).sqrt().reshape(
                phi.shape + r.shape).mean(dim=0) * self.v_scale]
        if components:
            return r, vc
        else:
            total_vc = vc[0]
            for thisvc in vc[1:]:
                total_vc = (total_vc ** 2 + thisvc ** 2).sqrt()
            return r, total_vc


class MilkyWayModel(DynamicalModel):
    def __init__(self, snap: SnapShot, potentials: List[nn.Module],
                 targets: List[nn.Module],
                 self_gravity_update: Union[float, torch.Tensor] = 0.2,
                 bar_angle: Union[float, torch.Tensor] = 27.,
                 r_0: Union[float, torch.Tensor] = 8.2,
                 z_0: Union[float, torch.Tensor] = 0.014,
                 v_scale: Union[float, torch.Tensor] = 240,
                 d_scale: Union[float, torch.Tensor] = 1.4,
                 v_sun: Union[List[float], Tuple[float, float, float],
                              torch.Tensor] = (11.1, 12.24 + 238.0, 7.25)
                 ):
        super(MilkyWayModel, self).__init__(snap, potentials, targets,
                                            self_gravity_update)
        self.bar_angle = nn.Parameter(torch.as_tensor(bar_angle),
                                      requires_grad=False)
        self.r_0 = nn.Parameter(torch.as_tensor(r_0), requires_grad=False)
        self.z_0 = nn.Parameter(torch.as_tensor(z_0), requires_grad=False)
        self.v_scale = nn.Parameter(torch.as_tensor(v_scale),
                                    requires_grad=False)
        self.d_scale = nn.Parameter(torch.as_tensor(d_scale),
                                    requires_grad=False)
        self.v_sun = nn.Parameter(torch.as_tensor(v_sun), requires_grad=False)

    @property
    def m_scale(self) -> torch.tensor:
        G = 4.302E-3  # Gravitational constant in astronomical units
        return self.d_scale * 1e3 * self.v_scale ** 2 / G

    @property
    def t_scale(self) -> torch.tensor:
        """1 iu in time in Gyr"""
        return self.d_scale / self.v_scale * 0.977813106  # note that 1km/s is almost 1kpc/Gyr

    @property
    def xyz(self) -> torch.tensor:
        """Return position of particles in relative to the Sun in cartesian coordinates with units kpc
        """
        ddtor = math.pi / 180.
        ang = self.bar_angle * ddtor
        pos = self.snap.positions
        xyz = torch.zeros_like(pos)
        inplane_gc_distance = (self.r_0 ** 2 - self.z_0 ** 2).sqrt()
        xyz[:, 0] = (pos[:, 0] * torch.cos(-ang) - pos[:, 1] * torch.sin(
            -ang)) * self.d_scale + inplane_gc_distance
        xyz[:, 1] = (pos[:, 0] * torch.sin(-ang) + pos[:, 1] * torch.cos(
            -ang)) * self.d_scale
        xyz[:, 2] = pos[:, 2] * self.d_scale - self.z_0
        return xyz

    @property
    def l_b_mu(self) -> torch.tensor:
        """Return array of particles in galactic (l,b,mu) coordinates. (l,b) in degrees. mu is distance modulus"""
        xyz = self.xyz
        l_b_mu = torch.zeros_like(xyz)
        d = (xyz[:, 0] ** 2 + xyz[:, 1] ** 2 + xyz[:, 2] ** 2).sqrt()
        l_b_mu[:, 0] = torch.atan2(xyz[:, 1], xyz[:, 0]) * 180 / math.pi
        b_offset = torch.asin(
            self.z_0 / self.r_0)  # the GC has z = -z_0, rotate b coordinate so this is at l,b=(0,0)
        l_b_mu[:, 1] = (torch.asin(xyz[:, 2] / d) + b_offset) * 180 / math.pi
        l_b_mu[:, 2] = 5 * (100 * d).log10()
        return l_b_mu

    @property
    def masses(self) -> torch.tensor:
        return self.snap.masses * self.m_scale

    @property
    def omega(self) -> torch.tensor:
        return self.snap.omega * self.v_scale / self.d_scale

    @omega.setter
    def omega(self, omega: float):
        self.snap.omega = omega / self.v_scale * self.d_scale

    @property
    def uvw(self) -> torch.tensor:
        """Return UVW velocities.
        """
        ddtor = math.pi / 180.
        ang = self.bar_angle * ddtor
        vxyz = torch.zeros_like(self.snap.positions)
        # sun moves at Vsun[0] towards galactic center i.e. other stars are moving away towards larger x
        vel = self.snap.velocities * self.v_scale
        vxyz[:, 0] = (vel[:, 0] * torch.cos(-ang) - vel[:, 1] * torch.sin(-ang)) + self.v_sun[0]
        # sun moves at Vsun[1] in direction of rotation, other stars are going slower than (0,-Vc,0)
        vxyz[:, 1] = (vel[:, 0] * torch.sin(-ang) + vel[:, 1] * torch.cos(-ang)) - self.v_sun[1]
        # sun is moving towards ngp i.e. other stars on average move at negative vz
        vxyz[:, 2] = vel[:, 2] - self.v_sun[2]
        return vxyz

    @property
    def vr(self) -> torch.tensor:
        """Return array of particles radial velocities in [km/s]"""
        xyz = self.xyz
        vxyz = self.uvw
        r = xyz.norm(dim=-1)
        vr = (xyz * vxyz).sum(dim=-1) / r
        return vr

    @property
    def mul_mub(self) -> torch.tensor:
        """Return proper motions of particles in [mas/yr] in (l, b).
        Proper motion in l is (rate of change of l)*cos(b)"""
        xyz = self.xyz
        vxyz = self.uvw
        r = xyz.norm(dim=-1)
        rxy = (xyz[:, 0] ** 2 + xyz[:, 1] ** 2).sqrt()
        # magic number comes from:  1 mas/yr = 4.74057 km/s at 1 kpc
        mul = (-vxyz[:, 0] * xyz[:, 1] / rxy + vxyz[:, 1] * xyz[:, 0] / rxy) / r / 4.74057
        mub = (-vxyz[:, 0] * xyz[:, 2] * xyz[:, 0] / rxy - vxyz[:, 1] * xyz[:, 2] * xyz[:, 1] / rxy + vxyz[:, 2] * rxy) / (
                    r ** 2) / 4.74057
        return torch.stack((mul, mub), dim=-1)
