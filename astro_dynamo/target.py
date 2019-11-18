import torch
import math
import numpy as np
from abc import ABC, abstractmethod


class Target(ABC):
    @abstractmethod
    def observe(self, snap):
        pass

    @property
    @abstractmethod
    def target(self):
        pass

    @property
    @abstractmethod
    def error(self):
        pass

    def loss(self, snap):
        return (((self.observe(snap) - self.target) / self.error) ** 2).sum() / len(self.target)


class RadialProfile(Target):
    def __init__(self, surface_density=None, r_range=(0, 10), rbins=50, device=None):
        self.device = torch.zeros((1,), device=device).device
        self.dr = (r_range[1] - r_range[0]) / rbins
        self.r_range = r_range
        self.rbins = rbins
        self.area = math.pi * (self.redge[1:] ** 2 - self.redge[:-1] ** 2)

        if surface_density is not None:
            # We take in the surface_denisty function and sample it 100 times in each bin
            oversample = 100
            nrsamples = rbins * oversample
            dr = (r_range[1] - r_range[0]) / nrsamples
            rsamples = torch.arange(r_range[0] + dr / 2, r_range[1], dr, device=device)
            fvalues = surface_density(rsamples)
            values = 2 * math.pi * rsamples * fvalues * dr
            self._target = values.view(rbins, oversample).sum(dim=-1) / self.area
        else:
            self._target = None

    @property
    def target(self):
        return self._target

    @property
    def error(self):
        # Fit fractional error
        return self.target

    @property
    def redge(self):
        return torch.linspace(self.r_range[0], self.r_range[1], self.rbins + 1, device=self.device)

    @property
    def rmid(self):
        return torch.arange(self.r_range[0] + self.dr / 2, self.r_range[1], self.dr, device=self.device)

    def observe(self, snap):
        assert snap.positions.device == self.device
        i = ((snap.rcyl - self.r_range[0]) / self.dr).floor().type(torch.long)
        gd = (i >= 0) & (i < self.rbins)
        mass_in_bin = torch.sparse.FloatTensor(i[gd].unsqueeze(0), snap.masses[gd], size=(self.rbins,)).to_dense()
        surface_density = mass_in_bin / self.area
        return surface_density

    def interpolate_surface_density(self, snap, r):
        my_r = self.rmid.cpu().numpy()
        my_surface_density = self.observe(snap).cpu().numpy()
        interpolated_surface_density = torch.as_tensor(np.interp(r, my_r, my_surface_density), device=self.device)
        return interpolated_surface_density


class Cartesian3DDensity(Target):
    def __init__(self, density=None, density_func=None, density_error=None, density_error_func=None,
                 x_range=(-10, 10), xbins=100, y_range=(-10, 10), ybins=100, z_range=(-2, 2), zbins=20,
                 device=None, shape=False):
        self.device = torch.zeros((1,), device=device).device
        self.dx = (x_range[1] - x_range[0]) / xbins
        self.dy = (y_range[1] - y_range[0]) / ybins
        self.dz = (z_range[1] - z_range[0]) / zbins
        self.x_range, self.y_range, self.z_range = x_range, y_range, z_range
        self.xbins, self.ybins, self.zbins = xbins, ybins, zbins
        self.volume = self.dx * self.dy * self.dz

        if density_func is not None:
            self.density = density_func(self.xmid[:, None, None],
                                        self.ymid[None, :, None],
                                        self.zmid[None, None, :])
        if density_error_func is not None:
            self.density_error = density_error_func(self.xmid[:, None, None],
                                                    self.ymid[None, :, None],
                                                    self.zmid[None, None, :])
        if density is not None:
            self.density = density
        if density_error is not None:
            self.density_error = density_error

    @property
    def target(self):
        return self.density

    @property
    def error(self):
        if self.density_error is None:
            return self.density
        else:
            return self.density_error

    @property
    def xedge(self):
        return torch.linspace(self.x_range[0], self.x_range[1], self.xbins + 1, device=self.device)

    @property
    def xmid(self):
        return torch.arange(self.x_range[0] + self.dx / 2, self.x_range[1], self.dx, device=self.device)

    @property
    def yedge(self):
        return torch.linspace(self.y_range[0], self.y_range[1], self.ybins + 1, device=self.device)

    @property
    def ymid(self):
        return torch.arange(self.y_range[0] + self.dy / 2, self.y_range[1], self.dy, device=self.device)

    @property
    def zedge(self):
        return torch.linspace(self.z_range[0], self.z_range[1], self.zbins + 1, device=self.device)

    @property
    def zmid(self):
        return torch.arange(self.z_range[0] + self.dz / 2, self.z_range[1], self.dz, device=self.device)

    def observe(self, snap):
        assert snap.positions.device == self.device
        ix = ((snap.x - self.x_range[0]) / self.dx).floor().type(torch.long)
        iy = ((snap.y - self.y_range[0]) / self.dy).floor().type(torch.long)
        iz = ((snap.z - self.z_range[0]) / self.dz).floor().type(torch.long)

        gd = (ix >= 0) & (ix < self.xbins) & (ix >= 0) & (iy < self.ybins) & (iz >= 0) & (iz < self.zbins)
        mass_in_bin = torch.sparse.FloatTensor(torch.stack((ix[gd], ix[gd], ix[gd]), dim=0),
                                               snap.masses[gd], size=(self.xbins, self.ybins, self.zbins)).to_dense()
        if self.shape:
            mass_in_bin *= self._target.sum() / mass_in_bin.sum()
        else:
            density = mass_in_bin / self.area
        return density
