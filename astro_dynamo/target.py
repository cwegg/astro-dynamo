import math

import torch
import torch.nn as nn


class SurfaceDensity(nn.Module):
    def __init__(self, r_range=(0, 10), r_bins=20):
        super(SurfaceDensity, self).__init__()
        self.dr = (r_range[1] - r_range[0]) / r_bins
        self.r_min = r_range[0]
        self.r_bins = r_bins
        redge = self.r_min + torch.arange(self.r_bins + 1) * self.dr
        self.register_buffer('area', math.pi * (redge[1:] ** 2 - redge[:-1] ** 2))

    def forward(self, snap):
        r_cyl = (snap.positions[:, 0] ** 2 + snap.positions[:, 1] ** 2).sqrt()
        i = ((r_cyl - self.r_min) / self.dr).floor().type(torch.long)
        gd = (i >= 0) & (i < self.r_bins)
        mass_in_bin = torch.sparse.FloatTensor(i[gd].unsqueeze(0), snap.masses[gd], size=(self.r_bins,)).to_dense()
        surface_density = mass_in_bin / self.area
        return surface_density

    def extra_repr(self):
        return f'r_min={self.r_min}, r_max={self.r_min+self.dr*self.r_bins}, r_bins={self.r_bins}'

    @property
    def rmid(self):
        return self.r_min + self.dr / 2 + self.dr * torch.arange(self.r_bins, device=self.area.device,
                                                                 dtype=self.area.dtype)

    def evalulate_function(self, surface_density):
        return surface_density(self.rmid)


class DiskKinematics(SurfaceDensity):
    def __init__(self, *args, **kw_args):
        super(DiskKinematics, self).__init__(*args, **kw_args)

    def forward(self, snap):
        r_cyl = (snap.positions[:, 0] ** 2 + snap.positions[:, 1] ** 2).sqrt()
        i = ((r_cyl - self.r_min) / self.dr).floor().type(torch.long)
        gd = (i >= 0) & (i < self.r_bins)

        mass_in_bin = torch.sparse.FloatTensor(i[gd].unsqueeze(0), snap.masses[gd], size=(self.r_bins,)).to_dense()
        vz_in_bin = torch.sparse.FloatTensor(i[gd].unsqueeze(0), snap.masses[gd] * snap.velocities[gd, 2],
                                             size=(self.r_bins,)).to_dense()
        vz_in_bin /= mass_in_bin
        return vz_in_bin
