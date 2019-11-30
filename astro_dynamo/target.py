import math

import torch
import torch.nn as nn


class SurfaceDensity(nn.Module):
    def __init__(self, r_range=(0, 10), r_bins=20, physical=False):
        super(SurfaceDensity, self).__init__()
        self.physical = physical
        self.dr = (r_range[1] - r_range[0]) / r_bins
        self.r_min = r_range[0]
        self.r_bins = r_bins
        redge = self.r_min + torch.arange(self.r_bins + 1) * self.dr
        self.register_buffer('area', math.pi * (redge[1:] ** 2 - redge[:-1] ** 2))


    def forward(self, model):
        snap = model.snap
        r_cyl = (snap.positions[:, 0] ** 2 + snap.positions[:, 1] ** 2).sqrt()
        if self.physical:
            r_cyl *= model.d_scale
        i = ((r_cyl - self.r_min) / self.dr).floor().type(torch.long)
        gd = (i >= 0) & (i < self.r_bins)
        mass_in_bin = torch.sparse.FloatTensor(i[gd].unsqueeze(0), snap.masses[gd], size=(self.r_bins,)).to_dense()
        surface_density = mass_in_bin / self.area
        if self.physical:
            surface_density *= model.m_scale/model.d_scale**2*1e-6 #in Msun/pc**2
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
    def __init__(self,*args, **kw_args):
        super(DiskKinematics, self).__init__(*args, **kw_args)

    def forward(self, model):
        snap = model.snap
        # Take the usual approach of binning using a sparse tensor so that we get gradients
        r_cyl = (snap.positions[:, 0] ** 2 + snap.positions[:, 1] ** 2).sqrt()
        if self.physical:
            r_cyl *= model.d_scale
        i = ((r_cyl - self.r_min) / self.dr).floor().type(torch.long)
        gd = (i >= 0) & (i < self.r_bins)

        masses = snap.masses[gd]
        i = i[gd].unsqueeze(0)

        v_r_cyl = (snap.positions[:, 0]*snap.velocities[:, 0] + snap.positions[:, 1]*snap.velocities[:, 1])/r_cyl
        v_phi = (snap.positions[:, 1]*snap.velocities[:, 0] - snap.positions[:, 0]*snap.velocities[:, 1])/r_cyl

        in_bin  = lambda x :  torch.sparse.FloatTensor(i, masses*x, size=(self.r_bins,)).to_dense()
        hist_mass = in_bin(1)
        hist_vr = in_bin(v_r_cyl[gd])/hist_mass
        hist_vr2 = in_bin(v_r_cyl[gd]**2)/hist_mass
        hist_vphi = in_bin(v_phi[gd])/hist_mass
        hist_vphi2 = in_bin(v_phi[gd]**2)/hist_mass
        hist_vz = in_bin(snap.velocities[gd,2])/hist_mass
        hist_vz2 = in_bin(snap.velocities[gd,2]**2)/hist_mass

        hist_vz_sig = (hist_vz2-hist_vz**2).sqrt()
        hist_vr_sig = (hist_vr2-hist_vr**2).sqrt()
        hist_vphi_sig = (hist_vphi2 - hist_vphi**2).sqrt()

        kin = torch.stack((hist_vphi,hist_vphi_sig,hist_vr,hist_vr_sig,hist_vz,hist_vz_sig))
        if self.physical:
            kin *= model.v_scale
        return kin


