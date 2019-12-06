import math
from typing import List, Union, Tuple, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from astro_dynamo.grid import Grid
from astro_dynamo.model import DynamicalModel
from scipy.interpolate import RegularGridInterpolator


class SurfaceDensity(nn.Module):
    def __init__(self, r_range: Union[List[float], Tuple[float], torch.Tensor]=(0., 10.),
                 r_bins: int=20,
                 physical: bool=False):
        super(SurfaceDensity, self).__init__()
        self.physical = physical
        self.dr = (r_range[1] - r_range[0]) / r_bins
        self.r_min = r_range[0]
        self.r_bins = r_bins
        redge = self.r_min + torch.arange(self.r_bins + 1) * self.dr
        self.register_buffer('area', math.pi * (redge[1:] ** 2 - redge[:-1] ** 2))

    def forward(self, model: DynamicalModel) -> torch.Tensor:
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

    def extra_repr(self) -> str:
        return f'r_min={self.r_min}, r_max={self.r_min+self.dr*self.r_bins}, r_bins={self.r_bins}'

    @property
    def rmid(self) -> torch.Tensor:
        return self.r_min + self.dr / 2 + self.dr * torch.arange(self.r_bins, device=self.area.device,
                                                                 dtype=self.area.dtype)

    def evalulate_function(self, surface_density: Callable[[torch.Tensor],torch.Tensor]) -> torch.Tensor:
        return surface_density(self.rmid)


class DiskKinematics(SurfaceDensity):
    def __init__(self,*args, **kw_args):
        super(DiskKinematics, self).__init__(*args, **kw_args)

    def forward(self, model: DynamicalModel) -> torch.Tensor:
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


class ParsecLuminosityFunction():
    def __init__(self,file):
        """Load a luminosity function downloaded from http://stev.oapd.inaf.it/cgi-bin/cmd .
        Should have a range of metalicities and ages. feh should be on a regular grid."""
        with open(file) as fp:
            line = fp.readline()
            while line and line[0]=='#':
                header = line.rstrip().strip("#").split()
                line = fp.readline()
        lf_df = pd.read_csv(file,sep='\s+',comment='#',names=header)
        self.zs = lf_df.Z.unique()
        self.ages = lf_df.age.unique()
        self.mags = lf_df.magbinc.unique()
        assert len(self.zs)*len(self.ages)*len(self.mags) == len(lf_df)
        lf_df.set_index(['age','Z','magbinc'],inplace=True)
        lf_df.sort_index(inplace=True)
        self.interpolators = {}
        self.grids = {}

        for col in lf_df.columns:
            grid = np.zeros((len(self.ages),len(self.zs),len(self.mags)))
            for i,age in enumerate(self.ages):
                for j,z in enumerate(self.zs):
                    grid[i,j,:]=lf_df.loc[(age,z,)][col]
            self.interpolators[col]=RegularGridInterpolator((self.ages,np.log10(self.zs/0.0198)),grid)
            self.grids[col] = grid
        self.fehs = np.log10(self.zs/0.0198)

    def get_single_lf(self,band,age,feh,mags=None):
        """Get the luminosity function for the specified age and feh. Samples at the specified mags, otherwise use the
        native mags from the loaded luminosty funciton"""
        lf = self.interpolators[band]((age,feh))
        if mags is None:
            return {'mag':self.mags, 'number':lf}
        else:
            cumlf = np.cumsum(lf)
            cum_lf_interpolated = np.interp(mags,self.mags,cumlf)
            resampled_lf = np.diff(cum_lf_interpolated,prepend=0)
            resampled_lf[0] = resampled_lf[1]
            return {'mag':mags, 'number':resampled_lf}

    def get_lf_feh_func(self,band,age,feh_func,mags=None):
        """Get the luminsoty function for the specified age and function specifying the feh distribution."""
        weights = feh_func(self.fehs)
        weights /= np.sum(weights)
        lf = None
        for weight, feh in zip(weights,self.fehs):
            this_lf = self.get_single_lf(band,age,feh,mags=mags)
            if lf is not None:
                lf+=weight*this_lf['number']
            else:
                lf=weight*this_lf['number']
        return {'mag':this_lf['mag'], 'number':lf}


def convolve_distance_modulus_with_lf(histogram, lf_number, distance_mod_dim=-1):
    """Helper function that convolves a histogram as a function of distance modulus with a luminosity function. Distance
    modulus dimension should be specified in distance_mod_dim, otherwise the last is assumed. Your magnitudes
    should have the same (regular) spacing in both histogram and luminosity function."""
    if distance_mod_dim is not -1:
        histogram = histogram.transpose(distance_mod_dim, -1)
    reshaped = histogram.reshape(-1, 1, histogram.shape[-1])
    if reshaped.dtype == torch.long:
        reshaped = reshaped.type(lf_number.dtype)
    for dim in range(len(reshaped.shape) - 1):
        lf_number = lf_number.unsqueeze(0)

    output = torch.nn.functional.conv1d(lf_number, reshaped.flip(dims=(-1,)))
    output = output.reshape(tuple(histogram.shape[:-1]) + (-1,))
    if distance_mod_dim is not -1:
        output = output.transpose(distance_mod_dim, -1)
    return output


class PositionMagnitude(nn.Module):
    def __init__(self, luminosity_function, l_range=(-90., 90.), n_l=90,
                 b_range=(-12., 12.), n_b=20,
                 mag_range=(11., 12.), n_mag=int(1 / 0.2 + 1)):
        """Target of a grid in galactic coordinates and magnitude. To convert from the snapshot distances to a magntiude
         distribution we need the luminosity function, which is assumed constant."""
        super(PositionMagnitude, self).__init__()
        self.register_buffer('luminosity_function', torch.as_tensor(luminosity_function['number'], dtype=torch.float))

        first_mu_bin = mag_range[1] - luminosity_function['mag'][-1]
        last_mu_bin = mag_range[0] - luminosity_function['mag'][0]
        n_mu_bins = len(luminosity_function['mag']) - n_mag + 1

        self.gridder = Grid(grid_edges=torch.tensor(((l_range[0], l_range[-1]),
                                                     (b_range[0], b_range[-1]),
                                                     (first_mu_bin, last_mu_bin))),
                            n=(n_l, n_b, n_mu_bins))

        self.register_buffer('apparent_mags', torch.linspace(mag_range[0],
                                                             mag_range[1],
                                                             n_mag))

        self._check_apparent_mags(luminosity_function['mag'])

    def _check_apparent_mags(self, lf_mags):
        """Getting the apparent magnitude correct is annoying. Here we do a check that everything is setup ok."""
        out = convolve_distance_modulus_with_lf(self.gridder.grid_data(torch.as_tensor([[0, 0, 0]])),
                                                self.luminosity_function)
        assert torch.allclose(self.apparent_mags, self.gridder.cell_midpoints[2][-1] + lf_mags[0] +
                              self.gridder.dx[2] * torch.arange(out.shape[2]))

    def forward(self, model):
        l_b_mu = self.gridder.grid_data(model.l_b_mu, weights=model.masses, method='nearest')
        out = convolve_distance_modulus_with_lf(l_b_mu, self.luminosity_function)
        return out

    def extra_repr(self):
        return f'mags={self.apparent_mags[0]}->{self.apparent_mags[-1]}'

