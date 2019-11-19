import torch
import math
from scipy.special import roots_legendre
import numpy as np
import scipy.optimize
import matplotlib.pylab as plt
from typing import Callable


# Functions fixed_quad and _cached_roots_legendre are adapted from scipy but adapted to pytorch, and the case of
# integration from 0->1. Copyright notice applies to just the modified _cached_roots_legendre and fixed_quad functions

# Copyright © 2001, 2002 Enthought, Inc.
# All rights reserved.

# Copyright © 2003-2013 SciPy Developers.
# All rights reserved.
def _cached_roots_legendre(n: int) -> (torch.tensor, torch.tensor):
    """
    Cache roots_legendre results to speed up calls of the fixed_quad
    function.
    """
    if n in _cached_roots_legendre.cache:
        return _cached_roots_legendre.cache[n]
    x, w = torch.tensor(roots_legendre(n), dtype=torch.float64)
    _cached_roots_legendre.cache[n] = (
        torch.tensor(0.5, dtype=torch.float64) * (x + torch.tensor(1, dtype=torch.float64)),
        torch.tensor(0.5, dtype=torch.float64) * w)
    return _cached_roots_legendre.cache[n]


_cached_roots_legendre.cache = dict()


def fixed_quad(func: Callable[[torch.tensor], torch.tensor], n: int = 5, dtype: torch.dtype = torch.float32,
               device: torch.device = None) -> torch.tensor:
    y, w = _cached_roots_legendre(n)
    return torch.sum(w.to(dtype=dtype, device=device) * func(y.to(dtype=dtype, device=device)), axis=-1)


class SpheroidalPotential:
    def __init__(self, rho_func: Callable[[torch.tensor], torch.tensor], q: torch.tensor = torch.tensor([1.0])):
        self.q = torch.as_tensor(q)
        self.rho = rho_func
        self.grid = None
        self.r_max, self.z_max = None, None

    def to(self, device=None, dtype=None, *args, **kwargs) -> 'SpheroidalPotential':
        if (dtype is None or self.q.dtype == dtype) and (device is None or self.q.device == device):
            return self
        else:
            new_pot = SpheroidalPotential(self.rho, q=self.q.to(device=device, dtype=dtype, *args, **kwargs))
            if self.grid is not None: new_pot.grid = self.grid.to(device=device, dtype=dtype, *args, **kwargs)
            new_pot.r_max, new_pot.z_max = self.r_max, self.z_max
            return new_pot

    def _f_compute(self, r_cyl: torch.tensor, z: torch.tensor, rel_tol: float = 1e-6, direction: str = 'r_cyl', *args,
                   **kwargs) -> torch.tensor:
        r_cyl, z = map(torch.as_tensor, (r_cyl, z))
        assert (r_cyl.dtype == z.dtype) and (r_cyl.device == z.device), "r_cyl and z should be same type on same device"

        if direction == 'r_cyl':
            # Change variables of the integral from BT's tau over 0->inf, to x = (1/tau-1)**3 over 0->1.
            # Tests suggested 3rd power generally provided better convergence than 1,2,4...
            def integrand(x):
                tau = (1 / x - 1) ** 3
                r_cyl_mat, z_mat, x, tau = torch.broadcast_tensors(r_cyl.unsqueeze(-1), z.unsqueeze(-1), x, tau)
                m = torch.sqrt(r_cyl_mat ** 2 / (tau + 1) + z_mat ** 2 / (tau + self.q ** 2))
                return self.rho(m, *args, **kwargs) / (tau + 1) ** 2 / torch.sqrt(tau + self.q ** 2) * 3 * tau \
                       / x / (1 - x)

            integral = r_cyl * self._fixedquad(integrand, rel_tol=rel_tol, dtype=z.dtype, device=z.device)

        elif direction == 'z':

            def integrand(x):
                tau = (1 / x - 1) ** 3
                r_cyl_mat, z_mat, x, tau = torch.broadcast_tensors(r_cyl.unsqueeze(-1), z.unsqueeze(-1), x, tau)
                m = torch.sqrt(r_cyl_mat ** 2 / (tau + 1) + z_mat ** 2 / (tau + self.q ** 2))
                return self.rho(m, *args, **kwargs) / (tau + 1) / (tau + self.q ** 2) ** 1.5 * 3 * tau \
                       / x / (1 - x)

            integral = z * self._fixedquad(integrand, rel_tol=rel_tol, dtype=z.dtype, device=z.device)

        else:
            raise ValueError("Direction should be ('r_cyl'|'z')")

        return -2 * math.pi * self.q * integral

    def f_r_cyl(self, r_cyl: torch.tensor, z: torch.tensor, *args, **kwargs) -> torch.tensor:
        """Return the force in cylindrical R direction at (r_cyl, z)"""
        return self._f_compute(r_cyl, z, direction='r_cyl', *args, **kwargs)

    def f_z(self, r_cyl: torch.tensor, z: torch.tensor, *args, **kwargs) -> torch.tensor:
        """Return the force in the z-direction at (r_cyl, z)"""
        return self._f_compute(r_cyl, z, direction='z', *args, **kwargs)

    def f_r(self, r_cyl: torch.tensor, z: torch.tensor, *args, **kwargs) -> torch.tensor:
        """Return the radial force at (r_cyl, z)"""
        r_cyl, z = map(torch.as_tensor, (r_cyl, z))
        r = torch.sqrt(r_cyl ** 2 + z ** 2)
        return (r_cyl * self.f_r_cyl(r_cyl, z, *args, **kwargs) +
                z * self.f_z(r_cyl, z, *args, **kwargs)) / r

    def f_theta(self, r_cyl: torch.tensor, z: torch.tensor, *args, **kwargs) -> torch.tensor:
        """Return the force in the theta direction at (r_cyl, z)"""
        r_cyl, z = map(torch.as_tensor, (r_cyl, z))
        r = torch.sqrt(r_cyl ** 2 + z ** 2)
        return (z * self.f_r_cyl(r_cyl, z, *args, **kwargs) -
                r_cyl * self.f_z(r_cyl, z, *args, **kwargs)) / r

    def vc2(self, r_cyl: torch.tensor, z: torch.tensor, *args, **kwargs) -> torch.tensor:
        """Return the squared circular velocity at (r_cyl, z)"""
        r_cyl, z = map(torch.as_tensor, (r_cyl, z))
        r = torch.sqrt(r_cyl ** 2 + z ** 2)
        return -self.f_r(r_cyl, z, *args, **kwargs) * r

    def pot_ellip(self, r_cyl: torch.tensor, z: torch.tensor, *args, **kwargs) -> torch.tensor:
        """Return the elipticity of the potential"""
        r_cyl, z = map(torch.as_tensor, (r_cyl, z))
        return torch.sqrt(z * self.f_r_cyl(r_cyl, z, *args, **kwargs) /
                          (r_cyl * self.f_z(r_cyl, z, *args, **kwargs)))

    @classmethod
    def spherical_to_cylindrical(cls, r: torch.tensor, ang: torch.tensor) -> (torch.tensor, torch.tensor):
        z = r * torch.sin(math.pi / 180 * ang)
        r_cyl = torch.sqrt(r ** 2 - z ** 2)
        return z, r_cyl

    @staticmethod
    def _fixedquad(func, n=None, n_max=100, n_min=10, rel_tol=1e-6, dtype=torch.float32, device=None) -> torch.tensor:
        """Integrate func from 0->1 using Gaussian quadrature of order n if set.
        Else provide answer with estimated relative error less than rel_tol (up to a
        maximum order of n_max"""
        if n is None:
            val = old_val = fixed_quad(func, n=n_min, dtype=dtype, device=device)
            for n in range(n_min + 5, n_max, 5):
                val = fixed_quad(func, n=n, dtype=dtype, device=device)
                rel_err = torch.max(torch.abs((val - old_val) / val))
                if rel_err < rel_tol:
                    break
                old_val = val
        else:
            val = fixed_quad(func, n=n, dtype=dtype, device=device)
        return val

    def grid_accelerations(self, r_max=10., z_max=10., r_bins=512, z_bins=1024):
        """Linear interpolate the gridded forces to the specified positions. This should be preceeded
        by a call to grid_acc to (re)compute the accelerations on the grid."""
        r = torch.linspace(0, r_max, r_bins, device=self.q.device)
        z = torch.linspace(-z_max, z_max, z_bins, device=self.q.device)
        self.r_max, self.z_max = r_max, z_max
        f_r_cyl = self.f_r_cyl(r, z.unsqueeze(-1))
        f_z = self.f_z(r, z.unsqueeze(-1))
        self.grid = torch.stack((f_r_cyl, f_z)).unsqueeze(0)  # .permute(0,1,3,2)

    def get_accelerations_cyl(self, positions):
        """Linear interpolate the gridded forces to the specified positions. This should be preceeded
        by a call to grid_acc to (re)compute the accelerations on the grid.
        Returns forces in (r_cyl, z) directions."""
        samples = torch.stack((2 * torch.sqrt(positions[..., 0] ** 2 + positions[..., 1] ** 2) / self.r_max - 1,
                               positions[..., 2] / self.z_max), dim=-1)
        samples = samples.unsqueeze(0).unsqueeze(2)
        return torch.nn.functional.grid_sample(self.grid, samples, mode='bilinear', align_corners=True).squeeze().t()

    def get_accelerations(self, positions):
        """Linear interpolate the gridded forces to the specified positions. This should be preceeded
        by a call to grid_acc to (re)compute the accelerations on the grid.
        Returns forces in xyz directions."""
        acc_cyl = self.get_accelerations_cyl(positions)
        acc = torch.zeros_like(positions)
        r_cyl = torch.sqrt(positions[..., 0] ** 2 + positions[..., 1] ** 2)
        acc[..., 0] = acc_cyl[..., 0] * positions[..., 0] / r_cyl
        acc[..., 1] = acc_cyl[..., 0] * positions[..., 1] / r_cyl
        acc[..., 2] = acc_cyl[..., 1]
        return acc


def fit_q_slice_to_snapshot(snap):
    """Fit a flattening to a particle snapshot. We assume that the density varies like rho ~ m**gamma with
    m**2 = Rcyl**2 + z**2/q**2 in this slice. This should therefore be a fairly narrow slice so this approximation
    is reasonable (it will also fail for very flattened systems). Returns (q,qerr)"""
    st = snap.z / snap.r
    mintheta = np.sin(30. / 180 * np.pi)
    mass, edges = np.histogram(st ** 2, np.linspace(mintheta ** 2, 1, 100), weights=snap.masses)
    x = 0.5 * (edges[:-1] + edges[1:])
    ctedges = 1 - np.sqrt(edges)
    vol = ctedges[:-1] - ctedges[1:]
    rho = mass / vol

    def f(x, a, b):
        return a * (b * x + 1)

    mass2, edges = np.histogram(st ** 2, np.linspace(mintheta ** 2, 1, 100), weights=snap.masses ** 2)
    Neff = mass ** 2 / mass2
    popt, pcov = scipy.optimize.curve_fit(f, x, rho, p0=[-0.8, 1.0], sigma=rho / np.sqrt(Neff))
    perr = np.sqrt(np.diag(pcov))
    q = 1 / np.sqrt(1 - popt[1])
    qerr = q * 0.5 * perr[1] / popt[1]
    return q, qerr


def fit_q_to_snapshot(snap, r_range=(1, 20), r_bins=10, plot=False):
    """Fit a flattening to a particle snapshot. We assume that the density varies like rho ~ m**gamma with
    m**2 = Rcyl**2 + z**2/q**2 in this slice. This should therefore be a fairly narrow slice so this approximation
    is reasonable (it will also fail for very flattened systems). Returns (q,qerr)"""
    r_bins = np.linspace(r_range[0], r_range[1], r_bins)
    qs, qerrs = np.zeros(len(r_bins) - 1), np.zeros(len(r_bins) - 1)
    for ir in range(len(r_bins) - 1):
        snap_slice = snap.dm[(snap.dm.r > r_bins[ir]) & (snap.dm.r <= r_bins[ir + 1])]
        qs[ir], qerrs[ir] = fit_q_slice_to_snapshot(snap_slice)
    q = np.sum(qs / qerrs ** 2) / np.sum(1 / qerrs ** 2)
    qerr = 1 / np.sqrt(np.sum(1 / qerrs ** 2))
    if plot:
        f, ax = plt.subplots()
        r = 0.5 * (r_bins[:-1] + r_bins[1:])
        ax.errorbar(r, qs, yerr=qerrs, fmt='o', color='k', ecolor='k')
        ax.axhline(y=q, color='r')
        ax.axhspan(ymin=q - qerr, ymax=q + qerr, color='r', alpha=0.2)
        ax.set_ylabel(r'$q_\rho$')
        ax.set_xlabel(r'$r$ [kpc]')
    return q, qerr


def fit_potential_to_snap(snap, rho_func, m_range=(1, 20), m_bins=50, q=None, *args, **kwargs):
    """Fit an ellipsoidal density function of the form  rho_func(m,p[0],p[1],....) to the snapshot.
    Returns pot : the best fitting potential.
    Must supply initial parameters for this fit.
    Can optionally supply q or this to be fit.
    Fit occurs in bins: np.linspace(m_range[0], m_range[1], m_bins).
    Set plot=True to compare fit to the snapshot."""
    if q is None:
        q, qerr = fit_q_to_snapshot(snap, r_range=m_range, r_bins=m_bins)
    popt, perr = fit_spheroidal_function_to_snap(snap, rho_func, q=q, *args, **kwargs)
    return SpheroidalPotential(lambda m: rho_func(m, *popt), q=q)


def fit_spheroidal_function_to_snap(snap, rho_func, init_parms, m_range=(1, 20), m_bins=50, q=None, plot=False):
    """Fit an ellipsoidal density function of the form  rho_func(m,p[0],p[1],....) to the snapshot.
    Returns p, perr : the best fitting parameters of rho_func and their errors.
    Must supply initial parameters for this fit.
    Can optionally supply q or this to be fit.
    Fit occurs in bins: np.linspace(m_range[0], m_range[1], m_bins).
    Set plot=True to compare fit to the snapshot."""
    if q is None:
        q, qerr = fit_q_to_snapshot(snap, r_range=m_range, r_bins=m_bins)

    m = np.sqrt((snap.rcyl) ** 2 + (snap.z / q) ** 2)
    m_bins = np.linspace(m_range[0], m_range[1], m_bins)
    (mass, medge) = np.histogram(m, m_bins, weights=snap.masses)

    volcorr = q
    vol = 4 * np.pi * (medge[1:] ** 3 - medge[:-1] ** 3) / 3 * volcorr
    mmid = 0.5 * (medge[1:] + medge[:-1])
    rho = mass / vol

    (mass2, medge) = np.histogram(m, m_bins, weights=snap.dm.masses)
    Neff = mass ** 2 / mass2
    rho_numpy = lambda x, *args, **kwargs: rho_func(x, *args, **kwargs).cpu().numpy()
    popt, pcov = scipy.optimize.curve_fit(rho_numpy, mmid, rho, p0=init_parms, sigma=rho / np.sqrt(Neff))
    perr = np.sqrt(np.diag(pcov))
    if plot:
        f, ax = plt.subplots()
        ax.plot(mmid, rho)
        ax.set_yscale('log')
        ax.plot(mmid, rho_func(mmid, *popt))
        ax.set_xlabel(r'$m$')
        ax.set_ylabel(r'$\rho$')
    return popt, perr
