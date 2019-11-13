import torch
import math
from scipy.special import roots_legendre


# Functions fixed_quad and _cached_roots_legendre are taken from scipy but adapted to pytorch, and the case of
# integration from 0->1
def _cached_roots_legendre(n):
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


def fixed_quad(func, n=5, dtype=torch.float32):
    y, w = _cached_roots_legendre(n)
    return torch.sum(w.to(dtype=dtype) * func(y.to(dtype=dtype)), axis=-1)


class SpheroidalPotential:
    def __init__(self, rho_func, q=1.0):
        self.q = q
        self.rho = rho_func

    def _f_compute(self, r_cyl, z, rel_tol, direction='r_cyl'):
        if rel_tol is None:
            rel_tol = 1e-6

        r_cyl, z = map(torch.as_tensor, (r_cyl, z))
        assert (r_cyl.dtype == z.dtype) and (r_cyl.device == z.device), "r_cyl and z should be same type on same device"

        if direction == 'r_cyl':
            # Change variables of the integral from BT's tau over 0->inf, to x = (1/tau-1)**3 over 0->1.
            # Tests suggested 3rd power generally provided better convergence than 1,2,4...
            def integrand(x):
                tau = (1 / x - 1) ** 3
                r_cyl_mat, z_mat, x, tau = torch.broadcast_tensors(r_cyl.unsqueeze(-1), z.unsqueeze(-1), x, tau)
                m = torch.sqrt(r_cyl_mat ** 2 / (tau + 1) + z_mat ** 2 / (tau + self.q ** 2))
                return self.rho(m) / (tau + 1) ** 2 / torch.sqrt(tau + self.q ** 2) * 3 * tau / x / (1 - x)

            integral = r_cyl * self._fixedquad(integrand, rel_tol=rel_tol, dtype=z.dtype)

        elif direction == 'z':

            def integrand(x):
                tau = (1 / x - 1) ** 3
                r_cyl_mat, z_mat, x, tau = torch.broadcast_tensors(r_cyl.unsqueeze(-1), z.unsqueeze(-1), x, tau)
                m = torch.sqrt(r_cyl_mat ** 2 / (tau + 1) + z_mat ** 2 / (tau + self.q ** 2))
                return self.rho(m) / (tau + 1) / (tau + self.q ** 2) ** 1.5 * 3 * tau / x / (1 - x)

            integral = z * self._fixedquad(integrand, rel_tol=rel_tol, dtype=z.dtype)

        else:
            raise ValueError("Direction should be ('r_cyl'|'z')")

        return -2 * math.pi * self.q * integral

    def f_r_cyl(self, r_cyl, z, rel_tol=None):
        """Return the force in cylindical R direction at (r_cyl, z)"""
        return self._f_compute(r_cyl, z, rel_tol, direction='r_cyl')

    def f_z(self, r_cyl, z, rel_tol=None):
        """Return the force in the z-direction at (r_cyl, z)"""
        return self._f_compute(r_cyl, z, rel_tol, direction='z')

    def f_r(self, r_cyl, z, *args, **kwargs):
        """Return the radial force at (r_cyl, z)"""
        r_cyl, z = map(torch.as_tensor, (r_cyl, z))
        r = torch.sqrt(r_cyl ** 2 + z ** 2)
        return (r_cyl * self.f_r_cyl(r_cyl, z, *args, **kwargs) +
                z * self.f_z(r_cyl, z, *args, **kwargs)) / r

    def f_theta(self, r_cyl, z, *args, **kwargs):
        """Return the force in the theta direction at (r_cyl, z)"""
        r_cyl, z = map(torch.as_tensor, (r_cyl, z))
        r = torch.sqrt(r_cyl ** 2 + z ** 2)
        return (z * self.f_r_cyl(r_cyl, z, *args, **kwargs) -
                r_cyl * self.f_z(r_cyl, z, *args, **kwargs)) / r

    def vc2(self, r_cyl, z, *args, **kwargs):
        """Return the squared circular velocity at (r_cyl, z)"""
        r_cyl, z = map(torch.as_tensor, (r_cyl, z))
        r = torch.sqrt(r_cyl ** 2 + z ** 2)
        return -self.f_r(r_cyl, z, *args, **kwargs) * r

    def pot_ellip(self, r_cyl, z, *args, **kwargs):
        """Return the elipticity of the potential"""
        r_cyl, z = map(torch.as_tensor, (r_cyl, z))
        return torch.sqrt(z * self.f_r_cyl(r, ang, *args, **kwargs) /
                          (r_cyl * self.f_z(r, ang, *args, **kwargs)))

    @classmethod
    def spherical_to_cylindrical(cls, r, ang):
        z = r * torch.sin(math.pi / 180 * ang)
        r_cyl = torch.sqrt(r ** 2 - z ** 2)
        return z, r_cyl

    @staticmethod
    def _fixedquad(func, n=None, n_max=100, n_min=10, rel_tol=1e-6, dtype=torch.float32):
        """Integrate func from 0->1 using Gaussian quadrature of order n if set.
        Else provide answer with estimated relative error less than rel_tol (up to a
        maximum order of n_max"""
        if n is None:
            val = old_val = fixed_quad(func, n=n_min, dtype=dtype)
            for n in range(n_min + 5, n_max, 5):
                val = fixed_quad(func, n=n, dtype=dtype)
                rel_err = torch.max(torch.abs((val - old_val) / val))
                if rel_err < rel_tol:
                    break
                old_val = val
        else:
            val = fixed_quad(integrand_closed, n=n, dtype=dtype)
        return val

    def get_accelerations(self, positions):
        """Linear interpolate the gridded forces to the specified positions. This should be preceeded
        by a call to grid_acc to (re)compute the accelerations on the grid."""
        positions.rcyl, positions.z
        samples = (positions / self.max).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        image = self.acc.permute(3, 2, 1, 0)  # change to:      C x W x H x D
        image = image.unsqueeze(0)  # change to:  1 x C x W x H x D
        return torch.nn.functional.grid_sample(image, samples, mode='bilinear').squeeze().t()
