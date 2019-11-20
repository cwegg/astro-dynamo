from functools import reduce
from typing import Union

import torch
import torch.sparse
from torch import nn


class Grid(nn.Module):
    """Class for gridding data. Assumes grids are symmertic about zero."""

    def __init__(self, grid_edges: torch.Tensor = torch.tensor((10., 10., 10.)),
                 n: Union[list, tuple] = (256, 256, 256),
                 data: torch.Tensor = None):
        super(Grid, self).__init__()
        if grid_edges.dim() == 1:
            grid_min = -grid_edges
            grid_max = grid_edges
        else:
            grid_min = grid_edges[..., 0]
            grid_max = grid_edges[..., 1]
        dx = (grid_max - grid_min) / (grid_max.new_tensor(n) - 1)
        if data is None:
            data = grid_edges.new_zeros(n)

        self.register_buffer('dx', dx)
        self.register_buffer('min', grid_min)
        self.register_buffer('max', grid_max)
        self.register_buffer('data', data)

    def extra_repr(self):
        return f'min={self.min}, max={self.max}, size={self.data.shape}'

    @property
    def n(self):
        """Returns grid size"""
        return self.data.shape

    @property
    def x(self) -> torch.Tensor:
        """Returns grid points in first dimension"""
        return torch.linspace(self.min[0].item(), self.max[0].item(), self.n[0])

    @property
    def y(self) -> torch.Tensor:
        """Returns grid points in second dimension"""
        return torch.linspace(self.min[1].item(), self.max[1].item(), self.n[1])

    @property
    def z(self) -> torch.Tensor:
        """Returns grid points in third dimension"""
        return torch.linspace(self.min[2].item(), self.max[2].item(), self.n[2])

    def ingrid(self, positions: torch.Tensor, h: float = None) -> torch.Tensor:
        """Test if positions are in the grid"""
        if h is None:
            return ((positions > self.min[None, :]) &
                    (positions < self.max[None, :])).all(dim=1)
        else:
            return ((positions > self.min[None, :] + h * self.dx[None, :]) &
                    (positions < self.max[None, :] - h * self.dx[None, :])).all(dim=1)

    def _float_idx(self, positions: torch.Tensor) -> torch.Tensor:
        return (positions - self.min[None, :]) / self.dx

    def _threed_to_oned(self, i: torch.Tensor) -> torch.Tensor:
        i1d = (i[:, 0] * self.n[1] + i[:, 1]) * self.n[2] + i[:, 2]
        return i1d

    def grid_data(self, positions: torch.Tensor, weights: torch.Tensor = None,
                  method: str = 'nearest', fractional_update: float = 1.0) -> torch.Tensor:
        """Places data from positions onto grid using method='nearest'|'cic'
        where cic=cloud in cell. Returns gridded data and stores it as class attribute
        data"""

        n_elements = reduce(lambda x, y: x * y, self.n)
        fi = self._float_idx(positions)
        if method == 'nearest':
            gd = self.ingrid(positions)
        else:
            gd = self.ingrid(positions, h=0.5)

        if weights is not None:
            weights = weights[gd]
        else:
            weights = positions.new_ones(gd.sum())

        if gd.sum() == 0:
            new_data = positions.new_zeros(self.n)
        elif method == 'nearest':
            i = (fi + 0.5).type(torch.int64)
            new_data = torch.sparse.FloatTensor(i[gd].t(), weights, size=self.n).to_dense().reshape(
                self.n).type(dtype=positions.dtype)
        elif method == 'cic':
            i = fi[gd, ...].floor()
            offset = fi[gd, ...] - i
            i = i.type(torch.int64)

            new_data = weights.new_zeros(n_elements)
            twidle = torch.tensor([0, 1], device=i.device)
            indexes = torch.cartesian_prod(*torch.split(twidle.repeat(3), 2))
            for offsetdims in indexes:
                thisweights = torch.ones_like(weights)
                for dimi, offsetdim in enumerate(offsetdims):
                    if offsetdim == 0:
                        thisweights *= (torch.tensor(1.0) - offset[..., dimi])
                    if offsetdim == 1:
                        thisweights *= offset[..., dimi]
                new_data += torch.bincount(self._threed_to_oned(i + offsetdims), minlength=n_elements,
                                           weights=thisweights * weights)
            new_data = new_data.reshape(self.n)
        else:
            raise ValueError(f'Method {method} not recognised. Allowed values are nearest|cic')

        self.data.lerp_(new_data, fractional_update)
        return self.data


class ForceGrid(Grid):
    def __init__(self,
                 grid_edges: Union[list, tuple, torch.Tensor] = torch.tensor((10., 10., 10.)),
                 n: Union[list, tuple, torch.Tensor] = (256, 256, 256),
                 data: torch.Tensor = None,
                 smoothing: float = 1.0):
        super().__init__(grid_edges, n, data)

        self.greenfft = None
        self.pot = None
        self.epsilon = smoothing
        self.register_buffer('acc', acc)

    def to(self, device: Union[torch.device, str]) -> 'ForceGrid':
        if self.min.device == device:
            return self
        else:
            grid_edges = torch.stack((self.min, self.max), dim=1).to(device)
            grid = ForceGrid(grid_edges=grid_edges, n=self.n, data=self.data.to(device))
            if self.acc is not None: grid.acc = self.acc.to(device)
            if self.pot is not None: grid.pot = self.pot.to(device)
            return grid

    @staticmethod
    def smoothing_pot(r: torch.Tensor, epsilon: float = 1.) -> torch.Tensor:
        """Taken from the manual of the Galaxy code by Jerry Sellwood."""
        x = r / epsilon
        pot = torch.as_tensor(-1.0) / r
        i = (x < 1)
        pot[i] = (-1.4 + x[i] ** 2 * (- 0.3 * x[i] ** 2 + 2. / 3 + 0.1 * x[i] ** 3)) / epsilon
        i = (x >= 1) & (x < 2)
        pot[i] = (-1.6 + 1.0 / (15.0 * x[i]) + x[i] ** 2 * (4. / 3 - x[i] + 0.3 * x[i] ** 2 - x[i] ** 3 / 30)) / epsilon
        return pot

    @staticmethod
    def complex_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Function to do complex multiplication: Pytorch has no complex dtype but
        stores complex numbers as an additional two long dimension"""
        return torch.stack([a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1],
                            a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]], dim=-1)

    def get_accelerations(self, positions: torch.Tensor) -> torch.Tensor:
        """Linear intepolate the gridded forces to the specified positions. This should be preceeded
        by a call to grid_acc to (re)compute the accelerations on the grid."""
        # torch.nn.functional.grid_sample is fast but frustrating to juggle the inputs. See the pytorch documentation
        samples = (positions / self.max).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        image = self.acc.permute(3, 2, 1, 0)  # change to:      C x W x H x D
        image = image.unsqueeze(0)  # change to:  1 x C x W x H x D
        return torch.nn.functional.grid_sample(image, samples, mode='bilinear', align_corners=True).squeeze().t()

    def get_potential(self, positions: torch.Tensor) -> torch.Tensor:
        """Linear intepolate the gridded potential to the specified positions. This should be preceeded
        by a call to grid_acc to (re)compute the potential on the grid."""
        # torch.nn.functional.grid_sample is fast but frustrating to juggle the inputs. See the pytorch documentation
        samples = (positions / self.max).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        image = self.pot.permute(2, 1, 0)  # change to:      C x W x H x D - with C=1 for potential
        image = image.unsqueeze(0).unsqueeze(0)  # change to:  1 x C x W x H x D - with C=1 for potential
        return torch.nn.functional.grid_sample(image, samples, mode='bilinear', align_corners=True).squeeze().t()

    def grid_accelerations(self, positions: torch.Tensor = None, weights: torch.Tensor = None,
                           method: str = 'nearest'):
        """Takes the brute force approach of removing periodic images by grid doubling in every dimension."""
        if positions is not None:
            self.grid_data(positions, weights, method)
        rho = self.data
        nx, ny, nz = rho.shape[0], rho.shape[1], rho.shape[2]

        padrho = rho.new_zeros([2 * nx, 2 * ny, 2 * nz])
        padrho[0:nx, 0:ny, 0:nz] = rho

        if self.greenfft is None or padrho.shape != self.greenfft.shape[:-1]:
            x = torch.arange(-nx, nx, dtype=rho.dtype, device=rho.device) * self.dx[0]
            y = torch.arange(-ny, ny, dtype=rho.dtype, device=rho.device) * self.dx[1]
            z = torch.arange(-nz, nz, dtype=rho.dtype, device=rho.device) * self.dx[2]
            self.greenfft = torch.rfft(ForceGrid.smoothing_pot(
                (x[:, None, None] ** 2 + y[None, :, None] ** 2 + z[None, None, :] ** 2).sqrt(),
                epsilon=self.epsilon),
                3, onesided=False)
        rhofft = torch.rfft(padrho, 3, onesided=False)
        del padrho
        rhofft = ForceGrid.complex_mul(rhofft, self.greenfft)
        self.pot = torch.irfft(rhofft, 3, onesided=False)
        del rhofft

        self.pot = self.pot[nx:, ny:, nz:]
        self.acc = self.pot.new_zeros(self.pot.shape + (3,))
        for dim in (0, 1, 2):
            self.acc[..., dim] = -self.__diff(self.pot, dim, d=self.dx[dim].item())

    @staticmethod
    def __diff(vector: torch.Tensor, dim: int, d: float = 1.0) -> torch.Tensor:
        """Helper function for differentiating the potential to get the accelerations"""
        ret = (vector.roll(-1, dim) - vector.roll(1, dim)) / (2 * d)
        if dim == 0:
            ret[0, ...] = (vector[1, ...] - vector[0, ...]) / d
            ret[-1, ...] = (vector[-1, ...] - vector[-2, ...]) / d
        if dim == 1:
            ret[:, 0, ...] = (vector[:, 1, ...] - vector[:, 0, ...]) / d
            ret[:, -1, ...] = (vector[:, -1, ...] - vector[:, -2, ...]) / d
        if dim == 2:
            ret[:, :, 0, ...] = (vector[:, :, 1, ...] - vector[:, :, 0, ...]) / d
            ret[:, :, -1, ...] = (vector[:, :, -1, ...] - vector[:, :, -2, ...]) / d
        return ret
