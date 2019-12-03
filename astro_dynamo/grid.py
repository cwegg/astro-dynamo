from typing import Union, List

import torch
import torch.sparse
from torch import nn


class Grid(nn.Module):
    """Class for gridding data. Assumes grids are symmertic about zero."""

    def __init__(self, grid_edges: torch.Tensor = torch.tensor((10., 10., 10.)),
                 n: Union[list, tuple] = (256, 256, 256)):
        super().__init__()
        if grid_edges.dim() == 1:
            grid_min = -grid_edges
            grid_max = grid_edges
        else:
            grid_min = grid_edges[..., 0]
            grid_max = grid_edges[..., 1]
        dx = (grid_max - grid_min) / (grid_max.new_tensor(n) - 1)
        self.register_buffer('dx', dx)
        self.register_buffer('n', torch.as_tensor(n, dtype=torch.long))
        self.register_buffer('min', grid_min)
        self.register_buffer('max', grid_max)

    def forward(self, snap):
        return self.grid_data(snap.positions, weights=snap.masses, method='nearest')

    def extra_repr(self):
        return f'min={self.min}, max={self.max}, size={self.n}'

    @property
    def n(self):
        """Returns grid size"""
        return self.data.shape

    @property
    def cell_edges(self) -> List[torch.Tensor]:
        return [torch.linspace(self.min[dim].item() - 0.5 * self.dx[dim].item(),
                               self.max[dim].item() + 0.5 * self.dx[dim].item(),
                               self.n[dim].item() + 1) for dim in range(len(self.n))]

    @property
    def cell_midpoints(self) -> List[torch.Tensor]:
        return [torch.linspace(self.min[dim].item(),
                               self.max[dim].item(),
                               self.n[dim].item()) for dim in range(len(self.n))]

    def ingrid(self, positions: torch.Tensor, h: float = None) -> torch.Tensor:
        """Test if positions are in the grid"""
        if h is None:
            return ((positions > self.min[None, :]) &
                    (positions < self.max[None, :])).all(dim=1)
        else:
            return ((positions > self.min[None, :] - h * self.dx[None, :]) &
                    (positions < self.max[None, :] + h * self.dx[None, :])).all(dim=1)

    def _float_idx(self, positions: torch.Tensor) -> torch.Tensor:
        return (positions - self.min[None, :]) / self.dx

    def grid_data(self, positions: torch.Tensor, weights: torch.Tensor = None,
                  method: str = 'nearest') -> torch.Tensor:
        """Places data from positions onto grid using method='nearest'|'cic'
        where cic=cloud in cell. Returns gridded data and stores it as class attribute
        data.

        Note that for nearest we place the data at the nearest grid point. This corresponds to grid edges at
        the property cell_edges. For cic then if the particle lies on a grid point it is entirely places in that
        cell, otherwise it is split over multiple cells.
        """
        dimensions = tuple(self.n)
        fi = self._float_idx(positions)

        if weights is None:
            weights = positions.new_ones(positions.shape[0])

        if method == 'nearest':
            i = (fi + 0.5).type(torch.int64)
            gd = ((i>=0) & (i<self.n[None,:])).all(dim=1)
            if gd.sum() == 0:
                return positions.new_zeros(dimensions)
            data = torch.sparse.FloatTensor(i[gd].t(), weights[gd], size=dimensions).to_dense().reshape(dimensions).type(
                dtype=positions.dtype)
        elif method == 'cic':

            dimensions = tuple(self.n + 2)
            i = fi.floor()
            offset = fi - i
            i = i.type(torch.int64) + 1

            gd = ((i>=1) & (i<=self.n[None, :])).all(dim=1)
            if gd.sum() == 0:
                return positions.new_zeros(dimensions)
            weights, i, offset = weights[gd], i[gd], offset[gd]

            data = weights.new_zeros(dimensions)
            if len(self.n) == 1:
                # 1d is easier to handle as a special case
                indexes = torch.tensor([[0], [1]], device=i.device)
            else:
                twidle = torch.tensor([0, 1], device=i.device)
                indexes = torch.cartesian_prod(*torch.split(twidle.repeat(len(self.n)), 2))
            for offsetdims in indexes:
                thisweights = torch.ones_like(weights)
                for dimi, offsetdim in enumerate(offsetdims):
                    if offsetdim == 0:
                        thisweights *= (torch.tensor(1.0) - offset[..., dimi])
                    if offsetdim == 1:
                        thisweights *= offset[..., dimi]
                data += torch.sparse.FloatTensor((i + offsetdims).t(), thisweights * weights,
                                                 size=dimensions).to_dense().type(dtype=positions.dtype)
            for dim in range(len(self.n)):
                data = data.narrow(dim, 1, data.shape[dim] - 2)
        else:
            raise ValueError(f'Method {method} not recognised. Allowed values are nearest|cic')

        return data


class ForceGrid(Grid):
    def __init__(self,
                 grid_edges: Union[list, tuple, torch.Tensor] = torch.tensor((10., 10., 10.)),
                 n: Union[list, tuple, torch.Tensor] = (256, 256, 256),
                 rho: torch.Tensor = None,
                 smoothing: float = 1.0):
        super().__init__(grid_edges, n)
        self.greenfft = None
        self.register_buffer('epsilon', torch.as_tensor(smoothing))
        self.register_buffer('acc', None)
        self.register_buffer('pot', None)
        if rho is None:
            rho = self.min.new_zeros(tuple(n))
        self.register_buffer('rho', rho)

    def update_density(self, positions: torch.Tensor, weights: torch.Tensor = None,
                       method: str = 'nearest', fractional_update: float = 1.0) -> None:
        new_data = self.grid_data(positions, weights, method)
        if fractional_update == 1.0:
            self.rho = new_data
        else:
            self.rho.lerp_(new_data, fractional_update)

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
            self.update_density(positions, weights, method)
        rho = self.rho
        nx, ny, nz = rho.shape[0], rho.shape[1], rho.shape[2]

        padrho = rho.new_zeros([2 * nx, 2 * ny, 2 * nz])
        padrho[0:nx, 0:ny, 0:nz] = rho

        if self.greenfft is None or padrho.shape != self.greenfft.shape[:-1] or \
                self.greenfft.device != rho.device or self.greenfft.dtype != rho.dtype:
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
