import torch
import torch.nn as nn

from .analysesnap import align_bar

_symmetrize_matrix = lambda x,dim : (x+x.flip(dims=[dim]))/2
class DynamicalModel(nn.Module):
    """DynamicalModels class. This containts a snapshot of the particles, the potentials
    in which they move, and the targets to which the model should be fitted.

    Attributes:
        snap:
            Should be a SnapShot whose masses will be optimised

        potentials:
            The potentials add. If self gravity is not required set self_gravity_update to None.
            If self gravity is required then the potential of the snapshot should be in potentials[0]
            and self_gravity_update represents how much update the running average of the density on
            each iteration. Default value is 0.2 which is then exponential average with timescale
            5 snapshots(=1/0.2).

        targets:
            A list of targets. Running
                model = DynamicalModel(snap, potentials, targets)
                current_target_list = model()
            will provide an list of these targets evaluated with the present model. These are then
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
        return [target(self.snap) for target in self.targets]

    def integrate(self, steps=256):
        with torch.no_grad():
            self.snap.leapfrog_steps(potentials=self.potentials, steps=steps)
            if self.self_gravity_update is not None:
                self.potentials[0].update_density(self.snap.positions, self.snap.masses.detach(),
                                                  fractional_update=self.self_gravity_update)

    def update_potential(self, dm_potential=None, update_velocities=True):
        with torch.no_grad():
            if update_velocities:
                old_accelerations = self.snap.get_accelerations(self.potentials, self.snap.positions)
                old_vc = torch.sum(-old_accelerations * self.snap.positions, dim=-1).sqrt()
            self.potentials[0].rho = _symmetrize_matrix(
                _symmetrize_matrix(_symmetrize_matrix(self.potentials[0].rho, 0), 1), 2)
            self.potentials[0].grid_accelerations()
            if dm_potential is not None:
                self.potentials[1]=dm_potential
            if update_velocities:
                new_accelerations = self.snap.get_accelerations(self.potentials, self.snap.positions)
                new_vc = torch.sum(-new_accelerations * self.snap.positions, dim=-1).sqrt()
                gd = torch.isfinite(new_vc / old_vc) & (new_vc / old_vc > 0)
                self.snap.velocities[gd, :] *= (new_vc / old_vc)[gd, None]
                print(f'gd: {gd.sum()} mean fractional vc change: {(new_vc / old_vc)[gd].mean()}')
            align_bar(self.snap)

    def resample(self, velocity_perturbation=0.01):
        """Resample the model to equal mass particles.

        Note that the snapshot changes and so the parameters of
        the model also change in a way that any optimiser that keeps parameter-by-parameter information e.g.
        gradients must also be update."""
        with torch.no_grad():
            self.snap = self.snap.resample(self.potentials, velocity_perturbation=velocity_perturbation)
            align_bar(self.snap)
