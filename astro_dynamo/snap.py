import torch
import math
import numpy as np
from enum import IntFlag
from torch.utils.data import WeightedRandomSampler

float_dtype = torch.float32

class ParticleType(IntFlag):
    """Enum for storing particle type: Gas, Star, DarkMatter"""
    # If the values are changed then SnapShot.__organise will need updating
    DarkMatter = 1
    Gas = 2
    Star = 4
    Baryonic = 6


class SnapShot:
    def __init__(self, file=None, positions=None, velocities=None,
                 masses=None, particle_type=None, time=0., omega=0., dt=None,
                 particle_type_mapping=None):
        if file is None and (positions is None or velocities is None or masses is None):
            raise TypeError('Need either a snapshot to load, or positions, velocities and masses.')
        if positions is not None:
            self.positions = torch.as_tensor(positions)
            self.velocities = torch.as_tensor(velocities)
            self.masses = torch.as_tensor(masses)
        if file is not None:
            snap = torch.tensor(np.loadtxt(file), dtype=float_dtype)
            self.positions = snap[:, 0:3]
            self.velocities = snap[:, 3:6]
            self.masses = snap[:, 6]
            if snap.shape[1] >= 8:
                particle_type = snap[:, 7].type(torch.uint8)
                if particle_type_mapping is not None:
                    mapped_particle_type = torch.zeros_like(particle_type)
                    for key, value in particle_type_mapping.items():
                        mapped_particle_type.masked_fill_(particle_type == key, value)
                    particle_type = mapped_particle_type
        if particle_type is None:
            particle_type = torch.full(self.masses.shape, ParticleType.Star, dtype=torch.uint8)
        self.__particletype = particle_type
        self.time = torch.as_tensor(time)
        self.omega = torch.as_tensor(omega)
        self.n = len(self.masses)
        self.dt = dt
        if self.dt is None:
            self.dt = torch.full(self.masses.shape, float('inf'), dtype=self.positions.dtype)
        self.starrange = None
        self.dmrange = None
        self.gasrange = None

    def to(self, device):
        """Moves the snapshot to the specified device. If already on the device returns self, otherwise returns a new
        SnapShot on the device."""
        if self.positions.device == device:
            return self
        else:
            newsnap = SnapShot(positions=self.positions.to(device),
                               velocities=self.velocities.to(device),
                               masses=self.masses.to(device),
                               particle_type=self.particletype.to(device),
                               dt=self.dt.to(device),
                               time=self.time.to(device), omega=self.omega.to(device))
            newsnap.dmrange = self.dmrange
            newsnap.starrange = self.starrange
            newsnap.gasrange = self.gasrange
            return newsnap

    def as_numpy_array(self):
        """Returns as a nmagic type matricx of dimension [:,7] with positions at [:,1:4], velocities at [:,4:7] and
        masses at [:,7]"""
        martrix = torch.cat((self.positions, self.velocities, self.masses.unsqueeze(dim=1)), dim=1)
        return martrix.to(torch.device('cpu')).numpy()


    def __organise(self):
        """We sort the particles by type. This improves speed because then we can just take a slice of the snapshot
        when we want stars/gas/dm."""
        ind = torch.argsort(self.__particletype, descending=True)
        self.positions = self.positions[ind, :]
        self.velocities = self.velocities[ind, :]
        self.masses = self.masses[ind]
        self.__particletype = self.__particletype[ind]
        counts = torch.bincount(self.__particletype, minlength=int(max(ParticleType)))
        self.starrange = (0, counts[ParticleType.Star])
        self.gasrange = (self.starrange[1], self.starrange[1] + counts[ParticleType.Gas])
        self.dmrange = (self.gasrange[1], self.gasrange[1] + counts[ParticleType.DarkMatter])

    @property
    def particletype(self):
        return self.__particletype

    @particletype.setter
    def particletype(self, particletype):
        self.__particletype = particletype
        self.__organise()

    @property
    def dm(self):
        if self.dmrange is None:
            self.__organise()
        return self[self.dmrange[0]:self.dmrange[1]]

    @property
    def gas(self):
        if self.gasrange is None:
            self.__organise()
        return self[self.gasrange[0]:self.gasrange[1]]

    @property
    def stars(self):
        if self.starrange is None:
            self.__organise()
        return self[self.starrange[0]:self.starrange[1]]

    @property
    def x(self):
        return self.positions[:, 0]

    @property
    def y(self):
        return self.positions[:, 1]

    @property
    def z(self):
        return self.positions[:, 2]

    @property
    def vx(self):
        return self.velocities[:, 0]

    @property
    def vy(self):
        return self.velocities[:, 1]

    @property
    def vz(self):
        return self.velocities[:, 2]

    @property
    def r(self):
        return self.positions.norm(dim=-1)

    @property
    def vr(self):
        return torch.tensordot(self.positions, self.velocities) / self.r

    @property
    def rcyl(self):
        return (self.positions[:, 0] ** 2 + self.positions[:, 1] ** 2).sqrt()

    def __getitem__(self, i):
        """Returns a new snapshot but should avoid unnessesary copies: if index
        is a slice that doesnt need a copy we will get views. Othewise we get a
        copy."""
        newsnap = SnapShot(positions=self.positions[i, :],
                           velocities=self.velocities[i, :],
                           masses=self.masses[i],
                           particle_type=self.particletype[i],
                           time=self.time, omega=self.omega)
        if self.dt is not None:
            newsnap.dt = self.dt[i]
        return newsnap

    def integrate(self, time, potentials, minsteps=1, maxsteps=2 ** 20, stepsperorbit=800, verbose=False):
        """"
        Integrate the snapshot until time. Use a minimum timestep of mindt (default 1e-6) and aim for
        stepsperorbit (default 800) steps per orbit.

        time may be either a scalar, in which case we integrate from self.time
        to time, or a tensor of times for each particle, in which case we integrate particle i from 0 to time[i].

        Strategy is to estimate for each particle the required number of steps rounded up to the nearest 2**N and
        integrate each group of particles with same number of timesteps together. If we find a particle that had
        an acceleration such that we arent getting enough steps per orbit we double the number of steps and retry.
        The timestep for each particle is remembered between calls.

        Note that we integrate in the inertial frame, but store the particles back to the corotating frame at the end.
        """

        time = torch.as_tensor(time)
        if time.nelement() == 1:
            blockstep = True
            initial_time = self.time
        else:
            assert (len(time) == self.n)
            blockstep = False
            initial_time = 0

        if blockstep and time == self.time:
            return

        if self.dt is None:
            if blockstep:
                self.dt = torch.full_like(self.masses, time - initial_time)
            else:
                self.dt = time.clone()

        # Given each particles dt then compute the optimal number of steps rounded up to the nearest 2**N
        steps = 2 ** (((time - initial_time) / self.dt).abs().log2().ceil()).clamp(min=0).type(torch.int32)
        # maxsteps = 2 ** (((time - initial_time) / mindt).abs().log2().ceil()).clamp(min=0).type(torch.int32)
        steps = steps.clamp(minsteps, maxsteps)  # dont exceed mindt stepsize, and make at least 1 step

        if not blockstep:
            steps.masked_fill_(time == initial_time, 0)

        thissteps = steps.min().clamp(1, None)
        while thissteps <= steps.max():
            i = (thissteps == steps)
            n_particles = i.sum()
            if n_particles > 0:
                positions = self.positions[i, :]
                velocities = self.velocities[i, :]
                dt = self.dt[i]
                if blockstep:
                    thisdt = (time - initial_time).view(1) / thissteps
                else:
                    thisdt = (time[i] - initial_time) / thissteps

                if verbose:
                    print(f'Steps: {thissteps} (of max {maxsteps}) with {i.sum()} particles')

                positions += velocities * thisdt[:, None] * 0.5
                timenow = initial_time + thisdt * 0.5
                for step in range(thissteps):
                    # Get accelerations in from corotating frame
                    accelerations = self.get_accelerations(potentials, self.corotating_frame(self.omega, timenow -
                                                                                             initial_time, positions))
                    self.corotating_frame(-self.omega, timenow - initial_time, accelerations,
                                          inplace=True)  # Move accelerations to inertial frame
                    velocities += accelerations * thisdt[:, None]
                    tau = 2 * math.pi * (
                            (positions.norm(dim=-1) + 1e-3) / (
                            accelerations.norm(dim=-1) + 1e-5)).sqrt() / stepsperorbit
                    dt = torch.min(dt, tau)
                    positions += velocities * thisdt[:, None]
                    timenow += thisdt
                positions -= velocities * thisdt[:, None] * 0.5
                if thissteps < maxsteps:
                    steps[i.nonzero()[:, 0][dt < thisdt]] *= 2
                    gd = i.nonzero()[:, 0][dt >= thisdt]
                    if len(gd) > 0:
                        self.positions[gd, :] = positions[dt >= thisdt]
                        self.velocities[gd, :] = velocities[dt >= thisdt]
                        self.dt[gd] = dt[dt >= thisdt]
                else:
                    self.positions[i, :] = positions
                    self.velocities[i, :] = velocities

            thissteps *= 2

        self.corotating_frame(self.omega, time - initial_time, self.positions, self.velocities, inplace=True)

        if blockstep:
            self.time = time

    @classmethod
    def get_accelerations(cls, potentials, positions):
        """Helper function to combine accelerations from list of potentials"""
        accelerations = potentials[0].get_accelerations(positions)
        for potential in potentials[1:]:
            accelerations.add_(potential.get_accelerations(positions))
        return accelerations

    @classmethod
    def ingrid(cls, potentials, positions):
        """Helper function to combine accelerations from list of potentials"""
        ingrid = torch.full_like(positions[..., 0], True, dtype=torch.bool)
        for potential in potentials:
            try:
                ingrid = ingrid & potential.ingrid(positions)
            except AttributeError:
                # Potential doesn't have an ingrid method: all assumed to be in-grid
                pass
        return ingrid

    def leapfrog_steps(self, potentials, steps, stepsperorbit=800, verbose=False, return_time=False):
        """"
        Integrate the snapshot for a set number of timesteps. Because the timestep is so there are stepsperorbit steps
        for each orbit (roughly) then this corresponds to integrating for roughly steps/stepsperorbit orbits.

        There are two reasons for prefering this over integrating for a length of time:
            - When fitting dynamical models we want all regions to be in equilibrium, regardless of orbital time, so
              we are limited by the longest timescale outer regions, and spend most of our time integrating the short
              timescale inner regions.
            - This is more efficient since we can be completely vectorised, just making a different timestep for each
              particle.

        Note that we integrate in the inertial frame, but store the particles back to the corotating frame at the end.
        """

        baddt = (self.dt == float("Inf"))
        if baddt.sum() > 0:
            print('Estimating dt')
            accelerations = self.get_accelerations(potentials, self.positions[baddt, :])
            self.dt[baddt] = 2 * math.pi * (
                    (self.positions[baddt, :].norm(dim=-1) + 1e-3) / (
                    accelerations.norm(dim=-1) + 1e-5)).sqrt() / stepsperorbit

        self.positions += self.velocities * self.dt[:, None] * 0.5
        timenow = self.dt * 0.5
        bad = (self.dt < 0)
        for step in range(steps):
            # Get accelerations in from corotating frame
            accelerations = self.get_accelerations(potentials,
                                                   self.corotating_frame(self.omega, timenow, self.positions))
            self.corotating_frame(-self.omega, timenow, accelerations,
                                  inplace=True)  # Move accelerations to inertial frame
            self.velocities += accelerations * self.dt[:, None]
            self.positions += self.velocities * self.dt[:, None]
            timenow += self.dt

        if verbose:
            print(f'Bad: {idx_bad.sum()} Total Bad seen: {bad.sum()}')
        self.positions -= self.velocities * self.dt[:, None] * 0.5
        self.corotating_frame(self.omega, timenow, self.positions, self.velocities, inplace=True)
        if return_time:
            return timenow

    @classmethod
    def rotate_snap(cls, angle, positions, velocities=None, inplace=False, deg=False):
        """Rotates the bar by angle about the z-axis.
        Returns positions, velocities in the corotating frame. Optionally updates the positions and velocities
        inplace"""
        if inplace:
            corotating_positions = positions
        else:
            corotating_positions = torch.zeros_like(positions)
            corotating_positions[..., 2] = positions[..., 2]

        # Idea is to make the 2x2 rotation matrix R and apply to the x-y plane for positions/velocities
        angle = torch.as_tensor(angle, device=positions.device, dtype=positions.dtype)
        if deg:
            angle *= math.pi / 180.
        cp, sp = torch.cos(angle), torch.sin(angle)
        corotating_positions[..., 0:2] = torch.stack((cp * positions[..., 0] - sp * positions[..., 1],
                                                      sp * positions[..., 0] + cp * positions[..., 1]), dim=-1)
        if velocities is None:
            return corotating_positions
        else:
            if inplace:
                corotating_velocities = velocities
            else:
                corotating_velocities = torch.zeros_like(velocities)
                corotating_velocities[..., 2] = velocities[..., 2]
            corotating_velocities[..., 0:2] = torch.stack((cp * velocities[..., 0] - sp * velocities[..., 1],
                                                           sp * velocities[..., 0] + cp * velocities[..., 1]), dim=-1)

            return corotating_positions, corotating_velocities

    @classmethod
    def corotating_frame(cls, omega, time, positions, velocities=None, inplace=False):
        """Uses the time to rotate positions (and optionally velocities) from the rotating to the corotating frame.
        Returns positions, velocities in the corotating frame. Optionally updates the positions and velocities inplace"""
        return cls.rotate_snap(omega * time, positions, velocities=velocities, inplace=inplace)

    def resample(self, potentials, verbose=False, velocity_perturbation=0.01):
        """Resamples the model, splitting the particles randomly proportional to mass. For particles with more than
        one copy then then the daughter particles are sampled along the orbit and given a small velocity perturbation
        of fraction velocity_perturbation (default 0.01) in the rotating frame."""

        gd = self.ingrid(potentials, self.positions).nonzero()[:, 0]
        i = torch.multinomial(self.masses[gd], self.n, replacement=True).sort().values

        # copy the resampled positions - the first copy of each particle is the parent particle and will
        # remain unchanged
        self.positions.copy_(self.positions[gd[i], :])
        self.velocities.copy_(self.velocities[gd[i], :])
        self.masses.fill_(self.masses[gd].sum() / self.n)
        # self.dt.fill_(1.0)
        self.dt.copy_(self.dt[gd[i]])

        accelerations = self.get_accelerations(potentials, self.positions)
        tau = 2 * math.pi * ((self.positions.norm(dim=-1) + 1e-3) / (accelerations.norm(dim=-1) + 1e-5)).sqrt()
        # compute the sample time for each child particle
        sample_time = torch.zeros_like(self.masses)
        uniq_i, inverse_indices, counts = torch.unique_consecutive(i, return_counts=True, return_inverse=True)
        for n_children in range(2, counts.max() + 1):
            number_of_n_children = (counts == n_children).sum()
            if number_of_n_children > 0:
                if verbose:
                    print('n_children', n_children, 'number_of_n_children:', number_of_n_children)
                sample_time[(counts[inverse_indices] == n_children).nonzero()[:, 0]] = torch.arange(n_children,
                                                                                                    dtype=sample_time.dtype,
                                                                                                    device=sample_time.device).repeat(
                    number_of_n_children) / n_children
        sample_time *= tau
        self.integrate(sample_time, potentials, minsteps=1024, maxsteps=8192, verbose=verbose)

        # perturb the velocities of the children in the rotating frame
        children = (sample_time > 0)
        self.velocities[children, 0] += velocity_perturbation * torch.randn_like(self.velocities[children, 0]) * (
                self.velocities[children, 0] - self.omega * self.positions[children, 1])
        self.velocities[children, 1] += velocity_perturbation * torch.randn_like(self.velocities[children, 1]) * (
                self.velocities[children, 1] + self.omega * self.positions[children, 0])
        self.velocities[children, 2] += velocity_perturbation * torch.randn_like(self.velocities[children, 2]) * \
                                        self.velocities[children, 2]
