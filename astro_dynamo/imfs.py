from abc import ABC, abstractmethod
from typing import Union, Sequence

import numpy as np


class IMF(ABC):
    """All IMFs should derive from this class"""

    @abstractmethod
    def number(self, mass: Union[float, np.array]) -> np.array:
        """The number of stars per unit mass born at mass"""
        pass

    @abstractmethod
    def integral(self, mass: Union[float, np.array]) -> np.array:
        """The number of stars per unit mass born with masses smaller than mass"""
        pass


class PowerLawIMF(IMF):
    def __init__(self, mass_breaks: Sequence[float] = (0.08, 0.5),
                 power_law_indicies: Sequence[float] = (-0.3, -1.3, -2.3)):
        """A power law IMF. The default corresponds to a Kroupa (2001) IMF."""
        if len(mass_breaks) + 1 != len(power_law_indicies):
            raise ValueError("power_law_indicies should be 1 element longer than breaks")

        self.mass_breaks = mass_breaks
        self.power_law_indicies = power_law_indicies

        # first compute the normalisations we need to ensure continuity at each break
        normalisations = [1.0]
        for i, mass_break in enumerate(self.mass_breaks):
            normalisations += [normalisations[i] * mass_break ** (power_law_indicies[i] - power_law_indicies[i + 1])]
        self.normalisations = np.asarray(normalisations)

        # then ensure that we the normalisation means that we have 1Msun in total
        self.normalisations /= self.integral(np.inf)

    def number(self, mass: Union[float, np.array]):
        """The number of stars per unit mass born at mass"""
        mass = np.asarray(mass)
        out = np.zeros_like(mass)
        for break_index, (normalisation, power_law_index) in enumerate(
                zip(self.normalisations, self.power_law_indicies)):
            m_min = (self.mass_breaks[break_index - 1] if break_index != 0 else 0.)
            m_max = (self.mass_breaks[break_index] if break_index < len(self.mass_breaks) else np.inf)
            out_indx = (mass >= m_min) & (mass < m_max)
            out[out_indx] = normalisation * mass[out_indx] ** power_law_index
        return out

    def integral(self, mass: Union[float, np.array]):
        """The number of stars per unit mass born with masses smaller than mass"""
        mass = np.asarray(mass)
        integral = np.zeros_like(mass)
        for break_index, (normalisation, power_law_index) in enumerate(
                zip(self.normalisations, self.power_law_indicies)):
            m_min = (self.mass_breaks[break_index - 1] if break_index != 0 else 0.)
            m_max = (self.mass_breaks[break_index] if break_index < len(self.mass_breaks) else np.inf)
            out_indx = (mass >= m_max)
            if power_law_index != -1.0:
                integral[out_indx] += normalisation*(m_max ** (power_law_index + 1) - m_min ** (power_law_index + 1)) / (
                            power_law_index + 1)
            else:
                integral[out_indx] += normalisation*(np.log(m_max) - np.log(m_min))
            out_indx = (mass >= m_min) & (mass < m_max)
            if power_law_index != -1.0:
                integral[out_indx] += normalisation*(mass[out_indx] ** (power_law_index + 1) - m_min ** (power_law_index + 1)) / (
                            power_law_index + 1)
            else:
                integral[out_indx] += normalisation*(np.log(mass[out_indx]) - np.log(m_min))
        return integral
