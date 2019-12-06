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
    def number_integral(self, mass: Union[float, np.array]) -> np.array:
        """The number of stars per unit mass born with masses smaller than mass"""
        pass

    @abstractmethod
    def mass_integral(self, mass: Union[float, np.array]) -> np.array:
        """The mass fraction of stars born with masses smaller than mass"""
        pass


class PowerLawIMF(IMF):
    def __init__(self, mass_breaks: Sequence[float] = (0.08, 0.5),
                 power_law_indicies: Sequence[float] = (-0.3, -1.3, -2.3)):
        """A power law IMF. Has power law slopes given by power_law_indicies where the number of stars from m to m+dm is
        m ** power_law_indicies dm.

        The default corresponds to a Kroupa (2001) IMF."""
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
        self.normalisations /= self.mass_integral(np.inf)

    def number(self, mass: Union[float, np.array]) -> np.array:
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

    def _integral(self, mass: Union[float, np.array], mass_power=0) -> np.array:
        """Helper function that returns \int phi(m)*m**mass_power dm so that the same code can be used
        with mass_power = 0 for number integral  and mass_power = 1 for mass integral"""
        mass = np.asarray(mass)
        integral = np.zeros_like(mass)
        for break_index, (normalisation, power_law_index) in enumerate(
                zip(self.normalisations, self.power_law_indicies)):
            m_min = (self.mass_breaks[break_index - 1] if break_index != 0 else 0.)
            m_max = (self.mass_breaks[break_index] if break_index < len(self.mass_breaks) else np.inf)
            out_indx = (mass >= m_max)
            exponent = power_law_index + mass_power + 1
            if exponent != 0.:
                integral[out_indx] += normalisation * (m_max ** exponent - m_min ** exponent) / exponent
            else:
                integral[out_indx] += normalisation * (np.log(m_max) - np.log(m_min))

            out_indx = (mass >= m_min) & (mass < m_max)
            if exponent != 0.:
                integral[out_indx] += normalisation * (mass[out_indx] ** exponent - m_min ** exponent) / exponent
            else:
                integral[out_indx] += normalisation * (np.log(mass[out_indx]) - np.log(m_min))
        return integral

    def number_integral(self, mass: Union[float, np.array]) -> np.array:
        """Returns the number of stars per unit mass born with masses smaller than mass"""
        return self._integral(mass, mass_power=0)

    def mass_integral(self, mass: Union[float, np.array]) -> np.array:
        """Returns the number of stars per unit mass born with masses smaller than mass"""
        return self._integral(mass, mass_power=1)
