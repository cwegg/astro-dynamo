import re
from itertools import tee
from typing import Union, List, Tuple, Callable, Dict

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

import astro_dynamo.imfs


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class ParsecLuminosityFunction():
    def __init__(self, file: str = '../data/parsec_luminosity_function.dat',
                 isochrone_file: str = None,
                 imf: astro_dynamo.imfs.IMF = None,
                 mag_range: Union[Tuple[float, float], List[float]] = (-5, 5),
                 d_mag: float = 0.2,
                 remnant_function: Callable[[
                                                np.array], np.array] = astro_dynamo.imfs.remnant_function_maraston98,
                 normalise: str = 'InitialMass',
                 mag_expressions: Dict = None):
        """Load a luminosity function with data downloaded from http://stev.oapd.inaf.it/cgi-bin/cmd .
        Should have a range of metalicities and ages."""

        if isochrone_file is None:
            if (normalise != 'InitialMass') | (imf is None) | (
                    mag_expressions is None):
                raise ValueError(
                    'Cannot change the IMF or normalisation from Padova LF, you should use Isochrones instead')

            lf_df = self._load_luminosity_function_from_file(file)
        else:
            if (normalise != 'CurrentMass') & (normalise != 'InitialMass'):
                raise ValueError(
                    f'normalise should be CurrentMass or InitalMass (found {normalise})')
            lf_df = self._create_luminosity_function_from_isochrones(
                isochrone_file, imf, mag_range, d_mag,
                remnant_function, normalise, mag_expressions)

        self.zs = lf_df.Z.unique()
        self.ages = lf_df.index.get_level_values('age').unique()
        self.mags = lf_df.index.get_level_values('magbinc').unique()
        self.mhs = lf_df.index.get_level_values('MH').unique()
        assert len(self.mhs) * len(self.ages) * len(self.mags) == len(lf_df)

        self.interpolators = {}
        self.grids = {}
        for col in lf_df.columns:
            grid = np.zeros((len(self.ages), len(self.mhs), len(self.mags)))
            for i, age in enumerate(self.ages):
                for j, mh in enumerate(self.mhs):
                    grid[i, j, :] = lf_df.loc[(age, mh,)][col]

            self.interpolators[col] = RegularGridInterpolator(
                (self.ages, self.mhs), grid)
            self.grids[col] = grid
        self.lf_df = lf_df

    def _load_parsec_file_to_df(self, file):
        header, columns = [], None
        with open(file) as fp:
            line = fp.readline()
            while line and line[0] == '#':
                header += [line]
                columns = line.rstrip().strip("#").split()
                line = fp.readline()
        _ = header.pop(-1)
        df = pd.read_csv(file, sep=r'\s+', comment='#', names=columns)
        self.header = header
        return df

    def _create_luminosity_function_from_isochrones(self, file, imf, mag_range,
                                                    d_mag, remnant_function,
                                                    normalise, mag_expressions):
        iso_df = self._load_parsec_file_to_df(file)
        if mag_expressions is not None:
            for mag, expression in mag_expressions.items():
                new_expression, matches = re.subn(r'(\w+mag)', r'iso_df.\1',
                                                  expression)
                if matches == 0:
                    raise ValueError(
                        "The expression {expression} for {mag} doesnt seem to contain any magnitudes.")
                iso_df[mag] = eval(new_expression)
        # we first compute how many stars lie on each segment i.e. between each
        # point on the isochrone.
        # To do this we use the number_integral of the imf and the initial mass
        # at each point om the isochrones
        if imf is None:
            imf = astro_dynamo.imfs.PowerLawIMF()
        iso_df['my_imf_int'] = imf.number_integral(iso_df.Mini)
        iso_df['my_imf_mass_int'] = imf.mass_integral(iso_df.Mini)

        # Strategy is to loop over individual isochrone constructing the LF for each. So first organise the isochrones
        mhs = iso_df.MH.unique()
        ages = 10 ** iso_df.logAge.unique()
        iso_df.set_index(['logAge', 'MH', 'Mini'], inplace=True)
        iso_df.sort_index(inplace=True)
        self.iso_df = iso_df
        # Create the output DataFrame of all the luminosity functions
        n_mags = np.round((mag_range[1] - mag_range[0]) / d_mag + 1).astype(int)
        mags = mag_range[0] + d_mag * np.arange(n_mags)
        index = pd.MultiIndex.from_product([ages, mhs, mags],
                                           names=['age', 'MH', 'magbinc'])
        lf_df = pd.DataFrame(index=index)

        # The code below is slow... it loops over all isochrones and isochrone segments, but it's not easy to vectorize

        def check_index(
                i):  # Function to check our isochrone point is on the LF
            return (i >= 0) & (i < n_mags - 1)

        for (logage, mh), single_iso in iso_df.groupby(level=(0, 1)):
            age = 10 ** logage
            # First we compute the mass in stars and remnants on this isochrone

            # Set the stellar mass to be the total mass of stars born with masses less than the last isochrone point
            # This ignores the mass loss along the isochrone, but the error due to this approximation is negligible
            # compared to the uncertainties in IMF
            max_stellar_mass = single_iso.index.get_level_values('Mini').max()
            lf_df.loc[(age, mh,), 'max_stellar_mass'] = max_stellar_mass
            total_stellar_mass = single_iso.my_imf_mass_int.max()
            lf_df.loc[(age, mh,), 'total_stellar_mass'] = total_stellar_mass

            # Integrate the remnant masses in log space i.e. mass in remnants = int Mf(Mi)*N(Mi)*Mi*dlog Mi
            logMi = np.linspace(np.log(max_stellar_mass), np.log(100.),
                                100)  # Integrate to 100Msun
            dlogMi = logMi[1] - logMi[0]
            Mi = np.exp(logMi)
            Mf = remnant_function(Mi)
            total_remnant_mass = np.sum(Mf * imf.number(Mi) * Mi * dlogMi)
            lf_df.loc[(age, mh,), 'total_remnant_mass'] = total_remnant_mass
            total_current_mass = total_remnant_mass + total_stellar_mass
            lf_df.loc[(age, mh,), 'total_current_mass'] = total_current_mass

            number_in_segment = single_iso.my_imf_int.diff().iloc[1:]
            for column in single_iso.columns.values:
                # loop over all columns, just making LFs for those that end in 'mag'
                if column[-3:] == 'mag':
                    band = column
                    lf = np.zeros(n_mags)
                    # compute where on the LF each point should be, both as a float and index.
                    f_index = (single_iso[column] - mag_range[0]) / d_mag
                    index = f_index.round().astype(int)
                    # we step over every segement (between 2 isochrone points) and place the stars
                    for n, (f_i_1, f_i_2), (i_1, i_2) in zip(number_in_segment,
                                                             pairwise(f_index),
                                                             pairwise(index)):
                        if f_i_1 > f_i_2:  # easier to just handle one ordering of segments, so swap if required
                            f_i_1, f_i_2 = f_i_2, f_i_1
                            i_1, i_2 = i_2, i_1
                        if i_1 == i_2:  # place all stars in one LF bin
                            if check_index(i_1):
                                lf[i_2] += n
                        else:  # Spread stars over more than one LF bin
                            if i_2 - i_1 > 1:  # Complete LF bins
                                if check_index(i_1 + 1) & check_index(i_2):
                                    lf[i_1 + 1:i_2] += n / (f_i_2 - f_i_1)
                            if check_index(i_1):  # Left hand LF bin
                                lf[i_1] += n * (0.5 - (f_i_1 - i_1)) / (
                                        f_i_2 - f_i_1)
                            if check_index(i_2):  # Right hand LF bin
                                lf[i_2] += n * (0.5 + (f_i_2 - i_2)) / (
                                        f_i_2 - f_i_1)
                    if normalise == 'CurrentMass':
                        lf /= total_current_mass
                    lf_df.loc[(age, mh,), band] = lf

        # We need to use the same metalicity measure for both isochrones and the LF, but the isochrones only give [
        # M/H], while LF give Z. We use [M/H], but compute Z from this for compatible of the resultant DataFrames.
        # Computation is the inverse of that shown on the parsec isochrone webpage.
        m_h_solar = 0.0207
        mh_raw = m_h_solar * 10 ** lf_df.index.get_level_values('MH')
        lf_df['Z'] = mh_raw * (1 - 0.2485) / (1 + mh_raw * (1 + 1.78))
        return lf_df

    def _load_luminosity_function_from_file(self, file: str):
        lf_df = self._load_parsec_file_to_df(file)
        # We need to use the same metalicity measure for  both isochrones and the LF, but the isochrones only give [
        # M/H], while LF give Z. Therefore compute [M/H] from Z. Computation taken from the pasrsec isochrone webpage.
        y = 0.2485 + 1.78 * lf_df.Z  # compute helium fraction
        x = 1 - y - lf_df.Z  # compute hydrogen fraction
        m_h_solar = 0.0207
        lf_df['MH'] = np.log10(lf_df.Z / x) - np.log10(m_h_solar)
        lf_df.set_index(['age', 'MH', 'magbinc'], inplace=True)
        lf_df.sort_index(inplace=True)
        return lf_df

    def get_single_lf(self, band, age, mh, mags=None):
        """Get the luminosity function for the specified age and feh. Samples at the specified mags, otherwise use the
        native mags from the loaded luminosty funciton"""
        lf = self.interpolators[band]((age, mh))
        if mags is None:
            return {'mag': self.mags, 'number': lf}
        else:
            cumlf = np.cumsum(lf)
            cum_lf_interpolated = np.interp(mags, self.mags, cumlf)
            resampled_lf = np.diff(cum_lf_interpolated, prepend=0)
            resampled_lf[0] = resampled_lf[1]
            return {'mag': mags, 'number': resampled_lf}

    def get_lf_mh_func(self, band, age, feh_func, mags=None):
        """Get the luminsoty function for the specified age and function specifying the feh distribution."""
        weights = feh_func(self.mhs)
        weights /= np.sum(weights)
        lf = None
        for weight, mh in zip(weights, self.mhs):
            this_lf = self.get_single_lf(band, age, mh, mags=mags)
            if lf is not None:
                lf += weight * this_lf['number']
            else:
                lf = weight * this_lf['number']
        return {'mag': this_lf['mag'], 'number': lf}
