"""Contains routines for computing the opacity-limited minimum Jeans mass and
hydrostatic core mass
"""

from constants import *
import numpy as np


def hydrogen_massfrac(Z):
    """Returns the metallicity-depdendent hydrogen mass fraction"""
    return 0.8 - 5 * Z * 0.014


def mean_molecular_weight(Z):
    """Returns the metallicity-dependent mean molecular weight of fully molecular gas"""
    return 32 / (14.4 - 47 * Z * 0.014)


def minimum_jeans_mass(beta=2, kappadust10=0.0232, blackbody_temp=2.73, Zd=1.0):
    """Returns the minimum Jeans mass for a given set of dust properties and
    radiation field, taking the greater of the radiation-heated and PdV work-heated
    values, following arXiv:2308.16268

    Parameters
    ----------
    beta: float or array_like, optional
        Dust spectral index
    kappadust10: float or array_like, optional
        Planck-mean dust opacity in cm^2/g at 10K, for the chosen dust model.
    Trad: float or array_like, optional
        Black-body temperature corresponding to the ambient radiation energy density,
        i.e. (urad / a)^(1/4)


    Returns
    -------
    mjmin: float or array_like
        Minimum Jeans mass in solar
    """
    mjmin_ff = minimum_jeans_mass_ff(beta, kappadust10, Zd)
    mjmin_rad = minimum_jeans_mass_rad(beta, kappadust10, blackbody_temp, Zd)

    return np.max([mjmin_ff, mjmin_rad], axis=0)


def minimum_jeans_mass_ff(beta=2.0, kappadust10=0.0232, Zd=1.0):
    """
    Returns the minimum Jeans mass assuming heating by PdV work only (Eq 19 in
    arXiv:2308.16268)

    Parameters
    ----------
    beta: float or array_like, optional
        Dust spectral index
    kappadust10: float or array_like, optional
        Planck-mean dust opacity in cm^2/g at 10K, for the chosen dust model.
    Zd: float or array_like, optional
        Dust abundance normalized to Solar neighborhood dust-to-gas ratio


    Returns
    -------
    mjmin: float or array_like
        Minimum Jeans mass in solar
    """

    Adust = Zd * kappadust10 * PROTONMASS_CGS / hydrogen_massfrac(Zd) * 10**-beta
    power_denom = 1 / (4 * beta + 7)
    mjmin = (
        (BOLTZMANN_CGS / mean_molecular_weight(Zd)) ** (power_denom * (9 * beta + 16))
        * (CGRAV / (RADCONSTANT_CGS * C_LIGHT_CGS)) ** (power_denom * (2 * beta + 4))
        * (
            np.pi ** (12 * beta + 21)
            * FJ ** (2 * beta + 3)
            / 2 ** (2 * beta + 3)
            / 3 ** (4 * beta + 7)
            / Adust
            / hydrogen_massfrac(Zd)
        )
        ** power_denom
        / GRAVITY_G_CGS ** (power_denom * (6 * beta + 10))
        / PROTONMASS_CGS ** (power_denom * (9 * beta + 15))
    )
    return mjmin / SOLAR_MASS_CGS


def minimum_jeans_mass_rad(beta=2.0, kappadust10=0.0232, blackbody_temp=2.73, Zd=1.0):
    """
    Returns the minimum Jeans mass assuming heating by radiation only (Eq 21 in
    arXiv:2308.16268)

    Parameters
    ----------
    beta: float or array_like, optional
        Dust spectral index
    kappadust10: float or array_like, optional
        Planck-mean dust opacity in cm^2/g at 10K, for the chosen dust model.
    Trad: float or array_like, optional
        Black-body temperature corresponding to the ambient radiation energy density,
        i.e. (urad / a)^(1/4)
    Zd: float or array_like, optional
        Dust abundance normalized to Solar neighborhood dust-to-gas ratio


    Returns
    -------
    mjmin: float or array_like
        Minimum Jeans mass in solar
    """

    Adust = Zd * kappadust10 * PROTONMASS_CGS / hydrogen_massfrac(Zd) * 10**-beta
    mjmin = (
        (np.pi**18 * CGRAV**2 * FJ**4 * hydrogen_massfrac(Zd) ** 2 / 11664) ** (1.0 / 6)
        * Adust ** (2 / 6)
        * BOLTZMANN_CGS ** (13 / 6)
        * blackbody_temp ** ((2 * beta + 5) / 6)
        / RADCONSTANT_CGS ** (2 / 6)
        / C_LIGHT_CGS ** (2 / 6)
        / GRAVITY_G_CGS ** (10 / 6)
        / PROTONMASS_CGS ** (15 / 6)
        / mean_molecular_weight(Zd) ** (13 / 6)
    )
    return mjmin / SOLAR_MASS_CGS


def larson_mass(beta=2, kappadust10=0.0232, blackbody_temp=2.73, Zd=1.0):
    """Returns the Larson hydrostatic core mass for a given set of dust properties and
    radiation field, taking the greater of the radiation-heated and PdV work-heated
    values, following arXiv:2308.16268

    Parameters
    ----------
    beta: float or array_like, optional
        Dust spectral index
    kappadust10: float or array_like, optional
        Planck-mean dust opacity in cm^2/g at 10K, for the chosen dust model.
    Trad: float or array_like, optional
        Black-body temperature corresponding to the ambient radiation energy density,
        i.e. (urad / a)^(1/4)
    Zd: float or array_like, optional
        Dust abundance normalized to Solar neighborhood dust-to-gas ratio

    Returns
    -------
    mlarson: float or array_like
        Larson mass in solar
    """
    mlarson_ff = larson_mass_ff(beta, kappadust10, Zd)
    mlarson_rad = larson_mass_rad(beta, kappadust10, blackbody_temp, Zd)

    return np.max([mlarson_ff, mlarson_rad], axis=0)


def larson_mass_ff(beta=2.0, kappadust10=0.0232, Zd=1.0):
    """
    Returns the Larson hydrostatic core mass assuming heating by PdV work only

    Parameters
    ----------
    beta: float or array_like, optional
        Dust spectral index
    kappadust10: float or array_like, optional
        Planck-mean dust opacity in cm^2/g at 10K, for the chosen dust model.
    Zd: float or array_like, optional
        Dust abundance normalized to Solar neighborhood dust-to-gas ratio


    Returns
    -------
    mlarson: float or array_like
        Larson mass in solar
    """

    Adust = Zd * kappadust10 * PROTONMASS_CGS / hydrogen_massfrac(Zd) * 10**-beta
    mlarson = (
        13.436053228129921
        * 2
        ** (
            (7 + 2 * beta - 2 * (3 + beta) * GAMMA1_H2)
            / ((7 + 4 * beta) * (-1 + GAMMA1_H2))
        )
        * FJ
        ** (
            (-7 + 2 * beta * (-1 + GAMMA1_H2) + 6 * GAMMA1_H2)
            / ((7 + 4 * beta) * (-1 + GAMMA1_H2))
        )
        * GRAVITY_G_CGS
        ** (
            (14 - 6 * beta * (-1 + GAMMA1_H2) - 13 * GAMMA1_H2)
            / ((7 + 4 * beta) * (-1 + GAMMA1_H2))
        )
        * T_DISS_H2 ** ((4 - 3 * GAMMA2_H2) / (2 - 2 * GAMMA2_H2))
        * T_EX_H2
        ** (
            (GAMMA1_H2 - GAMMA2_H2)
            / (2 - 2 * GAMMA1_H2 - 2 * GAMMA2_H2 + 2 * GAMMA1_H2 * GAMMA2_H2)
        )
        * (Adust * hydrogen_massfrac(Zd))
        ** ((-7 + 5 * GAMMA1_H2) / ((7 + 4 * beta) * (-1 + GAMMA1_H2)))
    ) / (
        ((RADCONSTANT_CGS * C_LIGHT_CGS) / CGRAV)
        ** (
            (2 * beta * (-1 + GAMMA1_H2) + GAMMA1_H2)
            / ((7 + 4 * beta) * (-1 + GAMMA1_H2))
        )
        * PROTONMASS_CGS
        ** (
            (3 * (-14 + 6 * beta * (-1 + GAMMA1_H2) + 13 * GAMMA1_H2))
            / (2.0 * (7 + 4 * beta) * (-1 + GAMMA1_H2))
        )
        * (mean_molecular_weight(Zd) / BOLTZMANN_CGS)
        ** (
            (-28 + 18 * beta * (-1 + GAMMA1_H2) + 29 * GAMMA1_H2)
            / (2.0 * (7 + 4 * beta) * (-1 + GAMMA1_H2))
        )
    )

    return mlarson / SOLAR_MASS_CGS


def larson_mass_rad(beta=2.0, kappadust10=0.0232, blackbody_temp=2.73, Zd=1.0):
    """
    Returns the Larson hydrostatic core mass assuming heating by radiation

    Parameters
    ----------
    beta: float or array_like, optional
        Dust spectral index
    kappadust10: float or array_like, optional
        Planck-mean dust opacity in cm^2/g at 10K and solar metallicity, for the chosen dust model.
    Trad: float or array_like, optional
        Black-body temperature corresponding to the ambient radiation energy density,
        i.e. (urad / a)^(1/4)
    Zd: float or array_like, optional
        Dust abundance normalized to Solar neighborhood dust-to-gas ratio

    Returns
    -------
    mlarson: float or array_like
        Larson mass in solar
    """

    Adust = Zd * kappadust10 * PROTONMASS_CGS / hydrogen_massfrac(Zd) * 10**-beta
    mlarson = (
        8.464183144814426
        * FJ
        * blackbody_temp ** ((-4 + 2 * beta + 3 / (-1 + GAMMA1_H2)) / 6.0)
        * T_DISS_H2 ** ((4 - 3 * GAMMA2_H2) / (2 - 2 * GAMMA2_H2))
        * T_EX_H2
        ** (
            (GAMMA1_H2 - GAMMA2_H2)
            / (2 - 2 * GAMMA1_H2 - 2 * GAMMA2_H2 + 2 * GAMMA1_H2 * GAMMA2_H2)
        )
        * (Adust * CGRAV * hydrogen_massfrac(Zd)) ** 0.3333333333333333
    ) / (
        (RADCONSTANT_CGS * C_LIGHT_CGS * FJ) ** 0.3333333333333333
        * GRAVITY_G_CGS**1.6666666666666667
        * PROTONMASS_CGS**2.5
        * (mean_molecular_weight(Zd) / BOLTZMANN_CGS) ** 2.1666666666666665
    )

    return mlarson / SOLAR_MASS_CGS
