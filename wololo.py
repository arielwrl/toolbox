# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 17:57:21 2016

@author: ariel
"""

import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u


## Planck Collab 2015, Paper XIII
#H0 = 67.7 # km / s / Mpc
#omega0 = 0.307

#c = 299792.458 # km / s

#VW
H0 = 70 # km / s / Mpc
omega0 = 0.3
c = 299792.458 # km / s

cosmo = FlatLambdaCDM(H0, omega0)

# Other constants
h     = 6.62606885e-27
c_cgs = 2.99792458e10

def redshift2lumdistance(z, simple=False):
    if simple:
        return z * c / H0

    cosmo = FlatLambdaCDM(H0, omega0)
    dl = cosmo.luminosity_distance(z) / u.Mpc
    return dl.value


def countstoflux(l_eff, counts):
    '''
    by counts I mean counts/second
    '''

    f = counts * h * c_cgs / l_eff

    return f

def magstologY(mag, z):
    dl = redshift2lumdistance(z)
    pho_norm = 3.826e33 / (4 * np.pi * ((3.086e24*dl) ** 2))
    logY = mag + 2.5 * np.log10(pho_norm)

    return logY

def logYtoflux(logY, z, lamb):

    dl = redshift2lumdistance(z)
    pho_norm = 3.826e33 / (4 * np.pi * ((3.086e24*dl) ** 2))
    mag = logY - 2.5 * np.log10(pho_norm)

    f_ni   = 3631. * ( 10 ** ( -0.4 * mag ) )
    f_lamb = janskystoergs(f_ni, lamb)

    return f_lamb

def logYtomag(logY, z):

    dl = redshift2lumdistance(z)
    pho_norm = 3.826e33 / (4 * np.pi * ((3.086e24*dl) ** 2))
    mag = logY - 2.5 * np.log10(pho_norm)

    return mag

def logYtoflux_alt(logY, z, band):

    dl = redshift2lumdistance(z)
    pho_norm = 3.826e33 / (4 * np.pi * ((3.086e24*dl) ** 2))
    mag = logY - 2.5 * np.log10(pho_norm)

    if band == 'FUV':
        flux = 1.40e-15 * (10**((18.82 - mag)/2.5))

    if band == 'NUV':
        flux = 2.06e-16 * (10**((20.08 - mag)/2.5))

    return flux

def calc_logYerr(logY, z, band, logYerr):

    flux = logYtoflux_alt(logY, z, band)

    fluxerr = logYerr * np.log(10) * flux

    return fluxerr

def galexcountomags(band, counts):
    """

    Convert GALEX counts/second to AB magnitudes.

    """

    if band=='NUV':
        mab = -2.5 * np.log10(counts) + 20.08
    if band=='FUV':
        mab = -2.5 * np.log10(counts) + 18.82

    return mab


def galexcountoflux(band, counts):
    """

    Convert GALEX counts/second to flux in erg/s/cm-2/A.

    """

    if band=='NUV':
        flux = 2.06e-16 * counts
    if band=='FUV':
        flux = 1.40e-15 * counts

    return flux


def sdssmagstoflux(band, mag, flux_unit='flam'):
    """

    This will convert the SDSS luptitudes to a flux in erg s-1 cm-2 angstrons-1

    """

    if flux_unit=='flam':
        def get_flux(mag,b,lamb):

            a = (2.5) / np.log(10.)
            lamb = lamb / 1000
            arg = (mag/a) + (np.log(b))
            x = (-2) * b * np.sinh(arg)
            f = ( 1 / (3.34e10 * (lamb ** 2))) * 3631 * x
            return f


    if flux_unit=='maggie':
        def get_flux(mag,b,lamb):

            a = (2.5) / np.log(10.)
            arg = (mag/a) + np.log(b)
            f = (-2) * b * np.sinh(arg)
            return f


    #Defining lambdaeffs and softening parameters:
    lambdaeffs = np.array([3543.,  4770.,  6231.,  7625.,  9134.])
    bs  = [1.4e-10,0.9e-10,1.2e-10,1.8e-10,7.4e-10]

    if band=='u':

        #AB correction:
        mag = mag -0.04

        #Calculating flux:
        flux = get_flux(mag, bs[0], lambdaeffs[0])

    if band=='g':

        #Calculating flux:
        flux = get_flux(mag, bs[1], lambdaeffs[1])

    if band=='r':

        #Calculating flux:
        flux = get_flux(mag, bs[2], lambdaeffs[2])

    if band=='i':

        #Calculating flux:
        flux = get_flux(mag, bs[3], lambdaeffs[3])

    if band=='z':

        #AB correction:
        mag = mag + 0.02

        #Calculating flux:
        flux = get_flux(mag, bs[4], lambdaeffs[4])

    return flux


def absmag(inmag, z):
    """

    Calculate the absolute magnitude of an object of given redshift
    and aparent magnitude.

    """
    dm    = cosmo.distmod(z)

    absolute = inmag - dm.value

    return absolute



def janskystoergs(influx, lamb):
    """

    Converts janskys to erg s-1 cm-2 angstrons-1 given the flux in janskys and
    the wavelenght in angstrons.

    """

    ergflux  =  ( 1 / (3.34e4 * (lamb ** 2)) ) * influx

    return ergflux


def ergstojanskys(influx,lamb):

    janskys = 3.34e4 * (lamb**2) * influx

    return janskys


def ergstoabmags(influx,lamb):
    janskys = ergstojanskys(influx,lamb)
    abmag = -2.5 * np.log10(janskys/3631.)
    return abmag


def nanomaggiestoergs(nanomaggies, band):
    lambdaeffs = {'FUV':1542.26, 'NUV':2274.37,'u':3543., 'g':4770., 'r':6231.
    , 'i':7625., 'z':9134.}
    janskys = 3.631e-06 * nanomaggies
    return janskystoergs(janskys, lambdaeffs[band])


def nanomaggiestoabmags(nanomaggies):
    jankys = 3.631e-06 * nanomaggies
    abmag = -2.5 * np.log10(jankys / 3631.)
    return abmag


def nanomaggiestoabmagserror(nanomaggies, nanomaggieserror):
    return np.absolute(2.5*np.log10(np.e)*nanomaggieserror/nanomaggies)


def vacuumtoair(vac):
    """

    Turn vacuum wavelenghts to air wavelenghts.
    Reference: Morton (1991, ApJS, 77, 119)

    """
    air = vac/(1.0 + 2.735182e-4 + 131.4182 / vac ** 2 + 2.76249e8 / vac ** 4)
    return air


def radectoindex(RA,DEC,nside=2048,nest=False):
    """

    Converts RA and Dec to a healpix index, default nside and nest are set to
    work with helpix maps from Planck

    """

    theta = (90 - DEC) * (np.pi / 180)
    phi = RA * (np.pi / 180)

    return hp.ang2pix(nside,theta,phi,nest)


def abmagstoflux(mag, band):

    lambdaeffs = {'FUV':1542.26, 'NUV':2274.37,'u':3543., 'g':4770., 'r':6231.
    , 'i':7625., 'z':9134., 'y':10314.06, 'j':12501.18, 'h':16354.34, 'k':22058.36}

    f_ni   = 3631. * ( 10 ** ( -0.4 * mag ) )
    f_lamb = janskystoergs(f_ni, lambdaeffs[band])

    return f_lamb

def abmagstoflux_wlpiv(mag, wl_piv):

    f_ni   = 3631. * ( 10 ** ( -0.4 * mag ) )
    f_lamb = janskystoergs(f_ni, wl_piv)

    return f_lamb


def luminosity(mag, wl_piv, z):

    f_lamb = abmagstoflux_wlpiv(mag, wl_piv)
    L      = np.log10( f_lamb * 4 * np.pi * (3.086e24*redshift2lumdistance(z)) ** 2  /  3.826e33 )

    return L


def calc_beta(FUV, NUV):
    return 2.286*(FUV-NUV) - 2.096
