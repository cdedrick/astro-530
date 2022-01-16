import numpy as np

from astropy import units as u
from astropy import constants as c
from astropy.modeling import models


### HW 1

def Planck(wn, T):
    '''
    Inputs:
        wn - wavenumbers in microns
        T - temperature in Kelvin
    Outputs:
        values of the Planck function in erg / s / cm^2 / sr / Hz^-1
    '''
    wn = wn * (1/u.micron)
    nu = wn.to(u.Hz, equivalencies=u.spectral())
    T *= u.K
    B_nu = 2 * c.h * nu ** 3 / c.c ** 2 * (np.exp(c.h * c.c * wn / (c.k_B * T)) - 1) ** -1
    B_nu = B_nu.to(u.erg / u.s / u.cm ** 2 / u.Hz)
    return B_nu.value
