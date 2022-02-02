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

### HW 2

def _trapz(y, x):
    # Integrate numerically using the trapezoid rule
    area = 0
    for i in range(len(x) - 1):
        dx = x[i+1] - x[i]
        area += (y[i] + y[i+1]) / 2 * dx 
    return area

def NIntegrate(func, a, b, dx, integrator = _trapz, **kwargs):
    '''
        numerically integrate 'func' from x = a to x = b with step size dx 
    '''
    n = int((b - a) * dx)
    x = 10 ** np.linspace(a, b, n)
    y = func(x, **kwargs)
    return _trapz(y, x)