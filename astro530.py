import numpy as np

from astropy import units as u
from astropy import constants as c
from astropy.modeling import models
from scipy.interpolate import interp1d


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
    dx = np.diff(x)
    area = (y[:-1] + y[1:]) / 2 * dx
    return np.sum(area)

# def NIntegrate(func, a, b, density, unit = None, integrator = _trapz, **kwargs):
#     '''
#         func - function to numerically integrate
#         a - lower bound
#         b - upper bound
#         density - number of subintervals per unit 
#     '''
#     n = round((b - a) * density)
#     x = np.linspace(a, b, n)
#     if unit != None:
#         x *= unit
#     y = func(x, **kwargs)
#     return _trapz(y, x)

### HW 4

def NIntegrate(func, a, b, density, log = True, unit = None, integrator = _trapz, **kwargs):
    '''
        Now improved to have log-spaced bins!
        func - function to numerically integrate
        a - lower bound
        b - upper bound
        density - number of subintervals per unit 
    '''
    n = round((b - a) * density)
    
    if log:
        x = np.logspace(a, b, n)
        if unit != None:
            x *= unit
        y = x * func(x, **kwargs)
        x = np.log(x)
    else:
        x = np.linspace(a, b, n)
        if unit != None:
            x *= unit
        y = func(x, **kwargs)
    
    return _trapz(y, x)

### HW 6

def _getPartition(species, table):
    T = 5040 / np.linspace(0.2, 2.0, 10)
    log_u = table.loc[species][0:10].to_numpy()
    
    # remove nans so scipy doesn't get mad
    good = np.where(~np.isnan(log_u))[0]
    T = T[good]
    log_u = log_u[good]
    
    # interpolate data points
    f = interp1d(T, log_u, fill_value = 'extrapolate')
    return f
    
def partition(species, T, table):
    if species == 'H-' or species == 'H+':
        u = np.ones_like(T)
    else:
        logU = _getPartition(species, table)
        u = 10 ** logU(T)
    return u 