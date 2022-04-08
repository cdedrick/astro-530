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
    T = 5040 / np.linspace(0.2, 2.0, 10) * u.K
    log_u = table.loc[species][0:10].to_numpy()
    
    # remove nans so scipy doesn't get mad
    good = np.where(~np.isnan(log_u))[0]
    T = T[good]
    log_u = log_u[good]
    
    # interpolate data points
    f = interp1d(T, log_u, fill_value = 'extrapolate')
    return f
    
def partition(species, T, table):
    if species == 'H-' or species == 'H+' or species == 'Li+':
        u = np.ones_like(T.value)
    else:
        logU = _getPartition(species, table)
        u = 10 ** logU(T)
    return u 

### HW 7

# Problem 14
def saha_phi(element, T, i_table, u_table):
    if element == 'H-':
        u0 = partition('H-', T, u_table)
        u1 = partition('H', T, u_table)
        I = 0.754 * u.eV
    else:
        u0 = partition(element, T, u_table)
        u1 = partition(element + '+', T, u_table)
        I = i_table.loc[element][3] * u.eV
        
    phi = (2 * np.pi * c.m_e)**(3/2) * (c.k_B * T)**(5/2) / c.h**3 * 2 * u1 / u0 * np.exp(-I/(c.k_B * T))
    return phi.to(u.barye)

# Problem 15

def chi(n, I):
    return I * (1 - 1/n**2)

def g_bf(l, n):
    limit = np.array([912, 3746, 8206, 14588]) * u.AA
    g = 1 - 0.3456 / (l * c.Ryd)**(1/3) * (l * c.Ryd / n**2 - 1/2)
    
    if n <= 4:
        ion_lim = np.where(l >= limit[n - 1])[0]
        if len(ion_lim) > 0:
            g[ion_lim] = 0
    return g

def g_ff(l, T):
    g = 1 + 0.3456 / (l * c.Ryd)**(1/3) * (l * c.k_B * T / (c.h * c.c) + 1/2)
    return g


def KH_bf(l, T, i_table):

    l = l.to(u.AA).value
    theta = 5040 * u.K / T 
    a0 = 1.0449e-26
    loge = 0.43429
    I = i_table.loc['H'][3]
    
    sum_terms = np.array([g_bf(u.Quantity([l]) * u.AA, n)[0]/n**3 * 10 ** (-chi(n, I) * theta) for n in range(1, 5)]) 

    K = a0 * l**3 * (np.sum(sum_terms, axis = 0) + loge/(2 * theta * I) * (10**(-chi(5, I) * theta) - 10**(-I * theta)))
    return K * u.cm **2

def KH_ff(l, T, i_table):

    l = l.to(u.AA).value
    theta = 5040 * u.K / T 
    a0 = 1.0449e-26
    loge = 0.43429
    I = i_table.loc['H'][3]
    
    K = a0 * l**3 * g_ff(l * u.AA, T) * loge/(2 * theta * I) * 10**(-theta * I)
    return K * u.cm **2

def KHminus_bf(l, T, Pe):

    l = l.to(u.AA).value
    Pe = Pe.cgs.value
    theta = 5040 * u.K / T 
    
    a = np.array([1.99654, -1.18267e-5, 2.64243e-6, -4.40524e-10, 3.23992e-14, -1.39568e-18, 2.78701e-23])
    a_bf = np.sum([a[n] * l ** n for n in range(7)], axis = 0) * 10 ** -18
    
    K = 4.158e-10 * a_bf * Pe * theta**(5/2) * 10**(0.754 * theta)
    ion_lim = np.where(l * u.AA >= c.h *c.c/ (0.754 * u.eV))[0]
    if len(ion_lim)> 0:
        K[ion_lim] = 0
    return K * u.cm **2

def KHminus_ff(l, T, Pe):

    l = l.to(u.AA).value
    Pe = Pe.cgs.value
    theta = 5040 * u.K / T 
    
    f0 = -2.2763 - 1.6850 * np.log10(l) + 0.76661 * np.log10(l)**2 - 0.053346 * np.log10(l)**3
    f1 = 15.2827 - 9.2846 * np.log10(l) + 1.99381 * np.log10(l)**2 - 0.142631 * np.log10(l)**3
    f2 = -197.789 + 190.266 * np.log10(l) - 67.9775 * np.log10(l)**2 + 10.6913 * np.log10(l)**3 - 0.625151 * np.log10(l)**4
    
    K = 1e-26 * Pe * 10**(f0 + f1 * np.log10(theta) + f2 * np.log10(theta)**2)
    return K * u.cm **2

def KH(l, T, Pe, i_table):
    return (KH_bf(l, T, i_table) + KH_ff(l, T, i_table) + KHminus_bf(l, T, Pe)) * (1 - np.exp(-c.h * c.c / (l * c.k_B * T))) + KHminus_ff(l, T, Pe) 

### HW 8

def eq9_8(Pe, Pg, A_j, Phi_j):
    num = np.sum(A_j * (Phi_j/Pe) / (1 + Phi_j/Pe))
    denom = np.sum(A_j * (1 + (Phi_j/Pe) / (1 + Phi_j/Pe)))
    return Pg * num/denom

def iterate_Pe(Pe, Pg, A_j, Phi_j, tol=1e-8):
    diff = np.inf
    while diff > tol:
        Pe_new = eq9_8(Pe, Pg, A_j, Phi_j)
        diff = np.abs(Pe - Pe_new).value
        Pe = Pe_new
    return Pe

def P_e(T, Pg, A_table, i_table, u_table, elements):
    '''
    Inputs:
        T - temperature
        log_Pg - gas pressure
    Output:
        electron pressure, as shown in Gray Eq. (9.8)
    '''
    
    A_j = A_table.loc[elements]['A'].to_numpy()
    Phi_j = u.Quantity([saha_phi(element, T, i_table, u_table) for element in elements])
    Phi_H = saha_phi('H', T, i_table, u_table)
    
    if T.value > 30000:
        Pe_guess = 0.5 * Pg 
    else:
        Pe_guess = np.sqrt(Pg * Phi_H)
        
    P_e = iterate_Pe(Pe_guess, Pg, A_j, Phi_j)
    
    return P_e