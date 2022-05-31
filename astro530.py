import numpy as np
import pandas as pd

from astropy import units as u
from astropy import constants as c
from astropy.modeling import models
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from scipy.special import wofz

e = c.e.esu

### HW 1

A_table = pd.read_pickle('./data/abundances.pkl')
i_table = pd.read_pickle('./data/ionization.pkl')
u_table = pd.read_pickle('./data/partition.pkl')

VALIIIC = pd.read_pickle('./data/VALIIIC_atm.pkl')

tau_500 = VALIIIC['tau_500'].to_numpy()

h = VALIIIC['h'].to_numpy() * u.km
s = h[0] - h

T = VALIIIC['T'].to_numpy() * u.K

micro = VALIIIC['V'].to_numpy() * u.km/u.s

nH = VALIIIC['n_H'].to_numpy() * u.cm**-3

ne = VALIIIC['n_e'].to_numpy() * u.cm**-3
Pe = ne * c.k_B * T

Pg = VALIIIC['Pgas/Ptotal'].to_numpy() * VALIIIC['Ptotal'].to_numpy() * u.barye

rho = VALIIIC['rho'].to_numpy() * u.g/u.cm**3


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

## HW 9:

def K_e(Pg, Pe, A):
    a_e = c.sigma_T
    return (a_e * Pe /(Pg-Pe) * np.sum(A)).cgs

def Kc_total(l, T, Pg, Pe, A_table = A_table, i_table=i_table, u_table=u_table, verbose=False):
#    Pe = P_e(T, Pg, A_table, i_table, u_table, elements)
    Phi_H = saha_phi('H', T, i_table, u_table)
    
    neutral_H = 1 / (1 + Phi_H / Pe)
    K_hydrogen = KH(l, T, Pe, i_table)
    
    A_j = A_table['A'].to_numpy()
    mmm = np.sum(A_table['A'] * A_table['weight'] * u.u).cgs
    
    K_electron = K_e(Pg, Pe, A_j)
    total_per_H = K_hydrogen * neutral_H + K_electron
    total_per_g = total_per_H / mmm
    
    if verbose:
        print('K(continuum) = {:.4g}'.format(total_per_g))
        print('K(H-_bf) = {:.4g}'.format(KHminus_bf(l, T, Pe) * neutral_H / mmm))
        print('K(H-_ff) = {:.4g}'.format(KHminus_ff(l, T, Pe) * neutral_H / mmm * (1 - np.exp(-c.h * c.c / (l * c.k_B * T)))))
        print('K(H_bf) = {:.4g}'.format(KH_bf(l, T, i_table) * neutral_H / mmm * (1 - np.exp(-c.h * c.c / (l * c.k_B * T)))))
        print('K(H_ff) = {:.4g}'.format(KH_ff(l, T, i_table) * neutral_H / mmm * (1 - np.exp(-c.h * c.c / (l * c.k_B * T)))))
        print('K(e-) = {:.4g}'.format(K_electron / mmm))
        print('\n')
    return total_per_g

### HW 10
def gamma_4(T, Pe, log_C4):
    Pe = Pe / u.barye
    T = T / u.K
    log_gamma4 = 19 + 2/3 * log_C4 + np.log10(Pe) - 5/6 * np.log10(T)
    return 10 ** log_gamma4 * 1/u.s

def gamma_6(T, Pg, log_C6):
    Pg = Pg / u.barye
    T = T / u.K
    log_gamma6 = 20 + 0.4 * log_C6 + np.log10(Pg) - 0.7 * np.log10(T)
    return 10 ** log_gamma6 * 1/u.s


def gamma(T, Pg, Pe, A_ul,log_C4, log_C6):
    gamma_rad = A_ul
    gamma4 = gamma_4(T, Pe,log_C4)
    gamma6 = gamma_6(T, Pg, log_C6)
    return gamma_rad + gamma4 + gamma6

def gaussian_width(l0, T, micro):
    m_Na = 22.99 * u.u
    therm = 2 * c.k_B * T / m_Na
    return l0 / c.c * np.sqrt(therm + micro**2)

def Hjerting(u, a):
    return np.real(wofz(u + 1j*a)) 

def osc_f(l, g_l, g_u, A_ul):
    A_gray = A_ul/ (4 * np.pi)
    l = l.to(u.AA).value
    return 1.884e-15 * g_u / g_l * l**2 * A_gray.value

def alpha_NaD(l, T, Pg, Pe, micro, which):
    
    i = which - 1
    
    l0 = np.array([5895.932, 5889.959]) * u.AA
    A_ul = np.array([0.614, 0.616]) * 10 ** 8 * 1/u.s
    log_C4 = np.array([-15.33, -15.17])
    log_C6 = np.array([-31.673, -31.674])
    
    g_l = np.array([2, 2]) 
    g_u = np.array([2, 4])
    
    g = gamma(T, Pg, Pe, A_ul[i], log_C4[i], log_C6[i]) / c.c * l0[i]**2
    Dl = gaussian_width(l0[i], T, micro) 

    a = g /(4 * np.pi * Dl)
    v = (l - l0[i]) / Dl

    H = Hjerting(v, a)  
    
    f = osc_f(l,g_l[i], g_u[i], A_ul[i])
    
    return np.sqrt(np.pi) * e**2 / (c.m_e * c.c**2) * f * H / Dl * l0[i]**2

def K_NaD(l, T, Pg, Pe, nH, rho, micro, which):
    A_Na = 2.14e-6
    f_e = 2/partition('Na', T, u_table)
    Phi_Na = saha_phi('Na', T, i_table, u_table)
    f_i = 1 / (1 + Phi_Na / Pe)
    
    alpha = alpha_NaD(l, T, Pg, Pe, micro, which).cgs 
    stim = 1 - np.exp(-c.h * c.c / (l * c.k_B * T))
    return alpha * nH / rho * A_Na * f_e * f_i * stim

### HW 11

def opacity(l, T, Pg, Pe, nH, rho, micro):
    K = Kc_total(l, T, Pg, Pe)
    for i in range(1, 3):
        K += K_NaD(l, T, Pg, Pe, nH, rho, micro, i).cgs
    return K

def tau_fn(l):
    K = opacity(l, T, Pg, Pe, nH, rho, micro)
    tau = np.array([trapz(K[:i+1] * rho[:i+1], s[:i+1]) for i in range(len(s))])
    return tau